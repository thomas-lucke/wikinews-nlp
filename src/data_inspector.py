"""Inspect the structure and quality of loaded Wikinews data."""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_EXTENSIONS = {".txt", ".json", ".jsonl", ".csv", ".tsv"}
EXT_TO_FORMAT = {
    ".json": "json",
    ".jsonl": "jsonl",
    ".csv": "csv",
    ".tsv": "tsv",
}


@dataclass
class RawProfile:
    """Results of pass 1 - structural analysis of raw files before any field mapping."""

    total_records: int
    detected_format: str
    detected_fields: list[str]
    file_count: int
    total_size_bytes: int
    sample_record: dict


@dataclass
class NormalisedValidation:
    """Results of pass 2 - quality checks on the normalised article list."""

    total_articles: int
    languages_found: dict[str, int]
    countries_found: dict[str, int]
    topics_found: dict[str, int]
    country_topic_counts: dict[tuple[str, str], int]
    missing_date_count: int
    missing_title_count: int
    very_short_article_count: int
    country_topics_below_minimum: list[tuple[str, str]]
    topics_missing_from_config: list[str]
    countries_missing_from_config: list[str]
    validation_passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _direct_child_files(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p.is_file()]


def _parse_one_jsonl(path: Path) -> Optional[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return None
            if isinstance(obj, dict):
                return obj
            return None
    return None


def _parse_one_json(path: Path) -> Optional[dict]:
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
        return None
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        return item
        return data
    return None


def _parse_one_delimited(path: Path, delimiter: str) -> Optional[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            return dict(row)
    return None


def _confirm_parse(raw_path: Path, fmt: str) -> bool:
    """Try to parse one record from the detected format. Returns True if parsing succeeds."""
    fmt_ext = {
        "json": ".json",
        "jsonl": ".jsonl",
        "csv": ".csv",
        "tsv": ".tsv",
        "directory-of-txt": ".txt",
    }
    try:
        if raw_path.is_file():
            candidates = [raw_path]
        else:
            ext = fmt_ext.get(fmt)
            if ext is None:
                return False
            candidates = sorted(p for p in _direct_child_files(raw_path) if p.suffix.lower() == ext)
        if not candidates:
            return False
        first = candidates[0]
        if fmt == "json":
            return _parse_one_json(first) is not None
        if fmt == "jsonl":
            return _parse_one_jsonl(first) is not None
        if fmt == "csv":
            return _parse_one_delimited(first, ",") is not None
        if fmt == "tsv":
            return _parse_one_delimited(first, "\t") is not None
        if fmt == "directory-of-txt":
            with first.open("r", encoding="utf-8", errors="replace") as fh:
                fh.read(1)
            return True
    except (OSError, UnicodeDecodeError):
        return False
    return False


def detect_format(raw_path: str) -> str:
    """Inspect raw_path and return a format string."""
    path = Path(raw_path)
    if not path.exists():
        return "unknown"

    if path.is_file():
        suffix = path.suffix.lower()
        name_lower = path.name.lower()
        if name_lower.endswith(".tar.gz") or suffix in {".zip", ".gz", ".tgz"}:
            return "zip"
        fmt = EXT_TO_FORMAT.get(suffix)
        if fmt is None:
            return "unknown"
        if not _confirm_parse(path, fmt):
            return "unknown"
        return fmt

    if not path.is_dir():
        return "unknown"

    files = _direct_child_files(path)
    data_files = [p for p in files if p.suffix.lower() in DATA_EXTENSIONS]
    if not data_files:
        return "unknown"

    counts: dict[str, int] = {}
    for p in data_files:
        ext = p.suffix.lower()
        counts[ext] = counts.get(ext, 0) + 1
    total = len(data_files)
    dominant_ext = max(counts, key=lambda k: counts[k])
    if counts[dominant_ext] / total < 0.8:
        return "unknown"

    if dominant_ext == ".txt":
        fmt = "directory-of-txt"
    else:
        fmt = EXT_TO_FORMAT.get(dominant_ext)
        if fmt is None:
            return "unknown"

    if not _confirm_parse(path, fmt):
        return "unknown"
    return fmt


def _load_json_file(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        list_keys = [k for k, v in data.items() if isinstance(v, list)]
        if len(list_keys) == 1:
            key = list_keys[0]
            items = data[key]
            kept = [r for r in items if isinstance(r, dict)]
            skipped = len(items) - len(kept)
            if skipped:
                logger.warning(
                    "Skipped %d non-dict entries under key %r in %s",
                    skipped,
                    key,
                    path,
                )
            return kept
        if len(list_keys) > 1:
            kept: list[dict] = []
            for key in list_keys:
                items = data[key]
                inner = [r for r in items if isinstance(r, dict)]
                skipped = len(items) - len(inner)
                if skipped:
                    logger.warning(
                        "Skipped %d non-dict entries under key %r in %s",
                        skipped,
                        key,
                        path,
                    )
                kept.extend(inner)
            return kept
        logger.warning(
            "JSON file %s has no list-valued keys; wrapping top-level dict in a list",
            path,
        )
        return [data]
    logger.warning("JSON file %s top-level is not list or dict; skipping", path)
    return []


def _load_jsonl_file(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping unparseable line %d in %s: %s", lineno, path, exc)
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                logger.warning("Skipping non-dict record on line %d in %s", lineno, path)
    return records


def _load_delimited_file(path: Path, delimiter: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        return [dict(row) for row in reader]


def load_raw_records(raw_path: str, detected_format: str) -> list[dict]:
    """Load raw records from raw_path according to detected_format."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"raw_path does not exist: {raw_path}")

    if detected_format == "unknown":
        raise ValueError(f"Cannot load records from unknown format at {raw_path}")

    if detected_format == "directory-of-txt":
        if not path.is_dir():
            raise ValueError(f"directory-of-txt expected a directory, got file: {raw_path}")
        files = sorted(
            (p for p in _direct_child_files(path) if p.suffix.lower() == ".txt"),
            key=lambda p: p.stem,
        )
        records: list[dict] = []
        for fp in files:
            with fp.open("r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            records.append(
                {
                    "text": text,
                    "id": fp.stem,
                    "language": None,
                    "topic": None,
                    "date": None,
                }
            )
        return records

    ext_for_format = {"json": ".json", "jsonl": ".jsonl", "csv": ".csv", "tsv": ".tsv"}
    if detected_format not in ext_for_format:
        raise ValueError(f"Unsupported detected_format: {detected_format}")
    dominant_ext = ext_for_format[detected_format]

    if path.is_file():
        files = [path]
    else:
        all_data_files = [
            p for p in _direct_child_files(path) if p.suffix.lower() in DATA_EXTENSIONS
        ]
        files = sorted(
            (p for p in all_data_files if p.suffix.lower() == dominant_ext),
            key=lambda p: p.name,
        )
        skipped = [p for p in all_data_files if p.suffix.lower() != dominant_ext]
        for sp in skipped:
            logger.warning(
                "Skipping %s - extension %s does not match dominant format %s",
                sp,
                sp.suffix.lower(),
                detected_format,
            )

    records: list[dict] = []
    for fp in files:
        if detected_format == "json":
            records.extend(_load_json_file(fp))
        elif detected_format == "jsonl":
            records.extend(_load_jsonl_file(fp))
        elif detected_format == "csv":
            records.extend(_load_delimited_file(fp, ","))
        elif detected_format == "tsv":
            records.extend(_load_delimited_file(fp, "\t"))
    return records


def raw_profile(raw_path: str, detected_format: str) -> RawProfile:
    """Count records, collect field names, and retrieve one sample record."""
    path = Path(raw_path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        if detected_format == "directory-of-txt":
            files = [p for p in _direct_child_files(path) if p.suffix.lower() == ".txt"]
        elif detected_format in EXT_TO_FORMAT.values():
            ext = "." + detected_format
            files = [p for p in _direct_child_files(path) if p.suffix.lower() == ext]
        else:
            files = [p for p in _direct_child_files(path) if p.suffix.lower() in DATA_EXTENSIONS]
    else:
        files = []

    file_count = len(files)
    total_size_bytes = sum(p.stat().st_size for p in files)

    records = load_raw_records(raw_path, detected_format)
    total_records = len(records)

    if detected_format == "directory-of-txt":
        detected_fields: list[str] = []
    else:
        field_set: dict[str, None] = {}
        for r in records:
            for key in r.keys():
                if key not in field_set:
                    field_set[key] = None
        detected_fields = list(field_set.keys())

    sample_record = records[0] if records else {}

    return RawProfile(
        total_records=total_records,
        detected_format=detected_format,
        detected_fields=detected_fields,
        file_count=file_count,
        total_size_bytes=total_size_bytes,
        sample_record=sample_record,
    )


def print_raw_profile(profile: RawProfile) -> None:
    """Log a structured summary of the RawProfile via logger.info()."""
    logger.info("Detected format: %s", profile.detected_format)
    logger.info("File count: %d", profile.file_count)
    logger.info("Total size (bytes): %d", profile.total_size_bytes)
    logger.info("Total parseable records: %d", profile.total_records)
    logger.info("Detected fields (%d): %s", len(profile.detected_fields), profile.detected_fields)

    if profile.sample_record:
        truncated: dict = {}
        for key, value in profile.sample_record.items():
            if isinstance(value, str) and len(value) > 100:
                truncated[key] = value[:100] + "…"
            else:
                truncated[key] = value
        logger.info("Sample record keys: %s", list(profile.sample_record.keys()))
        logger.info("Sample record (text fields truncated to 100 chars): %s", truncated)
    else:
        logger.info("Sample record: <none>")


def _normalise_topic_string(s: str) -> str:
    return s.lower().strip()


def category_profile(raw_path: str, detected_format: str) -> pd.DataFrame:
    """Count raw category labels per record. Returns a DataFrame with attrs metadata."""
    columns = ["category", "count", "percent"]
    records = load_raw_records(raw_path, detected_format)
    total_records = len(records)

    counts: dict[str, int] = {}
    missing_categories_count = 0
    any_categories_field = False

    for r in records:
        if "categories" not in r:
            missing_categories_count += 1
            continue
        any_categories_field = True
        cats = r["categories"]
        if cats is None or not isinstance(cats, (list, str)):
            missing_categories_count += 1
            continue
        if isinstance(cats, str):
            label = cats.strip()
            if not label:
                missing_categories_count += 1
                continue
            counts[label] = counts.get(label, 0) + 1
            continue
        in_record: set[str] = set()
        for item in cats:
            if isinstance(item, str):
                label = item.strip()
                if label:
                    in_record.add(label)
        if not in_record:
            missing_categories_count += 1
            continue
        for label in in_record:
            counts[label] = counts.get(label, 0) + 1

    if not any_categories_field:
        df = pd.DataFrame(columns=columns)
        df.attrs["total_records"] = total_records
        df.attrs["missing_categories_count"] = missing_categories_count
        return df

    sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    rows = [
        {
            "category": cat,
            "count": n,
            "percent": (n / total_records * 100) if total_records else 0.0,
        }
        for cat, n in sorted_items
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.attrs["total_records"] = total_records
    df.attrs["missing_categories_count"] = missing_categories_count
    return df


def validate_normalised(
    articles: list[dict],
    config: dict,
) -> NormalisedValidation:
    """Run quality checks on the normalised article list."""
    total = len(articles)
    languages_found: dict[str, int] = {}
    countries_found: dict[str, int] = {}
    topics_found: dict[str, int] = {}
    country_topic_counts: dict[tuple[str, str], int] = {}
    missing_date_count = 0
    missing_title_count = 0
    very_short_article_count = 0

    min_article_length = config.get("data", {}).get("min_article_length", 0)
    short_threshold = min_article_length * 5

    for a in articles:
        lang = a.get("language")
        if isinstance(lang, str) and lang:
            languages_found[lang] = languages_found.get(lang, 0) + 1
        country = a.get("country")
        if isinstance(country, str) and country:
            countries_found[country] = countries_found.get(country, 0) + 1
        topic = a.get("topic")
        if isinstance(topic, str) and topic:
            topics_found[topic] = topics_found.get(topic, 0) + 1
        if isinstance(country, str) and country and isinstance(topic, str) and topic:
            key = (country, topic)
            country_topic_counts[key] = country_topic_counts.get(key, 0) + 1
        if not a.get("date"):
            missing_date_count += 1
        if a.get("title", "") == "":
            missing_title_count += 1
        text = a.get("text", "")
        if isinstance(text, str) and len(text) < short_threshold:
            very_short_article_count += 1

    selected_topics_raw = list(config.get("topics", {}).get("selected", []) or [])
    selected_countries_raw = list(config.get("countries", {}).get("selected", []) or [])
    min_per_topic = config.get("topics", {}).get("articles_per_topic_min", 0)

    selected_topics_pairs = [(_normalise_topic_string(t), t) for t in selected_topics_raw]
    selected_countries_pairs = [(_normalise_topic_string(c), c) for c in selected_countries_raw]

    topics_missing_from_config = [
        raw for norm_t, raw in selected_topics_pairs if norm_t not in topics_found
    ]
    countries_missing_from_config = [
        raw for norm_c, raw in selected_countries_pairs if norm_c not in countries_found
    ]

    country_topics_below_minimum: list[tuple[str, str]] = []
    for norm_c, _raw_c in selected_countries_pairs:
        for norm_t, _raw_t in selected_topics_pairs:
            cnt = country_topic_counts.get((norm_c, norm_t), 0)
            if cnt < min_per_topic:
                country_topics_below_minimum.append((norm_c, norm_t))

    warnings: list[str] = []
    errors: list[str] = []

    if total == 0:
        errors.append("No articles after normalisation (total_articles == 0).")
    if selected_topics_raw and len(topics_missing_from_config) == len(selected_topics_raw):
        errors.append(f"All selected topics missing from articles: {selected_topics_raw}")
    if selected_countries_raw and len(countries_missing_from_config) == len(selected_countries_raw):
        errors.append(f"All selected countries missing from articles: {selected_countries_raw}")

    if total > 0 and missing_date_count > 0.05 * total:
        warnings.append(f"{missing_date_count}/{total} articles missing date (>5%).")
    if total > 0 and very_short_article_count > 0.10 * total:
        warnings.append(
            f"{very_short_article_count}/{total} articles shorter than "
            f"{short_threshold} chars (>10%)."
        )
    for pair in country_topics_below_minimum:
        warnings.append(
            f"(country='{pair[0]}', topic='{pair[1]}') has fewer than {min_per_topic} articles."
        )
    for t in topics_missing_from_config:
        warnings.append(f"Topic '{t}' from config not found in articles.")
    for c in countries_missing_from_config:
        warnings.append(f"Country '{c}' from config not found in articles.")

    validation_passed = not errors

    return NormalisedValidation(
        total_articles=total,
        languages_found=languages_found,
        countries_found=countries_found,
        topics_found=topics_found,
        country_topic_counts=country_topic_counts,
        missing_date_count=missing_date_count,
        missing_title_count=missing_title_count,
        very_short_article_count=very_short_article_count,
        country_topics_below_minimum=country_topics_below_minimum,
        topics_missing_from_config=topics_missing_from_config,
        countries_missing_from_config=countries_missing_from_config,
        validation_passed=validation_passed,
        warnings=warnings,
        errors=errors,
    )


def print_category_profile(df: pd.DataFrame) -> None:
    """Log the category profile table via logger.info()."""
    total = df.attrs.get("total_records", 0)
    missing = df.attrs.get("missing_categories_count", 0)
    logger.info("Total parseable records: %d", total)
    logger.info("Records with no usable categories: %d", missing)
    if df.empty:
        logger.info("  <no categories found>")
    else:
        logger.info("Categories loaded.")
    logger.info(
        "Review these category labels, update config.yaml topics.selected and "
        "countries.selected if needed, then rerun Cell 1 or reload config "
        "before continuing."
    )


def print_validation_report(report: NormalisedValidation) -> None:
    """
    Log a concise, human-scannable validation report.

    Headline statistics go to logger.info; actionable problems are surfaced via
    the stored warnings (logger.warning) and errors (logger.error). The full
    per-country and per-(country, topic) breakdowns are intentionally NOT logged
    line by line - they remain on the NormalisedValidation object for
    programmatic access, but dumping ~50 countries and ~80 country/topic pairs
    makes the report unreadable. Country is metadata only (not a pipeline axis),
    so it gets a single summary line.
    """
    logger.info("Total articles: %d", report.total_articles)
    logger.info("Languages: %s", report.languages_found)
    logger.info("Topics: %s", report.topics_found)

    # Country is best-effort metadata, not a sampling axis - one summary line only.
    n_countries = len(report.countries_found)
    if n_countries:
        top = sorted(report.countries_found.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_str = ", ".join(f"{name}={count}" for name, count in top)
        logger.info("Countries (metadata): %d distinct (top: %s)", n_countries, top_str)
    else:
        logger.info("Countries (metadata): none extracted")

    logger.info(
        "Data quality: %d missing dates, %d missing titles, %d very short",
        report.missing_date_count,
        report.missing_title_count,
        report.very_short_article_count,
    )

    for w in report.warnings:
        logger.warning(w)
    for e in report.errors:
        logger.error(e)

    if report.validation_passed:
        logger.info("Validation passed.")
    else:
        logger.error("Validation FAILED - review errors above.")
