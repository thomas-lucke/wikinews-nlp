"""Tests for src/data_inspector.py.

SPEC: data_inspector contains detect_format, raw_profile, category_profile,
and validate_normalised, plus three print_* helpers. All file-system tests
use tmp_path with real files; no mocking required.
"""
import json

import pandas as pd
import pytest

data_inspector = pytest.importorskip("src.data_inspector")


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------

def test_detect_format_directory_of_json_files(tmp_path):
    # SPEC: detect_format — "If dominant extension is .json: return 'json'."
    for i in range(3):
        (tmp_path / f"a{i}.json").write_text(json.dumps({"x": i}), encoding="utf-8")
    assert data_inspector.detect_format(str(tmp_path)) == "json"


def test_detect_format_directory_of_txt_files(tmp_path):
    # SPEC: detect_format — "If dominant extension is .txt: return 'directory-of-txt'."
    for i in range(3):
        (tmp_path / f"a{i}.txt").write_text(f"text {i}", encoding="utf-8")
    assert data_inspector.detect_format(str(tmp_path)) == "directory-of-txt"


def test_detect_format_jsonl_with_readme_present(tmp_path):
    # SPEC: detect_format — "compute dominance using recognised data files only ...
    # Ignore README.md, license files, and other non-data files for dominance."
    (tmp_path / "wikinews.jsonl").write_text(
        json.dumps({"text": "hi"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# readme", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("MIT", encoding="utf-8")
    assert data_inspector.detect_format(str(tmp_path)) == "jsonl"


def test_detect_format_empty_directory_returns_unknown(tmp_path):
    # SPEC: detect_format — "If there are no recognised data files ... return 'unknown'."
    assert data_inspector.detect_format(str(tmp_path)) == "unknown"


def test_detect_format_single_jsonl_file(tmp_path):
    # SPEC: detect_format step 1 — "If raw_path is a single file: check extension.
    # .jsonl → 'jsonl'".
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
    assert data_inspector.detect_format(str(p)) == "jsonl"


def test_detect_format_mixed_extensions_no_dominance(tmp_path):
    # SPEC: detect_format — "no extension reaches 80% dominance ... return 'unknown'."
    # 2 jsonl + 2 csv + 1 tsv = no single extension at 80%.
    (tmp_path / "a.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "b.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "c.csv").write_text("a,b\n", encoding="utf-8")
    (tmp_path / "d.csv").write_text("a,b\n", encoding="utf-8")
    (tmp_path / "e.tsv").write_text("a\tb\n", encoding="utf-8")
    assert data_inspector.detect_format(str(tmp_path)) == "unknown"


def test_detect_format_exactly_80_percent_dominant(tmp_path):
    # SPEC: detect_format step 2b — "at least 80% of recognised data files share
    # one extension (count / total >= 0.8 — exactly 80% counts as dominant)."
    # 4 of 5 = exactly 80% jsonl → must classify as 'jsonl', not 'unknown'.
    for i in range(4):
        (tmp_path / f"a{i}.jsonl").write_text(
            json.dumps({"text": "x"}) + "\n", encoding="utf-8"
        )
    (tmp_path / "outlier.csv").write_text("a,b\n", encoding="utf-8")
    assert data_inspector.detect_format(str(tmp_path)) == "jsonl"


# ---------------------------------------------------------------------------
# raw_profile
# ---------------------------------------------------------------------------

def test_raw_profile_returns_populated_dataclass(tmp_path):
    # SPEC: RawProfile — fields include total_records, detected_format,
    # detected_fields (union of keys), file_count, total_size_bytes, sample_record.
    p = tmp_path / "data.jsonl"
    p.write_text(
        json.dumps({"text": "a", "title": "t1"}) + "\n"
        + json.dumps({"text": "b", "lang": "en"}) + "\n",
        encoding="utf-8",
    )
    profile = data_inspector.raw_profile(str(p), "jsonl")
    assert profile.total_records == 2
    assert profile.detected_format == "jsonl"
    # detected_fields is union of all keys across records.
    assert set(profile.detected_fields) == {"text", "title", "lang"}
    assert profile.sample_record  # first valid record
    assert isinstance(profile.sample_record, dict)


# ---------------------------------------------------------------------------
# category_profile
# ---------------------------------------------------------------------------

def test_category_profile_sorts_by_count_descending(tmp_path):
    # SPEC: category_profile — "Return a DataFrame sorted by count descending,
    # then category ascending".
    records = [
        {"text": "a", "categories": ["Sports", "Germany"]},
        {"text": "b", "categories": ["Sports", "United States"]},
        {"text": "c", "categories": ["Sports"]},
        {"text": "d", "categories": ["Germany"]},
    ]
    p = tmp_path / "data.jsonl"
    p.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    df = data_inspector.category_profile(str(p), "jsonl")
    counts = df["count"].tolist()
    assert counts == sorted(counts, reverse=True), (
        "category_profile must sort by count descending"
    )
    # Sports = 3, Germany = 2, United States = 1
    assert df.iloc[0]["category"] == "Sports"
    assert int(df.iloc[0]["count"]) == 3


def test_category_profile_deduplicates_within_record(tmp_path):
    # SPEC: category_profile — "Deduplicate repeated category strings within
    # one record before counting so a malformed record cannot inflate one category."
    records = [{"text": "a", "categories": ["Sports", "Sports", "Sports"]}]
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    df = data_inspector.category_profile(str(p), "jsonl")
    row = df[df["category"] == "Sports"]
    assert int(row["count"].iloc[0]) == 1, (
        "Repeated category labels in one record must count once."
    )


def test_category_profile_records_without_categories_returns_empty(tmp_path):
    # SPEC: category_profile — "For non-Wikinews datasets with no 'categories'
    # field, the returned DataFrame is empty with the correct columns and attrs populated."
    records = [{"text": "a"}, {"text": "b"}]
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    df = data_inspector.category_profile(str(p), "jsonl")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["category", "count", "percent"]
    assert len(df) == 0
    assert df.attrs.get("total_records") == 2
    assert df.attrs.get("missing_categories_count") == 2


def test_category_profile_percent_relative_to_total_records(tmp_path):
    # SPEC: category_profile — "percent (float) — count / total parseable records * 100".
    records = [
        {"text": "a", "categories": ["X"]},
        {"text": "b", "categories": ["X"]},
        {"text": "c"},  # no categories — still counts toward total_records
        {"text": "d"},
    ]
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    df = data_inspector.category_profile(str(p), "jsonl")
    x_row = df[df["category"] == "X"].iloc[0]
    assert float(x_row["percent"]) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# validate_normalised — rules on the post-normalisation article list
# ---------------------------------------------------------------------------

def test_validate_normalised_zero_articles_fails(sample_config):
    # SPEC: validate_normalised — "Errors (set validation_passed = False):
    # total_articles == 0".
    report = data_inspector.validate_normalised([], sample_config)
    assert report.validation_passed is False
    assert report.total_articles == 0


def test_validate_normalised_topic_missing_from_articles_adds_warning(sample_config):
    # SPEC: validate_normalised — "Any topic in topics_missing_from_config" → warning.
    # Config selects Politics, Science, Sports — article only has "sports".
    articles = [{
        "id": "1", "title": "t", "text": "x" * 1000, "language": "en",
        "country": "germany", "topic": "sports", "date": "2021-01-01",
    }]
    report = data_inspector.validate_normalised(articles, sample_config)
    missing = {t.lower().strip() for t in report.topics_missing_from_config}
    assert "politics and conflicts" in missing or "science and technology" in missing


def test_validate_normalised_country_missing_from_articles_adds_warning(sample_config):
    # SPEC: validate_normalised — "Any country in countries_missing_from_config" → warning.
    articles = [{
        "id": "1", "title": "t", "text": "x" * 1000, "language": "en",
        "country": "germany", "topic": "sports", "date": "2021-01-01",
    }]
    report = data_inspector.validate_normalised(articles, sample_config)
    missing = {c.lower().strip() for c in report.countries_missing_from_config}
    assert "united states" in missing


def test_validate_normalised_all_selected_topics_missing_fails(sample_config):
    # SPEC: validate_normalised — "Errors: All selected topics are missing from topics_found".
    articles = [{
        "id": "1", "title": "t", "text": "x" * 1000, "language": "en",
        "country": "germany", "topic": "weather", "date": "2021-01-01",
    }]
    report = data_inspector.validate_normalised(articles, sample_config)
    assert report.validation_passed is False


def test_validate_normalised_missing_date_warning_above_5_percent(sample_config):
    # SPEC: validate_normalised — "Warnings (do not fail validation):
    # missing_date_count > 5% of total_articles".
    articles = []
    for i in range(20):
        articles.append({
            "id": str(i), "title": "t", "text": "x" * 1000, "language": "en",
            "country": "germany", "topic": "sports",
            "date": None if i < 5 else "2021-01-01",
        })
    report = data_inspector.validate_normalised(articles, sample_config)
    assert report.missing_date_count == 5
    # 25% missing dates >> 5% — must produce at least one warning entry.
    assert len(report.warnings) >= 1


def test_validate_normalised_topic_case_insensitive_match(sample_config):
    # SPEC: validate_normalised — "Topic casing: ... normalise both sides with
    # _normalise_topic_string (lowercase + strip) before comparing."
    articles = [{
        "id": "1", "title": "t", "text": "x" * 1000, "language": "en",
        "country": "germany", "topic": "sports", "date": "2021-01-01",
    }, {
        "id": "2", "title": "t", "text": "x" * 1000, "language": "en",
        "country": "united states", "topic": "politics and conflicts",
        "date": "2021-01-01",
    }, {
        "id": "3", "title": "t", "text": "x" * 1000, "language": "en",
        "country": "germany", "topic": "science and technology",
        "date": "2021-01-01",
    }]
    # Config uses mixed-case ("Sports", "Politics and conflicts", ...) — articles
    # use lowercase. Match must succeed via normalisation.
    report = data_inspector.validate_normalised(articles, sample_config)
    assert report.validation_passed is True


def test_validate_normalised_very_short_article_count_uses_min_length_multiplier(
    sample_config,
):
    # SPEC: validate_normalised — "very_short_article_count: count of articles
    # where len(text) < config['data']['min_article_length'] * 5."
    # min_article_length=100, so threshold is 500 chars.
    short = "x" * 200  # < 500
    long = "x" * 1000   # >= 500
    articles = [
        {"id": "s1", "title": "t", "text": short, "language": "en",
         "country": "germany", "topic": "sports", "date": "2021-01-01"},
        {"id": "l1", "title": "t", "text": long, "language": "en",
         "country": "germany", "topic": "sports", "date": "2021-01-01"},
    ]
    report = data_inspector.validate_normalised(articles, sample_config)
    assert report.very_short_article_count == 1


# ---------------------------------------------------------------------------
# print_* helpers — SPEC marks these as "intentionally not covered by unit
# tests" (log-output helpers). We only check that they do not raise on a
# populated input — formatted text is reviewer-judged in the notebook.
# ---------------------------------------------------------------------------

def test_print_raw_profile_does_not_raise(tmp_path):
    # SPEC: print_raw_profile — uses logger.info(). Must not crash on a populated
    # RawProfile. Per the "Functions intentionally not covered by unit tests"
    # subsection, we do not assert on output text.
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"text": "x"}) + "\n", encoding="utf-8")
    profile = data_inspector.raw_profile(str(p), "jsonl")
    data_inspector.print_raw_profile(profile)  # no exception is the assertion


def test_print_validation_report_does_not_raise(sample_config):
    # SPEC: print_validation_report — must not raise on valid report.
    report = data_inspector.validate_normalised([], sample_config)
    data_inspector.print_validation_report(report)
