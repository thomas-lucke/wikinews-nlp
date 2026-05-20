"""Normalize raw Wikinews articles into a consistent schema."""

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Optional

import pycountry

from src.data_inspector import (
    load_raw_records,  # noqa: F401  (re-exported for monkeypatching in tests)
)

logger = logging.getLogger(__name__)


FIELD_MAPPINGS: dict[str, str] = {
    "text": "text",
    "body": "text",
    "article_body": "text",
    "content": "text",
    "article_text": "text",
    "title": "title",
    "headline": "title",
    "article_title": "title",
    "date": "date",
    "published": "date",
    "publish_date": "date",
    "created_at": "date",
    "timestamp": "date",
    "language": "language",
    "lang": "language",
    "locale": "language",
    "country": "country",
    "country_name": "country",
    "location": "country",
    "topic": "topic",
    "category": "topic",
    "section": "topic",
    "label": "topic",
    "categories": "categories",
    "id": "id",
    "article_id": "id",
    "uid": "id",
    "pageid": "event_id",
}


@dataclass
class DroppedRecord:
    article_index: int
    reason: str
    field_values: dict


def _generate_stable_id(text: str) -> str:
    """Return first 16 hex chars of sha256(text)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalise_topic_string(s: str) -> str:
    """Lowercase and strip whitespace."""
    return s.lower().strip()


def _select_from_categories(categories: list[str], selected_labels: list[str]) -> Optional[str]:
    """Return the first configured label that matches any category (case-insensitive, stripped).

    Comparison normalises both sides with _normalise_topic_string. The returned
    value is the configured label, lowercased and stripped.
    """
    normalised_categories = {_normalise_topic_string(c) for c in categories if isinstance(c, str)}
    for label in selected_labels:
        if not isinstance(label, str):
            continue
        norm_label = _normalise_topic_string(label)
        if norm_label in normalised_categories:
            return norm_label
    return None


def _infer_text_field(raw_record: dict, min_article_length: int) -> Optional[str]:
    """Return the key (not value) of the longest string field not in FIELD_MAPPINGS.

    A candidate must satisfy:
      - key not in FIELD_MAPPINGS
      - value is a str
      - len(value.strip()) >= min_article_length
    """
    best_key: Optional[str] = None
    best_len = -1
    for key, value in raw_record.items():
        if key in FIELD_MAPPINGS:
            continue
        if not isinstance(value, str):
            continue
        stripped_len = len(value.strip())
        if stripped_len < min_article_length:
            continue
        if stripped_len > best_len:
            best_len = stripped_len
            best_key = key
    return best_key


def print_normalisation_summary(valid: list[dict], dropped: list[DroppedRecord]) -> None:
    """Log totals and per-group breakdowns via logger.info."""
    logger.info("Total valid articles: %d", len(valid))

    by_country: dict[str, int] = {}
    by_language: dict[str, int] = {}
    by_topic: dict[str, int] = {}
    for a in valid:
        c = a.get("country")
        if isinstance(c, str):
            by_country[c] = by_country.get(c, 0) + 1
        lang = a.get("language")
        if isinstance(lang, str):
            by_language[lang] = by_language.get(lang, 0) + 1
        t = a.get("topic")
        if isinstance(t, str):
            by_topic[t] = by_topic.get(t, 0) + 1

    logger.info("Valid by country: %s", by_country)
    logger.info("Valid by language: %s", by_language)
    logger.info("Valid by topic: %s", by_topic)

    logger.info("Total dropped records: %d", len(dropped))
    by_reason: dict[str, int] = {}
    for d in dropped:
        by_reason[d.reason] = by_reason.get(d.reason, 0) + 1
    logger.info("Dropped by reason: %s", by_reason)


_SOURCE_ID_RAW_KEYS = {"id", "article_id", "uid"}
_DEBUG_FIELDS = ("id", "title", "language", "topic", "country")


def _debug_field_values(working: dict, raw: dict) -> dict:
    out: dict = {}
    for key in _DEBUG_FIELDS:
        if key in working:
            out[key] = working[key]
        elif key in raw:
            out[key] = raw[key]
    return out


# Colloquial English country names that the ISO 3166 dataset (pycountry) does not
# expose via name / common_name / official_name. pycountry lists Russia as
# "Russian Federation" and Turkey as "Türkiye" (the 2022 ISO rename); Wikinews
# categories use the older colloquial forms. These aliases prevent a silent
# regression vs. the previous hand-maintained country list.
_COUNTRY_ALIASES: frozenset[str] = frozenset({"russia", "turkey"})

# Recognised country names for best-effort metadata extraction when no country
# filter is active. Sourced from pycountry (ISO 3166) so the list is complete and
# maintenance-free; matching is exact (see _extract_country_from_categories), so
# the broad coverage does not introduce substring false positives.
_KNOWN_COUNTRIES: frozenset[str] = (
    frozenset(
        _normalise_topic_string(name)
        for country in pycountry.countries
        for name in filter(
            None,
            (
                getattr(country, "name", None),
                getattr(country, "common_name", None),
                getattr(country, "official_name", None),
            ),
        )
    )
    | _COUNTRY_ALIASES
)


def _extract_country_from_categories(categories: list[str]) -> Optional[str]:
    """Return the first recognised country name found in categories, or None.

    Searches _KNOWN_COUNTRIES using the same normalisation as topic/country
    matching. Returns the normalised (lowercase, stripped) country string.
    """
    for cat in categories:
        if not isinstance(cat, str):
            continue
        norm = _normalise_topic_string(cat)
        if norm in _KNOWN_COUNTRIES:
            return norm
    return None


def _match_string_to_config(value: str, selected: list[str]) -> Optional[str]:
    """Return the first config label whose normalised form matches `value`, or None."""
    norm_value = _normalise_topic_string(value)
    for label in selected:
        if not isinstance(label, str):
            continue
        norm_label = _normalise_topic_string(label)
        if norm_label == norm_value:
            return norm_label
    return None


def normalise_articles(
    raw_path: str,
    detected_format: str,
    languages: list[str],
    topics: list[str],
    countries: Optional[list[str]],
    max_per_topic: int,
    min_article_length: int,
    random_seed: int = 42,
) -> tuple[list[dict], list[DroppedRecord]]:
    """Load raw records, map fields, filter, deduplicate, and sample by group.

    When countries is None, country filtering is skipped and country is extracted
    from the categories list as best-effort metadata (empty string if unavailable).
    Sampling groups by (language, topic) only — max_per_topic applies per
    language×topic pair regardless of country.

    When countries is a non-empty list, articles whose categories contain no
    matching country are dropped, and sampling groups by (country, language, topic).
    """
    if not languages:
        raise ValueError("languages must be non-empty")
    if not topics:
        raise ValueError("topics must be non-empty")
    if countries is not None and not countries:
        raise ValueError(
            "countries must be non-empty when provided (pass None to disable filtering)"
        )

    rng = random.Random(random_seed)

    raw_records = load_raw_records(raw_path, detected_format)

    languages_lower = {lang.lower().strip() for lang in languages if isinstance(lang, str)}

    survivors: list[dict] = []
    dropped: list[DroppedRecord] = []

    for idx, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            continue

        working: dict = {}
        has_source_id = False
        pageid_raw_value = None

        for raw_key, internal in FIELD_MAPPINGS.items():
            if raw_key in raw and internal not in working:
                working[internal] = raw[raw_key]
                if raw_key in _SOURCE_ID_RAW_KEYS:
                    has_source_id = True
                if raw_key == "pageid":
                    pageid_raw_value = raw[raw_key]

        text = working.get("text")
        # Handle text as list of strings (paragraphs) or single string
        if isinstance(text, list):
            text = " ".join(str(p) for p in text if p)
            working["text"] = text
        if not isinstance(text, str):
            inferred_key = _infer_text_field(raw, min_article_length)
            if inferred_key is not None:
                if inferred_key == "url":
                    logger.warning(
                        "Record %d: would infer 'url' as text field (bad heuristic),"
                        " dropping instead",
                        idx,
                    )
                    dropped.append(
                        DroppedRecord(
                            article_index=idx,
                            reason="no_valid_text_field",
                            field_values=_debug_field_values(working, raw),
                        )
                    )
                    continue
                text = raw[inferred_key]
                working["text"] = text
                logger.info("Record %d: inferred text from field %r", idx, inferred_key)

        if not isinstance(text, str):
            dropped.append(
                DroppedRecord(
                    article_index=idx,
                    reason="no_text_field",
                    field_values=_debug_field_values(working, raw),
                )
            )
            continue
        if len(text.strip()) < min_article_length:
            dropped.append(
                DroppedRecord(
                    article_index=idx,
                    reason="text_too_short",
                    field_values=_debug_field_values(working, raw),
                )
            )
            continue

        lang_raw = working.get("language")
        if not isinstance(lang_raw, str):
            dropped.append(
                DroppedRecord(
                    article_index=idx,
                    reason="language_not_in_config",
                    field_values=_debug_field_values(working, raw),
                )
            )
            continue
        lang_norm = lang_raw.lower().strip()
        if lang_norm not in languages_lower:
            dropped.append(
                DroppedRecord(
                    article_index=idx,
                    reason="language_not_in_config",
                    field_values=_debug_field_values(working, raw),
                )
            )
            continue
        working["language"] = lang_norm

        categories_raw = working.get("categories")
        topic_value: Optional[str] = None
        if isinstance(categories_raw, list):
            topic_value = _select_from_categories(categories_raw, topics)
        elif isinstance(working.get("topic"), str):
            topic_value = _match_string_to_config(working["topic"], topics)

        if topic_value is None:
            dropped.append(
                DroppedRecord(
                    article_index=idx,
                    reason="topic_not_in_config",
                    field_values=_debug_field_values(working, raw),
                )
            )
            continue

        working["topic"] = topic_value

        if countries is not None:
            country_value: Optional[str] = None
            if isinstance(categories_raw, list):
                country_value = _select_from_categories(categories_raw, countries)
            elif isinstance(working.get("country"), str):
                country_value = _match_string_to_config(working["country"], countries)

            if country_value is None:
                dropped.append(
                    DroppedRecord(
                        article_index=idx,
                        reason="country_not_in_config",
                        field_values=_debug_field_values(working, raw),
                    )
                )
                continue
            working["country"] = country_value
        else:
            # No country filter: extract country from categories as metadata only.
            # Uses the full known-countries pool from the raw categories list so
            # downstream analysis (NER plots, similarity grouping) gets meaningful
            # country labels rather than empty strings.
            meta_country: Optional[str] = None
            if isinstance(categories_raw, list):
                meta_country = _extract_country_from_categories(categories_raw)
            elif isinstance(working.get("country"), str):
                meta_country = _normalise_topic_string(working["country"])
            working["country"] = meta_country or ""

        working["_has_source_id"] = has_source_id
        working["_pageid_raw_value"] = pageid_raw_value
        working["_raw_index"] = idx
        survivors.append(working)

    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for w in survivors:
        h = _generate_stable_id(w["text"])
        if h in seen_hashes:
            dropped.append(
                DroppedRecord(
                    article_index=w["_raw_index"],
                    reason="duplicate",
                    field_values=_debug_field_values(w, {}),
                )
            )
            continue
        seen_hashes.add(h)
        deduped.append(w)

    # When countries filtering is active, group by (country, language, topic) so
    # max_per_topic is enforced per country×language×topic cell.
    # When countries is None, group by (language, topic) only — country is metadata
    # and must not fragment the pool, which would silently under-sample rare countries.
    groups: dict[tuple, list[dict]] = {}
    for w in deduped:
        if countries is None:
            key = (w["language"], w["topic"])
        else:
            key = (w["country"], w["language"], w["topic"])
        groups.setdefault(key, []).append(w)

    sampled: list[dict] = []
    for group in groups.values():
        if len(group) <= max_per_topic:
            sampled.extend(group)
            continue
        all_have_id = all(w["_has_source_id"] for w in group)
        if all_have_id:
            group_sorted = sorted(group, key=lambda w: str(w.get("id", "")))
        else:
            group_sorted = list(group)  # already in post-dedup position order
        sampled.extend(rng.sample(group_sorted, max_per_topic))

    final: list[dict] = []
    for w in sampled:
        text_val = w["text"]
        has_id = w.pop("_has_source_id")
        pageid_raw = w.pop("_pageid_raw_value")
        w.pop("_raw_index", None)

        if has_id:
            id_val = w.get("id")
            article_id = id_val if isinstance(id_val, str) else str(id_val)
        else:
            article_id = _generate_stable_id(text_val)

        event_id = str(pageid_raw) if pageid_raw is not None else None

        title = w.get("title", "")
        if not isinstance(title, str):
            title = ""

        date_val = w.get("date")
        if date_val is None:
            date_out = None
        else:
            date_out = date_val

        final.append(
            {
                "id": article_id,
                "event_id": event_id,
                "title": title,
                "text": text_val,
                "language": w["language"],
                "topic": w["topic"],
                "country": w["country"],
                "date": date_out,
            }
        )

    return final, dropped
