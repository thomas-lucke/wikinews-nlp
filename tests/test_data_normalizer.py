"""Tests for src/data_normalizer.py.

SPEC: data_normalizer maps raw field names to internal schema, deduplicates,
filters, and samples. Tests write real JSONL files to tmp_path so they
exercise the same load_raw_records path the production pipeline uses.
"""

import json
from pathlib import Path

import pytest

data_normalizer = pytest.importorskip("src.data_normalizer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def _wikinews_record(**overrides) -> dict:
    base = {
        "title": "Some Title",
        "text": "Berlin researchers announced a new finding. " * 10,
        "lang": "en",
        "categories": ["Sports", "Germany"],
        "pageid": 12345,
        "date": "2021-01-01",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Stable ID - pure-function tests
# ---------------------------------------------------------------------------


def test_generate_stable_id_is_deterministic():
    # SPEC: _generate_stable_id - "Return first 16 hex chars of sha256 of text.
    # Stable across reruns."
    a = data_normalizer._generate_stable_id("hello world")
    b = data_normalizer._generate_stable_id("hello world")
    assert a == b
    assert len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_generate_stable_id_differs_for_different_text():
    # SPEC: _generate_stable_id - distinct text should not collide.
    a = data_normalizer._generate_stable_id("text one")
    b = data_normalizer._generate_stable_id("text two")
    assert a != b


def test_normalise_topic_string_lowercases_and_strips():
    # SPEC: _normalise_topic_string - "Lowercase and strip whitespace."
    assert data_normalizer._normalise_topic_string("  Sports  ") == "sports"
    assert data_normalizer._normalise_topic_string("POLITICS") == "politics"


# ---------------------------------------------------------------------------
# Field mapping
# ---------------------------------------------------------------------------


def test_field_mappings_includes_required_aliases():
    # SPEC: FIELD_MAPPINGS - must include the listed aliases.
    fm = data_normalizer.FIELD_MAPPINGS
    assert fm["article_body"] == "text"
    assert fm["body"] == "text"
    assert fm["headline"] == "title"
    assert fm["publish_date"] == "date"
    assert fm["lang"] == "language"
    assert fm["category"] == "topic"
    assert fm["article_id"] == "id"
    # SPEC: "Wikinews pageid groups multilingual articles ... store as event_id;
    # do not use it as article['id']."
    assert fm["pageid"] == "event_id"


def test_field_mappings_does_not_map_url_to_id():
    # SPEC: FIELD_MAPPINGS - '"url" is NOT mapped to "id".'
    assert data_normalizer.FIELD_MAPPINGS.get("url") != "id"


def test_field_mappings_precedence_text_wins_over_body(tmp_path):
    # SPEC: FIELD_MAPPINGS - "Precedence rule: iterate FIELD_MAPPINGS in
    # insertion order ... The first FIELD_MAPPINGS key that is present in the
    # raw record wins for that internal field. Example: a raw record with both
    # 'text' and 'body' yields internal 'text' from the 'text' key, because
    # 'text' appears before 'body' in this dict."
    # Bucket B test: spec is precise, this test exercises the deterministic
    # behaviour the spec already defines.
    record_text = "Body text here. " * 30
    record_body = "OTHER body field value. " * 30
    record = {
        "text": record_text,
        "body": record_body,
        "lang": "en",
        "categories": ["Sports", "Germany"],
        "pageid": 1,
    }
    p = _write_jsonl(tmp_path, [record])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1
    assert articles[0]["text"].startswith("Body text here."), (
        "FIELD_MAPPINGS precedence: 'text' must win over 'body' "
        "(insertion order in FIELD_MAPPINGS)."
    )


def test_normalise_articles_maps_non_standard_field_names(tmp_path):
    # SPEC test list: normalise_articles with sample_raw_record - fields
    # correctly mapped (article_body→text, headline→title, etc.)
    record = {
        "article_body": "x " * 200,
        "headline": "Test Article",
        "publish_date": "2021-03-10",
        "lang": "en",
        "category": "Sports",
        "country": "Germany",
        "article_id": "raw001",
        "pageid": 12345,
    }
    p = _write_jsonl(tmp_path, [record])
    articles, _dropped = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1
    a = articles[0]
    assert a["title"] == "Test Article"
    assert a["language"] == "en"
    # SPEC: step 7 - topic and country stored lowercased + stripped.
    assert a["topic"] == "sports"
    assert a["country"] == "germany"
    assert a["id"] == "raw001"  # has_source_id → keep raw value
    assert a["event_id"] == "12345"  # SPEC: store as str(raw_value)
    assert a["date"] == "2021-03-10"


# ---------------------------------------------------------------------------
# Wikinews categories - topic + country resolved from same list
# ---------------------------------------------------------------------------


def test_wikinews_categories_resolve_to_configured_topic(tmp_path):
    # SPEC: normalise_articles step 3 - "For Wikinews, topic is resolved from the
    # raw categories list using _select_from_categories(categories, topics)."
    p = _write_jsonl(tmp_path, [_wikinews_record(categories=["Politics and conflicts", "Germany"])])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Politics and conflicts"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1
    assert articles[0]["topic"] == "politics and conflicts"


def test_wikinews_categories_resolve_to_configured_country(tmp_path):
    # SPEC: normalise_articles step 3 - "For Wikinews, country is resolved from
    # the raw categories list using _select_from_categories(categories, countries)."
    p = _write_jsonl(tmp_path, [_wikinews_record(categories=["Sports", "Germany"])])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1
    assert articles[0]["country"] == "germany"


def test_pageid_stored_as_event_id_not_id(tmp_path):
    # SPEC: "pageid groups multilingual articles ... Store it as event_id;
    # do not use it as the unique article id."
    p = _write_jsonl(tmp_path, [_wikinews_record(pageid=99999)])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    a = articles[0]
    assert a["event_id"] == "99999"
    assert a["id"] != "99999"
    assert a["id"] != 99999


# ---------------------------------------------------------------------------
# Drop reasons (DroppedRecord.reason values are fixed strings in the spec)
# ---------------------------------------------------------------------------


def test_dropped_record_text_too_short(tmp_path):
    # SPEC: normalise_articles step 3 - drop if "found text is shorter than
    # min_article_length after strip()" → reason: "text_too_short".
    p = _write_jsonl(tmp_path, [_wikinews_record(text="short")])
    _articles, dropped = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert any(d.reason == "text_too_short" for d in dropped)


def test_dropped_record_language_not_in_config(tmp_path):
    # SPEC: normalise_articles step 3 - drop if "Language present but not in
    # languages list → reason: 'language_not_in_config'".
    p = _write_jsonl(tmp_path, [_wikinews_record(lang="fr")])
    _articles, dropped = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en", "de"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert any(d.reason == "language_not_in_config" for d in dropped)


def test_dropped_record_country_not_in_config(tmp_path):
    # SPEC: normalise_articles step 3 - "no configured country matches the source
    # country signal → reason: 'country_not_in_config'".
    p = _write_jsonl(tmp_path, [_wikinews_record(categories=["Sports", "France"])])
    _articles, dropped = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany", "United States"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert any(d.reason == "country_not_in_config" for d in dropped)


def test_dropped_record_topic_not_in_config(tmp_path):
    # SPEC: normalise_articles step 3 - "no configured topic matches the source
    # topic value → reason: 'topic_not_in_config'".
    p = _write_jsonl(tmp_path, [_wikinews_record(categories=["Weather", "Germany"])])
    _articles, dropped = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports", "Politics and conflicts"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert any(d.reason == "topic_not_in_config" for d in dropped)


def test_dropped_record_duplicate(tmp_path):
    # SPEC: normalise_articles step 4 - "If the hash has been seen before in this
    # run, drop with reason 'duplicate'."
    r1 = _wikinews_record(text="Unique body text. " * 20, article_id="A")
    r2 = _wikinews_record(text="Unique body text. " * 20, article_id="B")
    p = _write_jsonl(tmp_path, [r1, r2])
    articles, dropped = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1
    assert any(d.reason == "duplicate" for d in dropped)


# ---------------------------------------------------------------------------
# Stable id assignment
# ---------------------------------------------------------------------------


def test_stable_id_consistent_across_runs(tmp_path):
    # SPEC: _generate_stable_id is "Stable across reruns." Running the same
    # normalisation twice on the same record produces the same id.
    rec = _wikinews_record()
    rec.pop("article_id", None)  # force hash fallback
    p = _write_jsonl(tmp_path, [rec])
    kwargs = dict(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    a1, _ = data_normalizer.normalise_articles(**kwargs)
    a2, _ = data_normalizer.normalise_articles(**kwargs)
    assert a1[0]["id"] == a2[0]["id"]


def test_source_id_preserved_when_present(tmp_path):
    # SPEC: normalise_articles step 6 - "If has_source_id is True: the 'id' field
    # in the working dict already holds the source value; keep it as-is."
    p = _write_jsonl(tmp_path, [_wikinews_record(article_id="MYID-001")])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert articles[0]["id"] == "MYID-001"


def test_id_generated_when_no_source_id(tmp_path):
    # SPEC: normalise_articles step 6 - "If has_source_id is False: generate id
    # via _generate_stable_id(article['text'])."
    rec = _wikinews_record()
    for key in ("id", "article_id", "uid"):
        rec.pop(key, None)
    p = _write_jsonl(tmp_path, [rec])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    expected = data_normalizer._generate_stable_id(articles[0]["text"])
    assert articles[0]["id"] == expected


# ---------------------------------------------------------------------------
# Validation of input arguments
# ---------------------------------------------------------------------------


def test_empty_languages_raises_value_error(tmp_path):
    # SPEC: normalise_articles - "Pipeline-level validation: raise ValueError if
    # languages, topics, or countries is empty."
    p = _write_jsonl(tmp_path, [_wikinews_record()])
    with pytest.raises(ValueError):
        data_normalizer.normalise_articles(
            raw_path=str(p),
            detected_format="jsonl",
            languages=[],
            topics=["Sports"],
            countries=["Germany"],
            max_per_topic=10,
            min_article_length=100,
            random_seed=42,
        )


def test_empty_topics_raises_value_error(tmp_path):
    # SPEC: normalise_articles - same rule for topics.
    p = _write_jsonl(tmp_path, [_wikinews_record()])
    with pytest.raises(ValueError):
        data_normalizer.normalise_articles(
            raw_path=str(p),
            detected_format="jsonl",
            languages=["en"],
            topics=[],
            countries=["Germany"],
            max_per_topic=10,
            min_article_length=100,
            random_seed=42,
        )


def test_empty_countries_raises_value_error(tmp_path):
    # SPEC: normalise_articles - same rule for countries.
    p = _write_jsonl(tmp_path, [_wikinews_record()])
    with pytest.raises(ValueError):
        data_normalizer.normalise_articles(
            raw_path=str(p),
            detected_format="jsonl",
            languages=["en"],
            topics=["Sports"],
            countries=[],
            max_per_topic=10,
            min_article_length=100,
            random_seed=42,
        )


# ---------------------------------------------------------------------------
# Output schema - every returned article must have the guaranteed fields
# ---------------------------------------------------------------------------

GUARANTEED_FIELDS = {"id", "event_id", "title", "text", "language", "topic", "country", "date"}


def test_output_articles_have_all_guaranteed_schema_fields(tmp_path):
    # SPEC: Article schema - "Guaranteed after normalise_articles()" - every
    # listed field must be present (some optional, but the key must exist).
    p = _write_jsonl(tmp_path, [_wikinews_record()])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert GUARANTEED_FIELDS.issubset(
        articles[0].keys()
    ), f"Missing guaranteed schema fields: {GUARANTEED_FIELDS - articles[0].keys()}"


def test_title_is_empty_string_when_missing_never_none(tmp_path):
    # SPEC: Article schema - '"title": str, # Empty string "" if not in source - never None'.
    rec = _wikinews_record()
    rec.pop("title", None)
    rec.pop("headline", None)
    rec.pop("article_title", None)
    p = _write_jsonl(tmp_path, [rec])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert articles[0]["title"] == ""
    assert articles[0]["title"] is not None


def test_date_is_none_when_missing(tmp_path):
    # SPEC: Article schema - '"date": Optional[str], # ... or None if missing.'
    rec = _wikinews_record()
    for k in ("date", "published", "publish_date", "created_at", "timestamp"):
        rec.pop(k, None)
    p = _write_jsonl(tmp_path, [rec])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert articles[0]["date"] is None


def test_topic_and_country_lowercased_and_stripped(tmp_path):
    # SPEC: normalise_articles step 7 - "Normalise topic and country: store
    # lowercased stripped strings."
    p = _write_jsonl(tmp_path, [_wikinews_record(categories=["  Sports  ", "Germany"])])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert articles[0]["topic"] == "sports"
    assert articles[0]["country"] == "germany"


# ---------------------------------------------------------------------------
# Sampling: max_per_topic cap and determinism
# ---------------------------------------------------------------------------


def test_max_per_topic_caps_group_size(tmp_path):
    # SPEC: normalise_articles step 5 - "if count > max_per_topic: ... sample
    # max_per_topic records".
    records = [
        _wikinews_record(text=f"Unique body {i} " * 30, article_id=f"id{i}") for i in range(30)
    ]
    p = _write_jsonl(tmp_path, records)
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=5,
        min_article_length=10,
        random_seed=42,
    )
    assert len(articles) == 5


def test_sampling_is_deterministic_with_same_seed(tmp_path):
    # SPEC: normalise_articles - "RNG: create ONE stateful object ... With the
    # same random_seed and source pool, the English articles selected in both
    # passes are identical."
    records = [
        _wikinews_record(text=f"Unique body {i} " * 30, article_id=f"id{i:03d}") for i in range(30)
    ]
    p = _write_jsonl(tmp_path, records)
    kwargs = dict(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=5,
        min_article_length=10,
        random_seed=42,
    )
    a1, _ = data_normalizer.normalise_articles(**kwargs)
    a2, _ = data_normalizer.normalise_articles(**kwargs)
    assert [a["id"] for a in a1] == [a["id"] for a in a2]


# ---------------------------------------------------------------------------
# Per-article robustness - a single bad record must not crash the function
# ---------------------------------------------------------------------------


def test_malformed_record_does_not_crash(tmp_path):
    # SPEC: Design principles - "A single bad record never crashes the pipeline."
    # SPEC: load_raw_records JSONL - "Skip blank lines and unparseable lines".
    p = tmp_path / "data.jsonl"
    p.write_text(
        "not valid json\n" + json.dumps(_wikinews_record()) + "\n" + "\n" + "{broken",
        encoding="utf-8",
    )
    articles, _dropped = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany"],
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1


# ---------------------------------------------------------------------------
# countries=None path (language-only grouping)
# ---------------------------------------------------------------------------


def test_countries_none_does_not_raise(tmp_path):
    # countries=None is valid; only countries=[] should raise.
    p = _write_jsonl(tmp_path, [_wikinews_record()])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=None,
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1


def test_countries_none_extracts_country_from_categories(tmp_path):
    # When countries=None, country is extracted from categories as metadata.
    p = _write_jsonl(tmp_path, [_wikinews_record(categories=["Sports", "Germany"])])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=None,
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1
    assert articles[0]["country"] == "germany"


def test_countries_none_country_empty_string_when_no_known_country(tmp_path):
    # When categories contain no recognised country, country defaults to "".
    p = _write_jsonl(tmp_path, [_wikinews_record(categories=["Sports", "SomeUnknownRegion"])])
    articles, _ = data_normalizer.normalise_articles(
        raw_path=str(p),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=None,
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 1
    assert articles[0]["country"] == ""


def test_countries_none_does_not_drop_by_country(tmp_path):
    # With countries=None, articles with no matching country must NOT be dropped.
    records = [
        _wikinews_record(
            text=f"Unique article text number {i}. " * 10, categories=["Sports", "France"]
        )
        for i in range(3)
    ]
    articles, dropped = data_normalizer.normalise_articles(
        raw_path=_write_jsonl(tmp_path, records),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=None,
        max_per_topic=10,
        min_article_length=100,
        random_seed=42,
    )
    assert len(articles) == 3
    assert not any(d.reason == "country_not_in_config" for d in dropped)


def test_countries_none_groups_by_language_topic_not_country(tmp_path):
    # max_per_topic must apply per (language, topic), not per (country, language, topic).
    # 6 records: 3 tagged "Germany", 3 tagged "France", all same language+topic.
    # max_per_topic=4 → should return 4 (not 3+3=6, and not split into two groups of 3).
    records = [
        _wikinews_record(text=f"Germany article {i}. " * 10, categories=["Sports", "Germany"])
        for i in range(3)
    ] + [
        _wikinews_record(text=f"France article {i}. " * 10, categories=["Sports", "France"])
        for i in range(3)
    ]
    articles, _ = data_normalizer.normalise_articles(
        raw_path=_write_jsonl(tmp_path, records),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=None,
        max_per_topic=4,
        min_article_length=10,
        random_seed=42,
    )
    assert len(articles) == 4


def test_countries_filter_still_groups_by_country_language_topic(tmp_path):
    # When countries is provided, each (country, language, topic) cell is capped
    # independently — so two countries can each contribute up to max_per_topic.
    records = [
        _wikinews_record(text=f"Germany article {i}. " * 10, categories=["Sports", "Germany"])
        for i in range(5)
    ] + [
        _wikinews_record(text=f"US article {i}. " * 10, categories=["Sports", "United States"])
        for i in range(5)
    ]
    articles, _ = data_normalizer.normalise_articles(
        raw_path=_write_jsonl(tmp_path, records),
        detected_format="jsonl",
        languages=["en"],
        topics=["Sports"],
        countries=["Germany", "United States"],
        max_per_topic=3,
        min_article_length=10,
        random_seed=42,
    )
    assert len(articles) == 6  # 3 per country cell
