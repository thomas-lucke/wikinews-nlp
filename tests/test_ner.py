"""Tests for src/ner.py.

SPEC: ner contains validate_ner_config, load_ner_pipeline, _chunk_text,
_resolve_overlapping_entities, run_ner, build_entity_dataframe, plotting
helpers, and investigate_ner_errors. All HuggingFace pipeline interaction
is mocked; mock returns the RAW format (entity_group, word) so the
key-rename in run_ner is exercised.

Per SPEC "Functions intentionally not covered by unit tests":
- plot_top_entities, plot_entity_dynamics: matplotlib, reviewer-judged.
- load_ner_pipeline: thin wrapper around transformers.pipeline().
"""

import pandas as pd
import pytest

ner = pytest.importorskip("src.ner")


# ---------------------------------------------------------------------------
# validate_ner_config - pipeline-level config validation
# ---------------------------------------------------------------------------


def test_validate_ner_config_raises_on_zero_chunk_size(sample_config):
    # SPEC: validate_ner_config - "Raise ValueError if config['ner']['chunk_size'] <= 0."
    sample_config["ner"]["chunk_size"] = 0
    with pytest.raises(ValueError):
        ner.validate_ner_config(sample_config)


def test_validate_ner_config_raises_on_negative_chunk_size(sample_config):
    # SPEC: validate_ner_config - "Raise ValueError if ... chunk_size <= 0."
    sample_config["ner"]["chunk_size"] = -10
    with pytest.raises(ValueError):
        ner.validate_ner_config(sample_config)


def test_validate_ner_config_raises_on_overlap_ge_chunk_size(sample_config):
    # SPEC: validate_ner_config - "Raise ValueError if chunk_overlap >= chunk_size."
    sample_config["ner"]["chunk_size"] = 100
    sample_config["ner"]["chunk_overlap"] = 100
    with pytest.raises(ValueError):
        ner.validate_ner_config(sample_config)


def test_validate_ner_config_raises_on_negative_overlap(sample_config):
    # SPEC: validate_ner_config - "Raise ValueError if chunk_overlap < 0."
    sample_config["ner"]["chunk_overlap"] = -1
    with pytest.raises(ValueError):
        ner.validate_ner_config(sample_config)


def test_validate_ner_config_accepts_valid(sample_config):
    # SPEC: validate_ner_config - must not raise on valid config.
    sample_config["ner"]["chunk_size"] = 400
    sample_config["ner"]["chunk_overlap"] = 50
    ner.validate_ner_config(sample_config)


# ---------------------------------------------------------------------------
# _chunk_text - pure function
# ---------------------------------------------------------------------------


def test_chunk_text_empty_returns_empty_list():
    # SPEC: _chunk_text - "Empty text: return empty list."
    assert ner._chunk_text("", 100, 10) == []


def test_chunk_text_raises_on_non_positive_size():
    # SPEC: _chunk_text - "chunk_size <= 0: raise ValueError."
    with pytest.raises(ValueError):
        ner._chunk_text("hello", 0, 0)
    with pytest.raises(ValueError):
        ner._chunk_text("hello", -1, 0)


def test_chunk_text_raises_on_overlap_ge_chunk_size():
    # SPEC: _chunk_text - "overlap >= chunk_size: raise ValueError (would cause
    # infinite loop)."
    with pytest.raises(ValueError):
        ner._chunk_text("hello world " * 50, 50, 50)


def test_chunk_text_returns_tuples_of_str_and_int():
    # SPEC: _chunk_text - "Returns list of (chunk_text, start_offset) where
    # start_offset is the character position of the chunk's first character ..."
    text = "Berlin is great. " * 50
    chunks = ner._chunk_text(text, 100, 20)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, tuple) and len(chunk) == 2
        chunk_str, offset = chunk
        assert isinstance(chunk_str, str)
        assert isinstance(offset, int)
        assert text[offset : offset + len(chunk_str)] == chunk_str


def test_chunk_text_short_input_returns_single_chunk():
    # SPEC: _chunk_text step 2 - "If remaining text fits within chunk_size: add
    # as final chunk and stop."
    text = "short input"
    chunks = ner._chunk_text(text, 100, 10)
    assert len(chunks) == 1
    assert chunks[0][0].strip() == text.strip()
    assert chunks[0][1] == 0


# ---------------------------------------------------------------------------
# _resolve_overlapping_entities - pure function on entity dicts
# ---------------------------------------------------------------------------


def test_resolve_overlapping_entities_deduplicates_exact_duplicates():
    # SPEC: _resolve_overlapping_entities - "Exact duplicates: same (start, end,
    # label) - keep the one with higher score."
    text = "Hello Berlin today and Berlin again."
    entities = [
        {"text": "Berlin", "label": "LOC", "start": 6, "end": 12, "score": 0.7},
        {"text": "Berlin", "label": "LOC", "start": 6, "end": 12, "score": 0.99},
    ]
    out = ner._resolve_overlapping_entities(entities, text)
    assert len(out) == 1
    assert out[0]["score"] == pytest.approx(0.99)


def test_resolve_overlapping_entities_keeps_longer_span():
    # SPEC: _resolve_overlapping_entities - "Partial overlaps ... keep the
    # entity with the larger (end - start) value."
    text = "Visited New York City last week."
    entities = [
        {"text": "New York", "label": "LOC", "start": 8, "end": 16, "score": 0.95},
        {"text": "New York City", "label": "LOC", "start": 8, "end": 21, "score": 0.92},
    ]
    out = ner._resolve_overlapping_entities(entities, text)
    assert len(out) == 1
    assert out[0]["end"] - out[0]["start"] == 13  # the longer span


def test_resolve_overlapping_entities_corrects_text_from_offsets():
    # SPEC: _resolve_overlapping_entities - "Offset validation: in-bounds
    # entities have their text canonicalised from cleaned_text[start:end];
    # the lossy pipeline `.word` is not trusted." (FIX-10)
    text = "Hello Berlin today."
    entities = [
        # `.word` is the lossy "##erlin"; offsets 6-12 correctly point to "Berlin".
        {"text": "##erlin", "label": "LOC", "start": 6, "end": 12, "score": 0.9},
    ]
    out = ner._resolve_overlapping_entities(entities, text)
    assert len(out) == 1
    assert out[0]["text"] == "Berlin", "text must be corrected to the source slice"


def test_resolve_overlapping_entities_discards_out_of_bounds_offsets():
    # SPEC: _resolve_overlapping_entities - "Out-of-bounds offsets indicate a
    # genuine bug and are discarded with a warning." (FIX-10)
    text = "Hello Berlin today."
    entities = [
        # start/end far past the end of the text - must be discarded.
        {"text": "Berlin", "label": "LOC", "start": 100, "end": 110, "score": 0.9},
    ]
    out = ner._resolve_overlapping_entities(entities, text)
    assert out == []


def test_resolve_overlapping_entities_sorts_by_start_ascending():
    # SPEC: _resolve_overlapping_entities - "Sort final list by start offset ascending."
    text = "Apple. Berlin. Charlie."
    entities = [
        {"text": "Charlie", "label": "PER", "start": 15, "end": 22, "score": 0.9},
        {"text": "Apple", "label": "ORG", "start": 0, "end": 5, "score": 0.9},
        {"text": "Berlin", "label": "LOC", "start": 7, "end": 13, "score": 0.9},
    ]
    out = ner._resolve_overlapping_entities(entities, text)
    starts = [e["start"] for e in out]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# run_ner - interaction with HuggingFace pipeline (mocked)
# ---------------------------------------------------------------------------


def test_run_ner_skips_articles_of_other_language(mock_ner_pipeline, sample_de_article):
    # SPEC: run_ner - "Only process articles where article['language'] == language."
    # SPEC: Article schema - "entities: Optional[list[dict]] ... None if NER not run."
    sample_de_article["cleaned_text"] = sample_de_article["text"]
    articles = [sample_de_article]
    out = ner.run_ner(articles, mock_ner_pipeline, language="en", chunk_size=400, chunk_overlap=50)
    # The German article was not processed for the English pass.
    assert out[0].get("entities") is None or "entities" not in out[0]
    mock_ner_pipeline.assert_not_called()


def test_run_ner_renames_entity_group_to_label(mock_ner_pipeline, sample_en_article):
    # SPEC: run_ner - "run_ner MUST rename this key when building each entity dict:
    # entity['label'] = raw_entity['entity_group']."
    cleaned = "Hello Berlin everyone."
    sample_en_article["cleaned_text"] = cleaned
    mock_ner_pipeline.return_value = [
        {"word": "Berlin", "entity_group": "LOC", "score": 0.99, "start": 6, "end": 12},
    ]
    out = ner.run_ner(
        [sample_en_article], mock_ner_pipeline, language="en", chunk_size=400, chunk_overlap=50
    )
    entities = out[0]["entities"]
    assert len(entities) == 1
    ent = entities[0]
    assert "label" in ent, "run_ner must rename entity_group → label"
    assert ent["label"] == "LOC"
    # SPEC: Entity schema fields must include text, label, start, end, score.
    for key in ("text", "label", "start", "end", "score"):
        assert key in ent


def test_run_ner_empty_pipeline_result_sets_empty_list(
    mock_ner_pipeline,
    sample_en_article,
):
    # SPEC: run_ner - "Set entities=[] (not None) for articles where NER ran
    # but found nothing".
    sample_en_article["cleaned_text"] = sample_en_article["text"]
    mock_ner_pipeline.return_value = []
    out = ner.run_ner(
        [sample_en_article], mock_ner_pipeline, language="en", chunk_size=400, chunk_overlap=50
    )
    assert out[0]["entities"] == []


def test_run_ner_missing_cleaned_text_sets_empty_list(
    mock_ner_pipeline,
    sample_en_article,
):
    # SPEC: run_ner - "For articles where 'cleaned_text' is missing
    # (preprocessing failed): set entities=[] and log a warning."
    sample_en_article.pop("cleaned_text", None)
    out = ner.run_ner(
        [sample_en_article], mock_ner_pipeline, language="en", chunk_size=400, chunk_overlap=50
    )
    assert out[0]["entities"] == []


def test_run_ner_per_article_error_does_not_crash(
    mock_ner_pipeline,
    sample_en_article,
    sample_de_article,
):
    # SPEC: run_ner - "Per-article errors: catch, log with article ID, set
    # entities=[], continue."
    sample_en_article["cleaned_text"] = "x"
    sample_de_article["cleaned_text"] = "y"

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    mock_ner_pipeline.side_effect = explode

    articles = [sample_en_article]
    try:
        out = ner.run_ner(
            articles, mock_ner_pipeline, language="en", chunk_size=400, chunk_overlap=50
        )
    except Exception as exc:
        pytest.fail(f"run_ner must catch per-article errors, got: {exc!r}")
    assert out[0]["entities"] == []


# ---------------------------------------------------------------------------
# build_entity_dataframe - schema and skip semantics
# ---------------------------------------------------------------------------

EXPECTED_ENTITY_DF_COLS = {
    "article_id",
    "event_id",
    "title",
    "date",
    "language",
    "topic",
    "entity_text",
    "entity_label",
    "score",
}


def test_build_entity_dataframe_columns(sample_en_article):
    # SPEC: build_entity_dataframe - "Columns: article_id, event_id, title, date,
    # language, topic, entity_text, entity_label, score".
    sample_en_article["entities"] = [
        {"text": "Berlin", "label": "LOC", "start": 0, "end": 6, "score": 0.99},
    ]
    df = ner.build_entity_dataframe([sample_en_article])
    assert EXPECTED_ENTITY_DF_COLS.issubset(set(df.columns))


def test_build_entity_dataframe_one_row_per_entity(sample_en_article):
    # SPEC: build_entity_dataframe - "Flatten all entities from all articles into
    # one row per entity occurrence."
    # SPEC test list: "build_entity_dataframe on article with 3 entities:
    # DataFrame has 3 rows"
    sample_en_article["entities"] = [
        {"text": "Berlin", "label": "LOC", "start": 0, "end": 6, "score": 0.9},
        {"text": "Schmidt", "label": "PER", "start": 7, "end": 14, "score": 0.9},
        {"text": "CDU", "label": "ORG", "start": 15, "end": 18, "score": 0.9},
    ]
    df = ner.build_entity_dataframe([sample_en_article])
    assert len(df) == 3


def test_build_entity_dataframe_skips_articles_without_entities_key(sample_en_article):
    # SPEC: build_entity_dataframe - "articles where 'entities' key is absent: skip".
    sample_en_article.pop("entities", None)
    df = ner.build_entity_dataframe([sample_en_article])
    assert len(df) == 0


def test_build_entity_dataframe_skips_none_entities(sample_en_article):
    # SPEC: build_entity_dataframe - "articles where entities is None: skip
    # (NER was not run for this article's language)".
    sample_en_article["entities"] = None
    df = ner.build_entity_dataframe([sample_en_article])
    assert len(df) == 0


def test_build_entity_dataframe_includes_empty_list_with_zero_rows(sample_en_article):
    # SPEC: build_entity_dataframe - "articles where entities is [] (empty list):
    # include in iteration, produces zero rows".
    sample_en_article["entities"] = []
    df = ner.build_entity_dataframe([sample_en_article])
    assert len(df) == 0


def test_build_entity_dataframe_substitutes_id_for_empty_title():
    # SPEC: build_entity_dataframe - "When title == '', substitute article['id']
    # as the display value: display_title = ... f'[id: {article['id']}]'."
    article = {
        "id": "X1",
        "event_id": None,
        "title": "",
        "date": None,
        "language": "en",
        "country": "germany",
        "topic": "sports",
        "entities": [
            {"text": "Berlin", "label": "LOC", "start": 0, "end": 6, "score": 0.9},
        ],
    }
    df = ner.build_entity_dataframe([article])
    assert df.iloc[0]["title"] == "[id: X1]"


def test_build_entity_dataframe_empty_input_returns_empty_with_columns():
    # SPEC: build_entity_dataframe - "Empty DataFrame (with correct columns)
    # if no entities found."
    df = ner.build_entity_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert EXPECTED_ENTITY_DF_COLS.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# investigate_ner_errors
# ---------------------------------------------------------------------------


def test_investigate_ner_errors_threshold(sample_en_article):
    # SPEC: investigate_ner_errors - "An entity is flagged as a candidate error
    # if score < error_score_threshold." Sorted ascending by score.
    sample_en_article["entities"] = [
        {"text": "Low", "label": "PER", "start": 0, "end": 3, "score": 0.30},
        {"text": "Mid", "label": "ORG", "start": 4, "end": 7, "score": 0.55},
        {"text": "High", "label": "LOC", "start": 8, "end": 12, "score": 0.95},
    ]
    df = ner.investigate_ner_errors(
        [sample_en_article],
        language="en",
        error_score_threshold=0.6,
    )
    # 0.30 and 0.55 below 0.60 → 2 rows; 0.95 above → excluded.
    assert len(df) == 2
    scores = df["score"].tolist()
    assert scores == sorted(scores), "must sort by score ascending"


def test_investigate_ner_errors_does_not_filter_by_misc_label_alone(sample_en_article):
    # SPEC: investigate_ner_errors - "MISC label alone is NOT treated as an error
    # - it is a valid label class."
    sample_en_article["entities"] = [
        # High score MISC → should NOT appear despite the MISC label.
        {"text": "Thing", "label": "MISC", "start": 0, "end": 5, "score": 0.99},
    ]
    df = ner.investigate_ner_errors(
        [sample_en_article],
        language="en",
        error_score_threshold=0.6,
    )
    assert len(df) == 0
