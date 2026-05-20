"""Tests for src/summarizer.py.

SPEC: summarizer contains validate_summarization_config, load_summarization_pipeline,
summarize_article, summarize_articles, build_summary_quality_dataframe.
HuggingFace pipeline is mocked via the mock_summ_pipeline fixture.

Per SPEC "Functions intentionally not covered by unit tests":
- load_summarization_pipeline: thin wrapper around transformers.pipeline().
"""

import pandas as pd
import pytest

summarizer = pytest.importorskip("src.summarizer")


# ---------------------------------------------------------------------------
# validate_summarization_config
# ---------------------------------------------------------------------------


def test_validate_summarization_config_raises_when_min_ge_max(sample_config):
    # SPEC: validate_summarization_config - "Raise ValueError if
    # min_summary_length >= max_summary_length."
    sample_config["summarization"]["min_summary_length"] = 200
    sample_config["summarization"]["max_summary_length"] = 200
    with pytest.raises(ValueError):
        summarizer.validate_summarization_config(sample_config)


def test_validate_summarization_config_raises_when_min_gt_max(sample_config):
    # SPEC: validate_summarization_config - same rule with strict >.
    sample_config["summarization"]["min_summary_length"] = 300
    sample_config["summarization"]["max_summary_length"] = 100
    with pytest.raises(ValueError):
        summarizer.validate_summarization_config(sample_config)


def test_validate_summarization_config_accepts_valid(sample_config):
    # SPEC: validate_summarization_config - must not raise on valid config.
    sample_config["summarization"]["min_summary_length"] = 50
    sample_config["summarization"]["max_summary_length"] = 200
    summarizer.validate_summarization_config(sample_config)


# ---------------------------------------------------------------------------
# summarize_article - single article path
# ---------------------------------------------------------------------------


def test_summarize_article_empty_string_returns_none(mock_summ_pipeline):
    # SPEC: summarize_article - "If text is empty string: log warning and return None."
    result = summarizer.summarize_article(
        text="",
        summ_pipeline=mock_summ_pipeline,
        min_length=50,
        max_length=200,
    )
    assert result is None


def test_summarize_article_short_input_returns_none(mock_summ_pipeline):
    # SPEC: summarize_article - "If token_count < min_length, log a warning and
    # return None."
    mock_summ_pipeline.tokenizer.encode.return_value = list(range(10))  # < 50
    result = summarizer.summarize_article(
        text="too short",
        summ_pipeline=mock_summ_pipeline,
        min_length=50,
        max_length=200,
    )
    assert result is None


def test_summarize_article_pipeline_called_with_pinned_kwargs(mock_summ_pipeline):
    # SPEC: summarize_article "Pipeline call signature" - "pass the parameters to
    # summ_pipeline as keyword arguments named exactly truncation=True,
    # min_length=..., and max_length=... Do not rename them, alias them, or pass
    # them positionally - tests and downstream tooling assert against these exact
    # kwarg names."
    summarizer.summarize_article(
        text="Some long article text. " * 20,
        summ_pipeline=mock_summ_pipeline,
        min_length=50,
        max_length=200,
    )
    assert mock_summ_pipeline.called
    _args, kwargs = mock_summ_pipeline.call_args
    assert (
        kwargs.get("truncation") is True
    ), "Pipeline must be called with truncation=True (exact kwarg name)."
    assert (
        kwargs.get("min_length") == 50
    ), "Pipeline must be called with min_length= as a kwarg (HF name)."
    # max_length is now adapted to input length inside summarize_article: the caller
    # passes 200 as the upper cap; the effective value is in (min_length, 200].
    effective_max = kwargs.get("max_length")
    assert isinstance(
        effective_max, int
    ), "Pipeline must be called with max_length= as a kwarg (HF name)."
    assert (
        50 < effective_max <= 200
    ), f"max_length must be capped within (min_length, configured_max], got {effective_max}."


def test_summarize_article_pipeline_exception_returns_none(mock_summ_pipeline):
    # SPEC: summarize_article - "If pipeline raises: log error and return None."
    mock_summ_pipeline.side_effect = RuntimeError("boom")
    result = summarizer.summarize_article(
        text="long text " * 100,
        summ_pipeline=mock_summ_pipeline,
        min_length=50,
        max_length=200,
    )
    assert result is None


def test_summarize_article_returns_summary_text(mock_summ_pipeline):
    # SPEC: summarize_article - "Returns: Summary string ..."
    # The pipeline returns [{"summary_text": "..."}]; the function must extract
    # that string.
    mock_summ_pipeline.return_value = [{"summary_text": "Concise summary."}]
    result = summarizer.summarize_article(
        text="long text " * 100,
        summ_pipeline=mock_summ_pipeline,
        min_length=50,
        max_length=200,
    )
    assert result == "Concise summary."


def test_summarize_article_uses_add_special_tokens_false_for_token_count(
    mock_summ_pipeline,
):
    # SPEC: summarize_article - "Use add_special_tokens=False to count only
    # content tokens, not BOS/EOS tokens added by the tokeniser."
    summarizer.summarize_article(
        text="long text " * 100,
        summ_pipeline=mock_summ_pipeline,
        min_length=50,
        max_length=200,
    )
    found = False
    for call in mock_summ_pipeline.tokenizer.encode.call_args_list:
        kwargs = call.kwargs
        if kwargs.get("add_special_tokens") is False:
            found = True
            break
    assert found, "summarize_article must count tokens with add_special_tokens=False"


# ---------------------------------------------------------------------------
# summarize_articles - orchestration over a list
# ---------------------------------------------------------------------------


def test_summarize_articles_only_processes_configured_languages(
    mock_summ_pipeline,
    sample_en_article,
    sample_de_article,
    sample_config,
):
    # SPEC: summarize_articles - "Run summarize_article on articles whose
    # language is in config['languages']['summarization']."
    # config has summarization=["en"] only.
    sample_en_article["cleaned_text"] = sample_en_article["text"]
    sample_de_article["cleaned_text"] = sample_de_article["text"]
    articles = [sample_en_article, sample_de_article]
    out = summarizer.summarize_articles(articles, mock_summ_pipeline, sample_config)
    # English article got a summary key.
    en = next(a for a in out if a["language"] == "en")
    assert "summary" in en
    # German article did not - SPEC: "Non-qualifying articles ... do not receive
    # a 'summary' key at all".
    de = next(a for a in out if a["language"] == "de")
    assert "summary" not in de


def test_summarize_articles_summary_can_be_none_on_failure(
    mock_summ_pipeline,
    sample_en_article,
    sample_config,
):
    # SPEC: Article schema - '"summary": Optional[str], # None if summarisation
    # failed or short-article guard triggered.'
    sample_en_article["cleaned_text"] = sample_en_article["text"]
    mock_summ_pipeline.side_effect = RuntimeError("boom")
    out = summarizer.summarize_articles(
        [sample_en_article],
        mock_summ_pipeline,
        sample_config,
    )
    assert out[0]["summary"] is None


# ---------------------------------------------------------------------------
# build_summary_quality_dataframe
# ---------------------------------------------------------------------------

QUALITY_COLS = {
    "article_id",
    "title",
    "country",
    "topic",
    "summary_char_count",
    "summary_sentence_count",
    "avg_sentence_chars",
    "missing_terminal_punctuation",
    "repeated_whitespace",
    "very_long_sentence",
    "issue_count",
}


def test_build_summary_quality_dataframe_columns():
    # SPEC: build_summary_quality_dataframe - required column set.
    article = {
        "id": "1",
        "title": "t",
        "country": "germany",
        "topic": "sports",
        "summary": "A normal summary.",
    }
    df = summarizer.build_summary_quality_dataframe([article])
    assert QUALITY_COLS.issubset(set(df.columns))


def test_build_summary_quality_dataframe_flags_missing_terminal_punctuation():
    # SPEC: build_summary_quality_dataframe - "missing_terminal_punctuation (bool)".
    article = {
        "id": "1",
        "title": "t",
        "country": "germany",
        "topic": "sports",
        "summary": "This summary does not end with terminal punctuation",
    }
    df = summarizer.build_summary_quality_dataframe([article])
    assert bool(df.iloc[0]["missing_terminal_punctuation"]) is True


def test_build_summary_quality_dataframe_flags_repeated_whitespace():
    # SPEC: build_summary_quality_dataframe - "repeated_whitespace (bool)".
    article = {
        "id": "1",
        "title": "t",
        "country": "germany",
        "topic": "sports",
        "summary": "This  has   repeated   whitespace.",
    }
    df = summarizer.build_summary_quality_dataframe([article])
    assert bool(df.iloc[0]["repeated_whitespace"]) is True


def test_build_summary_quality_dataframe_flags_very_long_sentence():
    # SPEC: build_summary_quality_dataframe - "very_long_sentence (bool;
    # any sentence > 250 chars)".
    article = {
        "id": "1",
        "title": "t",
        "country": "germany",
        "topic": "sports",
        "summary": ("word " * 100).strip() + ".",  # ~500 chars in one sentence
    }
    df = summarizer.build_summary_quality_dataframe([article])
    assert bool(df.iloc[0]["very_long_sentence"]) is True


def test_build_summary_quality_dataframe_excludes_articles_with_no_summary():
    # SPEC: build_summary_quality_dataframe - "Include only articles where
    # article.get('summary') is not None."
    articles = [
        {
            "id": "1",
            "title": "t",
            "country": "germany",
            "topic": "sports",
            "summary": "Has summary.",
        },
        {"id": "2", "title": "t", "country": "germany", "topic": "sports", "summary": None},
        {"id": "3", "title": "t", "country": "germany", "topic": "sports"},
    ]
    df = summarizer.build_summary_quality_dataframe(articles)
    assert len(df) == 1
    assert df.iloc[0]["article_id"] == "1"


def test_build_summary_quality_dataframe_empty_when_no_summaries():
    # SPEC: build_summary_quality_dataframe - "Empty with correct columns if no
    # summaries exist."
    df = summarizer.build_summary_quality_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert QUALITY_COLS.issubset(set(df.columns))


def test_build_summary_quality_dataframe_substitutes_id_for_empty_title():
    # SPEC: build_summary_quality_dataframe - "if article['title'] == '',
    # use f'[id: {article['id']}]' as the display value."
    article = {
        "id": "ABC",
        "title": "",
        "country": "germany",
        "topic": "sports",
        "summary": "Has a summary.",
    }
    df = summarizer.build_summary_quality_dataframe([article])
    assert df.iloc[0]["title"] == "[id: ABC]"
