"""Tests for src/similarity.py.

SPEC: similarity contains load_embedding_model, calculate_similarity,
score_all_articles, build_similarity_dataframe, plot_similarity_distribution,
explain_similarity_extremes. SentenceTransformer is mocked.

Per SPEC "Functions intentionally not covered by unit tests":
- plot_similarity_distribution: matplotlib, reviewer-judged.
- load_embedding_model: thin wrapper around SentenceTransformer().
"""
import pandas as pd
import pytest

similarity = pytest.importorskip("src.similarity")


# ---------------------------------------------------------------------------
# calculate_similarity
# ---------------------------------------------------------------------------

def test_calculate_similarity_returns_python_float(mock_embedding_model):
    # SPEC: calculate_similarity — "Extract the scalar value with
    # float(cos_sim_result[0][0]) before returning."
    result = similarity.calculate_similarity(
        original="some original text",
        summary="some summary",
        model=mock_embedding_model,
    )
    assert isinstance(result, float), (
        "calculate_similarity must return a Python float, not a torch.Tensor."
    )


def test_calculate_similarity_value_in_theoretical_range(mock_embedding_model):
    # SPEC: calculate_similarity — "Cosine similarity is theoretically in [-1, 1]."
    result = similarity.calculate_similarity(
        original="some original text",
        summary="some summary",
        model=mock_embedding_model,
    )
    assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# score_all_articles
# ---------------------------------------------------------------------------

def test_score_all_articles_skips_articles_without_summary(
    mock_embedding_model, sample_en_article, sample_de_article,
):
    # SPEC: score_all_articles — "Run calculate_similarity on every article where
    # ... article.get('summary') is not None."
    # German article has no 'summary' key at all (non-qualifying language).
    sample_en_article["cleaned_text"] = sample_en_article["text"]
    sample_en_article["summary"] = "an english summary"
    sample_de_article["cleaned_text"] = sample_de_article["text"]
    # NOTE: no 'summary' key on German article.
    articles = [sample_en_article, sample_de_article]
    out = similarity.score_all_articles(articles, mock_embedding_model)
    en = next(a for a in out if a["language"] == "en")
    de = next(a for a in out if a["language"] == "de")
    assert "similarity_score" in en
    assert "similarity_score" not in de


def test_score_all_articles_skips_articles_with_none_summary(
    mock_embedding_model, sample_en_article,
):
    # SPEC: score_all_articles — "article.get('summary') is not None" is the gate.
    sample_en_article["cleaned_text"] = sample_en_article["text"]
    sample_en_article["summary"] = None
    out = similarity.score_all_articles([sample_en_article], mock_embedding_model)
    assert "similarity_score" not in out[0]


def test_score_all_articles_per_article_error_does_not_crash(
    mock_embedding_model, sample_en_article,
):
    # SPEC: score_all_articles — "Per-article errors: catch, log, leave
    # 'similarity_score' unset."
    sample_en_article["cleaned_text"] = sample_en_article["text"]
    sample_en_article["summary"] = "summary text"
    mock_embedding_model.encode.side_effect = RuntimeError("boom")
    try:
        out = similarity.score_all_articles(
            [sample_en_article], mock_embedding_model
        )
    except Exception as exc:
        pytest.fail(
            f"score_all_articles must catch per-article errors, got: {exc!r}"
        )
    assert "similarity_score" not in out[0]


# ---------------------------------------------------------------------------
# build_similarity_dataframe
# ---------------------------------------------------------------------------

SIMILARITY_COLS = {"article_id", "title", "country", "topic", "similarity_score"}


def test_build_similarity_dataframe_columns_when_scored():
    # SPEC: build_similarity_dataframe — "Columns: article_id (str), title (str),
    # country (str), topic (str), similarity_score (float)."
    articles = [{
        "id": "1", "title": "t", "country": "germany", "topic": "sports",
        "similarity_score": 0.87,
    }]
    df = similarity.build_similarity_dataframe(articles)
    assert SIMILARITY_COLS.issubset(set(df.columns))
    assert len(df) == 1


def test_build_similarity_dataframe_empty_when_no_scored_articles():
    # SPEC test list: "build_similarity_dataframe on articles with no summaries:
    # returns empty DataFrame with correct columns".
    articles = [
        {"id": "1", "title": "t", "country": "germany", "topic": "sports"},
    ]
    df = similarity.build_similarity_dataframe(articles)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert SIMILARITY_COLS.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# explain_similarity_extremes — deterministic tie-breaking
# ---------------------------------------------------------------------------

def test_explain_similarity_extremes_returns_required_keys():
    # SPEC: explain_similarity_extremes — "Return: {'highest': ..., 'lowest': ...}".
    df = pd.DataFrame([
        {"article_id": "1", "title": "a", "country": "germany",
         "topic": "sports", "similarity_score": 0.1},
        {"article_id": "2", "title": "b", "country": "germany",
         "topic": "sports", "similarity_score": 0.9},
    ])
    result = similarity.explain_similarity_extremes(df, n=1)
    assert set(result.keys()) == {"highest", "lowest"}
    assert len(result["highest"]) == 1
    assert len(result["lowest"]) == 1


def test_explain_similarity_extremes_deterministic_with_ties():
    # SPEC: explain_similarity_extremes — "Tie-breaking: sort by str(article_id)
    # lexicographically for deterministic output."
    df = pd.DataFrame([
        {"article_id": "zzz", "title": "z", "country": "germany",
         "topic": "sports", "similarity_score": 0.5},
        {"article_id": "aaa", "title": "a", "country": "germany",
         "topic": "sports", "similarity_score": 0.5},
        {"article_id": "mmm", "title": "m", "country": "germany",
         "topic": "sports", "similarity_score": 0.5},
    ])
    result1 = similarity.explain_similarity_extremes(df, n=3)
    result2 = similarity.explain_similarity_extremes(df, n=3)
    ids1 = [r["article_id"] for r in result1["highest"]]
    ids2 = [r["article_id"] for r in result2["highest"]]
    assert ids1 == ids2, "ties must resolve to the same order on every call"


def test_explain_similarity_extremes_result_has_required_fields():
    # SPEC: explain_similarity_extremes — each result has keys article_id, title,
    # country, topic, similarity_score.
    df = pd.DataFrame([
        {"article_id": "1", "title": "a", "country": "germany",
         "topic": "sports", "similarity_score": 0.5},
    ])
    result = similarity.explain_similarity_extremes(df, n=1)
    assert set(result["highest"][0].keys()) >= {
        "article_id", "title", "country", "topic", "similarity_score",
    }
