"""Tests for src/topic_predictor.py.

SPEC: topic_predictor contains load_topic_pipeline, predict_topic,
predict_all_topics, evaluate_topic_predictions. The zero-shot pipeline
is mocked via the mock_topic_pipeline fixture.

Per SPEC "Functions intentionally not covered by unit tests":
- load_topic_pipeline: thin wrapper around transformers.pipeline().
"""
import pytest

topic_predictor = pytest.importorskip("src.topic_predictor")


# ---------------------------------------------------------------------------
# predict_topic
# ---------------------------------------------------------------------------

def test_predict_topic_empty_text_returns_none(mock_topic_pipeline):
    # SPEC: predict_topic — "If text is empty or pipeline raises: log warning
    # and return None."
    result = topic_predictor.predict_topic(
        text="",
        candidate_labels=["Sports", "Politics and conflicts"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
    )
    assert result is None


def test_predict_topic_pipeline_exception_returns_none(mock_topic_pipeline):
    # SPEC: predict_topic — same rule on pipeline exception.
    mock_topic_pipeline.side_effect = RuntimeError("boom")
    result = topic_predictor.predict_topic(
        text="A real article about sports.",
        candidate_labels=["Sports", "Politics and conflicts"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
    )
    assert result is None


def test_predict_topic_returns_highest_scoring_label(mock_topic_pipeline):
    # SPEC: predict_topic — "Return the label with the highest score."
    mock_topic_pipeline.return_value = {
        "sequence": "x",
        "labels": ["Politics and conflicts", "Sports", "Science and technology"],
        "scores": [0.7, 0.2, 0.1],
    }
    result = topic_predictor.predict_topic(
        text="A real article.",
        candidate_labels=["Politics and conflicts", "Sports", "Science and technology"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
    )
    assert result == "Politics and conflicts"


def test_predict_topic_passes_hypothesis_template(mock_topic_pipeline):
    # SPEC: predict_topic — "Pass hypothesis_template to the pipeline."
    template = "This news article is about {}."
    topic_predictor.predict_topic(
        text="A real article.",
        candidate_labels=["Sports"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template=template,
    )
    assert mock_topic_pipeline.called
    _, kwargs = mock_topic_pipeline.call_args
    assert kwargs.get("hypothesis_template") == template


# ---------------------------------------------------------------------------
# predict_all_topics — sampling determinism, original-list immutability
# ---------------------------------------------------------------------------

def _make_articles(n, language="en", country="germany", topic="sports"):
    out = []
    for i in range(n):
        out.append({
            "id": f"id{i:03d}",
            "title": f"Article {i}",
            "text": f"Body {i}",
            "cleaned_text": f"Body {i}. Sports content. " * 5,
            "language": language,
            "country": country,
            "topic": topic,
            "date": "2021-01-01",
            "event_id": None,
        })
    return out


def test_predict_all_topics_deterministic_with_same_seed(mock_topic_pipeline):
    # SPEC: predict_all_topics — "create ONE stateful RNG object ... Use this
    # single rng instance for ALL sampling calls."
    articles = _make_articles(20)
    out1 = topic_predictor.predict_all_topics(
        articles, candidate_labels=["Sports"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
        sample_size=5, random_seed=42,
    )
    out2 = topic_predictor.predict_all_topics(
        articles, candidate_labels=["Sports"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
        sample_size=5, random_seed=42,
    )
    assert [a["id"] for a in out1] == [a["id"] for a in out2]


def test_predict_all_topics_does_not_mutate_originals(mock_topic_pipeline):
    # SPEC: predict_all_topics — "Create a shallow copy of each sampled article
    # dict before adding 'predicted_topic' ... This prevents mutation of the
    # original articles list."
    # SPEC: Article schema — "Added by topic_predictor.predict_all_topics() to
    # copied sampled articles only ... The original article list is not mutated
    # and unsampled originals do not receive this key."
    articles = _make_articles(5)
    topic_predictor.predict_all_topics(
        articles, candidate_labels=["Sports"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
        sample_size=5, random_seed=42,
    )
    for a in articles:
        assert "predicted_topic" not in a, (
            "predict_all_topics must not mutate the original article list."
        )


def test_predict_all_topics_adds_predicted_topic_to_returned(mock_topic_pipeline):
    # SPEC: predict_all_topics — "Return only the copied, sampled articles."
    articles = _make_articles(5)
    out = topic_predictor.predict_all_topics(
        articles, candidate_labels=["Sports"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
        sample_size=5, random_seed=42,
    )
    for a in out:
        assert "predicted_topic" in a


def test_predict_all_topics_excludes_articles_without_cleaned_text(
    mock_topic_pipeline,
):
    # SPEC: predict_all_topics — "Pre-filter before sampling: exclude articles
    # where 'cleaned_text' is absent or empty."
    articles = _make_articles(5)
    articles[0].pop("cleaned_text")           # absent
    articles[1]["cleaned_text"] = ""          # empty
    out = topic_predictor.predict_all_topics(
        articles, candidate_labels=["Sports"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
        sample_size=10, random_seed=42,
    )
    out_ids = {a["id"] for a in out}
    assert articles[0]["id"] not in out_ids
    assert articles[1]["id"] not in out_ids


def test_predict_all_topics_per_article_error_sets_none(mock_topic_pipeline):
    # SPEC: predict_all_topics — "Per-article errors: catch, log, set
    # 'predicted_topic' = None on that article's copy."
    articles = _make_articles(3)
    mock_topic_pipeline.side_effect = RuntimeError("boom")
    try:
        out = topic_predictor.predict_all_topics(
            articles, candidate_labels=["Sports"],
            topic_pipeline=mock_topic_pipeline,
            hypothesis_template="This news article is about {}.",
            sample_size=3, random_seed=42,
        )
    except Exception as exc:
        pytest.fail(f"predict_all_topics must catch per-article errors: {exc!r}")
    for a in out:
        assert a["predicted_topic"] is None


def test_predict_all_topics_redistribution_fills_to_sample_size(mock_topic_pipeline):
    # SPEC: predict_all_topics — "Redistribution pass: after the initial per-group
    # pass, compute: remaining_slots = sample_size - len(selected_articles) ...
    # collect groups that still have unsampled articles ... draw min(per_group_extra,
    # available_unsampled) articles using the same rng object."
    # Bucket B test: with two (country, topic) groups and an imbalanced pool, the
    # redistribution pass must fill the remaining slots from the larger group.
    # quota = floor(sample_size / 2) = 5; small group has 3 → contributes 3;
    # remaining_slots = 10 - 5 - 3 = 2 must be filled from the large group.
    small = _make_articles(3, country="germany", topic="sports")
    large = _make_articles(20, country="united states", topic="sports")
    for i, a in enumerate(large):
        a["id"] = f"us{i:03d}"  # disambiguate ids across groups
    articles = small + large
    out = topic_predictor.predict_all_topics(
        articles, candidate_labels=["Sports"],
        topic_pipeline=mock_topic_pipeline,
        hypothesis_template="This news article is about {}.",
        sample_size=10, random_seed=42,
    )
    assert len(out) == 10, (
        "Redistribution must fill remaining slots when a group is undersized "
        "and another has surplus articles."
    )


# ---------------------------------------------------------------------------
# evaluate_topic_predictions
# ---------------------------------------------------------------------------

def test_evaluate_topic_predictions_case_insensitive_match():
    # SPEC: evaluate_topic_predictions — "Comparison MUST normalise both sides:
    # lower().strip() on both values before comparing."
    sampled = [{
        "id": "1", "title": "t", "country": "germany",
        "topic": "sports",                # lowercase from normalise_articles
        "predicted_topic": "Sports",      # original casing from config
    }]
    result = topic_predictor.evaluate_topic_predictions(sampled)
    # SPEC: "A direct equality check ... will ALWAYS return False" — a correct
    # implementation MUST produce match=True here.
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["results"][0]["match"] is True


def test_evaluate_topic_predictions_all_correct_gives_accuracy_one():
    # SPEC test list: "evaluate_topic_predictions with all correct predictions:
    # accuracy=1.0".
    sampled = [
        {"id": "1", "title": "t", "country": "germany",
         "topic": "sports", "predicted_topic": "Sports"},
        {"id": "2", "title": "t", "country": "united states",
         "topic": "politics and conflicts",
         "predicted_topic": "Politics and conflicts"},
    ]
    result = topic_predictor.evaluate_topic_predictions(sampled)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["correct"] == 2
    assert result["evaluated"] == 2


def test_evaluate_topic_predictions_excludes_none_from_denominator():
    # SPEC: evaluate_topic_predictions — "accuracy: float, # correct / evaluated
    # (excludes None predictions)".
    sampled = [
        {"id": "1", "title": "t", "country": "germany",
         "topic": "sports", "predicted_topic": "Sports"},
        {"id": "2", "title": "t", "country": "germany",
         "topic": "sports", "predicted_topic": None},
    ]
    result = topic_predictor.evaluate_topic_predictions(sampled)
    assert result["evaluated"] == 1
    assert result["total_sampled"] == 2
    assert result["accuracy"] == pytest.approx(1.0)


def test_evaluate_topic_predictions_zero_correct_gives_zero_accuracy():
    # SPEC: evaluate_topic_predictions — denominator excludes None, so a
    # population of one wrong prediction must yield accuracy=0.0.
    sampled = [
        {"id": "1", "title": "t", "country": "germany",
         "topic": "sports", "predicted_topic": "Politics and conflicts"},
    ]
    result = topic_predictor.evaluate_topic_predictions(sampled)
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["correct"] == 0
    assert result["evaluated"] == 1


def test_evaluate_topic_predictions_empty_evaluation_returns_zero_accuracy():
    # SPEC: evaluate_topic_predictions "Empty-evaluation rule" — "when
    # evaluated == 0 (every prediction was None, or sampled_articles is empty),
    # accuracy is 0.0 — not NaN. ... Do not return float('nan') or None."
    import math

    result_empty = topic_predictor.evaluate_topic_predictions([])
    assert result_empty["evaluated"] == 0
    assert result_empty["total_sampled"] == 0
    assert result_empty["accuracy"] == 0.0
    assert not math.isnan(result_empty["accuracy"]), (
        "accuracy must be 0.0 when evaluated==0, not NaN."
    )

    # Also the case where every prediction is None.
    all_none = [
        {"id": "1", "title": "t", "country": "germany",
         "topic": "sports", "predicted_topic": None},
        {"id": "2", "title": "t", "country": "germany",
         "topic": "sports", "predicted_topic": None},
    ]
    result_all_none = topic_predictor.evaluate_topic_predictions(all_none)
    assert result_all_none["evaluated"] == 0
    assert result_all_none["accuracy"] == 0.0
    assert not math.isnan(result_all_none["accuracy"])


def test_evaluate_topic_predictions_substitutes_id_for_empty_title():
    # SPEC: evaluate_topic_predictions — "Title handling: use the same
    # substitution as build_entity_dataframe — if article['title'] == '',
    # use f'[id: {article['id']}]' as the display value."
    sampled = [{
        "id": "ABC", "title": "", "country": "germany",
        "topic": "sports", "predicted_topic": "Sports",
    }]
    result = topic_predictor.evaluate_topic_predictions(sampled)
    assert result["results"][0]["title"] == "[id: ABC]"


def test_evaluate_topic_predictions_returns_required_keys():
    # SPEC: evaluate_topic_predictions — return dict with accuracy, correct,
    # evaluated, total_sampled, results.
    result = topic_predictor.evaluate_topic_predictions([])
    assert set(result.keys()) >= {
        "accuracy", "correct", "evaluated", "total_sampled", "results",
    }
