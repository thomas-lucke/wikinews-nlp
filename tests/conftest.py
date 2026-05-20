"""Shared fixtures for the wikinews-nlp test suite.

Tests are written before the implementation exists. Files use
``pytest.importorskip("src.<module>")`` at the top so collection succeeds
even when the implementation is absent - tests are skipped rather than
erroring at collection.
"""

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Article fixtures - shapes match SPEC "Article schema" exactly.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_en_article() -> dict:
    """Minimal valid English article after normalisation.

    SPEC: Article schema - fields guaranteed after normalise_articles().
    """
    return {
        "id": "abc123",
        "title": "Scientists discover new element",
        "text": (
            "Researchers at the University of Berlin announced today that they "
            "have discovered a new chemical element. The element, temporarily "
            "named Berlinium, was found during experiments at the particle "
            "accelerator. Professor Hans Schmidt led the research team."
        ),
        "language": "en",
        "country": "germany",
        "topic": "science and technology",
        "date": "2021-06-15",
        "event_id": None,
    }


@pytest.fixture
def sample_de_article() -> dict:
    """Minimal valid German article after normalisation.

    SPEC: Article schema - language is ISO 639-1 lowercase ("en" or "de").
    """
    return {
        "id": "def456",
        "title": "Bundestagswahl 2021",
        "text": (
            "Die Bundestagswahl findet am 26. September 2021 statt. "
            "Kanzlerkandidat Olaf Scholz der SPD liegt in Umfragen vorne. "
            "Angela Merkel tritt nach 16 Jahren als Bundeskanzlerin nicht "
            "mehr an. Die CDU/CSU kämpft um den Erhalt der Regierungsmacht."
        ),
        "language": "de",
        "country": "germany",
        "topic": "politics and conflicts",
        "date": "2021-09-01",
        "event_id": None,
    }


@pytest.fixture
def sample_raw_record() -> dict:
    """A raw record with non-standard field names, for normaliser tests.

    SPEC: FIELD_MAPPINGS - article_body→text, headline→title, publish_date→date,
    lang→language, category→topic, country→country, article_id→id, pageid→event_id.
    """
    return {
        "article_body": (
            "This is the article text. It is longer than one hundred "
            "characters to pass the minimum length check in the normaliser."
        ),
        "headline": "Test Article",
        "publish_date": "2021-03-10",
        "lang": "en",
        "category": "Sports",
        "country": "Germany",
        "article_id": "raw001",
        "pageid": 12345,
    }


@pytest.fixture
def sample_config() -> dict:
    """A minimal config matching the schema in SPEC ``config.yaml``."""
    return {
        "data": {
            "source_url": "https://example.com/data",
            "raw_path": "data/raw",
            "min_article_length": 100,
        },
        "topics": {
            "selected": ["Politics and conflicts", "Science and technology", "Sports"],
            "articles_per_topic_min": 10,
            "articles_per_topic_max": 20,
        },
        "countries": {"selected": ["United States", "Germany"]},
        "languages": {"ner": ["en", "de"], "summarization": ["en"]},
        "models": {
            "ner_english": "dslim/bert-base-NER",
            "ner_german": "Davlan/bert-base-multilingual-cased-ner-hrl",
            "spacy_english": "en_core_web_sm",
            "spacy_german": "de_core_news_sm",
            "summarization": "facebook/bart-large-cnn",
            "similarity": "sentence-transformers/all-MiniLM-L6-v2",
            "topic_prediction": "facebook/bart-large-mnli",
        },
        "summarization": {"min_summary_length": 50, "max_summary_length": 200},
        "ner": {"chunk_size": 400, "chunk_overlap": 50, "error_score_threshold": 0.6},
        "topic_prediction": {
            "sample_size": 30,
            "hypothesis_template": "This news article is about {}.",
        },
        "similarity": {"threshold": 0.8},
        "logging": {"log_file": "logs/pipeline.log"},
        "random_seed": 42,
    }


# ---------------------------------------------------------------------------
# Mocked external dependencies - shapes MUST match real library returns.
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ner_pipeline():
    """Mock HuggingFace NER pipeline.

    SPEC: ner.run_ner - "HuggingFace NER pipelines with aggregation_strategy='simple'
    return dicts with key 'entity_group', not 'label'. run_ner MUST rename this key."
    Mock returns the RAW pre-rename format so tests of run_ner exercise the rename logic.
    """
    pipeline = MagicMock()
    pipeline.return_value = [
        {"word": "Berlin", "entity_group": "LOC", "score": 0.98, "start": 5, "end": 11},
    ]
    return pipeline


@pytest.fixture
def mock_summ_pipeline():
    """Mock HuggingFace summarisation pipeline.

    SPEC: summarizer.summarize_article - pipeline has a ``.tokenizer`` attribute with an
    ``.encode()`` method that returns a list of token ids, and the pipeline call returns
    a list with a single dict containing key 'summary_text'.
    """
    pipeline = MagicMock()
    pipeline.return_value = [{"summary_text": "A short summary of the article."}]
    pipeline.tokenizer = MagicMock()
    # 100 fake token ids - passes default min_summary_length=50 guard.
    pipeline.tokenizer.encode.return_value = list(range(100))
    return pipeline


@pytest.fixture
def mock_topic_pipeline():
    """Mock HuggingFace zero-shot classification pipeline.

    SPEC: topic_predictor.predict_topic - pipeline returns a dict with 'labels' and
    'scores' aligned lists; the label with the highest score is the prediction.
    """
    pipeline = MagicMock()
    pipeline.return_value = {
        "sequence": "irrelevant",
        "labels": ["Sports", "Politics and conflicts", "Science and technology"],
        "scores": [0.9, 0.05, 0.05],
    }
    return pipeline


@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer.

    SPEC: similarity.calculate_similarity - model.encode() output is fed into
    util.cos_sim(), which expects 2D tensors of shape (1, dim).
    """
    import torch

    model = MagicMock()
    model.encode.return_value = torch.tensor([[0.5, 0.5, 0.5]])
    return model


# ---------------------------------------------------------------------------
# Auto-applied device mock - keeps CI off the GPU.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_device(monkeypatch):
    """Force CPU device in all tests.

    SPEC: CI note - "GPU and device detection must also be mocked in CI
    environments without CUDA." Patches are wrapped in try/except so that this
    autouse fixture does not error when implementation modules do not yet exist.
    """
    targets = (
        "src.ner.get_device",
        "src.summarizer.get_device",
        "src.topic_predictor.get_device",
    )
    for target in targets:
        try:
            monkeypatch.setattr(target, lambda: -1, raising=False)
        except (ImportError, ModuleNotFoundError, AttributeError):
            # Implementation module not yet present - fixture is a no-op.
            pass
