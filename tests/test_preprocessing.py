"""Tests for src/preprocessing.py.

SPEC: preprocessing contains clean_text (pure regex), _get_spacy_model
(cached spaCy loader), tokenize_and_tag, and preprocess_articles.
spaCy is mocked - no real model download is required.

The mwparserfromhell fallback branch in clean_text is intentionally NOT
unit-tested per SPEC "Functions intentionally not covered by unit tests"
- which branch executes depends on the environment, not the input.
"""

from unittest.mock import patch

import pytest

preprocessing = pytest.importorskip("src.preprocessing")


# ---------------------------------------------------------------------------
# Fake spaCy Doc / Token / Span - replicates the spaCy attribute surface
# used by tokenize_and_tag (token.pos_, token.text, token.is_space, doc.sents).
# ---------------------------------------------------------------------------


class FakeToken:
    def __init__(self, text: str, pos: str = "NOUN", is_space: bool = False):
        self.text = text
        self.pos_ = pos
        self.is_space = is_space


class FakeSpan:
    def __init__(self, text: str):
        self.text = text


class FakeDoc:
    def __init__(self, text: str):
        self._text = text
        # Split sentences on '. ' for a deterministic fake - spec only requires
        # that tokenize_and_tag returns spaCy's senter output, not any specific
        # sentencing algorithm.
        self.sents = [FakeSpan(s) for s in text.split(". ") if s]
        self._tokens = [FakeToken(tok, pos="NOUN") for tok in text.split() if tok]

    def __iter__(self):
        return iter(self._tokens)


class FakeNLP:
    def __init__(self):
        self.max_length = 1_000_000
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return FakeDoc(text)

    def pipe(self, texts):
        for t in texts:
            yield FakeDoc(t)


# ---------------------------------------------------------------------------
# clean_text - pure regex, no spaCy
# ---------------------------------------------------------------------------


def test_clean_text_resolves_mediawiki_link_with_display():
    # SPEC: clean_text step 2 - "[[target|display]] → 'display'".
    # SPEC test list: "clean_text on text with [[Berlin|Berlin, Germany]]:
    # returns 'Berlin, Germany'"
    text = "City: [[Berlin|Berlin, Germany]]"
    out = preprocessing.clean_text(text)
    assert "Berlin, Germany" in out
    assert "[[" not in out and "]]" not in out


def test_clean_text_resolves_mediawiki_link_no_pipe():
    # SPEC: clean_text step 2 - "[[target]] → 'target'".
    out = preprocessing.clean_text("Visit [[Paris]] today.")
    assert "Paris" in out
    assert "[[" not in out


def test_clean_text_strips_mediawiki_template():
    # SPEC: clean_text step 1 - remove "{{...}}" templates.
    # SPEC test list: "clean_text on text with {{cite web|url=...}}: template removed"
    out = preprocessing.clean_text("Result {{cite web|url=http://x.com}} done.")
    assert "{{" not in out and "}}" not in out
    assert "cite" not in out  # template body fully removed


def test_clean_text_strips_html_tags():
    # SPEC: clean_text step 3 - HTML tags r'<[^>]+>' → ''.
    out = preprocessing.clean_text("Hello <b>bold</b> <i>italic</i> world")
    assert "<" not in out and ">" not in out
    assert "bold" in out and "italic" in out


def test_clean_text_strips_urls():
    # SPEC: clean_text step 4 - URLs r'https?://\S+' → ''.
    out = preprocessing.clean_text("See https://example.com/page for details")
    assert "https://" not in out
    assert "example.com" not in out


def test_clean_text_collapses_whitespace():
    # SPEC: clean_text step 5 - "Multiple whitespace/newlines → single space."
    out = preprocessing.clean_text("a  \n\t  b\n\nc")
    assert out == "a b c"


def test_clean_text_strips_leading_trailing_whitespace():
    # SPEC: clean_text step 6 - "Strip leading and trailing whitespace."
    out = preprocessing.clean_text("   hello world   ")
    assert out == "hello world"


def test_clean_text_keeps_punctuation():
    # SPEC: clean_text - "Punctuation is NOT removed (required for sentence splitting)."
    out = preprocessing.clean_text("Hello, world! Is it ok?")
    assert "," in out
    assert "!" in out
    assert "?" in out


# ---------------------------------------------------------------------------
# tokenize_and_tag - returns shape-correct dict
# ---------------------------------------------------------------------------


def test_tokenize_and_tag_returns_expected_keys():
    # SPEC: tokenize_and_tag - "Returns: dict with 'sentences', 'tokens', 'pos_tags'."
    fake = FakeNLP()
    with patch.object(preprocessing, "_get_spacy_model", return_value=fake):
        out = preprocessing.tokenize_and_tag(
            "Berlin is great. The team won.", "en", "en_core_web_sm"
        )
    assert set(out.keys()) == {"sentences", "tokens", "pos_tags"}
    assert isinstance(out["sentences"], list)
    assert isinstance(out["tokens"], list)
    assert isinstance(out["pos_tags"], list)


def test_tokenize_and_tag_excludes_whitespace_tokens():
    # SPEC: tokenize_and_tag - "Tokens list excludes whitespace tokens (token.is_space)."
    class DocWithWhitespace:
        def __init__(self):
            self.sents = [FakeSpan("Hello world.")]
            self._tokens = [
                FakeToken("Hello", "PROPN", is_space=False),
                FakeToken(" ", "SPACE", is_space=True),
                FakeToken("world", "NOUN", is_space=False),
            ]

        def __iter__(self):
            return iter(self._tokens)

    class FakeNLPwithWS:
        max_length = 1_000_000

        def __call__(self, text):
            return DocWithWhitespace()

        def pipe(self, texts):
            for _ in texts:
                yield DocWithWhitespace()

    with patch.object(preprocessing, "_get_spacy_model", return_value=FakeNLPwithWS()):
        out = preprocessing.tokenize_and_tag("Hello world.", "en", "en_core_web_sm")
    assert " " not in out["tokens"]
    assert "Hello" in out["tokens"]
    assert "world" in out["tokens"]


def test_tokenize_and_tag_pos_tags_are_pairs():
    # SPEC: Article schema - '"pos_tags": list[tuple[str, str]],
    # # [(token, universal_pos_tag), ...]'
    fake = FakeNLP()
    with patch.object(preprocessing, "_get_spacy_model", return_value=fake):
        out = preprocessing.tokenize_and_tag("Hello world", "en", "en_core_web_sm")
    assert len(out["pos_tags"]) > 0
    for pair in out["pos_tags"]:
        assert isinstance(pair, tuple) and len(pair) == 2
        assert isinstance(pair[0], str)
        assert isinstance(pair[1], str)


# ---------------------------------------------------------------------------
# _get_spacy_model - caching behaviour
# ---------------------------------------------------------------------------


def test_get_spacy_model_caches_by_language():
    # SPEC: _get_spacy_model - "Return cached spaCy model for language. Load on
    # first call ..."
    preprocessing._SPACY_MODELS.clear()
    call_counter = {"n": 0}

    def fake_load(name):
        call_counter["n"] += 1
        return FakeNLP()

    with patch.object(preprocessing.spacy, "load", side_effect=fake_load):
        m1 = preprocessing._get_spacy_model("en", "en_core_web_sm")
        m2 = preprocessing._get_spacy_model("en", "en_core_web_sm")
    assert m1 is m2
    assert call_counter["n"] == 1  # loaded only once

    preprocessing._SPACY_MODELS.clear()


def test_get_spacy_model_raises_runtime_error_if_model_missing():
    # SPEC: _get_spacy_model - "Raise RuntimeError with installation instructions
    # if model not found." Real spaCy raises OSError when the model isn't installed.
    preprocessing._SPACY_MODELS.clear()
    with patch.object(preprocessing.spacy, "load", side_effect=OSError("no model")):
        with pytest.raises(RuntimeError):
            preprocessing._get_spacy_model("en", "en_core_web_sm")
    preprocessing._SPACY_MODELS.clear()


# ---------------------------------------------------------------------------
# preprocess_articles - adds fields, skips unknown languages, robust to errors
# ---------------------------------------------------------------------------


def test_preprocess_articles_adds_required_fields(sample_en_article, sample_config):
    # SPEC: Article schema - "Added by preprocessing.preprocess_articles():
    # cleaned_text, sentences, tokens, pos_tags".
    preprocessing._SPACY_MODELS.clear()
    articles = [sample_en_article]
    with patch.object(preprocessing, "_get_spacy_model", return_value=FakeNLP()):
        out = preprocessing.preprocess_articles(articles, sample_config)
    a = out[0]
    for field in ("cleaned_text", "sentences", "tokens", "pos_tags"):
        assert field in a, f"preprocess_articles must add field {field}"


def test_preprocess_articles_skips_unrecognised_language(sample_config):
    # SPEC: preprocess_articles - "Articles with an unrecognised language field
    # are skipped (logged as warning, fields not added, article remains in list)."
    preprocessing._SPACY_MODELS.clear()
    articles = [
        {
            "id": "x1",
            "title": "t",
            "text": "hello world",
            "language": "fr",  # not in spacy_english / spacy_german
            "country": "germany",
            "topic": "sports",
            "date": None,
        }
    ]
    with patch.object(preprocessing, "_get_spacy_model", return_value=FakeNLP()):
        out = preprocessing.preprocess_articles(articles, sample_config)
    assert len(out) == 1
    assert "cleaned_text" not in out[0]
    assert "sentences" not in out[0]


def test_preprocess_articles_per_article_error_does_not_crash(
    sample_en_article,
    sample_config,
):
    # SPEC: preprocess_articles - "Per-article errors are caught, logged with
    # article ID, and the article is left in the list without preprocessing fields."
    preprocessing._SPACY_MODELS.clear()

    class ExplodingNLP:
        max_length = 1_000_000

        def __call__(self, text):
            raise RuntimeError("boom")

        def pipe(self, texts):
            raise RuntimeError("boom")

    articles = [sample_en_article]
    with patch.object(preprocessing, "_get_spacy_model", return_value=ExplodingNLP()):
        try:
            out = preprocessing.preprocess_articles(articles, sample_config)
        except Exception as exc:
            pytest.fail(f"preprocess_articles must catch per-article errors, got: {exc!r}")
    # Article remains in the list with no preprocessing fields added.
    assert len(out) == 1
    assert "cleaned_text" not in out[0] or out[0].get("cleaned_text") is None


def test_preprocess_articles_reads_model_names_from_config(
    sample_en_article,
    sample_config,
):
    # SPEC: preprocess_articles - "spaCy model names are read from
    # config['models']['spacy_english'] and config['models']['spacy_german']."
    preprocessing._SPACY_MODELS.clear()
    captured = {}

    def fake_get(language, model_name):
        captured[language] = model_name
        return FakeNLP()

    with patch.object(preprocessing, "_get_spacy_model", side_effect=fake_get):
        preprocessing.preprocess_articles([sample_en_article], sample_config)
    assert captured.get("en") == sample_config["models"]["spacy_english"]
