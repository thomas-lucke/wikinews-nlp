"""Text preprocessing utilities (cleaning, chunking, language detection)."""

import logging
import re
from typing import Optional

import spacy

logger = logging.getLogger(__name__)

_SPACY_MODELS: dict[str, spacy.Language] = {}


_WIKI_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_TEMPLATE_FALLBACK_RE = re.compile(r"\{\{[^}]*\}\}")


def _get_spacy_model(language: str, model_name: str) -> spacy.Language:
    """Return cached spaCy model for language, loading on first call."""
    if language in _SPACY_MODELS:
        return _SPACY_MODELS[language]
    try:
        nlp = spacy.load(model_name)
    except OSError:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found. "
            f"Run: python -m spacy download {model_name}"
        )
    nlp.max_length = 2_000_000
    _SPACY_MODELS[language] = nlp
    logger.info("Loaded spaCy model: %s", model_name)
    return nlp


def clean_text(text: str) -> str:
    """Remove MediaWiki markup, HTML, URLs and collapse whitespace."""
    try:
        import mwparserfromhell
        cleaned = mwparserfromhell.parse(text).strip_code()
    except ImportError:
        logger.warning(
            "mwparserfromhell not installed; falling back to regex template removal — "
            "nested templates may not be fully removed."
        )
        cleaned = _TEMPLATE_FALLBACK_RE.sub("", text)

    cleaned = _WIKI_LINK_RE.sub(r"\1", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def tokenize_and_tag(text: str, language: str, model_name: str) -> dict:
    """Split into sentences, tokenise, and tag POS using cached spaCy model."""
    nlp = _get_spacy_model(language, model_name)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    tokens = [tok.text for tok in doc if not tok.is_space]
    pos_tags = [(tok.text, tok.pos_) for tok in doc if not tok.is_space]
    return {
        "sentences": sentences,
        "tokens": tokens,
        "pos_tags": pos_tags,
    }


_LANG_TO_CONFIG_KEY = {
    "en": "spacy_english",
    "de": "spacy_german",
}


def preprocess_articles(articles: list[dict], config: dict) -> list[dict]:
    """Run clean_text + spaCy nlp.pipe per language group, mutating articles in place."""
    models_cfg = config.get("models", {})

    by_language: dict[str, list[dict]] = {}
    for article in articles:
        lang = article.get("language")
        if lang not in _LANG_TO_CONFIG_KEY:
            logger.warning(
                "Article %r has unrecognised language %r; skipping preprocessing.",
                article.get("id"),
                lang,
            )
            continue
        by_language.setdefault(lang, []).append(article)

    for lang, group in by_language.items():
        config_key = _LANG_TO_CONFIG_KEY[lang]
        model_name = models_cfg.get(config_key)
        if not model_name:
            logger.warning(
                "Missing spaCy model name for language %r in config (key=%r); "
                "skipping %d articles.",
                lang,
                config_key,
                len(group),
            )
            continue

        try:
            nlp = _get_spacy_model(lang, model_name)
        except RuntimeError:
            logger.exception(
                "Failed to load spaCy model %r for language %r; "
                "leaving %d articles unprocessed.",
                model_name,
                lang,
                len(group),
            )
            continue

        cleaned_pairs: list[tuple[dict, Optional[str]]] = []
        for article in group:
            try:
                cleaned = clean_text(article.get("text", ""))
            except Exception:
                logger.exception(
                    "clean_text failed for article %r; skipping.", article.get("id")
                )
                cleaned_pairs.append((article, None))
                continue
            cleaned_pairs.append((article, cleaned))

        articles_to_process = [(a, c) for a, c in cleaned_pairs if c is not None]
        cleaned_texts = [c for _, c in articles_to_process]

        try:
            docs = list(nlp.pipe(cleaned_texts))
        except Exception:
            logger.exception(
                "nlp.pipe failed for language %r batch of %d articles.",
                lang,
                len(cleaned_texts),
            )
            continue

        for (article, cleaned), doc in zip(articles_to_process, docs):
            try:
                article["cleaned_text"] = cleaned
                article["sentences"] = [sent.text for sent in doc.sents]
                article["tokens"] = [tok.text for tok in doc if not tok.is_space]
                article["pos_tags"] = [
                    (tok.text, tok.pos_) for tok in doc if not tok.is_space
                ]
            except Exception:
                logger.exception(
                    "Failed to extract spaCy fields for article %r.",
                    article.get("id"),
                )

    return articles
