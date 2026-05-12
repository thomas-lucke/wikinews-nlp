"""Article summarization using a transformer-based model."""

import logging
import re
from typing import Optional

import pandas as pd
from transformers import pipeline as hf_pipeline

from src.utils import get_device

logger = logging.getLogger(__name__)


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]")
_REPEATED_WS_RE = re.compile(r"\s{2,}")
_TERMINAL_PUNCT = (".", "!", "?")

_SUMMARY_QUALITY_COLUMNS = [
    "article_id", "title", "country", "topic",
    "summary_char_count", "summary_sentence_count", "avg_sentence_chars",
    "missing_terminal_punctuation", "repeated_whitespace", "very_long_sentence",
    "issue_count",
]


def validate_summarization_config(config: dict) -> None:
    """Raise ValueError if min_summary_length >= max_summary_length."""
    summ_cfg = config["summarization"]
    min_len = summ_cfg["min_summary_length"]
    max_len = summ_cfg["max_summary_length"]
    if min_len >= max_len:
        raise ValueError(
            f"summarization.min_summary_length ({min_len}) must be < "
            f"max_summary_length ({max_len})"
        )


def load_summarization_pipeline(model_name: str) -> object:
    """Load a HuggingFace summarization pipeline."""
    pipe = hf_pipeline("summarization", model=model_name, device=get_device())
    logger.info("Loaded summarization pipeline: %s", model_name)
    return pipe


def summarize_article(
    text: str,
    summ_pipeline: object,
    min_length: int,
    max_length: int,
) -> Optional[str]:
    """Summarise a single article. Return None on empty, short, or pipeline failure."""
    if text == "":
        logger.warning("summarize_article: empty text input, returning None.")
        return None

    token_count = len(summ_pipeline.tokenizer.encode(text, add_special_tokens=False))
    if token_count < min_length:
        logger.warning(
            "summarize_article: input token_count=%d < min_length=%d, returning None.",
            token_count, min_length,
        )
        return None

    try:
        result = summ_pipeline(
            text, truncation=True, min_length=min_length, max_length=max_length
        )
    except Exception:
        logger.exception("summarize_article: pipeline raised; returning None.")
        return None

    return result[0]["summary_text"]


def summarize_articles(
    articles: list[dict],
    summ_pipeline: object,
    config: dict,
) -> list[dict]:
    """Summarise qualifying articles in place; non-qualifying articles get no 'summary' key."""
    summ_languages = set(config["languages"]["summarization"])
    min_length = config["summarization"]["min_summary_length"]
    max_length = config["summarization"]["max_summary_length"]

    qualifying = [a for a in articles if a.get("language") in summ_languages]
    n_total = len(qualifying)
    n_done = 0

    for article in qualifying:
        article["summary"] = summarize_article(
            article.get("cleaned_text", ""),
            summ_pipeline,
            min_length,
            max_length,
        )
        n_done += 1
        if n_done % 5 == 0:
            logger.info("Summarised %d/%d articles", n_done, n_total)

    return articles


def _summary_quality_row(article: dict) -> dict:
    summary = article["summary"]
    char_count = len(summary)
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(summary)]
    sentences = [s for s in sentences if s]
    sentence_count = len(sentences)
    if sentence_count > 0:
        avg_sentence_chars = sum(len(s) for s in sentences) / sentence_count
    else:
        avg_sentence_chars = 0.0

    missing_terminal = not summary.endswith(_TERMINAL_PUNCT)
    repeated_ws = bool(_REPEATED_WS_RE.search(summary))
    very_long = any(len(s) > 250 for s in sentences)

    issue_count = int(missing_terminal) + int(repeated_ws) + int(very_long)

    title = article.get("title", "")
    if not title:
        title = f"[id: {article.get('id')}]"

    return {
        "article_id": article.get("id"),
        "title": title,
        "country": article.get("country"),
        "topic": article.get("topic"),
        "summary_char_count": char_count,
        "summary_sentence_count": sentence_count,
        "avg_sentence_chars": avg_sentence_chars,
        "missing_terminal_punctuation": missing_terminal,
        "repeated_whitespace": repeated_ws,
        "very_long_sentence": very_long,
        "issue_count": issue_count,
    }


def build_summary_quality_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Quality flags per generated summary. Empty DF with correct columns if no summaries."""
    rows = [
        _summary_quality_row(a)
        for a in articles
        if a.get("summary") is not None
    ]
    if not rows:
        return pd.DataFrame(columns=_SUMMARY_QUALITY_COLUMNS)
    return pd.DataFrame(rows, columns=_SUMMARY_QUALITY_COLUMNS)
