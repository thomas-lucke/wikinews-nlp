"""Named entity recognition for English and German articles."""

import logging
from typing import cast

import pandas as pd
from transformers import Pipeline

from src.utils import get_device, make_hf_pipeline

logger = logging.getLogger(__name__)


def validate_ner_config(config: dict) -> None:
    """Raise ValueError if NER chunk config values are inconsistent."""
    ner_cfg = config["ner"]
    chunk_size = ner_cfg["chunk_size"]
    chunk_overlap = ner_cfg["chunk_overlap"]
    if chunk_size <= 0:
        raise ValueError(f"ner.chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"ner.chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"ner.chunk_overlap ({chunk_overlap}) must be < ner.chunk_size ({chunk_size})"
        )


def load_ner_pipeline(model_name: str, language: str) -> Pipeline:
    """Load a HuggingFace NER pipeline with word-level aggregation.

    Uses aggregation_strategy="average" (a word-level strategy) rather than
    "simple". "simple" merges consecutive same-label tokens but splits a word
    when the model tags its BERT subword pieces inconsistently, producing
    fragmentary "##"-prefixed entities. "average" aggregates per whole word, so
    subword fragments cannot occur. See FIX-10.
    """
    try:
        pipe = make_hf_pipeline(
            "ner",
            model=model_name,
            device=get_device(),
            aggregation_strategy="average",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load NER pipeline {model_name!r} for language {language!r}: {exc}"
        ) from exc
    logger.info("Loaded NER pipeline %s for language %s", model_name, language)
    return pipe


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    """Split text into overlapping chunks at whitespace boundaries.

    Returns list of (chunk_str, start_offset) tuples.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    if not text:
        return []

    chunks: list[tuple[str, int]] = []
    start = 0
    text_len = len(text)

    while True:
        if text_len - start <= chunk_size:
            chunks.append((text[start:], start))
            break

        window_end = start + chunk_size
        end = -1
        for i in range(window_end, start, -1):
            if text[i - 1].isspace():
                end = i - 1
                break
        hard_break = False
        if end == -1 or end <= start:
            end = window_end
            hard_break = True
            logger.warning(
                "Hard-break chunk at offset %d: no whitespace within %d chars",
                start,
                chunk_size,
            )

        chunks.append((text[start:end], start))

        next_start = end - overlap
        lower_bound = end - chunk_size
        if next_start > start and not hard_break:
            adjusted = -1
            search_floor = max(lower_bound, 0)
            for i in range(next_start, search_floor, -1):
                if text[i - 1].isspace():
                    adjusted = i
                    break
            if adjusted != -1:
                next_start = adjusted

        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks


def _resolve_overlapping_entities(
    entities: list[dict],
    cleaned_text: str,
) -> list[dict]:
    """Deduplicate exact + partial overlaps, validate offsets, sort by start."""
    if not entities:
        return []

    by_key: dict[tuple[int, int, str], dict] = {}
    for ent in entities:
        key = (ent["start"], ent["end"], ent["label"])
        existing = by_key.get(key)
        if existing is None or ent["score"] > existing["score"]:
            by_key[key] = ent

    candidates = sorted(
        by_key.values(), key=lambda e: (e["start"], -(e["end"] - e["start"]), -e["score"])
    )

    kept: list[dict] = []
    for ent in candidates:
        replaced = False
        drop_self = False
        new_kept: list[dict] = []
        for prev in kept:
            if ent["end"] <= prev["start"] or ent["start"] >= prev["end"]:
                new_kept.append(prev)
                continue
            ent_span = ent["end"] - ent["start"]
            prev_span = prev["end"] - prev["start"]
            if ent_span > prev_span:
                replaced = True
                continue  # drop prev
            if ent_span < prev_span:
                new_kept.append(prev)
                drop_self = True
                continue
            if ent["score"] > prev["score"]:
                replaced = True
                continue
            new_kept.append(prev)
            drop_self = True
        if drop_self and not replaced:
            kept = new_kept
            continue
        kept = new_kept
        kept.append(ent)

    validated: list[dict] = []
    for ent in kept:
        start, end = ent["start"], ent["end"]
        if 0 <= start < end <= len(cleaned_text):
            # The fast tokenizer's character offsets are reliable; the pipeline's
            # reconstructed `.word` is lossy (spurious spaces like "U. S." for
            # "U.S.", leftover "##" subword markers). Take the canonical source
            # slice as the entity text rather than trusting `.word`. See FIX-10.
            ent["text"] = cleaned_text[start:end]
            validated.append(ent)
        else:
            logger.warning(
                "Discarding entity with out-of-bounds offsets: text=%r start=%d end=%d text_len=%d",
                ent.get("text"),
                start,
                end,
                len(cleaned_text),
            )

    validated.sort(key=lambda e: e["start"])
    return validated


_ENTITY_DF_COLUMNS = [
    "article_id",
    "event_id",
    "title",
    "date",
    "language",
    "topic",
    "entity_text",
    "entity_label",
    "score",
]


def _convert_raw_entity(raw: dict) -> dict:
    return {
        "text": raw.get("word", raw.get("text", "")),
        "label": raw.get("entity_group", raw.get("label", "")),
        "start": int(raw.get("start", 0)),
        "end": int(raw.get("end", 0)),
        "score": float(raw.get("score", 0.0)),
    }


def run_ner(
    articles: list[dict],
    ner_pipeline: Pipeline,
    language: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """Run NER on all articles where article['language'] == language."""
    for article in articles:
        if article.get("language") != language:
            article["entities"] = None
            continue

        cleaned = article.get("cleaned_text")
        if cleaned is None:
            logger.warning(
                "Article %r missing cleaned_text; setting entities=[].",
                article.get("id"),
            )
            article["entities"] = []
            continue
        if not isinstance(cleaned, str) or cleaned == "":
            article["entities"] = []
            continue

        try:
            if len(cleaned) <= chunk_size:
                raw_entities = cast(list[dict], ner_pipeline(cleaned))
                # Route through _resolve_overlapping_entities even for the
                # single-chunk case so entity text is canonicalised from the
                # source offsets identically to the chunked path. See FIX-10.
                entities = _resolve_overlapping_entities(
                    [_convert_raw_entity(e) for e in raw_entities], cleaned
                )
            else:
                chunks = _chunk_text(cleaned, chunk_size, chunk_overlap)
                collected: list[dict] = []
                for chunk_text, chunk_offset in chunks:
                    chunk_raw = cast(list[dict], ner_pipeline(chunk_text))
                    for raw in chunk_raw:
                        ent = _convert_raw_entity(raw)
                        ent["start"] += chunk_offset
                        ent["end"] += chunk_offset
                        collected.append(ent)
                entities = _resolve_overlapping_entities(collected, cleaned)
            article["entities"] = entities
        except Exception:
            logger.exception(
                "NER failed for article %r; setting entities=[].",
                article.get("id"),
            )
            article["entities"] = []

    return articles


def _display_title(article: dict) -> str:
    title = article.get("title", "")
    if title:
        return title
    return f"[id: {article.get('id')}]"


def build_entity_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Flatten all entities into one row per occurrence."""
    rows: list[dict] = []
    for article in articles:
        if "entities" not in article:
            continue
        entities = article["entities"]
        if entities is None:
            continue
        title = _display_title(article)
        for ent in entities:
            rows.append(
                {
                    "article_id": article.get("id"),
                    "event_id": article.get("event_id"),
                    "title": title,
                    "date": article.get("date"),
                    "language": article.get("language"),
                    "topic": article.get("topic"),
                    "entity_text": ent.get("text"),
                    "entity_label": ent.get("label"),
                    "score": ent.get("score"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=_ENTITY_DF_COLUMNS)
    return pd.DataFrame(rows, columns=_ENTITY_DF_COLUMNS)


def _filter_df(df: pd.DataFrame, language: str) -> pd.DataFrame:
    return df[df["language"] == language]


def plot_top_entities(df: pd.DataFrame, top_n: int, language: str) -> None:
    """Plot horizontal bar chart of top_n entity_text values by distinct article_id count."""
    import matplotlib.pyplot as plt

    subset = _filter_df(df, language)
    if subset.empty:
        logger.warning("plot_top_entities: no rows for language=%r", language)
        return

    counts = (
        subset.groupby("entity_text")["article_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
    )
    if counts.empty:
        logger.warning("plot_top_entities: no entities after grouping.")
        return

    title = f"Top {top_n} entities - {language.upper()}"

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(counts))))
    counts.iloc[::-1].plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Article count")
    ax.set_ylabel("Entity")
    fig.tight_layout()


def plot_entity_dynamics(
    df: pd.DataFrame,
    entity_names: list[str],
    language: str,
) -> None:
    """Plot one line per named entity showing distinct-article count per year-month."""
    import matplotlib.pyplot as plt

    subset = _filter_df(df, language)
    subset = subset[subset["entity_text"].isin(entity_names)]
    if subset.empty:
        logger.warning(
            "plot_entity_dynamics: no rows matching entity names for language=%r",
            language,
        )
        return

    parsed = pd.to_datetime(subset["date"], errors="coerce")
    subset = subset.assign(_parsed_date=parsed).dropna(subset=["_parsed_date"])
    if subset.empty:
        logger.warning("plot_entity_dynamics: no parseable dates after coercion.")
        return

    subset = subset.assign(year_month=subset["_parsed_date"].dt.to_period("M"))
    grouped = (
        subset.groupby(["entity_text", "year_month"])["article_id"]
        .nunique()
        .reset_index(name="article_count")
    )

    title = f"Entity dynamics - {language.upper()}"

    fig, ax = plt.subplots(figsize=(10, 5))
    for ent_name in entity_names:
        line = grouped[grouped["entity_text"] == ent_name].sort_values("year_month")
        if line.empty:
            continue
        if len(line) < 3:
            logger.warning(
                "plot_entity_dynamics: entity %r has fewer than 3 data points (%d).",
                ent_name,
                len(line),
            )
        x_labels = [str(p) for p in line["year_month"]]
        ax.plot(x_labels, line["article_count"].tolist(), marker="o", label=ent_name)

    ax.set_title(title)
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Article count")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()


def investigate_ner_errors(
    articles: list[dict],
    language: str,
    error_score_threshold: float,
) -> pd.DataFrame:
    """Return low-confidence entities, sorted by score ascending."""
    columns = [
        "article_id",
        "event_id",
        "title",
        "entity_text",
        "entity_label",
        "score",
    ]
    rows: list[dict] = []
    for article in articles:
        if article.get("language") != language:
            continue
        entities = article.get("entities")
        if entities is None:
            continue
        title = _display_title(article)
        for ent in entities:
            score = ent.get("score", 0.0)
            if score < error_score_threshold:
                rows.append(
                    {
                        "article_id": article.get("id"),
                        "event_id": article.get("event_id"),
                        "title": title,
                        "entity_text": ent.get("text"),
                        "entity_label": ent.get("label"),
                        "score": score,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("score", ascending=True)
        .reset_index(drop=True)
    )
