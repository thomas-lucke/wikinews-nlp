"""Sentence-embedding-based similarity scoring between articles."""

import logging
from typing import cast

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from torch import Tensor

logger = logging.getLogger(__name__)


_SIMILARITY_COLUMNS = ["article_id", "title", "topic", "similarity_score"]


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and return a SentenceTransformer model."""
    model = SentenceTransformer(model_name)
    logger.info("Loaded SentenceTransformer model: %s", model_name)
    return model


def calculate_similarity(
    original: str,
    summary: str,
    model: SentenceTransformer,
) -> float:
    """Encode both strings and return cosine similarity as a Python float in [-1, 1]."""
    original_embedding = cast(
        Tensor, model.encode(original, convert_to_tensor=True, show_progress_bar=False)
    )
    summary_embedding = cast(
        Tensor, model.encode(summary, convert_to_tensor=True, show_progress_bar=False)
    )
    similarity = util.cos_sim(original_embedding, summary_embedding)
    return float(similarity[0][0])


def score_all_articles(
    articles: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    """Compute similarity_score for qualifying articles via two batched encode calls.

    Batching collapses 2N progress bars into one and lets the model parallelise
    embeddings internally (default batch_size=32). On encode failure, all
    qualifying articles are left without a similarity_score and an error is logged.
    """
    qualifying: list[dict] = []
    originals: list[str] = []
    summaries: list[str] = []
    for article in articles:
        cleaned = article.get("cleaned_text")
        summary = article.get("summary")
        if cleaned is None or cleaned == "" or summary is None:
            continue
        qualifying.append(article)
        originals.append(cleaned)
        summaries.append(summary)

    if not qualifying:
        return articles

    try:
        originals_emb = cast(
            Tensor,
            model.encode(originals, convert_to_tensor=True, show_progress_bar=True),
        )
        summaries_emb = cast(
            Tensor,
            model.encode(summaries, convert_to_tensor=True, show_progress_bar=False),
        )
        sims = util.cos_sim(originals_emb, summaries_emb).diagonal().tolist()
    except Exception:
        logger.exception(
            "score_all_articles: batch encode failed for %d articles; "
            "leaving similarity_score unset.",
            len(qualifying),
        )
        return articles

    for article, sim in zip(qualifying, sims):
        article["similarity_score"] = float(sim)
    return articles


def build_similarity_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from articles that have similarity_score set."""
    rows: list[dict] = []
    for article in articles:
        if "similarity_score" not in article:
            continue
        rows.append(
            {
                "article_id": article.get("id"),
                "title": article.get("title", ""),
                "topic": article.get("topic"),
                "similarity_score": article["similarity_score"],
            }
        )
    if not rows:
        return pd.DataFrame(columns=_SIMILARITY_COLUMNS)
    return pd.DataFrame(rows, columns=_SIMILARITY_COLUMNS)


def plot_similarity_distribution(df: pd.DataFrame, threshold: float) -> None:
    """Histogram per topic; shared x-axis [-1.0, 1.0], independent y.

    Topic is the primary sampling axis (per ADR 0005, country is best-effort
    metadata, not a sampling axis). Splitting by (country, topic) fragmented
    60 articles into ~80 mostly-empty subplots; per-topic gives a readable
    picture in 3 subplots.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        logger.warning("plot_similarity_distribution: empty DataFrame; no plot.")
        return

    topics = sorted({t for t in df["topic"] if t is not None})
    n_subplots = len(topics)
    if n_subplots == 0:
        logger.warning("plot_similarity_distribution: no topics.")
        return

    fig, axes = plt.subplots(
        n_subplots,
        1,
        figsize=(7, 4 * n_subplots),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    bins = 20
    bin_range = (-1.0, 1.0)

    for ax, topic in zip(axes[:, 0], topics):
        subset = df[df["topic"] == topic]
        article_count = len(subset)
        ax.hist(subset["similarity_score"], bins=bins, range=bin_range)
        ax.axvline(threshold, linestyle="--", color="red")
        ax.set_title(f"{topic} (n={article_count})")
        ax.set_xlim(bin_range)
        ax.set_xlabel("Similarity score")
        ax.set_ylabel("Article count")

    fig.suptitle("Similarity score distribution by topic")
    fig.tight_layout()


def plot_similarity_boxplot(df: pd.DataFrame, threshold: float) -> None:
    """Per-topic boxplot of similarity scores — single image side-by-side comparison."""
    import matplotlib.pyplot as plt

    if df.empty:
        logger.warning("plot_similarity_boxplot: empty DataFrame; no plot.")
        return

    topics = sorted({t for t in df["topic"] if t is not None})
    if not topics:
        logger.warning("plot_similarity_boxplot: no topics.")
        return

    data = [df.loc[df["topic"] == t, "similarity_score"].tolist() for t in topics]
    tick_labels = [f"{t}\n(n={len(d)})" for t, d in zip(topics, data)]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(topics)), 5))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels)
    ax.axhline(threshold, linestyle="--", color="red", label=f"threshold={threshold}")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Cosine similarity (original vs. summary)")
    ax.set_title("Similarity score distribution by topic")
    ax.legend(loc="lower right")
    fig.tight_layout()


def summarize_similarity_stats(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per-topic and overall similarity statistics: count, mean, median, std, % above threshold."""
    columns = ["count", "mean", "median", "std", "pct_above_threshold"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    def stats_row(scores: pd.Series) -> dict:
        return {
            "count": int(len(scores)),
            "mean": round(float(scores.mean()), 3),
            "median": round(float(scores.median()), 3),
            "std": round(float(scores.std()), 3),
            "pct_above_threshold": round(float((scores >= threshold).mean()) * 100, 1),
        }

    rows: dict[str, dict] = {}
    for topic in sorted({t for t in df["topic"] if t is not None}):
        rows[topic] = stats_row(df.loc[df["topic"] == topic, "similarity_score"])
    rows["(all)"] = stats_row(df["similarity_score"])

    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def explain_similarity_extremes(df: pd.DataFrame, n: int = 3) -> dict:
    """
    Return the top-n and bottom-n articles by similarity_score, ties broken by str(article_id).
    """
    columns = ["article_id", "title", "topic", "similarity_score"]
    if df.empty:
        return {"highest": [], "lowest": []}

    working = df.copy()
    working["_id_str"] = working["article_id"].astype(str)

    highest = working.sort_values(by=["similarity_score", "_id_str"], ascending=[False, True]).head(
        n
    )
    lowest = working.sort_values(by=["similarity_score", "_id_str"], ascending=[True, True]).head(n)

    def to_records(frame: pd.DataFrame) -> list[dict]:
        return [{col: row[col] for col in columns} for _, row in frame.iterrows()]

    return {"highest": to_records(highest), "lowest": to_records(lowest)}
