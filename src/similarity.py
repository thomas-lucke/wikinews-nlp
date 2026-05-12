"""Sentence-embedding-based similarity scoring between articles."""

import logging
from typing import Optional

import pandas as pd
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


_SIMILARITY_COLUMNS = ["article_id", "title", "country", "topic", "similarity_score"]


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
    original_embedding = model.encode(original)
    summary_embedding = model.encode(summary)
    similarity = util.cos_sim(original_embedding, summary_embedding)
    return float(similarity[0][0])


def score_all_articles(
    articles: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    """Compute similarity_score in-place for qualifying articles."""
    for article in articles:
        cleaned = article.get("cleaned_text")
        summary = article.get("summary")
        if cleaned is None or cleaned == "" or summary is None:
            continue
        try:
            article["similarity_score"] = calculate_similarity(cleaned, summary, model)
        except Exception:
            logger.exception(
                "calculate_similarity failed for article %r; leaving similarity_score unset.",
                article.get("id"),
            )
    return articles


def build_similarity_dataframe(articles: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from articles that have similarity_score set."""
    rows: list[dict] = []
    for article in articles:
        if "similarity_score" not in article:
            continue
        rows.append({
            "article_id": article.get("id"),
            "title": article.get("title", ""),
            "country": article.get("country"),
            "topic": article.get("topic"),
            "similarity_score": article["similarity_score"],
        })
    if not rows:
        return pd.DataFrame(columns=_SIMILARITY_COLUMNS)
    return pd.DataFrame(rows, columns=_SIMILARITY_COLUMNS)


def plot_similarity_distribution(df: pd.DataFrame, threshold: float) -> None:
    """Histogram per (country, topic) pair; shared x-axis [-1.0, 1.0], independent y."""
    import matplotlib.pyplot as plt

    if df.empty:
        logger.warning("plot_similarity_distribution: empty DataFrame; no plot.")
        return

    pairs = sorted(
        {(c, t) for c, t in zip(df["country"], df["topic"])},
        key=lambda p: (str(p[0]), str(p[1])),
    )
    n_subplots = len(pairs)
    if n_subplots == 0:
        logger.warning("plot_similarity_distribution: no (country, topic) pairs.")
        return

    fig, axes = plt.subplots(
        1, n_subplots,
        figsize=(max(4 * n_subplots, 6), 4),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    bins = 20
    bin_range = (-1.0, 1.0)

    for ax, (country, topic) in zip(axes[0], pairs):
        subset = df[(df["country"] == country) & (df["topic"] == topic)]
        article_count = len(subset)
        ax.hist(subset["similarity_score"], bins=bins, range=bin_range)
        ax.axvline(threshold, linestyle="--", color="red")
        ax.set_title(f"{country} — {topic} (n={article_count})")
        ax.set_xlim(bin_range)
        ax.set_xlabel("Similarity score")
        ax.set_ylabel("Article count")

    fig.suptitle("Similarity score distribution by country and topic")
    fig.tight_layout()


def explain_similarity_extremes(df: pd.DataFrame, n: int = 3) -> dict:
    """Return the top-n and bottom-n articles by similarity_score, ties broken by str(article_id)."""
    columns = ["article_id", "title", "country", "topic", "similarity_score"]
    if df.empty:
        return {"highest": [], "lowest": []}

    working = df.copy()
    working["_id_str"] = working["article_id"].astype(str)

    highest = working.sort_values(
        by=["similarity_score", "_id_str"], ascending=[False, True]
    ).head(n)
    lowest = working.sort_values(
        by=["similarity_score", "_id_str"], ascending=[True, True]
    ).head(n)

    def to_records(frame: pd.DataFrame) -> list[dict]:
        return [
            {col: row[col] for col in columns}
            for _, row in frame.iterrows()
        ]

    return {"highest": to_records(highest), "lowest": to_records(lowest)}
