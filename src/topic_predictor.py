"""Zero-shot topic prediction over the configured topic set."""

import logging
import math
import random
from typing import Optional

from transformers import pipeline as hf_pipeline

from src.utils import get_device

logger = logging.getLogger(__name__)


def load_topic_pipeline(model_name: str) -> object:
    """Load and return a HuggingFace zero-shot-classification pipeline."""
    pipeline_obj = hf_pipeline(
        "zero-shot-classification", model=model_name, device=get_device()
    )
    logger.info("Loaded zero-shot topic pipeline: %s", model_name)
    return pipeline_obj


def predict_topic(
    text: str,
    candidate_labels: list[str],
    topic_pipeline: object,
    hypothesis_template: str,
) -> Optional[str]:
    """Run zero-shot classification and return the highest-scoring label, or None."""
    if not text:
        logger.warning("predict_topic called with empty text; returning None.")
        return None

    try:
        result = topic_pipeline(
            text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
        )
    except Exception:
        logger.warning("predict_topic pipeline raised; returning None.", exc_info=True)
        return None

    try:
        return result["labels"][0]
    except (KeyError, IndexError, TypeError):
        logger.warning(
            "predict_topic could not extract top label from result; returning None."
        )
        return None


def predict_all_topics(
    articles: list[dict],
    candidate_labels: list[str],
    topic_pipeline: object,
    hypothesis_template: str,
    sample_size: int,
    random_seed: int = 42,
) -> list[dict]:
    """Balanced-sample articles and run predict_topic on each sampled article."""
    rng = random.Random(random_seed)

    for article in articles:
        if article.get("language") != "en":
            logger.warning(
                "predict_all_topics: article id=%r has language=%r (expected 'en'); "
                "results from bart-large-mnli on non-English text are unreliable.",
                article.get("id"),
                article.get("language"),
            )

    eligible: list[dict] = []
    excluded = 0
    for article in articles:
        if article.get("cleaned_text", "") == "":
            excluded += 1
            continue
        eligible.append(article)
    if excluded > 0:
        logger.info(
            "predict_all_topics: excluded %d articles with missing/empty cleaned_text.",
            excluded,
        )

    if not eligible or sample_size <= 0:
        return []

    groups: dict[tuple, list[dict]] = {}
    for article in eligible:
        key = (article.get("country"), article.get("topic"))
        groups.setdefault(key, []).append(article)

    for key in groups:
        groups[key].sort(key=lambda a: str(a["id"]))

    n_groups = len(groups)
    quota = sample_size // n_groups if n_groups > 0 else 0

    selected_ids: set = set()
    selected: list[dict] = []
    for key, group in groups.items():
        k = min(quota, len(group))
        if k <= 0:
            continue
        picked = rng.sample(group, k)
        for article in picked:
            selected_ids.add(article["id"])
            selected.append(article)

    remaining_slots = sample_size - len(selected)
    if remaining_slots > 0:
        groups_with_remainder: list[tuple] = []
        for key, group in groups.items():
            available = [a for a in group if a["id"] not in selected_ids]
            if available:
                groups_with_remainder.append(key)

        if groups_with_remainder:
            per_group_extra = math.ceil(remaining_slots / len(groups_with_remainder))
            for key in groups_with_remainder:
                if remaining_slots <= 0:
                    break
                pool = [a for a in groups[key] if a["id"] not in selected_ids]
                if not pool:
                    continue
                want = min(per_group_extra, remaining_slots)
                if len(pool) < want:
                    picked = rng.sample(pool, len(pool))
                else:
                    picked = rng.sample(pool, want)
                for article in picked:
                    selected_ids.add(article["id"])
                    selected.append(article)
                    remaining_slots -= 1
                    if remaining_slots <= 0:
                        break

    sampled_copies: list[dict] = []
    for original in selected:
        copy = {**original, "predicted_topic": None}
        try:
            copy["predicted_topic"] = predict_topic(
                original.get("cleaned_text", ""),
                candidate_labels,
                topic_pipeline,
                hypothesis_template,
            )
        except Exception:
            logger.exception(
                "predict_all_topics: predict_topic failed for article id=%r; "
                "predicted_topic set to None.",
                original.get("id"),
            )
            copy["predicted_topic"] = None
        sampled_copies.append(copy)

    return sampled_copies


def evaluate_topic_predictions(sampled_articles: list[dict]) -> dict:
    """Compare predicted_topic to topic for each sampled article (case-insensitive)."""
    results: list[dict] = []
    correct = 0
    evaluated = 0

    for article in sampled_articles:
        predicted = article.get("predicted_topic")
        if predicted is None:
            continue

        gold = article.get("topic", "")
        match = gold.lower().strip() == predicted.lower().strip()
        if match:
            correct += 1
        evaluated += 1

        title = article.get("title", "")
        if title == "":
            title = f"[id: {article['id']}]"

        results.append(
            {
                "title": title,
                "match": match,
                "country": article.get("country"),
                "topic": gold,
                "predicted_topic": predicted,
            }
        )

    accuracy = (correct / evaluated) if evaluated > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "evaluated": evaluated,
        "total_sampled": len(sampled_articles),
        "results": results,
    }
