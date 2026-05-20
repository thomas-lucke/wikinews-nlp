"""Zero-shot topic prediction over the configured topic set."""

import logging
import math
import random
from typing import Optional, cast

from transformers import Pipeline

from src.utils import get_device, make_hf_pipeline

logger = logging.getLogger(__name__)


def load_topic_pipeline(model_name: str) -> Pipeline:
    """Load and return a HuggingFace zero-shot-classification pipeline."""
    pipeline_obj = make_hf_pipeline(
        "zero-shot-classification", model=model_name, device=get_device()
    )
    logger.info("Loaded zero-shot topic pipeline: %s", model_name)
    return pipeline_obj


def predict_topic(
    text: str,
    candidate_labels: list[str],
    topic_pipeline: Pipeline,
    hypothesis_template: str,
) -> Optional[str]:
    """Run zero-shot classification and return the highest-scoring label, or None."""
    if not text:
        logger.warning("predict_topic called with empty text; returning None.")
        return None

    try:
        result = cast(
            dict,
            topic_pipeline(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template=hypothesis_template,
            ),
        )
    except Exception:
        logger.warning("predict_topic pipeline raised; returning None.", exc_info=True)
        return None

    try:
        return result["labels"][0]
    except (KeyError, IndexError, TypeError):
        logger.warning("predict_topic could not extract top label from result; returning None.")
        return None


def predict_all_topics(
    articles: list[dict],
    candidate_labels: list[str],
    topic_pipeline: Pipeline,
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


def plot_topic_confusion_matrix(eval_results: dict, candidate_labels: list[str]) -> None:
    """Heatmap of true topic vs predicted topic counts.

    Diagonal cells are correct predictions; off-diagonal cells are errors.
    Order of rows/columns follows candidate_labels for deterministic layout.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    results = eval_results.get("results", [])
    if not results:
        logger.warning("plot_topic_confusion_matrix: no results to plot.")
        return

    labels = list(candidate_labels)
    label_index = {label.lower().strip(): i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for row in results:
        true_key = (row.get("topic") or "").lower().strip()
        pred_key = (row.get("predicted_topic") or "").lower().strip()
        if true_key not in label_index or pred_key not in label_index:
            continue
        matrix[label_index[true_key], label_index[pred_key]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted topic")
    ax.set_ylabel("True topic")
    ax.set_title("Confusion matrix — true vs predicted topic")
    max_val = matrix.max() if matrix.size else 0
    for i in range(n):
        for j in range(n):
            colour = "white" if max_val > 0 and matrix[i, j] > max_val / 2 else "black"
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color=colour)
    fig.colorbar(im, ax=ax, label="Article count")
    fig.tight_layout()


def plot_topic_error_breakdown(eval_results: dict) -> None:
    """Three-panel breakdown of topic-prediction errors.

    Panel 1: errors grouped by TRUE topic (which topics get misinterpreted).
    Panel 2: errors grouped by PREDICTED topic (which predictions are unreliable).
    Panel 3: overall correct / wrong / None counts.
    """
    from collections import Counter

    import matplotlib.pyplot as plt

    results = eval_results.get("results", [])
    if not results:
        logger.warning("plot_topic_error_breakdown: no results to plot.")
        return

    errors_by_true: Counter = Counter()
    errors_by_pred: Counter = Counter()
    correct = 0
    wrong = 0
    for row in results:
        if row.get("match"):
            correct += 1
        else:
            wrong += 1
            errors_by_true[row.get("topic", "(unknown)")] += 1
            errors_by_pred[row.get("predicted_topic", "(unknown)")] += 1

    total_sampled = eval_results.get("total_sampled", 0)
    evaluated = eval_results.get("evaluated", 0)
    none_count = max(total_sampled - evaluated, 0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    def _bar(ax, counter: Counter, title: str, colour: str) -> None:
        if not counter:
            ax.text(0.5, 0.5, "No errors", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        sorted_items = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
        labels = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        ax.bar(labels, values, color=colour)
        ax.set_title(title)
        ax.set_ylabel("Wrong predictions")
        ax.tick_params(axis="x", rotation=30)

    _bar(
        axes[0],
        errors_by_true,
        "Errors by TRUE topic\n(which topics get misinterpreted?)",
        "steelblue",
    )
    _bar(
        axes[1],
        errors_by_pred,
        "Errors by PREDICTED topic\n(which predictions are unreliable?)",
        "coral",
    )

    accuracy_pct = (correct / evaluated * 100) if evaluated > 0 else 0.0
    axes[2].bar(
        ["Correct", "Wrong", "None"],
        [correct, wrong, none_count],
        color=["seagreen", "indianred", "gray"],
    )
    axes[2].set_title(f"Overall: {correct}/{evaluated} correct ({accuracy_pct:.0f}%)")
    axes[2].set_ylabel("Article count")

    fig.suptitle("Topic prediction error breakdown")
    fig.tight_layout()
