"""Utility helpers for the Wikinews NLP pipeline."""

import gc
import logging
import sys
from pathlib import Path
from typing import cast

import torch
from transformers import Pipeline
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

RANDOM_SEED: int = 42


def setup_logging(log_file: str = "logs/pipeline.log", level: int = logging.INFO) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(level)


def get_device() -> int:
    """Return the HuggingFace device index for model loading.

    Reads PIPELINE_DEVICE env var (set from config["device"] in the notebook):
      "auto"  → GPU (0) if CUDA available, else CPU (-1)
      "cpu"   → always CPU (-1)
      "cuda"  → always GPU (0); raises RuntimeError if CUDA is unavailable
    """
    import os

    setting = os.environ.get("PIPELINE_DEVICE", "auto").lower().strip()
    if setting == "cpu":
        return -1
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested in config but no CUDA GPU found.")
        return 0
    return 0 if torch.cuda.is_available() else -1


def release_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")
    else:
        logger.info("Model released (CPU run — no GPU cache to clear).")


def make_hf_pipeline(task: str, **kwargs) -> Pipeline:
    """Typed wrapper around `transformers.pipeline()`.

    The transformers stub has ~40 overloads keyed off the `task` literal and
    Pylance mis-resolves several of them. Taking `task` as a plain `str` here
    bypasses the literal-overload matching. The cast keeps the return type
    concrete so call sites don't need their own type assertions.
    """
    return cast(
        Pipeline, hf_pipeline(task, **kwargs)
    )  # pyright: ignore[reportCallIssue, reportArgumentType]
