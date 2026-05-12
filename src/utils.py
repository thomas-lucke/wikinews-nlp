"""Utility helpers for the Wikinews NLP pipeline."""

import gc
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

RANDOM_SEED: int = 42


def setup_logging(
    log_file: str = "logs/pipeline.log", level: int = logging.INFO
) -> None:
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
    return 0 if torch.cuda.is_available() else -1


def release_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU cache cleared.")
