"""Shared helpers: logging setup, JSON/YAML I/O, model-id slugify, seeds."""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Configure root logger to write to logs/run_<timestamp>.log and stderr.

    Safe to call more than once — old handlers are cleared so re-running cells
    in a notebook doesn't duplicate log lines.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path


def slugify_model_id(model_id: str) -> str:
    """Turn 'openai/whisper-large-v3' into 'openai__whisper-large-v3' for filenames."""
    return model_id.replace("/", "__")


def read_yaml(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def set_seeds(seed: int = 42) -> None:
    """Make results reproducible where libraries support it."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
