"""Data utilities for the MABe mouse behavior detection competition."""

from __future__ import annotations

from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COMPETITION = "mabe-mouse-behavior-detection"
DEFAULT_DATA_ROOT = _WORKSPACE_ROOT / "data"


def raw_data_dir() -> Path:
    """Directory where raw Kaggle downloads are stored."""
    return DEFAULT_DATA_ROOT / "raw"


def processed_data_dir() -> Path:
    """Directory reserved for processed artifacts."""
    return DEFAULT_DATA_ROOT / "processed"


from .dataset import BehaviorClip, BehaviorDataset, discover_samples, train_val_split  # noqa: E402
from .download import build_parser, download_competition, main  # noqa: E402

__all__ = [
    "DEFAULT_COMPETITION",
    "DEFAULT_DATA_ROOT",
    "BehaviorClip",
    "BehaviorDataset",
    "build_parser",
    "download_competition",
    "discover_samples",
    "main",
    "processed_data_dir",
    "raw_data_dir",
    "train_val_split",
]
