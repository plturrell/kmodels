"""Data utilities for the PhysioNet ECG Image Digitization competition."""

from __future__ import annotations

from pathlib import Path

# Root of this workspace (two levels up from this file)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COMPETITION = "physionet-ecg-image-digitization"
DEFAULT_DATA_ROOT = _WORKSPACE_ROOT / "data"


def raw_data_dir() -> Path:
    """Return the directory where raw Kaggle downloads are stored."""
    return DEFAULT_DATA_ROOT / "raw"


def processed_data_dir() -> Path:
    """Return the directory reserved for intermediate processed artifacts."""
    return DEFAULT_DATA_ROOT / "processed"


from .dataset import (  # noqa: E402
    ECGDigitizationDataset,
    ECGSample,
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA_CSV,
    DEFAULT_SIGNAL_DIR,
    create_dataloaders,
    discover_samples,
    load_samples_from_metadata,
    train_val_split,
)
from .download import build_parser, download_competition, main  # noqa: E402

__all__ = [
    "DEFAULT_COMPETITION",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_IMAGE_DIR",
    "DEFAULT_METADATA_CSV",
    "DEFAULT_SIGNAL_DIR",
    "ECGDigitizationDataset",
    "ECGSample",
    "build_parser",
    "create_dataloaders",
    "discover_samples",
    "download_competition",
    "load_samples_from_metadata",
    "main",
    "processed_data_dir",
    "raw_data_dir",
    "train_val_split",
]
