"""Data utilities for the CAFA 6 Protein Function Prediction competition."""

from __future__ import annotations

from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COMPETITION = "cafa-6-protein-function-prediction"
DEFAULT_DATA_ROOT = _WORKSPACE_ROOT / "data"


def raw_data_dir() -> Path:
    """Return the directory where raw Kaggle downloads are stored."""
    return DEFAULT_DATA_ROOT / "raw"


def processed_data_dir() -> Path:
    """Return the directory reserved for intermediate processed artifacts."""
    return DEFAULT_DATA_ROOT / "processed"

from .dataset import (  # noqa: E402
    AMINO_ACIDS,
    ProteinDataset,
    ProteinSample,
    build_samples,
    create_dataloaders,
    load_go_terms,
    load_go_terms_long_format,
    load_sequences_from_fasta,
    load_split_from_json,
    train_val_split,
)

__all__ = [
    "DEFAULT_COMPETITION",
    "DEFAULT_DATA_ROOT",
    "AMINO_ACIDS",
    "ProteinDataset",
    "ProteinSample",
    "build_samples",
    "create_dataloaders",
    "load_go_terms",
    "load_go_terms_long_format",
    "load_sequences_from_fasta",
    "load_split_from_json",
    "processed_data_dir",
    "raw_data_dir",
    "train_val_split",
]
