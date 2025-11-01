"""CLI to train a neural feature extractor for Hull Tactical Market Prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .config.experiment import (
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from .solvers.feature_extractor_solver import run_feature_extraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a neural feature extractor for Hull Tactical Market Prediction."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to the training CSV.")
    parser.add_argument("--test-csv", type=Path, help="Optional path to the test CSV.")
    parser.add_argument("--target-column", default="forward_returns", help="Target column in the training CSV.")
    parser.add_argument("--id-column", default="date_id", help="Identifier column present in train/test files.")
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional columns to drop before feature generation (repeat as needed).",
    )
    parser.add_argument(
        "--model",
        default="cnn",
        choices=("cnn", "lstm"),
        help="Model architecture to use (default: cnn).",
    )
    parser.add_argument(
        "--hidden-dim",
        action="append",
        dest="hidden_dims",
        type=int,
        help="Hidden layer dimension for the neural network (repeat for multiple layers).",
    )
    parser.add_argument("--embedding-dim", type=int, default=64, help="Dimension of the output feature embeddings.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the neural network.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for AdamW.")
_
