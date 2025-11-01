"""Data utilities for the NFL Big Data Bowl 2026 Prediction competition."""

from .download import build_parser, download_competition, main
from .loaders import (
    DEFAULT_DATA_ROOT,
    available_train_weeks,
    load_submission_format,
    load_test_input,
    load_train_inputs,
    load_train_outputs,
    load_train_week_input,
    load_train_week_output,
    load_train_week_pair,
)
from .trajectory_dataset import (
    TrajectoryDataset,
    TrajectorySample,
    collate_trajectories,
)

__all__ = [
    "DEFAULT_DATA_ROOT",
    "TrajectoryDataset",
    "TrajectorySample",
    "available_train_weeks",
    "build_parser",
    "collate_trajectories",
    "download_competition",
    "load_submission_format",
    "load_test_input",
    "load_train_inputs",
    "load_train_outputs",
    "load_train_week_input",
    "load_train_week_output",
    "load_train_week_pair",
    "main",
]
