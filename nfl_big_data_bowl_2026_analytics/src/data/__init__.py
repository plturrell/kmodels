"""Data access helpers for the analytics workspace."""

try:
    from .download import build_parser, download_competition, main  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    build_parser = None
    download_competition = None
    main = None

from .dataset import (
    AnalyticsDataset,
    FeatureBundle,
    apply_feature_flags,
    compute_feature_stats,
    create_dataloader,
    extract_numeric_features,
    load_training_dataframe,
    split_train_validation,
)
from .loaders import (
    DEFAULT_BUNDLE_DIRNAME,
    DEFAULT_DOWNLOAD_ROOT,
    available_train_weeks,
    load_supplementary,
    load_train_inputs,
    load_train_outputs,
    load_train_week_input,
    load_train_week_output,
    load_train_week_pair,
    resolve_bundle_root,
)

__all__ = [
    "AnalyticsDataset",
    "FeatureBundle",
    "apply_feature_flags",
    "compute_feature_stats",
    "create_dataloader",
    "extract_numeric_features",
    "load_training_dataframe",
    "split_train_validation",
    "DEFAULT_BUNDLE_DIRNAME",
    "DEFAULT_DOWNLOAD_ROOT",
    "available_train_weeks",
    "load_supplementary",
    "load_train_inputs",
    "load_train_outputs",
    "load_train_week_input",
    "load_train_week_output",
    "load_train_week_pair",
    "resolve_bundle_root",
]

if build_parser is not None and download_competition is not None and main is not None:
    __all__.extend(["build_parser", "download_competition", "main"])
