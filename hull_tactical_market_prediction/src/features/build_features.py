"""Feature engineering helpers for Hull Tactical Market Prediction.

The dataset is delivered as a collection of daily factors and market signals.
These utilities keep the scaffolding light-weight so you can iterate quickly
outside Kaggle notebooks while still supporting richer tabular features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureBuilderConfig:
    """Configuration for the default feature builder."""

    drop_columns: Sequence[str] = ()
    lag_steps: Sequence[int] = (1, 5, 21)
    max_nan_ratio: float | None = 0.75
    drop_constant: bool = True
    rolling_windows: Sequence[int] = (5, 21)
    rolling_stats: Sequence[str] = field(default_factory=lambda: ("mean", "std"))


def _clean_numeric_frame(
    numeric: pd.DataFrame, config: FeatureBuilderConfig
) -> pd.DataFrame:
    cleaned = numeric.copy()

    if config.max_nan_ratio is not None:
        nan_ratio = cleaned.isna().mean()
        high_nan_cols = nan_ratio[nan_ratio > config.max_nan_ratio].index.tolist()
        if high_nan_cols:
            cleaned = cleaned.drop(columns=high_nan_cols)
            LOGGER.debug("Dropped %d high-missing features", len(high_nan_cols))

    if config.drop_constant and not cleaned.empty:
        constant_cols = cleaned.nunique(dropna=True)
        constant_cols = constant_cols[constant_cols <= 1].index.tolist()
        if constant_cols:
            cleaned = cleaned.drop(columns=constant_cols)
            LOGGER.debug("Dropped %d near-constant features", len(constant_cols))

    return cleaned


def _build_rolling_features(
    numeric: pd.DataFrame, config: FeatureBuilderConfig
) -> list[pd.DataFrame]:
    rolling_frames: list[pd.DataFrame] = []
    if not config.rolling_windows:
        return rolling_frames

    for window in config.rolling_windows:
        if window <= 0:
            LOGGER.warning("Ignoring non-positive rolling window: %s", window)
            continue
        rolling_window = numeric.rolling(window=window, min_periods=1)
        for stat in config.rolling_stats:
            if stat == "mean":
                roll_df = rolling_window.mean()
                suffix = f"_roll{window}_mean"
            elif stat == "std":
                roll_df = rolling_window.std(ddof=0)
                suffix = f"_roll{window}_std"
            elif stat == "min":
                roll_df = rolling_window.min()
                suffix = f"_roll{window}_min"
            elif stat == "max":
                roll_df = rolling_window.max()
                suffix = f"_roll{window}_max"
            else:  # pragma: no cover - defensive branch
                LOGGER.warning("Unsupported rolling stat '%s'; skipping", stat)
                continue

            roll_df.columns = [f"{col}{suffix}" for col in numeric.columns]
            rolling_frames.append(roll_df)
    return rolling_frames


def build_feature_frame(
    df: pd.DataFrame,
    config: FeatureBuilderConfig | None = None,
) -> pd.DataFrame:
    """Return a numeric feature matrix with engineered columns."""
    cfg = config or FeatureBuilderConfig()

    drop_cols = set(cfg.drop_columns)
    base = df.drop(columns=list(drop_cols), errors="ignore")
    numeric = base.select_dtypes(include=["number", "bool"]).astype(float)

    numeric = _clean_numeric_frame(numeric, cfg)
    features = [numeric]
    features.extend(_build_rolling_features(numeric, cfg))

    # Volatility features
    for window in cfg.rolling_windows:
        volatility = numeric.rolling(window=window).std()
        volatility.columns = [f"{col}_vol{window}" for col in numeric.columns]
        features.append(volatility)

    # Momentum features
    for short_window, long_window in [(5, 21), (21, 60)]:
        short_ma = numeric.rolling(window=short_window).mean()
        long_ma = numeric.rolling(window=long_window).mean()
        momentum = short_ma - long_ma
        momentum.columns = [f"{col}_mom{short_window}_{long_window}" for col in numeric.columns]
        features.append(momentum)

    for lag in cfg.lag_steps:
        lagged = numeric.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in numeric.columns]
        features.append(lagged)

    feature_frame = pd.concat(features, axis=1)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    return feature_frame


def align_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FeatureBuilderConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same transformation to train and test dataframes."""
    cfg = config or FeatureBuilderConfig()
    train_features = build_feature_frame(train_df, cfg)
    test_features = build_feature_frame(test_df, cfg)

    missing_cols = train_features.columns.difference(test_features.columns)
    for col in missing_cols:
        test_features[col] = np.nan
    test_features = test_features[train_features.columns]
    return train_features, test_features
