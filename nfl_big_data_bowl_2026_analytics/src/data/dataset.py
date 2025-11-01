"""Dataset assembly utilities for the NFL analytics workspace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config.experiment import DatasetConfig, FeatureConfig, TrainingConfig
from .loaders import (
    available_train_weeks,
    load_supplementary,
    load_train_week_pair,
    resolve_bundle_root,
)


MERGE_KEYS = ["game_id", "play_id", "nfl_id", "frame_id"]


def load_training_dataframe(
    dataset_cfg: DatasetConfig,
    *,
    weeks: Optional[Sequence[Tuple[int, int]]] = None,
    base_dir: Optional[Path | str] = None,
    feature_cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Load and merge weekly input/output CSVs into a single dataframe."""
    bundle_root = resolve_bundle_root(base_dir or dataset_cfg.data_root)
    if weeks is None:
        weeks = available_train_weeks(bundle_root)

    merged_frames: List[pd.DataFrame] = []
    for season, week in weeks:
        inputs, outputs = load_train_week_pair(
            season,
            week,
            base_dir=bundle_root,
        )
        outputs = outputs.rename(columns={"x": "target_x", "y": "target_y"})
        combined = outputs.merge(inputs, on=MERGE_KEYS, how="left", suffixes=("_target", ""))
        combined["season"] = season
        combined["week"] = week
        merged_frames.append(combined)

    if not merged_frames:
        raise RuntimeError("No training frames were assembled from the provided weeks.")

    dataset = pd.concat(merged_frames, ignore_index=True)

    if dataset_cfg.include_supplementary:
        supplementary = load_supplementary(base_dir=bundle_root)
        if not supplementary.empty:
            supplementary = supplementary.drop_duplicates(subset=["game_id", "play_id"])
            dataset = dataset.merge(
                supplementary,
                on=["game_id", "play_id"],
                how="left",
                suffixes=("", "_supp"),
            )

    if feature_cfg is not None:
        dataset = apply_feature_flags(dataset, feature_cfg)

    return dataset


def split_train_validation(
    dataframe: pd.DataFrame,
    config: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/validation partitions."""
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be within (0, 1).")

    val_fraction = config.val_fraction
    seed = config.seed

    groups = None
    fold_column = config.dataset.fold_column
    if fold_column and fold_column in dataframe.columns:
        groups = dataframe[fold_column].to_numpy()

    indices = np.arange(len(dataframe))
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(indices, groups=groups))
    else:
        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )

    train_df = dataframe.iloc[train_idx].reset_index(drop=True)
    val_df = dataframe.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def _numeric_feature_columns(
    dataframe: pd.DataFrame,
    *,
    id_columns: Iterable[str],
    target_columns: Iterable[str],
) -> List[str]:
    """Return numeric feature column names, excluding ids/targets."""
    numeric_cols: List[str] = []
    excluded = set(id_columns) | set(target_columns)
    for column, dtype in dataframe.dtypes.items():
        if column in excluded:
            continue
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(column)
    return numeric_cols


@dataclass
class FeatureBundle:
    features: np.ndarray
    targets: np.ndarray
    ids: pd.DataFrame
    feature_columns: List[str]
    target_columns: List[str]
    id_columns: List[str]


def extract_numeric_features(
    dataframe: pd.DataFrame,
    dataset_cfg: DatasetConfig,
) -> FeatureBundle:
    """Extract numeric features/targets from the dataframe."""
    df = dataframe.copy()
    for column in df.columns:
        if pd.api.types.is_bool_dtype(df[column]):
            df[column] = df[column].astype(np.float32)

    feature_columns = _numeric_feature_columns(
        df,
        id_columns=dataset_cfg.id_columns,
        target_columns=dataset_cfg.target_columns,
    )

    features = df[feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
    targets = df[dataset_cfg.target_columns].to_numpy(dtype=np.float32)
    identifiers = df[dataset_cfg.id_columns].copy()

    return FeatureBundle(
        features=features,
        targets=targets,
        ids=identifiers,
        feature_columns=feature_columns,
        target_columns=list(dataset_cfg.target_columns),
        id_columns=list(dataset_cfg.id_columns),
    )


class AnalyticsDataset(Dataset):
    """Torch dataset wrapping feature/target arrays."""

    def __init__(
        self,
        bundle: FeatureBundle,
        *,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ) -> None:
        self.features = bundle.features.copy()
        self.targets = bundle.targets.copy()
        self.ids_frame = bundle.ids.reset_index(drop=True)
        self.feature_columns = bundle.feature_columns
        self.target_columns = bundle.target_columns
        self.id_columns = bundle.id_columns

        if feature_mean is not None and feature_std is not None:
            self._normalise_inplace(feature_mean, feature_std)

    def _normalise_inplace(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> None:
        mean = feature_mean.astype(np.float32)
        std = feature_std.astype(np.float32)
        std = np.where(std == 0, 1.0, std)
        self.features = (self.features - mean) / std

    def __len__(self) -> int:  # noqa: D401
        return self.features.shape[0]

    def __getitem__(self, index: int):
        feature_tensor = torch.from_numpy(self.features[index])
        target_tensor = torch.from_numpy(self.targets[index])
        identifier = {
            column: self.ids_frame.at[index, column]
            for column in self.id_columns
        }
        return feature_tensor, target_tensor, identifier


def create_dataloader(
    dataset: AnalyticsDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )


def _compute_distance_to_ball_land(row: pd.Series) -> float:
    try:
        dx = float(row["x"]) - float(row["ball_land_x"])
        dy = float(row["y"]) - float(row["ball_land_y"])
        return float(np.hypot(dx, dy))
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _parse_game_clock(value: object) -> float:
    if not isinstance(value, str):
        return float("nan")
    parts = value.strip().split(":")
    if len(parts) != 2:
        return float("nan")
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return float(minutes * 60 + seconds)
    except ValueError:
        return float("nan")


def apply_feature_flags(
    dataframe: pd.DataFrame,
    feature_cfg: FeatureConfig,
) -> pd.DataFrame:
    """Augment dataframe with optional engineered features."""
    df = dataframe.copy()
    if feature_cfg.use_pairwise_distance and all(
        column in df.columns for column in ("x", "y", "ball_land_x", "ball_land_y")
    ):
        df["distance_to_ball_land"] = df.apply(_compute_distance_to_ball_land, axis=1)

    if feature_cfg.use_game_clock_seconds and "game_clock" in df.columns:
        df["game_clock_seconds"] = df["game_clock"].map(_parse_game_clock)
        df["is_two_minute_drill"] = df["game_clock_seconds"].apply(
            lambda t: float(t <= 120.0) if np.isfinite(t) else 0.0
        )

    return df


def compute_feature_stats(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)
