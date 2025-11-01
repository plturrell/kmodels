"""LightningDataModule for Hull Tactical Market Prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config.experiment import ExperimentConfig, TrainerConfig
from ..features.build_features import FeatureBuilderConfig, align_features, build_feature_frame


@dataclass
class PreparedFeatures:
    train_features: pd.DataFrame
    train_target: pd.Series
    train_ids: pd.Series
    test_features: Optional[pd.DataFrame]
    test_ids: Optional[pd.Series]
    feature_means: np.ndarray
    feature_stds: np.ndarray


def _standardise_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    values = frame.to_numpy(dtype=np.float32)
    means = np.nanmean(values, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    centered = values - means
    stds = np.nanstd(centered, axis=0)
    stds = np.where((stds > 0) & np.isfinite(stds), stds, 1.0)
    normalised = centered / stds
    normalised = np.where(np.isfinite(normalised), normalised, 0.0)
    return pd.DataFrame(normalised, columns=frame.columns, index=frame.index), means.astype(np.float32), stds.astype(np.float32)


def _apply_standardisation(frame: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> pd.DataFrame:
    values = frame.to_numpy(dtype=np.float32)
    normalised = (values - means) / stds
    normalised = np.where(np.isfinite(normalised), normalised, 0.0)
    return pd.DataFrame(normalised, columns=frame.columns, index=frame.index)


def prepare_features(config: ExperimentConfig) -> PreparedFeatures:
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv) if config.test_csv else None

    feature_cfg = FeatureBuilderConfig(
        drop_columns=[config.target_column, config.id_column, *config.features.drop_columns],
        lag_steps=tuple(config.features.lag_steps),
        max_nan_ratio=config.features.max_nan_ratio,
        drop_constant=config.features.drop_constant,
        rolling_windows=tuple(config.features.rolling_windows),
        rolling_stats=tuple(config.features.rolling_stats),
    )

    if test_df is not None:
        train_features, test_features = align_features(train_df, test_df, feature_cfg)
    else:
        train_features = build_feature_frame(train_df, feature_cfg)
        test_features = None

    train_norm, means, stds = _standardise_frame(train_features)
    if test_features is not None:
        test_norm = _apply_standardisation(test_features, means, stds)
    else:
        test_norm = None

    target = train_df[config.target_column].astype(np.float32)
    train_ids = train_df[config.id_column]
    test_ids = test_df[config.id_column] if test_df is not None else None
    return PreparedFeatures(train_norm, target, train_ids, test_norm, test_ids, means, stds)


def create_dataloaders(
    prepared: PreparedFeatures,
    trainer_cfg: TrainerConfig,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], np.ndarray]:
    features_tensor = torch.from_numpy(prepared.train_features.to_numpy(dtype=np.float32))
    target_tensor = torch.from_numpy(prepared.train_target.to_numpy(dtype=np.float32)).unsqueeze(-1)

    num_samples = features_tensor.shape[0]
    val_size = max(1, int(num_samples * trainer_cfg.val_fraction))
    train_size = num_samples - val_size
    if train_size <= 0:
        raise ValueError("Validation fraction too large for dataset size.")

    if trainer_cfg.time_series_holdout:
        train_idx = slice(0, train_size)
        val_idx = slice(train_size, num_samples)
        train_features = features_tensor[train_idx]
        train_targets = target_tensor[train_idx]
        val_features = features_tensor[val_idx]
        val_targets = target_tensor[val_idx]
    else:
        permutation = torch.randperm(num_samples)
        train_indices = permutation[:train_size]
        val_indices = permutation[train_size:]
        train_features = features_tensor[train_indices]
        train_targets = target_tensor[train_indices]
        val_features = features_tensor[val_indices]
        val_targets = target_tensor[val_indices]

    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    test_loader: Optional[DataLoader]
    if prepared.test_features is not None:
        test_tensor = torch.from_numpy(prepared.test_features.to_numpy(dtype=np.float32))
        test_dataset = TensorDataset(test_tensor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=trainer_cfg.batch_size,
            shuffle=False,
            num_workers=trainer_cfg.num_workers,
            pin_memory=True,
        )
    else:
        test_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=not trainer_cfg.time_series_holdout,
        num_workers=trainer_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        num_workers=trainer_cfg.num_workers,
        pin_memory=True,
    )
    if trainer_cfg.time_series_holdout:
        val_indices_array = np.arange(train_size, num_samples)
    else:
        val_indices_array = val_indices.detach().cpu().numpy() if hasattr(val_indices, "detach") else np.array(val_indices)

    return train_loader, val_loader, test_loader, val_indices_array
