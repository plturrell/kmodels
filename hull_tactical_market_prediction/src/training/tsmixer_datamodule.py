"""Dataloader for the TSMixer model.

This module handles the sequence-to-sequence data preparation required by TSMixer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config.experiment import TrainerConfig
from .datamodule import PreparedFeatures


@dataclass
class TSMixerDataConfig:
    sequence_length: int = 60
    prediction_length: int = 1


class TSMixerDataset(Dataset):
    """A PyTorch Dataset for creating sequence-to-sequence samples."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
        prediction_length: int,
    ):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_length + 1

    def __getitem__(self, idx):
        seq_start = idx
        seq_end = idx + self.sequence_length
        target_start = seq_end
        target_end = target_start + self.prediction_length

        return (
            torch.from_numpy(self.features[seq_start:seq_end]).float(),
            torch.from_numpy(self.targets[target_start:target_end]).float(),
        )


def create_tsmixer_dataloaders(
    prepared: PreparedFeatures,
    trainer_config: TrainerConfig,
    data_config: TSMixerDataConfig,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create DataLoaders for the TSMixer model."""

    # Time-series holdout for validation
    val_size = int(len(prepared.train_features) * trainer_config.val_fraction)
    train_size = len(prepared.train_features) - val_size

    train_features = prepared.train_features.iloc[:train_size]
    val_features = prepared.train_features.iloc[train_size:]
    train_target = prepared.train_target.iloc[:train_size]
    val_target = prepared.train_target.iloc[train_size:]

    train_dataset = TSMixerDataset(
        train_features.to_numpy(),
        train_target.to_numpy(),
        data_config.sequence_length,
        data_config.prediction_length,
    )

    val_dataset = TSMixerDataset(
        val_features.to_numpy(),
        val_target.to_numpy(),
        data_config.sequence_length,
        data_config.prediction_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
        pin_memory=True,
    )

    test_loader = None
    if prepared.test_features is not None:
        # For inference, we take the last `sequence_length` points of the training data
        # to predict the first point of the test set.
        test_features_full = pd.concat([prepared.train_features, prepared.test_features])
        test_dataset = TSMixerDataset(
            test_features_full.to_numpy(),
            np.zeros(len(test_features_full)),  # Dummy targets
            data_config.sequence_length,
            data_config.prediction_length,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=False,
            num_workers=trainer_config.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
