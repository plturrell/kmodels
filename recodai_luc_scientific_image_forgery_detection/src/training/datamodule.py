"""Lightning data module for the RecoDAI forgery competition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..config.training import AugmentationConfig, ExperimentConfig
from ..data.dataset import ForgeryDataset, Sample, load_samples, train_val_split
from ..data.transforms import create_transforms


@dataclass
class _Dataloaders:
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader


class ForgeryDataModule(pl.LightningDataModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.class_names: Sequence[str] = ("authentic", "forged")
        self._train_dataset: Optional[ForgeryDataset] = None
        self._val_dataset: Optional[ForgeryDataset] = None
        self.class_weights: Optional[torch.Tensor] = None

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        samples = load_samples(self.config.data_root)
        train_samples, val_samples = train_val_split(
            samples,
            val_fraction=self.config.val_fraction,
            seed=self.config.seed,
        )

        if self.config.max_train_samples:
            train_samples = train_samples[: self.config.max_train_samples]
        if self.config.max_val_samples:
            val_samples = val_samples[: self.config.max_val_samples]

        self.class_weights = None
        if self.config.use_class_weights:
            counts = Counter(sample.label for sample in train_samples)
            num_classes = len(self.class_names)
            total = sum(counts.values())
            weights = []
            for idx in range(num_classes):
                count = counts.get(idx, 0)
                if count == 0 or total == 0:
                    weights.append(0.0)
                else:
                    weights.append(total / (num_classes * count))
            self.class_weights = torch.tensor(weights, dtype=torch.float32)

        # Create augmentation transforms
        train_transform = create_transforms(self.config.augmentation, is_training=True)
        val_transform = create_transforms(self.config.augmentation, is_training=False)

        self._train_dataset = ForgeryDataset(train_samples, transforms=train_transform)
        self._val_dataset = ForgeryDataset(val_samples, transforms=val_transform)

    def train_dataloader(self):  # noqa: D401
        if self._train_dataset is None:
            raise RuntimeError("Data module has not been setup.")
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):  # noqa: D401
        if self._val_dataset is None:
            raise RuntimeError("Data module has not been setup.")
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

