"""Lightning data module for ECG digitisation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..config.training import AugmentationConfig, ExperimentConfig
from ..data.dataset import (
    ECGDigitizationDataset,
    ECGSample,
    create_dataloaders,
    discover_samples,
    load_samples_from_metadata,
)
from ..features.transforms import build_eval_transform, build_train_transform


def _prepare_samples(config: ExperimentConfig) -> List[ECGSample]:
    if config.train_csv and Path(config.train_csv).exists():
        return load_samples_from_metadata(
            config.train_csv,
            image_root=config.image_dir,
            signal_root=config.signal_dir,
            id_column=config.id_column,
            image_column=config.image_column,
            signal_column=config.signal_column,
            lead_column=config.lead_column,
        )
    return discover_samples(
        config.image_dir,
        signal_dir=config.signal_dir,
        assume_lead_from_parent=True,
    )


def _group_key_factory(column: Optional[str]):
    if column is None:
        return None

    def group_key(sample: ECGSample) -> str:
        value = sample.metadata.get(column)
        if value is not None:
            return str(value)
        return sample.ecg_id.split("_")[0]

    return group_key


@dataclass
class _LoaderBundle:
    train: DataLoader
    val: DataLoader


class ECGDataModule(pl.LightningDataModule):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self._loaders: Optional[_LoaderBundle] = None
        self.signal_length: Optional[int] = config.signal_length
        self.signal_channels: Optional[int] = config.signal_channels

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        samples = _prepare_samples(self.config)
        if not samples:
            raise RuntimeError("No training samples discovered.")

        aug = self.config.augmentation
        train_tf = build_train_transform(
            image_size=aug.image_size,
            hflip_prob=aug.horizontal_flip,
            vflip_prob=aug.vertical_flip,
            affine_degrees=aug.random_rotation,
        )
        val_tf = build_eval_transform(image_size=aug.image_size)

        train_loader, val_loader = create_dataloaders(
            samples,
            batch_size=self.config.batch_size,
            val_fraction=self.config.val_fraction,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            train_transforms=train_tf,
            val_transforms=val_tf,
            preload_signals=self.config.preload_signals,
            group_key=_group_key_factory(self.config.group_column),
        )

        self._loaders = _LoaderBundle(train=train_loader, val=val_loader)

        if self.signal_length is None or self.signal_channels is None:
            dataset = train_loader.dataset  # type: ignore[attr-defined]
            if len(dataset) == 0:  # pragma: no cover - defensive guard
                raise RuntimeError("Training dataset is empty.")
            sample = dataset[0]
            signal = sample.get("signal")
            if signal is not None:
                if signal.ndim == 1:
                    inferred_channels = 1
                    inferred_length = signal.shape[0]
                elif signal.ndim == 2:
                    inferred_channels = signal.shape[0]
                    inferred_length = signal.shape[1]
                else:
                    raise RuntimeError(
                        f"Unsupported signal dimensions: {tuple(signal.shape)}"
                    )

                if self.signal_length is None:
                    self.signal_length = inferred_length
                elif self.signal_length != inferred_length:
                    raise RuntimeError(
                        "Configured signal_length does not match data-derived length "
                        f"({self.signal_length} vs {inferred_length})."
                    )

                if self.signal_channels is None:
                    self.signal_channels = inferred_channels
                elif self.signal_channels != inferred_channels:
                    raise RuntimeError(
                        "Configured signal_channels does not match data-derived channels "
                        f"({self.signal_channels} vs {inferred_channels})."
                    )
            else:
                if self.signal_length is None:
                    raise RuntimeError(
                        "Unable to infer signal length because no signal tensors were observed."
                    )
                if self.signal_channels is None:
                    self.signal_channels = 1

    def train_dataloader(self) -> DataLoader:
        if self._loaders is None:
            raise RuntimeError("Data module has not been setup yet.")
        return self._loaders.train

    def val_dataloader(self) -> DataLoader:
        if self._loaders is None:
            raise RuntimeError("Data module has not been setup yet.")
        return self._loaders.val

    def predict_dataloader(self, test_samples: List[ECGSample]) -> DataLoader:
        eval_tf = build_eval_transform(image_size=self.config.augmentation.image_size)
        dataset = ECGDigitizationDataset(
            test_samples,
            transforms=eval_tf,
            preload_signals=self.config.preload_signals,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


