"""PyTorch Lightning data module for the MABe competition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from ..data import BehaviorDataset, BehaviorClip, discover_samples, train_val_split


@dataclass
class SequenceTransformConfig:
    array_key: str
    sequence_length: int
    time_axis: int
    center: bool
    standardize: bool


def _load_sequence(path: Path, array_key: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if array_key not in data:
                raise KeyError(
                    f"Array key '{array_key}' not present in {path.name}. "
                    f"Available keys: {', '.join(data.files)}"
                )
            array = data[array_key]
    elif suffix == ".npy":
        array = np.load(path)
    else:
        raise ValueError(f"Unsupported sequence format: {path.suffix}")
    return np.asarray(array, dtype=np.float32)


def _prepare_sequence(
    array: np.ndarray,
    *,
    sequence_length: int,
    time_axis: int,
    center: bool,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim < 2:
        raise ValueError(
            f"Expected array with >= 2 dimensions (time + features), received {array.shape}"
        )
    if time_axis != 0:
        array = np.moveaxis(array, time_axis, 0)
    total_frames = array.shape[0]
    target_length = sequence_length if sequence_length > 0 else total_frames
    if total_frames >= target_length:
        indices = np.linspace(0, total_frames - 1, target_length).astype(int)
        array = array[indices]
        mask = np.ones(target_length, dtype=np.float32)
    else:
        pad_count = target_length - total_frames
        pad_shape = (pad_count,) + array.shape[1:]
        padding = np.zeros(pad_shape, dtype=array.dtype)
        array = np.concatenate([array, padding], axis=0)
        mask = np.zeros(target_length, dtype=np.float32)
        mask[: total_frames] = 1.0

    flat = array.reshape(target_length, -1).astype(np.float32, copy=False)
    if center:
        flat = flat - flat.mean(axis=0, keepdims=True)
    if standardize:
        std = flat.std(axis=0, keepdims=True)
        flat = flat / np.where(std < 1e-6, 1.0, std)
    return flat, mask


def _build_transform(
    *,
    transform_cfg: SequenceTransformConfig,
    label_to_index: Optional[Dict[object, int]],
    target_column: Optional[str],
):
    def transform(clip: BehaviorClip) -> Dict[str, object]:
        array = _load_sequence(clip.asset_path, array_key=transform_cfg.array_key)
        flat, mask = _prepare_sequence(
            array,
            sequence_length=transform_cfg.sequence_length,
            time_axis=transform_cfg.time_axis,
            center=transform_cfg.center,
            standardize=transform_cfg.standardize,
        )
        sample: Dict[str, object] = {
            "sample_id": clip.sample_id,
            "inputs": torch.from_numpy(flat),
            "mask": torch.from_numpy(mask),
        }
        if label_to_index is not None and target_column:
            target_value = clip.metadata.get(target_column)
            if target_value is None:
                raise ValueError(
                    f"Sample {clip.sample_id} missing target column '{target_column}'."
                )
            sample["target"] = torch.tensor(label_to_index[target_value], dtype=torch.long)
        return sample

    return transform


class BehaviorLightningDataModule(pl.LightningDataModule):
    """Lightning wrapper around the behaviour sequence datasets."""

    def __init__(
        self,
        *,
        train_csv: Path,
        asset_root: Optional[Path],
        asset_column: str,
        target_column: str,
        id_column: str,
        metadata_columns: Iterable[str],
        batch_size: int,
        val_batch_size: Optional[int],
        val_fraction: float,
        num_workers: int,
        seed: int,
        transform_cfg: SequenceTransformConfig,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.train_csv = train_csv
        self.asset_root = asset_root
        self.asset_column = asset_column
        self.target_column = target_column
        self.id_column = id_column
        self.metadata_columns = list(metadata_columns)
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.val_fraction = val_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.transform_cfg = transform_cfg
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

        self._train_dataset: Optional[BehaviorDataset] = None
        self._val_dataset: Optional[BehaviorDataset] = None
        self.label_to_index: Dict[object, int] = {}
        self.index_to_label: List[object] = []
        self.input_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        samples = discover_samples(
            self.train_csv,
            self.asset_root or self.train_csv.parent,
            asset_column=self.asset_column,
            label_column=self.target_column,
            id_column=self.id_column,
            metadata_columns=self._metadata_columns_with_ids(),
        )

        train_samples = [clip for clip in samples if self.target_column in clip.metadata]
        if not train_samples:
            raise RuntimeError(
                f"No training samples found with target column '{self.target_column}'."
            )

        labels = sorted(
            {clip.metadata[self.target_column] for clip in train_samples},
            key=lambda value: str(value),
        )
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        self.index_to_label = labels

        if self.val_fraction > 0:
            train_samples, val_samples = train_val_split(
                train_samples,
                test_size=self.val_fraction,
                random_state=self.seed,
                stratify_by=lambda clip: clip.metadata[self.target_column],
            )
        else:
            val_samples = []

        if self.max_train_samples:
            train_samples = train_samples[: self.max_train_samples]
        if self.max_val_samples and val_samples:
            val_samples = val_samples[: self.max_val_samples]

        transform = _build_transform(
            transform_cfg=self.transform_cfg,
            label_to_index=self.label_to_index,
            target_column=self.target_column,
        )
        self._train_dataset = BehaviorDataset(train_samples, transform=transform)

        if val_samples:
            val_transform = _build_transform(
                transform_cfg=self.transform_cfg,
                label_to_index=self.label_to_index,
                target_column=self.target_column,
            )
            self._val_dataset = BehaviorDataset(val_samples, transform=val_transform)
        else:
            self._val_dataset = None

        sample_inputs = self._train_dataset[0]["inputs"]
        if sample_inputs.ndim != 2:
            raise ValueError(
                f"Expected sequence tensor shape (T, F), received {tuple(sample_inputs.shape)}"
            )
        self.input_dim = sample_inputs.shape[-1]

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Data module not setup. Call setup() before requesting loaders.")
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_dataset is None:
            return None
        return DataLoader(
            self._val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _metadata_columns_with_ids(self) -> List[str]:
        columns: List[str] = []
        for column in self.metadata_columns:
            if column and column not in columns:
                columns.append(column)
        for column in (self.target_column, self.id_column):
            if column and column not in columns:
                columns.append(column)
        return columns

    def build_test_dataset(self, *, test_csv: Path, asset_root: Optional[Path]) -> BehaviorDataset:
        if not self.index_to_label:
            raise RuntimeError("Call setup() before building the test dataset.")

        metadata_columns = [self.id_column] if self.id_column else []
        samples = discover_samples(
            test_csv,
            asset_root or test_csv.parent,
            asset_column=self.asset_column,
            label_column=None,
            id_column=self.id_column,
            metadata_columns=metadata_columns,
        )
        transform = _build_transform(
            transform_cfg=self.transform_cfg,
            label_to_index=None,
            target_column=None,
        )
        return BehaviorDataset(samples, transform=transform)


