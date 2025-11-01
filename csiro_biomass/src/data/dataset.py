"""Dataset helpers for the CSIRO Image2Biomass competition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetConfig:
    image_column: str
    target_columns: Optional[Sequence[str]]
    metadata_columns: Sequence[str]
    metadata_mean: Optional[np.ndarray]
    metadata_std: Optional[np.ndarray]


class BiomassDataset(Dataset):
    """Simple dataset that returns image tensor, target vector, metadata vector, and id."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        image_column: str,
        transforms: Optional[Callable] = None,
        target_columns: Optional[Sequence[str]] = None,
        metadata_columns: Optional[Sequence[str]] = None,
        metadata_mean: Optional[np.ndarray] = None,
        metadata_std: Optional[np.ndarray] = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_column = image_column
        self.transforms = transforms
        self.target_columns = tuple(target_columns) if target_columns else tuple()
        self.metadata_columns = tuple(metadata_columns) if metadata_columns else tuple()
        self.metadata_mean = (
            np.asarray(metadata_mean, dtype="float32")
            if metadata_mean is not None
            else None
        )
        self.metadata_std = (
            np.asarray(metadata_std, dtype="float32")
            if metadata_std is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, filename: str) -> Image.Image:
        path = self.image_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = self._load_image(str(row[self.image_column]))
        if self.transforms:
            image = self.transforms(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        if self.target_columns:
            target_values = row[list(self.target_columns)].to_numpy(dtype="float32")
            target = torch.from_numpy(target_values)
        else:
            target = torch.empty(0, dtype=torch.float32)

        if self.metadata_columns:
            meta_values = row[list(self.metadata_columns)].to_numpy(dtype="float32")
            if self.metadata_mean is not None and self.metadata_std is not None:
                meta_values = (meta_values - self.metadata_mean) / self.metadata_std
            metadata = torch.from_numpy(meta_values)
        else:
            metadata = torch.empty(0, dtype=torch.float32)

        identifier = str(row[self.image_column])
        return image, target, metadata, identifier


class PairedBiomassDataset(Dataset):
    """Dataset that provides temporally paired samples for physics-informed objectives."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_dir: Path,
        image_column: str,
        time_column: str,
        group_column: str,
        transforms: Optional[Callable] = None,
        target_columns: Optional[Sequence[str]] = None,
        metadata_columns: Optional[Sequence[str]] = None,
        metadata_mean: Optional[np.ndarray] = None,
        metadata_std: Optional[np.ndarray] = None,
    ) -> None:
        self.inner = BiomassDataset(
            frame,
            image_dir=image_dir,
            image_column=image_column,
            transforms=transforms,
            target_columns=target_columns,
            metadata_columns=metadata_columns,
            metadata_mean=metadata_mean,
            metadata_std=metadata_std,
        )
        base_frame = self.inner.frame
        if time_column not in base_frame.columns or group_column not in base_frame.columns:
            self.pairs: List[Tuple[int, int, float]] = []
            return

        annotated = base_frame[[time_column, group_column]].copy()
        annotated[time_column] = pd.to_datetime(annotated[time_column], errors="coerce")
        annotated = annotated.dropna(subset=[time_column, group_column])

        pairs: List[Tuple[int, int, float]] = []
        for _, group in annotated.groupby(group_column):
            group_sorted = group.sort_values(time_column)
            indices = group_sorted.index.to_list()
            times = group_sorted[time_column].to_numpy()
            if len(indices) < 2:
                continue
            deltas = np.diff(times).astype("timedelta64[D]")
            for (idx_a, idx_b), delta in zip(zip(indices[:-1], indices[1:]), deltas):
                delta_days = max(float(delta / np.timedelta64(1, "D")), 1.0)
                pairs.append((idx_a, idx_b, delta_days))
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        idx_a, idx_b, delta_days = self.pairs[index]
        img_a, target_a, meta_a, _ = self.inner[idx_a]
        img_b, target_b, meta_b, _ = self.inner[idx_b]
        delta_tensor = torch.tensor(delta_days, dtype=torch.float32)
        return img_a, target_a, meta_a, img_b, target_b, meta_b, delta_tensor

def create_inference_loader(
    df: pd.DataFrame,
    image_dir: Path,
    *,
    image_column: str,
    transforms: Callable,
    metadata_columns: Sequence[str],
    metadata_mean: Optional[Sequence[float]],
    metadata_std: Optional[Sequence[float]],
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    dataset = BiomassDataset(
        df,
        image_dir=image_dir,
        image_column=image_column,
        transforms=transforms,
        target_columns=None,
        metadata_columns=metadata_columns,
        metadata_mean=metadata_mean,
        metadata_std=metadata_std,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
