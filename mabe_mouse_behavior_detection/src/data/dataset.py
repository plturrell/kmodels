"""Lightweight dataset scaffolding for MABe mouse behavior detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BehaviorClip:
    """Container describing a single labelled asset."""

    sample_id: str
    asset_path: Path
    label_path: Optional[Path]
    metadata: Dict[str, object]


class BehaviorDataset(Dataset):
    """PyTorch Dataset wrapper around :class:`BehaviorClip` entries."""

    def __init__(
        self,
        samples: Sequence[BehaviorClip],
        transform: Optional[Callable[[BehaviorClip], Dict[str, object]]] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:  # noqa: D401
        """Return dataset length."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:  # noqa: D401
        """Return a dictionary describing the selected sample."""
        clip = self.samples[index]
        if self.transform is not None:
            return self.transform(clip)
        return {
            "sample_id": clip.sample_id,
            "asset_path": clip.asset_path,
            "label_path": clip.label_path,
            "metadata": clip.metadata,
        }


def discover_samples(
    metadata_csv: Path,
    asset_root: Path,
    *,
    asset_column: str = "video_path",
    label_column: Optional[str] = "label_path",
    id_column: Optional[str] = None,
    metadata_columns: Optional[Iterable[str]] = None,
) -> List[BehaviorClip]:
    """Create :class:`BehaviorClip` objects from a manifest CSV.

    Parameters
    ----------
    metadata_csv:
        CSV file that references the competition assets (usually `train.csv`).
    asset_root:
        Directory containing the primary competition assets (videos or pose features).
    asset_column:
        Column in the CSV that contains relative asset paths.
    label_column:
        Column containing label JSON paths (if provided by the competition).
    id_column:
        Optional CSV column used as the unique identifier for each sample.
    metadata_columns:
        Extra columns to preserve in the metadata dictionary.
    """
    df = pd.read_csv(metadata_csv)
    required_cols = [asset_column]
    if id_column:
        required_cols.append(id_column)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"CSV {metadata_csv} missing required column(s): {', '.join(missing)}"
        )

    metadata_cols = list(metadata_columns or [])
    samples: List[BehaviorClip] = []
    for _, row in df.iterrows():
        asset_value = row[asset_column]
        if pd.isna(asset_value):
            raise ValueError(f"Missing asset path in column '{asset_column}'.")
        asset_path = asset_root / str(asset_value)
        label_path = None
        if label_column and pd.notna(row.get(label_column)):
            label_path = asset_root / str(row[label_column])

        if id_column:
            raw_id = row[id_column]
            sample_id = str(raw_id) if not pd.isna(raw_id) else asset_path.stem
        else:
            sample_id = asset_path.stem

        metadata = {}
        for col in metadata_cols:
            if col not in row:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            metadata[col] = value
        samples.append(
            BehaviorClip(
                sample_id=sample_id,
                asset_path=asset_path,
                label_path=label_path if label_path is not None else None,
                metadata=metadata,
            )
        )
    return samples


def train_val_split(
    samples: Sequence[BehaviorClip],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by: Optional[Callable[[BehaviorClip], object]] = None,
) -> Tuple[List[BehaviorClip], List[BehaviorClip]]:
    """Split samples into train/validation lists."""
    stratify_labels = None
    if stratify_by is not None:
        stratify_labels = [stratify_by(sample) for sample in samples]
    train_samples, val_samples = train_test_split(
        list(samples), test_size=test_size, random_state=random_state, stratify=stratify_labels
    )
    return train_samples, val_samples


__all__ = ["BehaviorClip", "BehaviorDataset", "discover_samples", "train_val_split"]
