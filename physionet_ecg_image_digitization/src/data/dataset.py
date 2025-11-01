"""Dataset helpers tailored for the PhysioNet ECG Image Digitization challenge.

The implementation mirrors patterns from other competition workspaces in this
repository (for example the RecodAI forgery dataset utilities) so that
augmentation hooks, dataclasses, and deterministic splits feel familiar.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

COMPETITION_SLUG = "physionet-ecg-image-digitization"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_ROOT = PACKAGE_ROOT / "data" / "physionet_ecg_image_digitization"
DEFAULT_DATA_ROOT = DEFAULT_STORAGE_ROOT / "raw" / COMPETITION_SLUG
DEFAULT_IMAGE_DIR = DEFAULT_DATA_ROOT / "train_images"
DEFAULT_SIGNAL_DIR = DEFAULT_DATA_ROOT / "train_signals"
DEFAULT_METADATA_CSV = DEFAULT_DATA_ROOT / "train.csv"


@dataclass(frozen=True)
class ECGSample:
    """Container describing a paired ECG image and digitised waveform."""

    ecg_id: str
    image_path: Path
    signal_path: Optional[Path] = None
    lead: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


def _resolve_relative(path: Path, root: Path) -> Path:
    resolved = (root / path).resolve() if not path.is_absolute() else path
    if not resolved.exists():
        raise FileNotFoundError(f"Expected file not found: {resolved}")
    return resolved


def load_samples_from_metadata(
    csv_path: Path,
    *,
    image_root: Optional[Path] = None,
    signal_root: Optional[Path] = None,
    id_column: str = "ecg_id",
    image_column: str = "image",
    signal_column: str = "signal",
    lead_column: Optional[str] = "lead",
) -> List[ECGSample]:
    """Create samples from a metadata CSV describing image/signal locations.

    The CSV may contain relative paths. ``image_root`` and ``signal_root`` are
    joined with those relative values. Extra columns are preserved in the
    ``metadata`` field for downstream analysis.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in (id_column, image_column) if col not in df.columns]
    if missing:
        raise KeyError(
            f"CSV {csv_path} missing required columns: {', '.join(missing)}"
        )

    samples: List[ECGSample] = []
    drop_keys = {id_column, image_column}
    if signal_column:
        drop_keys.add(signal_column)
    if lead_column:
        drop_keys.add(lead_column)
    for _, row in df.iterrows():
        row_dict = {
            key: (None if pd.isna(value) else value) for key, value in row.items()
        }
        raw_id = row_dict.get(id_column)
        if raw_id is None:
            raise KeyError(f"Row missing ID column '{id_column}'.")
        ecg_id = str(raw_id)

        image_raw = row_dict.get(image_column)
        if image_raw is None or str(image_raw).strip() == "":
            raise FileNotFoundError(f"No image path provided for {ecg_id}.")
        image_path = _resolve_relative(
            Path(str(image_raw)), image_root or csv_path.parent
        )

        signal_path = None
        if signal_column and row_dict.get(signal_column) not in (None, ""):
            signal_path = _resolve_relative(
                Path(str(row_dict[signal_column])), signal_root or csv_path.parent
            )

        lead_value = None
        if lead_column and row_dict.get(lead_column) not in (None, ""):
            lead_value = str(row_dict[lead_column])

        metadata = {
            key: value
            for key, value in row_dict.items()
            if key not in drop_keys and value is not None
        }
        samples.append(
            ECGSample(
                ecg_id=ecg_id,
                image_path=image_path,
                signal_path=signal_path,
                lead=lead_value,
                metadata=metadata,
            )
        )
    return samples


def discover_samples(
    image_dir: Path,
    *,
    signal_dir: Optional[Path] = None,
    image_glob: str = "**/*.png",
    signal_suffix: str = ".npy",
    assume_lead_from_parent: bool = True,
) -> List[ECGSample]:
    """Infer samples by pairing images with signals based on filename stems."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if signal_dir is not None and not signal_dir.exists():
        raise FileNotFoundError(f"Signal directory not found: {signal_dir}")

    samples: List[ECGSample] = []
    for image_path in sorted(image_dir.glob(image_glob)):
        if not image_path.is_file():
            continue
        stem = image_path.stem
        signal_path = None
        if signal_dir is not None:
            candidate = signal_dir / f"{stem}{signal_suffix}"
            if candidate.exists():
                signal_path = candidate
        lead = image_path.parent.name if assume_lead_from_parent else None
        samples.append(
            ECGSample(
                ecg_id=stem,
                image_path=image_path,
                signal_path=signal_path,
                lead=lead,
                metadata={},
            )
        )
    if not samples:
        raise RuntimeError(
            f"No samples discovered in {image_dir} (glob='{image_glob}')."
        )
    return samples


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        array = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def _load_signal(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        array = np.loadtxt(path, delimiter=",")
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict):
            array = payload.get("signal") or payload.get("values")
        else:
            array = payload
        array = np.asarray(array, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported signal format: {path.suffix}")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        tensor = torch.from_numpy(array).unsqueeze(0)
    elif array.ndim == 2:
        tensor = torch.from_numpy(array)
        if tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.transpose(0, 1)
    else:
        tensor = torch.from_numpy(array.reshape(array.shape[0], -1))
    return tensor


class ECGDigitizationDataset(Dataset):
    """Return tensors representing image and waveform pairs."""

    def __init__(
        self,
        samples: Sequence[ECGSample],
        *,
        transforms: Optional[
            Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
        ] = None,
        preload_signals: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("Dataset initialised with an empty sample list.")
        self.samples = list(samples)
        self.transforms = transforms
        self._preload_signals = preload_signals
        self._signal_cache: Dict[str, torch.Tensor] = {}
        if preload_signals:
            for sample in self.samples:
                if sample.signal_path is not None:
                    self._signal_cache[sample.ecg_id] = _load_signal(
                        sample.signal_path
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_signal(self, sample: ECGSample) -> Optional[torch.Tensor]:
        if sample.signal_path is None:
            return None
        if sample.ecg_id in self._signal_cache:
            return self._signal_cache[sample.ecg_id]
        waveform = _load_signal(sample.signal_path)
        if self._preload_signals:
            self._signal_cache[sample.ecg_id] = waveform
        return waveform

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | None]:
        sample = self.samples[index]
        batch: Dict[str, torch.Tensor | str | None] = {
            "id": sample.ecg_id,
            "lead": sample.lead,
            "image": _load_image(sample.image_path),
        }
        waveform = self._resolve_signal(sample)
        if waveform is not None:
            batch["signal"] = waveform
        if self.transforms is not None:
            batch = self.transforms(batch)  # type: ignore[assignment]
        return batch


def train_val_split(
    samples: Sequence[ECGSample],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    group_key: Optional[Callable[[ECGSample], str]] = None,
) -> Tuple[List[ECGSample], List[ECGSample]]:
    """Deterministic split mirroring the forgery dataset helper."""
    rng = random.Random(seed)
    if group_key is not None:
        groups: Dict[str, List[ECGSample]] = {}
        for sample in samples:
            key = group_key(sample)
            groups.setdefault(key, []).append(sample)
        keys = list(groups.keys())
        rng.shuffle(keys)
        split_idx = int(len(keys) * (1 - val_fraction))
        train_keys = set(keys[:split_idx])
        train_samples = [s for key in train_keys for s in groups[key]]
        val_samples = [s for key in keys[split_idx:] for s in groups[key]]
    else:
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * (1 - val_fraction))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def _default_collate(
    batch: Sequence[Dict[str, torch.Tensor | str | None]]
) -> Dict[str, torch.Tensor | List[str] | None]:
    images = torch.stack(
        [item["image"] for item in batch if isinstance(item["image"], torch.Tensor)]
    )  # type: ignore[index]
    signal_tensors = [
        item["signal"] for item in batch if isinstance(item.get("signal"), torch.Tensor)
    ]
    signals = torch.stack(signal_tensors) if len(signal_tensors) == len(batch) else None
    ids = [str(item.get("id", "")) for item in batch]
    leads = [item.get("lead") for item in batch]
    return {"image": images, "signal": signals, "id": ids, "lead": leads}


def create_dataloaders(
    samples: Sequence[ECGSample],
    *,
    batch_size: int = 8,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    train_transforms: Optional[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ] = None,
    val_transforms: Optional[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ] = None,
    preload_signals: bool = False,
    group_key: Optional[Callable[[ECGSample], str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders with shared transforms."""
    train_samples, val_samples = train_val_split(
        samples, val_fraction=val_fraction, seed=seed, group_key=group_key
    )
    train_dataset = ECGDigitizationDataset(
        train_samples,
        transforms=train_transforms,
        preload_signals=preload_signals,
    )
    val_dataset = ECGDigitizationDataset(
        val_samples,
        transforms=val_transforms,
        preload_signals=preload_signals,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_default_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_default_collate,
    )
    return train_loader, val_loader


__all__ = [
    "COMPETITION_SLUG",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_IMAGE_DIR",
    "DEFAULT_METADATA_CSV",
    "DEFAULT_SIGNAL_DIR",
    "DEFAULT_STORAGE_ROOT",
    "ECGDigitizationDataset",
    "ECGSample",
    "create_dataloaders",
    "discover_samples",
    "load_samples_from_metadata",
    "train_val_split",
]
