"""PyTorch dataset helpers for the forgery detection competition."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "recodai-luc-scientific-image-forgery-detection"
)
TRAIN_IMAGE_DIR = DEFAULT_DATA_ROOT / "train_images"
MASK_DIR = DEFAULT_DATA_ROOT / "train_masks"


@dataclasses.dataclass(frozen=True)
class Sample:
    """Container describing a single image/mask pair."""

    image_path: Path
    mask_path: Optional[Path]
    label: int  # 0 = authentic, 1 = forged


def _discover_samples(data_root: Path) -> List[Sample]:
    """Return the list of available samples on disk."""
    train_dir = data_root / "train_images"
    mask_dir = data_root / "train_masks"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train_images directory: {train_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing train_masks directory: {mask_dir}")

    label_map = {"authentic": 0, "forged": 1}
    samples: List[Sample] = []
    for label_name, label_idx in label_map.items():
        class_dir = train_dir / label_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected directory not found: {class_dir}")
        for image_path in sorted(class_dir.glob("*.png")):
            mask_path = mask_dir / f"{image_path.stem}.npy"
            if not mask_path.exists():
                mask_path = None
            samples.append(
                Sample(image_path=image_path, mask_path=mask_path, label=label_idx)
            )
    return samples


class ForgeryDataset(Dataset):
    """Return image, mask, and label tensors for training."""

    def __init__(
        self,
        samples: Sequence[Sample],
        transforms: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        ensure_mask_size: Optional[Tuple[int, int]] = (512, 648),
    ) -> None:
        self.samples = list(samples)
        self.transforms = transforms
        self.ensure_mask_size = ensure_mask_size

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            array = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)  # C, H, W
        return tensor

    def _load_mask(self, sample: Sample, spatial_shape: Tuple[int, int]) -> torch.Tensor:
        if sample.mask_path is None:
            return torch.zeros(1, *spatial_shape, dtype=torch.float32)
        mask = np.load(sample.mask_path)
        tensor = torch.from_numpy(mask.astype(np.float32))

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] == 1:
                pass
            elif tensor.shape[-1] == 1:
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor[:1]
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        else:
            tensor = tensor.reshape(1, *tensor.shape[-2:])

        if tensor.shape[1:] != spatial_shape:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=spatial_shape,
                mode="nearest",
            ).squeeze(0)

        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = self._load_image(sample.image_path)
        spatial_shape = image.shape[1:]
        if self.ensure_mask_size:
            spatial_shape = self.ensure_mask_size
            if image.shape[1:] != spatial_shape:
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=spatial_shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        mask = self._load_mask(sample, spatial_shape)
        label = torch.tensor(sample.label, dtype=torch.long)

        batch = {"image": image, "mask": mask, "label": label}
        if self.transforms is not None:
            batch = self.transforms(batch)
            image = batch["image"]
            mask = batch["mask"]
            if self.ensure_mask_size:
                target_shape = self.ensure_mask_size
                if image.shape[1:] != target_shape:
                    image = F.interpolate(
                        image.unsqueeze(0),
                        size=target_shape,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                if mask.shape[1:] != target_shape:
                    mask = F.interpolate(
                        mask.unsqueeze(0),
                        size=target_shape,
                        mode="nearest",
                    ).squeeze(0)
            batch = {"image": image, "mask": mask, "label": batch.get("label", label)}
        return batch


def train_val_split(
    samples: Sequence[Sample],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Deterministic train/validation split stratified by label."""
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)
        split_idx = int(len(group) * (1 - val_fraction))
        train_samples.extend(group[:split_idx])
        val_samples.extend(group[split_idx:])
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def create_dataloaders(
    data_root: Path = DEFAULT_DATA_ROOT,
    batch_size: int = 4,
    val_fraction: float = 0.2,
    num_workers: int = 0,
    train_transforms: Optional[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ] = None,
    val_transforms: Optional[
        Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
    ] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/validation dataloaders ready for training."""
    samples = _discover_samples(data_root)
    train_samples, val_samples = train_val_split(
        samples, val_fraction=val_fraction, seed=seed
    )

    train_dataset = ForgeryDataset(train_samples, transforms=train_transforms)
    val_dataset = ForgeryDataset(val_samples, transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def load_samples(data_root: Path = DEFAULT_DATA_ROOT) -> List[Sample]:
    """Public helper to list available samples on disk."""
    return _discover_samples(data_root)
