"""Dataset utilities for self-supervised contrastive pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .contrastive_transforms import ContrastiveAugConfig, build_contrastive_augs


@dataclass(frozen=True)
class ContrastiveSample:
    image_path: Path


def _discover_image_paths(data_root: Path) -> List[ContrastiveSample]:
    samples: List[ContrastiveSample] = []
    for label_dir in ["authentic", "forged"]:
        class_dir = data_root / "train_images" / label_dir
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.glob("*.png")):
            samples.append(ContrastiveSample(image_path=image_path))
    return samples


class ContrastivePairDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        transform: Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]] = None,
    ) -> None:
        self.data_root = data_root
        self.samples = _discover_image_paths(data_root)
        if not self.samples:
            raise FileNotFoundError(f"No training images found under {data_root}")
        self.transform = transform or build_contrastive_augs(ContrastiveAugConfig())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.image_path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
        augmented = self.transform(arr)
        view1 = torch.from_numpy(augmented["view1"]).permute(2, 0, 1).float() / 255.0
        view2 = torch.from_numpy(augmented["view2"]).permute(2, 0, 1).float() / 255.0
        return {"view1": view1, "view2": view2}


__all__ = ["ContrastivePairDataset", "ContrastiveAugConfig", "build_contrastive_augs"]


