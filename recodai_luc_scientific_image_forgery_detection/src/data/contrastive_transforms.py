"""Contrastive-learning augmentations with forgery-aware perturbations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import albumentations as A
import numpy as np


@dataclass
class ContrastiveAugConfig:
    image_size: Tuple[int, int] = (256, 256)
    strong_color_jitter: float = 0.4
    gaussian_blur_prob: float = 0.5
    solarize_prob: float = 0.2
    synthetic_forgery_prob: float = 0.4
    copy_paste_patch_ratio: Tuple[float, float] = (0.05, 0.2)


class SyntheticForgeryAug(A.ImageOnlyTransform):
    """Create synthetic splicing by copy-pasting random patches."""

    def __init__(self, patch_ratio: Tuple[float, float], p: float = 0.5):
        super().__init__(p=p)
        self.patch_ratio = patch_ratio

    def apply(self, img: np.ndarray, **params):  # type: ignore[override]
        h, w, _ = img.shape
        rng = np.random.default_rng()
        patch_h = int(h * rng.uniform(*self.patch_ratio))
        patch_w = int(w * rng.uniform(*self.patch_ratio))
        if patch_h <= 0 or patch_w <= 0:
            return img

        y0 = rng.integers(0, max(1, h - patch_h))
        x0 = rng.integers(0, max(1, w - patch_w))
        y1 = rng.integers(0, max(1, h - patch_h))
        x1 = rng.integers(0, max(1, w - patch_w))

        patch = img[y0 : y0 + patch_h, x0 : x0 + patch_w].copy()

        alpha = rng.uniform(0.6, 1.0)
        blend = alpha * patch + (1 - alpha) * img[y1 : y1 + patch_h, x1 : x1 + patch_w]
        img[y1 : y1 + patch_h, x1 : x1 + patch_w] = np.clip(blend, 0, 255)
        return img


class ContrastiveViewGenerator:
    """Callable wrapper that produces two augmented views per image."""

    def __init__(self, config: ContrastiveAugConfig) -> None:
        target_h, target_w = config.image_size
        self.resize = A.Resize(height=target_h, width=target_w)
        self.base_transforms = A.Compose(
            [
                A.SmallestMaxSize(max(target_h, target_w), p=1.0),
                A.RandomCrop(height=target_h, width=target_w, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ColorJitter(
                    brightness=config.strong_color_jitter,
                    contrast=config.strong_color_jitter,
                    saturation=config.strong_color_jitter,
                    hue=config.strong_color_jitter / 2,
                    p=0.8,
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussianBlur(blur_limit=5, p=config.gaussian_blur_prob),
                A.Solarize(p=config.solarize_prob),
                SyntheticForgeryAug(
                    patch_ratio=config.copy_paste_patch_ratio,
                    p=config.synthetic_forgery_prob,
                ),
                A.ISONoise(p=0.3),
            ],
            additional_targets={"view2": "image"},
        )

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:  # noqa: D401
        augmented = self.base_transforms(image=image, view2=image)
        img1 = self.resize(image=augmented["image"])["image"]
        img2 = self.resize(image=augmented["view2"])["image"]
        return {"view1": img1, "view2": img2}


def build_contrastive_augs(config: ContrastiveAugConfig) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    """Return a callable producing two augmented views for contrastive learning."""

    return ContrastiveViewGenerator(config)


__all__ = ["ContrastiveAugConfig", "SyntheticForgeryAug", "build_contrastive_augs"]


