"""Lightweight augmentation utilities shared across training scripts."""

from __future__ import annotations

import random
from typing import Callable, Dict

import torch
from torchvision.transforms import functional as F

IMAGE_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGE_DEFAULT_STD = (0.5, 0.5, 0.5)


def _ensure_tensor(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected image to be a torch.Tensor.")
    if image.dtype != torch.float32:
        image = image.float()
    if image.max().item() > 1.0:
        image = image / 255.0
    return image


def _apply_affine(
    image: torch.Tensor,
    *,
    degrees: float,
    translate: float,
    scale_jitter: float,
) -> torch.Tensor:
    angle = random.uniform(-degrees, degrees)
    max_dx = translate * image.shape[2]
    max_dy = translate * image.shape[1]
    translations = (
        random.uniform(-max_dx, max_dx),
        random.uniform(-max_dy, max_dy),
    )
    scale = 1.0 + random.uniform(-scale_jitter, scale_jitter)
    return F.affine(
        image,
        angle=angle,
        translate=translations,
        scale=scale,
        shear=[0.0, 0.0],
    )


def build_train_transform(
    image_size: int = 512,
    *,
    mean: tuple[float, float, float] = IMAGE_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGE_DEFAULT_STD,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.0,
    brightness: float = 0.1,
    contrast: float = 0.1,
    affine_degrees: float = 3.0,
    affine_translate: float = 0.03,
    affine_scale: float = 0.05,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Return a callable that augments image tensors in-place."""

    def transform(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = _ensure_tensor(batch["image"])
        image = F.resize(image, (image_size, image_size), antialias=True)
        if random.random() < hflip_prob:
            image = torch.flip(image, dims=[2])
        if random.random() < vflip_prob:
            image = torch.flip(image, dims=[1])
        if affine_degrees or affine_translate or affine_scale:
            image = _apply_affine(
                image,
                degrees=affine_degrees,
                translate=affine_translate,
                scale_jitter=affine_scale,
            )
        if brightness > 0:
            factor = 1.0 + random.uniform(-brightness, brightness)
            image = F.adjust_brightness(image, factor)
        if contrast > 0:
            factor = 1.0 + random.uniform(-contrast, contrast)
            image = F.adjust_contrast(image, factor)
        image = torch.clamp(image, 0.0, 1.0)
        image = F.normalize(image, mean=mean, std=std)
        batch["image"] = image
        if "signal" in batch and isinstance(batch["signal"], torch.Tensor):
            batch["signal"] = batch["signal"].float()
        return batch

    return transform


def build_eval_transform(
    image_size: int = 512,
    *,
    mean: tuple[float, float, float] = IMAGE_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGE_DEFAULT_STD,
) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Return deterministic transforms shared by validation and test."""

    def transform(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = _ensure_tensor(batch["image"])
        image = F.resize(image, (image_size, image_size), antialias=True)
        image = F.normalize(image, mean=mean, std=std)
        batch["image"] = image
        if "signal" in batch and isinstance(batch["signal"], torch.Tensor):
            batch["signal"] = batch["signal"].float()
        return batch

    return transform


def denormalise_image(
    image: torch.Tensor,
    *,
    mean: tuple[float, float, float] = IMAGE_DEFAULT_MEAN,
    std: tuple[float, float, float] = IMAGE_DEFAULT_STD,
) -> torch.Tensor:
    """Undo normalisation for visualisation."""
    for channel, (m, s) in enumerate(zip(mean, std)):
        image[channel] = image[channel] * s + m
    return torch.clamp(image, 0.0, 1.0)


__all__ = [
    "IMAGE_DEFAULT_MEAN",
    "IMAGE_DEFAULT_STD",
    "build_eval_transform",
    "build_train_transform",
    "denormalise_image",
]
