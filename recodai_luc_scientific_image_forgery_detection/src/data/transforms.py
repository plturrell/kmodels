"""Augmentation transforms for forgery detection using Albumentations."""

from __future__ import annotations

from typing import Dict

import albumentations as A
import numpy as np
import torch

from ..config.training import AugmentationConfig


def build_train_transforms(config: AugmentationConfig) -> A.Compose:
    """Build training augmentation pipeline.
    
    Args:
        config: Augmentation configuration
    
    Returns:
        Albumentations composition
    """
    transforms = []
    
    # Geometric transforms
    if config.horizontal_flip > 0:
        transforms.append(A.HorizontalFlip(p=config.horizontal_flip))
    
    if config.vertical_flip > 0:
        transforms.append(A.VerticalFlip(p=config.vertical_flip))
    
    if config.rotation_probability > 0:
        transforms.append(
            A.Rotate(limit=90, p=config.rotation_probability, border_mode=0)
        )
    
    # Color augmentations
    if config.jitter_strength > 0:
        transforms.append(
            A.ColorJitter(
                brightness=config.jitter_strength,
                contrast=config.jitter_strength,
                saturation=config.jitter_strength,
                hue=config.jitter_strength * 0.5,
                p=0.5,
            )
        )
    
    # Forgery-specific augmentations
    transforms.extend([
        # JPEG compression artifacts (common in forgeries)
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        
        # Noise (can hide forgery traces)
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        
        # Blur/sharpen (forgeries often have inconsistent sharpness)
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
        ], p=0.3),
        
        # Elastic deformation (subtle warping)
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        
        # Random brightness/contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        
        # Cutout (helps with robustness)
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.2,
        ),
    ])
    
    return A.Compose(
        transforms,
        additional_targets={"mask": "mask"},
    )


def build_val_transforms() -> A.Compose:
    """Build validation transforms (no augmentation).
    
    Returns:
        Albumentations composition
    """
    return A.Compose(
        [],
        additional_targets={"mask": "mask"},
    )


class AlbumentationsWrapper:
    """Wrapper to apply Albumentations to PyTorch tensors."""
    
    def __init__(self, transform: A.Compose):
        """Initialize wrapper.
        
        Args:
            transform: Albumentations composition
        """
        self.transform = transform
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transforms to batch.
        
        Args:
            batch: Dictionary with 'image', 'mask', 'label' tensors
        
        Returns:
            Transformed batch
        """
        # Convert tensors to numpy (C, H, W) -> (H, W, C)
        image = batch["image"].permute(1, 2, 0).numpy()
        mask = batch["mask"].squeeze(0).numpy()  # (1, H, W) -> (H, W)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        
        # Convert back to tensors
        image = torch.from_numpy(transformed["image"]).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = torch.from_numpy(transformed["mask"]).unsqueeze(0)  # (H, W) -> (1, H, W)
        
        return {
            "image": image,
            "mask": mask,
            "label": batch["label"],
        }


def create_transforms(
    config: AugmentationConfig,
    is_training: bool = True,
) -> AlbumentationsWrapper:
    """Create transform wrapper for dataset.
    
    Args:
        config: Augmentation configuration
        is_training: Whether to use training augmentations
    
    Returns:
        Transform wrapper
    """
    if is_training:
        transform = build_train_transforms(config)
    else:
        transform = build_val_transforms()
    
    return AlbumentationsWrapper(transform)


__all__ = [
    "build_train_transforms",
    "build_val_transforms",
    "AlbumentationsWrapper",
    "create_transforms",
]
