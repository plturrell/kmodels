"""Test-Time Augmentation (TTA) for biomass prediction."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TTAWrapper:
    """Test-Time Augmentation wrapper for biomass models."""
    
    def __init__(
        self,
        model: nn.Module,
        transforms: List[Callable] = None,
        merge_mode: str = "mean",
    ):
        """Initialize TTA wrapper.
        
        Args:
            model: Trained model
            transforms: List of augmentation functions (if None, use default)
            merge_mode: How to merge predictions ('mean', 'median', 'gmean')
        """
        self.model = model
        self.merge_mode = merge_mode
        
        if transforms is None:
            self.transforms = self._default_transforms()
        else:
            self.transforms = transforms
    
    def _default_transforms(self) -> List[Callable]:
        """Default TTA transforms for biomass images."""
        return [
            lambda x: x,  # Original
            lambda x: torch.flip(x, dims=[3]),  # Horizontal flip
            lambda x: torch.flip(x, dims=[2]),  # Vertical flip
            lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3]),  # Both flips
        ]
    
    def __call__(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        """Apply TTA and merge predictions.
        
        Args:
            x: Input image tensor (batch, channels, height, width)
            metadata: Optional metadata tensor
        
        Returns:
            Merged predictions
        """
        predictions = []
        
        for transform in self.transforms:
            # Apply transform
            x_aug = transform(x)
            
            # Forward pass
            if metadata is not None:
                pred = self.model(x_aug, metadata)
            else:
                pred = self.model(x_aug)
            
            predictions.append(pred)
        
        # Merge predictions
        predictions = torch.stack(predictions)
        
        if self.merge_mode == "mean":
            return predictions.mean(dim=0)
        elif self.merge_mode == "median":
            return predictions.median(dim=0)[0]
        elif self.merge_mode == "gmean":
            # Geometric mean (for positive values)
            return torch.exp(torch.log(predictions + 1e-8).mean(dim=0))
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")


class MultiCropTTA:
    """Multi-crop TTA for biomass prediction."""
    
    def __init__(
        self,
        model: nn.Module,
        crop_size: int = 352,
        n_crops: int = 5,
        merge_mode: str = "mean",
    ):
        """Initialize multi-crop TTA.
        
        Args:
            model: Trained model
            crop_size: Size of crops
            n_crops: Number of random crops
            merge_mode: How to merge predictions
        """
        self.model = model
        self.crop_size = crop_size
        self.n_crops = n_crops
        self.merge_mode = merge_mode
    
    def __call__(self, image: Image.Image, metadata: torch.Tensor = None) -> torch.Tensor:
        """Apply multi-crop TTA.
        
        Args:
            image: PIL Image
            metadata: Optional metadata tensor
        
        Returns:
            Merged predictions
        """
        predictions = []
        
        # Center crop
        center_crop = A.CenterCrop(self.crop_size, self.crop_size)
        crop_transform = A.Compose([
            center_crop,
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Center crop prediction
        center_img = crop_transform(image=np.array(image))["image"]
        center_img = center_img.unsqueeze(0)
        
        if metadata is not None:
            pred = self.model(center_img, metadata)
        else:
            pred = self.model(center_img)
        predictions.append(pred)
        
        # Random crops
        random_crop = A.RandomCrop(self.crop_size, self.crop_size)
        for _ in range(self.n_crops - 1):
            crop_transform = A.Compose([
                random_crop,
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            
            crop_img = crop_transform(image=np.array(image))["image"]
            crop_img = crop_img.unsqueeze(0)
            
            if metadata is not None:
                pred = self.model(crop_img, metadata)
            else:
                pred = self.model(crop_img)
            predictions.append(pred)
        
        # Merge predictions
        predictions = torch.stack(predictions)
        
        if self.merge_mode == "mean":
            return predictions.mean(dim=0)
        elif self.merge_mode == "median":
            return predictions.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")


__all__ = [
    "TTAWrapper",
    "MultiCropTTA",
]

