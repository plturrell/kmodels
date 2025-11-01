"""Pre-trained encoder-decoder models using segmentation_models_pytorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import segmentation_models_pytorch as smp
import torch
from torch import nn


@dataclass
class PretrainedModelConfig:
    """Configuration for pre-trained models."""
    
    # Architecture
    architecture: str = "Unet"  # Unet, UnetPlusPlus, FPN, DeepLabV3Plus, etc.
    encoder_name: str = "resnet34"  # resnet34, efficientnet-b0, etc.
    encoder_weights: str = "imagenet"  # imagenet, ssl, swsl, or None
    
    # Model parameters
    in_channels: int = 3
    num_classes: int = 2  # Classification classes
    segmentation_classes: int = 1  # Segmentation classes (binary mask)
    
    # Training
    activation: Optional[str] = None  # None for logits
    dropout: float = 0.3


class PretrainedForgeryModel(nn.Module):
    """Forgery detection model with pre-trained encoder."""
    
    def __init__(self, config: PretrainedModelConfig):
        """Initialize model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Get segmentation model class
        model_class = getattr(smp, config.architecture)
        
        # Create segmentation model
        self.segmentation_model = model_class(
            encoder_name=config.encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=config.in_channels,
            classes=config.segmentation_classes,
            activation=config.activation,
        )
        
        # Get encoder output channels
        encoder = self.segmentation_model.encoder
        if hasattr(encoder, 'out_channels'):
            encoder_channels = encoder.out_channels[-1]
        else:
            # Fallback: run a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, config.in_channels, 224, 224)
                encoder_features = encoder(dummy_input)
                encoder_channels = encoder_features[-1].shape[1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(encoder_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_classes),
        )
    
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            image: Input image tensor (B, C, H, W)
        
        Returns:
            Tuple of (class_logits, mask_logits)
        """
        # Get encoder features
        features = self.segmentation_model.encoder(image)
        
        # Classification from deepest features
        class_logits = self.classifier(features[-1])
        
        # Segmentation from decoder
        mask_logits = self.segmentation_model.decoder(*features)
        mask_logits = self.segmentation_model.segmentation_head(mask_logits)
        
        return class_logits, mask_logits


def build_pretrained_model(
    architecture: str = "Unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    **kwargs,
) -> PretrainedForgeryModel:
    """Build a pre-trained forgery detection model.
    
    Args:
        architecture: Model architecture (Unet, UnetPlusPlus, FPN, etc.)
        encoder_name: Encoder backbone (resnet34, efficientnet-b0, etc.)
        encoder_weights: Pre-trained weights (imagenet, ssl, swsl, None)
        **kwargs: Additional config parameters
    
    Returns:
        PretrainedForgeryModel instance
    
    Examples:
        >>> # ResNet34 U-Net with ImageNet weights
        >>> model = build_pretrained_model("Unet", "resnet34", "imagenet")
        
        >>> # EfficientNet-B0 U-Net++ with ImageNet weights
        >>> model = build_pretrained_model("UnetPlusPlus", "efficientnet-b0", "imagenet")
        
        >>> # ResNet50 FPN with ImageNet weights
        >>> model = build_pretrained_model("FPN", "resnet50", "imagenet")
    """
    config = PretrainedModelConfig(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        **kwargs,
    )
    return PretrainedForgeryModel(config)


# Popular encoder configurations
ENCODER_CONFIGS = {
    "resnet34": {"encoder_name": "resnet34", "encoder_weights": "imagenet"},
    "resnet50": {"encoder_name": "resnet50", "encoder_weights": "imagenet"},
    "efficientnet-b0": {"encoder_name": "efficientnet-b0", "encoder_weights": "imagenet"},
    "efficientnet-b3": {"encoder_name": "efficientnet-b3", "encoder_weights": "imagenet"},
    "resnext50": {"encoder_name": "resnext50_32x4d", "encoder_weights": "imagenet"},
    "densenet121": {"encoder_name": "densenet121", "encoder_weights": "imagenet"},
}


__all__ = [
    "PretrainedModelConfig",
    "PretrainedForgeryModel",
    "build_pretrained_model",
    "ENCODER_CONFIGS",
]

