"""Dual-stream forgery detection model combining spatial and frequency cues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .pretrained import PretrainedForgeryModel, PretrainedModelConfig, build_pretrained_model


@dataclass
class DualStreamConfig:
    """Configuration governing the dual-stream model composition."""

    spatial: PretrainedModelConfig = PretrainedModelConfig()
    num_classes: int = 2
    frequency_channels: int = 3
    fusion_weight: float = 0.5  # Weight for spatial logits in the final fusion


class FrequencyBranch(nn.Module):
    """Simple convolutional encoder operating on Fourier magnitude maps."""

    def __init__(self, in_channels: int, out_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(out_dim, out_dim)

    def forward(self, frequency_map: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(frequency_map)
        return self.head(embedding)


class DualStreamForgeryModel(nn.Module):
    """Combine a spatial segmentation model with a frequency classifier."""

    def __init__(
        self,
        config: DualStreamConfig,
        spatial_model: Optional[PretrainedForgeryModel] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_model = spatial_model or PretrainedForgeryModel(config.spatial)
        self.frequency_branch = FrequencyBranch(
            in_channels=config.frequency_channels,
            out_dim=256,
        )
        self.frequency_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, config.num_classes),
        )
        self.fusion_weight = config.fusion_weight

    def _build_frequency_map(self, image: torch.Tensor) -> torch.Tensor:
        """Compute log-spectrum magnitude maps from the RGB image."""

        # image: (B, C, H, W)
        fft = torch.fft.fft2(image.float())
        magnitude = torch.abs(fft)
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
        magnitude = torch.log1p(magnitude)
        # Min-max normalise per sample for stability
        b = magnitude.shape[0]
        magnitude = magnitude.view(b, -1)
        min_vals = magnitude.min(dim=1)[0].unsqueeze(-1)
        max_vals = magnitude.max(dim=1)[0].unsqueeze(-1)
        norm = (magnitude - min_vals) / (max_vals - min_vals + 1e-6)
        norm = norm.view_as(image)
        return norm

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        class_logits_spatial, mask_logits = self.spatial_model(image)
        freq_map = self._build_frequency_map(image)
        freq_features = self.frequency_branch(freq_map)
        class_logits_freq = self.frequency_classifier(freq_features)

        fused_logits = (
            self.fusion_weight * class_logits_spatial
            + (1.0 - self.fusion_weight) * class_logits_freq
        )
        return fused_logits, mask_logits


def build_dual_stream_model(config: Optional[DualStreamConfig] = None) -> DualStreamForgeryModel:
    return DualStreamForgeryModel(config or DualStreamConfig())


__all__ = [
    "DualStreamConfig",
    "DualStreamForgeryModel",
    "build_dual_stream_model",
]


