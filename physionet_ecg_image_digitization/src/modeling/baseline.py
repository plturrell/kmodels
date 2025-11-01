"""Baseline neural architectures for ECG waveform regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def _build_backbone(
    name: str,
    *,
    pretrained: bool = True,
    in_channels: int = 3,
) -> tuple[nn.Module, int]:
    """Return a torchvision backbone and embedding dimension."""
    name = name.lower()
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision backbone: {name}")
    builder = getattr(models, name)
    kwargs = {}
    weights_attr = f"{name}_weights"
    if pretrained:
        try:
            weights_enum = getattr(models, weights_attr)
        except AttributeError:
            weights_enum = None
        if weights_enum is not None:
            default_weights = getattr(weights_enum, "DEFAULT", None)
            if default_weights is None:
                # Fallback to the first available enum entry if DEFAULT is missing.
                default_weights = list(weights_enum)[0]
            kwargs["weights"] = default_weights
        else:
            kwargs["pretrained"] = True  # legacy API (torchvision < 0.13)
    else:
        if hasattr(models, weights_attr):
            kwargs["weights"] = None
        else:
            kwargs["pretrained"] = False
    backbone = builder(**kwargs)

    if hasattr(backbone, "conv1") and in_channels != 3:
        conv1 = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=False,
        )

    if hasattr(backbone, "fc"):
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        if isinstance(backbone.classifier, nn.Sequential):
            *_, last = backbone.classifier
            in_features = last.in_features  # type: ignore[attr-defined]
            backbone.classifier = nn.Identity()
        else:
            in_features = backbone.classifier.in_features  # type: ignore[attr-defined]
            backbone.classifier = nn.Identity()
    else:
        raise RuntimeError(
            f"Backbone {name} does not expose a standard classification head."
        )
    return backbone, in_features


class ECGResNetRegressor(nn.Module):
    """ResNet-style encoder paired with a lightweight regression head."""

    def __init__(
        self,
        *,
        signal_length: int,
        signal_channels: int = 1,
        backbone: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.signal_length = signal_length
        self.signal_channels = signal_channels
        self.encoder, embedding_dim = _build_backbone(
            backbone, pretrained=pretrained, in_channels=in_channels
        )
        output_dim = signal_length * signal_channels
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        waveform = self.head(features)
        batch = waveform.shape[0]
        if self.signal_channels > 1:
            waveform = waveform.view(batch, self.signal_channels, self.signal_length)
        else:
            waveform = waveform.view(batch, self.signal_length)
        return waveform


class Conv1DRefiner(nn.Module):
    """Optional module to refine coarse predictions with 1D convolutions."""

    def __init__(
        self,
        signal_length: int,
        hidden_channels: int = 128,
        layers: int = 3,
        channels: int = 1,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = 1
        for _ in range(layers):
            blocks.append(nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2))
            blocks.append(nn.ReLU(inplace=True))
            in_channels = hidden_channels
        blocks.append(nn.Conv1d(hidden_channels, 1, kernel_size=1))
        self.net = nn.Sequential(*blocks)
        self.signal_length = signal_length
        self.expected_channels = channels

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            batch, length = waveform.shape
            channels = 1
            waveform_reshaped = waveform.unsqueeze(1)
        elif waveform.ndim == 3:
            batch, channels, length = waveform.shape
            waveform_reshaped = waveform
        else:  # pragma: no cover - defensive guard
            raise ValueError(
                f"Expected waveform tensor of rank 2 or 3, received shape {tuple(waveform.shape)}"
            )

        if self.expected_channels is not None and channels not in (self.expected_channels, 1):
            raise RuntimeError(
                "Refiner received a waveform with an unexpected number of channels: "
                f"expected {self.expected_channels}, got {channels}."
            )

        if length != self.signal_length:
            waveform_reshaped = nn.functional.interpolate(
                waveform_reshaped,
                size=self.signal_length,
                mode="linear",
                align_corners=False,
            )

        refined = self.net(
            waveform_reshaped.view(batch * channels, 1, self.signal_length)
        ).view(batch, channels, self.signal_length)

        if channels == 1:
            refined = refined.squeeze(1)
        return refined


@dataclass
class BaselineModelConfig:
    signal_length: int = 1000
    signal_channels: int = 1
    backbone: str = "resnet18"
    pretrained: bool = True
    hidden_dim: int = 512
    dropout: float = 0.1
    use_refiner: bool = False
    refiner_channels: int = 128
    refiner_layers: int = 3


def build_baseline_model(cfg: BaselineModelConfig) -> nn.Module:
    """Factory mirroring other projects for quick instantiation."""
    model = ECGResNetRegressor(
        signal_length=cfg.signal_length,
        signal_channels=cfg.signal_channels,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )
    if cfg.use_refiner:
        refiner = Conv1DRefiner(
            signal_length=cfg.signal_length,
            hidden_channels=cfg.refiner_channels,
            layers=cfg.refiner_layers,
            channels=cfg.signal_channels,
        )

        class WithRefiner(nn.Module):
            def __init__(self, base: nn.Module, refinement: nn.Module) -> None:
                super().__init__()
                self.base = base
                self.refinement = refinement

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                coarse = self.base(images)
                return self.refinement(coarse)

        return WithRefiner(model, refiner)
    return model


__all__ = [
    "BaselineModelConfig",
    "Conv1DRefiner",
    "ECGResNetRegressor",
    "build_baseline_model",
]
