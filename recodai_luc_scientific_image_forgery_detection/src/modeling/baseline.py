"""Joint segmentation + classification baseline for forgery detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VisionBaselineConfig:
    in_channels: int = 3
    base_channels: int = 32
    encoder_depth: int = 4
    segmentation_channels: int = 1
    num_classes: int = 2
    classification_hidden: int = 256
    dropout: float = 0.3
    bilinear: bool = True


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_bn: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.extend(
            [
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn),
            ]
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv_in = in_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            conv_in = in_ch
        self.conv = _ConvBlock(conv_in, out_ch)
        self.bilinear = bilinear

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.bilinear:
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x], dim=1)
        else:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ForgeryBaseline(nn.Module):
    """Simple U-Net style encoder-decoder with classification head."""

    def __init__(self, config: VisionBaselineConfig) -> None:
        super().__init__()
        self.config = config

        channels = [config.base_channels * (2 ** i) for i in range(config.encoder_depth)]
        self.inc = _ConvBlock(config.in_channels, channels[0])
        downs = []
        for idx in range(1, config.encoder_depth):
            downs.append(_Down(channels[idx - 1], channels[idx]))
        self.downs = nn.ModuleList(downs)

        ups = []
        for idx in range(config.encoder_depth - 1, 0, -1):
            in_ch = channels[idx] + channels[idx - 1]
            ups.append(_Up(in_ch, channels[idx - 1], bilinear=config.bilinear))
        self.ups = nn.ModuleList(ups)

        self.mask_head = nn.Conv2d(channels[0], config.segmentation_channels, kernel_size=1)

        clf_in = channels[-1]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(clf_in, config.classification_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.classification_hidden, config.num_classes),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inc(image)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        class_logits = self.classifier(x)

        for idx, up in enumerate(self.ups, start=1):
            skip = skips[-(idx + 1)]
            x = up(x, skip)

        mask_logits = self.mask_head(x)
        return class_logits, mask_logits


def build_vision_baseline(config: VisionBaselineConfig | None = None) -> ForgeryBaseline:
    return ForgeryBaseline(config or VisionBaselineConfig())


__all__ = ["VisionBaselineConfig", "ForgeryBaseline", "build_vision_baseline"]
