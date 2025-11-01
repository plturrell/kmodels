"""PyTorch implementation of the TSMixer model.

Adapted from https://github.com/ditschuk/pytorch-tsmixer
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --- Utility Functions ---

def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)


feature_to_time = time_to_feature


# --- Layers ---

class TimeBatchNorm2d(nn.BatchNorm1d):
    """A batch normalization layer that normalizes over the last two dimensions."""

    def __init__(self, normalized_shape: tuple[int, int]):
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, but got {x.ndim}D tensor instead.")

        x = x.reshape(x.shape[0], -1, 1)
        x = super().forward(x)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)
        return x


class FeatureMixing(nn.Module):
    """A module for feature mixing."""

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        super().__init__()

        self.norm_before = (
            norm_type((sequence_length, input_channels))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((sequence_length, output_channels))
            if not normalize_before
            else nn.Identity()
        )

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)

        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.projection(x)
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x_proj + x
        return self.norm_after(x)


class TimeMixing(nn.Module):
    """Applies a transformation over the time dimension of a sequence."""

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_temp = feature_to_time(x)
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)
        return self.norm(x + x_res)


class MixerLayer(nn.Module):
    """A residual block that combines time and feature mixing."""

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = FeatureMixing(
            sequence_length,
            input_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
            normalize_before=normalize_before,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x


# --- TSMixer Model ---

class TSMixer(nn.Module):
    """TSMixer model for time series forecasting."""

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        output_channels: int = 1,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 64,
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        super().__init__()

        activation_fn = getattr(F, activation_fn)
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type_callable = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_channels,
            input_channels,  # Output channels of mixer layers should be the same as input
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type_callable,
        )

        self.temporal_projection = nn.Linear(sequence_length, prediction_length)
        self.output_projection = nn.Linear(input_channels, output_channels)

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                MixerLayer(input_channels=input_channels, output_channels=output_channels, **kwargs)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        x = self.mixer_layers(x_hist)
        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)
        x = self.output_projection(x)
        return x
