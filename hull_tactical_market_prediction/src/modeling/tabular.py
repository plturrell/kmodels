"""Neural architectures for Hull Tactical Market Prediction."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _resolve_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class TabularRegressor(nn.Module):
    """Simple feed-forward network with optional batch norm and dropout.
    
    Supports heteroscedastic regression by outputting both mean and variance predictions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        dropout: float,
        activation: str = "gelu",
        *,
        batch_norm: bool = True,
        output_variance: bool = False,
    ) -> None:
        super().__init__()
        self.output_variance = output_variance
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(_resolve_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(current_dim, 1)
        if output_variance:
            self.variance_head = nn.Sequential(
                nn.Linear(current_dim, current_dim // 2),
                _resolve_activation(activation),
                nn.Linear(current_dim // 2, 1),
                nn.Softplus(),  # Ensure variance is positive
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        mean = self.mean_head(hidden).squeeze(-1)
        
        if self.output_variance:
            variance = self.variance_head(hidden).squeeze(-1)
            return mean, variance
        return mean
