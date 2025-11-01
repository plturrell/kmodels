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
    """Simple feed-forward network with optional batch norm and dropout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        dropout: float,
        activation: str = "gelu",
        *,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
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
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
