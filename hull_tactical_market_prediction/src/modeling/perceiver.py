"""Perceiver-style regressor for tabular features."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False)
        self.cross_ff = FeedForward(dim, mult=2, dropout=dropout)

        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False)
        self.self_ff = FeedForward(dim, mult=2, dropout=dropout)

        self.layernorm_latent = nn.LayerNorm(dim)
        self.layernorm_cross = nn.LayerNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        latent_norm = self.layernorm_latent(latents.transpose(0, 1)).transpose(0, 1)
        cross_output, _ = self.cross_attn(latent_norm, inputs, inputs)
        latents = latents + cross_output
        latents = latents + self.cross_ff(latents.transpose(0, 1)).transpose(0, 1)

        latent_norm = self.layernorm_cross(latents.transpose(0, 1)).transpose(0, 1)
        self_output, _ = self.self_attn(latent_norm, latent_norm, latent_norm)
        latents = latents + self_output
        latents = latents + self.self_ff(latents.transpose(0, 1)).transpose(0, 1)
        return latents


class PerceiverRegressor(nn.Module):
    """Perceiver encoder with mean pooled latents for regression."""

    def __init__(
        self,
        input_dim: int,
        *,
        num_latents: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_mult: int,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim))

        self.blocks = nn.ModuleList(
            [
                PerceiverBlock(latent_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(latent_dim)
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * ff_mult, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.size(0)
        inputs = self.input_proj(features).unsqueeze(0)  # (1, batch, latent_dim)

        latents = self.latents.unsqueeze(1).expand(-1, batch_size, -1)  # (num_latents, batch, latent_dim)
        for block in self.blocks:
            latents = block(latents, inputs)

        latents = self.final_ln(latents.transpose(0, 1))  # (batch, num_latents, latent_dim)
        pooled = latents.mean(dim=1)
        output = self.head(pooled).squeeze(-1)
        return output
