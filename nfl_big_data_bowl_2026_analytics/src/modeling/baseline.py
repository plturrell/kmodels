"""Baseline and advanced regressors for the analytics task."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from ..config.experiment import ModelConfig

try:
    from .graph_model import GATPlayerTracker
except ImportError:  # pragma: no cover - optional dependency
    GATPlayerTracker = None


class MLPRegressor(nn.Module):
    """Simple multilayer perceptron for trajectory regression."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.network(features)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x + self.block(x)


class ResidualMLPRegressor(nn.Module):
    """Residual MLP with DeepMind-style pre-activation blocks."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)
        return self.output_head(x)


class PerceiverRegressor(nn.Module):
    """Lightweight Perceiver-style latent attention regressor."""

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.cross_ff = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(latent_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(latent_dim, latent_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # noqa: D401
        batch_size = features.size(0)
        inputs = self.input_proj(features).unsqueeze(0)  # (1, batch, latent_dim)
        latents = self.latents.unsqueeze(1).expand(-1, batch_size, -1)

        for attn, ff in zip(self.cross_attn, self.cross_ff):
            attn_latents, _ = attn(latents, inputs, inputs)
            latents = latents + attn_latents
            latents = latents + ff(latents.transpose(0, 1)).transpose(0, 1)

        pooled = latents.mean(dim=0)
        return self.decoder(pooled)


def build_baseline_model(
    input_dim: int,
    output_dim: int,
    model_cfg: ModelConfig,
) -> nn.Module:
    """Construct a regressor based on the requested architecture."""
    architecture = model_cfg.architecture.lower()

    if architecture == "mlp":
        hidden_dims: Iterable[int] = model_cfg.hidden_dims or [256, 128]
        return MLPRegressor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=model_cfg.dropout,
            output_dim=output_dim,
        )

    if architecture == "residual_mlp":
        hidden_dim = model_cfg.hidden_dims[0] if model_cfg.hidden_dims else 512
        return ResidualMLPRegressor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=max(int(model_cfg.num_layers), 1),
            dropout=model_cfg.dropout,
        )

    if architecture == "gat":
        if GATPlayerTracker is None:
            raise ImportError(
                "torch_geometric is not installed; install it to use the GAT architecture."
            )
        hidden_dim = model_cfg.hidden_dims[0] if model_cfg.hidden_dims else 256
        return GATPlayerTracker(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_heads=max(int(model_cfg.n_heads), 1),
            dropout=model_cfg.dropout,
        )

    if architecture == "perceiver":
        return PerceiverRegressor(
            input_dim=input_dim,
            latent_dim=model_cfg.latent_dim,
            num_latents=model_cfg.num_latents,
            num_layers=max(int(model_cfg.num_layers), 1),
            num_heads=max(int(model_cfg.n_heads), 1),
            dropout=model_cfg.dropout,
            output_dim=output_dim,
        )

    raise ValueError(f"Unsupported architecture: {model_cfg.architecture}")
