"""Diffusion utilities for multi-modal trajectory prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


@dataclass
class DiffusionConfig:
    timesteps: int = 100
    noise_schedule: str = "cosine"
    prediction_type: str = "velocity"


class TrajectoryDiffusion(nn.Module):
    def __init__(self, model: nn.Module, config: DiffusionConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        if config.noise_schedule == "cosine":
            betas = cosine_beta_schedule(config.timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, config.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = torch.randn_like(x_start) if noise is None else noise
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def forward(self, node_feats: torch.Tensor, positions: torch.Tensor, *, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch = target.size(0)
        device = target.device
        t = torch.randint(0, self.config.timesteps, (batch,), device=device, dtype=torch.long)
        noise = torch.randn_like(target)
        self.q_sample(target, t, noise=noise)
        model_pred, _ = self.model(node_feats, positions, mask=mask)
        model_pred = model_pred.gather(dim=1, index=t.view(batch, 1, 1).expand(-1, model_pred.size(1), 2))
        target_noise = noise if self.config.prediction_type == "velocity" else target
        loss = torch.mean((model_pred.squeeze(1) - target_noise.mean(dim=1)) ** 2)
        return loss, model_pred.squeeze(1)

    @torch.no_grad()
    def sample(self, node_feats: torch.Tensor, positions: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, time_steps, _, _ = node_feats.shape
        sample = torch.randn(batch, time_steps, 2, device=node_feats.device)
        for step in reversed(range(self.config.timesteps)):
            model_pred, _ = self.model(node_feats, positions, mask=mask)
            eps_theta = model_pred.mean(dim=1)
            alpha = self.alphas_cumprod[step]
            beta = self.betas[step]
            mean = (1 / torch.sqrt(alpha)) * (sample - ((1 - alpha) / torch.sqrt(1 - self.alphas_cumprod[step])) * eps_theta)
            if step > 0:
                noise = torch.randn_like(sample)
                sample = mean + torch.sqrt(beta) * noise
            else:
                sample = mean
        return sample
