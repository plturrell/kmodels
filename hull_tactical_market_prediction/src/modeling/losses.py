"""Custom loss functions for Hull Tactical Market Prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


LossMode = Literal["mse", "sharpe", "sharpe_mse"]


@dataclass
class LossConfig:
    mode: LossMode = "mse"
    sharpe_risk_free: float = 0.0
    sharpe_lambda: float = 1.0
    mse_weight: float = 0.5


class LossFactory:
    """Compute regression losses with optional Sharpe-style objective."""

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mode = self.config.mode
        if mode == "mse":
            return F.mse_loss(preds, target)
        if mode == "sharpe":
            return self._negative_sharpe(preds, target)
        if mode == "sharpe_mse":
            sharpe_loss = self._negative_sharpe(preds, target)
            mse_loss = F.mse_loss(preds, target)
            return self.config.mse_weight * mse_loss + (1 - self.config.mse_weight) * sharpe_loss
        raise ValueError(f"Unsupported loss mode: {mode}")

    def _negative_sharpe(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        excess = preds - self.config.sharpe_risk_free
        diff = excess - target
        mean_return = torch.mean(excess - target)
        volatility = torch.std(excess - target)
        volatility = torch.clamp(volatility, min=1e-6)
        sharpe = mean_return / volatility
        penalty = self.config.sharpe_lambda * torch.mean(diff**2)
        return -sharpe + penalty
