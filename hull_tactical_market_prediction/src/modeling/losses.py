"""Custom loss functions for Hull Tactical Market Prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


LossMode = Literal["mse", "sharpe", "sharpe_mse", "heteroscedastic"]


@dataclass
class LossConfig:
    mode: LossMode = "mse"
    sharpe_risk_free: float = 0.0
    sharpe_lambda: float = 1.0
    mse_weight: float = 0.5
    heteroscedastic_weight: float = 0.5  # Weight for variance prediction in heteroscedastic loss


class LossFactory:
    """Compute regression losses with optional Sharpe-style objective."""

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(
        self, 
        preds: torch.Tensor, 
        target: torch.Tensor,
        pred_variance: Optional[torch.Tensor] = None,
        target_uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = self.config.mode
        if mode == "mse":
            return F.mse_loss(preds, target)
        if mode == "sharpe":
            return self._negative_sharpe(preds, target)
        if mode == "sharpe_mse":
            sharpe_loss = self._negative_sharpe(preds, target)
            mse_loss = F.mse_loss(preds, target)
            return self.config.mse_weight * mse_loss + (1 - self.config.mse_weight) * sharpe_loss
        if mode == "heteroscedastic":
            return self._heteroscedastic_loss(preds, target, pred_variance, target_uncertainty)
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
    
    def _heteroscedastic_loss(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        pred_variance: Optional[torch.Tensor],
        target_uncertainty: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Heteroscedastic loss that models prediction uncertainty.
        
        Uses negative log-likelihood of Gaussian with predicted mean and variance.
        If target_uncertainty is provided (from Kalman smoothing), anchors predictions to it.
        """
        if pred_variance is None:
            # Fallback to MSE if no variance prediction
            return F.mse_loss(preds, target)
        
        # Ensure variance is positive and stable
        pred_variance = torch.clamp(pred_variance, min=1e-6)
        
        # Negative log-likelihood of Gaussian: 0.5 * log(2*pi*var) + 0.5 * (pred - target)^2 / var
        precision = 1.0 / pred_variance
        squared_error = (preds - target) ** 2
        nll = 0.5 * torch.log(2 * torch.tensor(3.14159, device=preds.device) * pred_variance) + 0.5 * precision * squared_error
        
        loss = torch.mean(nll)
        
        # If target uncertainties are available, add regularization to align predictions
        if target_uncertainty is not None:
            # Scale target_uncertainty to match pred_variance scale if needed
            target_var = target_uncertainty ** 2
            target_var = torch.clamp(target_var, min=1e-6)
            # KL divergence or MSE between predicted and target uncertainties
            var_mismatch = F.mse_loss(pred_variance, target_var)
            loss = loss + self.config.heteroscedastic_weight * var_mismatch
        
        return loss
