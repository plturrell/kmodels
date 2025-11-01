"""Custom loss functions for the biomass competition."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """A loss function that incorporates a simplified, learnable physics/biology growth model."""

    def __init__(self, physics_weight: float = 0.1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.physics_weight = physics_weight

        # Define learnable parameters for the growth model
        # growth_rate = C1 * sunlight + C2 * rainfall - C3 (decay)
        self.growth_factor_sunlight = nn.Parameter(torch.randn(1))
        self.growth_factor_rainfall = nn.Parameter(torch.randn(1))
        self.decay_factor = nn.Parameter(torch.randn(1))

    def _estimate_growth(self, metadata_t1: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        A simplified, learnable growth model.
        Assumes metadata tensor has columns for 'sunlight' and 'rainfall' at known indices.
        This needs to be configured based on the actual metadata columns.
        """
        # Placeholder indices - these must match the actual data
        sunlight_idx, rainfall_idx = -2, -1 # Assuming they are the last two columns

        sunlight = metadata_t1[:, sunlight_idx]
        rainfall = metadata_t1[:, rainfall_idx]

        # The growth model equation
        growth_rate = (
            self.growth_factor_sunlight * sunlight + 
            self.growth_factor_rainfall * rainfall - 
            torch.relu(self.decay_factor) # Ensure decay is non-negative
        )
        
        expected_change = growth_rate * dt
        return expected_change

    def forward(self, 
              pred_t1: torch.Tensor, target_t1: torch.Tensor, metadata_t1: torch.Tensor,
              pred_t2: torch.Tensor, target_t2: torch.Tensor, metadata_t2: torch.Tensor,
              dt: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total loss.
        """
        # 1. Standard Data Loss (MSE)
        loss_data_t1 = self.mse_loss(pred_t1, target_t1)
        loss_data_t2 = self.mse_loss(pred_t2, target_t2)
        loss_data = (loss_data_t1 + loss_data_t2) / 2.0

        # 2. Physics-Informed Loss
        biomass_pred_t1 = pred_t1[:, 0]
        biomass_pred_t2 = pred_t2[:, 0]
        predicted_change = biomass_pred_t2 - biomass_pred_t1
        
        expected_change = self._estimate_growth(metadata_t1, dt)

        loss_physics = self.mse_loss(predicted_change, expected_change)

        # 3. Total Loss
        total_loss = loss_data + self.physics_weight * loss_physics
        return total_loss

class GaussianNLLLoss(nn.Module):
    """Computes the Negative Log-Likelihood for a Gaussian distribution."""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mean: The predicted mean of the Gaussian distribution.
            log_var: The predicted log variance of the Gaussian distribution.
            target: The ground truth value.
        """
        # Ensure variance is positive
        log_var = torch.clamp(log_var, min=-10, max=10)
        variance = torch.exp(log_var)
        
        # Calculate the negative log-likelihood
        nll = 0.5 * (torch.log(2 * np.pi * variance) + ((target - mean) ** 2) / variance)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll
