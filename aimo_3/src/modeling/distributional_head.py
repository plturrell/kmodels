"""
Distributional Prediction Head for Uncertainty Quantification.

Predicts probability distributions over latent states rather than point estimates,
enabling uncertainty-aware world model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DistributionalHead(nn.Module):
    """
    Distributional prediction head that outputs mean and variance.
    
    Supports:
    - Gaussian predictions with learned variance
    - Mixture of Gaussians for multimodal predictions
    - Uncertainty decomposition (aleatoric vs epistemic)
    - Calibrated uncertainty estimates
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_components: int = 1,
        min_variance: float = 1e-6,
        max_variance: float = 10.0,
        dropout: float = 0.1,
    ):
        """
        Initialize distributional head.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output latent dimension
            num_components: Number of mixture components (1 = single Gaussian)
            min_variance: Minimum allowed variance (for numerical stability)
            max_variance: Maximum allowed variance (to prevent explosion)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_components = num_components
        self.min_variance = min_variance
        self.max_variance = max_variance
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim),
        )
        
        if num_components == 1:
            # Single Gaussian: predict mean and log-variance
            self.mean_head = nn.Linear(input_dim, output_dim)
            self.logvar_head = nn.Linear(input_dim, output_dim)
        else:
            # Mixture of Gaussians
            self.mean_heads = nn.ModuleList([
                nn.Linear(input_dim, output_dim)
                for _ in range(num_components)
            ])
            self.logvar_heads = nn.ModuleList([
                nn.Linear(input_dim, output_dim)
                for _ in range(num_components)
            ])
            # Mixture weights (logits)
            self.mixture_logits = nn.Linear(input_dim, num_components)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: predict distribution parameters.
        
        Args:
            x: Input features, shape [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (mean, variance, mixture_weights):
            - mean: Predicted mean, shape [batch_size, output_dim] or 
                   [batch_size, num_components, output_dim] for mixtures
            - variance: Predicted variance, same shape as mean
            - mixture_weights: Mixture component weights [batch_size, num_components]
                              or None for single Gaussian
        """
        # Extract features
        features = self.feature_extractor(x)
        
        if self.num_components == 1:
            # Single Gaussian
            mean = self.mean_head(features)
            logvar = self.logvar_head(features)
            
            # Convert log-variance to variance with bounds
            variance = self._logvar_to_variance(logvar)
            
            return mean, variance, None
        else:
            # Mixture of Gaussians
            means = torch.stack([
                head(features) for head in self.mean_heads
            ], dim=-2)  # [B, K, D] or [B, T, K, D]
            
            logvars = torch.stack([
                head(features) for head in self.logvar_heads
            ], dim=-2)
            
            variances = self._logvar_to_variance(logvars)
            
            # Mixture weights
            mixture_logits = self.mixture_logits(features)  # [B, K] or [B, T, K]
            mixture_weights = F.softmax(mixture_logits, dim=-1)
            
            return means, variances, mixture_weights
    
    def _logvar_to_variance(self, logvar: torch.Tensor) -> torch.Tensor:
        """Convert log-variance to variance with clamping."""
        # Clamp log-variance to reasonable range
        log_min = math.log(self.min_variance)
        log_max = math.log(self.max_variance)
        logvar = torch.clamp(logvar, log_min, log_max)
        
        # Convert to variance
        variance = torch.exp(logvar)
        
        return variance
    
    def sample(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        mixture_weights: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from predicted distribution.
        
        Args:
            mean: Predicted mean
            variance: Predicted variance
            mixture_weights: Mixture weights (for MoG)
            num_samples: Number of samples to draw
            
        Returns:
            Samples, shape [num_samples, batch_size, output_dim]
        """
        if self.num_components == 1:
            # Single Gaussian sampling
            std = torch.sqrt(variance)
            epsilon = torch.randn(
                num_samples, *mean.shape,
                device=mean.device, dtype=mean.dtype
            )
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * epsilon
        else:
            # Mixture of Gaussians sampling
            if mixture_weights is None:
                raise ValueError("mixture_weights must be provided when num_components > 1")
            batch_size = mean.shape[0]
            
            # Sample component indices
            component_indices = torch.multinomial(
                mixture_weights,
                num_samples=num_samples,
                replacement=True,
            )  # [B, num_samples]
            
            # Sample from selected components
            samples_list: list[torch.Tensor] = []
            for i in range(num_samples):
                batch_samples: list[torch.Tensor] = []
                for b in range(batch_size):
                    k = component_indices[b, i]
                    component_mean = mean[b, k]
                    component_var = variance[b, k]
                    component_std = torch.sqrt(component_var)
                    
                    epsilon = torch.randn_like(component_mean)
                    sample = component_mean + component_std * epsilon
                    batch_samples.append(sample)
                
                samples_list.append(torch.stack(batch_samples))
            
            samples = torch.stack(samples_list)  # [num_samples, B, D]
        
        return samples
    
    def nll_loss(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        target: torch.Tensor,
        mixture_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            mean: Predicted mean
            variance: Predicted variance
            target: Target values
            mixture_weights: Mixture weights (for MoG)
            
        Returns:
            NLL loss scalar
        """
        if self.num_components == 1:
            # Single Gaussian NLL
            # L = 0.5 * (log(2π) + log(var) + (target - mean)^2 / var)
            squared_error = (target - mean) ** 2
            nll = 0.5 * (
                math.log(2 * math.pi)
                + torch.log(variance)
                + squared_error / variance
            )
            return nll.mean()
        else:
            # Mixture of Gaussians NLL
            # L = -log(Σ_k π_k * N(target | μ_k, σ_k^2))
            if mixture_weights is None:
                raise ValueError("mixture_weights must be provided when num_components > 1")
            
            # Compute log probabilities for each component
            log_probs: list[torch.Tensor] = []
            for k in range(self.num_components):
                component_mean = mean[..., k, :]
                component_var = variance[..., k, :]
                
                squared_error = (target.unsqueeze(-2) - component_mean) ** 2
                log_prob = -0.5 * (
                    math.log(2 * math.pi)
                    + torch.log(component_var)
                    + squared_error / component_var
                )
                log_probs.append(log_prob.sum(dim=-1))  # Sum over dimensions
            
            log_probs_t = torch.stack(log_probs, dim=-1)  # [B, K]
            
            # Weight by mixture probabilities
            log_mixture_probs = torch.log(mixture_weights + 1e-8)
            log_prob_total = torch.logsumexp(log_probs_t + log_mixture_probs, dim=-1)
            
            nll = -log_prob_total.mean()
            return nll


class EnsembleHead(nn.Module):
    """
    Ensemble of prediction heads for epistemic uncertainty estimation.
    
    Epistemic uncertainty (model uncertainty) is captured by disagreement
    between ensemble members.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int = 5,
        **kwargs,
    ):
        """
        Initialize ensemble head.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_heads: Number of ensemble members
            **kwargs: Arguments passed to individual DistributionalHead
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Create ensemble of distributional heads
        self.heads = nn.ModuleList([
            DistributionalHead(input_dim, output_dim, **kwargs)
            for _ in range(num_heads)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Returns:
            Tuple of (mean, aleatoric_variance, epistemic_variance):
            - mean: Ensemble mean prediction
            - aleatoric_variance: Average predictive variance (aleatoric)
            - epistemic_variance: Variance of means (epistemic)
        """
        predictions = [head(x) for head in self.heads]
        
        # Stack predictions
        means = torch.stack([p[0] for p in predictions], dim=0)  # [num_heads, B, D]
        variances = torch.stack([p[1] for p in predictions], dim=0)
        
        # Compute ensemble statistics
        ensemble_mean = means.mean(dim=0)  # [B, D]
        aleatoric_variance = variances.mean(dim=0)  # Average predictive variance
        epistemic_variance = means.var(dim=0)  # Variance of predictions
        
        return ensemble_mean, aleatoric_variance, epistemic_variance
    
    def total_uncertainty(
        self,
        aleatoric_variance: torch.Tensor,
        epistemic_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total uncertainty (aleatoric + epistemic)."""
        return aleatoric_variance + epistemic_variance


def create_distributional_head(
    input_dim: int,
    output_dim: int,
    ensemble: bool = False,
    num_components: int = 1,
    num_heads: int = 5,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create distributional head.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        ensemble: Whether to use ensemble for epistemic uncertainty
        num_components: Number of mixture components
        num_heads: Number of ensemble members (if ensemble=True)
        **kwargs: Additional arguments
        
    Returns:
        DistributionalHead or EnsembleHead
    """
    if ensemble:
        return EnsembleHead(
            input_dim=input_dim,
            output_dim=output_dim,
            num_components=num_components,
            num_heads=num_heads,
            **kwargs,
        )
    else:
        return DistributionalHead(
            input_dim=input_dim,
            output_dim=output_dim,
            num_components=num_components,
            **kwargs,
        )
