"""Advanced loss functions for ECG signal regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber loss) - more robust to outliers than MSE."""
    
    def __init__(self, beta: float = 1.0):
        """Initialize Smooth L1 loss.
        
        Args:
            beta: Threshold for switching between L1 and L2
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute smooth L1 loss."""
        return F.smooth_l1_loss(pred, target, beta=self.beta)


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss - weight different signal regions differently."""
    
    def __init__(self, weight_fn: str = "uniform"):
        """Initialize weighted MSE loss.
        
        Args:
            weight_fn: Weighting function ('uniform', 'center', 'edges')
        """
        super().__init__()
        self.weight_fn = weight_fn
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss."""
        batch_size, signal_length = pred.shape
        
        if self.weight_fn == "uniform":
            weights = torch.ones_like(pred)
        elif self.weight_fn == "center":
            # Weight center more
            center = signal_length // 2
            distances = torch.abs(torch.arange(signal_length, device=pred.device) - center)
            weights = 1.0 / (1.0 + distances / signal_length)
            weights = weights.unsqueeze(0).expand(batch_size, -1)
        elif self.weight_fn == "edges":
            # Weight edges more
            center = signal_length // 2
            distances = torch.abs(torch.arange(signal_length, device=pred.device) - center)
            weights = distances / signal_length
            weights = weights.unsqueeze(0).expand(batch_size, -1)
        else:
            raise ValueError(f"Unknown weight function: {self.weight_fn}")
        
        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # Weighted MSE
        squared_error = (pred - target) ** 2
        return (squared_error * weights).sum() / batch_size


class PerceptualLoss(nn.Module):
    """Perceptual loss using 1D convolutions to extract features."""
    
    def __init__(self, hidden_channels: int = 64):
        """Initialize perceptual loss.
        
        Args:
            hidden_channels: Number of channels in feature extractor
        """
        super().__init__()
        
        # Simple 1D CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        
        # Freeze feature extractor (optional)
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # Add channel dimension
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        
        # Extract features
        pred_features = self.features(pred)
        target_features = self.features(target)
        
        # MSE on features
        return F.mse_loss(pred_features, target_features)


class FrequencyDomainLoss(nn.Module):
    """Loss in frequency domain using FFT."""
    
    def __init__(self, alpha: float = 0.5):
        """Initialize frequency domain loss.
        
        Args:
            alpha: Weight for frequency domain loss (1-alpha for time domain)
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain loss."""
        # Time domain loss
        time_loss = F.mse_loss(pred, target)
        
        # Frequency domain loss
        pred_fft = torch.fft.rfft(pred, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        
        # MSE on magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        freq_loss = F.mse_loss(pred_mag, target_mag)
        
        # Combined loss
        return (1 - self.alpha) * time_loss + self.alpha * freq_loss


class CombinedLoss(nn.Module):
    """Combined loss with multiple components."""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        mae_weight: float = 0.5,
        smooth_l1_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        frequency_weight: float = 0.0,
    ):
        """Initialize combined loss.
        
        Args:
            mse_weight: Weight for MSE loss
            mae_weight: Weight for MAE loss
            smooth_l1_weight: Weight for Smooth L1 loss
            perceptual_weight: Weight for perceptual loss
            frequency_weight: Weight for frequency domain loss
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.smooth_l1_weight = smooth_l1_weight
        self.perceptual_weight = perceptual_weight
        self.frequency_weight = frequency_weight
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        if frequency_weight > 0:
            self.frequency_loss = FrequencyDomainLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        loss = 0.0
        
        if self.mse_weight > 0:
            loss += self.mse_weight * F.mse_loss(pred, target)
        
        if self.mae_weight > 0:
            loss += self.mae_weight * F.l1_loss(pred, target)
        
        if self.smooth_l1_weight > 0:
            loss += self.smooth_l1_weight * F.smooth_l1_loss(pred, target)
        
        if self.perceptual_weight > 0:
            loss += self.perceptual_weight * self.perceptual_loss(pred, target)
        
        if self.frequency_weight > 0:
            loss += self.frequency_weight * self.frequency_loss(pred, target)
        
        return loss


__all__ = [
    "SmoothL1Loss",
    "WeightedMSELoss",
    "PerceptualLoss",
    "FrequencyDomainLoss",
    "CombinedLoss",
]

