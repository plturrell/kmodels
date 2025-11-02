"""Advanced loss functions for forgery detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Logits (B, C)
            targets: Class labels (B,)
        
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        """Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            inputs: Logits (B, 1, H, W)
            targets: Binary masks (B, 1, H, W)
        
        Returns:
            Loss value
        """
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        """Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss.
        
        Args:
            inputs: Logits (B, 1, H, W)
            targets: Binary masks (B, 1, H, W)
        
        Returns:
            Loss value
        """
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1 - tversky


class CombinedLoss(nn.Module):
    """Combined loss for segmentation (Dice + BCE)."""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        """Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            inputs: Logits (B, 1, H, W)
            targets: Binary masks (B, 1, H, W)
        
        Returns:
            Loss value
        """
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce


class ContrastiveLoss(nn.Module):
    """NT-Xent loss for region-level contrastive supervision."""

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        if z_i.shape != z_j.shape:
            raise ValueError("Contrastive pairs must share the same shape")

        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)
        representations = nn.functional.normalize(representations, dim=1)
        similarity = representations @ representations.T

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity.device)
        similarity = similarity / self.temperature
        similarity = similarity.masked_fill(mask, float("-inf"))

        positives = torch.arange(batch_size, device=similarity.device)
        targets = torch.cat([positives + batch_size, positives], dim=0)

        loss = nn.functional.cross_entropy(similarity, targets)
        return loss


__all__ = [
    "FocalLoss",
    "DiceLoss",
    "TverskyLoss",
    "CombinedLoss",
    "ContrastiveLoss",
]

