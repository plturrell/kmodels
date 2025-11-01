"""Metric helpers borrowed from the CSIRO biomass workspace pattern."""

from __future__ import annotations

from typing import Dict

import torch


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """Return simple accuracy statistics."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy}


def compute_segmentation_metrics(
    mask_logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Return Dice and IoU scores."""
    probs = torch.sigmoid(mask_logits)
    preds = (probs > threshold).float()
    masks = (masks > threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    mask_area = masks.sum(dim=(1, 2, 3))
    union = pred_area + mask_area - intersection

    dice = (2 * intersection + eps) / (pred_area + mask_area + eps)
    iou = (intersection + eps) / (union + eps)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
    }
