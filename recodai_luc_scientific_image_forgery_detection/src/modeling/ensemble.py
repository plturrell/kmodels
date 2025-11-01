"""Ensemble methods for forgery detection models."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models with weighted averaging."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """Initialize ensemble.
        
        Args:
            models: List of models to ensemble
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        self.register_buffer("weights", torch.tensor(weights))
    
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ensemble.
        
        Args:
            image: Input image tensor
        
        Returns:
            Tuple of (class_logits, mask_logits)
        """
        class_logits_list = []
        mask_logits_list = []
        
        for model in self.models:
            class_logits, mask_logits = model(image)
            class_logits_list.append(class_logits)
            mask_logits_list.append(mask_logits)
        
        # Stack and weight
        class_logits = torch.stack(class_logits_list)  # (n_models, batch, classes)
        mask_logits = torch.stack(mask_logits_list)  # (n_models, batch, 1, H, W)
        
        # Weighted average
        weights = self.weights.view(-1, 1, 1)  # (n_models, 1, 1)
        class_logits = (class_logits * weights).sum(dim=0)
        
        weights_mask = self.weights.view(-1, 1, 1, 1, 1)  # (n_models, 1, 1, 1, 1)
        mask_logits = (mask_logits * weights_mask).sum(dim=0)
        
        return class_logits, mask_logits


def load_models_from_checkpoints(
    checkpoint_paths: List[Path],
    model_factory,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[nn.Module]:
    """Load multiple models from checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        model_factory: Function that returns a new model instance
        device: Device to load models on
    
    Returns:
        List of loaded models
    """
    models = []
    
    for checkpoint_path in checkpoint_paths:
        model = model_factory()
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "state_dict" in checkpoint:
            # Lightning checkpoint
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        models.append(model)
    
    return models


def create_fold_ensemble(
    cv_output_dir: Path,
    model_factory,
    n_folds: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> ModelEnsemble:
    """Create ensemble from cross-validation folds.
    
    Args:
        cv_output_dir: Directory containing fold subdirectories
        model_factory: Function that returns a new model instance
        n_folds: Number of folds
        device: Device to load models on
    
    Returns:
        ModelEnsemble instance
    """
    checkpoint_paths = []
    
    for fold_idx in range(1, n_folds + 1):
        fold_dir = cv_output_dir / f"fold_{fold_idx}"
        checkpoint_path = fold_dir / "checkpoints" / "best.ckpt"
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found for fold {fold_idx}")
            continue
        
        checkpoint_paths.append(checkpoint_path)
    
    if not checkpoint_paths:
        raise FileNotFoundError("No fold checkpoints found")
    
    print(f"Loading {len(checkpoint_paths)} models for ensemble")
    models = load_models_from_checkpoints(checkpoint_paths, model_factory, device)
    
    return ModelEnsemble(models)


__all__ = [
    "ModelEnsemble",
    "load_models_from_checkpoints",
    "create_fold_ensemble",
]

