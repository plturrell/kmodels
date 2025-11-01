"""
Cross-validation utilities for CAFA 6 protein function prediction.

Implements stratified k-fold cross-validation with support for multi-label data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold

from ..data import ProteinSample

LOGGER = logging.getLogger(__name__)


class StratifiedMultiLabelKFold:
    """Stratified k-fold cross-validation for multi-label data.
    
    Stratifies based on the number of labels per sample to maintain
    label distribution across folds.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """Initialize stratified k-fold splitter.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        samples: Sequence[ProteinSample],
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate train/val indices for each fold.
        
        Args:
            samples: Sequence of protein samples
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Group samples by number of GO terms
        label_counts = np.array([len(s.go_terms) for s in samples])
        
        # Create bins for stratification
        # Use quantiles to create balanced bins
        n_bins = min(10, len(np.unique(label_counts)))
        bins = np.percentile(label_counts, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        stratify_labels = np.digitize(label_counts, bins)
        
        # Use sklearn's KFold with manual stratification
        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        
        # Sort indices by stratify labels to ensure balanced splits
        sorted_indices = np.argsort(stratify_labels)
        
        splits = []
        for train_idx, val_idx in kfold.split(sorted_indices):
            # Map back to original indices
            train_indices = sorted_indices[train_idx].tolist()
            val_indices = sorted_indices[val_idx].tolist()
            splits.append((train_indices, val_indices))
        
        LOGGER.info(f"Created {self.n_splits} stratified folds")
        return splits


def cross_validate(
    samples: Sequence[ProteinSample],
    train_fn: Callable,
    eval_fn: Callable,
    n_splits: int = 5,
    random_state: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """Perform k-fold cross-validation.
    
    Args:
        samples: Sequence of protein samples
        train_fn: Function that trains a model given (train_samples, val_samples, fold_idx)
                  Returns trained model
        eval_fn: Function that evaluates a model given (model, val_samples, fold_idx)
                 Returns dictionary of metrics
        n_splits: Number of folds
        random_state: Random seed
        output_dir: Optional directory to save fold results
    
    Returns:
        Dictionary mapping metric names to lists of scores across folds
    """
    LOGGER.info(f"Starting {n_splits}-fold cross-validation on {len(samples)} samples")
    
    # Create folds
    splitter = StratifiedMultiLabelKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = splitter.split(samples)
    
    # Store results
    all_metrics: Dict[str, List[float]] = {}
    
    # Run each fold
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Fold {fold_idx + 1}/{n_splits}")
        LOGGER.info(f"{'='*60}")
        
        # Split samples
        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]
        
        LOGGER.info(f"Train: {len(train_samples)} samples")
        LOGGER.info(f"Val: {len(val_samples)} samples")
        
        # Train model
        LOGGER.info("Training model...")
        model = train_fn(train_samples, val_samples, fold_idx)
        
        # Evaluate model
        LOGGER.info("Evaluating model...")
        metrics = eval_fn(model, val_samples, fold_idx)
        
        # Store metrics
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)
        
        # Log fold results
        LOGGER.info(f"\nFold {fold_idx + 1} results:")
        for metric_name, value in metrics.items():
            LOGGER.info(f"  {metric_name}: {value:.4f}")
    
    # Calculate summary statistics
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("Cross-Validation Summary")
    LOGGER.info(f"{'='*60}")
    
    summary = {}
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[f"{metric_name}_mean"] = mean_val
        summary[f"{metric_name}_std"] = std_val
        LOGGER.info(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save results if output directory provided
    if output_dir:
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'fold_results': all_metrics,
            'summary': summary,
            'n_splits': n_splits,
            'n_samples': len(samples),
        }
        
        output_path = output_dir / "cv_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        LOGGER.info(f"\nSaved CV results to {output_path}")
    
    return all_metrics


__all__ = [
    "StratifiedMultiLabelKFold",
    "cross_validate",
]

