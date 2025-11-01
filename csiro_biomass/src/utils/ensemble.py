"""Ensemble utilities for combining predictions from multiple models.

Supports various ensemble strategies:
- Simple averaging
- Weighted averaging
- Rank averaging
- Stacking (meta-learning)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_predictions(prediction_files: Sequence[Path]) -> List[pd.DataFrame]:
    """Load prediction CSV files.
    
    Args:
        prediction_files: List of paths to prediction CSV files
    
    Returns:
        List of DataFrames with predictions
    """
    predictions = []
    for file_path in prediction_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {file_path}")
        df = pd.read_csv(file_path)
        predictions.append(df)
        LOGGER.info(f"Loaded {len(df)} predictions from {file_path.name}")
    
    return predictions


def simple_average_ensemble(
    prediction_files: Sequence[Path],
    output_path: Path,
    weights: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Create ensemble by averaging predictions.
    
    Args:
        prediction_files: List of paths to prediction CSV files
        output_path: Path to save ensemble predictions
        weights: Optional weights for each model (default: equal weights)
    
    Returns:
        DataFrame with ensemble predictions
    """
    predictions = load_predictions(prediction_files)
    
    # Validate all have same structure
    base_df = predictions[0].copy()
    for i, df in enumerate(predictions[1:], 1):
        if not df['sample_id'].equals(base_df['sample_id']):
            raise ValueError(f"Sample IDs don't match between file 0 and {i}")
    
    # Set weights
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)
    else:
        if len(weights) != len(predictions):
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of predictions ({len(predictions)})")
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    LOGGER.info(f"Ensemble weights: {weights}")
    
    # Average predictions
    ensemble_df = base_df.copy()
    ensemble_df['target'] = sum(
        df['target'].values * weight
        for df, weight in zip(predictions, weights)
    )
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved ensemble predictions to {output_path}")
    
    return ensemble_df


def rank_average_ensemble(
    prediction_files: Sequence[Path],
    output_path: Path,
) -> pd.DataFrame:
    """Create ensemble by averaging ranks instead of raw predictions.
    
    This can be more robust when models have different scales.
    
    Args:
        prediction_files: List of paths to prediction CSV files
        output_path: Path to save ensemble predictions
    
    Returns:
        DataFrame with ensemble predictions
    """
    predictions = load_predictions(prediction_files)
    
    # Validate structure
    base_df = predictions[0].copy()
    for i, df in enumerate(predictions[1:], 1):
        if not df['sample_id'].equals(base_df['sample_id']):
            raise ValueError(f"Sample IDs don't match between file 0 and {i}")
    
    # Convert to ranks
    ranks = []
    for df in predictions:
        rank = df['target'].rank(method='average', pct=True)  # Percentile ranks
        ranks.append(rank.values)
    
    # Average ranks
    avg_rank = np.mean(ranks, axis=0)
    
    # Create ensemble
    ensemble_df = base_df.copy()
    ensemble_df['target'] = avg_rank
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved rank-averaged ensemble to {output_path}")
    
    return ensemble_df


def weighted_ensemble_by_performance(
    prediction_files: Sequence[Path],
    validation_metrics: Sequence[float],
    output_path: Path,
    metric_type: str = "rmse",
) -> pd.DataFrame:
    """Create weighted ensemble based on validation performance.
    
    Args:
        prediction_files: List of paths to prediction CSV files
        validation_metrics: Validation metric for each model
        output_path: Path to save ensemble predictions
        metric_type: Type of metric ("rmse" or "mae" - lower is better)
    
    Returns:
        DataFrame with ensemble predictions
    """
    if len(prediction_files) != len(validation_metrics):
        raise ValueError("Number of prediction files must match number of metrics")
    
    # Convert metrics to weights (inverse for RMSE/MAE)
    if metric_type.lower() in ["rmse", "mae"]:
        # Lower is better, so use inverse
        inv_metrics = [1.0 / m for m in validation_metrics]
        weights = [m / sum(inv_metrics) for m in inv_metrics]
    else:
        # Higher is better (e.g., R2)
        weights = [m / sum(validation_metrics) for m in validation_metrics]
    
    LOGGER.info(f"Performance-based weights: {weights}")
    
    return simple_average_ensemble(prediction_files, output_path, weights)


def ensemble_cross_validation_folds(
    fold_dirs: Sequence[Path],
    output_path: Path,
    prediction_filename: str = "validation_predictions.csv",
) -> pd.DataFrame:
    """Ensemble predictions from cross-validation folds.
    
    Args:
        fold_dirs: List of directories containing fold results
        output_path: Path to save ensemble predictions
        prediction_filename: Name of prediction file in each fold directory
    
    Returns:
        DataFrame with ensemble predictions
    """
    prediction_files = []
    for fold_dir in fold_dirs:
        pred_file = fold_dir / prediction_filename
        if pred_file.exists():
            prediction_files.append(pred_file)
        else:
            LOGGER.warning(f"Prediction file not found: {pred_file}")
    
    if not prediction_files:
        raise ValueError("No prediction files found in fold directories")
    
    LOGGER.info(f"Ensembling {len(prediction_files)} folds")
    return simple_average_ensemble(prediction_files, output_path)


def create_stacked_features(
    prediction_files: Sequence[Path],
) -> pd.DataFrame:
    """Create features for stacking ensemble (meta-learning).
    
    Args:
        prediction_files: List of paths to prediction CSV files
    
    Returns:
        DataFrame with predictions from all models as features
    """
    predictions = load_predictions(prediction_files)
    
    # Start with sample IDs
    stacked = predictions[0][['sample_id']].copy()
    
    # Add predictions from each model as features
    for i, df in enumerate(predictions):
        stacked[f'model_{i}_pred'] = df['target'].values
    
    return stacked


__all__ = [
    "simple_average_ensemble",
    "rank_average_ensemble",
    "weighted_ensemble_by_performance",
    "ensemble_cross_validation_folds",
    "create_stacked_features",
    "load_predictions",
]

