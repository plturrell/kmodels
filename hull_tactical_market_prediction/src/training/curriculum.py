"""Curriculum learning utilities for temporal sampling with uncertainty gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


DifficultyMetric = Literal["volatility", "ensemble_disagreement", "prediction_confidence"]


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    enabled: bool = False
    start_ratio: float = 0.1  # Start with easiest 10% of samples
    difficulty_metric: DifficultyMetric = "volatility"
    adaptive_dropout: bool = False  # Tie dropout to sample uncertainty
    min_difficulty_epoch: int = 0  # Epoch to start introducing harder samples
    max_difficulty_epoch: int = 10  # Epoch to reach full difficulty range


def compute_volatility_difficulty(
    features: pd.DataFrame | np.ndarray,
    window: int = 5,
    regime_aware: bool = True,
) -> np.ndarray:
    """Compute difficulty scores based on recent volatility with regime awareness.
    
    Args:
        features: Feature matrix (samples x features)
        window: Rolling window size for volatility calculation
        regime_aware: If True, detect regime changes and weight by regime stability
        
    Returns:
        Difficulty scores (higher = more difficult)
    """
    if isinstance(features, pd.DataFrame):
        features_array = features.values
    else:
        features_array = np.asarray(features)
    
    if features_array.shape[0] < window:
        return np.ones(features_array.shape[0])
    
    # Compute rolling standard deviation across all features
    volatilities = []
    regime_changes = np.zeros(len(features_array))
    
    for i in range(len(features_array)):
        start_idx = max(0, i - window + 1)
        window_data = features_array[start_idx:i + 1]
        vol = np.std(window_data, axis=0)
        volatilities.append(np.mean(vol))
        
        # Detect regime changes: large shifts in feature means
        if i > 0 and regime_aware:
            prev_window = features_array[max(0, i - window):i] if i >= window else features_array[:i]
            if len(prev_window) > 0:
                mean_shift = np.mean(np.abs(window_data[-1] - np.mean(prev_window, axis=0)))
                regime_changes[i] = mean_shift
    
    volatilities = np.array(volatilities)
    
    if regime_aware:
        # Normalize regime changes
        if regime_changes.max() > 0:
            regime_changes = regime_changes / (regime_changes.max() + 1e-8)
        # Combine volatility with regime change penalty
        # High volatility + regime change = very difficult
        difficulty = volatilities * (1.0 + regime_changes)
    else:
        difficulty = volatilities
    
    return difficulty


def compute_ensemble_disagreement(
    predictions: list[np.ndarray] | np.ndarray,
    checkpoint_paths: Optional[list] = None,
) -> np.ndarray:
    """Compute difficulty based on ensemble model disagreement.
    
    Args:
        predictions: List of prediction arrays from different models, or single array
        checkpoint_paths: Optional list of checkpoint paths to load and evaluate
        
    Returns:
        Disagreement scores (higher = more difficult)
    """
    if isinstance(predictions, np.ndarray):
        predictions = [predictions]
    
    if len(predictions) < 2:
        # If only one prediction, fall back to volatility or return zeros
        if len(predictions) == 0:
            return np.array([])
        # Use prediction variance as proxy
        pred = predictions[0]
        return np.abs(pred - np.mean(pred))
    
    pred_array = np.stack(predictions)
    # Standard deviation across models indicates disagreement
    disagreement = np.std(pred_array, axis=0)
    return disagreement.flatten()


def compute_prediction_confidence_difficulty(
    predictions: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute difficulty based on prediction confidence.
    
    Args:
        predictions: Model predictions
        uncertainties: Optional uncertainty estimates
        
    Returns:
        Difficulty scores (higher uncertainty = more difficult)
    """
    if uncertainties is not None:
        return uncertainties.flatten()
    
    # Use prediction variance as proxy for uncertainty
    # For regression, use absolute deviation from mean as difficulty proxy
    mean_pred = np.mean(predictions)
    difficulty = np.abs(predictions - mean_pred).flatten()
    return difficulty


def rank_samples_by_difficulty(
    features: pd.DataFrame | np.ndarray,
    difficulty_metric: DifficultyMetric = "volatility",
    predictions: Optional[list[np.ndarray] | np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    checkpoint_paths: Optional[list] = None,
) -> np.ndarray:
    """Rank samples by difficulty using the specified metric.
    
    Args:
        features: Feature matrix
        difficulty_metric: Metric to use for ranking
        predictions: Optional list of model predictions (for ensemble_disagreement)
        uncertainties: Optional uncertainty estimates (for prediction_confidence)
        
    Returns:
        Sorted indices (easiest first)
    """
    if difficulty_metric == "volatility":
        difficulty_scores = compute_volatility_difficulty(features)
    elif difficulty_metric == "ensemble_disagreement":
        if predictions is None or (isinstance(predictions, list) and len(predictions) < 2):
            # Fallback to volatility if no ensemble predictions available
            difficulty_scores = compute_volatility_difficulty(features)
        else:
            difficulty_scores = compute_ensemble_disagreement(predictions, checkpoint_paths)
    elif difficulty_metric == "prediction_confidence":
        if predictions is None or len(predictions) == 0:
            difficulty_scores = compute_volatility_difficulty(features)
        else:
            difficulty_scores = compute_prediction_confidence_difficulty(
                predictions[0] if isinstance(predictions, list) else predictions,
                uncertainties,
            )
    else:
        raise ValueError(f"Unknown difficulty metric: {difficulty_metric}")
    
    # Return indices sorted by difficulty (easiest first)
    return np.argsort(difficulty_scores)


def get_curriculum_schedule(
    epoch: int,
    total_samples: int,
    config: CurriculumConfig,
) -> tuple[int, int]:
    """Get current curriculum schedule (start and end indices).
    
    Args:
        epoch: Current training epoch
        total_samples: Total number of training samples
        config: Curriculum configuration
        
    Returns:
        Tuple of (start_idx, end_idx) for current curriculum stage
    """
    if not config.enabled or epoch < config.min_difficulty_epoch:
        # Use only easiest samples
        end_idx = int(total_samples * config.start_ratio)
        return 0, end_idx
    
    if epoch >= config.max_difficulty_epoch:
        # Use all samples
        return 0, total_samples
    
    # Gradually increase difficulty range
    progress = (epoch - config.min_difficulty_epoch) / (
        config.max_difficulty_epoch - config.min_difficulty_epoch
    )
    end_ratio = config.start_ratio + (1.0 - config.start_ratio) * progress
    end_idx = int(total_samples * end_ratio)
    
    return 0, end_idx


def create_curriculum_sampler_weights(
    difficulty_ranks: np.ndarray,
    epoch: int,
    config: CurriculumConfig,
) -> np.ndarray:
    """Create sampling weights for curriculum learning.
    
    Args:
        difficulty_ranks: Rank of each sample (0 = easiest, N-1 = hardest)
        epoch: Current training epoch
        config: Curriculum configuration
        
    Returns:
        Sampling weights (higher = more likely to be sampled)
    """
    if not config.enabled:
        return np.ones(len(difficulty_ranks))
    
    _, end_idx = get_curriculum_schedule(epoch, len(difficulty_ranks), config)
    
    # Create weights: easier samples get higher weight
    weights = np.ones(len(difficulty_ranks))
    
    # Only samples within current curriculum stage can be sampled
    weights[end_idx:] = 0.0
    
    # Gradually increase weight for harder samples within allowed range
    if end_idx > 0:
        # Exponential decay: easiest samples get highest weight
        decay = np.exp(-difficulty_ranks[:end_idx] / (end_idx * 0.5))
        weights[:end_idx] = decay
    
    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.sum()
    
    return weights

