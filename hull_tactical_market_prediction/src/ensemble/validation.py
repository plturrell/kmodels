"""Validation utilities for ensemble predictions and submissions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def validate_submission(
    submission_path: Path | str,
    expected_samples: Optional[int] = None,
    id_column: str = "date_id",
    prediction_column: str = "prediction",
) -> Dict[str, bool]:
    """Validate submission file before Kaggle upload.
    
    Args:
        submission_path: Path to submission CSV
        expected_samples: Expected number of predictions (optional)
        id_column: ID column name
        prediction_column: Prediction column name
        
    Returns:
        Dictionary of validation results
    """
    results = {}
    
    submission_path = Path(submission_path)
    if not submission_path.exists():
        return {"file_exists": False, "error": f"File not found: {submission_path}"}
    
    try:
        df = pd.read_csv(submission_path)
        
        # Check required columns
        results["has_id_column"] = id_column in df.columns
        results["has_prediction_column"] = prediction_column in df.columns
        
        if not results["has_id_column"]:
            return {**results, "error": f"Missing required column: {id_column}"}
        if not results["has_prediction_column"]:
            return {**results, "error": f"Missing required column: {prediction_column}"}
        
        # Check sample count
        results["sample_count"] = len(df)
        if expected_samples is not None:
            results["correct_count"] = len(df) == expected_samples
            if not results["correct_count"]:
                results["error"] = f"Expected {expected_samples} predictions, got {len(df)}"
        
        # Check for missing values
        results["no_missing_ids"] = df[id_column].isna().sum() == 0
        results["no_missing_predictions"] = df[prediction_column].isna().sum() == 0
        
        if not results["no_missing_predictions"]:
            results["error"] = f"Found {df[prediction_column].isna().sum()} missing predictions"
        
        # Check for duplicates
        results["no_duplicate_ids"] = df[id_column].duplicated().sum() == 0
        
        # Check prediction variance
        predictions = df[prediction_column]
        results["has_variance"] = predictions.std() > 0
        results["prediction_std"] = float(predictions.std())
        results["prediction_mean"] = float(predictions.mean())
        results["prediction_range"] = [float(predictions.min()), float(predictions.max())]
        
        # Check for infinite values
        results["no_infinite"] = np.isfinite(predictions).all()
        
        # Check for reasonable range (warn if extreme)
        results["reasonable_range"] = (
            predictions.abs().max() < 1.0 and  # Not too extreme
            predictions.abs().min() >= -1.0
        )
        
        # Overall validation
        results["valid"] = all([
            results.get("has_id_column", False),
            results.get("has_prediction_column", False),
            results.get("no_missing_predictions", False),
            results.get("no_duplicate_ids", False),
            results.get("has_variance", False),
            results.get("no_infinite", False),
        ])
        
    except Exception as e:
        results["valid"] = False
        results["error"] = str(e)
    
    return results


def analyze_model_diversity(
    predictions_dict: Dict[str, pd.Series | np.ndarray],
    method: str = "spearman",
) -> Dict[str, float]:
    """Analyze diversity between model predictions.
    
    Ensures models aren't just copying each other - true ensemble benefit.
    
    Args:
        predictions_dict: Dictionary mapping model_id to predictions
        method: Correlation method ('spearman' or 'pearson')
        
    Returns:
        Dictionary of diversity scores (1.0 = perfectly diverse, 0.0 = identical)
    """
    diversity_scores = {}
    models = list(predictions_dict.keys())
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:
                pred1 = predictions_dict[model1]
                pred2 = predictions_dict[model2]
                
                # Ensure same length
                min_len = min(len(pred1), len(pred2))
                pred1_aligned = pred1[:min_len] if isinstance(pred1, pd.Series) else pred1[:min_len]
                pred2_aligned = pred2[:min_len] if isinstance(pred2, pd.Series) else pred2[:min_len]
                
                if method == "spearman":
                    corr, _ = spearmanr(pred1_aligned, pred2_aligned)
                else:
                    corr = np.corrcoef(pred1_aligned, pred2_aligned)[0, 1]
                
                # Diversity = 1 - absolute correlation
                # 1.0 = perfectly diverse (uncorrelated)
                # 0.0 = identical predictions
                diversity = 1.0 - abs(corr) if not np.isnan(corr) else 0.5
                diversity_scores[f"{model1}_{model2}"] = float(diversity)
    
    return diversity_scores


def calculate_submission_confidence(
    adaptive_ensemble,
    current_regime: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Calculate confidence score for submission based on ensemble state.
    
    Args:
        adaptive_ensemble: Adaptive ensemble instance
        current_regime: Current market regime
        weights: Pre-calculated weights (optional)
        
    Returns:
        Dictionary with confidence metrics
    """
    if weights is None:
        weights = adaptive_ensemble.calculate_adaptive_weights(current_regime=current_regime)
    
    if not weights:
        return {"confidence_score": 0.0, "reason": "No models registered"}
    
    # Best model weight (higher = more confidence)
    best_model_weight = max(weights.values())
    
    # Weight concentration (how much weight is in top models)
    sorted_weights = sorted(weights.values(), reverse=True)
    top2_concentration = sum(sorted_weights[:2]) if len(sorted_weights) >= 2 else sorted_weights[0]
    
    # Model count (more models = more robust)
    model_count_factor = min(len(weights) / 5.0, 1.0)  # Normalize to max 5 models
    
    # Get model statistics for additional confidence factors
    stats = adaptive_ensemble.get_model_stats()
    if len(stats) > 0:
        avg_sharpe = stats['recent_sharpe'].mean()
        sharpe_factor = min(avg_sharpe / 2.0, 1.0)  # Normalize (assume max Sharpe ~2)
    else:
        sharpe_factor = 0.5
    
    # Calculate overall confidence (weighted combination)
    confidence_score = (
        best_model_weight * 0.3 +  # Best model strength
        top2_concentration * 0.3 +  # Ensemble concentration
        model_count_factor * 0.2 +  # Robustness
        sharpe_factor * 0.2  # Performance quality
    )
    
    return {
        "confidence_score": float(confidence_score),
        "best_model_weight": float(best_model_weight),
        "top2_concentration": float(top2_concentration),
        "model_count": len(weights),
        "avg_sharpe": float(avg_sharpe) if len(stats) > 0 else 0.0,
        "regime": current_regime or "unknown",
    }


def calculate_regime_stability(
    regime_detector,
    recent_features: pd.DataFrame,
    window: int = 21,
) -> float:
    """Calculate how stable the current regime is.
    
    Args:
        regime_detector: Regime detector instance
        recent_features: Recent feature dataframe
        window: Window size for stability calculation
        
    Returns:
        Stability score (0-1, higher = more stable)
    """
    if len(recent_features) < 2:
        return 0.5  # Default moderate stability
    
    # Detect regimes for recent period
    recent_regimes = regime_detector.detect_regime(recent_features.tail(window))
    
    if len(recent_regimes) == 0:
        return 0.5
    
    # Stability = proportion of most common regime
    regime_counts = recent_regimes.value_counts()
    if len(regime_counts) > 0:
        most_common_count = regime_counts.iloc[0]
        stability = most_common_count / len(recent_regimes)
    else:
        stability = 0.5
    
    return float(stability)

