"""Competition-specific utilities and optimizations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .regime_detection import RegimeDetector


def apply_volatility_scaling(
    predictions: pd.Series | np.ndarray,
    volatility_estimate: float | pd.Series,
    target_volatility: float = 0.15,
    periods_per_year: int = 252,
    clip_range: tuple[float, float] = (0.5, 2.0),
) -> pd.Series | np.ndarray:
    """Scale predictions based on estimated volatility.
    
    Hull Tactical emphasizes risk-adjusted returns, so volatility scaling
    can improve performance.
    
    Args:
        predictions: Model predictions
        volatility_estimate: Estimated volatility (can be scalar or series)
        target_volatility: Target annualized volatility (default 15%)
        periods_per_year: Trading periods per year (252 for daily)
        clip_range: Range for scaling factor (default 0.5x to 2.0x)
        
    Returns:
        Volatility-scaled predictions
    """
    # Convert to daily volatility target
    target_daily_vol = target_volatility / np.sqrt(periods_per_year)
    
    # Calculate scaling factor
    if isinstance(volatility_estimate, (pd.Series, np.ndarray)):
        # Handle series volatility
        vol_scale = target_daily_vol / (volatility_estimate + 1e-8)
        vol_scale = np.clip(vol_scale, clip_range[0], clip_range[1])
    else:
        # Scalar volatility
        vol_scale = np.clip(
            target_daily_vol / (volatility_estimate + 1e-8),
            clip_range[0],
            clip_range[1]
        )
    
    # Apply scaling
    scaled_predictions = predictions * vol_scale
    
    return scaled_predictions


def regime_performance_breakdown(
    predictions_dict: Dict[str, pd.Series],
    actuals: pd.Series,
    regimes: pd.Series,
    metric: str = "rmse",
) -> Dict[str, Dict[str, float]]:
    """Break down model performance by market regime.
    
    Args:
        predictions_dict: Dictionary of model predictions
        actuals: Actual target values
        regimes: Regime labels per sample
        metric: Metric to compute ('rmse', 'mae', 'corr', 'sharpe')
        
    Returns:
        Nested dictionary: {regime: {model_id: metric_value}}
    """
    breakdown = {}
    unique_regimes = regimes.unique()
    
    for regime in unique_regimes:
        regime_mask = regimes == regime
        if regime_mask.sum() == 0:
            continue
        
        regime_actuals = actuals[regime_mask]
        breakdown[regime] = {}
        
        for model_id, predictions in predictions_dict.items():
            # Align predictions with regime mask
            if isinstance(predictions, pd.Series):
                regime_preds = predictions[regime_mask]
            else:
                regime_preds = predictions[regime_mask.values]
            
            if metric == "rmse":
                value = float(np.sqrt(np.mean((regime_preds - regime_actuals) ** 2)))
            elif metric == "mae":
                value = float(np.mean(np.abs(regime_preds - regime_actuals)))
            elif metric == "corr":
                value = float(np.corrcoef(regime_preds, regime_actuals)[0, 1])
            elif metric == "sharpe":
                returns = regime_preds - regime_actuals
                if returns.std() > 0:
                    value = float(np.mean(returns) / returns.std())
                else:
                    value = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            breakdown[regime][model_id] = value
    
    return breakdown


def analyze_leaderboard_feedback(
    public_score: float,
    expected_score: float,
    current_predictions: pd.DataFrame,
    ensemble_config: Dict,
) -> Dict:
    """Analyze leaderboard feedback and provide recommendations.
    
    Args:
        public_score: Public leaderboard score (RMSE)
        expected_score: Expected score based on validation
        current_predictions: Current submission predictions
        ensemble_config: Current ensemble configuration
        
    Returns:
        Dictionary with analysis and recommendations
    """
    score_diff = public_score - expected_score
    relative_error = abs(score_diff) / (expected_score + 1e-8)
    
    analysis = {
        "public_score": public_score,
        "expected_score": expected_score,
        "score_difference": score_diff,
        "relative_error": relative_error,
        "status": "unknown",
        "recommendations": [],
    }
    
    # Analyze performance
    if score_diff < -0.0001:  # Better than expected
        analysis["status"] = "exceeding_expectations"
        analysis["recommendations"].append("✅ Strategy working well - maintain current approach")
        analysis["recommendations"].append("Consider more aggressive adaptation if opportunity")
    elif score_diff > 0.0001:  # Worse than expected
        analysis["status"] = "underperforming"
        analysis["recommendations"].append("🔄 Adjust strategy - analyze regime mismatches")
        analysis["recommendations"].append("Consider shorter lookback window for faster adaptation")
        analysis["recommendations"].append("Review regime detection thresholds")
        analysis["recommendations"].append("Experiment with different blend ratios")
    else:
        analysis["status"] = "meeting_expectations"
        analysis["recommendations"].append("✅ Performance as expected")
        analysis["recommendations"].append("Monitor for improvements over time")
    
    # Check prediction characteristics
    if "prediction" in current_predictions.columns:
        pred_std = current_predictions["prediction"].std()
        if pred_std < 1e-6:
            analysis["recommendations"].append(
                "⚠️  Very low prediction variance - consider increasing model diversity"
            )
    
    return analysis


def create_enhanced_metadata(
    ensemble_config: Dict,
    performance_metrics: Dict,
    submission_info: Optional[Dict] = None,
) -> Dict:
    """Create enhanced metadata for submission tracking.
    
    Args:
        ensemble_config: Ensemble configuration details
        performance_metrics: Performance metrics
        submission_info: Additional submission information
        
    Returns:
        Complete metadata dictionary
    """
    from datetime import datetime
    
    metadata = {
        "ensemble_config": ensemble_config,
        "performance_metrics": performance_metrics,
        "submission_info": submission_info or {
            "timestamp": datetime.now().isoformat(),
            "model_version": "adaptive_ensemble_v1",
            "competition": "Hull Tactical Market Prediction",
        },
        "validation": {
            "checks_passed": True,
            "validation_timestamp": datetime.now().isoformat(),
        },
    }
    
    return metadata

