"""Optimization utilities for adaptive ensemble."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .adaptive_ensemble import AdaptiveEnsemble
from .regime_detection import RegimeDetector


def optimize_regime_thresholds(
    regime_detector: RegimeDetector,
    features: pd.DataFrame,
    returns: Optional[pd.Series] = None,
    method: str = "percentile",
) -> Dict[str, float]:
    """Optimize regime detection thresholds based on actual market behavior.
    
    Args:
        regime_detector: Regime detector instance
        features: Feature dataframe
        returns: Optional return series for performance-based optimization
        method: Optimization method ('percentile' or 'statistical')
        
    Returns:
        Dictionary of optimized thresholds
    """
    thresholds = {}
    
    # Get feature groups
    vol_cols = [c for c in features.columns if c.startswith('V')]
    market_cols = [c for c in features.columns if c.startswith('M')]
    
    if method == "percentile":
        # Use percentile-based optimization
        if vol_cols:
            vol_df = features[vol_cols]
            vol_agg = vol_df.abs().mean(axis=1)
            # Optimize: use 75th percentile as baseline, but allow tuning
            thresholds['high_vol'] = float(vol_agg.quantile(0.75))
            thresholds['vol_percentile'] = 0.75
        
        if market_cols:
            market_df = features[market_cols]
            # Compute momentum strength
            momentum = market_df.rolling(window=21, min_periods=1).mean().diff().abs().mean(axis=1)
            thresholds['trending'] = float(momentum.quantile(0.60))
            thresholds['momentum_percentile'] = 0.60
    
    elif method == "statistical" and returns is not None:
        # Performance-based optimization: find thresholds that maximize Sharpe separation
        if vol_cols:
            vol_df = features[vol_cols]
            vol_agg = vol_df.abs().mean(axis=1)
            
            # Try different percentiles and find one that best separates performance
            best_threshold = None
            best_sharpe_diff = -np.inf
            
            for p in np.arange(0.6, 0.9, 0.05):
                threshold = vol_agg.quantile(p)
                high_vol_mask = vol_agg >= threshold
                
                if high_vol_mask.sum() > 10 and (~high_vol_mask).sum() > 10:
                    high_vol_sharpe = returns[high_vol_mask].mean() / (returns[high_vol_mask].std() + 1e-6)
                    normal_sharpe = returns[~high_vol_mask].mean() / (returns[~high_vol_mask].std() + 1e-6)
                    sharpe_diff = abs(high_vol_sharpe - normal_sharpe)
                    
                    if sharpe_diff > best_sharpe_diff:
                        best_sharpe_diff = sharpe_diff
                        best_threshold = threshold
            
            if best_threshold is not None:
                thresholds['high_vol'] = float(best_threshold)
    
    return thresholds


def calculate_dynamic_blend_ratio(
    ensemble_confidence: float,
    regime_stability: float,
    base_meta_weight: float = 0.7,
) -> float:
    """Calculate optimal meta-model vs adaptive blend ratio.
    
    Args:
        ensemble_confidence: Confidence in ensemble predictions (0-1)
        regime_stability: How stable the current regime is (0-1)
        base_meta_weight: Base weight for meta-model (default 0.7)
        
    Returns:
        Optimal meta-model weight (clamped to [0.5, 0.9])
    """
    # Increase meta weight when regime is stable and confidence high
    adjustment = regime_stability * ensemble_confidence * 0.2
    optimal_weight = base_meta_weight + adjustment
    
    # Clamp to reasonable bounds
    return float(np.clip(optimal_weight, 0.5, 0.9))


def analyze_regime_specialization(adaptive_ensemble: AdaptiveEnsemble) -> Dict[str, Dict]:
    """Comprehensive analysis of model performance by regime.
    
    Args:
        adaptive_ensemble: Adaptive ensemble instance
        
    Returns:
        Dictionary with regime specialization details per model
    """
    specialization = {}
    
    for model_id, diag in adaptive_ensemble.models.items():
        if diag.regime_correlation:
            # Calculate average Sharpe per regime
            regime_performance = {}
            for regime, sharpe_list in diag.regime_correlation.items():
                if sharpe_list:
                    regime_performance[regime] = {
                        'mean_sharpe': np.mean(sharpe_list),
                        'std_sharpe': np.std(sharpe_list),
                        'n_observations': len(sharpe_list),
                    }
            
            if regime_performance:
                # Find best regime
                best_regime = max(
                    regime_performance.items(),
                    key=lambda x: x[1]['mean_sharpe']
                )[0]
                
                specialization[model_id] = {
                    'best_regime': best_regime,
                    'regime_performance': regime_performance,
                    'overall_performance': diag.recent_performance,
                }
    
    return specialization


def get_fast_adaptation_config() -> Dict:
    """Get configuration for faster weight adaptation.
    
    Returns:
        Dictionary with optimized parameters for quick adaptation
    """
    return {
        'lookback_window': 21,  # Reduced from 63 for faster learning
        'min_active_periods': 10,  # Reduced from 21
        'sharpe_decay_factor': 0.90,  # Faster decay (was 0.95)
        'min_weight_floor': 0.05,
    }


def get_stable_adaptation_config() -> Dict:
    """Get configuration for stable, conservative adaptation.
    
    Returns:
        Dictionary with conservative parameters
    """
    return {
        'lookback_window': 126,  # Longer window for stability
        'min_active_periods': 42,  # More periods required
        'sharpe_decay_factor': 0.98,  # Slower decay
        'min_weight_floor': 0.10,  # Higher floor for stability
    }

