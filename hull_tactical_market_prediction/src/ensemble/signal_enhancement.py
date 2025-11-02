"""Enhance signal quality to improve risk-adjusted returns."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class SignalEnhancer:
    """
    Enhance signal quality to improve risk-adjusted returns
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize signal enhancer.
        
        Args:
            confidence_threshold: Minimum confidence for signal (default: 0.6)
        """
        self.confidence_threshold = confidence_threshold

    def volatility_regime_filtering(
        self,
        predictions: pd.Series,
        volatility: pd.Series,
        high_vol_threshold: float = 0.02,
    ) -> pd.Series:
        """
        Reduce position sizes in high volatility regimes where signal quality degrades
        
        Args:
            predictions: Model predictions
            volatility: Volatility series
            high_vol_threshold: Threshold for high volatility (default: 0.02)
            
        Returns:
            Volatility-filtered predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(volatility.index)
        pred_aligned = predictions.reindex(common_idx)
        vol_aligned = volatility.reindex(common_idx)

        # Identify high volatility periods
        high_vol_mask = vol_aligned > high_vol_threshold

        # Reduce positions in high vol regimes
        filtered_predictions = pred_aligned.copy()
        filtered_predictions[high_vol_mask] = filtered_predictions[high_vol_mask] * 0.7

        return filtered_predictions

    def momentum_confirmation(
        self, predictions: pd.Series, price_trend: pd.Series, lookback: int = 5
    ) -> pd.Series:
        """
        Only take positions that align with short-term momentum
        
        Args:
            predictions: Model predictions
            price_trend: Price or trend series
            lookback: Lookback period for momentum (default: 5)
            
        Returns:
            Momentum-filtered predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(price_trend.index)
        pred_aligned = predictions.reindex(common_idx)
        trend_aligned = price_trend.reindex(common_idx)

        # Calculate short-term momentum
        momentum = trend_aligned.diff(lookback) / trend_aligned.shift(lookback)
        momentum = momentum.fillna(0)

        # Filter: only take positions where prediction aligns with momentum
        aligned_mask = (pred_aligned * momentum) > 0

        # Reduce position size for non-aligned signals
        enhanced_predictions = pred_aligned.copy()
        enhanced_predictions[~aligned_mask] = enhanced_predictions[~aligned_mask] * 0.5

        return enhanced_predictions

    def signal_confidence_weighting(
        self, predictions: pd.Series, model_confidence: pd.Series
    ) -> pd.Series:
        """
        Weight predictions by model confidence scores
        
        Args:
            predictions: Model predictions
            model_confidence: Confidence scores (0-1)
            
        Returns:
            Confidence-weighted predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(model_confidence.index)
        pred_aligned = predictions.reindex(common_idx)
        confidence_aligned = model_confidence.reindex(common_idx)

        # Ensure confidence is between 0 and 1
        confidence_clipped = np.clip(confidence_aligned, 0.1, 1.0)
        confidence_clipped = confidence_clipped.fillna(0.5)

        return pred_aligned * confidence_clipped

    def regime_conditional_boosting(
        self,
        predictions: pd.Series,
        regime_labels: pd.Series,
        regime_performance: Dict,
    ) -> pd.Series:
        """
        Boost signals in regimes where model performs best
        
        Args:
            predictions: Model predictions
            regime_labels: Regime label series
            regime_performance: Dictionary of regime -> performance metrics
            
        Returns:
            Regime-boosted predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(regime_labels.index)
        pred_aligned = predictions.reindex(common_idx)
        regime_aligned = regime_labels.reindex(common_idx)

        boosted_predictions = pred_aligned.copy()

        for regime, performance in regime_performance.items():
            regime_mask = regime_aligned == regime

            # Boost by regime-specific Sharpe ratio
            sharpe = performance.get('sharpe', 0)
            boost_factor = min(2.0, 1.0 + sharpe / 10)
            boosted_predictions[regime_mask] = boosted_predictions[regime_mask] * boost_factor

        return boosted_predictions

