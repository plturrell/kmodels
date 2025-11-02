"""Advanced risk management techniques to improve Sharpe ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdvancedRiskManager:
    """
    Advanced risk management techniques to improve Sharpe ratio
    """

    def __init__(self, target_correlation: float = 0.3):
        """
        Initialize risk manager.
        
        Args:
            target_correlation: Target correlation with market (default: 0.3)
        """
        self.target_correlation = target_correlation

    def correlation_penalty(
        self, predictions: pd.Series, market_correlation: pd.Series
    ) -> pd.Series:
        """
        Penalize predictions that are highly correlated with market moves
        to capture true alpha, not beta
        
        Args:
            predictions: Model predictions
            market_correlation: Rolling correlation with market
            
        Returns:
            Correlation-penalized predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(market_correlation.index)
        pred_aligned = predictions.reindex(common_idx)
        corr_aligned = market_correlation.reindex(common_idx)

        # Calculate penalty factor based on correlation
        correlation_penalty = 1 - np.abs(corr_aligned)

        # Apply minimum penalty to avoid zero positions
        correlation_penalty = np.clip(correlation_penalty, 0.3, 1.0)
        correlation_penalty = correlation_penalty.fillna(0.7)

        return pred_aligned * correlation_penalty

    def var_based_position_limits(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        var_percentile: float = 0.05,
    ) -> pd.Series:
        """
        Use Value at Risk to limit position sizes
        
        Args:
            predictions: Model predictions
            returns: Historical returns
            var_percentile: VaR percentile (default: 0.05 for 5%)
            
        Returns:
            VaR-limited predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(returns.index)
        pred_aligned = predictions.reindex(common_idx)
        returns_aligned = returns.reindex(common_idx)

        # Calculate rolling VaR
        rolling_var = returns_aligned.rolling(window=63, min_periods=10).quantile(
            var_percentile
        )

        # Reduce positions when VaR is extreme
        var_zscore = (rolling_var - rolling_var.mean()) / (rolling_var.std() + 1e-8)
        var_zscore = var_zscore.fillna(0)

        # Scaling based on VaR z-score
        var_scale = np.where(
            var_zscore < -2,
            0.5,  # Severe risk - cut positions
            np.where(
                var_zscore < -1,
                0.8,  # Elevated risk - reduce positions
                1.0,  # Normal risk
            ),
        )

        return pred_aligned * var_scale

    def tail_risk_hedging(
        self,
        predictions: pd.Series,
        volatility_skew: pd.Series,
        hedge_ratio: float = 0.1,
    ) -> pd.Series:
        """
        Implicit tail risk hedging by reducing positions when tail risk is high
        
        Args:
            predictions: Model predictions
            volatility_skew: Volatility skew indicator (proxy for tail risk)
            hedge_ratio: Hedging ratio (default: 0.1)
            
        Returns:
            Tail-risk-hedged predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(volatility_skew.index)
        pred_aligned = predictions.reindex(common_idx)
        skew_aligned = volatility_skew.reindex(common_idx)

        # Use volatility skew as tail risk proxy
        tail_risk_indicator = skew_aligned.rolling(window=21, min_periods=5).mean()

        # Normalize and create hedge factor
        tail_risk_mean = tail_risk_indicator.mean()
        tail_risk_std = tail_risk_indicator.std()
        if tail_risk_std > 0:
            tail_risk_z = (tail_risk_indicator - tail_risk_mean) / tail_risk_std
        else:
            tail_risk_z = pd.Series(0, index=tail_risk_indicator.index)

        tail_risk_z = tail_risk_z.fillna(0)
        hedge_factor = 1 - (hedge_ratio * np.clip(tail_risk_z, 0, 3) / 3)

        return pred_aligned * hedge_factor

