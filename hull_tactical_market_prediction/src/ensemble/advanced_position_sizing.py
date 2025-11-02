"""Advanced position sizing to maximize Sharpe ratio."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class AdvancedPositionSizer:
    """
    Advanced position sizing to maximize Sharpe ratio
    """

    def __init__(self, target_volatility: float = 0.15, max_leverage: float = 2.0):
        """
        Initialize position sizer.
        
        Args:
            target_volatility: Target annualized volatility (default: 15%)
            max_leverage: Maximum leverage multiplier (default: 2.0)
        """
        self.target_vol = target_volatility / np.sqrt(252)  # Annual to daily
        self.max_leverage = max_leverage

    def volatility_scaling(
        self, predictions: pd.Series, volatility_estimate: pd.Series
    ) -> pd.Series:
        """
        Scale positions based on predicted volatility to maintain target risk
        
        Args:
            predictions: Model predictions
            volatility_estimate: Estimated volatility series
            
        Returns:
            Volatility-scaled predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(volatility_estimate.index)
        pred_aligned = predictions.reindex(common_idx)
        vol_aligned = volatility_estimate.reindex(common_idx)

        # Avoid division by zero
        vol_scaled = vol_aligned.replace(0, self.target_vol)
        vol_scaled = vol_scaled.fillna(self.target_vol)

        # Calculate scaling factor
        scale_factor = self.target_vol / vol_scaled

        # Apply leverage constraints
        scale_factor = np.clip(scale_factor, 0.1, self.max_leverage)

        return pred_aligned * scale_factor

    def kelly_position_sizing(
        self,
        predictions: pd.Series,
        win_rate: float,
        avg_win_loss_ratio: float,
    ) -> pd.Series:
        """
        Apply Kelly Criterion for optimal position sizing
        
        Args:
            predictions: Model predictions
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            
        Returns:
            Kelly-scaled predictions
        """
        # Kelly fraction: f* = p - (1-p)/b
        # where p = win rate, b = win/loss ratio
        kelly_fraction = win_rate - (1 - win_rate) / avg_win_loss_ratio

        # Conservative Kelly (half Kelly)
        conservative_kelly = kelly_fraction * 0.5

        # Apply to predictions
        kelly_scalar = np.clip(conservative_kelly, 0.01, 0.2)

        return predictions * kelly_scalar

    def dynamic_volatility_targeting(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        lookback: int = 21,
    ) -> pd.Series:
        """
        Dynamic volatility targeting based on recent market conditions
        
        Args:
            predictions: Model predictions
            returns: Historical returns
            lookback: Lookback window for volatility estimation
            
        Returns:
            Dynamically scaled predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(returns.index)
        pred_aligned = predictions.reindex(common_idx)
        returns_aligned = returns.reindex(common_idx)

        rolling_vol = returns_aligned.rolling(window=lookback, min_periods=5).std()

        # Adaptive target volatility - increase in low vol regimes, decrease in high vol
        vol_regime = rolling_vol.rolling(window=63, min_periods=10).rank(pct=True)

        # Regime-based target adjustment
        adaptive_target = self.target_vol * (1.2 - 0.4 * vol_regime)  # 0.8x to 1.2x target

        scale_factors = adaptive_target / (rolling_vol + 1e-8)
        scale_factors = np.clip(scale_factors, 0.5, 2.0)
        scale_factors = scale_factors.fillna(1.0)

        return pred_aligned * scale_factors

    def drawdown_aware_sizing(
        self,
        predictions: pd.Series,
        portfolio_value: pd.Series,
        max_drawdown_limit: float = 0.10,
    ) -> pd.Series:
        """
        Reduce position sizes during drawdowns to protect capital
        
        Args:
            predictions: Model predictions
            portfolio_value: Portfolio value series
            max_drawdown_limit: Maximum drawdown threshold (default: 10%)
            
        Returns:
            Drawdown-aware scaled predictions
        """
        # Align indices
        common_idx = predictions.index.intersection(portfolio_value.index)
        pred_aligned = predictions.reindex(common_idx)
        portfolio_aligned = portfolio_value.reindex(common_idx)

        # Calculate running max and current drawdown
        running_max = portfolio_aligned.expanding().max()
        current_drawdown = (portfolio_aligned - running_max) / running_max

        # Drawdown-based scaling
        drawdown_penalty = np.where(
            current_drawdown < -max_drawdown_limit,
            0.5,  # Cut positions by 50% during severe drawdowns
            1.0,  # Normal sizing otherwise
        )

        return pred_aligned * drawdown_penalty

