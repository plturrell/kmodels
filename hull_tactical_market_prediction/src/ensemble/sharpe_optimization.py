"""Direct optimization of ensemble weights for maximum Sharpe ratio."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class SharpeOptimizer:
    """
    Direct optimization of ensemble weights for maximum Sharpe ratio
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize Sharpe optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation (default: 0.0)
        """
        self.risk_free_rate = risk_free_rate

    def optimize_ensemble_weights(
        self, model_returns: Dict[str, pd.Series], lookback: int = 126
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights to maximize Sharpe ratio over lookback period
        
        Args:
            model_returns: Dictionary of model_id -> return series
            lookback: Lookback period for optimization (default: 126 days)
            
        Returns:
            Dictionary of optimized weights
        """
        # Align return series
        returns_df = pd.DataFrame(model_returns).dropna()

        if len(returns_df) < lookback:
            lookback = len(returns_df)

        if len(returns_df) == 0:
            # Return equal weights if no data
            return {model: 1 / len(model_returns) for model in model_returns.keys()}

        # Use recent data for optimization
        recent_returns = returns_df.iloc[-lookback:]

        def negative_sharpe(weights):
            """Negative Sharpe ratio for minimization"""
            portfolio_returns = recent_returns.dot(weights)
            excess_returns = portfolio_returns - self.risk_free_rate
            std = excess_returns.std()
            if std == 0:
                return 1e10  # Penalize zero volatility
            sharpe = excess_returns.mean() / std * np.sqrt(252)
            return -sharpe

        # Constraints: weights sum to 1, positive weights
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0.05, 1.0) for _ in range(len(model_returns))]  # Minimum 5% per model

        # Initial guess (equal weights)
        initial_weights = np.ones(len(model_returns)) / len(model_returns)

        # Optimize
        try:
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                optimized_weights = dict(zip(model_returns.keys(), result.x))
            else:
                # Fallback to equal weights
                optimized_weights = {
                    model: 1 / len(model_returns) for model in model_returns.keys()
                }
        except Exception:
            # Fallback to equal weights on error
            optimized_weights = {
                model: 1 / len(model_returns) for model in model_returns.keys()
            }

        return optimized_weights

    def rolling_sharpe_optimization(
        self,
        model_returns: Dict[str, pd.Series],
        optimization_window: int = 63,
        update_frequency: int = 21,
    ) -> pd.DataFrame:
        """
        Rolling optimization of ensemble weights
        
        Args:
            model_returns: Dictionary of model_id -> return series
            optimization_window: Window for optimization (default: 63 days)
            update_frequency: How often to update weights (default: 21 days)
            
        Returns:
            DataFrame of optimized weights over time
        """
        returns_df = pd.DataFrame(model_returns).dropna()
        dates = returns_df.index
        optimized_weights = []

        for i in range(optimization_window, len(returns_df), update_frequency):
            window_returns = returns_df.iloc[i - optimization_window : i]

            # Optimize weights for this window
            window_weights = self._optimize_window_weights(window_returns)
            optimized_weights.extend(
                [window_weights] * min(update_frequency, len(returns_df) - i)
            )

        # Create weights DataFrame
        if len(optimized_weights) < len(returns_df):
            # Pad with last optimized weights
            last_weights = (
                optimized_weights[-1]
                if optimized_weights
                else {model: 1 / len(model_returns) for model in model_returns.keys()}
            )
            optimized_weights.extend([last_weights] * (len(returns_df) - len(optimized_weights)))

        weights_df = pd.DataFrame(
            optimized_weights, index=dates[: len(optimized_weights)]
        )
        return weights_df

    def _optimize_window_weights(self, window_returns: pd.DataFrame) -> Dict[str, float]:
        """Optimize weights for a single window"""

        def negative_sharpe(weights):
            portfolio_returns = window_returns.dot(weights)
            std = portfolio_returns.std()
            if std == 0:
                return 1e10
            sharpe = portfolio_returns.mean() / std * np.sqrt(252)
            return -sharpe

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0.05, 1.0) for _ in range(window_returns.shape[1])]
        initial_weights = np.ones(window_returns.shape[1]) / window_returns.shape[1]

        try:
            result = minimize(
                negative_sharpe,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                return dict(zip(window_returns.columns, result.x))
            else:
                return {
                    model: 1 / window_returns.shape[1] for model in window_returns.columns
                }
        except Exception:
            return {
                model: 1 / window_returns.shape[1] for model in window_returns.columns
            }

