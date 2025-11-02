"""Financial-specific learning tests for market prediction models."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from ..utils.financial_metrics import sharpe_ratio
except ImportError:
    # Fallback implementation
    def sharpe_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


class FinancialLearningAssessment:
    """Financial-specific learning tests for market prediction models."""

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        """Initialize financial learning assessment.
        
        Args:
            risk_free_rate: Risk-free rate (daily, default 0.0)
            periods_per_year: Number of periods per year (default 252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def regime_robustness_test(
        self,
        predictions: Dict[str, pd.Series],
        actual_returns: pd.Series,
        regime_labels: pd.Series,
    ) -> Dict:
        """Test if model maintains performance across different market regimes.
        
        True learning = consistent performance across regimes
        Memorization = regime-specific performance drops
        
        Args:
            predictions: Dictionary of model_id -> prediction series
            actual_returns: Actual return series
            regime_labels: Series of regime labels (e.g., 'high_vol', 'trending')
            
        Returns:
            Dictionary with regime robustness metrics
        """
        regime_performance = {}
        min_samples = 10  # Minimum samples per regime

        # Align indices
        common_index = actual_returns.index.intersection(regime_labels.index)
        if len(common_index) == 0:
            return {
                'regime_performance': {},
                'consistency_scores': {},
                'overall_consistency': 0.0,
                'is_learning': False,
                'note': 'No common indices between returns and regime labels',
            }

        actual_aligned = actual_returns.reindex(common_index)
        regime_aligned = regime_labels.reindex(common_index)

        for regime in regime_aligned.dropna().unique():
            regime_mask = regime_aligned == regime
            if regime_mask.sum() >= min_samples:
                regime_actual = actual_aligned[regime_mask]

                # Calculate regime-specific Sharpe ratios for each model
                regime_sharpes = {}
                for model_id, pred_series in predictions.items():
                    # Align predictions to common index
                    pred_aligned = pred_series.reindex(common_index)
                    if pred_aligned.isna().sum() < len(pred_aligned) * 0.5:  # At least 50% non-NA
                        regime_preds = pred_aligned[regime_mask].dropna()
                        
                        if len(regime_preds) >= min_samples:
                            # Calculate strategy returns: sign(prediction) * actual_return
                            strategy_returns = np.sign(regime_preds.values) * regime_actual.reindex(
                                regime_preds.index
                            ).values
                            
                            if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
                                sharpe = sharpe_ratio(
                                    pd.Series(strategy_returns),
                                    self.risk_free_rate,
                                    self.periods_per_year,
                                )
                                regime_sharpes[model_id] = sharpe

                if regime_sharpes:
                    regime_performance[regime] = regime_sharpes

        # Calculate performance consistency across regimes
        consistency_scores = {}
        for model_id in predictions.keys():
            model_sharpes = [
                perf.get(model_id, np.nan) for perf in regime_performance.values()
            ]
            model_sharpes = [s for s in model_sharpes if not np.isnan(s)]
            
            if len(model_sharpes) > 1:
                mean_sharpe = np.mean(model_sharpes)
                std_sharpe = np.std(model_sharpes)
                
                # Consistency: 1 - (coefficient of variation)
                if abs(mean_sharpe) > 1e-8:
                    consistency = 1 - (std_sharpe / abs(mean_sharpe))
                else:
                    consistency = 0.0
                
                consistency_scores[model_id] = max(0.0, min(1.0, consistency))
            elif len(model_sharpes) == 1:
                consistency_scores[model_id] = 1.0  # Single regime, perfect consistency
            else:
                consistency_scores[model_id] = 0.0

        overall_consistency = (
            float(np.mean(list(consistency_scores.values())))
            if consistency_scores
            else 0.0
        )

        return {
            'regime_performance': {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in regime_performance.items()
            },
            'consistency_scores': {k: float(v) for k, v in consistency_scores.items()},
            'overall_consistency': overall_consistency,
            'is_learning': overall_consistency > 0.6,
            'num_regimes': len(regime_performance),
        }

    def alpha_persistence_test(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
        lookback_periods: List[int] = [21, 63, 126],
    ) -> Dict:
        """Test if model alpha persists over different time horizons.
        
        True learning = persistent alpha across time horizons
        Overfitting = alpha decays rapidly
        
        Args:
            predictions: Prediction series
            actual_returns: Actual return series
            lookback_periods: List of lookback periods in days
            
        Returns:
            Dictionary with alpha persistence metrics
        """
        # Align indices
        common_index = predictions.index.intersection(actual_returns.index)
        if len(common_index) == 0:
            return {
                'alpha_persistence': {},
                'persistence_score': 0.0,
                'is_learning': False,
                'note': 'No common indices',
            }

        pred_aligned = predictions.reindex(common_index).dropna()
        actual_aligned = actual_returns.reindex(common_index)

        alpha_persistence = {}

        for period in lookback_periods:
            if period > len(pred_aligned):
                continue

            rolling_alphas = []

            for i in range(period, len(pred_aligned)):
                window_preds = pred_aligned.iloc[i - period : i]
                window_actual = actual_aligned.reindex(window_preds.index)

                if len(window_preds) == period and window_actual.notna().sum() >= period * 0.8:
                    # Calculate window alpha (excess returns over market mean)
                    strategy_returns = np.sign(window_preds.values) * window_actual.dropna().values[
                        : len(window_preds)
                    ]

                    if len(strategy_returns) > 0:
                        strategy_cumulative = np.sum(strategy_returns)
                        market_cumulative = window_actual.dropna().values[: len(window_preds)].sum()
                        alpha = strategy_cumulative - market_cumulative
                        rolling_alphas.append(alpha)

            if rolling_alphas:
                alpha_persistence[f'{period}_day'] = {
                    'mean_alpha': float(np.mean(rolling_alphas)),
                    'alpha_std': float(np.std(rolling_alphas)),
                    'positive_alpha_ratio': float(np.mean(np.array(rolling_alphas) > 0)),
                    'num_windows': len(rolling_alphas),
                }

        # Calculate persistence score
        positive_ratios = [v['positive_alpha_ratio'] for v in alpha_persistence.values()]
        persistence_score = float(np.mean(positive_ratios)) if positive_ratios else 0.0

        return {
            'alpha_persistence': alpha_persistence,
            'persistence_score': persistence_score,
            'is_learning': persistence_score > 0.6,  # Consistent positive alpha
        }

    def capacity_robustness_test(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
        capacity_factors: List[float] = [0.5, 0.8, 0.9],
    ) -> Dict:
        """Test if strategy degrades gracefully with capacity constraints.
        
        True learning = robust to capacity constraints
        Overfitting = severe degradation with capacity limits
        
        Args:
            predictions: Prediction series
            actual_returns: Actual return series
            capacity_factors: List of capacity factors (0-1)
            
        Returns:
            Dictionary with capacity robustness metrics
        """
        # Align indices
        common_index = predictions.index.intersection(actual_returns.index)
        if len(common_index) == 0:
            return {
                'capacity_robustness': {},
                'robustness_score': 0.0,
                'is_learning': False,
                'note': 'No common indices',
            }

        pred_aligned = predictions.reindex(common_index).dropna()
        actual_aligned = actual_returns.reindex(common_index)

        # Calculate base Sharpe
        strategy_returns_base = np.sign(pred_aligned.values) * actual_aligned.reindex(
            pred_aligned.index
        ).values
        base_sharpe = sharpe_ratio(
            pd.Series(strategy_returns_base).dropna(),
            self.risk_free_rate,
            self.periods_per_year,
        )

        capacity_ratios = {}
        for capacity in capacity_factors:
            # Simulate capacity constraints by scaling predictions
            constrained_preds = pred_aligned * capacity
            constrained_returns = (
                np.sign(constrained_preds.values)
                * actual_aligned.reindex(constrained_preds.index).values
            )

            constrained_sharpe = sharpe_ratio(
                pd.Series(constrained_returns).dropna(),
                self.risk_free_rate,
                self.periods_per_year,
            )

            if abs(base_sharpe) > 1e-8:
                degradation = (base_sharpe - constrained_sharpe) / abs(base_sharpe)
            else:
                degradation = 1.0 if constrained_sharpe < base_sharpe else 0.0

            capacity_ratios[f'capacity_{int(capacity*100)}%'] = {
                'sharpe': float(constrained_sharpe),
                'degradation': float(degradation),
            }

        # Calculate robustness score (lower degradation = more robust)
        degradations = [abs(v['degradation']) for v in capacity_ratios.values()]
        robustness_score = float(1.0 - min(1.0, np.mean(degradations))) if degradations else 0.0

        return {
            'base_sharpe': float(base_sharpe),
            'capacity_robustness': capacity_ratios,
            'robustness_score': robustness_score,
            'is_learning': robustness_score > 0.7,  # Graceful degradation
        }

