"""Extreme stress tests that push models beyond normal operating conditions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import warnings

try:
    from ..utils.financial_metrics import sharpe_ratio
except ImportError:
    def sharpe_ratio(returns, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        if isinstance(returns, pd.Series):
            returns = returns.values
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


class ExtremeStressTester:
    """Extreme stress tests that push models beyond normal operating conditions."""

    def __init__(self, confidence_level: float = 0.99):
        """Initialize stress tester.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level

    def adversarial_regime_shock_test(
        self,
        predictions: Dict[str, pd.Series],
        actual_returns: pd.Series,
        shock_scenarios: List[Dict],
    ) -> Dict:
        """Test model resilience against extreme regime shocks.
        
        Tests against:
        - Flash crashes
        - Liquidity crises
        - Volatility explosions
        - Correlation breakdowns
        
        Args:
            predictions: Dictionary of model_id -> prediction series
            actual_returns: Actual return series
            shock_scenarios: List of shock scenario configurations
            
        Returns:
            Dictionary with shock test results
        """
        shock_results = {}

        for scenario in shock_scenarios:
            scenario_name = scenario['name']
            shock_mask = self._create_shock_mask(actual_returns.index, scenario)

            if shock_mask.sum() > 0:
                shock_returns = actual_returns[shock_mask]
                shock_performance = {}

                for model_id, pred_series in predictions.items():
                    aligned_preds = pred_series.reindex(actual_returns.index)
                    shock_preds = aligned_preds[shock_mask].dropna()

                    if len(shock_preds) > 0:
                        # Align shock returns with predictions
                        common_idx = shock_preds.index.intersection(shock_returns.index)
                        if len(common_idx) > 0:
                            shock_preds_aligned = shock_preds.reindex(common_idx)
                            shock_returns_aligned = shock_returns.reindex(common_idx)

                            # Calculate shock performance
                            shock_strategy_returns = (
                                np.sign(shock_preds_aligned.values) * shock_returns_aligned.values
                            )

                            if len(shock_strategy_returns) > 0 and np.std(shock_strategy_returns) > 0:
                                shock_sharpe = sharpe_ratio(
                                    pd.Series(shock_strategy_returns),
                                    periods_per_year=252,
                                )
                                max_dd = self._calculate_max_drawdown(pd.Series(shock_strategy_returns))

                                shock_performance[model_id] = {
                                    'sharpe': float(shock_sharpe),
                                    'max_drawdown': float(max_dd),
                                    'hit_rate': float((shock_strategy_returns > 0).mean()),
                                    'volatility': float(np.std(shock_strategy_returns)),
                                    'samples': len(shock_strategy_returns),
                                }

                if shock_performance:
                    shock_results[scenario_name] = shock_performance

        # Calculate shock resilience scores
        resilience_scores = {}
        for model_id in predictions.keys():
            model_shock_scores = []
            for scenario, performance in shock_results.items():
                if model_id in performance:
                    model_perf = performance[model_id]
                    # Score based on Sharpe and drawdown during shocks
                    sharpe_score = max(0, model_perf['sharpe'])
                    dd_penalty = min(1, abs(model_perf['max_drawdown']))
                    shock_score = sharpe_score * (1 - dd_penalty)
                    model_shock_scores.append(shock_score)

            if model_shock_scores:
                resilience_scores[model_id] = float(np.mean(model_shock_scores))

        overall_resilience = (
            float(np.mean(list(resilience_scores.values()))) if resilience_scores else 0.0
        )

        return {
            'shock_results': shock_results,
            'resilience_scores': resilience_scores,
            'overall_resilience': overall_resilience,
            'passed_stress_test': overall_resilience > 0.3,
        }

    def multi_scale_consistency_test(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
        time_scales: List[str] = ['1D', '1W', '1M', '3M'],
    ) -> Dict:
        """Test if learning is consistent across multiple time scales.
        
        True learning should work across scales, overfitting is scale-specific.
        
        Args:
            predictions: Prediction series
            actual_returns: Actual return series
            time_scales: List of pandas time scale strings
            
        Returns:
            Dictionary with multi-scale consistency metrics
        """
        # Align indices first
        common_idx = predictions.index.intersection(actual_returns.index)
        if len(common_idx) == 0:
            return {'scale_performance': {}, 'multi_scale_consistency': 0.0, 'passed_multi_scale': False}

        pred_aligned = predictions.reindex(common_idx).dropna()
        actual_aligned = actual_returns.reindex(pred_aligned.index).dropna()
        pred_aligned = pred_aligned.reindex(actual_aligned.index).dropna()
        
        if len(pred_aligned) < 10:
            return {'scale_performance': {}, 'multi_scale_consistency': 0.0, 'passed_multi_scale': False}

        # Check if index is datetime-like
        is_datetime_index = isinstance(pred_aligned.index, pd.DatetimeIndex)
        
        if not is_datetime_index:
            # Try to convert if it looks like datetime
            try:
                if pd.api.types.is_numeric_dtype(pred_aligned.index):
                    # Assume integer date_id - use numeric grouping
                    return self._numeric_multi_scale_test(pred_aligned, actual_aligned)
                else:
                    # Try converting to datetime
                    pred_aligned.index = pd.to_datetime(pred_aligned.index)
                    actual_aligned.index = pd.to_datetime(actual_aligned.index)
                    is_datetime_index = True
            except Exception:
                # Use numeric grouping as fallback
                return self._numeric_multi_scale_test(pred_aligned, actual_aligned)

        # If we have datetime index, use resampling
        if is_datetime_index:
            scale_performance = {}
            
            for scale in time_scales:
                # Replace deprecated 'M' with 'ME'
                if scale == 'M':
                    scale = 'ME'
                elif scale == '3M':
                    scale = '3ME'
                
                try:
                    # Resample to different time scales
                    returns_resampled = actual_aligned.resample(scale).sum()
                    predictions_resampled = pred_aligned.resample(scale).mean()

                    # Align indices
                    common_index = returns_resampled.index.intersection(predictions_resampled.index)
                    if len(common_index) > 5:  # Minimum samples
                        scale_returns = returns_resampled.loc[common_index]
                        scale_preds = predictions_resampled.loc[common_index]

                        # Calculate scale performance
                        scale_strategy_returns = scale_preds.values * scale_returns.values
                        if len(scale_strategy_returns) > 0 and np.std(scale_strategy_returns) > 0:
                            scale_sharpe = sharpe_ratio(
                                pd.Series(scale_strategy_returns),
                                periods_per_year=252,
                            )
                            scale_hit_rate = (scale_strategy_returns > 0).mean()
                            try:
                                correlation = np.corrcoef(scale_preds.values, scale_returns.values)[0, 1]
                                correlation = float(correlation) if not np.isnan(correlation) else 0.0
                            except Exception:
                                correlation = 0.0

                            scale_performance[scale] = {
                                'sharpe': float(scale_sharpe),
                                'hit_rate': float(scale_hit_rate),
                                'correlation': correlation,
                                'samples': len(common_index),
                            }
                except Exception as e:
                    # Continue to next scale if this one fails
                    warnings.warn(f"Failed to test scale {scale}: {e}")
                    continue

            # Calculate multi-scale consistency
            sharpes = [
                v['sharpe'] for v in scale_performance.values() 
                if not np.isnan(v.get('sharpe', np.nan)) and not np.isinf(v.get('sharpe', 0))
            ]
            hit_rates = [
                v['hit_rate'] for v in scale_performance.values() 
                if not np.isnan(v.get('hit_rate', np.nan))
            ]

            if sharpes and len(sharpes) > 1:
                mean_sharpe = np.mean(np.abs(sharpes))
                scale_consistency = (
                    1 - (np.std(sharpes) / (mean_sharpe + 1e-10)) if mean_sharpe > 0 else 0.0
                )
                hit_consistency = 1 - np.std(hit_rates) if len(hit_rates) > 1 else 0.5
                multi_scale_score = (scale_consistency + hit_consistency) / 2
            elif sharpes:
                # Only one scale worked, use it directly
                multi_scale_score = min(1.0, abs(sharpes[0]) / 2.0) if sharpes[0] != 0 else 0.0
            else:
                multi_scale_score = 0.0

            return {
                'scale_performance': scale_performance,
                'multi_scale_consistency': float(multi_scale_score),
                'passed_multi_scale': multi_scale_score > 0.6,
            }
        
        # Final fallback to numeric
        return self._numeric_multi_scale_test(pred_aligned, actual_aligned)

    def _numeric_multi_scale_test(
        self, predictions: pd.Series, actual_returns: pd.Series
    ) -> Dict:
        """Multi-scale test using numeric grouping when datetime not available.
        
        Uses rolling window aggregation to simulate different time scales:
        - 1D: window=1 (daily)
        - 1W: window=5 (weekly, ~5 trading days)
        - 1M: window=21 (monthly, ~21 trading days)
        - 3M: window=63 (quarterly, ~63 trading days)
        """
        # Use rolling windows as proxy for time scales
        # Map to approximate trading day equivalents
        windows = {
            '1D': 1,
            '1W': 5,
            '1M': 21,
            '3M': 63,
        }
        scale_performance = {}

        common_idx = predictions.index.intersection(actual_returns.index)
        pred_aligned = predictions.reindex(common_idx).dropna()
        actual_aligned = actual_returns.reindex(pred_aligned.index).dropna()
        pred_aligned = pred_aligned.reindex(actual_aligned.index).dropna()
        
        if len(pred_aligned) < 10:
            return {
                'scale_performance': {},
                'multi_scale_consistency': 0.0,
                'passed_multi_scale': False
            }

        # Sort by index to ensure proper ordering
        if not pred_aligned.index.is_monotonic_increasing:
            pred_aligned = pred_aligned.sort_index()
            actual_aligned = actual_aligned.reindex(pred_aligned.index)

        for scale_name, window in windows.items():
            if len(pred_aligned) > window * 3:  # Need at least 3 windows for meaningful stats
                try:
                    # Use rolling windows: for returns sum, for predictions mean
                    # Returns need to be summed (compounding)
                    returns_rolling = actual_aligned.rolling(window=window, min_periods=window).sum().dropna()
                    # Predictions are averaged (signal smoothing)
                    preds_rolling = pred_aligned.rolling(window=window, min_periods=window).mean().dropna()
                    
                    # Align indices
                    common_roll_idx = returns_rolling.index.intersection(preds_rolling.index)
                    if len(common_roll_idx) > 5:  # Minimum samples for reliable stats
                        scale_returns = returns_rolling.loc[common_roll_idx]
                        scale_preds = preds_rolling.loc[common_roll_idx]

                        # Calculate scale performance
                        scale_strategy_returns = scale_preds.values * scale_returns.values
                        
                        if len(scale_strategy_returns) > 0 and np.std(scale_strategy_returns) > 1e-10:
                            scale_sharpe = sharpe_ratio(
                                pd.Series(scale_strategy_returns),
                                periods_per_year=252,
                            )
                            
                            # Handle infinite or NaN Sharpe
                            if np.isnan(scale_sharpe) or np.isinf(scale_sharpe):
                                continue
                                
                            scale_hit_rate = (scale_strategy_returns > 0).mean()
                            
                            try:
                                correlation = np.corrcoef(scale_preds.values, scale_returns.values)[0, 1]
                                correlation = float(correlation) if not np.isnan(correlation) else 0.0
                            except Exception:
                                correlation = 0.0

                            scale_performance[scale_name] = {
                                'sharpe': float(scale_sharpe),
                                'hit_rate': float(scale_hit_rate),
                                'correlation': correlation,
                                'samples': len(common_roll_idx),
                            }
                except Exception as e:
                    warnings.warn(f"Failed to test numeric scale {scale_name} (window={window}): {e}")
                    continue

        # Calculate multi-scale consistency
        sharpes = [
            v.get('sharpe', 0) for v in scale_performance.values()
            if not np.isnan(v.get('sharpe', np.nan)) and not np.isinf(v.get('sharpe', 0))
        ]
        hit_rates = [
            v.get('hit_rate', 0) for v in scale_performance.values()
            if not np.isnan(v.get('hit_rate', np.nan))
        ]

        if sharpes and len(sharpes) > 1:
            mean_sharpe = np.mean(np.abs(sharpes))
            scale_consistency = (
                1 - (np.std(sharpes) / (mean_sharpe + 1e-10)) if mean_sharpe > 0 else 0.0
            )
            hit_consistency = 1 - np.std(hit_rates) if len(hit_rates) > 1 and np.std(hit_rates) > 0 else 0.5
            multi_scale_score = (scale_consistency + hit_consistency) / 2
        elif sharpes:
            # Only one scale worked, score based on Sharpe magnitude
            multi_scale_score = min(1.0, max(0.0, abs(sharpes[0]) / 3.0))
        else:
            multi_scale_score = 0.0

        return {
            'scale_performance': scale_performance,
            'multi_scale_consistency': float(multi_scale_score),
            'passed_multi_scale': multi_scale_score > 0.6,
        }

    def information_theoretic_learning_test(
        self,
        model_predictions: pd.Series,
        actual_returns: pd.Series,
        feature_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Test learning using information theory measures.
        
        Args:
            model_predictions: Model predictions
            actual_returns: Actual returns
            feature_matrix: Optional feature matrix for conditional entropy
            
        Returns:
            Dictionary with information-theoretic metrics
        """
        # Align indices
        common_idx = model_predictions.index.intersection(actual_returns.index)
        if len(common_idx) == 0:
            return {
                'mutual_information': 0.0,
                'conditional_entropy': 1.0,
                'information_gain_over_time': 0.0,
                'learning_efficiency_score': 0.0,
                'efficient_learner': False,
            }

        pred_aligned = model_predictions.reindex(common_idx).dropna()
        actual_aligned = actual_returns.reindex(pred_aligned.index)

        # Calculate mutual information (simplified using correlation)
        if len(pred_aligned) > 0 and len(actual_aligned) > 0:
            correlation = np.corrcoef(pred_aligned.values, actual_aligned.values)[0, 1]
            # Approximate MI using correlation (Monte Carlo approach for true MI is more complex)
            mi_pred_actual = abs(correlation) if not np.isnan(correlation) else 0.0

            # Conditional entropy (simplified)
            # High correlation = low conditional entropy
            cond_entropy = max(0, 1 - abs(correlation)) if not np.isnan(correlation) else 1.0

            # Information gain over time (split in half and compare)
            mid_point = len(pred_aligned) // 2
            if mid_point > 10:
                pred_first = pred_aligned.iloc[:mid_point]
                pred_second = pred_aligned.iloc[mid_point:]
                actual_first = actual_aligned.iloc[:mid_point]
                actual_second = actual_aligned.iloc[mid_point:]

                corr_first = np.corrcoef(pred_first.values, actual_first.values)[0, 1]
                corr_second = np.corrcoef(pred_second.values, actual_second.values)[0, 1]

                # Information gain: improvement in second half
                information_gain = max(0, abs(corr_second) - abs(corr_first)) if not (
                    np.isnan(corr_first) or np.isnan(corr_second)
                ) else 0.0
            else:
                information_gain = 0.0

            # Learning efficiency score
            learning_efficiency = mi_pred_actual * 0.4 + (1 - cond_entropy) * 0.3 + information_gain * 0.3
        else:
            mi_pred_actual = 0.0
            cond_entropy = 1.0
            information_gain = 0.0
            learning_efficiency = 0.0

        return {
            'mutual_information': float(mi_pred_actual),
            'conditional_entropy': float(cond_entropy),
            'information_gain_over_time': float(information_gain),
            'learning_efficiency_score': float(learning_efficiency),
            'efficient_learner': learning_efficiency > 0.7,
        }

    def _create_shock_mask(self, index: pd.Index, scenario: Dict) -> pd.Series:
        """Create mask for specific shock scenario."""
        if scenario['type'] == 'volatility_explosion':
            returns = scenario.get('returns')
            if returns is not None:
                try:
                    vol_rolling = returns.rolling(21, min_periods=10).std()
                    if vol_rolling.notna().sum() > 0:
                        vol_threshold = vol_rolling.quantile(0.95)
                        high_vol_periods = vol_rolling > vol_threshold
                        return high_vol_periods.reindex(index, fill_value=False)
                except Exception:
                    pass

        elif scenario['type'] == 'flash_crash':
            returns = scenario.get('returns')
            if returns is not None:
                try:
                    crash_threshold = returns.quantile(0.01)  # Bottom 1%
                    crash_periods = returns < crash_threshold
                    return crash_periods.reindex(index, fill_value=False)
                except Exception:
                    pass

        return pd.Series(False, index=index)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min())
        except Exception:
            return 0.0

