"""Test model robustness against worst-case adversarial attacks."""

from __future__ import annotations

from typing import Dict, List, Callable, Optional

import numpy as np
import pandas as pd
import warnings


class AdversarialRobustnessTester:
    """Test model robustness against worst-case adversarial attacks.
    
    Inspired by adversarial machine learning in computer vision.
    """

    def __init__(self, attack_budget: float = 0.1):
        """Initialize adversarial robustness tester.
        
        Args:
            attack_budget: Maximum perturbation budget (as fraction of feature std)
        """
        self.attack_budget = attack_budget

    def worst_case_feature_perturbation_test(
        self,
        model_predict_fn: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        feature_importance: Optional[Dict] = None,
    ) -> Dict:
        """Find worst-case feature perturbations that break the model.
        
        Robust models should resist small adversarial changes.
        
        Args:
            model_predict_fn: Function that takes X and returns predictions
            X: Feature matrix
            y: Target series (for evaluation)
            feature_importance: Optional dict of feature_idx -> importance
            
        Returns:
            Dictionary with adversarial robustness metrics
        """
        try:
            baseline_performance = self._evaluate_model(model_predict_fn, X, y)
            adversarial_performances = {}

            # Random perturbation
            X_random = self._create_adversarial_perturbation(
                X, feature_importance, 'random', self.attack_budget
            )
            adversarial_performances['random'] = self._evaluate_adversarial_attack(
                model_predict_fn, X_random, y, baseline_performance
            )

            # Targeted perturbation (on important features)
            if feature_importance:
                X_targeted = self._create_adversarial_perturbation(
                    X, feature_importance, 'targeted', self.attack_budget
                )
                adversarial_performances['targeted'] = self._evaluate_adversarial_attack(
                    model_predict_fn, X_targeted, y, baseline_performance
                )

            # Gradient-based perturbation (simplified)
            X_gradient = self._create_adversarial_perturbation(
                X, feature_importance, 'gradient_based', self.attack_budget
            )
            adversarial_performances['gradient_based'] = self._evaluate_adversarial_attack(
                model_predict_fn, X_gradient, y, baseline_performance
            )

            overall_robustness = np.mean(
                [v['robustness'] for v in adversarial_performances.values()]
            )

            return {
                'adversarial_performances': {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in adversarial_performances.items()
                },
                'baseline_performance': float(baseline_performance),
                'overall_robustness': float(overall_robustness),
                'adversarially_robust': overall_robustness > 0.7,
            }

        except Exception as e:
            return {
                'adversarial_performances': {},
                'overall_robustness': 0.0,
                'adversarially_robust': False,
                'error': str(e),
            }

    def temporal_adversarial_attack_test(
        self,
        model_predict_fn: Callable,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: Optional[pd.DatetimeIndex] = None,
    ) -> Dict:
        """Test robustness against temporal adversarial attacks.
        
        Tests:
        - Data drift attacks
        - Distribution shift attacks
        - Concept drift attacks
        
        Args:
            model_predict_fn: Function that takes X and returns predictions
            X: Feature matrix
            y: Target series
            time_index: Optional datetime index for temporal attacks
            
        Returns:
            Dictionary with temporal robustness metrics
        """
        try:
            baseline_perf = self._evaluate_model(model_predict_fn, X, y)
            attack_results = {}

            # Gradual drift
            X_drift, y_drift = self._simulate_gradual_drift(X, y, time_index)
            attack_results['gradual_drift'] = self._evaluate_adversarial_attack(
                model_predict_fn, X_drift, y_drift, baseline_perf
            )

            # Sudden shift
            X_shift, y_shift = self._simulate_sudden_shift(X, y, time_index)
            attack_results['sudden_shift'] = self._evaluate_adversarial_attack(
                model_predict_fn, X_shift, y_shift, baseline_perf
            )

            # Cyclical attack
            X_cyclical, y_cyclical = self._simulate_cyclical_attack(X, y, time_index)
            attack_results['cyclical_attack'] = self._evaluate_adversarial_attack(
                model_predict_fn, X_cyclical, y_cyclical, baseline_perf
            )

            temporal_robustness = np.mean([v['robustness'] for v in attack_results.values()])

            return {
                'temporal_attack_results': {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in attack_results.items()
                },
                'baseline_performance': float(baseline_perf),
                'temporal_robustness': float(temporal_robustness),
                'temporally_robust': temporal_robustness > 0.6,
            }

        except Exception as e:
            return {
                'temporal_attack_results': {},
                'temporal_robustness': 0.0,
                'temporally_robust': False,
                'error': str(e),
            }

    def _create_adversarial_perturbation(
        self,
        X: pd.DataFrame,
        feature_importance: Optional[Dict],
        attack_type: str,
        budget: float,
    ) -> pd.DataFrame:
        """Create adversarial perturbation."""
        X_perturbed = X.copy()

        if attack_type == 'random':
            # Random noise on all features
            for col in X.columns:
                if X[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    noise = np.random.normal(0, X[col].std() * budget, len(X))
                    X_perturbed[col] = X_perturbed[col] + noise

        elif attack_type == 'targeted' and feature_importance:
            # Target important features
            for feat_idx, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                if feat_idx < X.shape[1]:
                    col = X.columns[feat_idx] if hasattr(X, 'columns') else feat_idx
                    if isinstance(col, str) and col in X.columns:
                        if X[col].dtype in [np.float64, np.float32]:
                            noise = np.random.normal(0, X[col].std() * budget * importance, len(X))
                            X_perturbed[col] = X_perturbed[col] + noise

        elif attack_type == 'gradient_based':
            # Simplified gradient-based: perturb features correlated with target
            # In practice, would use actual gradients
            for col in X.columns[:min(20, X.shape[1])]:  # Limit to 20 features
                if X[col].dtype in [np.float64, np.float32]:
                    noise = np.random.normal(0, X[col].std() * budget * 0.5, len(X))
                    X_perturbed[col] = X_perturbed[col] + noise

        return X_perturbed

    def _simulate_gradual_drift(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: Optional[pd.DatetimeIndex],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Simulate gradual distribution drift."""
        X_drifted = X.copy()
        y_drifted = y.copy()

        # Add gradual linear shift
        n_samples = len(X_drifted)
        drift_factor = np.linspace(0, self.attack_budget * 2, n_samples)

        for col in X.columns[:min(10, X.shape[1])]:
            if X[col].dtype in [np.float64, np.float32]:
                X_drifted[col] = X_drifted[col] + drift_factor * X[col].std()

        return X_drifted, y_drifted

    def _simulate_sudden_shift(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: Optional[pd.DatetimeIndex],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Simulate sudden distribution shift."""
        X_shifted = X.copy()
        y_shifted = y.copy()

        # Sudden shift halfway through
        mid_point = len(X_shifted) // 2
        shift_magnitude = self.attack_budget * 3

        for col in X.columns[:min(10, X.shape[1])]:
            if X[col].dtype in [np.float64, np.float32]:
                shift = np.zeros(len(X_shifted))
                shift[mid_point:] = shift_magnitude * X[col].std()
                X_shifted[col] = X_shifted[col] + shift

        return X_shifted, y_shifted

    def _simulate_cyclical_attack(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_index: Optional[pd.DatetimeIndex],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Simulate cyclical attack pattern."""
        X_cyclical = X.copy()
        y_cyclical = y.copy()

        n_samples = len(X_cyclical)
        cycle_length = n_samples // 4

        for col in X.columns[:min(10, X.shape[1])]:
            if X[col].dtype in [np.float64, np.float32]:
                cycle = np.sin(2 * np.pi * np.arange(n_samples) / cycle_length)
                attack_pattern = cycle * self.attack_budget * X[col].std()
                X_cyclical[col] = X_cyclical[col] + attack_pattern

        return X_cyclical, y_cyclical

    def _evaluate_model(self, model_predict_fn: Callable, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate model performance (RMSE)."""
        try:
            preds = model_predict_fn(X)
            if isinstance(preds, pd.Series):
                preds = preds.values
            if isinstance(y, pd.Series):
                y = y.values

            # Use negative RMSE as performance (higher is better)
            rmse = np.sqrt(np.mean((preds - y) ** 2))
            return float(-rmse)  # Negative so robustness = 1 - drop/performance
        except Exception:
            return 0.0

    def _evaluate_adversarial_attack(
        self,
        model_predict_fn: Callable,
        X_attacked: pd.DataFrame,
        y: pd.Series,
        baseline_performance: float,
    ) -> Dict:
        """Evaluate adversarial attack impact."""
        try:
            attacked_performance = self._evaluate_model(model_predict_fn, X_attacked, y)
            performance_drop = baseline_performance - attacked_performance

            # Robustness: resistance to performance drop
            if abs(baseline_performance) > 1e-8:
                robustness = 1 - min(1.0, abs(performance_drop) / abs(baseline_performance))
            else:
                robustness = 0.5  # Neutral if baseline is zero

            return {
                'performance': attacked_performance,
                'performance_drop': performance_drop,
                'robustness': max(0.0, robustness),
            }
        except Exception:
            return {'performance': 0.0, 'performance_drop': 0.0, 'robustness': 0.0}

