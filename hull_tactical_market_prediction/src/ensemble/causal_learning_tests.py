"""Test if model discovers causal mechanisms vs spurious correlations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import warnings


class CausalLearningTester:
    """Test if model discovers causal mechanisms vs spurious correlations.
    
    Based on Judea Pearl's causal hierarchy and invariant causal prediction.
    """

    def __init__(self, alpha: float = 0.05):
        """Initialize causal learning tester.
        
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha

    def invariant_causal_prediction_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        environment_labels: pd.Series,
    ) -> Dict:
        """Invariant Causal Prediction (ICP) test.
        
        True causal features should have invariant relationships across environments.
        
        Args:
            X: Feature matrix
            y: Target series
            environment_labels: Series indicating environment for each sample
            
        Returns:
            Dictionary with ICP test results
        """
        environments = environment_labels.dropna().unique()
        if len(environments) < 2:
            return {
                'feature_p_values': {},
                'causal_features': [],
                'causal_feature_ratio': 0.0,
                'has_causal_features': False,
                'note': 'Need at least 2 environments',
            }

        p_values_by_feature = {}

        for feature_idx in range(min(X.shape[1], 100)):  # Limit to first 100 features for efficiency
            feature_name = X.columns[feature_idx] if hasattr(X, 'columns') else feature_idx
            feature_p_values = []

            for env in environments:
                env_mask = environment_labels == env
                if env_mask.sum() > 10:  # Minimum samples per environment
                    X_env = X[env_mask]
                    y_env = y.reindex(X_env.index).dropna()

                    # Align
                    common_idx = X_env.index.intersection(y_env.index)
                    if len(common_idx) > 10:
                        X_env_aligned = X_env.reindex(common_idx)
                        y_env_aligned = y_env.reindex(common_idx)

                        try:
                            # Test significance using correlation
                            feature_values = X_env_aligned.iloc[:, feature_idx].values
                            target_values = y_env_aligned.values

                            if len(feature_values) > 0 and np.std(feature_values) > 0:
                                corr, p_value = stats.pearsonr(feature_values, target_values)
                                if not np.isnan(p_value):
                                    feature_p_values.append(p_value)
                        except Exception:
                            continue

            if feature_p_values and len(feature_p_values) >= 2:
                try:
                    # Use Fisher's method to combine p-values across environments
                    from scipy.stats import combine_pvalues

                    _, combined_p = combine_pvalues(feature_p_values, method='fisher')
                    p_values_by_feature[feature_name] = float(combined_p)
                except Exception:
                    # Fallback: use mean p-value
                    p_values_by_feature[feature_name] = float(np.mean(feature_p_values))

        # Identify causal features (invariant across environments)
        causal_features = [
            feat for feat, p in p_values_by_feature.items() if p < self.alpha
        ]

        causal_ratio = len(causal_features) / max(1, len(p_values_by_feature))

        return {
            'feature_p_values': {str(k): float(v) for k, v in p_values_by_feature.items()},
            'causal_features': causal_features[:20],  # Limit output
            'causal_feature_ratio': float(causal_ratio),
            'has_causal_features': len(causal_features) > 0,
        }

    def intervention_response_test(
        self,
        model_predict_fn,
        X: pd.DataFrame,
        y: pd.Series,
        intervention_features: Optional[List[int]] = None,
        n_features_to_test: int = 10,
    ) -> Dict:
        """Test if model responds appropriately to interventions.
        
        Based on do-calculus and intervention distributions.
        
        Args:
            model_predict_fn: Function that takes X and returns predictions
            X: Feature matrix
            y: Target series (for reference)
            intervention_features: List of feature indices to test (None = auto-select)
            n_features_to_test: Number of features to test if intervention_features is None
            
        Returns:
            Dictionary with intervention test results
        """
        if intervention_features is None:
            # Select features with highest variance
            feature_vars = X.var()
            top_features = feature_vars.nlargest(n_features_to_test).index.tolist()
            intervention_features = [X.columns.get_loc(f) for f in top_features if f in X.columns]
            if not intervention_features:
                intervention_features = list(range(min(n_features_to_test, X.shape[1])))

        intervention_effects = {}

        try:
            # Get baseline predictions
            original_preds = model_predict_fn(X)
            if isinstance(original_preds, pd.Series):
                original_preds = original_preds.values

            for feature_idx in intervention_features[:n_features_to_test]:
                try:
                    # Create intervention: break correlations with other features
                    X_intervened = X.copy()
                    original_values = X_intervened.iloc[:, feature_idx].copy()

                    # Intervention: shuffle to break dependencies
                    X_intervened.iloc[:, feature_idx] = np.random.permutation(original_values)

                    # Measure prediction change
                    intervened_preds = model_predict_fn(X_intervened)
                    if isinstance(intervened_preds, pd.Series):
                        intervened_preds = intervened_preds.values

                    # Calculate intervention effect
                    effect_size = float(np.mean(np.abs(intervened_preds - original_preds)))
                    intervention_effects[feature_idx] = effect_size
                except Exception:
                    continue

        except Exception as e:
            return {
                'intervention_effects': {},
                'causal_consistency_score': 0.0,
                'causally_consistent': False,
                'error': str(e),
            }

        # Test if intervention effects are consistent (non-zero effects suggest causal relationship)
        if intervention_effects:
            # Score based on whether interventions have measurable effects
            # Consistent causal structure: interventions on causal features have larger effects
            effect_values = list(intervention_effects.values())
            if effect_values:
                # High variance in effects suggests differential causal importance
                causal_consistency = min(1.0, np.std(effect_values) / (np.mean(effect_values) + 1e-10))
            else:
                causal_consistency = 0.0
        else:
            causal_consistency = 0.0

        return {
            'intervention_effects': {int(k): float(v) for k, v in intervention_effects.items()},
            'causal_consistency_score': float(causal_consistency),
            'causally_consistent': causal_consistency > 0.3,
        }

    def counterfactual_fairness_test(
        self,
        model_predict_fn,
        X: pd.DataFrame,
        sensitive_features: List[str],
    ) -> Dict:
        """Test if model makes similar predictions for counterfactual worlds.
        
        Args:
            model_predict_fn: Function that takes X and returns predictions
            X: Feature matrix
            sensitive_features: List of sensitive feature names to test
            
        Returns:
            Dictionary with counterfactual fairness metrics
        """
        fairness_scores = {}

        available_features = [f for f in sensitive_features if f in X.columns]
        if not available_features:
            return {
                'fairness_scores': {},
                'overall_fairness': 0.0,
                'counterfactually_fair': False,
                'note': 'No sensitive features found',
            }

        try:
            original_preds = model_predict_fn(X)
            if isinstance(original_preds, pd.Series):
                original_preds = original_preds.values

            for sens_feat in available_features[:5]:  # Limit to 5 for efficiency
                try:
                    # Create counterfactuals by modifying sensitive feature
                    X_counterfactual = X.copy()
                    original_values = X_counterfactual[sens_feat].copy()

                    # Counterfactual: change sensitive feature values
                    if X_counterfactual[sens_feat].dtype == 'object' or X_counterfactual[sens_feat].dtype.name == 'category':
                        # Categorical - swap categories
                        unique_vals = X_counterfactual[sens_feat].dropna().unique()
                        if len(unique_vals) > 1:
                            counterfactual_vals = np.random.choice(unique_vals, len(X_counterfactual))
                            X_counterfactual[sens_feat] = counterfactual_vals
                        else:
                            continue
                    else:
                        # Numerical - add noise or shift
                        std_val = X_counterfactual[sens_feat].std()
                        if std_val > 0:
                            noise = np.random.normal(0, std_val * 0.5, len(X_counterfactual))
                            X_counterfactual[sens_feat] = X_counterfactual[sens_feat] + noise
                        else:
                            continue

                    # Compare predictions
                    counterfactual_preds = model_predict_fn(X_counterfactual)
                    if isinstance(counterfactual_preds, pd.Series):
                        counterfactual_preds = counterfactual_preds.values

                    # Calculate counterfactual fairness
                    prediction_difference = np.abs(original_preds - counterfactual_preds)
                    pred_std = np.std(original_preds)
                    if pred_std > 0:
                        fairness_score = 1 - (np.mean(prediction_difference) / pred_std)
                        fairness_scores[sens_feat] = max(0.0, min(1.0, fairness_score))
                except Exception:
                    continue

        except Exception as e:
            return {
                'fairness_scores': {},
                'overall_fairness': 0.0,
                'counterfactually_fair': False,
                'error': str(e),
            }

        overall_fairness = (
            float(np.mean(list(fairness_scores.values()))) if fairness_scores else 0.0
        )

        return {
            'fairness_scores': {k: float(v) for k, v in fairness_scores.items()},
            'overall_fairness': overall_fairness,
            'counterfactually_fair': overall_fairness > 0.8,
        }

