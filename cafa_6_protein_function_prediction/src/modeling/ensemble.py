"""
Ensemble methods for combining multiple protein function prediction models.

Implements various ensemble strategies including voting, averaging, and stacking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor combining multiple models."""
    
    def __init__(self, method: str = "average"):
        """Initialize ensemble predictor.
        
        Args:
            method: Ensemble method ('average', 'weighted_average', 'max', 'stacking')
        """
        self.method = method
        self.models: List = []
        self.weights: Optional[np.ndarray] = None
        self.meta_model: Optional[LogisticRegression] = None
    
    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble.
        
        Args:
            model: Model object with predict_proba or predict method
            weight: Weight for weighted averaging
        """
        self.models.append(model)
        if self.weights is None:
            self.weights = np.array([weight])
        else:
            self.weights = np.append(self.weights, weight)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using ensemble.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Probability predictions (n_samples, n_labels)
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            elif hasattr(model, 'predict'):
                pred = model.predict(X)
                # Convert to probabilities if binary
                if pred.dtype == bool or np.all(np.isin(pred, [0, 1])):
                    pred = pred.astype(float)
            else:
                raise ValueError(f"Model {model} has no predict_proba or predict method")
            predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_models, n_samples, n_labels)
        
        # Combine predictions based on method
        if self.method == "average":
            return np.mean(predictions, axis=0)
        
        elif self.method == "weighted_average":
            # Normalize weights
            weights = self.weights / np.sum(self.weights)
            weights = weights.reshape(-1, 1, 1)  # (n_models, 1, 1)
            return np.sum(predictions * weights, axis=0)
        
        elif self.method == "max":
            return np.max(predictions, axis=0)
        
        elif self.method == "stacking":
            if self.meta_model is None:
                raise ValueError("Meta-model not trained for stacking")
            # Use predictions as features for meta-model
            n_samples, n_labels = predictions.shape[1], predictions.shape[2]
            stacked_features = predictions.transpose(1, 0, 2).reshape(n_samples, -1)
            return self.meta_model.predict_proba(stacked_features)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def fit_stacking(self, X: np.ndarray, y: np.ndarray):
        """Train meta-model for stacking ensemble.
        
        Args:
            X: Feature matrix for base models
            y: True labels
        """
        if self.method != "stacking":
            raise ValueError("fit_stacking only for stacking method")
        
        # Get predictions from base models
        predictions = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X).astype(float)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_models, n_samples, n_labels)
        n_samples, n_labels = predictions.shape[1], predictions.shape[2]
        
        # Stack predictions as features
        stacked_features = predictions.transpose(1, 0, 2).reshape(n_samples, -1)
        
        # Train meta-model
        self.meta_model = LogisticRegression(max_iter=200, solver='lbfgs')
        self.meta_model.fit(stacked_features, y)
        
        LOGGER.info("Trained stacking meta-model")


def optimize_ensemble_weights(
    predictions_list: List[np.ndarray],
    ground_truth: np.ndarray,
    metric_fn,
) -> np.ndarray:
    """Optimize ensemble weights to maximize a metric.
    
    Args:
        predictions_list: List of prediction arrays from different models
        ground_truth: True labels
        metric_fn: Function that takes (y_true, y_pred) and returns a score
    
    Returns:
        Optimal weights for each model
    """
    from scipy.optimize import minimize
    
    n_models = len(predictions_list)
    predictions = np.array(predictions_list)  # (n_models, n_samples, n_labels)
    
    def objective(weights):
        """Negative metric (for minimization)."""
        weights = weights / np.sum(weights)  # Normalize
        weights = weights.reshape(-1, 1, 1)
        ensemble_pred = np.sum(predictions * weights, axis=0)
        score = metric_fn(ground_truth, ensemble_pred)
        return -score  # Negative for minimization
    
    # Initial weights (equal)
    initial_weights = np.ones(n_models) / n_models
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
    )
    
    optimal_weights = result.x
    LOGGER.info(f"Optimized ensemble weights: {optimal_weights}")
    
    return optimal_weights


__all__ = [
    "EnsemblePredictor",
    "optimize_ensemble_weights",
]

