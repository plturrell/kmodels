"""Evaluation metrics for AIMO problem solving."""

from typing import Dict, List, Optional

import numpy as np
import torch


class AIMOMetrics:
    """Metrics for evaluating AIMO problem-solving models."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions: List[int] = []
        self.targets: List[int] = []
        self.problem_ids: List[str] = []

    def update(self, predictions: List[int], targets: List[int], problem_ids: Optional[List[str]] = None):
        """
        Update metrics with new predictions.

        Args:
            predictions: List of predicted answers
            targets: List of ground truth answers
            problem_ids: Optional list of problem IDs
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        if problem_ids:
            self.problem_ids.extend(problem_ids)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        if not self.predictions or not self.targets:
            return {}

        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {
            "accuracy": self._accuracy(predictions, targets),
            "exact_match": self._exact_match(predictions, targets),
            "mean_absolute_error": self._mean_absolute_error(predictions, targets),
            "mean_squared_error": self._mean_squared_error(predictions, targets),
            "within_range_1": self._within_range(predictions, targets, range_val=1),
            "within_range_10": self._within_range(predictions, targets, range_val=10),
            "within_range_100": self._within_range(predictions, targets, range_val=100),
        }

        return metrics

    def _accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute accuracy (exact match)."""
        return float(np.mean(predictions == targets))

    def _exact_match(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute exact match rate (same as accuracy)."""
        return self._accuracy(predictions, targets)

    def _mean_absolute_error(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean absolute error."""
        return float(np.mean(np.abs(predictions - targets)))

    def _mean_squared_error(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean squared error."""
        return float(np.mean((predictions - targets) ** 2))

    def _within_range(self, predictions: np.ndarray, targets: np.ndarray, range_val: int) -> float:
        """Compute fraction of predictions within range_val of target."""
        return float(np.mean(np.abs(predictions - targets) <= range_val))

    def get_per_problem_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get metrics per problem.

        Returns:
            Dictionary mapping problem_id to metrics
        """
        if not self.problem_ids:
            return {}

        per_problem = {}
        for problem_id in set(self.problem_ids):
            indices = [i for i, pid in enumerate(self.problem_ids) if pid == problem_id]
            preds = [self.predictions[i] for i in indices]
            targs = [self.targets[i] for i in indices]

            per_problem[problem_id] = {
                "accuracy": self._accuracy(np.array(preds), np.array(targs)),
                "mae": self._mean_absolute_error(np.array(preds), np.array(targs)),
            }

        return per_problem


def compute_penalized_accuracy(
    predictions: Dict[str, int],
    ground_truth: Dict[str, List[int]],
) -> float:
    """
    Compute penalized accuracy as per AIMO competition rules.

    For each problem:
    - Both answers correct: score = 1
    - One correct, one incorrect: score = 0.5
    - Both incorrect: score = 0

    Args:
        predictions: Dictionary mapping problem_id to predicted answer
        ground_truth: Dictionary mapping problem_id to list of two correct answers

    Returns:
        Total penalized accuracy score
    """
    total_score = 0.0

    for problem_id, correct_answers in ground_truth.items():
        if problem_id not in predictions:
            continue

        predicted = predictions[problem_id]
        correct_count = sum(1 for ans in correct_answers if ans == predicted)

        if correct_count == 2:
            score = 1.0
        elif correct_count == 1:
            score = 0.5
        else:
            score = 0.0

        total_score += score

    return total_score

