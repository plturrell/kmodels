"""Tests for training components."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from training.metrics import AIMOMetrics, compute_penalized_accuracy


def test_aimo_metrics_accuracy():
    """Test accuracy computation."""
    metrics = AIMOMetrics()
    metrics.update([1, 2, 3], [1, 2, 4])
    
    computed = metrics.compute()
    assert "accuracy" in computed
    assert computed["accuracy"] == pytest.approx(2/3, abs=0.01)


def test_aimo_metrics_mae():
    """Test mean absolute error."""
    metrics = AIMOMetrics()
    metrics.update([1, 2, 3], [1, 3, 5])
    
    computed = metrics.compute()
    assert "mean_absolute_error" in computed
    assert computed["mean_absolute_error"] == pytest.approx(1.0, abs=0.01)


def test_aimo_metrics_within_range():
    """Test within range metric."""
    metrics = AIMOMetrics()
    metrics.update([10, 20, 30], [10, 21, 35])
    
    computed = metrics.compute()
    assert "within_range_1" in computed
    assert computed["within_range_1"] == pytest.approx(1/3, abs=0.01)


def test_penalized_accuracy_both_correct():
    """Test penalized accuracy when both answers correct."""
    predictions = {"problem_1": 42}
    ground_truth = {"problem_1": [42, 42]}
    
    score = compute_penalized_accuracy(predictions, ground_truth)
    assert score == 1.0


def test_penalized_accuracy_one_correct():
    """Test penalized accuracy when one answer correct."""
    predictions = {"problem_1": 42}
    ground_truth = {"problem_1": [42, 100]}
    
    score = compute_penalized_accuracy(predictions, ground_truth)
    assert score == 0.5


def test_penalized_accuracy_none_correct():
    """Test penalized accuracy when no answers correct."""
    predictions = {"problem_1": 0}
    ground_truth = {"problem_1": [42, 100]}
    
    score = compute_penalized_accuracy(predictions, ground_truth)
    assert score == 0.0


def test_metrics_reset():
    """Test metrics reset."""
    metrics = AIMOMetrics()
    metrics.update([1, 2], [1, 2])
    metrics.reset()
    
    computed = metrics.compute()
    assert computed == {}


def test_per_problem_metrics():
    """Test per-problem metrics."""
    metrics = AIMOMetrics()
    metrics.update(
        [1, 2, 3],
        [1, 2, 4],
        ["prob1", "prob2", "prob1"],
    )
    
    per_problem = metrics.get_per_problem_metrics()
    assert "prob1" in per_problem
    assert "prob2" in per_problem
    assert per_problem["prob1"]["accuracy"] == pytest.approx(0.5, abs=0.01)

