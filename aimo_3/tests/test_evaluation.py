"""Tests for evaluation API."""

import pytest

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evaluation.api import AIMOEvaluator, validate_answer


def test_validate_answer():
    """Test answer validation."""
    assert validate_answer(0) is True
    assert validate_answer(99999) is True
    assert validate_answer(50000) is True
    assert validate_answer(-1) is False
    assert validate_answer(100000) is False
    assert validate_answer("123") is False
    assert validate_answer(123.5) is False


def test_evaluator_initialization(tmp_path):
    """Test evaluator initialization."""
    evaluator = AIMOEvaluator(output_dir=tmp_path)
    assert evaluator.output_dir == tmp_path
    assert len(evaluator.answers) == 0


def test_submit_answer_for_problem(tmp_path):
    """Test submitting answers."""
    evaluator = AIMOEvaluator(output_dir=tmp_path)
    
    # Valid answer
    assert evaluator.submit_answer_for_problem("test_001", 42) is True
    assert evaluator.answers["test_001"] == 42
    
    # Invalid answer
    assert evaluator.submit_answer_for_problem("test_002", 100000) is False
    assert "test_002" not in evaluator.answers


def test_generate_submission_file(tmp_path):
    """Test generating submission file."""
    evaluator = AIMOEvaluator(output_dir=tmp_path)
    evaluator.answers = {
        "problem_001": 42,
        "problem_002": 100,
    }
    
    submission_path = evaluator.generate_submission_file()
    assert submission_path.exists()
    
    # Check file contents
    with open(submission_path) as f:
        lines = f.readlines()
        assert lines[0].strip() == "problem_id,answer"
        assert "problem_001,42" in [line.strip() for line in lines[1:]]
        assert "problem_002,100" in [line.strip() for line in lines[1:]]


def test_calculate_penalized_accuracy(tmp_path):
    """Test penalized accuracy calculation."""
    evaluator = AIMOEvaluator(output_dir=tmp_path)
    evaluator.answers = {
        "problem_001": 42,
        "problem_002": 100,
    }
    
    ground_truth = {
        "problem_001": [42, 42],  # Both correct
        "problem_002": [100, 200],  # One correct
    }
    
    score = evaluator.calculate_penalized_accuracy(ground_truth)
    assert score == 1.5  # 1.0 + 0.5

