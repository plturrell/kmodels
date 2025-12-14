"""Tests for data loading and processing."""

import json
import tempfile
from pathlib import Path

import pytest

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.loader import load_problem, load_problems
from data.preprocessing import clean_problem_text, extract_math_expressions


def test_load_problem_json(tmp_path, sample_problem):
    """Test loading a problem from JSON file."""
    problem_file = tmp_path / "test.json"
    with open(problem_file, "w") as f:
        json.dump(sample_problem, f)

    loaded = load_problem(problem_file)
    assert loaded["problem_id"] == "test_001"
    assert "statement" in loaded


def test_load_problems(tmp_path, sample_problem):
    """Test loading multiple problems."""
    # Create multiple problem files
    for i in range(3):
        problem_file = tmp_path / f"problem_{i}.json"
        problem = sample_problem.copy()
        problem["problem_id"] = f"test_{i:03d}"
        with open(problem_file, "w") as f:
            json.dump(problem, f)

    problems = load_problems(tmp_path)
    assert len(problems) == 3
    assert all("problem_id" in p for p in problems)


def test_extract_math_expressions():
    """Test extracting math expressions from LaTeX."""
    text = "Calculate $x + y$ and $z^2$."
    expressions = extract_math_expressions(text)
    assert len(expressions) >= 2


def test_clean_problem_text():
    """Test cleaning problem text."""
    text = "  Problem  with   extra   spaces  "
    cleaned = clean_problem_text(text)
    assert "  " not in cleaned
    assert cleaned.strip() == cleaned

