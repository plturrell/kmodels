"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    return {
        "problem_id": "test_001",
        "statement": "What is $2 + 2$?",
        "answer": 4,
    }

