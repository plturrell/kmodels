"""Tests for analysis solver."""

import pytest

from ..src.solvers.analysis_solver import AnalysisSolver


def test_analysis_solver_initialization():
    """Test analysis solver initialization."""
    solver = AnalysisSolver()
    assert solver.domain == "analysis"


def test_can_solve():
    """Test can_solve method."""
    solver = AnalysisSolver()

    assert solver.can_solve("Find the limit as x approaches 2")
    assert solver.can_solve("Derivative of x²")
    assert solver.can_solve("Integral of x²")
    assert not solver.can_solve("Find the area of a triangle")


def test_limit():
    """Test limit solving."""
    solver = AnalysisSolver()

    # Simple limit
    problem = "Find the limit as x approaches 2 of x²"
    answer = solver.solve(problem)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_derivative():
    """Test derivative solving."""
    solver = AnalysisSolver()

    # Derivative problem
    problem = "Derivative of x³ at x = 2"
    answer = solver.solve(problem)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_integral():
    """Test integral solving."""
    solver = AnalysisSolver()

    # Integral problem
    problem = "Integral from 0 to 1 of x² dx"
    answer = solver.solve(problem)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999


def test_sequence():
    """Test sequence/series solving."""
    solver = AnalysisSolver()

    # Sum problem
    problem = "Sum of first 10 terms"
    answer = solver.solve(problem)
    assert isinstance(answer, int)
    assert 0 <= answer <= 99999

