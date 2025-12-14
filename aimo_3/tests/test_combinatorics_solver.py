"""Tests for combinatorics solver."""

import pytest

from ..src.solvers.combinatorics_solver import CombinatoricsSolver, GraphSolver


def test_combinatorics_solver_initialization():
    """Test combinatorics solver initialization."""
    solver = CombinatoricsSolver()
    assert solver.domain == "combinatorics"


def test_permutation():
    """Test permutation solving."""
    solver = CombinatoricsSolver()
    
    # P(5, 3) = 60
    problem = "Find P(5, 3)"
    answer = solver.solve(problem)
    assert answer == 60


def test_combination():
    """Test combination solving."""
    solver = CombinatoricsSolver()
    
    # C(5, 3) = 10
    problem = "Find C(5, 3) or 5 choose 3"
    answer = solver.solve(problem)
    assert answer == 10


def test_factorial():
    """Test factorial solving."""
    solver = CombinatoricsSolver()
    
    # 5! = 120
    problem = "Find 5!"
    answer = solver.solve(problem)
    assert answer == 120


def test_can_solve():
    """Test can_solve method."""
    solver = CombinatoricsSolver()
    
    assert solver.can_solve("How many permutations of 5 objects?")
    assert solver.can_solve("Find the combination C(10, 3)")
    assert not solver.can_solve("Find the area of a triangle")


def test_graph_solver():
    """Test graph solver."""
    solver = GraphSolver()
    
    assert solver.domain == "graph"
    assert solver.can_solve("How many paths in a graph?")
    assert not solver.can_solve("Solve for x: 2x + 5 = 13")


def test_counting_problems():
    """Test counting problems."""
    solver = CombinatoricsSolver()
    
    # "How many ways to arrange 5 objects" -> 5! = 120
    problem = "How many ways to arrange 5 objects?"
    answer = solver.solve(problem)
    assert answer == 120

