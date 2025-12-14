"""Unified solver that integrates all domain solvers."""

from typing import Optional

from .algebra_solver import AlgebraSolver
from .analysis_solver import AnalysisSolver
from .combinatorics_solver import CombinatoricsSolver, GraphSolver
from .domain_router import DomainRouter
from .geometry_solver_wrapper import GeometrySolverWrapper
from .number_theory_solver import NumberTheorySolver


class UnifiedSolver:
    """
    Unified solver that integrates all domain-specific solvers.
    
    Routes problems to appropriate solvers and provides fallback mechanisms.
    """

    def __init__(self):
        """Initialize unified solver with all domain solvers."""
        self.router = DomainRouter()

        # Register all solvers
        self.router.register_solver(GeometrySolverWrapper())
        self.router.register_solver(AlgebraSolver())
        self.router.register_solver(NumberTheorySolver())
        self.router.register_solver(CombinatoricsSolver())
        self.router.register_solver(GraphSolver())
        self.router.register_solver(AnalysisSolver())

    def solve(self, problem_statement: str) -> int:
        """
        Solve problem using appropriate domain solver.

        Args:
            problem_statement: Problem statement

        Returns:
            Integer answer in [0, 99999]
        """
        result = self.router.solve(problem_statement)
        return result.answer

    def solve_with_metadata(self, problem_statement: str):
        """Solve and return result with metadata."""
        return self.router.solve(problem_statement)

