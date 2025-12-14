"""Multi-domain solver architecture."""

from .base import BaseSolver, SolverResult
from .domain_router import DomainRouter
from .algebra_solver import AlgebraSolver
from .number_theory_solver import NumberTheorySolver
from .geometry_solver_wrapper import GeometrySolverWrapper
from .analysis_solver import AnalysisSolver
from .combinatorics_solver import CombinatoricsSolver, GraphSolver

__all__ = [
    "BaseSolver",
    "SolverResult",
    "DomainRouter",
    "AlgebraSolver",
    "NumberTheorySolver",
    "GeometrySolverWrapper",
    "CombinatoricsSolver",
    "GraphSolver",
    "AnalysisSolver",
]

