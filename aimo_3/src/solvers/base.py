"""Base solver interface for multi-domain architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SolverResult:
    """Result from a solver."""

    answer: int
    confidence: float = 0.0
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSolver(ABC):
    """
    Base interface for all domain-specific solvers.
    
    All solvers must implement the solve() method that takes a problem
    statement and returns an integer answer in [0, 99999].
    """

    def __init__(self, domain: str):
        """
        Initialize solver.

        Args:
            domain: Domain name (e.g., "geometry", "algebra", "number_theory")
        """
        self.domain = domain

    @abstractmethod
    def solve(self, problem_statement: str) -> int:
        """
        Solve a problem in this domain.

        Args:
            problem_statement: Problem statement (LaTeX format)

        Returns:
            Integer answer in [0, 99999]
        """
        pass

    @abstractmethod
    def can_solve(self, problem_statement: str) -> bool:
        """
        Check if this solver can handle the problem.

        Args:
            problem_statement: Problem statement

        Returns:
            True if solver can handle this problem
        """
        pass

    def solve_with_metadata(self, problem_statement: str) -> SolverResult:
        """
        Solve problem and return result with metadata.

        Args:
            problem_statement: Problem statement

        Returns:
            SolverResult with answer and metadata
        """
        try:
            answer = self.solve(problem_statement)
            confidence = 1.0 if self.can_solve(problem_statement) else 0.5
            return SolverResult(
                answer=answer,
                confidence=confidence,
                method=self.domain,
                metadata={"domain": self.domain},
            )
        except Exception as e:
            return SolverResult(
                answer=0,
                confidence=0.0,
                method=self.domain,
                metadata={"error": str(e)},
            )

