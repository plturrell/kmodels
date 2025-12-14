"""Wrapper for geometry solver to integrate with multi-domain architecture."""

from typing import Optional

from ..geometry.solver import GeometrySolver
from .base import BaseSolver


class GeometrySolverWrapper(BaseSolver):
    """
    Wrapper for GeometrySolver to integrate with multi-domain architecture.
    """

    def __init__(
        self,
        geometry_solver: Optional[GeometrySolver] = None,
        measure_stability: bool = False
    ):
        """
        Initialize geometry solver wrapper.

        Args:
            geometry_solver: Optional GeometrySolver instance (creates new if None)
            measure_stability: Whether to compute Lyapunov stability metrics
        """
        super().__init__("geometry")
        if geometry_solver:
            self.geometry_solver = geometry_solver
        else:
            self.geometry_solver = GeometrySolver(measure_stability=measure_stability)

    def solve(self, problem_statement: str) -> int:
        """
        Solve geometry problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Integer answer
        """
        answer = self.geometry_solver.solve(problem_statement)
        
        # Capture proof token if stability measurement enabled
        if self.measure_stability:
            self.last_proof_token = self._create_proof_token()
        
        return answer
    
    def get_last_proof_token(self):
        """
        Get the last proof token with stability information.
        
        Returns:
            ProofToken or None
        """
        return self.last_proof_token
    
    def _create_proof_token(self):
        """Create proof token from last solver execution."""
        from ..orchestration.stability_tracker import ProofToken, StabilityStatus
        
        # Simplified: create token with basic stability info
        # In full implementation, would extract from solver's proof trace
        stability = StabilityStatus(
            status="stable",  # Would compute from proof trace
            confidence=0.8,  # Would compute from proof quality
        )
        
        return ProofToken(
            token_id="proof_1",
            theorem_name="unknown",
            stability=stability,
        )

    def can_solve(self, problem_statement: str) -> bool:
        """Check if problem is geometric."""
        problem_lower = problem_statement.lower()
        geometry_keywords = [
            "triangle", "circle", "angle", "length", "area", "perimeter",
            "parallel", "perpendicular", "tangent", "inscribed", "circumscribed",
            "point", "line", "segment", "polygon", "radius", "diameter",
        ]
        return any(keyword in problem_lower for keyword in geometry_keywords)
    
    def get_last_proof_token(self):
        """Get the proof token from the last solve() call."""
        return self.geometry_solver.get_last_proof_token()


