"""Wrapper for geometry solver to integrate with multi-domain architecture."""

from typing import Optional, TYPE_CHECKING

from ..geometry.solver import GeometrySolver
from .base import BaseSolver

if TYPE_CHECKING:
    from ..geometry.metadata_schema import ProofToken as GeometryProofToken
    from ..orchestration.stability_tracker import ProofToken as OrchestrationProofToken


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
        self.measure_stability = measure_stability
        self.last_proof_token: Optional["OrchestrationProofToken"] = None
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
        if self.measure_stability:
            geo_token = self.geometry_solver.get_last_proof_token()
            self.last_proof_token = self._convert_proof_token(geo_token) if geo_token is not None else None
        return answer

    def can_solve(self, problem_statement: str) -> bool:
        """Check if problem is geometric."""
        problem_lower = problem_statement.lower()
        geometry_keywords = [
            "triangle", "circle", "angle", "length", "area", "perimeter",
            "parallel", "perpendicular", "tangent", "inscribed", "circumscribed",
            "point", "line", "segment", "polygon", "radius", "diameter",
        ]
        return any(keyword in problem_lower for keyword in geometry_keywords)
    
    def get_last_proof_token(self) -> Optional["OrchestrationProofToken"]:
        """Get the proof token from the last solve() call."""
        return self.last_proof_token

    def _convert_proof_token(self, token: "GeometryProofToken") -> "OrchestrationProofToken":
        """Convert geometry ProofToken to orchestration ProofToken."""
        from ..orchestration.stability_tracker import ProofToken, StabilityStatus

        theorem_name = "unknown"
        if token.proof_sequence:
            theorem_name = str(token.proof_sequence[-1][0])

        stability = None
        if token.stability is not None:
            stability = StabilityStatus(
                status=str(token.stability.status),
                lyapunov_exponent=float(token.stability.lambda_max),
                confidence=float(token.stability.confidence),
            )

        return ProofToken(
            token_id=token.token_id,
            theorem_name=theorem_name,
            stability=stability,
            metadata={"problem_id": token.problem_id},
        )


