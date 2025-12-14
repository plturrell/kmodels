"""Advanced ensemble framework for multiple solvers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Protocol

from .llm_base import LLMSolver
from .symbolic_solver import SymbolicSolver
from .hybrid_solver import HybridSolver

from ..utils.solution_verifier import SolutionVerifier


class SupportsSolve(Protocol):
    def solve(self, problem_statement: str) -> int: ...


class SolverResult:
    """Result from a single solver."""

    def __init__(self, answer: int, confidence: float = 1.0, solver_name: str = "unknown", reasoning: Optional[str] = None):
        """
        Initialize solver result.

        Args:
            answer: Predicted answer
            confidence: Confidence score [0, 1]
            solver_name: Name of the solver
            reasoning: Optional reasoning provided
        """
        self.answer = answer
        self.confidence = confidence
        self.solver_name = solver_name
        self.reasoning = reasoning


class EnsembleSolver:
    """
    Advanced ensemble framework for combining multiple solvers.
    """

    def __init__(
        self,
        solvers: List[SupportsSolve],
        weights: Optional[List[float]] = None,
        method: str = "weighted_majority",
        use_verification: bool = True,
        min_agreement: float = 0.5,
    ):
        """
        Initialize ensemble solver.

        Args:
            solvers: List of solver instances
            weights: Optional weights for each solver (default: equal weights)
            method: Ensemble method:
                - "majority_vote": Simple majority voting
                - "weighted_majority": Weighted majority voting
                - "confidence_weighted": Weight by solver confidence
                - "stacking": Use meta-learner (requires training)
                - "consensus": Require agreement between solvers
            use_verification: Whether to verify solutions
            min_agreement: Minimum agreement threshold for consensus method
        """
        self.solvers: List[SupportsSolve] = solvers
        if solvers:
            self.weights = weights or [1.0 / len(solvers)] * len(solvers)
            if len(self.weights) != len(solvers):
                self.weights = [1.0 / len(solvers)] * len(solvers)
        else:
            self.weights = []

        self.method = method
        self.use_verification = use_verification
        self.min_agreement = min_agreement
        self.verifier: Optional[SolutionVerifier] = SolutionVerifier() if use_verification else None

        # Performance tracking
        self.solver_performance: Dict[str, Dict[str, float]] = {}

    def solve(self, problem_statement: str) -> int:
        """
        Solve problem using ensemble of solvers.

        Args:
            problem_statement: LaTeX problem statement

        Returns:
            Answer as integer in [0, 99999]
        """
        results = self._get_solver_results(problem_statement)

        if not results:
            return 0

        # Filter by verification if enabled
        if self.use_verification and self.verifier:
            verified_results = []
            for result in results:
                is_valid, confidence, _ = self.verifier.verify(problem_statement, result.answer)
                if is_valid:
                    result.confidence *= confidence  # Adjust confidence
                    verified_results.append(result)
            results = verified_results if verified_results else results

        # Combine results based on method
        if self.method == "majority_vote":
            return self._majority_vote(results)
        elif self.method == "weighted_majority":
            return self._weighted_majority(results)
        elif self.method == "confidence_weighted":
            return self._confidence_weighted(results)
        elif self.method == "consensus":
            return self._consensus(results)
        elif self.method == "best_confidence":
            return self._best_confidence(results)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def _get_solver_results(self, problem_statement: str) -> List[SolverResult]:
        """Get results from all solvers."""
        results: List[SolverResult] = []

        for i, solver in enumerate(self.solvers):
            try:
                answer = solver.solve(problem_statement)
                if 0 <= answer <= 99999:
                    # Try to get confidence if solver supports it
                    confidence = float(getattr(solver, "confidence", 1.0))
                    solver_name = type(solver).__name__
                    
                    results.append(SolverResult(
                        answer=answer,
                        confidence=confidence * (self.weights[i] if i < len(self.weights) else 1.0),  # Apply weight
                        solver_name=solver_name,
                    ))
            except Exception as e:
                print(f"Solver {type(solver).__name__} failed: {e}")

        return results

    def _majority_vote(self, results: List[SolverResult]) -> int:
        """Simple majority voting."""
        from collections import Counter
        answers = [r.answer for r in results]
        counter = Counter(answers)
        return counter.most_common(1)[0][0]

    def _weighted_majority(self, results: List[SolverResult]) -> int:
        """Weighted majority voting."""
        weighted_votes: defaultdict[int, float] = defaultdict(float)

        for result in results:
            weighted_votes[result.answer] += result.confidence

        return max(weighted_votes.items(), key=lambda x: x[1])[0]

    def _confidence_weighted(self, results: List[SolverResult]) -> int:
        """Weight by solver confidence scores."""
        weighted_sum: defaultdict[int, float] = defaultdict(float)
        total_weight: defaultdict[int, float] = defaultdict(float)

        for result in results:
            weighted_sum[result.answer] += result.answer * result.confidence
            total_weight[result.answer] += result.confidence

        # Return answer with highest weighted average
        best_answer: Optional[int] = None
        best_score: float = -1.0

        for answer in weighted_sum:
            score = weighted_sum[answer] / total_weight[answer] if total_weight[answer] > 0 else 0.0
            if score > best_score:
                best_score = score
                best_answer = answer

        return best_answer if best_answer is not None else results[0].answer

    def _consensus(self, results: List[SolverResult]) -> int:
        """Require consensus between solvers."""
        from collections import Counter
        answers = [r.answer for r in results]
        counter = Counter(answers)

        if not counter:
            return 0

        most_common_answer, count = counter.most_common(1)[0]
        agreement_ratio = count / len(results)

        if agreement_ratio >= self.min_agreement:
            return most_common_answer
        else:
            # Fallback to weighted majority if no consensus
            return self._weighted_majority(results)

    def _best_confidence(self, results: List[SolverResult]) -> int:
        """Return answer from solver with highest confidence."""
        if not results:
            return 0
        return max(results, key=lambda r: r.confidence).answer

    def update_solver_weights(self, problem_id: str, correct_answer: int):
        """
        Update solver weights based on performance.

        Args:
            problem_id: Problem identifier
            correct_answer: Correct answer
        """
        # This would track performance and adjust weights
        # Implementation would require historical performance data
        pass

    def get_ensemble_confidence(self, problem_statement: str) -> float:
        """
        Get confidence score for ensemble prediction.

        Args:
            problem_statement: Problem statement

        Returns:
            Confidence score [0, 1]
        """
        results = self._get_solver_results(problem_statement)

        if not results:
            return 0.0

        # Calculate agreement
        from collections import Counter
        answers = [r.answer for r in results]
        counter = Counter(answers)
        if counter:
            most_common_count = counter.most_common(1)[0][1]
            agreement = most_common_count / len(results)

            # Average confidence
            avg_confidence = sum(r.confidence for r in results) / len(results)

            # Combined confidence
            return (agreement + avg_confidence) / 2.0

        return 0.0

