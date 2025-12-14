"""Stability metrics computation for geometric proofs."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

from .state import State
from .theorems import Theorem


@dataclass
class ReasoningStabilityMetrics:
    """Lightweight container for reasoning stability results.

    This matches the expectations of geometry.search, which uses it as a
    return type for search stability analysis.
    """

    status: str
    lambda_max: float
    confidence: float


def estimate_search_stability(
    search_fn,
    initial_state: State,
    search_params: Dict[str, float],
    perturbation_epsilon: float,
    num_runs: int,
    horizon_steps: int,
) -> ReasoningStabilityMetrics:
    """Estimate search stability under small perturbations.

    This is a lightweight implementation intended primarily to unblock
    downstream imports. It runs the search multiple times with perturbed
    parameters and aggregates a simple stability score based on how often
    the final state changes.
    """

    if num_runs <= 0:
        return ReasoningStabilityMetrics(
            status="stable",
            lambda_max=0.0,
            confidence=1.0,
        )

    baseline_final, _ = search_fn(initial_state, search_params)
    divergences = 0

    for _ in range(num_runs):
        perturbed_params = dict(search_params)
        # Simple perturbation of max_iterations
        if "max_iterations" in perturbed_params:
            perturbed_params["max_iterations"] = max(
                1,
                int(perturbed_params["max_iterations"] * (1.0 + perturbation_epsilon)),
            )
        final_state, _ = search_fn(initial_state, perturbed_params)
        if final_state is not baseline_final:
            divergences += 1

    divergence_ratio = divergences / num_runs

    # Map divergence ratio to a rough lambda_max and confidence
    lambda_max = math.log1p(divergence_ratio) if divergence_ratio > 0 else 0.0
    confidence = 1.0 - divergence_ratio
    status = "stable" if divergence_ratio < 0.33 else ("unstable" if divergence_ratio > 0.66 else "marginal")

    return ReasoningStabilityMetrics(
        status=status,
        lambda_max=lambda_max,
        confidence=confidence,
    )


class StabilityMetrics:
    """
    Computes Lyapunov stability metrics for geometric proofs.
    
    Measures stability of proof sequences and theorem applications.
    """

    @staticmethod
    def compute_proof_stability(
        initial_state: State,
        final_state: State,
        theorem_sequence: List[Tuple[Theorem, Dict[str, str]]],
    ) -> Dict[str, float]:
        """
        Compute stability metrics for a proof sequence.

        Args:
            initial_state: Initial state
            final_state: Final state after proof
            theorem_sequence: Sequence of (theorem, match) tuples

        Returns:
            Dictionary with stability metrics
        """
        if not theorem_sequence:
            return {
                "status": "stable",
                "lyapunov_exponent": 0.0,
                "confidence": 1.0,
            }

        # Compute Lyapunov exponent (simplified)
        # Measures sensitivity to initial conditions
        lyapunov = StabilityMetrics._compute_lyapunov_exponent(
            initial_state, final_state, theorem_sequence
        )

        # Determine stability status
        if lyapunov < -0.1:
            status = "stable"
            confidence = 0.9
        elif lyapunov < 0.1:
            status = "marginally_stable"
            confidence = 0.6
        else:
            status = "unstable"
            confidence = 0.3

        return {
            "status": status,
            "lyapunov_exponent": lyapunov,
            "confidence": confidence,
            "proof_length": len(theorem_sequence),
        }

    @staticmethod
    def _compute_lyapunov_exponent(
        initial_state: State,
        final_state: State,
        theorem_sequence: List[Tuple[Theorem, Dict[str, str]]],
    ) -> float:
        """
        Compute Lyapunov exponent for proof sequence.

        Args:
            initial_state: Initial state
            final_state: Final state
            theorem_sequence: Proof sequence

        Returns:
            Lyapunov exponent (negative = stable, positive = unstable)
        """
        if not theorem_sequence:
            return 0.0

        # Simplified computation:
        # Measure how much the state changes relative to proof length
        initial_prop_count = len(initial_state.propositions)
        final_prop_count = len(final_state.propositions)
        
        prop_growth = (final_prop_count - initial_prop_count) / max(len(theorem_sequence), 1)
        
        # Normalize
        if initial_prop_count > 0:
            normalized_growth = prop_growth / initial_prop_count
        else:
            normalized_growth = prop_growth

        # Lyapunov exponent: negative for stable (bounded growth)
        # Positive for unstable (unbounded growth)
        lyapunov = math.log(1 + abs(normalized_growth)) if normalized_growth != 0 else 0.0
        
        # Make it negative for stable proofs (bounded)
        if normalized_growth < 1.0:  # Bounded growth
            lyapunov = -lyapunov

        return lyapunov

    @staticmethod
    def compute_theorem_stability(
        theorem: Theorem,
        state: State,
        match: Dict[str, str],
    ) -> float:
        """
        Compute stability of a single theorem application.

        Args:
            theorem: Theorem being applied
            state: Current state
            match: Match dictionary

        Returns:
            Stability score (0-1, higher = more stable)
        """
        # Simplified: measure how well theorem matches state
        # More specific matches = more stable
        
        # Check if theorem conditions are well-satisfied
        conditions_met = theorem.check_conditions(state, match)
        
        if not conditions_met:
            return 0.0

        # Measure match quality (simplified)
        match_quality = len(match) / 10.0  # Normalize
        match_quality = min(match_quality, 1.0)

        return match_quality
