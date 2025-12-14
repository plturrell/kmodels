"""Solution verification system for AIMO problems."""

import re
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp


class SolutionVerifier:
    """
    Verifies solutions to mathematical problems using multiple strategies.
    """

    def __init__(self):
        """Initialize verifier."""
        pass

    def verify(
        self,
        problem_statement: str,
        answer: int,
        verification_methods: Optional[List[str]] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify a solution using multiple methods.

        Args:
            problem_statement: Problem statement
            answer: Proposed answer
            verification_methods: List of verification methods to use

        Returns:
            (is_valid, confidence, details) tuple
        """
        if verification_methods is None:
            verification_methods = ["range_check", "sanity_check", "symbolic_verification"]

        results: Dict[str, Any] = {}
        validations = []

        for method in verification_methods:
            if method == "range_check":
                is_valid, confidence = self._range_check(answer)
                results["range_check"] = {"valid": is_valid, "confidence": confidence}
                validations.append((is_valid, confidence))

            elif method == "sanity_check":
                is_valid, confidence = self._sanity_check(problem_statement, answer)
                results["sanity_check"] = {"valid": is_valid, "confidence": confidence}
                validations.append((is_valid, confidence))

            elif method == "symbolic_verification":
                is_valid, confidence = self._symbolic_verification(problem_statement, answer)
                results["symbolic_verification"] = {"valid": is_valid, "confidence": confidence}
                validations.append((is_valid, confidence))

            elif method == "back_substitution":
                is_valid, confidence = self._back_substitution(problem_statement, answer)
                results["back_substitution"] = {"valid": is_valid, "confidence": confidence}
                validations.append((is_valid, confidence))

        # Aggregate results
        overall_valid = all(v[0] for v in validations)
        overall_confidence = sum(v[1] for v in validations) / len(validations) if validations else 0.0

        return overall_valid, overall_confidence, results

    def _range_check(self, answer: int) -> Tuple[bool, float]:
        """Check if answer is in valid range."""
        is_valid = 0 <= answer <= 99999
        confidence = 1.0 if is_valid else 0.0
        return is_valid, confidence

    def _sanity_check(self, problem_statement: str, answer: int) -> Tuple[bool, float]:
        """
        Perform sanity checks on the answer.

        Args:
            problem_statement: Problem statement
            answer: Proposed answer

        Returns:
            (is_valid, confidence)
        """
        confidence = 0.5  # Start with neutral confidence

        # Check for obvious issues
        if answer < 0:
            return False, 0.0

        # Check if answer seems reasonable based on problem type
        problem_lower = problem_statement.lower()

        # For remainder problems, answer should be less than divisor
        if "remainder" in problem_lower or "mod" in problem_lower:
            # Try to extract divisor
            mod_match = re.search(r'(?:mod|divided by|÷)\s*(\d+)', problem_lower)
            if mod_match:
                divisor = int(mod_match.group(1))
                if answer >= divisor:
                    return False, 0.2  # Low confidence that it's wrong

        # For factorial problems, answer should be positive
        if "factorial" in problem_lower or "!" in problem_statement:
            if answer < 1:
                return False, 0.0

        # For problems asking for "number of", answer should be non-negative integer
        if "number of" in problem_lower or "how many" in problem_lower:
            if answer < 0:
                return False, 0.0

        return True, confidence

    def _symbolic_verification(self, problem_statement: str, answer: int) -> Tuple[bool, float]:
        """
        Attempt symbolic verification using SymPy.

        Args:
            problem_statement: Problem statement
            answer: Proposed answer

        Returns:
            (is_valid, confidence)
        """
        try:
            # Try to extract equations or expressions
            # This is a simplified version - full implementation would parse LaTeX

            # Look for equality statements
            equality_match = re.search(r'(\w+)\s*=\s*(\d+)', problem_statement)
            if equality_match:
                # Try to verify
                # This is a placeholder - would need full LaTeX parsing
                return True, 0.6

            # If no clear equation, can't verify symbolically
            return True, 0.3  # Neutral - can't verify but no reason to reject

        except Exception:
            return True, 0.3  # If verification fails, don't reject

    def _back_substitution(self, problem_statement: str, answer: int) -> Tuple[bool, float]:
        """
        Verify by back-substituting answer into problem.

        Args:
            problem_statement: Problem statement
            answer: Proposed answer

        Returns:
            (is_valid, confidence)
        """
        # This would require parsing the problem and checking if answer satisfies conditions
        # For now, return neutral
        return True, 0.4

    def verify_with_reasoning(
        self,
        problem_statement: str,
        answer: int,
        reasoning: str,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify solution including reasoning check.

        Args:
            problem_statement: Problem statement
            answer: Proposed answer
            reasoning: Reasoning steps provided

        Returns:
            (is_valid, confidence, details)
        """
        # First verify answer
        is_valid, confidence, details = self.verify(problem_statement, answer)

        # Check if reasoning mentions the answer
        reasoning_lower = reasoning.lower()
        answer_mentioned = str(answer) in reasoning_lower

        if answer_mentioned:
            confidence = min(confidence + 0.1, 1.0)
        else:
            confidence = max(confidence - 0.1, 0.0)

        details["reasoning_check"] = {
            "answer_mentioned": answer_mentioned,
            "adjusted_confidence": confidence,
        }

        return is_valid, confidence, details


def verify_solution(problem: str, answer: int) -> bool:
    """
    Convenience function to verify a solution.

    Args:
        problem: Problem statement
        answer: Proposed answer

    Returns:
        True if solution appears valid
    """
    verifier = SolutionVerifier()
    is_valid, confidence, _ = verifier.verify(problem, answer)
    return is_valid and confidence > 0.5

