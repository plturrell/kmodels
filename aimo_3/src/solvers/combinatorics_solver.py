"""Combinatorics solver for counting, probability, and graph problems."""

import math
import re
from typing import List, Optional

from .base import BaseSolver


class CombinatoricsSolver(BaseSolver):
    """
    Solver for combinatorics problems.
    
    Handles:
    - Permutations and combinations
    - Probability calculations
    - Graph counting problems
    - Arrangement problems
    """

    def __init__(self):
        """Initialize combinatorics solver."""
        super().__init__("combinatorics")

    def solve(self, problem_statement: str) -> int:
        """
        Solve combinatorics problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Integer answer
        """
        problem_lower = problem_statement.lower()

        # Permutations
        if "permutation" in problem_lower or "arrange" in problem_lower or "order" in problem_lower:
            return self._solve_permutation(problem_statement)

        # Combinations
        if "combination" in problem_lower or "choose" in problem_lower or "select" in problem_lower:
            return self._solve_combination(problem_statement)

        # Probability
        if "probability" in problem_lower or "prob" in problem_lower or "chance" in problem_lower:
            return self._solve_probability(problem_statement)

        # Ways/Paths
        if "ways" in problem_lower or "paths" in problem_lower or "how many" in problem_lower:
            return self._solve_counting(problem_statement)

        # Factorial
        if "factorial" in problem_lower or "!" in problem_statement:
            return self._solve_factorial(problem_statement)

        return 0

    def can_solve(self, problem_statement: str) -> bool:
        """Check if problem is combinatorics."""
        problem_lower = problem_statement.lower()
        comb_keywords = [
            "permutation", "combination", "choose", "arrange", "order",
            "probability", "ways", "paths", "factorial", "select",
        ]
        return any(keyword in problem_lower for keyword in comb_keywords)

    def _solve_permutation(self, problem_statement: str) -> int:
        """Solve permutation problem: P(n, r) = n! / (n-r)!."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if len(numbers) >= 2:
            n = numbers[0]
            r = numbers[1] if len(numbers) > 1 else n
            # P(n, r) = n! / (n-r)!
            if n >= r and n - r >= 0:
                return math.perm(n, r)

        # Try to extract from "n choose r" or "P(n, r)" patterns
        perm_pattern = r'P\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        match = re.search(perm_pattern, problem_statement)
        if match:
            n = int(match.group(1))
            r = int(match.group(2))
            return math.perm(n, r)

        return 0

    def _solve_combination(self, problem_statement: str) -> int:
        """Solve combination problem: C(n, r) = n! / (r! * (n-r)!)."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if len(numbers) >= 2:
            n = numbers[0]
            r = numbers[1] if len(numbers) > 1 else n
            # C(n, r) = n! / (r! * (n-r)!)
            if n >= r and n - r >= 0:
                return math.comb(n, r)

        # Try to extract from "C(n, r)" or "n choose r" patterns
        comb_patterns = [
            r'C\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)',
            r'(\d+)\s+choose\s+(\d+)',
            r'(\d+)\s+C\s+(\d+)',
        ]
        for pattern in comb_patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE)
            if match:
                n = int(match.group(1))
                r = int(match.group(2))
                return math.comb(n, r)

        return 0

    def _solve_probability(self, problem_statement: str) -> int:
        """Solve probability problem."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if len(numbers) >= 2:
            # Simple probability: favorable / total
            favorable = numbers[0]
            total = numbers[1]
            if total > 0:
                # Return as integer (multiply by 100 for percentage, or return as-is)
                # For AIMO, might need to return probability * some_factor
                return favorable  # Simplified - would need more context

        # Pattern: "probability of X is Y/Z"
        prob_pattern = r'probability.*?(\d+)\s*/\s*(\d+)'
        match = re.search(prob_pattern, problem_statement, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            den = int(match.group(2))
            if den > 0:
                # Return numerator (or could return simplified fraction)
                return num

        return 0

    def _solve_counting(self, problem_statement: str) -> int:
        """Solve counting problem (ways, paths, etc.)."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if not numbers:
            return 0

        # Try to identify counting pattern
        # "How many ways to arrange n objects" -> n!
        if "arrange" in problem_statement.lower() and len(numbers) >= 1:
            n = numbers[0]
            return math.factorial(n)

        # "How many ways to choose r from n" -> C(n, r)
        if "choose" in problem_statement.lower() and len(numbers) >= 2:
            n = numbers[0]
            r = numbers[1]
            return math.comb(n, r)

        # Default: return first number (simplified)
        return numbers[0] if numbers else 0

    def _solve_factorial(self, problem_statement: str) -> int:
        """Solve factorial problem: n!."""
        # Extract number before !
        fact_pattern = r'(\d+)\s*!'
        match = re.search(fact_pattern, problem_statement)
        if match:
            n = int(match.group(1))
            if n <= 20:  # Reasonable limit for factorial
                return math.factorial(n)

        # Extract numbers and assume first is n
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]
        if numbers:
            n = numbers[0]
            if n <= 20:
                return math.factorial(n)

        return 0


class GraphSolver(BaseSolver):
    """
    Solver for graph theory problems.
    
    Handles:
    - Graph counting
    - Path counting
    - Tree problems
    """

    def __init__(self):
        """Initialize graph solver."""
        super().__init__("graph")

    def solve(self, problem_statement: str) -> int:
        """Solve graph problem."""
        problem_lower = problem_statement.lower()

        # Path counting
        if "path" in problem_lower:
            return self._solve_paths(problem_statement)

        # Graph counting
        if "graph" in problem_lower or "vertex" in problem_lower or "edge" in problem_lower:
            return self._solve_graph_counting(problem_statement)

        return 0

    def can_solve(self, problem_statement: str) -> bool:
        """Check if problem is graph theory."""
        problem_lower = problem_statement.lower()
        graph_keywords = ["graph", "vertex", "edge", "path", "tree", "cycle"]
        return any(keyword in problem_lower for keyword in graph_keywords)

    def _solve_paths(self, problem_statement: str) -> int:
        """Solve path counting problem."""
        # Extract numbers (vertices, edges)
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if len(numbers) >= 2:
            # Simple path counting: if n vertices, number of paths might be related
            n = numbers[0]
            # Simplified: return some function of n
            # In practice, would need to parse graph structure
            return n * (n - 1) // 2  # Complete graph edges

        return 0

    def _solve_graph_counting(self, problem_statement: str) -> int:
        """Solve graph counting problem."""
        # Extract numbers
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if numbers:
            n = numbers[0]
            # Number of possible graphs on n vertices: 2^(n choose 2)
            if n <= 10:  # Reasonable limit
                num_edges = math.comb(n, 2)
                return 2 ** num_edges if num_edges <= 20 else 0

        return 0

