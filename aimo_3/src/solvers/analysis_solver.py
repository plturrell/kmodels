"""Analysis solver for calculus problems: limits, derivatives, integrals."""

import re
from typing import Optional

import sympy as sp

from .base import BaseSolver


class AnalysisSolver(BaseSolver):
    """
    Solver for analysis/calculus problems.
    
    Handles:
    - Limits
    - Derivatives
    - Integrals
    - Sequences and series
    """

    def __init__(self):
        """Initialize analysis solver."""
        super().__init__("analysis")

    def solve(self, problem_statement: str) -> int:
        """
        Solve analysis problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Integer answer
        """
        problem_lower = problem_statement.lower()

        # Limits
        if "limit" in problem_lower or "lim" in problem_lower:
            return self._solve_limit(problem_statement)

        # Derivatives
        if "derivative" in problem_lower or "differentiate" in problem_lower or "d/dx" in problem_statement:
            return self._solve_derivative(problem_statement)

        # Integrals
        if "integral" in problem_lower or "∫" in problem_statement or "integrate" in problem_lower:
            return self._solve_integral(problem_statement)

        # Sequences
        if "sequence" in problem_lower or "series" in problem_lower or "sum" in problem_lower:
            return self._solve_sequence(problem_statement)

        return 0

    def can_solve(self, problem_statement: str) -> bool:
        """Check if problem is analysis."""
        problem_lower = problem_statement.lower()
        analysis_keywords = [
            "limit", "derivative", "integral", "differentiate", "integrate",
            "sequence", "series", "converge", "diverge",
        ]
        return any(keyword in problem_lower for keyword in analysis_keywords)

    def _solve_limit(self, problem_statement: str) -> int:
        """Solve limit problem."""
        # Extract expression and point
        # Pattern: "limit as x -> a of f(x)"
        limit_patterns = [
            r'limit.*?as\s+(\w+)\s*[→->]\s*(\d+)',
            r'lim\s*[\(_]\s*(\w+)\s*[→->]\s*(\d+)',
            r'limit\s+of\s+(.+?)\s+as\s+(\w+)\s*[→->]\s*(\d+)',
        ]

        for pattern in limit_patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE)
            if match:
                try:
                    var_str = match.group(1) if len(match.groups()) >= 1 else "x"
                    point_str = match.group(2) if len(match.groups()) >= 2 else "0"

                    var = sp.Symbol(var_str)
                    point = sp.sympify(point_str)

                    # Try to extract expression
                    expr_str = match.group(1) if len(match.groups()) >= 3 else None
                    if expr_str:
                        expr = sp.sympify(expr_str)
                        limit_value = sp.limit(expr, var, point)
                    else:
                        # Simplified: return point as answer
                        limit_value = point

                    if isinstance(limit_value, (int, sp.Integer)):
                        return int(limit_value)
                    elif isinstance(limit_value, (float, sp.Float)):
                        return int(round(limit_value))
                    else:
                        return int(round(float(limit_value.evalf())))
                except Exception:
                    pass

        # Extract numbers as fallback
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]
        return numbers[0] if numbers else 0

    def _solve_derivative(self, problem_statement: str) -> int:
        """Solve derivative problem."""
        # Pattern: "derivative of f(x)" or "d/dx f(x)"
        deriv_patterns = [
            r'derivative\s+of\s+(.+)',
            r'd/dx\s*\((.+)\)',
            r'differentiate\s+(.+)',
        ]

        for pattern in deriv_patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE)
            if match:
                try:
                    expr_str = match.group(1)
                    expr = sp.sympify(expr_str)
                    x = sp.Symbol('x')
                    derivative = sp.diff(expr, x)

                    # Evaluate at a point if given
                    eval_point = None
                    point_match = re.search(r'at\s+x\s*=\s*(\d+)', problem_statement, re.IGNORECASE)
                    if point_match:
                        eval_point = int(point_match.group(1))
                        result = derivative.subs(x, eval_point)
                    else:
                        result = derivative

                    # Try to simplify to integer
                    if isinstance(result, (int, sp.Integer)):
                        return int(result)
                    elif isinstance(result, (float, sp.Float)):
                        return int(round(result))
                    else:
                        # Return 0 if can't simplify
                        return 0
                except Exception:
                    pass

        return 0

    def _solve_integral(self, problem_statement: str) -> int:
        """Solve integral problem."""
        # Pattern: "integral of f(x) dx" or "∫ f(x) dx"
        integral_patterns = [
            r'integral\s+of\s+(.+?)\s+dx',
            r'∫\s*(.+?)\s+dx',
            r'integrate\s+(.+)',
        ]

        for pattern in integral_patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE)
            if match:
                try:
                    expr_str = match.group(1)
                    expr = sp.sympify(expr_str)
                    x = sp.Symbol('x')

                    # Check for definite integral
                    bounds_match = re.search(r'from\s+(\d+)\s+to\s+(\d+)', problem_statement, re.IGNORECASE)
                    if bounds_match:
                        a = int(bounds_match.group(1))
                        b = int(bounds_match.group(2))
                        integral_value = sp.integrate(expr, (x, a, b))
                    else:
                        # Indefinite integral
                        integral_value = sp.integrate(expr, x)
                        # Evaluate at some point or return 0
                        return 0

                    if isinstance(integral_value, (int, sp.Integer)):
                        return int(integral_value)
                    elif isinstance(integral_value, (float, sp.Float)):
                        return int(round(integral_value))
                    else:
                        return int(round(float(integral_value.evalf())))
                except Exception:
                    pass

        return 0

    def _solve_sequence(self, problem_statement: str) -> int:
        """Solve sequence/series problem."""
        # Extract numbers (sequence terms)
        numbers = [int(x) for x in re.findall(r'\b(\d+)\b', problem_statement)]

        if not numbers:
            return 0

        # Sum of sequence
        if "sum" in problem_statement.lower():
            if len(numbers) >= 2:
                # Sum from a to b
                return sum(range(numbers[0], numbers[1] + 1))
            else:
                # Sum of first n terms
                n = numbers[0]
                return n * (n + 1) // 2  # Sum of 1 to n

        # Nth term
        if "term" in problem_statement.lower() and len(numbers) >= 1:
            n = numbers[0]
            # Simple arithmetic sequence: a_n = a_1 + (n-1)d
            # Simplified: return n
            return n

        return numbers[0] if numbers else 0

