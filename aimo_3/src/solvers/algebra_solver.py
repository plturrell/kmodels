"""Algebra solver using SymPy for symbolic manipulation."""

import re
from typing import Dict, List, Optional

import sympy as sp

from .base import BaseSolver


class AlgebraSolver(BaseSolver):
    """
    Solver for algebraic problems using SymPy.
    
    Handles:
    - Equation solving (linear, quadratic, systems)
    - Polynomial operations
    - Symbolic manipulation
    - Basic inequalities
    """

    def __init__(self):
        """Initialize algebra solver."""
        super().__init__("algebra")
        self.symbols: Dict[str, sp.Symbol] = {}

    def solve(self, problem_statement: str) -> int:
        """
        Solve algebraic problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Integer answer
        """
        # Extract equations from problem
        equations = self._extract_equations(problem_statement)

        if not equations:
            return 0

        # Extract variables
        variables = self._extract_variables(problem_statement)

        # Extract goal (what to find)
        goal = self._extract_goal(problem_statement)

        # Solve system
        try:
            answer = self._solve_system(equations, variables, goal)
            return int(answer) if answer is not None else 0
        except Exception as e:
            print(f"Algebra solver error: {e}")
            return 0

    def can_solve(self, problem_statement: str) -> bool:
        """Check if problem is algebraic."""
        problem_lower = problem_statement.lower()
        algebra_keywords = [
            "equation", "solve", "polynomial", "factor", "expand",
            "quadratic", "linear", "system", "variable",
        ]
        return any(keyword in problem_lower for keyword in algebra_keywords)

    def _extract_equations(self, problem_statement: str) -> List[sp.Eq]:
        """Extract equations from problem statement."""
        equations = []

        # Pattern: "x = 5", "x + y = 10", etc.
        equation_patterns = [
            r'([A-Za-z0-9\s\+\-\*\/\(\)]+)\s*=\s*([A-Za-z0-9\s\+\-\*\/\(\)]+)',
        ]

        for pattern in equation_patterns:
            matches = re.finditer(pattern, problem_statement)
            for match in matches:
                left_str = match.group(1).strip()
                right_str = match.group(2).strip()

                try:
                    # Try to parse as SymPy expressions
                    left_expr = sp.sympify(left_str)
                    right_expr = sp.sympify(right_str)
                    equations.append(sp.Eq(left_expr, right_expr))
                except Exception:
                    pass

        return equations

    def _extract_variables(self, problem_statement: str) -> List[str]:
        """Extract variable names from problem."""
        # Find single letters that appear in equations
        var_pattern = r'\b([a-zA-Z])\b'
        matches = re.findall(var_pattern, problem_statement)
        # Filter common words
        common_words = {"the", "and", "or", "is", "are", "find", "what"}
        variables = [v for v in set(matches) if v.lower() not in common_words]
        return variables

    def _extract_goal(self, problem_statement: str) -> Optional[str]:
        """Extract what to find from problem."""
        find_patterns = [
            r'find\s+([A-Za-z0-9\s\+\-\*\/]+)',
            r'what\s+is\s+([A-Za-z0-9\s\+\-\*\/]+)',
            r'compute\s+([A-Za-z0-9\s\+\-\*\/]+)',
        ]

        for pattern in find_patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _solve_system(self, equations: List[sp.Eq], variables: List[str], goal: Optional[str]) -> Optional[sp.Number]:
        """Solve system of equations."""
        if not equations:
            return None

        # Get all symbols
        all_symbols = set()
        for eq in equations:
            all_symbols.update(eq.free_symbols)

        if not all_symbols:
            return None

        # Solve system
        try:
            solution = sp.solve(equations, list(all_symbols), dict=True)

            if not solution:
                return None

            # Extract goal value
            if goal:
                goal_expr = sp.sympify(goal)
                # Substitute solution
                result = goal_expr.subs(solution[0])
                if isinstance(result, (int, sp.Integer)):
                    return int(result)
                elif isinstance(result, (float, sp.Float)):
                    return int(round(result))
                else:
                    return int(round(float(result.evalf())))

            # If no goal, return first variable value
            first_var = list(all_symbols)[0]
            value = solution[0].get(first_var)
            if value is not None:
                if isinstance(value, (int, sp.Integer)):
                    return int(value)
                elif isinstance(value, (float, sp.Float)):
                    return int(round(value))
                else:
                    return int(round(float(value.evalf())))

        except Exception as e:
            print(f"Error solving system: {e}")
            return None

        return None

