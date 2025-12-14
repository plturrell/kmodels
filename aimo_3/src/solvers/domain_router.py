"""Domain router for multi-domain problem solving."""

from typing import Dict, List, Optional

from .base import BaseSolver, SolverResult


class DomainRouter:
    """
    Routes problems to appropriate domain-specific solvers.
    
    Uses keyword matching and problem structure analysis to determine
    which solver should handle a given problem.
    """

    def __init__(self):
        """Initialize domain router."""
        self.solvers: Dict[str, BaseSolver] = {}
        self.domain_keywords = {
            "geometry": [
                "triangle", "circle", "angle", "length", "area", "perimeter",
                "parallel", "perpendicular", "tangent", "inscribed", "circumscribed",
                "point", "line", "segment", "polygon", "radius", "diameter",
            ],
            "algebra": [
                "polynomial", "equation", "inequality", "solve", "factor",
                "expand", "simplify", "quadratic", "linear", "system",
                "variable", "coefficient", "root", "zero",
            ],
            "number_theory": [
                "modulo", "mod", "divisible", "prime", "gcd", "lcm",
                "remainder", "factor", "multiple", "divisor", "congruent",
                "integer", "digit", "sum of digits",
            ],
            "combinatorics": [
                "count", "permutation", "combination", "choose", "arrange",
                "probability", "ways", "paths", "factorial", "select",
            ],
            "graph": [
                "graph", "vertex", "edge", "path", "tree", "cycle",
            ],
            "analysis": [
                "limit", "derivative", "integral", "differentiate", "integrate",
                "sequence", "series", "converge", "diverge", "calculus",
            ],
        }

    def register_solver(self, solver: BaseSolver) -> None:
        """
        Register a domain solver.

        Args:
            solver: Solver instance
        """
        self.solvers[solver.domain] = solver

    def route(self, problem_statement: str) -> Optional[str]:
        """
        Determine which domain should handle the problem.

        Args:
            problem_statement: Problem statement

        Returns:
            Domain name or None if cannot determine
        """
        problem_lower = problem_statement.lower()

        # Count keyword matches for each domain
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return None

        # Return domain with highest score
        return max(domain_scores.items(), key=lambda x: x[1])[0]

    def solve(self, problem_statement: str) -> SolverResult:
        """
        Route problem to appropriate solver and solve.

        Args:
            problem_statement: Problem statement

        Returns:
            SolverResult with answer
        """
        # Try to route to specific domain
        domain = self.route(problem_statement)

        if domain and domain in self.solvers:
            solver = self.solvers[domain]
            if solver.can_solve(problem_statement):
                return solver.solve_with_metadata(problem_statement)

        # Fallback: try all solvers
        results = []
        for solver in self.solvers.values():
            if solver.can_solve(problem_statement):
                result = solver.solve_with_metadata(problem_statement)
                if result.confidence > 0:
                    results.append(result)

        if results:
            # Return result with highest confidence
            return max(results, key=lambda r: r.confidence)

        # Default: return zero answer
        return SolverResult(
            answer=0,
            confidence=0.0,
            method="unknown",
            metadata={"error": "No solver could handle this problem"},
        )

    def solve_with_fallback(self, problem_statement: str) -> int:
        """
        Solve with fallback strategy.

        Args:
            problem_statement: Problem statement

        Returns:
            Integer answer
        """
        result = self.solve(problem_statement)
        return result.answer

