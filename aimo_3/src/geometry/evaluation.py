"""Evaluation function F: (G_n, Φ_n) → ℝ."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import sympy as sp

from .primitives import Circle, Line, Point
from .relations import RelationType
from .scene_graph import GeometricSceneGraph
from .state import State


class EvaluationEngine:
    """
    Evaluation function F: (G_n, Φ_n) → ℝ.
    
    Extracts constraints from final state and computes numeric answer.
    """

    def __init__(self):
        """Initialize evaluation engine."""
        self.symbols: Dict[str, sp.Symbol] = {}
        self.equations: List[sp.Eq] = []
        self.use_cache: bool = True
        self.solve_cache: Dict[str, int] = {}

    def evaluate(self, state: State, problem_goal: str) -> int:
        """
        Evaluate final state to compute answer.

        Args:
            state: Final state (G_n, Φ_n)
            problem_goal: Goal statement from problem

        Returns:
            Integer answer k in [0, 99999]
        """
        # Extract constraints from state
        constraints = self._extract_constraints(state)

        # Extract goal quantity
        goal_quantity = self._extract_goal_quantity(problem_goal, state)

        # Solve constraint system
        answer = self._solve_constraints(constraints, goal_quantity)

        return answer

    def _extract_constraints(self, state: State) -> List[sp.Eq]:
        """
        Extract equations from state propositions and graph.

        Args:
            state: Current state

        Returns:
            List of symbolic equations
        """
        equations = []

        # Extract from propositions
        for prop in state.propositions:
            eq = self._proposition_to_equation(prop)
            if eq:
                equations.append(eq)

        # Extract from graph structure
        graph_eqs = self._extract_graph_constraints(state.graph)
        equations.extend(graph_eqs)

        return equations

    def _proposition_to_equation(self, proposition: str) -> Optional[sp.Eq]:
        """Convert proposition to symbolic equation."""
        # Pythagorean: a² + b² = c²
        pythagorean_match = re.search(
            r'pythagorean.*?:\s*(\w+)²\s*\+\s*(\w+)²\s*=\s*(\w+)²', proposition, re.IGNORECASE
        )
        if pythagorean_match:
            a, b, c = pythagorean_match.groups()
            a_sym = self._get_symbol(a)
            b_sym = self._get_symbol(b)
            c_sym = self._get_symbol(c)
            return sp.Eq(a_sym**2 + b_sym**2, c_sym**2)

        # Angle sum: ∠A + ∠B + ∠C = 180
        angle_sum_match = re.search(
            r'angle_sum.*?:\s*∠(\w+)\s*\+\s*∠(\w+)\s*\+\s*∠(\w+)\s*=\s*180', proposition, re.IGNORECASE
        )
        if angle_sum_match:
            A, B, C = angle_sum_match.groups()
            A_sym = self._get_symbol(f"angle_{A}")
            B_sym = self._get_symbol(f"angle_{B}")
            C_sym = self._get_symbol(f"angle_{C}")
            return sp.Eq(A_sym + B_sym + C_sym, 180)

        # Isosceles base angles: ∠B = ∠C
        isosceles_match = re.search(
            r'isosceles_base_angles.*?:\s*∠(\w+)\s*=\s*∠(\w+)', proposition, re.IGNORECASE
        )
        if isosceles_match:
            B, C = isosceles_match.groups()
            B_sym = self._get_symbol(f"angle_{B}")
            C_sym = self._get_symbol(f"angle_{C}")
            return sp.Eq(B_sym, C_sym)

        # Equilateral: all angles 60°
        equilateral_match = re.search(
            r'equilateral.*?:\s*∠(\w+)\s*=\s*∠(\w+)\s*=\s*∠(\w+)\s*=\s*60', proposition, re.IGNORECASE
        )
        if equilateral_match:
            A, B, C = equilateral_match.groups()
            A_sym = self._get_symbol(f"angle_{A}")
            return sp.Eq(A_sym, 60)

        # Triangle height: h = 2*area / base
        height_match = re.search(
            r'triangle_height.*?:\s*h\s*=\s*2\*area\s*/\s*base', proposition, re.IGNORECASE
        )
        if height_match:
            # Would need to extract actual values
            pass

        # Distance formula: AB = √[(x_B-x_A)² + (y_B-y_A)²]
        distance_match = re.search(
            r'distance_formula.*?:\s*(\w+)\s*=\s*√\[\(x_(\w+)-x_(\w+)\)²\s*\+\s*\(y_(\w+)-y_(\w+)\)²\]', 
            proposition, re.IGNORECASE
        )
        if distance_match:
            AB, B, A, B2, A2 = distance_match.groups()
            # Extract coordinates if available
            x_A = self._get_symbol(f"x_{A}")
            y_A = self._get_symbol(f"y_{A}")
            x_B = self._get_symbol(f"x_{B}")
            y_B = self._get_symbol(f"y_{B}")
            AB_sym = self._get_symbol(AB)
            return sp.Eq(AB_sym**2, (x_B - x_A)**2 + (y_B - y_A)**2)

        # Midpoint formula
        midpoint_match = re.search(
            r'midpoint_formula.*?:\s*(\w+)\s*=\s*\(\(x_(\w+)\+x_(\w+)\)/2,\s*\(y_(\w+)\+y_(\w+)\)/2\)',
            proposition, re.IGNORECASE
        )
        if midpoint_match:
            # Would extract midpoint coordinates
            pass

        # Slope formula: m = (y_B-y_A)/(x_B-x_A)
        slope_match = re.search(
            r'slope_formula.*?:\s*m\s*=\s*\(y_(\w+)-y_(\w+)\)/\(x_(\w+)-x_(\w+)\)',
            proposition, re.IGNORECASE
        )
        if slope_match:
            # Would extract slope
            pass

        # Equal lengths: AB = CD
        equal_match = re.search(r'(\w+)\s*=\s*(\w+)', proposition)
        if equal_match:
            left, right = equal_match.groups()
            left_sym = self._get_symbol(left)
            right_sym = self._get_symbol(right)
            return sp.Eq(left_sym, right_sym)

        return None

    def _extract_graph_constraints(self, graph: GeometricSceneGraph) -> List[sp.Eq]:
        """Extract constraints from graph structure."""
        equations = []

        # Extract from relations
        relations = graph.get_relations()

        for relation in relations:
            if relation.relation_type == RelationType.EQUAL_LENGTH:
                # Two segments are equal
                if isinstance(relation.source, Line) and isinstance(relation.target, Line):
                    source_sym = self._get_symbol(f"len_{relation.source}")
                    target_sym = self._get_symbol(f"len_{relation.target}")
                    equations.append(sp.Eq(source_sym, target_sym))

            elif relation.relation_type == RelationType.RIGHT_ANGLE:
                # Right angle constraint
                # Would need to extract angle measure
                pass

        return equations

    def _extract_goal_quantity(self, problem_goal: str, state: State) -> Optional[sp.Symbol]:
        """
        Extract the quantity to compute from goal statement.

        Args:
            problem_goal: Goal statement
            state: Current state

        Returns:
            Symbol representing goal quantity
        """
        # Look for common patterns
        # "Find the length of AB"
        length_match = re.search(r'length\s+of\s+([A-Z][A-Z]?)', problem_goal, re.IGNORECASE)
        if length_match:
            segment = length_match.group(1)
            return self._get_symbol(f"len_{segment}")

        # "Find the measure of angle ABC"
        angle_match = re.search(r'(?:measure\s+of\s+)?angle\s+([A-Z])([A-Z])([A-Z])', problem_goal, re.IGNORECASE)
        if angle_match:
            vertex = angle_match.group(2)
            return self._get_symbol(f"angle_{vertex}")

        # "Find the area of triangle ABC"
        area_match = re.search(r'area\s+of\s+triangle\s+([A-Z])([A-Z])([A-Z])', problem_goal, re.IGNORECASE)
        if area_match:
            triangle = "".join(area_match.groups())
            return self._get_symbol(f"area_{triangle}")

        # Generic: look for "Find X" or "What is X"
        find_match = re.search(r'(?:find|what is|compute)\s+([A-Z][A-Z]?)', problem_goal, re.IGNORECASE)
        if find_match:
            quantity = find_match.group(1)
            return self._get_symbol(quantity)

        return None

    def _solve_constraints(
        self, constraints: List[sp.Eq], goal_quantity: Optional[sp.Symbol]
    ) -> int:
        """
        Solve constraint system to find goal quantity.

        Args:
            constraints: List of equations
            goal_quantity: Symbol to solve for

        Returns:
            Integer answer
        """
        # Check cache
        if self.use_cache:
            cache_key = str((tuple(str(eq) for eq in constraints), str(goal_quantity)))
            cached = self.solve_cache.get(cache_key)
            if cached is not None:
                return cached
        if not constraints:
            return 0

        if goal_quantity is None:
            # Try to solve for any variable
            all_symbols = set()
            for eq in constraints:
                all_symbols.update(eq.free_symbols)
            if all_symbols:
                goal_quantity = list(all_symbols)[0]

        if goal_quantity is None:
            return 0

        try:
            # Try solving system first
            if len(constraints) > 1:
                # System of equations
                solution = sp.solve(constraints, goal_quantity, dict=True, manual=True)
            else:
                # Single equation
                solution = sp.solve(constraints[0], goal_quantity, dict=True, manual=True)

            if solution:
                # Extract numeric value
                value = solution[0].get(goal_quantity) if isinstance(solution, list) else solution.get(goal_quantity)
                if value is not None:
                    # Convert to integer
                    if isinstance(value, (int, sp.Integer)):
                        answer = int(value)
                    elif isinstance(value, (float, sp.Float)):
                        answer = int(round(value))
                    else:
                        # Try to evaluate
                        answer = int(round(float(value.evalf())))

                    # Ensure in valid range
                    if 0 <= answer <= 99999:
                        # Cache result
                        if self.use_cache:
                            cache_key = str((tuple(str(eq) for eq in constraints), str(goal_quantity)))
                            self.solve_cache[cache_key] = answer
                        return answer

            # Fallback: try numerical solving
            try:
                from scipy.optimize import fsolve
                import numpy as np

                def equations(vars):
                    # Convert symbolic equations to numerical functions
                    subs_dict = {goal_quantity: vars[0]}
                    return [float(eq.subs(subs_dict).evalf()) for eq in constraints]

                initial_guess = [1.0]
                result = fsolve(equations, initial_guess)
                answer = int(round(result[0]))
                if 0 <= answer <= 99999:
                    return answer
            except ImportError:
                pass  # scipy not available
            except Exception:
                pass

        except Exception as e:
            print(f"Error solving constraints: {e}")

        return 0

    def _get_symbol(self, name: str) -> sp.Symbol:
        """Get or create symbol for a quantity."""
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name, real=True, positive=True)
        return self.symbols[name]


def evaluate_state(state: State, problem_goal: str) -> int:
    """
    Convenience function to evaluate a state.

    Args:
        state: Final state
        problem_goal: Goal statement

    Returns:
        Integer answer
    """
    engine = EvaluationEngine()
    return engine.evaluate(state, problem_goal)

