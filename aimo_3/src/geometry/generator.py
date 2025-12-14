"""Problem generator: Synthesize novel geometric problems (S, k)."""

import random
from typing import Dict, List, Optional, Tuple

import sympy as sp

from .primitives import Circle, Line, Point
from .relations import Relation, RelationType
from .scene_graph import GeometricSceneGraph
from .state import State


class ProblemGenerator:
    """
    Generate geometric problems by sampling valid initial graphs G
    and applying inverse search to synthesize problems (S, k).
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize problem generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            # Note: sympy doesn't have a global random seed, but we use it deterministically

    def generate_triangle_problem(
        self,
        triangle_type: str = "right",
        difficulty: str = "medium",
    ) -> Tuple[str, int]:
        """
        Generate a triangle geometry problem.

        Args:
            triangle_type: Type of triangle ("right", "equilateral", "isosceles", "scalene")
            difficulty: Problem difficulty ("easy", "medium", "hard")

        Returns:
            (problem_statement, answer) tuple
        """
        if triangle_type == "right":
            return self._generate_right_triangle_problem(difficulty)
        elif triangle_type == "equilateral":
            return self._generate_equilateral_triangle_problem(difficulty)
        elif triangle_type == "isosceles":
            return self._generate_isosceles_triangle_problem(difficulty)
        elif triangle_type == "scalene":
            return self._generate_scalene_triangle_problem(difficulty)
        else:
            raise ValueError(f"Unknown triangle type: {triangle_type}")

    def generate_circle_problem(
        self,
        circle_type: str = "inscribed",
        difficulty: str = "medium",
    ) -> Tuple[str, int]:
        """
        Generate a circle geometry problem.

        Args:
            circle_type: Type of circle problem ("inscribed", "circumscribed", "tangent", "chord")
            difficulty: Problem difficulty ("easy", "medium", "hard")

        Returns:
            (problem_statement, answer) tuple
        """
        if circle_type == "inscribed":
            return self._generate_inscribed_circle_problem(difficulty)
        elif circle_type == "circumscribed":
            return self._generate_circumscribed_circle_problem(difficulty)
        elif circle_type == "tangent":
            return self._generate_tangent_circle_problem(difficulty)
        elif circle_type == "chord":
            return self._generate_chord_problem(difficulty)
        else:
            raise ValueError(f"Unknown circle type: {circle_type}")

    def generate_coordinate_problem(
        self,
        problem_type: str = "distance",
        difficulty: str = "medium",
    ) -> Tuple[str, int]:
        """
        Generate a coordinate geometry problem.

        Args:
            problem_type: Type of coordinate problem ("distance", "midpoint", "slope", "area")
            difficulty: Problem difficulty ("easy", "medium", "hard")

        Returns:
            (problem_statement, answer) tuple
        """
        if problem_type == "distance":
            return self._generate_distance_problem(difficulty)
        elif problem_type == "midpoint":
            return self._generate_midpoint_problem(difficulty)
        elif problem_type == "slope":
            return self._generate_slope_problem(difficulty)
        elif problem_type == "area":
            return self._generate_area_problem(difficulty)
        else:
            raise ValueError(f"Unknown coordinate problem type: {problem_type}")

    def _generate_right_triangle_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate right triangle problem using Pythagorean theorem."""
        # Generate Pythagorean triple
        triples = [
            (3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25),
            (20, 21, 29), (9, 40, 41), (12, 35, 37), (11, 60, 61),
        ]

        if difficulty == "easy":
            a, b, c = random.choice(triples[:3])
        elif difficulty == "medium":
            a, b, c = random.choice(triples[2:6])
        else:  # hard
            a, b, c = random.choice(triples[4:])

        # Randomly choose which sides to give and which to find
        sides = ["AC", "BC", "AB"]
        given_sides = random.sample(sides, 2)
        unknown_side = [s for s in sides if s not in given_sides][0]

        # Map to actual values
        side_values = {"AC": a, "BC": b, "AB": c}
        given_values = {s: side_values[s] for s in given_sides}
        answer = side_values[unknown_side]

        # Generate problem statement
        problem = (
            f"In right triangle ABC with right angle at C, "
            f"if {given_sides[0]} = {given_values[given_sides[0]]} and "
            f"{given_sides[1]} = {given_values[given_sides[1]]}, "
            f"find the length of {unknown_side}."
        )

        return problem, answer

    def _generate_equilateral_triangle_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate equilateral triangle problem."""
        if difficulty == "easy":
            side = random.randint(5, 15)
        elif difficulty == "medium":
            side = random.randint(10, 30)
        else:  # hard
            side = random.randint(20, 50)

        # Problem: find height or area
        problem_type = random.choice(["height", "area"])

        if problem_type == "height":
            height = int(side * sp.sqrt(3) / 2)
            problem = (
                f"In equilateral triangle ABC with side length {side}, "
                f"find the height from vertex A to side BC."
            )
            return problem, height
        else:  # area
            area = int(side**2 * sp.sqrt(3) / 4)
            problem = (
                f"In equilateral triangle ABC with side length {side}, "
                f"find the area."
            )
            return problem, area

    def _generate_isosceles_triangle_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate isosceles triangle problem."""
        if difficulty == "easy":
            base = random.randint(6, 12)
            equal_side = random.randint(5, 15)
        elif difficulty == "medium":
            base = random.randint(10, 20)
            equal_side = random.randint(10, 25)
        else:  # hard
            base = random.randint(15, 30)
            equal_side = random.randint(15, 40)

        # Use Pythagorean theorem for height
        height = int(sp.sqrt(equal_side**2 - (base / 2) ** 2))

        problem = (
            f"In isosceles triangle ABC with AB = AC = {equal_side} and "
            f"base BC = {base}, find the height from vertex A to base BC."
        )

        return problem, height

    def _generate_scalene_triangle_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate scalene triangle problem."""
        # Use Heron's formula or angle sum
        if difficulty == "easy":
            sides = sorted([random.randint(5, 10) for _ in range(3)])
        elif difficulty == "medium":
            sides = sorted([random.randint(8, 15) for _ in range(3)])
        else:  # hard
            sides = sorted([random.randint(10, 20) for _ in range(3)])

        # Ensure triangle inequality
        while sides[0] + sides[1] <= sides[2]:
            sides = sorted([random.randint(5, 20) for _ in range(3)])

        # Use angle sum theorem
        angles = [random.randint(30, 80) for _ in range(2)]
        third_angle = 180 - sum(angles)

        problem = (
            f"In triangle ABC, if ∠A = {angles[0]}° and ∠B = {angles[1]}°, "
            f"find the measure of ∠C."
        )

        return problem, third_angle

    def _generate_inscribed_circle_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate inscribed circle (incircle) problem."""
        if difficulty == "easy":
            # Right triangle with simple dimensions
            a, b, c = random.choice([(3, 4, 5), (5, 12, 13), (6, 8, 10)])
        elif difficulty == "medium":
            a, b, c = random.choice([(8, 15, 17), (7, 24, 25), (9, 12, 15)])
        else:  # hard
            a, b, c = random.choice([(20, 21, 29), (12, 35, 37), (15, 20, 25)])

        # Inradius formula: r = (a + b - c) / 2 for right triangle
        # Or more generally: r = area / semiperimeter
        s = (a + b + c) / 2
        area = sp.sqrt(s * (s - a) * (s - b) * (s - c))
        inradius = int(area / s)

        problem = (
            f"In right triangle ABC with right angle at C, "
            f"if AC = {a}, BC = {b}, and AB = {c}, "
            f"find the radius of the inscribed circle."
        )

        return problem, inradius

    def _generate_circumscribed_circle_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate circumscribed circle (circumcircle) problem."""
        if difficulty == "easy":
            side = random.randint(5, 10)
        elif difficulty == "medium":
            side = random.randint(8, 15)
        else:  # hard
            side = random.randint(12, 25)

        # For equilateral triangle: R = side / sqrt(3)
        circumradius = int(side / sp.sqrt(3))

        problem = (
            f"In equilateral triangle ABC with side length {side}, "
            f"find the radius of the circumscribed circle."
        )

        return problem, circumradius

    def _generate_tangent_circle_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate tangent circle problem."""
        if difficulty == "easy":
            r1 = random.randint(3, 8)
            r2 = random.randint(3, 8)
        elif difficulty == "medium":
            r1 = random.randint(5, 12)
            r2 = random.randint(5, 12)
        else:  # hard
            r1 = random.randint(8, 20)
            r2 = random.randint(8, 20)

        # Distance between centers of two externally tangent circles
        distance = r1 + r2

        problem = (
            f"Two circles with radii {r1} and {r2} are externally tangent. "
            f"Find the distance between their centers."
        )

        return problem, distance

    def _generate_chord_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate chord length problem."""
        if difficulty == "easy":
            radius = random.randint(5, 10)
            chord_distance = random.randint(2, radius - 1)
        elif difficulty == "medium":
            radius = random.randint(8, 15)
            chord_distance = random.randint(3, radius - 2)
        else:  # hard
            radius = random.randint(10, 20)
            chord_distance = random.randint(4, radius - 3)

        # Chord length: 2 * sqrt(r^2 - d^2) where d is distance from center
        chord_length = int(2 * sp.sqrt(radius**2 - chord_distance**2))

        problem = (
            f"In a circle with radius {radius}, a chord is at distance {chord_distance} "
            f"from the center. Find the length of the chord."
        )

        return problem, chord_length

    def _generate_distance_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate distance between two points problem."""
        if difficulty == "easy":
            x1, y1 = random.randint(-5, 5), random.randint(-5, 5)
            x2, y2 = random.randint(-5, 5), random.randint(-5, 5)
        elif difficulty == "medium":
            x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
            x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        else:  # hard
            x1, y1 = random.randint(-20, 20), random.randint(-20, 20)
            x2, y2 = random.randint(-20, 20), random.randint(-20, 20)

        # Distance formula: sqrt((x2-x1)^2 + (y2-y1)^2)
        distance = int(sp.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        problem = (
            f"Find the distance between points A({x1}, {y1}) and B({x2}, {y2})."
        )

        return problem, distance

    def _generate_midpoint_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate midpoint coordinate problem."""
        if difficulty == "easy":
            x1, y1 = random.randint(-5, 5), random.randint(-5, 5)
            x2, y2 = random.randint(-5, 5), random.randint(-5, 5)
        elif difficulty == "medium":
            x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
            x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        else:  # hard
            x1, y1 = random.randint(-20, 20), random.randint(-20, 20)
            x2, y2 = random.randint(-20, 20), random.randint(-20, 20)

        # Midpoint: ((x1+x2)/2, (y1+y2)/2)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Ask for sum of coordinates
        answer = int(mid_x + mid_y)

        problem = (
            f"Find the sum of the x and y coordinates of the midpoint "
            f"of segment AB, where A({x1}, {y1}) and B({x2}, {y2})."
        )

        return problem, answer

    def _generate_slope_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate slope problem."""
        if difficulty == "easy":
            x1, y1 = random.randint(-5, 5), random.randint(-5, 5)
            x2, y2 = random.randint(-5, 5), random.randint(-5, 5)
        elif difficulty == "medium":
            x1, y1 = random.randint(-10, 10), random.randint(-10, 10)
            x2, y2 = random.randint(-10, 10), random.randint(-10, 10)
        else:  # hard
            x1, y1 = random.randint(-20, 20), random.randint(-20, 20)
            x2, y2 = random.randint(-20, 20), random.randint(-20, 20)

        # Avoid division by zero
        while x2 == x1:
            x2 = random.randint(-20, 20)

        # Slope: (y2 - y1) / (x2 - x1)
        slope = (y2 - y1) / (x2 - x1)

        # Return numerator of simplified fraction
        if slope == int(slope):
            answer = int(slope)
        else:
            # Return simplified numerator
            from fractions import Fraction
            frac = Fraction(slope).limit_denominator(100)
            answer = frac.numerator

        problem = (
            f"Find the slope of the line passing through points "
            f"A({x1}, {y1}) and B({x2}, {y2})."
        )

        return problem, answer

    def _generate_area_problem(self, difficulty: str) -> Tuple[str, int]:
        """Generate area problem (triangle or rectangle)."""
        shape = random.choice(["triangle", "rectangle"])

        if shape == "triangle":
            if difficulty == "easy":
                base = random.randint(4, 10)
                height = random.randint(4, 10)
            elif difficulty == "medium":
                base = random.randint(6, 15)
                height = random.randint(6, 15)
            else:  # hard
                base = random.randint(10, 25)
                height = random.randint(10, 25)

            area = int(base * height / 2)

            problem = (
                f"Find the area of triangle ABC with base BC = {base} "
                f"and height from A to BC = {height}."
            )

        else:  # rectangle
            if difficulty == "easy":
                length = random.randint(4, 10)
                width = random.randint(4, 10)
            elif difficulty == "medium":
                length = random.randint(6, 15)
                width = random.randint(6, 15)
            else:  # hard
                length = random.randint(10, 25)
                width = random.randint(10, 25)

            area = length * width

            problem = (
                f"Find the area of rectangle ABCD with length AB = {length} "
                f"and width BC = {width}."
            )

        return problem, area

    def generate_batch(
        self,
        problem_types: List[Dict[str, str]],
        num_problems: int = 10,
    ) -> List[Tuple[str, int, Dict[str, str]]]:
        """
        Generate a batch of problems.

        Args:
            problem_types: List of dicts with keys: "family", "type", "difficulty"
            num_problems: Number of problems to generate

        Returns:
            List of (problem_statement, answer, metadata) tuples
        """
        problems = []

        for _ in range(num_problems):
            # Select random problem type if not specified
            if not problem_types:
                family = random.choice(["triangle", "circle", "coordinate"])
                if family == "triangle":
                    ptype = random.choice(["right", "equilateral", "isosceles", "scalene"])
                elif family == "circle":
                    ptype = random.choice(["inscribed", "circumscribed", "tangent", "chord"])
                else:  # coordinate
                    ptype = random.choice(["distance", "midpoint", "slope", "area"])
                difficulty = random.choice(["easy", "medium", "hard"])
            else:
                spec = random.choice(problem_types)
                family = spec["family"]
                ptype = spec["type"]
                difficulty = spec["difficulty"]

            # Generate problem
            if family == "triangle":
                problem, answer = self.generate_triangle_problem(ptype, difficulty)
            elif family == "circle":
                problem, answer = self.generate_circle_problem(ptype, difficulty)
            else:  # coordinate
                problem, answer = self.generate_coordinate_problem(ptype, difficulty)

            metadata = {
                "family": family,
                "type": ptype,
                "difficulty": difficulty,
            }

            problems.append((problem, answer, metadata))

        return problems


def generate_problems(
    family: str,
    problem_type: str,
    difficulty: str = "medium",
    num_problems: int = 1,
    seed: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Convenience function to generate problems.

    Args:
        family: Problem family ("triangle", "circle", "coordinate")
        problem_type: Specific problem type
        difficulty: Difficulty level
        num_problems: Number of problems to generate
        seed: Random seed

    Returns:
        List of (problem_statement, answer) tuples
    """
    generator = ProblemGenerator(seed=seed)

    problems = []
    for _ in range(num_problems):
        if family == "triangle":
            problem, answer = generator.generate_triangle_problem(problem_type, difficulty)
        elif family == "circle":
            problem, answer = generator.generate_circle_problem(problem_type, difficulty)
        elif family == "coordinate":
            problem, answer = generator.generate_coordinate_problem(problem_type, difficulty)
        else:
            raise ValueError(f"Unknown family: {family}")

        problems.append((problem, answer))

    return problems

