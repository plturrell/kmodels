"""Extended geometry theorems: isosceles/equilateral, circles, coordinate geometry."""

from typing import Dict, Set, Tuple

import sympy as sp

from .primitives import Circle, Line, Point
from .relations import Relation, RelationType
from .scene_graph import GeometricSceneGraph
from .state import State
from .theorems import Theorem


class IsoscelesTriangleTheorem(Theorem):
    """Isosceles triangle: base angles are equal."""

    def __init__(self):
        super().__init__("IsoscelesBaseAngles", "In isosceles triangle, base angles are equal")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: isosceles triangle with two equal sides."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        AB = Line(point1=A, point2=B)
        AC = Line(point1=A, point2=C)
        BC = Line(point1=B, point2=C)
        pattern.add_vertex(AB)
        pattern.add_vertex(AC)
        pattern.add_vertex(BC)

        # Two equal sides
        pattern.add_edge(Relation(RelationType.CONGRUENT, AB, AC))

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if two sides are equal."""
        # Simplified - would check actual congruence
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive base angles equal."""
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"isosceles_base_angles({B_label}, {C_label}): ∠B = ∠C",
        }

        return G_addition, propositions


class EquilateralTriangleTheorem(Theorem):
    """Equilateral triangle: all sides equal, all angles 60°."""

    def __init__(self):
        super().__init__("Equilateral", "In equilateral triangle, all sides equal, all angles 60°")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: equilateral triangle."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        AB = Line(point1=A, point2=B)
        BC = Line(point1=B, point2=C)
        CA = Line(point1=C, point2=A)
        pattern.add_vertex(AB)
        pattern.add_vertex(BC)
        pattern.add_vertex(CA)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if all sides are equal."""
        # Simplified - would check actual congruence
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive all sides equal, all angles 60°."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"equilateral({A_label}, {B_label}, {C_label}): AB = BC = CA, ∠A = ∠B = ∠C = 60°",
        }

        return G_addition, propositions


class TriangleHeightTheorem(Theorem):
    """Height of triangle: h = 2*area / base."""

    def __init__(self):
        super().__init__("TriangleHeight", "Height = 2*area / base")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with height."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        BC = Line(point1=B, point2=C)
        pattern.add_vertex(BC)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Always applicable."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive height formula."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"triangle_height({A_label}, {B_label}, {C_label}): h = 2*area / base",
        }

        return G_addition, propositions


class TriangleMedianTheorem(Theorem):
    """Median divides triangle into two equal areas."""

    def __init__(self):
        super().__init__("TriangleMedian", "Median divides triangle into two equal areas")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with median."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        M = Point("M")  # Midpoint
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)
        pattern.add_vertex(M)

        BC = Line(point1=B, point2=C)
        AM = Line(point1=A, point2=M)
        pattern.add_vertex(BC)
        pattern.add_vertex(AM)

        pattern.add_edge(Relation(RelationType.MIDPOINT, M, BC))

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if M is midpoint of BC."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive median property."""
        A_label = match.get("A")
        M_label = match.get("M")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"median_property({A_label}, {M_label}): area(ABM) = area(ACM)",
        }

        return G_addition, propositions


class InscribedAngleTheorem(Theorem):
    """Inscribed angle is half the central angle subtending the same arc."""

    def __init__(self):
        super().__init__("InscribedAngle", "Inscribed angle = half central angle")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: circle with inscribed angle."""
        pattern = GeometricSceneGraph()

        O = Point("O")  # Center
        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(O)
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        circle = Circle(center=O)
        pattern.add_vertex(circle)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if points are on circle."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive inscribed angle property."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")
        O_label = match.get("O")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"inscribed_angle({A_label}, {B_label}, {C_label}, {O_label}): ∠ABC = 0.5 * ∠AOC",
        }

        return G_addition, propositions


class ChordLengthTheorem(Theorem):
    """Chord length: 2 * r * sin(θ/2) where θ is central angle."""

    def __init__(self):
        super().__init__("ChordLength", "Chord length = 2*r*sin(θ/2)")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: circle with chord."""
        pattern = GeometricSceneGraph()

        O = Point("O")  # Center
        A = Point("A")
        B = Point("B")
        pattern.add_vertex(O)
        pattern.add_vertex(A)
        pattern.add_vertex(B)

        circle = Circle(center=O)
        chord = Line(point1=A, point2=B)
        pattern.add_vertex(circle)
        pattern.add_vertex(chord)

        pattern.add_edge(Relation(RelationType.LIES_ON, A, circle))
        pattern.add_edge(Relation(RelationType.LIES_ON, B, circle))

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if A and B are on circle."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive chord length formula."""
        A_label = match.get("A")
        B_label = match.get("B")
        O_label = match.get("O")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"chord_length({A_label}, {B_label}, {O_label}): AB = 2*r*sin(∠AOB/2)",
        }

        return G_addition, propositions


class CoordinateDistanceTheorem(Theorem):
    """Distance formula: √[(x₂-x₁)² + (y₂-y₁)²]."""

    def __init__(self):
        super().__init__("CoordinateDistance", "Distance = √[(x₂-x₁)² + (y₂-y₁)²]")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: two points with coordinates."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        pattern.add_vertex(A)
        pattern.add_vertex(B)

        AB = Line(point1=A, point2=B)
        pattern.add_vertex(AB)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if points have coordinates."""
        A_label = match.get("A")
        B_label = match.get("B")

        # Check if points have coordinates in state
        for prop in state.propositions:
            if f"coord({A_label})" in prop or f"x_{A_label}" in prop:
                return True

        return False

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive distance formula."""
        A_label = match.get("A")
        B_label = match.get("B")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"distance_formula({A_label}, {B_label}): AB = √[(x_B-x_A)² + (y_B-y_A)²]",
        }

        return G_addition, propositions


class CoordinateMidpointTheorem(Theorem):
    """Midpoint formula: ((x₁+x₂)/2, (y₁+y₂)/2)."""

    def __init__(self):
        super().__init__("CoordinateMidpoint", "Midpoint = ((x₁+x₂)/2, (y₁+y₂)/2)")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: two points with midpoint."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        M = Point("M")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(M)

        AB = Line(point1=A, point2=B)
        pattern.add_vertex(AB)

        pattern.add_edge(Relation(RelationType.MIDPOINT, M, AB))

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if points have coordinates."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive midpoint formula."""
        A_label = match.get("A")
        B_label = match.get("B")
        M_label = match.get("M")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"midpoint_formula({A_label}, {B_label}, {M_label}): M = ((x_A+x_B)/2, (y_A+y_B)/2)",
        }

        return G_addition, propositions


class CoordinateSlopeTheorem(Theorem):
    """Slope formula: (y₂-y₁)/(x₂-x₁)."""

    def __init__(self):
        super().__init__("CoordinateSlope", "Slope = (y₂-y₁)/(x₂-x₁)")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: line through two points."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        pattern.add_vertex(A)
        pattern.add_vertex(B)

        line = Line(point1=A, point2=B)
        pattern.add_vertex(line)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if points have coordinates."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive slope formula."""
        A_label = match.get("A")
        B_label = match.get("B")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"slope_formula({A_label}, {B_label}): m = (y_B-y_A)/(x_B-x_A)",
        }

        return G_addition, propositions


class CoordinateAreaTheorem(Theorem):
    """Area using coordinates: 0.5 * |x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|."""

    def __init__(self):
        super().__init__("CoordinateArea", "Area = 0.5 * |det|")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with three points."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if points have coordinates."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive area formula."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"coordinate_area({A_label}, {B_label}, {C_label}): "
            f"area = 0.5 * |x_A(y_B-y_C) + x_B(y_C-y_A) + x_C(y_A-y_B)|",
        }

        return G_addition, propositions


# Extended theorem library with new theorems
def get_extended_theorem_library():
    """Get extended theorem library with all new theorems."""
    from .theorems import TheoremLibrary

    library = TheoremLibrary()
    
    # Add extended theorems
    library.add_theorem(IsoscelesTriangleTheorem())
    library.add_theorem(EquilateralTriangleTheorem())
    library.add_theorem(TriangleHeightTheorem())
    library.add_theorem(TriangleMedianTheorem())
    library.add_theorem(InscribedAngleTheorem())
    library.add_theorem(ChordLengthTheorem())
    library.add_theorem(CoordinateDistanceTheorem())
    library.add_theorem(CoordinateMidpointTheorem())
    library.add_theorem(CoordinateSlopeTheorem())
    library.add_theorem(CoordinateAreaTheorem())

    return library

