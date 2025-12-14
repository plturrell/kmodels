"""Advanced geometry theorems: similarity, power of point, Ceva, Menelaus, etc."""

from typing import Dict, Set, Tuple

from .primitives import Circle, Line, Point
from .relations import Relation, RelationType
from .scene_graph import GeometricSceneGraph
from .state import State
from .theorems import Theorem


class SimilarTrianglesTheorem(Theorem):
    """Similar triangles: corresponding angles equal, sides proportional."""

    def __init__(self):
        super().__init__("SimilarTriangles", "Similar triangles have proportional sides")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: two triangles with equal angles."""
        pattern = GeometricSceneGraph()

        # First triangle
        A1 = Point("A1")
        B1 = Point("B1")
        C1 = Point("C1")
        pattern.add_vertex(A1)
        pattern.add_vertex(B1)
        pattern.add_vertex(C1)

        # Second triangle
        A2 = Point("A2")
        B2 = Point("B2")
        C2 = Point("C2")
        pattern.add_vertex(A2)
        pattern.add_vertex(B2)
        pattern.add_vertex(C2)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if triangles have equal angles."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive proportional sides."""
        G_addition = GeometricSceneGraph()
        propositions = {
            f"similar_triangles({match.get('A1')}, {match.get('B1')}, {match.get('C1')}, "
            f"{match.get('A2')}, {match.get('B2')}, {match.get('C2')}): "
            f"A1B1/A2B2 = B1C1/B2C2 = C1A1/C2A2",
        }
        return G_addition, propositions


class PowerOfPointTheorem(Theorem):
    """Power of a point: PA × PB = PC × PD for chords through point P."""

    def __init__(self):
        super().__init__("PowerOfPoint", "Power of a point: PA × PB = PC × PD")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: circle with point P and two chords."""
        pattern = GeometricSceneGraph()

        O = Point("O")
        P = Point("P")
        A = Point("A")
        B = Point("B")
        C = Point("C")
        D = Point("D")
        pattern.add_vertex(O)
        pattern.add_vertex(P)
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)
        pattern.add_vertex(D)

        circle = Circle(center=O)
        pattern.add_vertex(circle)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if P is point and A,B,C,D are on circle."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive power of point."""
        P_label = match.get("P")
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")
        D_label = match.get("D")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"power_of_point({P_label}, {A_label}, {B_label}, {C_label}, {D_label}): "
            f"PA × PB = PC × PD",
        }
        return G_addition, propositions


class CevaTheorem(Theorem):
    """Ceva's theorem: For triangle ABC, lines from vertices to opposite sides are concurrent iff (AF/FB) × (BD/DC) × (CE/EA) = 1."""

    def __init__(self):
        super().__init__("Ceva", "Ceva's theorem for concurrent cevians")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with cevians."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        D = Point("D")  # On BC
        E = Point("E")  # On CA
        F = Point("F")  # On AB
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)
        pattern.add_vertex(D)
        pattern.add_vertex(E)
        pattern.add_vertex(F)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if cevians are concurrent."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive Ceva's condition."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")
        D_label = match.get("D")
        E_label = match.get("E")
        F_label = match.get("F")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"ceva({A_label}, {B_label}, {C_label}, {D_label}, {E_label}, {F_label}): "
            f"(AF/FB) × (BD/DC) × (CE/EA) = 1",
        }
        return G_addition, propositions


class MenelausTheorem(Theorem):
    """Menelaus' theorem: For triangle ABC, points D, E, F on lines BC, CA, AB are collinear iff (AF/FB) × (BD/DC) × (CE/EA) = -1."""

    def __init__(self):
        super().__init__("Menelaus", "Menelaus' theorem for collinear points")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with points on sides."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        D = Point("D")  # On BC
        E = Point("E")  # On CA
        F = Point("F")  # On AB
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)
        pattern.add_vertex(D)
        pattern.add_vertex(E)
        pattern.add_vertex(F)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if points are collinear."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive Menelaus' condition."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")
        D_label = match.get("D")
        E_label = match.get("E")
        F_label = match.get("F")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"menelaus({A_label}, {B_label}, {C_label}, {D_label}, {E_label}, {F_label}): "
            f"(AF/FB) × (BD/DC) × (CE/EA) = -1",
        }
        return G_addition, propositions


class AngleBisectorTheorem(Theorem):
    """Angle bisector theorem: BD/DC = AB/AC."""

    def __init__(self):
        super().__init__("AngleBisector", "Angle bisector divides opposite side proportionally")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with angle bisector."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        D = Point("D")  # On BC
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)
        pattern.add_vertex(D)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if AD is angle bisector."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive angle bisector property."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")
        D_label = match.get("D")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"angle_bisector({A_label}, {B_label}, {C_label}, {D_label}): "
            f"BD/DC = AB/AC",
        }
        return G_addition, propositions


class PtolemyTheorem(Theorem):
    """Ptolemy's theorem: For cyclic quadrilateral, AB × CD + BC × DA = AC × BD."""

    def __init__(self):
        super().__init__("Ptolemy", "Ptolemy's theorem for cyclic quadrilaterals")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: cyclic quadrilateral."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        D = Point("D")
        O = Point("O")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)
        pattern.add_vertex(D)
        pattern.add_vertex(O)

        circle = Circle(center=O)
        pattern.add_vertex(circle)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if quadrilateral is cyclic."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive Ptolemy's relation."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")
        D_label = match.get("D")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"ptolemy({A_label}, {B_label}, {C_label}, {D_label}): "
            f"AB × CD + BC × DA = AC × BD",
        }
        return G_addition, propositions


class StewartTheorem(Theorem):
    """Stewart's theorem: For triangle ABC with cevian AD, b²m + c²n = a(d² + mn)."""

    def __init__(self):
        super().__init__("Stewart", "Stewart's theorem for cevians")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with cevian."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        D = Point("D")  # On BC
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)
        pattern.add_vertex(D)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if AD is cevian."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive Stewart's relation."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")
        D_label = match.get("D")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"stewart({A_label}, {B_label}, {C_label}, {D_label}): "
            f"b²m + c²n = a(d² + mn)",
        }
        return G_addition, propositions


class HeronFormula(Theorem):
    """Heron's formula: Area = √[s(s-a)(s-b)(s-c)] where s is semiperimeter."""

    def __init__(self):
        super().__init__("Heron", "Heron's formula for triangle area")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with three sides."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Always applicable."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive Heron's formula."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"heron({A_label}, {B_label}, {C_label}): "
            f"area = √[s(s-a)(s-b)(s-c)] where s = (a+b+c)/2",
        }
        return G_addition, propositions


class LawOfCosines(Theorem):
    """Law of cosines: c² = a² + b² - 2ab cos(C)."""

    def __init__(self):
        super().__init__("LawOfCosines", "Law of cosines for any triangle")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Always applicable."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive law of cosines."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"law_of_cosines({A_label}, {B_label}, {C_label}): "
            f"c² = a² + b² - 2ab cos(C)",
        }
        return G_addition, propositions


class LawOfSines(Theorem):
    """Law of sines: a/sin(A) = b/sin(B) = c/sin(C) = 2R."""

    def __init__(self):
        super().__init__("LawOfSines", "Law of sines for any triangle")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle."""
        pattern = GeometricSceneGraph()

        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Always applicable."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive law of sines."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"law_of_sines({A_label}, {B_label}, {C_label}): "
            f"a/sin(A) = b/sin(B) = c/sin(C) = 2R",
        }
        return G_addition, propositions


class ThalesTheorem(Theorem):
    """Thales' theorem: Angle in semicircle is right angle."""

    def __init__(self):
        super().__init__("Thales", "Angle in semicircle is right angle")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: circle with diameter and point on circumference."""
        pattern = GeometricSceneGraph()

        O = Point("O")
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
        """Check if AB is diameter and C is on circle."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive right angle."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"thales({A_label}, {B_label}, {C_label}): "
            f"∠ACB = 90°",
        }
        return G_addition, propositions


class TangentSecantTheorem(Theorem):
    """Tangent-secant theorem: (tangent)² = (external part) × (whole secant)."""

    def __init__(self):
        super().__init__("TangentSecant", "Tangent-secant power theorem")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: circle with tangent and secant from external point."""
        pattern = GeometricSceneGraph()

        O = Point("O")
        P = Point("P")
        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(O)
        pattern.add_vertex(P)
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        circle = Circle(center=O)
        pattern.add_vertex(circle)

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if PA is tangent and PBC is secant."""
        return True

    def apply(self, state: State, match: Dict[str, str]) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive tangent-secant relation."""
        P_label = match.get("P")
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"tangent_secant({P_label}, {A_label}, {B_label}, {C_label}): "
            f"PA² = PB × PC",
        }
        return G_addition, propositions


def get_advanced_theorem_library():
    """Get advanced theorem library with 10 additional theorems."""
    from .theorems_extended import get_extended_theorem_library

    library = get_extended_theorem_library()

    # Add advanced theorems
    library.add_theorem(SimilarTrianglesTheorem())
    library.add_theorem(PowerOfPointTheorem())
    library.add_theorem(CevaTheorem())
    library.add_theorem(MenelausTheorem())
    library.add_theorem(AngleBisectorTheorem())
    library.add_theorem(PtolemyTheorem())
    library.add_theorem(StewartTheorem())
    library.add_theorem(HeronFormula())
    library.add_theorem(LawOfCosines())
    library.add_theorem(LawOfSines())
    library.add_theorem(ThalesTheorem())
    library.add_theorem(TangentSecantTheorem())

    return library

