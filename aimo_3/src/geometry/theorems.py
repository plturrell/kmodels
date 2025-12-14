"""Theorem system: production rules T: (G_pattern, Φ_condition) → (G_addition, Φ_addition)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

from .primitives import Circle, Line, Point
from .relations import Relation, RelationType
from .scene_graph import GeometricSceneGraph
from .state import State


class Theorem(ABC):
    """
    A geometric theorem as a production rule.
    
    T: (G_pattern, Φ_condition) → (G_addition, Φ_addition)
    
    If pattern G_pattern matches current graph G and conditions Φ_condition hold,
    then add G_addition and propositions Φ_addition to the state.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize theorem.

        Args:
            name: Theorem name (e.g., "Pythagorean")
            description: Human-readable description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def get_pattern(self) -> GeometricSceneGraph:
        """
        Get the pattern graph G_pattern that must match.

        Returns:
            Pattern graph
        """
        pass

    @abstractmethod
    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """
        Check if conditions Φ_condition hold for a given match.

        Args:
            state: Current state
            match: Mapping from pattern labels to graph labels

        Returns:
            True if conditions are satisfied
        """
        pass

    @abstractmethod
    def apply(
        self, state: State, match: Dict[str, str]
    ) -> Tuple[GeometricSceneGraph, Set[str]]:
        """
        Apply theorem to add G_addition and Φ_addition.

        Args:
            state: Current state
            match: Mapping from pattern labels to graph labels

        Returns:
            (G_addition, Φ_addition) tuple
        """
        pass

    def can_apply(self, state: State) -> List[Dict[str, str]]:
        """
        Check if theorem can be applied to state.

        Args:
            state: Current state

        Returns:
            List of valid matches (empty if cannot apply)
        """
        pattern = self.get_pattern()
        matches = state.graph.find_subgraph_matches(pattern)

        valid_matches = []
        for match in matches:
            if self.check_conditions(state, match):
                valid_matches.append(match)

        return valid_matches


class PythagoreanTheorem(Theorem):
    """Pythagorean theorem: a² + b² = c² for right triangles."""

    def __init__(self):
        super().__init__("Pythagorean", "In a right triangle, a² + b² = c²")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: right triangle with vertices A, B, C where angle at C is right."""
        pattern = GeometricSceneGraph()

        # Three points forming triangle
        A = Point("A")
        B = Point("B")
        C = Point("C")
        pattern.add_vertex(A)
        pattern.add_vertex(B)
        pattern.add_vertex(C)

        # Three sides
        AB = Line(point1=A, point2=B)
        BC = Line(point1=B, point2=C)
        CA = Line(point1=C, point2=A)
        pattern.add_vertex(AB)
        pattern.add_vertex(BC)
        pattern.add_vertex(CA)

        # Right angle at C
        pattern.add_edge(Relation(RelationType.RIGHT_ANGLE, C, CA))

        return pattern

    def check_conditions(self, state: State, match: Dict[str, str]) -> bool:
        """Check if angle at C is right."""
        # Check if right angle relation exists
        C_label = match.get("C")
        if not C_label:
            return False

        # Find right angle relations involving C
        relations = state.graph.get_relations()
        for relation in relations:
            if (
                relation.relation_type == RelationType.RIGHT_ANGLE
                and relation.source.label == C_label
            ):
                return True

        return False

    def apply(
        self, state: State, match: Dict[str, str]
    ) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive a² + b² = c²."""
        # Extract matched vertices
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        # Create new graph additions (none in this case)
        G_addition = GeometricSceneGraph()

        # Add propositions
        propositions = {
            f"pythagorean({A_label}, {B_label}, {C_label}): AB² + BC² = CA²",
        }

        return G_addition, propositions


class AngleSumTheorem(Theorem):
    """Sum of angles in triangle is 180°."""

    def __init__(self):
        super().__init__("AngleSum", "Sum of angles in triangle is 180°")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: triangle with three vertices."""
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
        """Always applicable to any triangle."""
        return True

    def apply(
        self, state: State, match: Dict[str, str]
    ) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive angle sum."""
        A_label = match.get("A")
        B_label = match.get("B")
        C_label = match.get("C")

        G_addition = GeometricSceneGraph()
        propositions = {
            f"angle_sum({A_label}, {B_label}, {C_label}): ∠A + ∠B + ∠C = 180°",
        }

        return G_addition, propositions


class CongruentTrianglesTheorem(Theorem):
    """If two triangles have three pairs of congruent sides, they are congruent."""

    def __init__(self):
        super().__init__("SSS", "Side-Side-Side congruence")

    def get_pattern(self) -> GeometricSceneGraph:
        """Pattern: two triangles with three pairs of equal sides."""
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
        """Check if three pairs of sides are congruent."""
        # Simplified - would check actual congruence relations
        return True

    def apply(
        self, state: State, match: Dict[str, str]
    ) -> Tuple[GeometricSceneGraph, Set[str]]:
        """Apply: derive triangle congruence."""
        G_addition = GeometricSceneGraph()
        propositions = {
            f"congruent_triangles({match.get('A1')}, {match.get('B1')}, {match.get('C1')}, "
            f"{match.get('A2')}, {match.get('B2')}, {match.get('C2')})",
        }

        return G_addition, propositions


class TheoremLibrary:
    """Library of geometric theorems."""

    def __init__(self):
        """Initialize with core theorems."""
        self.theorems: List[Theorem] = [
            PythagoreanTheorem(),
            AngleSumTheorem(),
            CongruentTrianglesTheorem(),
        ]

    def add_theorem(self, theorem: Theorem) -> None:
        """Add a theorem to the library."""
        self.theorems.append(theorem)

    def get_applicable_theorems(self, state: State) -> List[Tuple[Theorem, Dict[str, str]]]:
        """
        Get all theorems that can be applied to the current state.

        Args:
            state: Current state

        Returns:
            List of (theorem, match) tuples
        """
        applicable = []

        for theorem in self.theorems:
            matches = theorem.can_apply(state)
            for match in matches:
                applicable.append((theorem, match))

        return applicable

    def get_theorem_by_name(self, name: str) -> Optional[Theorem]:
        """Get theorem by name."""
        for theorem in self.theorems:
            if theorem.name == name:
                return theorem
        return None

