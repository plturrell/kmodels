"""Geometric relations between primitives."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .primitives import Circle, Line, Point


class RelationType(Enum):
    """Types of geometric relations."""

    # Incidence relations
    LIES_ON = "lies_on"  # Point lies on line/circle
    CONTAINS = "contains"  # Line/circle contains point

    # Line relations
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    INTERSECTS = "intersects"

    # Distance/measure relations
    CONGRUENT = "congruent"  # Segments/circles are congruent
    EQUAL_LENGTH = "equal_length"
    EQUAL_ANGLE = "equal_angle"
    RATIO = "ratio"  # Length ratio

    # Circle relations
    TANGENT = "tangent"  # Line tangent to circle, or circles tangent
    SECANT = "secant"  # Line intersects circle

    # Angle relations
    RIGHT_ANGLE = "right_angle"
    ACUTE = "acute"
    OBTUSE = "obtuse"

    # Position relations
    BETWEEN = "between"  # Point between two other points
    MIDPOINT = "midpoint"  # Point is midpoint of segment
    CENTER = "center"  # Point is center of circle


@dataclass
class Relation:
    """
    A relation between geometric primitives.
    
    Relations are edges in the scene graph, labeled with RelationType
    and optionally containing parameters (e.g., angle measure, ratio value).
    """

    relation_type: RelationType
    source: Any  # Point, Line, or Circle
    target: Any  # Point, Line, or Circle
    parameters: Dict[str, Any] = field(default_factory=dict)  # Additional parameters (angle, ratio, etc.)

    def __str__(self) -> str:
        params_str = f"({self.parameters})" if self.parameters else ""
        return f"{self.relation_type.value}({self.source}, {self.target}){params_str}"

    def __hash__(self) -> int:
        return hash((self.relation_type, id(self.source), id(self.target), tuple(sorted(self.parameters.items()))))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Relation):
            return False
        return (
            self.relation_type == other.relation_type
            and self.source == other.source
            and self.target == other.target
            and self.parameters == other.parameters
        )

    def matches_pattern(self, pattern: "Relation") -> bool:
        """
        Check if this relation matches a pattern relation.
        
        Pattern matching allows for variable binding in theorem application.
        """
        if self.relation_type != pattern.relation_type:
            return False
        # More sophisticated matching would be needed for variables
        return True

