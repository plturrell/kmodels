"""Geometric primitives: points, lines, circles."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Point:
    """
    A point in the geometric scene.
    
    Points are identified by labels (e.g., "A", "B", "P") and may have
    coordinates if the coordinate system is established.
    """
    label: str
    x: Optional[float] = None
    y: Optional[float] = None

    def __str__(self) -> str:
        return self.label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.label == other.label

    def has_coordinates(self) -> bool:
        """Check if point has assigned coordinates."""
        return self.x is not None and self.y is not None

    def coordinates(self) -> Optional[Tuple[float, float]]:
        """Get coordinates if available."""
        if self.has_coordinates():
            return (self.x, self.y)
        return None


@dataclass(frozen=True)
class Line:
    """
    A line in the geometric scene.
    
    Lines are defined by two points or by a point and direction.
    """
    label: Optional[str] = None
    point1: Optional[Point] = None
    point2: Optional[Point] = None
    # Alternative: line through point with given direction/slope
    through_point: Optional[Point] = None
    direction: Optional[Tuple[float, float]] = None

    def __str__(self) -> str:
        if self.label:
            return self.label
        if self.point1 and self.point2:
            return f"line({self.point1.label}, {self.point2.label})"
        return "line"

    def __hash__(self) -> int:
        if self.label:
            return hash(self.label)
        if self.point1 and self.point2:
            return hash((self.point1.label, self.point2.label))
        return id(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Line):
            return False
        if self.label and other.label:
            return self.label == other.label
        if self.point1 and self.point2 and other.point1 and other.point2:
            return (self.point1 == other.point1 and self.point2 == other.point2) or \
                   (self.point1 == other.point2 and self.point2 == other.point1)
        return False

    def contains_point(self, point: Point) -> bool:
        """Check if line contains a point."""
        if self.point1 == point or self.point2 == point:
            return True
        if self.through_point == point:
            return True
        return False


@dataclass(frozen=True)
class Circle:
    """
    A circle in the geometric scene.
    
    Circles are defined by center and radius, or by three points.
    """
    label: Optional[str] = None
    center: Optional[Point] = None
    radius: Optional[float] = None
    # Alternative: circle through three points
    point1: Optional[Point] = None
    point2: Optional[Point] = None
    point3: Optional[Point] = None

    def __str__(self) -> str:
        if self.label:
            return self.label
        if self.center:
            return f"circle(center={self.center.label}, r={self.radius})"
        return "circle"

    def __hash__(self) -> int:
        if self.label:
            return hash(self.label)
        if self.center:
            return hash((self.center.label, self.radius))
        return id(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Circle):
            return False
        if self.label and other.label:
            return self.label == other.label
        if self.center and other.center:
            return self.center == other.center and self.radius == other.radius
        return False

    def contains_point(self, point: Point) -> bool:
        """Check if circle contains a point (on circumference)."""
        # This would require coordinate computation
        # For now, return False - would be computed during evaluation
        return False

