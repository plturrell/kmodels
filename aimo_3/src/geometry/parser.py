"""LaTeX parser: Interpretation function I: S → G."""

import re
from typing import Dict, List, Optional, Tuple

from .primitives import Circle, Line, Point
from .relations import Relation, RelationType
from .scene_graph import GeometricSceneGraph


class GeometryParser:
    """
    Parses LaTeX problem statements into Geometric Scene Graphs.
    
    Implements the interpretation function I: S → G.
    """

    def __init__(self):
        """Initialize parser."""
        self.point_pattern = re.compile(r'\b([A-Z])\b')  # Single uppercase letters
        self.triangle_pattern = re.compile(r'triangle\s*([A-Z])([A-Z])([A-Z])', re.IGNORECASE)
        self.circle_pattern = re.compile(r'circle\s*\(([A-Z])\)', re.IGNORECASE)

    def parse(self, problem_statement: str) -> GeometricSceneGraph:
        """
        Parse problem statement into scene graph.

        Args:
            problem_statement: LaTeX problem statement

        Returns:
            GeometricSceneGraph G_initial
        """
        graph = GeometricSceneGraph()

        # Extract points
        points = self._extract_points(problem_statement)
        for point in points:
            graph.add_vertex(point)

        # Extract lines
        lines = self._extract_lines(problem_statement, points)
        for line in lines:
            graph.add_vertex(line)

        # Extract circles
        circles = self._extract_circles(problem_statement, points)
        for circle in circles:
            graph.add_vertex(circle)

        # Extract relations
        relations = self._extract_relations(problem_statement, points, lines, circles)
        for relation in relations:
            graph.add_edge(relation)

        return graph

    def _extract_points(self, text: str) -> List[Point]:
        """Extract point labels from text."""
        # Find all uppercase letters that likely represent points
        point_labels = set()

        # Triangle notation: triangle ABC
        triangle_matches = self.triangle_pattern.findall(text)
        for match in triangle_matches:
            point_labels.update(match)

        # Single letter points: point A, vertex B, etc.
        point_mentions = re.findall(r'(?:point|vertex|corner)\s+([A-Z])', text, re.IGNORECASE)
        point_labels.update(point_mentions)

        # General uppercase letters in math mode
        math_points = re.findall(r'\$([A-Z])\$', text)
        point_labels.update(math_points)

        # Also look for standalone uppercase letters in geometric context
        standalone = re.findall(r'\b([A-Z])\b', text)
        # Filter to common geometric point labels
        common_labels = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        point_labels.update(p for p in standalone if p in common_labels)

        return [Point(label=label) for label in sorted(point_labels)]

    def _extract_lines(self, text: str, points: List[Point]) -> List[Line]:
        """Extract lines from text."""
        lines = []
        point_dict = {p.label: p for p in points}

        # Line through two points: line AB, segment AB
        line_patterns = [
            r'(?:line|segment|side)\s+([A-Z])([A-Z])',
            r'([A-Z])([A-Z])\s*(?:line|segment)',
        ]

        for pattern in line_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                label1, label2 = match[0], match[1]
                if label1 in point_dict and label2 in point_dict:
                    line = Line(
                        point1=point_dict[label1],
                        point2=point_dict[label2],
                    )
                    if line not in lines:
                        lines.append(line)

        # Named lines: line l, line m
        named_lines = re.findall(r'line\s+([a-z])', text, re.IGNORECASE)
        for name in named_lines:
            line = Line(label=name)
            lines.append(line)

        return lines

    def _extract_circles(self, text: str, points: List[Point]) -> List[Circle]:
        """Extract circles from text."""
        circles = []
        point_dict = {p.label: p for p in points}

        # Circle with center: circle(O), circle with center A
        circle_matches = self.circle_pattern.findall(text)
        for center_label in circle_matches:
            if center_label in point_dict:
                circle = Circle(center=point_dict[center_label])
                circles.append(circle)

        # Circle through points: circle through A, B, C
        circle_through = re.findall(
            r'circle\s+through\s+([A-Z]),\s*([A-Z]),\s*([A-Z])',
            text,
            re.IGNORECASE,
        )
        for match in circle_through:
            labels = [match[0], match[1], match[2]]
            if all(l in point_dict for l in labels):
                circle = Circle(
                    point1=point_dict[labels[0]],
                    point2=point_dict[labels[1]],
                    point3=point_dict[labels[2]],
                )
                circles.append(circle)

        return circles

    def _extract_relations(
        self,
        text: str,
        points: List[Point],
        lines: List[Line],
        circles: List[Circle],
    ) -> List[Relation]:
        """Extract geometric relations from text."""
        relations = []
        text_lower = text.lower()

        point_dict = {p.label: p for p in points}
        line_dict = {}
        for line in lines:
            if line.label:
                line_dict[line.label] = line
            elif line.point1 and line.point2:
                key = f"{line.point1.label}{line.point2.label}"
                line_dict[key] = line

        # Parallel lines
        parallel_pattern = r'([A-Z])([A-Z])\s*(?:\|\||parallel)\s*([A-Z])([A-Z])'
        for match in re.finditer(parallel_pattern, text, re.IGNORECASE):
            line1_key = match.group(1) + match.group(2)
            line2_key = match.group(3) + match.group(4)
            if line1_key in line_dict and line2_key in line_dict:
                relations.append(
                    Relation(
                        RelationType.PARALLEL,
                        line_dict[line1_key],
                        line_dict[line2_key],
                    )
                )

        # Perpendicular lines
        perp_pattern = r'([A-Z])([A-Z])\s*[⊥⟂]\s*([A-Z])([A-Z])|perpendicular'
        for match in re.finditer(perp_pattern, text, re.IGNORECASE):
            # Extract line pairs from context
            pass  # Simplified - would need more sophisticated parsing

        # Point on line
        for line in lines:
            if line.point1 and line.point2:
                relations.append(
                    Relation(RelationType.LIES_ON, line.point1, line)
                )
                relations.append(
                    Relation(RelationType.LIES_ON, line.point2, line)
                )

        # Right angle
        right_angle_pattern = r'right\s+angle|90\s*°|∠\s*([A-Z])([A-Z])([A-Z])\s*=\s*90'
        if re.search(right_angle_pattern, text, re.IGNORECASE):
            # Extract angle vertices
            angle_matches = re.findall(r'∠\s*([A-Z])([A-Z])([A-Z])', text)
            for match in angle_matches:
                vertex_label = match[1]
                if vertex_label in point_dict:
                    # Create relation (simplified)
                    pass

        # Equal segments
        equal_pattern = r'([A-Z])([A-Z])\s*=\s*([A-Z])([A-Z])'
        for match in re.finditer(equal_pattern, text):
            seg1_key = match.group(1) + match.group(2)
            seg2_key = match.group(3) + match.group(4)
            if seg1_key in line_dict and seg2_key in line_dict:
                relations.append(
                    Relation(
                        RelationType.CONGRUENT,
                        line_dict[seg1_key],
                        line_dict[seg2_key],
                    )
                )

        return relations


def parse_problem(problem_statement: str) -> GeometricSceneGraph:
    """
    Convenience function to parse a problem statement.

    Args:
        problem_statement: LaTeX problem statement

    Returns:
        GeometricSceneGraph
    """
    parser = GeometryParser()
    return parser.parse(problem_statement)

