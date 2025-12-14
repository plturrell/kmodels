"""Geometric Scene Graph: directed, labeled multigraph G = (V, E)."""

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .primitives import Circle, Line, Point
from .relations import Relation, RelationType


class GeometricSceneGraph:
    """
    Geometric Scene Graph G = (V, E).
    
    Vertices V are geometric primitives (points, lines, circles).
    Edges E are relations between primitives.
    """

    def __init__(self):
        """Initialize empty scene graph."""
        self.graph = nx.MultiDiGraph()
        self.vertices: Dict[str, Any] = {}  # Label -> primitive mapping
        self.vertex_types: Dict[str, str] = {}  # Label -> type mapping

    def add_vertex(self, primitive: Any) -> None:
        """
        Add a primitive to the graph.

        Args:
            primitive: Point, Line, or Circle instance
        """
        label = self._get_label(primitive)
        self.graph.add_node(label, primitive=primitive, type=type(primitive).__name__)
        self.vertices[label] = primitive
        self.vertex_types[label] = type(primitive).__name__

    def add_edge(self, relation: Relation) -> None:
        """
        Add a relation (edge) to the graph.

        Args:
            relation: Relation instance
        """
        source_label = self._get_label(relation.source)
        target_label = self._get_label(relation.target)

        # Ensure vertices exist
        if source_label not in self.vertices:
            self.add_vertex(relation.source)
        if target_label not in self.vertices:
            self.add_vertex(relation.target)

        # Add edge with relation as data
        self.graph.add_edge(
            source_label,
            target_label,
            relation_type=relation.relation_type,
            parameters=relation.parameters,
            relation=relation,
        )

    def get_vertices(self, vertex_type: Optional[str] = None) -> List[Any]:
        """
        Get all vertices, optionally filtered by type.

        Args:
            vertex_type: Optional type filter ("Point", "Line", "Circle")

        Returns:
            List of primitives
        """
        if vertex_type:
            return [
                self.vertices[label]
                for label, vtype in self.vertex_types.items()
                if vtype == vertex_type
            ]
        return list(self.vertices.values())

    def get_relations(
        self,
        source: Optional[Any] = None,
        target: Optional[Any] = None,
        relation_type: Optional[RelationType] = None,
    ) -> List[Relation]:
        """
        Get relations matching criteria.

        Args:
            source: Optional source primitive
            target: Optional target primitive
            relation_type: Optional relation type filter

        Returns:
            List of matching relations
        """
        relations = []

        source_label = self._get_label(source) if source else None
        target_label = self._get_label(target) if target else None

        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if source_label and u != source_label:
                continue
            if target_label and v != target_label:
                continue
            if relation_type and data.get("relation_type") != relation_type:
                continue

            relation = data.get("relation")
            if relation:
                relations.append(relation)

        return relations

    def has_subgraph(self, pattern_graph: "GeometricSceneGraph") -> bool:
        """
        Check if this graph contains a subgraph matching the pattern.

        Args:
            pattern_graph: Pattern graph to match

        Returns:
            True if pattern is found
        """
        # Use NetworkX subgraph isomorphism
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            self.graph,
            pattern_graph.graph,
            node_match=self._node_match,
            edge_match=self._edge_match,
        )
        return matcher.subgraph_is_isomorphic()

    def find_subgraph_matches(self, pattern_graph: "GeometricSceneGraph") -> List[Dict[str, str]]:
        """
        Find all subgraph matches for a pattern.

        Args:
            pattern_graph: Pattern graph to match

        Returns:
            List of mappings from pattern labels to graph labels
        """
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            self.graph,
            pattern_graph.graph,
            node_match=self._node_match,
            edge_match=self._edge_match,
        )

        matches = []
        for mapping in matcher.subgraph_isomorphisms_iter():
            matches.append(mapping)

        return matches

    def _node_match(self, n1: Dict, n2: Dict) -> bool:
        """Node matching function for isomorphism."""
        return n1.get("type") == n2.get("type")

    def _edge_match(self, e1: Dict, e2: Dict) -> bool:
        """Edge matching function for isomorphism."""
        return e1.get("relation_type") == e2.get("relation_type")

    def _get_label(self, primitive: Any) -> str:
        """Get label for a primitive."""
        if hasattr(primitive, "label") and primitive.label:
            return primitive.label
        if isinstance(primitive, Point):
            return primitive.label
        if isinstance(primitive, Line) and primitive.point1 and primitive.point2:
            return f"line_{primitive.point1.label}_{primitive.point2.label}"
        if isinstance(primitive, Circle) and primitive.center:
            return f"circle_{primitive.center.label}"
        return str(id(primitive))

    def copy(self) -> "GeometricSceneGraph":
        """Create a deep copy of the graph."""
        new_graph = GeometricSceneGraph()
        new_graph.graph = self.graph.copy()
        new_graph.vertices = self.vertices.copy()
        new_graph.vertex_types = self.vertex_types.copy()
        return new_graph

    def merge(self, other: "GeometricSceneGraph") -> "GeometricSceneGraph":
        """
        Merge another graph into this one.

        Args:
            other: Graph to merge

        Returns:
            New merged graph
        """
        merged = self.copy()

        # Add vertices from other
        for vertex in other.get_vertices():
            merged.add_vertex(vertex)

        # Add edges from other
        for relation in other.get_relations():
            merged.add_edge(relation)

        return merged

    def __str__(self) -> str:
        """String representation of graph."""
        vertices_str = ", ".join(str(v) for v in self.get_vertices())
        edges_str = ", ".join(str(r) for r in self.get_relations())
        return f"Graph(V=[{vertices_str}], E=[{edges_str}])"

