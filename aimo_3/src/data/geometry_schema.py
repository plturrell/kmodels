from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .schema import MathProblem


# ---------------------------------------------------------------------------
# Geometry scene graph and proof trace
# ---------------------------------------------------------------------------


class GeoNode(BaseModel):
    """A node in the geometric scene graph (point, line, circle, etc.)."""

    id: str
    type: str  # e.g. "point", "line", "circle"
    label: Optional[str] = None  # e.g. "A", "B", "(O)"


class GeoEdge(BaseModel):
    """A labeled relation between geometric entities.

    The `relation` field should be drawn from a controlled vocabulary, e.g.::

        "incident", "parallel", "perpendicular", "tangent",
        "equal_angle", "equal_segment", "concyclic", ...
    """

    source: str
    target: str
    relation: str


class GeometryScene(BaseModel):
    """The geometric world model G = (V, E)."""

    nodes: List[GeoNode]
    edges: List[GeoEdge]


class TheoremApplication(BaseModel):
    """One application of a theorem T in the deductive search."""

    theorem: str
    matchedNodes: List[str] = []
    matchedEdges: List[str] = []
    addedNodes: List[GeoNode] = []
    addedEdges: List[GeoEdge] = []
    addedProps: List[str] = []  # human-readable propositions from Φ_addition
    comment: Optional[str] = None


class GeometryTrace(BaseModel):
    """Full trace of reasoning over a geometric scene.

    This mirrors the formal sequence S_0, ..., S_n in the project spec.
    """

    initialScene: GeometryScene
    theoremApplications: List[TheoremApplication]
    finalScene: GeometryScene
    derivedProps: List[str] = []  # selected propositions from Φ_n


class GeometryPayload(BaseModel):
    """Typed payload to be stored under MathProblem.extension["geometry"]."""

    scene: GeometryScene
    trace: GeometryTrace
    goalDescription: Optional[str] = None  # natural-language target description
    targetQuantity: Optional[str] = None  # e.g. "angle_BAD" or "length_BD"
    extra: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def attach_geometry(problem: MathProblem, payload: GeometryPayload) -> MathProblem:
    """Attach a geometry payload into a MathProblem.extension.

    This keeps the core schema stable while allowing rich geometry structure
    for parsing and proof traces.
    """

    problem.extension.setdefault("geometry", payload.model_dump())
    return problem

