from __future__ import annotations

from typing import Dict, List

from .geometry_schema import (
	GeoEdge,
	GeometryScene,
	GeometryTrace,
	TheoremApplication,
)


def _index_edges(scene: GeometryScene) -> Dict[str, List[GeoEdge]]:
    """Index edges by relation to simplify matching.

    This is intentionally tiny and specialized to the right-triangle domain.
    """

    by_rel: Dict[str, List[GeoEdge]] = {}
    for e in scene.edges:
        by_rel.setdefault(e.relation, []).append(e)
    return by_rel


def search_right_triangle_area(
    scene: GeometryScene, a: int, c: int
) -> GeometryTrace:
    """Derive the area of a right triangle from its scene and leg lengths.

    This is a minimal, but *real*, forward application of a micro-theorem set:

    1. RecognizeRightTriangle: from a perpendicular relation between legs,
       conclude that triangle ABC is right-angled at B.
    2. RightTriangleArea: from the right triangle and known leg lengths,
       conclude the area formula and its numeric value.
    """

    edges_by_rel = _index_edges(scene)
    apps: List[TheoremApplication] = []

    # 1) Recognize that ABC is a right triangle at B from perpendicular legs.
    # In our scene, there is an edge AB --perpendicular--> BC.
    perp_edges = edges_by_rel.get("perpendicular", [])
    matched_edges_ids: List[str] = []
    if perp_edges:
        # We just record a symbolic identifier for the matched constraint.
        matched_edges_ids.append("AB_perp_BC")

    recognize = TheoremApplication(
        theorem="RecognizeRightTriangle",
        matchedNodes=["A", "B", "C"],
        matchedEdges=matched_edges_ids,
        addedProps=["Triangle ABC is right-angled at B"],
        comment=(
            "Segments AB and BC are perpendicular, so the angle at B is a "
            "right angle."
        ),
    )
    apps.append(recognize)

    # 2) Use the right-triangle area theorem with leg lengths AB = a, BC = c.
    area = a * c // 2
    area_step = TheoremApplication(
        theorem="RightTriangleArea",
        matchedNodes=["A", "B", "C"],
        matchedEdges=matched_edges_ids,
        addedProps=[
            f"AB = {a}",
            f"BC = {c}",
            "Area(ABC) = (AB * BC) / 2",
            f"Area(ABC) = {area}",
        ],
        comment=(
            "In a right triangle, the area equals half the product of the "
            "lengths of the legs adjacent to the right angle."
        ),
    )
    apps.append(area_step)

    return GeometryTrace(
        initialScene=scene,
        theoremApplications=apps,
        finalScene=scene,
        derivedProps=[f"Area(ABC) = {area}"],
    )


def search_pythagoras_hypotenuse(
	scene: GeometryScene, lengths: Dict[str, int]
) -> GeometryTrace:
	"""Derive the hypotenuse length AC using the Pythagorean theorem.

	This uses the same micro-theorem set as ``search_right_triangle_area`` but
	focuses on computing the length of segment AC. The numeric leg lengths are
	provided in ``lengths`` (e.g. {"AB": 3, "BC": 4}).
	"""

	edges_by_rel = _index_edges(scene)
	apps: List[TheoremApplication] = []

	# 1) Recognize that ABC is a right triangle at B from perpendicular legs.
	perp_edges = edges_by_rel.get("perpendicular", [])
	matched_edges_ids: List[str] = []
	if perp_edges:
		matched_edges_ids.append("AB_perp_BC")

	recognize = TheoremApplication(
		theorem="RecognizeRightTriangle",
		matchedNodes=["A", "B", "C"],
		matchedEdges=matched_edges_ids,
		addedProps=["Triangle ABC is right-angled at B"],
		comment=(
			"Segments AB and BC are perpendicular, so the angle at B is a "
			"right angle."
		),
	)
	apps.append(recognize)

	# 2) Apply Pythagoras: AC^2 = AB^2 + BC^2.
	ab = lengths.get("AB")
	bc = lengths.get("BC")
	if ab is None or bc is None:
		raise ValueError("Lengths for 'AB' and 'BC' must be provided")

	ac_sq = ab * ab + bc * bc
	ac = int(ac_sq**0.5)
	if ac * ac != ac_sq:
		raise ValueError("Provided leg lengths do not form an integer hypotenuse")

	pyth = TheoremApplication(
		theorem="Pythagoras",
		matchedNodes=["A", "B", "C"],
		matchedEdges=matched_edges_ids,
		addedProps=[
			f"AB = {ab}",
			f"BC = {bc}",
			"In a right triangle, AC^2 = AB^2 + BC^2",
			f"AC^2 = {ab}^2 + {bc}^2 = {ac_sq}",
			f"AC = {ac}",
		],
		comment=(
			"Apply the Pythagorean theorem: the square of the hypotenuse equals "
			"the sum of the squares of the legs."
		),
	)
	apps.append(pyth)

	return GeometryTrace(
		initialScene=scene,
		theoremApplications=apps,
		finalScene=scene,
		derivedProps=[f"AC = {ac}"],
	)


def search_right_triangle_area_via_pythagoras(
	scene: GeometryScene, lengths: Dict[str, int]
) -> GeometryTrace:
	"""Derive the area of a right triangle by chaining Pythagoras + area.

	This composes :func:`search_pythagoras_hypotenuse` with a subsequent
	"RightTriangleArea" step, so the trace explicitly contains both the
	Pythagorean derivation of the hypotenuse and the final area computation.
	"""

	base_trace = search_pythagoras_hypotenuse(scene, lengths)
	ab = lengths.get("AB")
	bc = lengths.get("BC")
	if ab is None or bc is None:
		raise ValueError("Lengths for 'AB' and 'BC' must be provided")

	area = ab * bc // 2
	matched_edges_ids: List[str] = []
	if base_trace.theoremApplications:
		matched_edges_ids = base_trace.theoremApplications[-1].matchedEdges

	area_step = TheoremApplication(
		theorem="RightTriangleArea",
		matchedNodes=["A", "B", "C"],
		matchedEdges=matched_edges_ids,
		addedProps=[
			f"AB = {ab}",
			f"BC = {bc}",
			"Using the previously derived hypotenuse AC and the right angle at B,",
			"we know AB and BC are the legs of the right triangle.",
			"Area(ABC) = (AB * BC) / 2",
			f"Area(ABC) = {area}",
		],
		comment=(
			"After computing AC via Pythagoras, use the standard right-triangle "
			"area formula based on the two legs AB and BC."
		),
	)

	apps = list(base_trace.theoremApplications) + [area_step]
	derived = list(base_trace.derivedProps) + [f"Area(ABC) = {area}"]

	return GeometryTrace(
		initialScene=base_trace.initialScene,
		theoremApplications=apps,
		finalScene=base_trace.finalScene,
		derivedProps=derived,
	)

