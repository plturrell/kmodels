from __future__ import annotations

import argparse
import json
import random
import time
from typing import List

from .schema import (
    Difficulty,
    MathProblem,
    Owner,
    SourceInfo,
    SourceType,
    SpanKind,
    SpanStream,
    Split,
    TokenSpan,
    TopicTag,
)
from .geometry_schema import (
	GeoEdge,
	GeoNode,
	GeometryPayload,
	GeometryScene,
	attach_geometry,
)
from .geometry_engine import (
	search_pythagoras_hypotenuse,
	search_right_triangle_area,
	search_right_triangle_area_via_pythagoras,
)


def _approx_tokens(text: str) -> int:
    # Very rough heuristic; real tokenization happens downstream.
    return max(1, int(len(text.split()) * 1.3))


def _make_right_triangle_problem(idx: int) -> MathProblem:
    """Generate one synthetic right-triangle area problem.

    Configuration: triangle ABC with right angle at B, AB = a, BC = c.
    Target: area(ABC) = a * c / 2 (integer).
    """

    a = random.randint(2, 20)
    c = 2 * random.randint(2, 20)  # ensure area integer
    area = a * c // 2

    problem_latex = (
        f"Triangle $ABC$ is right-angled at $B$. The legs have lengths "
        f"$AB = {a}$ and $BC = {c}$. Compute the area of triangle $ABC$."
    )
    solution_latex = (
        "Since $ABC$ is right-angled at $B$, the legs $AB$ and $BC$ are "
        "perpendicular. The area of a right triangle is given by "
        "${1 \\over 2} \\cdot AB \\cdot BC$, so\\n"
        f"$[ABC] = {{1 \\over 2}} \\cdot {a} \\cdot {c} = {area}$."
    )

    # Scene graph: points and segment-entities with basic relations.
    nodes = [
        GeoNode(id="A", type="point", label="A"),
        GeoNode(id="B", type="point", label="B"),
        GeoNode(id="C", type="point", label="C"),
        GeoNode(id="AB", type="segment", label="AB"),
        GeoNode(id="BC", type="segment", label="BC"),
        GeoNode(id="AC", type="segment", label="AC"),
    ]
    edges = [
        GeoEdge(source="A", target="AB", relation="incident"),
        GeoEdge(source="B", target="AB", relation="incident"),
        GeoEdge(source="B", target="BC", relation="incident"),
        GeoEdge(source="C", target="BC", relation="incident"),
        GeoEdge(source="A", target="AC", relation="incident"),
        GeoEdge(source="C", target="AC", relation="incident"),
        GeoEdge(source="AB", target="BC", relation="perpendicular"),
        GeoEdge(source="AC", target="BC", relation="hypotenuse"),
    ]
    initial_scene = GeometryScene(nodes=nodes, edges=edges)

    # Run a tiny, but real, search over the scene using our micro-theorem set.
    trace = search_right_triangle_area(initial_scene, a=a, c=c)

    payload = GeometryPayload(
        scene=initial_scene,
        trace=trace,
        goalDescription="Compute the area of right triangle ABC.",
        targetQuantity="area_ABC",
    )

    now_ms = int(time.time() * 1000)
    problem_id = f"geom_right_triangle_{idx}"

    spans: List[TokenSpan] = []
    spans.append(
        TokenSpan(
            stream=SpanStream.PROMPT,
            role="user",
            kind=SpanKind.PROBLEM,
            text=problem_latex,
            approxTokens=_approx_tokens(problem_latex),
        )
    )
    spans.append(
        TokenSpan(
            stream=SpanStream.COMPLETION,
            role="assistant",
            kind=SpanKind.SCRATCHPAD,
            text=solution_latex,
            approxTokens=_approx_tokens(solution_latex),
        )
    )
    spans.append(
        TokenSpan(
            stream=SpanStream.COMPLETION,
            role="assistant",
            kind=SpanKind.FINAL_ANSWER,
            text=str(area),
            approxTokens=1,
        )
    )

    mp = MathProblem(
        id=problem_id,
        fullyQualifiedName=f"aimo.geom.right_triangle.{idx}",
        version=1,
        createdAt=now_ms,
        updatedAt=now_ms,
        name=f"Right triangle area #{idx}",
        language="en",
        problemLatex=problem_latex,
        problemPlaintext=problem_latex,
        answer=area,
        topicTags=[TopicTag.GEOMETRY],
        difficulty=Difficulty.TRAINING_EASY,
        source=SourceInfo(
            sourceType=SourceType.SYNTHETIC,
            reference="geometry_right_triangle_v1",
        ),
        license="CC-BY-4.0",
        tags=["geometry", "right_triangle", "area"],
        domains=["olympiad_math"],
        owners=[
            Owner(name="aimo_generator", role="generator"),
        ],
        estimatedTimeMinutes=1,
        reviewStatus=None,
        solutionLatex=solution_latex,
        solutionSteps=[],
        toolTrace=[],
        split=Split.TRAIN,
        tokenSpans=spans,
    )

    return attach_geometry(mp, payload)


def _make_pythagoras_hypotenuse_problem(idx: int) -> MathProblem:
    """Generate a synthetic right-triangle hypotenuse problem.

    Triangle ABC is right-angled at B. Legs AB and BC are given; the task is to
    compute the hypotenuse AC via the Pythagorean theorem.
    """

    primitive_triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
    base_a, base_b, base_c = random.choice(primitive_triples)
    scale = random.randint(1, 10)
    ab = base_a * scale
    bc = base_b * scale
    ac = base_c * scale

    problem_latex = (
        f"Triangle $ABC$ is right-angled at $B$. The legs have lengths "
        f"$AB = {ab}$ and $BC = {bc}$. Compute the length of the hypotenuse $AC$."
    )
    solution_latex = (
        "Since $ABC$ is right-angled at $B$, we can apply the Pythagorean "
        "theorem: $AC^2 = AB^2 + BC^2$. Substituting the given values, we get "
        f"$AC^2 = {ab}^2 + {bc}^2 = {ab**2 + bc**2}$, so $AC = {ac}$."
    )

    # Same scene graph as the area problems.
    nodes = [
        GeoNode(id="A", type="point", label="A"),
        GeoNode(id="B", type="point", label="B"),
        GeoNode(id="C", type="point", label="C"),
        GeoNode(id="AB", type="segment", label="AB"),
        GeoNode(id="BC", type="segment", label="BC"),
        GeoNode(id="AC", type="segment", label="AC"),
    ]
    edges = [
        GeoEdge(source="A", target="AB", relation="incident"),
        GeoEdge(source="B", target="AB", relation="incident"),
        GeoEdge(source="B", target="BC", relation="incident"),
        GeoEdge(source="C", target="BC", relation="incident"),
        GeoEdge(source="A", target="AC", relation="incident"),
        GeoEdge(source="C", target="AC", relation="incident"),
        GeoEdge(source="AB", target="BC", relation="perpendicular"),
        GeoEdge(source="AC", target="BC", relation="hypotenuse"),
    ]
    initial_scene = GeometryScene(nodes=nodes, edges=edges)

    trace = search_pythagoras_hypotenuse(
        initial_scene, lengths={"AB": ab, "BC": bc}
    )

    payload = GeometryPayload(
        scene=initial_scene,
        trace=trace,
        goalDescription="Compute the length of hypotenuse AC.",
        targetQuantity="length_AC",
    )

    now_ms = int(time.time() * 1000)
    problem_id = f"geom_pythag_hyp_{idx}"

    spans: List[TokenSpan] = []
    spans.append(
        TokenSpan(
            stream=SpanStream.PROMPT,
            role="user",
            kind=SpanKind.PROBLEM,
            text=problem_latex,
            approxTokens=_approx_tokens(problem_latex),
        )
    )
    spans.append(
        TokenSpan(
            stream=SpanStream.COMPLETION,
            role="assistant",
            kind=SpanKind.SCRATCHPAD,
            text=solution_latex,
            approxTokens=_approx_tokens(solution_latex),
        )
    )
    spans.append(
        TokenSpan(
            stream=SpanStream.COMPLETION,
            role="assistant",
            kind=SpanKind.FINAL_ANSWER,
            text=str(ac),
            approxTokens=1,
        )
    )

    mp = MathProblem(
        id=problem_id,
        fullyQualifiedName=f"aimo.geom.pythag_hyp.{idx}",
        version=1,
        createdAt=now_ms,
        updatedAt=now_ms,
        name=f"Pythagorean hypotenuse #{idx}",
        language="en",
        problemLatex=problem_latex,
        problemPlaintext=problem_latex,
        answer=ac,
        topicTags=[TopicTag.GEOMETRY],
        difficulty=Difficulty.TRAINING_EASY,
        source=SourceInfo(
            sourceType=SourceType.SYNTHETIC,
            reference="geometry_pythag_hyp_v1",
        ),
        license="CC-BY-4.0",
        tags=["geometry", "right_triangle", "pythagoras"],
        domains=["olympiad_math"],
        owners=[Owner(name="aimo_generator", role="generator")],
        estimatedTimeMinutes=1,
        reviewStatus=None,
        solutionLatex=solution_latex,
        solutionSteps=[],
        toolTrace=[],
        split=Split.TRAIN,
        tokenSpans=spans,
    )

    return attach_geometry(mp, payload)


def _make_right_triangle_area_chained_problem(idx: int) -> MathProblem:
    """Area-of-right-triangle problem whose trace chains through Pythagoras.

    We first derive AC via the Pythagorean theorem, then use the standard
    area formula for a right triangle.
    """

    primitive_triples = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25)]
    base_a, base_b, base_c = random.choice(primitive_triples)
    scale = random.randint(1, 10)
    ab = base_a * scale
    bc = base_b * scale
    ac = base_c * scale
    area = ab * bc // 2

    problem_latex = (
        f"Triangle $ABC$ is right-angled at $B$. The legs have lengths "
        f"$AB = {ab}$ and $BC = {bc}$. Compute the area of triangle $ABC$."
    )
    solution_latex = (
        "First, apply the Pythagorean theorem to find the hypotenuse: "
        f"$AC^2 = {ab}^2 + {bc}^2 = {ab**2 + bc**2}$, so $AC = {ac}$. "
        "Since $ABC$ is right-angled at $B$, its area is "
        "${1 \\over 2} AB \\cdot BC$, so "
        f"$[ABC] = {{1 \\over 2}} \\cdot {ab} \\cdot {bc} = {area}$."
    )

    # Same scene graph as the other right-triangle problems.
    nodes = [
        GeoNode(id="A", type="point", label="A"),
        GeoNode(id="B", type="point", label="B"),
        GeoNode(id="C", type="point", label="C"),
        GeoNode(id="AB", type="segment", label="AB"),
        GeoNode(id="BC", type="segment", label="BC"),
        GeoNode(id="AC", type="segment", label="AC"),
    ]
    edges = [
        GeoEdge(source="A", target="AB", relation="incident"),
        GeoEdge(source="B", target="AB", relation="incident"),
        GeoEdge(source="B", target="BC", relation="incident"),
        GeoEdge(source="C", target="BC", relation="incident"),
        GeoEdge(source="A", target="AC", relation="incident"),
        GeoEdge(source="C", target="AC", relation="incident"),
        GeoEdge(source="AB", target="BC", relation="perpendicular"),
        GeoEdge(source="AC", target="BC", relation="hypotenuse"),
    ]
    initial_scene = GeometryScene(nodes=nodes, edges=edges)

    trace = search_right_triangle_area_via_pythagoras(
        initial_scene, lengths={"AB": ab, "BC": bc}
    )

    payload = GeometryPayload(
        scene=initial_scene,
        trace=trace,
        goalDescription="Compute the area of right triangle ABC.",
        targetQuantity="area_ABC",
    )

    now_ms = int(time.time() * 1000)
    problem_id = f"geom_right_triangle_chain_{idx}"

    spans: List[TokenSpan] = []
    spans.append(
        TokenSpan(
            stream=SpanStream.PROMPT,
            role="user",
            kind=SpanKind.PROBLEM,
            text=problem_latex,
            approxTokens=_approx_tokens(problem_latex),
        )
    )
    spans.append(
        TokenSpan(
            stream=SpanStream.COMPLETION,
            role="assistant",
            kind=SpanKind.SCRATCHPAD,
            text=solution_latex,
            approxTokens=_approx_tokens(solution_latex),
        )
    )
    spans.append(
        TokenSpan(
            stream=SpanStream.COMPLETION,
            role="assistant",
            kind=SpanKind.FINAL_ANSWER,
            text=str(area),
            approxTokens=1,
        )
    )

    mp = MathProblem(
        id=problem_id,
        fullyQualifiedName=f"aimo.geom.right_triangle_chain.{idx}",
        version=1,
        createdAt=now_ms,
        updatedAt=now_ms,
        name=f"Right triangle area (chained) #{idx}",
        language="en",
        problemLatex=problem_latex,
        problemPlaintext=problem_latex,
        answer=area,
        topicTags=[TopicTag.GEOMETRY],
        difficulty=Difficulty.TRAINING_EASY,
        source=SourceInfo(
            sourceType=SourceType.SYNTHETIC,
            reference="geometry_right_triangle_chain_v1",
        ),
        license="CC-BY-4.0",
        tags=["geometry", "right_triangle", "pythagoras", "area"],
        domains=["olympiad_math"],
        owners=[Owner(name="aimo_generator", role="generator")],
        estimatedTimeMinutes=1,
        reviewStatus=None,
        solutionLatex=solution_latex,
        solutionSteps=[],
        toolTrace=[],
        split=Split.TRAIN,
        tokenSpans=spans,
    )

    return attach_geometry(mp, payload)


def generate_examples(num_examples: int, kind: str = "area") -> List[MathProblem]:
    """Generate a batch of geometry problems.

    Args:
        kind: "area" for direct area-of-right-triangle problems,
              "hypotenuse" for Pythagoras problems,
              "area_chained" to chain Pythagoras + area.
    """

    if kind == "hypotenuse":
        return [
            _make_pythagoras_hypotenuse_problem(i)
            for i in range(1, num_examples + 1)
        ]
    if kind == "area_chained":
        return [
            _make_right_triangle_area_chained_problem(i)
            for i in range(1, num_examples + 1)
        ]
    # default: simple area problems
    return [_make_right_triangle_problem(i) for i in range(1, num_examples + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic geometry problems."
    )
    parser.add_argument(
        "--num", type=int, default=10, help="Number of problems to generate"
    )
    parser.add_argument("--out", type=str, required=True, help="Output JSONL path")
    parser.add_argument(
        "--kind",
        choices=["area", "hypotenuse", "area_chained"],
        default="area",
        help="Type of geometry problem to generate",
    )
    args = parser.parse_args()

    problems = generate_examples(args.num, kind=args.kind)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

