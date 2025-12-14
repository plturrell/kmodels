"""Script to generate training data using the problem generator."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.geometry.generator import ProblemGenerator


_AIMO3_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _AIMO3_ROOT / "data"


def _safe_output_file(path: Path) -> Path:
    """Constrain outputs to a file name under aimo_3/data/."""
    return (_DATA_DIR / Path(path).name).resolve()


def main():
    parser = argparse.ArgumentParser(description="Generate geometric training problems")
    parser.add_argument(
        "--num_problems",
        type=int,
        default=100,
        help="Number of problems to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="balanced",
        choices=["balanced", "triangle_heavy", "circle_heavy", "coordinate_heavy"],
        help="Problem distribution strategy",
    )

    args = parser.parse_args()

    generator = ProblemGenerator(seed=args.seed)

    # Define problem distribution
    if args.distribution == "balanced":
        problem_types = [
            # Triangles
            {"family": "triangle", "type": "right", "difficulty": "easy"},
            {"family": "triangle", "type": "right", "difficulty": "medium"},
            {"family": "triangle", "type": "right", "difficulty": "hard"},
            {"family": "triangle", "type": "equilateral", "difficulty": "medium"},
            {"family": "triangle", "type": "isosceles", "difficulty": "medium"},
            {"family": "triangle", "type": "scalene", "difficulty": "hard"},
            # Circles
            {"family": "circle", "type": "inscribed", "difficulty": "easy"},
            {"family": "circle", "type": "inscribed", "difficulty": "medium"},
            {"family": "circle", "type": "circumscribed", "difficulty": "medium"},
            {"family": "circle", "type": "tangent", "difficulty": "medium"},
            {"family": "circle", "type": "chord", "difficulty": "hard"},
            # Coordinate
            {"family": "coordinate", "type": "distance", "difficulty": "easy"},
            {"family": "coordinate", "type": "distance", "difficulty": "medium"},
            {"family": "coordinate", "type": "midpoint", "difficulty": "medium"},
            {"family": "coordinate", "type": "slope", "difficulty": "medium"},
            {"family": "coordinate", "type": "area", "difficulty": "hard"},
        ]
    elif args.distribution == "triangle_heavy":
        problem_types = [
            {"family": "triangle", "type": "right", "difficulty": d}
            for d in ["easy", "medium", "hard"]
        ] * 5
        problem_types.extend([
            {"family": "circle", "type": "inscribed", "difficulty": "medium"},
            {"family": "coordinate", "type": "distance", "difficulty": "medium"},
        ])
    elif args.distribution == "circle_heavy":
        problem_types = [
            {"family": "circle", "type": t, "difficulty": d}
            for t in ["inscribed", "circumscribed", "tangent", "chord"]
            for d in ["easy", "medium", "hard"]
        ]
        problem_types.extend([
            {"family": "triangle", "type": "right", "difficulty": "medium"},
            {"family": "coordinate", "type": "distance", "difficulty": "medium"},
        ])
    else:  # coordinate_heavy
        problem_types = [
            {"family": "coordinate", "type": t, "difficulty": d}
            for t in ["distance", "midpoint", "slope", "area"]
            for d in ["easy", "medium", "hard"]
        ]
        problem_types.extend([
            {"family": "triangle", "type": "right", "difficulty": "medium"},
            {"family": "circle", "type": "inscribed", "difficulty": "medium"},
        ])

    # Generate problems
    print(f"Generating {args.num_problems} problems...")
    problems = generator.generate_batch(problem_types, num_problems=args.num_problems)

    # Format output
    output_data = []
    for i, (problem, answer, metadata) in enumerate(problems):
        output_data.append({
            "problem_id": f"generated_{i+1:04d}",
            "statement": problem,
            "answer": answer,
            "metadata": metadata,
        })

    # Save to file
    output_file = _safe_output_file(Path("generated_problems.json"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Generated {len(output_data)} problems and saved to {output_file}")

    # Print statistics
    family_counts = {}
    type_counts = {}
    difficulty_counts = {}

    for item in output_data:
        meta = item["metadata"]
        family = meta["family"]
        ptype = meta["type"]
        difficulty = meta["difficulty"]

        family_counts[family] = family_counts.get(family, 0) + 1
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    print("\nStatistics:")
    print(f"  By family: {family_counts}")
    print(f"  By type: {type_counts}")
    print(f"  By difficulty: {difficulty_counts}")


if __name__ == "__main__":
    main()

