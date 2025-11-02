"""CLI for CAFA-6 learning diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..analysis.learning_assessment import run_learning_assessment


def _load_dataframe(path: Path, id_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        raise KeyError(f"Identifier column '{id_column}' not present in {path}")
    return frame


def _infer_class_names(frame: pd.DataFrame, id_column: str) -> List[str]:
    candidates: List[str] = []
    for column in frame.columns:
        if column == id_column or column.endswith('_actual'):
            continue
        candidates.append(column)
    if not candidates:
        raise ValueError("Unable to infer class names from prediction frame.")
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess learning behaviour for CAFA-6 baselines.")
    parser.add_argument(
        "--predictions",
        type=Path,
        action="append",
        required=True,
        help="Path(s) to prediction CSV files containing probabilities and *_actual columns.",
    )
    parser.add_argument("--class-names", type=Path, help="Optional JSON file listing class names in order.")
    parser.add_argument("--id-column", default="accession", help="Identifier column present in prediction files.")
    parser.add_argument(
        "--feature-matrix",
        type=Path,
        help="Optional CSV with handcrafted features for compression tests.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for binary decisions (default: 0.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/learning_assessment"),
        help="Directory to store assessment artefacts.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    prediction_paths = [path.resolve() for path in args.predictions]
    frames = [_load_dataframe(path, args.id_column) for path in prediction_paths]

    if args.class_names:
        class_names = json.loads(Path(args.class_names).read_text())
    else:
        class_names = _infer_class_names(frames[0], args.id_column)

    for frame in frames:
        missing = [f"{name}_actual" for name in class_names if f"{name}_actual" not in frame.columns]
        if missing:
            raise KeyError(f"Missing ground-truth columns: {', '.join(missing)}")

    primary = frames[0]
    diversity_inputs: Dict[str, pd.DataFrame] = {}
    if len(frames) > 1:
        for path, frame in zip(prediction_paths, frames):
            diversity_inputs[path.stem] = frame

    feature_frame = None
    if args.feature_matrix:
        feature_frame = pd.read_csv(args.feature_matrix)
        if args.id_column in feature_frame.columns:
            feature_frame = (
                feature_frame.set_index(args.id_column)
                .reindex(primary[args.id_column])
                .fillna(0.0)
            )

    assessment = run_learning_assessment(
        oof_predictions=primary,
        class_names=class_names,
        threshold=args.threshold,
        diversity_inputs=diversity_inputs if len(diversity_inputs) >= 2 else None,
        feature_frame=feature_frame,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "learning_assessment.json"
    output_path.write_text(json.dumps(assessment.to_dict(), indent=2), encoding="utf-8")

    print("=" * 70)
    print("CAFA-6 LEARNING ASSESSMENT")
    print("=" * 70)
    gen = assessment.generalization
    print(f"Macro F1: {gen.macro_f1:.4f} (baseline {gen.baseline_macro_f1:.4f})")
    print(f"Micro F1: {gen.micro_f1:.4f} (baseline {gen.baseline_micro_f1:.4f})")
    print(f"Coverage: {gen.coverage:.3f} (baseline {gen.baseline_coverage:.3f})")
    if assessment.diversity is not None:
        div = assessment.diversity
        print(f"Diversity mean: {div.mean_diversity:.3f} | Learning={div.is_learning}")
    if assessment.compression is not None:
        comp = assessment.compression
        print(f"Compression ratio: {comp.compression_ratio:.3f} | Learning={comp.is_learning}")
    print(f"\nLearning score: {assessment.learning_score:.2%}")
    print(f"Tests passed: {assessment.passed_tests}/{assessment.total_tests}")
    print(f"Report saved to: {output_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
