"""CLI for CAFA-6 domain learning diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..analysis.domain_learning_tests import run_domain_learning_assessment


def _load_predictions(path: Path, id_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        raise KeyError(f"Identifier column '{id_column}' not found in {path}")
    return frame


def _infer_class_names(frame: pd.DataFrame, id_column: str) -> list[str]:
    candidates: list[str] = []
    for column in frame.columns:
        if column == id_column or column.endswith('_actual'):
            continue
        candidates.append(column)
    if not candidates:
        raise ValueError('Unable to infer class names from predictions.')
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Domain learning assessment for CAFA-6 predictions.')
    parser.add_argument('--predictions', type=Path, required=True, help='Path to OOF prediction CSV file.')
    parser.add_argument('--class-names', type=Path, help='Optional JSON file containing ordered class names.')
    parser.add_argument('--id-column', default='accession', help='Identifier column present in the prediction file.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold used for binary decisions (default: 0.5).')
    parser.add_argument('--min-group-size', type=int, default=20, help='Minimum samples required to evaluate a group (default: 20).')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/domain_assessment'), help='Directory to store the report.')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    predictions_path = args.predictions.resolve()
    frame = _load_predictions(predictions_path, args.id_column)

    if args.class_names:
        class_names = json.loads(Path(args.class_names).read_text())
    else:
        class_names = _infer_class_names(frame, args.id_column)

    target_cols = [f"{name}_actual" for name in class_names]
    missing = [col for col in target_cols if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing ground-truth columns: {', '.join(missing)}")

    report = run_domain_learning_assessment(
        oof_predictions=frame,
        class_names=class_names,
        threshold=args.threshold,
        min_group_size=args.min_group_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / 'domain_learning_assessment.json'
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding='utf-8')

    print('=' * 70)
    print('CAFA-6 DOMAIN LEARNING ASSESSMENT')
    print('=' * 70)
    for group in report.density_groups:
        if group.sample_count == 0:
            continue
        print(f"Group {group.group}: n={group.sample_count} macroF1={group.macro_f1:.3f} microF1={group.micro_f1:.3f}")
    if report.class_bias is not None:
        bias = report.class_bias
        print(f"Class bias max |bias|={bias.max_abs_bias:.3f} | Learning={bias.is_learning}")
    print(f"\nTests passed: {report.passed_tests}/{report.total_tests}")
    print(f"Learning score: {report.learning_score:.2%}")
    print(f"Report saved to: {output_path}")
    print('=' * 70)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
