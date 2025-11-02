"""CLI to run domain learning diagnostics for CSIRO biomass models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..analysis.domain_learning_tests import run_domain_learning_assessment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Domain learning assessment for CSIRO biomass predictions.")
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to the training CSV (long format).")
    parser.add_argument("--oof-prediction", type=Path, required=True, help="Path to OOF prediction CSV.")
    parser.add_argument(
        "--id-column",
        default="sample_id",
        help="Identifier column shared between train and prediction files.",
    )
    parser.add_argument(
        "--target-name-column",
        default="target_name",
        help="Target name column in the training data (long format).",
    )
    parser.add_argument(
        "--target-value-column",
        default="target",
        help="Target value column in the training data (long format).",
    )
    parser.add_argument(
        "--species-column",
        default="Species",
        help="Optional species column in the training data.",
    )
    parser.add_argument(
        "--state-column",
        default="State",
        help="Optional state column in the training data.",
    )
    parser.add_argument(
        "--metadata-column",
        action="append",
        default=[],
        help="Numeric metadata column(s) for bias checks (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("competitions/csiro_biomass/outputs/domain_assessment"),
        help="Directory to store the assessment report.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_df = pd.read_csv(args.train_csv)
    prediction_df = pd.read_csv(args.oof_prediction)

    # Infer target names from training data
    target_names = sorted(train_df[args.target_name_column].dropna().unique().tolist())
    if not target_names:
        raise ValueError("Could not infer target names from training data.")

    # Ensure predictions have actual columns; merge when needed
    actual_cols = [f"{name}_actual" for name in target_names]
    if any(col not in prediction_df.columns for col in actual_cols):
        pivot = (
            train_df.pivot_table(
                index=args.id_column,
                columns=args.target_name_column,
                values=args.target_value_column,
            )
            .reset_index()
            .rename(columns={name: f"{name}_actual" for name in target_names})
        )
        prediction_df = prediction_df.merge(pivot, on=args.id_column, how="left")

    # Prepare grouping series
    species_series = None
    if args.species_column in train_df.columns:
        species_series = train_df.groupby(args.id_column)[args.species_column].first()
        species_series.name = args.species_column

    state_series = None
    if args.state_column in train_df.columns:
        state_series = train_df.groupby(args.id_column)[args.state_column].first()
        state_series.name = args.state_column

    metadata_df = None
    if args.metadata_column:
        metadata_df = (
            train_df.groupby(args.id_column)[args.metadata_column].mean()
            .reset_index()
            .fillna(0.0)
        )

    report = run_domain_learning_assessment(
        oof_predictions=prediction_df,
        target_names=target_names,
        id_column=args.id_column,
        species_series=species_series,
        state_series=state_series,
        metadata_features=metadata_df,
        metadata_columns=args.metadata_column,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "domain_learning_assessment.json"
    with output_path.open("w") as fp:
        json.dump(report.to_dict(), fp, indent=2)

    print("=" * 70)
    print("CSIRO BIOMASS DOMAIN LEARNING ASSESSMENT")
    print("=" * 70)
    if report.group_stability:
        gs = report.group_stability
        print(f"Group metric ({gs.metric}) coefficient of variation: {gs.coefficient_of_variation:.3f} | Learning={gs.is_learning}")
    if report.metadata_bias:
        for feature, res in report.metadata_bias.items():
            print(f"Metadata bias {feature}: max |corr|={res.max_abs_correlation:.3f} | Learning={res.is_learning}")
    print(f"\nTests passed: {report.passed_tests}/{report.total_tests}")
    print(f"Learning score: {report.learning_score:.2%}")
    print(f"Report saved to: {output_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
