"""Create an ensemble by averaging the predictions of multiple models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an ensemble by averaging predictions.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Path to a prediction (submission) CSV. Repeat for multiple models.",
    )
    parser.add_argument("--id-column", default="date_id", help="ID column name.")
    parser.add_argument("--prediction-column", default="prediction", help="Prediction column name.")
    parser.add_argument("--output-path", type=Path, default="submission_ensembled.csv", help="Path for the final submission file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.prediction:
        raise ValueError("At least one prediction file must be provided.")

    # Load and merge prediction files
    pred_dfs = [pd.read_csv(p) for p in args.prediction]
    merged_preds = pred_dfs[0]
    for i, df in enumerate(pred_dfs[1:]):
        merged_preds = pd.merge(
            merged_preds,
            df,
            on=args.id_column,
            suffixes=("", f"_model_{i + 1}"),
        )

    # Average the predictions
    pred_cols = [col for col in merged_preds.columns if args.prediction_column in col]
    if not pred_cols:
        raise ValueError(
            f"No columns containing '{args.prediction_column}' found after merging predictions."
        )
    ensembled = merged_preds[pred_cols].mean(axis=1)

    # Create submission file using the requested prediction column name
    submission = merged_preds[[args.id_column]].copy()
    submission[args.prediction_column] = ensembled
    submission.to_csv(args.output_path, index=False)

    print(f"Ensembling complete. Final submission saved to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
