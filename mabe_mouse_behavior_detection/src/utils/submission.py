"""Submission utilities for the MABe mouse behaviour detection workspace."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def build_submission(
    run_dir: Path,
    sample_submission: Path,
    *,
    output_path: Path,
    id_column: str = "row_id",
    prediction_column: str = "prediction",
    extra_columns: Iterable[str] | None = None,
    allow_missing: bool = False,
) -> Path:
    """Project the run-level predictions onto the Kaggle sample submission order."""
    run_dir = run_dir.resolve()
    submission_csv = run_dir / "submission.csv"
    if not submission_csv.exists():
        raise FileNotFoundError(f"Expected submission.csv in {run_dir}")

    predictions = pd.read_csv(submission_csv)
    required_cols: List[str] = [id_column, prediction_column]
    if extra_columns:
        for column in extra_columns:
            if column not in predictions.columns:
                raise KeyError(f"Column '{column}' not present in {submission_csv}")
            required_cols.append(column)

    missing = [column for column in required_cols if column not in predictions.columns]
    if missing:
        raise KeyError(f"Prediction file missing columns: {', '.join(missing)}")

    sample_df = pd.read_csv(sample_submission)
    merged = sample_df[[id_column]].merge(
        predictions[required_cols],
        on=id_column,
        how="left",
    )
    if not allow_missing and merged[prediction_column].isna().any():
        missing_ids = merged[merged[prediction_column].isna()][id_column].tolist()
        raise ValueError(
            f"{len(missing_ids)} rows missing predictions after merge. "
            f"Examples: {missing_ids[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align run predictions with the Kaggle sample submission.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing submission.csv.")
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-column", default="row_id")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument(
        "--extra-column",
        dest="extra_columns",
        action="append",
        default=[],
        help="Additional probability columns to keep alongside the main prediction.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not raise if some rows are missing predictions after alignment.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    build_submission(
        args.run_dir,
        args.sample_submission,
        output_path=args.output,
        id_column=args.id_column,
        prediction_column=args.prediction_column,
        extra_columns=args.extra_columns,
        allow_missing=args.allow_missing,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

