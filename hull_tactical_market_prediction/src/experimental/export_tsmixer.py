"""Export Hull Tactical data to the Autoformer/TSMixer CSV layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write train/test CSVs compatible with TSMixer basic recipe."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Source train.csv from Kaggle download.")
    parser.add_argument("--test-csv", type=Path, required=True, help="Source test.csv from Kaggle download.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/research_exports/tsmixer"),
        help="Directory to store exported CSVs.",
    )
    parser.add_argument(
        "--target-column",
        default="forward_returns",
        help="Target column used for forecasting.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="date_id",
        help="Column representing chronological order.",
    )
    parser.add_argument(
        "--known-future-column",
        default="is_scored",
        help="Boolean column available for future steps in the competition test file.",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help="Value used to fill missing observations in the pivoted matrices.",
    )
    return parser


def _ensure_sorted(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    return df.sort_values(timestamp_col).reset_index(drop=True)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_df = _ensure_sorted(pd.read_csv(args.train_csv), args.timestamp_column)
    test_df = _ensure_sorted(pd.read_csv(args.test_csv), args.timestamp_column)

    # TSMixer expects the target column first followed by covariates.
    feature_cols = [c for c in train_df.columns if c not in {args.target_column}]
    ordered_train = train_df[[args.target_column, *feature_cols]]
    ordered_test = test_df[[c for c in ordered_train.columns if c in test_df.columns]]

    train_output = args.output_dir / "train.csv"
    test_output = args.output_dir / "test.csv"

    _write_csv(ordered_train, train_output)
    _write_csv(ordered_test, test_output)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
