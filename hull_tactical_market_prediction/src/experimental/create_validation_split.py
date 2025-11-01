"""Utility to build a time-aware validation split for Hull Tactical data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def create_time_aware_validation(
    train_path: Path,
    output_dir: Path,
    val_ratio: float,
    date_column: str,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading training data from {train_path}")
    train_data = _load_frame(train_path)

    if date_column not in train_data.columns:
        print(
            f"Date column '{date_column}' not found. Falling back to shuffled split "
            f"(set --date-column to an existing field for chronological split)."
        )
        rng = np.random.default_rng(seed)
        mask = rng.random(len(train_data)) < (1 - val_ratio)
        train_split = train_data[mask]
        val_split = train_data[~mask]
    else:
        train_data = train_data.sort_values(date_column)
        unique_dates = train_data[date_column].unique()
        split_idx = int(len(unique_dates) * (1 - val_ratio))
        train_dates = unique_dates[:split_idx]
        val_dates = unique_dates[split_idx:]
        train_split = train_data[train_data[date_column].isin(train_dates)]
        val_split = train_data[train_data[date_column].isin(val_dates)]

    print(
        f"Train rows: {len(train_split):,} | Validation rows: {len(val_split):,} | "
        f"Validation ratio: {val_ratio:.2f}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path_out = output_dir / "train_split.parquet"
    val_path_out = output_dir / "validation.parquet"

    print(f"Saving train split to {train_path_out}")
    train_split.to_parquet(train_path_out, index=False)

    print(f"Saving validation split to {val_path_out}")
    val_split.to_parquet(val_path_out, index=False)

    return train_split, val_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a time-aware validation split.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/data/raw/train.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--date-column", default="date_id")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the shuffled split fallback.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_time_aware_validation(
        train_path=args.train_path,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        date_column=args.date_column,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
