"""CLI to assess learning behaviour of CSIRO biomass models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..analysis.learning_assessment import run_learning_assessment


def _infer_target_names(
    train_df: pd.DataFrame,
    *,
    target_name_column: str,
) -> List[str]:
    names = sorted(train_df[target_name_column].dropna().unique().tolist())
    if not names:
        raise ValueError("Could not infer target names from training data.")
    return names


def _pivot_training_targets(
    train_df: pd.DataFrame,
    *,
    id_column: str,
    target_name_column: str,
    target_value_column: str,
) -> pd.DataFrame:
    pivot = (
        train_df.pivot_table(
            index=id_column,
            columns=target_name_column,
            values=target_value_column,
        )
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def _load_metadata_features(
    train_df: pd.DataFrame,
    *,
    id_column: str,
    metadata_columns: Sequence[str],
) -> pd.DataFrame:
    if not metadata_columns:
        return pd.DataFrame()
    grouped = train_df.groupby(id_column)[list(metadata_columns)].first().reset_index()
    numeric_cols = grouped.select_dtypes(include=[np.number]).columns.tolist()
    keep_cols = [col for col in [id_column, *numeric_cols] if col in grouped.columns]
    if len(keep_cols) <= 1:
        return pd.DataFrame()
    grouped = grouped[keep_cols].copy()
    grouped[numeric_cols] = grouped[numeric_cols].fillna(0.0)
    return grouped


def _load_oof_predictions(
    paths: Sequence[Path],
    *,
    target_names: Sequence[str],
    id_column: str,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if id_column not in df.columns:
            raise KeyError(f"Identifier column '{id_column}' not found in {path}")
        frames.append(df)
    if not frames:
        raise ValueError("No OOF prediction files provided.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=[id_column], keep="last")

    pred_cols_missing = [col for col in target_names if col not in combined.columns]
    if pred_cols_missing:
        raise KeyError(f"Missing prediction columns: {', '.join(pred_cols_missing)}")
    return combined


def _prepare_diversity_inputs(
    paths: Sequence[Path],
    *,
    target_names: Sequence[str],
    id_column: str,
) -> Dict[str, pd.DataFrame]:
    diversity_inputs: Dict[str, pd.DataFrame] = {}
    for path in paths:
        df = pd.read_csv(path)
        base_name = path.parent.name
        if base_name == "version_0":
            # If we are inside a Lightning version directory, use the parent folder instead.
            base_name = path.parents[1].name
        _cols_missing = [col for col in target_names if col not in df.columns]
        if _cols_missing:
            continue
        df = df.drop_duplicates(subset=[id_column])
        diversity_inputs.setdefault(base_name, df)
    return diversity_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess CSIRO biomass learning behaviour.")
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to training CSV.")
    parser.add_argument(
        "--oof-prediction",
        type=Path,
        action="append",
        required=True,
        help="Path(s) to out-of-fold prediction CSV files.",
    )
    parser.add_argument("--id-column", default="sample_id", help="Identifier column name.")
    parser.add_argument(
        "--target-name-column",
        default="target_name",
        help="Column holding target names (long format).",
    )
    parser.add_argument(
        "--target-value-column",
        default="target",
        help="Column holding target values (long format).",
    )
    parser.add_argument(
        "--metadata-column",
        action="append",
        default=[],
        help="Optional metadata columns to use for compression tests (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("competitions/csiro_biomass/outputs/learning_assessment"),
        help="Directory to store the assessment report.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_df = pd.read_csv(args.train_csv)
    target_names = _infer_target_names(
        train_df,
        target_name_column=args.target_name_column,
    )
    target_pivot = _pivot_training_targets(
        train_df,
        id_column=args.id_column,
        target_name_column=args.target_name_column,
        target_value_column=args.target_value_column,
    )
    metadata_features = _load_metadata_features(
        train_df,
        id_column=args.id_column,
        metadata_columns=args.metadata_column,
    )

    oof_paths = [Path(p) for p in args.oof_prediction]
    combined_oof = _load_oof_predictions(
        oof_paths,
        target_names=target_names,
        id_column=args.id_column,
    )

    # Ensure actual targets are present, merging when necessary.
    actual_cols = [f"{name}_actual" for name in target_names]
    if any(col not in combined_oof.columns for col in actual_cols):
        combined_oof = combined_oof.merge(
            target_pivot[[args.id_column, *target_names]],
            on=args.id_column,
            how="left",
            suffixes=("", "_actual"),
        )
        # Recompute actual column list in case they were appended without suffix.
        for name in target_names:
            col = f"{name}_actual"
            if col not in combined_oof.columns and name in combined_oof.columns:
                combined_oof.rename(columns={name: col}, inplace=True)

    diversity_inputs = _prepare_diversity_inputs(
        oof_paths,
        target_names=target_names,
        id_column=args.id_column,
    )

    feature_frame = None
    if not metadata_features.empty:
        feature_frame = metadata_features.set_index(args.id_column).reindex(
            combined_oof[args.id_column]
        )
        feature_frame = feature_frame.fillna(0.0)

    assessment = run_learning_assessment(
        oof_predictions=combined_oof,
        target_names=target_names,
        diversity_inputs=diversity_inputs if len(diversity_inputs) >= 2 else None,
        feature_frame=feature_frame,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "learning_assessment.json"
    with report_path.open("w") as fp:
        json.dump(assessment.to_dict(), fp, indent=2)

    print("=" * 72)
    print("CSIRO BIOMASS LEARNING ASSESSMENT")
    print("=" * 72)
    gen = assessment.generalization
    print(f"Generalization RMSE: {gen.aggregate_rmse:.4f}")
    print(f"Baseline RMSE:       {gen.aggregate_baseline_rmse:.4f}")
    print(f"Gap (model-baseline): {gen.aggregate_gap:.4f}  | Learning={gen.is_learning}")
    if assessment.diversity is not None:
        div = assessment.diversity
        print(f"Diversity mean: {div.mean_diversity:.3f} | Learning={div.is_learning}")
    if assessment.compression is not None:
        comp = assessment.compression
        print(f"Compression ratio: {comp.compression_ratio:.3f} | Learning={comp.is_learning}")
    print(f"\nLearning score: {assessment.learning_score:.2%}")
    print(
        f"Tests passed: {assessment.passed_tests}/{assessment.total_tests} | Genuine learning? "
        f"{assessment.is_genuinely_learning}"
    )
    print(f"\nReport saved to: {report_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
