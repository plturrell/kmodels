"""Compare Perceiver two-phase and TSMixer outputs against a validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def calculate_metrics(pred_path: Path, validation_path: Path, id_column: str, target_column: str) -> Dict[str, float]:
    predictions = _load_frame(pred_path)

    if {f"{target_column}_pred", f"{target_column}_true"}.issubset(predictions.columns):
        frame = predictions[[id_column, f"{target_column}_pred", f"{target_column}_true"]]
    else:
        validation = _load_frame(validation_path)
        if target_column in predictions.columns:
            predictions = predictions.rename(columns={target_column: f"{target_column}_pred"})
        if target_column in validation.columns:
            validation = validation.rename(columns={target_column: f"{target_column}_true"})
        frame = predictions.merge(validation, on=id_column, suffixes=("_pred", "_true"))

    if frame.empty:
        raise ValueError("No overlapping rows between predictions and validation.")

    errors = frame[f"{target_column}_pred"] - frame[f"{target_column}_true"]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    returns = frame[f"{target_column}_pred"].to_numpy()
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252))
    return {
        "rmse": rmse,
        "sharpe": sharpe,
        "count": int(len(frame)),
    }


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Perceiver two-phase + TSMixer predictions.")
    parser.add_argument("--validation", type=Path, required=True, help="Validation file with ground truth.")
    parser.add_argument("--id-column", default="id", help="Join column shared across predictions and validation.")
    parser.add_argument("--target-column", default="forward_returns")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "perceiver-mse-20",
            "perceiver-two-phase-sharpe",
            "tsmixer_submission",
        ],
        help="Models to evaluate. Use NAME=PATH to point to a specific file.",
    )
    parser.add_argument("--prediction-dir", type=Path, default=Path("competitions/hull_tactical_market_prediction/outputs/tabular_nn"))
    parser.add_argument("--tsmixer-pred", type=Path, default=Path("submissions/tsmixer_submission.parquet"))
    return parser


DEFAULT_PATHS = {
    "perceiver-mse-20": Path("competitions/hull_tactical_market_prediction/outputs/tabular_nn/perceiver-mse-20/submission.csv"),
    "perceiver-two-phase-sharpe": Path("competitions/hull_tactical_market_prediction/outputs/tabular_nn/perceiver-two-phase-sharpe/submission.csv"),
}


def resolve_path(model_name: str, args: argparse.Namespace) -> Path:
    if model_name == "tsmixer_submission":
        return args.tsmixer_pred
    return DEFAULT_PATHS.get(model_name, args.prediction_dir / model_name / "submission.csv")


def main() -> None:
    args = build_parser().parse_args()
    validation_path = args.validation.resolve()
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found: {validation_path}")

    results = {}
    for model_entry in args.models:
        if "=" in model_entry:
            model, explicit_path = model_entry.split("=", 1)
            pred_path = Path(explicit_path)
        else:
            model = model_entry
            pred_path = resolve_path(model, args)
        pred_path = pred_path.resolve()
        if not pred_path.exists():
            print(f"Skipping {model}: prediction file not found ({pred_path})")
            continue
        metrics = calculate_metrics(pred_path, validation_path, args.id_column, args.target_column)
        results[model] = metrics

    if not results:
        print("No models evaluated.")
        return

    df = pd.DataFrame(results).T
    print("\n=== Model Comparison ===")
    print(df.round(6))
    print("\nJSON export:\n", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
