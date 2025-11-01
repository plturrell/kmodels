"""Utility to stage Hull Tactical data for google-research/tsmixer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


EXPORT_TRAIN = Path("competitions/hull_tactical_market_prediction/research_exports/tsmixer/train.csv")
EXPORT_TEST = Path("competitions/hull_tactical_market_prediction/research_exports/tsmixer/test.csv")


def _copy_with_warning(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}. Run export_tsmixer.py first.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Copied {source} -> {destination}")


def _generate_config_snippet(dataset_name: str, train_file: Path, test_file: Path, feature_columns: Optional[list[str]] = None) -> dict:
    snippet = {
        dataset_name: {
            "data_path": f"dataset/{dataset_name}/{train_file.name}",
            "test_data_path": f"dataset/{dataset_name}/{test_file.name}",
            "pred_len": 1,
            "input_len": 60,
            "feature_columns": feature_columns or ["feature_1", "feature_2"],
            "target_columns": ["forward_returns"],
            "freq": "D",
        }
    }
    return snippet


def setup_dataset(tsmixer_root: Path, dataset_name: str, src_train: Path, src_test: Path) -> dict:
    basic_dir = tsmixer_root / "tsmixer_basic"
    if not basic_dir.exists():
        raise FileNotFoundError(
            f"Expected tsmixer_basic under {tsmixer_root}. Clone google-research/google-research first."
        )

    dataset_dir = basic_dir / "dataset" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    target_train = dataset_dir / "train.csv"
    target_test = dataset_dir / "test.csv"

    _copy_with_warning(src_train, target_train)
    _copy_with_warning(src_test, target_test)

    # Try to infer feature columns from header
    feature_columns: Optional[list[str]] = None
    try:
        import pandas as pd

        header_df = pd.read_csv(target_train, nrows=0)
        columns = list(header_df.columns)
        feature_columns = [col for col in columns if col not in {"forward_returns", "date_id", "date", "asset", "id"}]
    except Exception as exc:  # pragma: no cover - optional dependency fallback
        print(f"Warning: could not infer feature columns ({exc}). Using placeholder names.")

    snippet = _generate_config_snippet(dataset_name, target_train, target_test, feature_columns)
    config_path = dataset_dir / "dataset_config.json"
    config_path.write_text(json.dumps(snippet, indent=2))
    print(f"Config snippet written to {config_path}")
    print("\n=== Add this to tsmixer_basic/configs/datasets.json ===\n")
    print(json.dumps(snippet, indent=2))
    return snippet


def run_tsmixer_training(tsmixer_root: Path, dataset_name: str) -> None:
    script_path = tsmixer_root / "run_tuned_hparam.sh"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find {script_path}")
    cmd = ["sh", str(script_path), dataset_name]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def convert_predictions(tsmixer_root: Path, dataset_name: str, output_dir: Path) -> None:
    predictions_dir = tsmixer_root / "tsmixer_basic" / "outputs" / dataset_name
    if not predictions_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {predictions_dir}")
    parquet_files = sorted(predictions_dir.glob("*.parquet"), key=os.path.getmtime)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet predictions found in {predictions_dir}")
    latest_pred = parquet_files[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{dataset_name}_submission.parquet"
    shutil.copy2(latest_pred, target)
    print(f"Copied latest predictions to {target}. Convert to Kaggle format with convert_submission.py if needed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage Hull Tactical data for TSMixer experiments.")
    parser.add_argument("--tsmixer-root", type=Path, required=True, help="Path to google-research/tsmixer directory.")
    parser.add_argument("--dataset-name", default="hull_tactical")
    parser.add_argument("--source-train", type=Path, default=EXPORT_TRAIN)
    parser.add_argument("--source-test", type=Path, default=EXPORT_TEST)
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--convert-predictions", action="store_true")
    parser.add_argument("--pred-output", type=Path, default=Path("submissions"))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    tsmixer_root = args.tsmixer_root.resolve()
    snippet = setup_dataset(tsmixer_root, args.dataset_name, args.source_train.resolve(), args.source_test.resolve())

    if args.run_training:
        run_tsmixer_training(tsmixer_root, args.dataset_name)

    if args.convert_predictions:
        convert_predictions(tsmixer_root, args.dataset_name, args.pred_output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
