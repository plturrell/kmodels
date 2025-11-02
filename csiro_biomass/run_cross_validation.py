#!/usr/bin/env python
"""Cross-validation driver that wraps the main training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from competitions.csiro_biomass.src.config.experiment import ExperimentConfig
from competitions.csiro_biomass.src.train import run_cross_validation

LOGGER = logging.getLogger(__name__)


def _parse_int_list(raw: Optional[str], *, default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        return list(default)
    try:
        return [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers, got '{raw}'") from exc


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    data_dir = args.data_dir

    train_csv = args.train_csv or data_dir / "train.csv"
    image_dir = args.image_dir or data_dir
    test_csv = args.test_csv or (data_dir / "test.csv" if args.test_csv_default and (data_dir / "test.csv").exists() else None)
    sample_submission = (
        args.sample_submission
        or (
            data_dir / "sample_submission.csv"
            if args.sample_submission_default and (data_dir / "sample_submission.csv").exists()
            else None
        )
    )

    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found at {train_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found at {image_dir}")

    config = ExperimentConfig(
        train_csv=train_csv,
        image_dir=image_dir,
        output_dir=args.output_dir,
        test_csv=test_csv,
        sample_submission=sample_submission,
        fractal_csv=args.fractal_csv,
        use_metadata=not args.no_metadata,
        image_column=args.image_column,
        id_column=args.id_column,
        target_name_column=args.target_name_column,
        target_value_column=args.target_value_column,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        constraint_tolerance=args.constraint_tolerance,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        save_oof=args.save_oof,
    )

    config.backbone.name = args.model or "efficientnet_b3"
    config.backbone.pretrained = not args.no_pretrained
    if args.dropout is not None:
        config.backbone.dropout = args.dropout

    config.fusion.fusion_type = args.fusion_type
    config.fusion.perceiver_latents = args.perceiver_latents
    config.fusion.perceiver_layers = args.perceiver_layers
    config.fusion.perceiver_heads = args.perceiver_heads
    config.fusion.perceiver_dropout = args.perceiver_dropout
    config.fusion.fusion_dropout = args.fusion_dropout
    config.fusion.use_layernorm = not args.no_tab_layernorm
    config.fusion.tabular_hidden_dims = _parse_int_list(args.tab_hidden_dims, default=[128, 64])
    config.fusion.fusion_hidden_dims = _parse_int_list(args.fusion_hidden_dims, default=[512, 256])

    config.optimizer.learning_rate = args.learning_rate
    config.optimizer.weight_decay = args.weight_decay
    config.optimizer.use_scheduler = not args.no_scheduler
    config.optimizer.scheduler_t_max = args.scheduler_t_max
    config.optimizer.warmup_epochs = args.warmup_epochs
    config.optimizer.ema_decay = None if args.no_ema else args.ema_decay

    config.curriculum.enable = not args.no_curriculum
    if args.curriculum_target:
        config.curriculum.target_column = args.curriculum_target

    config.snapshots.num_snapshots = max(int(args.snapshot_count), 0)

    config.regularization.mixup_alpha = max(float(args.mixup_alpha), 0.0)
    config.regularization.mixup_prob = float(min(max(args.mixup_prob, 0.0), 1.0))

    return config


def summarise_histories(run_dir: Path) -> List[dict]:
    history_path = run_dir / "cv_history.json"
    if not history_path.exists():
        return []
    try:
        histories: List[List[dict]] = json.loads(history_path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Failed to parse %s (%s)", history_path, exc)
        return []

    summaries: List[dict] = []
    for fold_idx, history in enumerate(histories, start=1):
        if not history:
            summaries.append({"fold": fold_idx})
            continue
        best_entry = min(history, key=lambda row: row.get("val_rmse", float("inf")))
        summaries.append(
            {
                "fold": fold_idx,
                "best_epoch": int(best_entry.get("epoch", 0)),
                "best_val_rmse": float(best_entry.get("val_rmse", float("nan"))),
                "best_val_mae": float(best_entry.get("val_mae", float("nan"))),
            }
        )
    return summaries


def collect_prediction_files(fold_dirs: Iterable[Path], filename: Optional[str]) -> List[Path]:
    if not filename:
        return []
    collected: List[Path] = []
    for fold_dir in fold_dirs:
        primary = fold_dir / filename
        if primary.exists():
            collected.append(primary)
            continue
        nested = sorted(fold_dir.glob(f"**/{filename}"))
        if nested:
            collected.append(nested[0])
    return collected


def average_prediction_files(prediction_files: Sequence[Path], output_path: Path) -> Path:
    if not prediction_files:
        raise ValueError("No prediction files supplied for ensembling.")

    frames = [pd.read_csv(path) for path in prediction_files]
    reference_cols = list(frames[0].columns)
    for path, frame in zip(prediction_files[1:], frames[1:]):
        if list(frame.columns) != reference_cols:
            raise ValueError(f"Column mismatch when ensembling predictions; {path} does not match schema.")

    numeric_cols = frames[0].select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        # Attempt to coerce remaining columns except the first identifier.
        numeric_cols = reference_cols[1:]

    if not numeric_cols:
        raise ValueError("No numeric columns found to average in prediction files.")

    # Work on copies so we can coerce numeric data safely.
    coerced_frames: List[pd.DataFrame] = []
    for frame in frames:
        coerced = frame.copy()
        coerced[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        coerced_frames.append(coerced)

    id_columns = [col for col in reference_cols if col not in numeric_cols]
    ensemble = coerced_frames[0][id_columns].copy()
    ensemble[numeric_cols] = 0.0

    weight = 1.0 / len(coerced_frames)
    for frame in coerced_frames:
        ensemble[numeric_cols] += frame[numeric_cols].to_numpy() * weight

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output_path, index=False)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-validation for CSIRO Biomass.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--cv-group-column", type=str, help="Optional grouping column for GroupKFold.")

    parser.add_argument("--data-dir", type=Path, default=Path("csiro_biomass_extract"), help="Root directory for competition files.")
    parser.add_argument("--train-csv", type=Path, help="Explicit path to train.csv; defaults to <data-dir>/train.csv.")
    parser.add_argument("--image-dir", type=Path, help="Directory containing image files; defaults to --data-dir.")
    parser.add_argument("--test-csv", type=Path, help="Optional path to test.csv.")
    parser.add_argument("--sample-submission", type=Path, help="Optional path to sample_submission.csv.")
    parser.add_argument("--fractal-csv", type=Path, help="Optional path to precomputed fractal features.")
    parser.add_argument("--test-csv-default", action="store_true", help="Attempt to use <data-dir>/test.csv when --test-csv is omitted.")
    parser.add_argument(
        "--sample-submission-default",
        action="store_true",
        help="Attempt to use <data-dir>/sample_submission.csv when --sample-submission is omitted.",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cross_validation"), help="Directory for CV outputs.")

    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (baseline mode).")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device.")
    parser.add_argument("--scheduler-t-max", type=int, default=15, help="Cosine scheduler T_max.")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Warmup epochs before cosine decay.")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable learning-rate scheduler.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay value.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA updates.")
    parser.add_argument("--snapshot-count", type=int, default=0, help="Number of checkpoints to average.")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup alpha parameter.")
    parser.add_argument("--mixup-prob", type=float, default=0.0, help="Probability of applying mixup.")

    parser.add_argument("--model", dest="model", help="Backbone architecture name (alias: --backbone).")
    parser.add_argument("--backbone", dest="model", help=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, help="Dropout applied to the backbone head.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights.")

    parser.add_argument("--fusion-type", choices=["mlp", "perceiver"], default="mlp", help="Metadata fusion strategy.")
    parser.add_argument("--perceiver-latents", type=int, default=32, help="Perceiver latent tokens.")
    parser.add_argument("--perceiver-layers", type=int, default=3, help="Perceiver layers.")
    parser.add_argument("--perceiver-heads", type=int, default=4, help="Perceiver attention heads.")
    parser.add_argument("--perceiver-dropout", type=float, default=0.1, help="Perceiver dropout.")
    parser.add_argument("--fusion-dropout", type=float, default=0.25, help="Dropout for fusion head.")
    parser.add_argument("--no-tab-layernorm", action="store_true", help="Disable tabular LayerNorm.")
    parser.add_argument("--tab-hidden-dims", help="Comma-separated hidden sizes for tabular encoder (default: 128,64).")
    parser.add_argument("--fusion-hidden-dims", help="Comma-separated hidden sizes for fusion head (default: 512,256).")

    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum sampler.")
    parser.add_argument("--curriculum-target", type=str, help="Column used to stage curriculum sampling.")

    parser.add_argument("--no-metadata", action="store_true", help="Ignore metadata features.")
    parser.add_argument("--image-column", type=str, help="Override image column name.")
    parser.add_argument("--target-name-column", type=str, default="target_name", help="Target name column.")
    parser.add_argument("--target-value-column", type=str, default="target", help="Target value column.")
    parser.add_argument("--id-column", type=str, default="sample_id", help="Identifier column.")
    parser.add_argument("--constraint-tolerance", type=float, default=0.5, help="Constraint repair tolerance.")
    parser.add_argument("--max-train-samples", type=int, help="Optional cap on training samples.")
    parser.add_argument("--max-val-samples", type=int, help="Optional cap on validation samples.")
    parser.add_argument("--save-oof", action="store_true", help="Persist out-of-fold predictions (if supported).")

    parser.add_argument(
        "--prediction-filename",
        type=str,
        default="submission.csv",
        help="Filename to collect from each fold for ensembling (set empty to skip).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)

    LOGGER.info("Starting %d-fold cross-validation", args.num_folds)
    result = run_cross_validation(
        config,
        n_folds=args.num_folds,
        group_column=args.cv_group_column,
    )

    run_dir = Path(result["run_dir"])
    summaries = summarise_histories(run_dir)
    cv_results = {
        "run_dir": result["run_dir"],
        "num_folds": len(result.get("fold_rmse", [])),
        "rmse_mean": result.get("rmse_mean"),
        "rmse_std": result.get("rmse_std"),
        "fold_rmse": result.get("fold_rmse", []),
        "fold_summaries": summaries,
    }

    results_path = run_dir / "cv_results.json"
    results_path.write_text(json.dumps(cv_results, indent=2))
    LOGGER.info("Cross-validation summary saved to %s", results_path)

    fold_dirs = [Path(path) for path in result.get("fold_dirs", [])]
    prediction_files = collect_prediction_files(fold_dirs, args.prediction_filename)
    if prediction_files:
        ensemble_path = run_dir / "ensemble_predictions.csv"
        try:
            average_prediction_files(prediction_files, ensemble_path)
            LOGGER.info("Ensemble created from %d folds at %s", len(prediction_files), ensemble_path)
        except (ValueError, OSError) as exc:
            LOGGER.warning("Failed to create ensemble: %s", exc)
    else:
        LOGGER.info("No prediction files matched '%s'; skipping ensemble.", args.prediction_filename)

    LOGGER.info(
        "Cross-validation complete: RMSE %.4f ± %.4f",
        cv_results["rmse_mean"],
        cv_results["rmse_std"],
    )
    LOGGER.info("Outputs available under %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
