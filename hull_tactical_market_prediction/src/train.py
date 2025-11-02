"""CLI entrypoint to train a light baseline for Hull Tactical Market Prediction."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import joblib
import numpy as np
import pandas as pd

from .features.build_features import FeatureBuilderConfig, align_features, build_feature_frame
from .modeling.baseline import BaselineConfig, train_baseline


LOGGER = logging.getLogger(__name__)


def infer_task(target: pd.Series) -> Literal["classification", "regression"]:
    unique = target.dropna().unique()
    if target.dtype.kind in "biu" and len(unique) <= 10:
        return "classification"
    if set(unique) <= {0, 1}:
        return "classification"
    return "regression"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a gradient boosting baseline for Hull Tactical Market Prediction."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to the training CSV.")
    parser.add_argument(
        "--test-csv",
        type=Path,
        help="Optional path to the test CSV. If provided, a submission file is produced.",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        help="Optional sample submission to mirror output column order.",
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help="Target column in the training file (default: target).",
    )
    parser.add_argument(
        "--id-column",
        default="id",
        help="Identifier column present in train/test/sample files (default: id).",
    )
    parser.add_argument(
        "--submission-column",
        help="Column name to use in the submission file when --sample-submission is not provided.",
    )
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Extra column(s) to drop before modeling. Repeat as needed.",
    )
    parser.add_argument(
        "--max-nan-ratio",
        type=float,
        default=0.75,
        help=(
            "Drop numeric features whose missing-value ratio exceeds this threshold. "
            "Set to 1 (or higher) to keep every column."
        ),
    )
    parser.add_argument(
        "--keep-constant-features",
        action="store_true",
        help="Retain features with a single unique value (default behaviour drops them).",
    )
    parser.add_argument(
        "--lag-step",
        action="append",
        type=int,
        dest="lag_steps",
        help="Lag steps to append as additional features. Repeat for multiple values.",
    )
    parser.add_argument(
        "--rolling-window",
        action="append",
        type=int,
        dest="rolling_windows",
        help="Rolling window size used to build mean/std features. Repeat for multiple values.",
    )
    parser.add_argument(
        "--no-rolling-features",
        action="store_true",
        help="Disable rolling window statistics.",
    )
    parser.add_argument(
        "--rolling-stat",
        action="append",
        dest="rolling_stats",
        choices=("mean", "std", "min", "max"),
        help="Rolling statistics to compute (default: mean and std). Repeat for multiple values.",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "classification", "regression"),
        default="auto",
        help="Set the prediction task explicitly or infer from the target (default: auto).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5).",
    )
    parser.add_argument(
        "--no-time-series-cv",
        action="store_true",
        help="Use shuffled KFold instead of TimeSeriesSplit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/outputs/baseline"),
        help="Directory to store run artifacts.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional run name. Defaults to a timestamped folder when omitted.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the model and CV splitters (default: 42).",
    )
    parser.add_argument(
        "--gbm-max-depth",
        type=int,
        default=6,
        help="Maximum depth for the hist gradient boosting trees (default: 6).",
    )
    parser.add_argument(
        "--gbm-max-iter",
        type=int,
        help="Number of boosting iterations. Defaults to 250 for classification and 400 for regression.",
    )
    parser.add_argument(
        "--gbm-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for the gradient boosting model (default: 0.05).",
    )
    parser.add_argument(
        "--gbm-min-samples-leaf",
        type=int,
        help="Minimum samples per leaf for the gradient boosting model (default: sklearn default of 20).",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Persist the trained model as model.joblib inside the run directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test with minimal settings (first 200 rows, 1 epoch, minimal features).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity."
    )
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_config: FeatureBuilderConfig,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if test_df is not None and not test_df.empty:
        train_features, test_features = align_features(train_df, test_df, feature_config)
        return train_features, test_features
    return build_feature_frame(train_df, feature_config), None


def _prepare_run_dir(output_dir: Path, run_name: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = output_dir / run_name
    else:
        timestamp = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
        run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    LOGGER.info("Loading training data from %s", args.train_csv)
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv) if args.test_csv else None

    if args.smoke:
        LOGGER.info("Smoke test mode: limiting to first 200 rows")
        train_df = train_df.head(200)
        if test_df is not None:
            test_df = test_df.head(50)

    if args.target_column not in train_df.columns:
        raise ValueError(f"Target column '{args.target_column}' not found in train CSV.")
    if args.id_column not in train_df.columns:
        LOGGER.warning("ID column '%s' missing in train CSV; continuing without it.", args.id_column)

    default_cfg = FeatureBuilderConfig()
    if args.smoke:
        # Smoke test: minimal feature engineering
        lag_steps = (1,)
        rolling_windows = (5,)
        rolling_stats = ("mean",)
        max_nan_ratio = 0.95
    else:
        lag_steps = tuple(args.lag_steps) if args.lag_steps else default_cfg.lag_steps
        max_nan_ratio = args.max_nan_ratio
        if max_nan_ratio is not None and max_nan_ratio >= 1:
            max_nan_ratio = None
        rolling_windows: tuple[int, ...]
        if args.no_rolling_features:
            rolling_windows = ()
        else:
            rolling_windows = (
                tuple(args.rolling_windows) if args.rolling_windows else default_cfg.rolling_windows
            )
        rolling_stats = tuple(args.rolling_stats) if args.rolling_stats else tuple(default_cfg.rolling_stats)

    feature_config = FeatureBuilderConfig(
        drop_columns=[args.target_column, args.id_column, *args.drop_column],
        lag_steps=lag_steps,
        max_nan_ratio=max_nan_ratio,
        drop_constant=not args.keep_constant_features,
        rolling_windows=rolling_windows,
        rolling_stats=rolling_stats,
    )

    train_features, test_features = _prepare_features(
        train_df=train_df,
        test_df=test_df,
        feature_config=feature_config,
    )

    target = train_df[args.target_column]
    task = args.task
    if task == "auto":
        task = infer_task(target)
        LOGGER.info("Inferred task type: %s", task)

    submission_column = args.submission_column or args.target_column

    if args.smoke:
        # Smoke test: minimal model settings
        cv_splits = 2
        gbm_max_iter = 10
        gbm_max_depth = 3
    else:
        cv_splits = args.cv_splits
        gbm_max_iter = args.gbm_max_iter
        gbm_max_depth = args.gbm_max_depth

    baseline_config = BaselineConfig(
        task=task,
        n_splits=cv_splits,
        time_series_cv=not args.no_time_series_cv,
        random_state=args.random_state,
        max_depth=gbm_max_depth,
        max_iter=gbm_max_iter,
        learning_rate=args.gbm_learning_rate,
        min_samples_leaf=args.gbm_min_samples_leaf,
    )

    LOGGER.info("Training baseline model.")
    model, metrics, oof_preds = train_baseline(train_features, target, baseline_config)

    run_dir = _prepare_run_dir(args.output_dir, args.run_name)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    run_config = {
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv) if args.test_csv else None,
        "sample_submission": str(args.sample_submission) if args.sample_submission else None,
        "target_column": args.target_column,
        "id_column": args.id_column,
        "submission_column": submission_column,
        "drop_columns": args.drop_column,
        "lag_steps": list(feature_config.lag_steps),
        "max_nan_ratio": feature_config.max_nan_ratio,
        "drop_constant": feature_config.drop_constant,
        "rolling_windows": list(feature_config.rolling_windows),
        "rolling_stats": list(feature_config.rolling_stats),
        "task": task,
        "cv_splits": args.cv_splits,
        "time_series_cv": not args.no_time_series_cv,
        "random_state": args.random_state,
        "gbm_max_depth": args.gbm_max_depth,
        "gbm_max_iter": args.gbm_max_iter,
        "gbm_learning_rate": args.gbm_learning_rate,
        "gbm_min_samples_leaf": args.gbm_min_samples_leaf,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    if oof_preds is not None and not oof_preds.empty:
        oof_df = pd.DataFrame({f"{args.target_column}_oof": oof_preds, args.id_column: train_df[args.id_column]})
        oof_path = run_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        LOGGER.info("Saved out-of-fold predictions to %s", oof_path)

    if args.save_model:
        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path)
        LOGGER.info("Saved fitted model to %s", model_path)

    if test_df is not None and test_features is not None:
        LOGGER.info("Generating predictions for the test set.")
        if args.id_column not in test_df.columns:
            raise ValueError(f"ID column '{args.id_column}' missing from test CSV.")

        if task == "classification":
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(test_features)
                if probas.ndim == 2 and probas.shape[1] == 2:
                    preds = probas[:, 1]
                else:
                    preds = np.argmax(probas, axis=1)
            else:
                preds = model.predict(test_features)
        else:
            preds = model.predict(test_features)

        submission: pd.DataFrame
        if args.sample_submission:
            submission = pd.read_csv(args.sample_submission)
            target_cols = [col for col in submission.columns if col != args.id_column]
            if len(target_cols) != 1:
                raise ValueError(
                    "Sample submission must contain exactly one target column besides the ID."
                )
            submission[target_cols[0]] = preds
        else:
            submission = pd.DataFrame(
                {
                    args.id_column: test_df[args.id_column],
                    submission_column: preds,
                }
            )

        submission_path = run_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        latest_path = args.output_dir / "latest_submission.csv"
        submission.to_csv(latest_path, index=False)
        LOGGER.info("Saved submission to %s", submission_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
