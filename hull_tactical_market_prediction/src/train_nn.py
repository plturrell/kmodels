"""CLI to train the neural solver for Hull Tactical Market Prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .config.experiment import (
    CurriculumConfig,
    ExperimentConfig,
    FeatureConfig,
    KalmanConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from .solvers.tabular_solver import run_experiment


def _parse_list(values: Optional[List[str]]) -> Optional[List[int]]:
    if not values:
        return None
    return [int(v) for v in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a neural tabular solver for Hull Tactical Market Prediction."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to the training CSV.")
    parser.add_argument("--test-csv", type=Path, help="Optional path to the test CSV.")
    parser.add_argument("--sample-submission", type=Path, help="Optional sample submission path.")
    parser.add_argument("--target-column", default="forward_returns", help="Target column in the training CSV.")
    parser.add_argument("--id-column", default="date_id", help="Identifier column present in train/test files.")
    parser.add_argument("--submission-column", default="prediction", help="Column name produced for submissions.")
    parser.add_argument(
        "--drop-column",
        action="append",
        default=[],
        help="Additional columns to drop before feature generation (repeat as needed).",
    )
    parser.add_argument(
        "--lag-step",
        action="append",
        dest="lag_steps",
        type=int,
        help="Lag steps to include in the feature builder (repeat as needed).",
    )
    parser.add_argument(
        "--rolling-window",
        action="append",
        dest="rolling_windows",
        type=int,
        help="Rolling window sizes for statistics (repeat as needed).",
    )
    parser.add_argument(
        "--rolling-stat",
        action="append",
        dest="rolling_stats",
        choices=("mean", "std", "min", "max"),
        help="Rolling statistics to compute.",
    )
    parser.add_argument(
        "--max-nan-ratio",
        type=float,
        default=0.5,
        help="Drop features with missing-value ratio above this threshold.",
    )
    parser.add_argument(
        "--no-drop-constant",
        action="store_true",
        help="Keep features that are constant across the training dataset.",
    )
    parser.add_argument(
        "--model",
        default="mlp",
        choices=("mlp", "perceiver"),
        help="Model architecture to use (default: mlp).",
    )
    parser.add_argument(
        "--hidden-dim",
        action="append",
        dest="hidden_dims",
        type=int,
        help="Hidden layer dimension for the neural network (repeat for multiple layers).",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the neural network.")
    parser.add_argument(
        "--activation",
        default="gelu",
        choices=("relu", "leaky_relu", "elu", "selu", "gelu"),
        help="Activation function for hidden layers.",
    )
    parser.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Disable batch normalisation layers.",
    )
    parser.add_argument(
        "--perceiver-latent-dim",
        type=int,
        default=128,
        help="Latent dimension for the Perceiver model.",
    )
    parser.add_argument(
        "--perceiver-num-latents",
        type=int,
        default=16,
        help="Number of latent vectors for the Perceiver model.",
    )
    parser.add_argument(
        "--perceiver-layers",
        type=int,
        default=6,
        help="Number of Perceiver blocks.",
    )
    parser.add_argument(
        "--perceiver-heads",
        type=int,
        default=4,
        help="Number of attention heads in Perceiver blocks.",
    )
    parser.add_argument(
        "--perceiver-dropout",
        type=float,
        default=0.1,
        help="Dropout applied inside Perceiver blocks.",
    )
    parser.add_argument(
        "--perceiver-ff-mult",
        type=int,
        default=4,
        help="Feed-forward multiplier for Perceiver readout head.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW.")
    parser.add_argument(
        "--loss-mode",
        default="mse",
        choices=("mse", "sharpe", "sharpe_mse", "heteroscedastic"),
        help="Objective to optimize.",
    )
    parser.add_argument(
        "--heteroscedastic-weight",
        type=float,
        default=0.5,
        help="Weight for variance mismatch regularization in heteroscedastic loss (default: 0.5).",
    )
    parser.add_argument(
        "--sharpe-risk-free",
        type=float,
        default=0.0,
        help="Risk-free rate used for Sharpe-style loss.",
    )
    parser.add_argument(
        "--sharpe-lambda",
        type=float,
        default=1.0,
        help="Penalty weight for Sharpe loss variance regularisation.",
    )
    parser.add_argument(
        "--loss-mse-weight",
        type=float,
        default=0.5,
        help="Blend factor when using sharpe_mse objective (0-1).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint from a previous run (expects a model_state.pt file).",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training and evaluation.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.12,
        help="Fraction of the data reserved for validation (used for time-series holdout).",
    )
    parser.add_argument(
        "--time-series-holdout",
        action="store_true",
        help="Use chronological split for validation (default).",
    )
    parser.add_argument(
        "--shuffled-holdout",
        action="store_true",
        help="Use shuffled validation split instead of chronological holdout.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test with minimal settings (first 200 rows, 1 epoch, minimal features).",
    )
    parser.add_argument(
        "--use-kalman-smoothing",
        action="store_true",
        help="Apply Kalman smoothing to denoise target values.",
    )
    parser.add_argument(
        "--kalman-noise-variance",
        type=float,
        default=0.01,
        help="Process noise variance for Kalman smoother (default: 0.01).",
    )
    parser.add_argument(
        "--kalman-observation-noise",
        type=float,
        help="Observation noise variance (default: learned from data).",
    )
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Enable curriculum learning with uncertainty gating.",
    )
    parser.add_argument(
        "--curriculum-start-ratio",
        type=float,
        default=0.1,
        help="Start with easiest fraction of samples (default: 0.1).",
    )
    parser.add_argument(
        "--curriculum-difficulty-metric",
        choices=("volatility", "ensemble_disagreement", "prediction_confidence"),
        default="volatility",
        help="Metric for ranking sample difficulty (default: volatility).",
    )
    parser.add_argument(
        "--curriculum-min-epoch",
        type=int,
        default=0,
        help="Epoch to start introducing harder samples (default: 0).",
    )
    parser.add_argument(
        "--curriculum-max-epoch",
        type=int,
        default=10,
        help="Epoch to reach full difficulty range (default: 10).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the default output directory.")
    parser.add_argument("--run-name", help="Optional name for the run directory.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    time_series_holdout = True
    if args.shuffled_holdout:
        time_series_holdout = False
    elif args.time_series_holdout:
        time_series_holdout = True

    if args.smoke:
        # Smoke test: minimal settings
        epochs = 1
        batch_size = 64
        hidden_dims = [64]
        lag_steps = [1]
        rolling_windows = [5]
        rolling_stats = ["mean"]
        max_nan_ratio = 0.95
        dropout = 0.1
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        hidden_dims = args.hidden_dims if args.hidden_dims else [512, 256, 128]
        lag_steps = args.lag_steps if args.lag_steps else [1, 5, 21]
        rolling_windows = args.rolling_windows if args.rolling_windows else [5, 21]
        rolling_stats = args.rolling_stats if args.rolling_stats else ["mean", "std"]
        max_nan_ratio = args.max_nan_ratio
        dropout = args.dropout

    trainer_cfg = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=4,
        val_fraction=args.val_fraction,
        time_series_holdout=time_series_holdout,
    )

    feature_cfg = FeatureConfig(
        drop_columns=args.drop_column,
        max_nan_ratio=max_nan_ratio,
        drop_constant=not args.no_drop_constant,
        lag_steps=lag_steps,
        rolling_windows=rolling_windows,
        rolling_stats=rolling_stats,
    )

    model_cfg = ModelConfig(
        model_type=args.model,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=args.activation,
        batch_norm=not args.no_batch_norm,
        perceiver_latent_dim=args.perceiver_latent_dim,
        perceiver_num_latents=args.perceiver_num_latents,
        perceiver_layers=args.perceiver_layers,
        perceiver_heads=args.perceiver_heads,
        perceiver_dropout=args.perceiver_dropout,
        perceiver_ff_mult=args.perceiver_ff_mult,
    )

    optimizer_cfg = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_cfg = LossConfig(
        mode=args.loss_mode,
        sharpe_risk_free=args.sharpe_risk_free,
        sharpe_lambda=args.sharpe_lambda,
        mse_weight=args.loss_mse_weight,
        heteroscedastic_weight=args.heteroscedastic_weight,
    )

    kalman_cfg = KalmanConfig(
        enabled=args.use_kalman_smoothing,
        process_noise=args.kalman_noise_variance,
        observation_noise=args.kalman_observation_noise,
    )

    curriculum_cfg = CurriculumConfig(
        enabled=args.use_curriculum,
        start_ratio=args.curriculum_start_ratio,
        difficulty_metric=args.curriculum_difficulty_metric,
        min_difficulty_epoch=args.curriculum_min_epoch,
        max_difficulty_epoch=args.curriculum_max_epoch,
    )

    exp_cfg = ExperimentConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        sample_submission=args.sample_submission,
        target_column=args.target_column,
        id_column=args.id_column,
        submission_column=args.submission_column,
        seed=args.seed,
        max_samples=200 if args.smoke else None,
        model=model_cfg,
        optimizer=optimizer_cfg,
        trainer=trainer_cfg,
        features=feature_cfg,
        loss=loss_cfg,
        kalman=kalman_cfg,
        curriculum=curriculum_cfg,
    )

    if args.output_dir:
        exp_cfg.output_dir = args.output_dir

    run_experiment(exp_cfg, run_name=args.run_name, checkpoint=args.checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
