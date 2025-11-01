"""CLI to train the TSMixer model for Hull Tactical Market Prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from .config.experiment import (
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from .solvers.tsmixer_solver import TSMixerDataConfig, run_tsmixer_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a TSMixer model for Hull Tactical Market Prediction."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to the training CSV.")
    parser.add_argument("--test-csv", type=Path, help="Optional path to the test CSV.")
    parser.add_argument("--target-column", default="forward_returns", help="Target column in the training CSV.")
    parser.add_argument("--id-column", default="date_id", help="Identifier column present in train/test files.")
    parser.add_argument("--submission-column", default="prediction", help="Column name produced for submissions.")
    parser.add_argument("--sequence-length", type=int, default=60, help="Length of the input sequence for the model.")
    parser.add_argument("--prediction-length", type=int, default=1, help="Length of the prediction sequence.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training and evaluation.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for AdamW.")
    parser.add_argument("--num-blocks", type=int, default=2, help="Number of mixer blocks in the model.")
    parser.add_argument("--ff-dim", type=int, default=128, help="Dimension of the feed-forward network in the mixer layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, help="Override the default output directory.")
    parser.add_argument("--run-name", help="Optional name for the run directory.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    trainer_cfg = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=4,
        val_fraction=0.15,
        time_series_holdout=True,
    )

    feature_cfg = FeatureConfig()

    model_cfg = ModelConfig(
        model_type="tsmixer",
        perceiver_layers=args.num_blocks,  # Re-using for num_blocks
        hidden_dims=[args.ff_dim],
        dropout=args.dropout,
    )

    optimizer_cfg = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    data_cfg = TSMixerDataConfig(
        sequence_length=args.sequence_length,
        prediction_length=args.prediction_length,
    )

    exp_cfg = ExperimentConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        target_column=args.target_column,
        id_column=args.id_column,
        submission_column=args.submission_column,
        seed=args.seed,
        model=model_cfg,
        optimizer=optimizer_cfg,
        trainer=trainer_cfg,
        features=feature_cfg,
    )

    if args.output_dir:
        exp_cfg.output_dir = args.output_dir

    run_tsmixer_experiment(exp_cfg, data_cfg, run_name=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
