"""Hyperparameter optimization for the NFL Big Data Bowl 2026 models using Optuna."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
from .config.experiment import (
    DatasetConfig,
    FeatureConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from .train import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize model hyperparameters.")
    parser.add_argument("--data-root", default="competitions/nfl_big_data_bowl_2026_analytics/data/raw")
    parser.add_argument("--bundle-dirname", default="nfl-big-data-bowl-2026-analytics")
    parser.add_argument("--output-dir", default="competitions/nfl_big_data_bowl_2026_analytics/outputs/optimization")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials to run.")
    return parser


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Objective function for Optuna to optimize."""

    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hidden_dims = trial.suggest_categorical("hidden_dims", ["512,256", "1024,512", "512,256,128"])
    architecture = trial.suggest_categorical("architecture", ["mlp", "residual_mlp", "gat"])

    # Create a training config from the suggested hyperparameters
    config = TrainingConfig(
        dataset=DatasetConfig(
            data_root=Path(args.data_root),
            bundle_dirname=args.bundle_dirname,
            target_columns=["target_x", "target_y"],
        ),
        model=ModelConfig(
            architecture=architecture,
            hidden_dims=[int(d) for d in hidden_dims.split(",")],
            dropout=dropout,
        ),
        features=FeatureConfig(),
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        experiment_root=Path(args.output_dir),
        run_name=f"trial_{trial.number}",
        epochs=50,  # Set a higher ceiling for epochs
        batch_size=1024,
        val_fraction=0.15,
        early_stopping_patience=3,  # Enable early stopping
    )

    result = run_experiment(config)
    return result.get("best_val_rmse", float("inf"))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
