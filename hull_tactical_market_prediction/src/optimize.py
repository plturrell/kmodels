"""Hyperparameter optimization for the baseline model using Optuna."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .features.build_features import FeatureBuilderConfig, build_feature_frame
from .modeling.baseline import BaselineConfig, _make_cv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize the baseline model's hyperparameters.")
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to the training CSV.")
    parser.add_argument("--target-column", default="forward_returns", help="Target column name.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials to run.")
    return parser


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, config: BaselineConfig) -> float:
    """Objective function for Optuna to optimize."""

    # Define the hyperparameter search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 1e-1, log=True),
    }

    model = HistGradientBoostingRegressor(**params, random_state=config.random_state)
    estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    splitter = _make_cv(config, y)
    scores = []

    for train_idx, valid_idx in splitter.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_valid)
        scores.append(mean_squared_error(y_valid, preds, squared=False))

    return sum(scores) / len(scores)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    train_df = pd.read_csv(args.train_csv)
    feature_config = FeatureBuilderConfig()
    train_features = build_feature_frame(train_df, feature_config)
    target = train_df[args.target_column]

    baseline_config = BaselineConfig(task="regression")

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_features, target, baseline_config), n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
