"""Baseline modeling utilities for Hull Tactical Market Prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline


LOGGER = logging.getLogger(__name__)

Task = Literal["classification", "regression"]


@dataclass
class BaselineConfig:
    """Configuration for the gradient boosting baseline."""

    task: Task
    n_splits: int = 5
    time_series_cv: bool = True
    random_state: int = 42
    max_depth: int | None = 6
    max_iter: int | None = None
    learning_rate: float = 0.05
    min_samples_leaf: int | None = None


def _build_estimator(config: BaselineConfig) -> Pipeline:
    if config.task == "classification":
        max_iter = config.max_iter if config.max_iter is not None else 250
        model = HistGradientBoostingClassifier(
            max_depth=config.max_depth,
            max_iter=max_iter,
            learning_rate=config.learning_rate,
            l2_regularization=0.0,
            random_state=config.random_state,
            min_samples_leaf=config.min_samples_leaf,
        )
    else:
        max_iter = config.max_iter if config.max_iter is not None else 400
        model = HistGradientBoostingRegressor(
            max_depth=config.max_depth,
            max_iter=max_iter,
            learning_rate=config.learning_rate,
            l2_regularization=0.0,
            random_state=config.random_state,
            min_samples_leaf=config.min_samples_leaf,
        )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _make_cv(
    config: BaselineConfig, y: Sequence[int | float | bool]
) -> KFold | StratifiedKFold | TimeSeriesSplit:
    if config.time_series_cv:
        return TimeSeriesSplit(n_splits=config.n_splits)

    if config.task == "classification":
        return StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.random_state,
        )
    return KFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    config: BaselineConfig,
) -> tuple[Pipeline, dict[str, float], pd.Series]:
    """Fit the baseline model and report simple cross-validation diagnostics."""
    estimator = _build_estimator(config)
    splitter = _make_cv(config, y)

    metrics: dict[str, float] = {}
    scores: list[float] = []
    oof_preds = pd.Series(index=y.index, dtype=float)

    try:
        for train_idx, valid_idx in splitter.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            estimator.fit(X_train, y_train)

            if config.task == "classification":
                model_step = estimator.named_steps.get("model")
                labels: np.ndarray
                preds: np.ndarray
                if model_step is not None and hasattr(estimator, "predict_proba"):
                    probas = estimator.predict_proba(X_valid)
                    if probas.ndim == 2 and probas.shape[1] == 2:
                        preds = probas[:, 1]
                        labels = (preds >= 0.5).astype(int)
                    else:
                        labels = np.argmax(probas, axis=1)
                        preds = labels
                else:
                    labels = estimator.predict(X_valid)
                    preds = labels
                fold_score = accuracy_score(y_valid, labels)
            else:
                preds = estimator.predict(X_valid)
                fold_score = float(np.sqrt(mean_squared_error(y_valid, preds)))

            oof_preds.iloc[valid_idx] = preds
            scores.append(fold_score)

        if scores:
            metric_name = "accuracy" if config.task == "classification" else "rmse"
            metrics[f"{metric_name}_mean"] = float(np.mean(scores))
            metrics[f"{metric_name}_std"] = float(np.std(scores))
    except ValueError as exc:  # pragma: no cover - defensive fallback
        LOGGER.warning("Cross-validation failed: %s", exc)

    estimator.fit(X, y)
    return estimator, metrics, oof_preds
