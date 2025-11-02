"""Learning diagnostics for the CSIRO biomass models.

The helpers mirror the Hull Tactical learning metrics while adapting to
multi-output regression and the simpler ensemble structure used here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from zlib import compress


TARGET_SUFF = "_actual"


def _ensure_columns(frame: pd.DataFrame, expected: Sequence[str]) -> None:
    missing = [col for col in expected if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")


def _rmse(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - actual) ** 2, axis=0))


def _mae(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - actual), axis=0)


def _flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    return matrix.reshape(-1)


@dataclass
class GeneralizationResult:
    per_target_rmse: Dict[str, float]
    per_target_baseline_rmse: Dict[str, float]
    per_target_gap: Dict[str, float]
    aggregate_rmse: float
    aggregate_baseline_rmse: float
    aggregate_gap: float
    effect_size: float
    p_value: float
    is_learning: bool


def compute_generalization_gap(
    oof_predictions: pd.DataFrame,
    target_names: Sequence[str],
    *,
    n_bootstrap: int = 200,
) -> GeneralizationResult:
    """Compare OOF predictions against a naive baseline for each target.

    Expects prediction frame with columns <target>, <target>_actual.
    """
    pred_cols = list(target_names)
    actual_cols = [f"{name}{TARGET_SUFF}" for name in target_names]
    _ensure_columns(oof_predictions, [*pred_cols, *actual_cols])

    preds = oof_predictions[pred_cols].to_numpy(dtype=np.float64)
    actuals = oof_predictions[actual_cols].to_numpy(dtype=np.float64)

    baseline = actuals.mean(axis=0, keepdims=True)
    baseline_preds = np.repeat(baseline, repeats=len(actuals), axis=0)

    rmse = _rmse(preds, actuals)
    baseline_rmse = _rmse(baseline_preds, actuals)
    gap = rmse - baseline_rmse

    aggregate_rmse = float(np.mean(rmse))
    aggregate_baseline_rmse = float(np.mean(baseline_rmse))
    aggregate_gap = aggregate_rmse - aggregate_baseline_rmse

    # Bootstrap p-value comparing prediction errors vs baseline
    rng = np.random.default_rng(42)
    bootstrap_diffs: List[float] = []
    baseline_diffs: List[float] = []
    errors = np.square(preds - actuals).mean(axis=1)
    baseline_errors = np.square(baseline_preds - actuals).mean(axis=1)

    n_samples = len(errors)
    for _ in range(min(n_bootstrap, n_samples)):
        idx = rng.integers(0, n_samples, size=n_samples)
        bootstrap_diffs.append(float(np.sqrt(errors[idx].mean())))
        baseline_diffs.append(float(np.sqrt(baseline_errors[idx].mean())))

    if len(bootstrap_diffs) > 1 and len(baseline_diffs) > 1:
        _, p_value = stats.ttest_ind(baseline_diffs, bootstrap_diffs, equal_var=False)
        effect_size = float((aggregate_baseline_rmse - aggregate_rmse) / (aggregate_rmse + 1e-8))
    else:
        p_value = 1.0
        effect_size = 0.0

    is_learning = bool(aggregate_rmse < aggregate_baseline_rmse and p_value < 0.05)

    return GeneralizationResult(
        per_target_rmse={name: float(val) for name, val in zip(target_names, rmse)},
        per_target_baseline_rmse={name: float(val) for name, val in zip(target_names, baseline_rmse)},
        per_target_gap={name: float(val) for name, val in zip(target_names, gap)},
        aggregate_rmse=aggregate_rmse,
        aggregate_baseline_rmse=aggregate_baseline_rmse,
        aggregate_gap=aggregate_gap,
        effect_size=float(effect_size),
        p_value=float(p_value),
        is_learning=is_learning,
    )


@dataclass
class DiversityResult:
    mean_diversity: float
    min_diversity: float
    max_correlation: float
    is_learning: bool


def compute_ensemble_diversity(
    prediction_sets: Mapping[str, pd.DataFrame],
    target_names: Sequence[str],
) -> DiversityResult:
    """Measure pairwise diversity across multiple OOF prediction sets."""
    if len(prediction_sets) < 2:
        return DiversityResult(
            mean_diversity=0.0,
            min_diversity=0.0,
            max_correlation=1.0,
            is_learning=False,
        )

    flattened: Dict[str, np.ndarray] = {}
    for name, df in prediction_sets.items():
        _ensure_columns(df, target_names)
        flattened[name] = _flatten_matrix(df[target_names].to_numpy(dtype=np.float64))

    correlations: List[float] = []
    diversities: List[float] = []
    model_names = list(flattened)
    for i, name_a in enumerate(model_names):
        vec_a = flattened[name_a]
        for name_b in model_names[i + 1 :]:
            vec_b = flattened[name_b]
            if len(vec_a) != len(vec_b):
                raise ValueError(f"Prediction length mismatch between '{name_a}' and '{name_b}'")
            corr = float(np.corrcoef(vec_a, vec_b)[0, 1])
            if np.isnan(corr):
                continue
            correlations.append(corr)
            diversities.append(1.0 - abs(corr))

    if not correlations:
        return DiversityResult(
            mean_diversity=0.0,
            min_diversity=0.0,
            max_correlation=1.0,
            is_learning=False,
        )

    mean_diversity = float(np.mean(diversities))
    min_diversity = float(np.min(diversities))
    max_correlation = float(np.max(np.abs(correlations)))

    return DiversityResult(
        mean_diversity=mean_diversity,
        min_diversity=min_diversity,
        max_correlation=max_correlation,
        is_learning=bool(mean_diversity > 0.25),
    )


@dataclass
class CompressionResult:
    prediction_complexity: int
    data_complexity: int
    model_complexity: int
    compression_ratio: float
    mdl_score: float
    is_learning: bool


def compute_compression_test(
    predictions: pd.DataFrame,
    feature_frame: Optional[pd.DataFrame],
    target_names: Sequence[str],
    *,
    model_complexity: int = 1000,
) -> CompressionResult:
    """Compare compressed size of predictions against features."""
    _ensure_columns(predictions, target_names)
    pred_bytes = predictions[target_names].to_numpy(dtype=np.float64).tobytes()
    prediction_complexity = len(compress(pred_bytes))

    if feature_frame is not None and not feature_frame.empty:
        feature_bytes = feature_frame.to_numpy(dtype=np.float64, copy=True).tobytes()
    else:
        feature_bytes = b"\x00"
    data_complexity = len(compress(feature_bytes))
    compression_ratio = prediction_complexity / max(data_complexity, 1)
    mdl_score = (prediction_complexity + model_complexity) / max(data_complexity, 1)

    return CompressionResult(
        prediction_complexity=prediction_complexity,
        data_complexity=data_complexity,
        model_complexity=model_complexity,
        compression_ratio=float(compression_ratio),
        mdl_score=float(mdl_score),
        is_learning=bool(compression_ratio < 0.9),
    )


@dataclass
class LearningAssessmentResult:
    generalization: GeneralizationResult
    diversity: Optional[DiversityResult]
    compression: Optional[CompressionResult]
    learning_score: float
    passed_tests: int
    total_tests: int
    is_genuinely_learning: bool

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "generalization": asdict(self.generalization),
            "learning_score": self.learning_score,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "is_genuinely_learning": self.is_genuinely_learning,
        }
        if self.diversity is not None:
            payload["diversity"] = asdict(self.diversity)
        if self.compression is not None:
            payload["compression"] = asdict(self.compression)
        return payload


def aggregate_learning_score(results: Iterable[bool]) -> float:
    flags = list(results)
    if not flags:
        return 0.0
    return float(np.mean(flags))


def run_learning_assessment(
    *,
    oof_predictions: pd.DataFrame,
    target_names: Sequence[str],
    diversity_inputs: Optional[Mapping[str, pd.DataFrame]] = None,
    feature_frame: Optional[pd.DataFrame] = None,
) -> LearningAssessmentResult:
    generalization = compute_generalization_gap(oof_predictions, target_names)

    diversity = None
    if diversity_inputs:
        diversity = compute_ensemble_diversity(diversity_inputs, target_names)

    compression = None
    if feature_frame is not None:
        compression = compute_compression_test(oof_predictions, feature_frame, target_names)

    boolean_results = [generalization.is_learning]
    if diversity is not None:
        boolean_results.append(diversity.is_learning)
    if compression is not None:
        boolean_results.append(compression.is_learning)

    learning_score = aggregate_learning_score(boolean_results)
    passed_tests = sum(bool(flag) for flag in boolean_results)
    total_tests = len(boolean_results)

    return LearningAssessmentResult(
        generalization=generalization,
        diversity=diversity,
        compression=compression,
        learning_score=learning_score,
        passed_tests=passed_tests,
        total_tests=total_tests,
        is_genuinely_learning=bool(learning_score >= 0.66),
    )
