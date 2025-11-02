"""Learning diagnostics for CAFA-6 protein function prediction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from zlib import compress


def _binarise(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(np.int32)


def _compute_f1_scores(
    predicted: np.ndarray,
    actual: np.ndarray,
    class_names: Sequence[str],
) -> tuple[Dict[str, float], float, float]:
    tp = (predicted & actual).sum(axis=0).astype(np.float64)
    fp = (predicted & (1 - actual)).sum(axis=0).astype(np.float64)
    fn = ((1 - predicted) & actual).sum(axis=0).astype(np.float64)

    per_label: Dict[str, float] = {}
    for idx, name in enumerate(class_names):
        denom = 2 * tp[idx] + fp[idx] + fn[idx]
        f1 = 0.0 if denom == 0 else float(2 * tp[idx] / denom)
        per_label[name] = f1

    macro = float(np.mean(list(per_label.values()))) if per_label else 0.0

    tp_total = tp.sum()
    fp_total = fp.sum()
    fn_total = fn.sum()
    denom_micro = 2 * tp_total + fp_total + fn_total
    micro = float(2 * tp_total / denom_micro) if denom_micro > 0 else 0.0
    return per_label, macro, micro


@dataclass
class GeneralizationResult:
    macro_f1: float
    micro_f1: float
    baseline_macro_f1: float
    baseline_micro_f1: float
    coverage: float
    baseline_coverage: float
    per_class_f1: Dict[str, float]
    baseline_per_class_f1: Dict[str, float]
    is_learning: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DiversityResult:
    mean_diversity: float
    min_diversity: float
    max_correlation: float
    is_learning: bool

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CompressionResult:
    prediction_complexity: int
    data_complexity: int
    model_complexity: int
    compression_ratio: float
    mdl_score: float
    is_learning: bool

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


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
            "generalization": self.generalization.to_dict(),
            "learning_score": self.learning_score,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "is_genuinely_learning": self.is_genuinely_learning,
        }
        if self.diversity is not None:
            payload["diversity"] = self.diversity.to_dict()
        if self.compression is not None:
            payload["compression"] = self.compression.to_dict()
        return payload


def compute_generalization_gap(
    oof_predictions: pd.DataFrame,
    class_names: Sequence[str],
    *,
    threshold: float = 0.5,
) -> GeneralizationResult:
    prob_matrix = oof_predictions[class_names].to_numpy(dtype=np.float64)
    target_cols = [f"{name}_actual" for name in class_names]
    missing = [col for col in target_cols if col not in oof_predictions.columns]
    if missing:
        raise KeyError(f"Missing ground-truth columns: {', '.join(missing)}")
    actual_matrix = oof_predictions[target_cols].to_numpy(dtype=np.float64)
    actual_binary = (actual_matrix > 0.5).astype(np.int32)

    predicted_binary = _binarise(prob_matrix, threshold)
    baseline_binary = np.zeros_like(actual_binary, dtype=np.int32)

    per_label_f1, macro_f1, micro_f1 = _compute_f1_scores(predicted_binary, actual_binary, class_names)
    baseline_per_label_f1, baseline_macro_f1, baseline_micro_f1 = _compute_f1_scores(
        baseline_binary, actual_binary, class_names
    )

    coverage = float((predicted_binary.sum(axis=1) > 0).mean())
    baseline_coverage = 0.0

    is_learning = (macro_f1 > baseline_macro_f1 + 0.02) or (micro_f1 > baseline_micro_f1 + 0.02)
    return GeneralizationResult(
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        baseline_macro_f1=baseline_macro_f1,
        baseline_micro_f1=baseline_micro_f1,
        coverage=coverage,
        baseline_coverage=baseline_coverage,
        per_class_f1=per_label_f1,
        baseline_per_class_f1=baseline_per_label_f1,
        is_learning=is_learning,
    )


def compute_ensemble_diversity(
    prediction_sets: Mapping[str, pd.DataFrame],
    class_names: Sequence[str],
) -> DiversityResult:
    if len(prediction_sets) < 2:
        return DiversityResult(mean_diversity=0.0, min_diversity=0.0, max_correlation=1.0, is_learning=False)

    flattened: Dict[str, np.ndarray] = {}
    for name, frame in prediction_sets.items():
        missing = [col for col in class_names if col not in frame.columns]
        if missing:
            raise KeyError(f"Prediction frame '{name}' missing columns: {', '.join(missing)}")
        flattened[name] = frame[class_names].to_numpy(dtype=np.float64).reshape(-1)

    correlations: list[float] = []
    diversities: list[float] = []
    keys = list(flattened)
    for i, key_a in enumerate(keys):
        vec_a = flattened[key_a]
        for key_b in keys[i + 1 :]:
            vec_b = flattened[key_b]
            if vec_a.shape != vec_b.shape:
                raise ValueError(f"Prediction length mismatch between '{key_a}' and '{key_b}'.")
            corr = float(np.corrcoef(vec_a, vec_b)[0, 1])
            if np.isnan(corr):
                continue
            correlations.append(corr)
            diversities.append(1.0 - abs(corr))

    if not correlations:
        return DiversityResult(mean_diversity=0.0, min_diversity=0.0, max_correlation=1.0, is_learning=False)

    mean_diversity = float(np.mean(diversities))
    min_diversity = float(np.min(diversities))
    max_correlation = float(np.max(np.abs(correlations)))
    return DiversityResult(
        mean_diversity=mean_diversity,
        min_diversity=min_diversity,
        max_correlation=max_correlation,
        is_learning=mean_diversity > 0.25,
    )


def compute_compression_test(
    probabilities: pd.DataFrame,
    feature_frame: Optional[pd.DataFrame],
    class_names: Sequence[str],
    *,
    model_complexity: int = 2000,
) -> CompressionResult:
    prob_bytes = probabilities[class_names].to_numpy(dtype=np.float64).tobytes()
    prediction_complexity = len(compress(prob_bytes))

    if feature_frame is not None and not feature_frame.empty:
        feature_bytes = feature_frame.to_numpy(dtype=np.float64).tobytes()
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
        is_learning=compression_ratio < 1.0,
    )


def aggregate_learning_score(results: Iterable[bool]) -> float:
    results = list(results)
    if not results:
        return 0.0
    return float(np.mean(results))


def run_learning_assessment(
    *,
    oof_predictions: pd.DataFrame,
    class_names: Sequence[str],
    threshold: float = 0.5,
    diversity_inputs: Optional[Mapping[str, pd.DataFrame]] = None,
    feature_frame: Optional[pd.DataFrame] = None,
) -> LearningAssessmentResult:
    generalization = compute_generalization_gap(oof_predictions, class_names, threshold=threshold)

    diversity = None
    if diversity_inputs:
        diversity = compute_ensemble_diversity(diversity_inputs, class_names)

    compression = None
    if feature_frame is not None:
        compression = compute_compression_test(oof_predictions, feature_frame, class_names)

    checks = [generalization.is_learning]
    if diversity is not None:
        checks.append(diversity.is_learning)
    if compression is not None:
        checks.append(compression.is_learning)

    learning_score = aggregate_learning_score(checks)
    return LearningAssessmentResult(
        generalization=generalization,
        diversity=diversity,
        compression=compression,
        learning_score=learning_score,
        passed_tests=sum(bool(x) for x in checks),
        total_tests=len(checks),
        is_genuinely_learning=learning_score >= 0.66,
    )
