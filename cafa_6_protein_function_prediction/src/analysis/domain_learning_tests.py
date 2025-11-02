"""Domain-specific diagnostics for CAFA-6 learning behaviour."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .learning_assessment import _binarise, _compute_f1_scores


@dataclass
class DensityGroupMetrics:
    group: str
    sample_count: int
    macro_f1: float
    micro_f1: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ClassBiasResult:
    max_abs_bias: float
    per_class_bias: Dict[str, float]
    top_positive: List[tuple[str, float]]
    top_negative: List[tuple[str, float]]
    is_learning: bool

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["top_positive"] = [(k, float(v)) for k, v in self.top_positive]
        payload["top_negative"] = [(k, float(v)) for k, v in self.top_negative]
        return payload


@dataclass
class DomainLearningReport:
    density_groups: List[DensityGroupMetrics]
    class_bias: Optional[ClassBiasResult]
    passed_tests: int
    total_tests: int
    learning_score: float

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "density_groups": [group.to_dict() for group in self.density_groups],
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "learning_score": self.learning_score,
        }
        if self.class_bias is not None:
            payload["class_bias"] = self.class_bias.to_dict()
        return payload


def _compute_group_metrics(
    mask: np.ndarray,
    predicted_binary: np.ndarray,
    actual_binary: np.ndarray,
    class_names: Sequence[str],
    group_name: str,
) -> DensityGroupMetrics:
    if mask.sum() == 0:
        return DensityGroupMetrics(group=group_name, sample_count=0, macro_f1=0.0, micro_f1=0.0)
    preds = predicted_binary[mask]
    actual = actual_binary[mask]
    _, macro, micro = _compute_f1_scores(preds, actual, class_names)
    return DensityGroupMetrics(group=group_name, sample_count=int(mask.sum()), macro_f1=macro, micro_f1=micro)


def _compute_class_bias(
    probabilities: np.ndarray,
    actual_binary: np.ndarray,
    class_names: Sequence[str],
) -> ClassBiasResult:
    pred_positive_rate = probabilities.mean(axis=0)
    actual_rate = actual_binary.mean(axis=0)
    bias = pred_positive_rate - actual_rate
    bias_lookup = {name: float(value) for name, value in zip(class_names, bias)}
    max_abs = float(np.max(np.abs(bias))) if bias.size else 0.0
    sorted_pairs = sorted(bias_lookup.items(), key=lambda item: item[1], reverse=True)
    top_positive = sorted_pairs[:5]
    top_negative = sorted(sorted_pairs, key=lambda item: item[1])[:5]
    is_learning = max_abs < 0.15
    return ClassBiasResult(
        max_abs_bias=max_abs,
        per_class_bias=bias_lookup,
        top_positive=top_positive,
        top_negative=top_negative,
        is_learning=is_learning,
    )


def run_domain_learning_assessment(
    *,
    oof_predictions: pd.DataFrame,
    class_names: Sequence[str],
    threshold: float = 0.5,
    min_group_size: int = 20,
) -> DomainLearningReport:
    prob_matrix = oof_predictions[class_names].to_numpy(dtype=np.float64)
    target_cols = [f"{name}_actual" for name in class_names]
    actual_matrix = oof_predictions[target_cols].to_numpy(dtype=np.float64)
    actual_binary = (actual_matrix > 0.5).astype(np.int32)

    predicted_binary = _binarise(prob_matrix, threshold)

    annotation_counts = actual_binary.sum(axis=1)
    groups = [
        ("no_annotations", annotation_counts == 0),
        ("sparse", (annotation_counts >= 1) & (annotation_counts <= 3)),
        ("medium", (annotation_counts >= 4) & (annotation_counts <= 10)),
        ("dense", annotation_counts > 10),
    ]

    density_metrics: List[DensityGroupMetrics] = []
    tests_total = 0
    tests_passed = 0

    for group_name, mask in groups:
        metrics = _compute_group_metrics(mask, predicted_binary, actual_binary, class_names, group_name)
        if metrics.sample_count >= min_group_size:
            tests_total += 1
            if metrics.macro_f1 >= 0.05:
                tests_passed += 1
        density_metrics.append(metrics)

    class_bias = None
    if prob_matrix.size:
        class_bias = _compute_class_bias(prob_matrix, actual_binary, class_names)
        tests_total += 1
        if class_bias.is_learning:
            tests_passed += 1

    learning_score = float(tests_passed / tests_total) if tests_total else 0.0
    return DomainLearningReport(
        density_groups=density_metrics,
        class_bias=class_bias,
        passed_tests=tests_passed,
        total_tests=tests_total,
        learning_score=learning_score,
    )
