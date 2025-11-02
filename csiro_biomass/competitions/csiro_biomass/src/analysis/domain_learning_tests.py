"""Domain-specific learning diagnostics for the CSIRO biomass task."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class GroupStabilityResult:
    metric: str
    group_scores: Dict[str, float]
    coefficient_of_variation: float
    min_group_score: float
    is_learning: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class MetadataBiasResult:
    feature: str
    correlations: Dict[str, float]
    max_abs_correlation: float
    is_learning: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DomainLearningReport:
    group_stability: Optional[GroupStabilityResult]
    metadata_bias: Optional[Dict[str, MetadataBiasResult]]
    passed_tests: int
    total_tests: int
    learning_score: float

    def to_dict(self) -> Dict[str, object]:
        payload = {
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "learning_score": self.learning_score,
        }
        if self.group_stability is not None:
            payload["group_stability"] = self.group_stability.to_dict()
        if self.metadata_bias:
            payload["metadata_bias"] = {
                feature: result.to_dict() for feature, result in self.metadata_bias.items()
            }
        return payload


def _rmse_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    target_names: Sequence[str],
) -> Dict[str, float]:
    grouped = df.groupby(group_col)
    scores: Dict[str, float] = {}
    for group_name, frame in grouped:
        residuals = []
        for name in target_names:
            pred_col = name
            actual_col = f"{name}_actual"
            if pred_col in frame and actual_col in frame:
                diff = frame[pred_col].to_numpy(dtype=np.float64) - frame[actual_col].to_numpy(dtype=np.float64)
                residuals.append(diff)
        if not residuals:
            continue
        residual_matrix = np.stack(residuals, axis=1)
        rmse = float(np.sqrt(np.mean(residual_matrix**2)))
        scores[str(group_name)] = rmse
    return scores


def evaluate_group_stability(
    df: pd.DataFrame,
    *,
    group_col: str,
    target_names: Sequence[str],
    min_samples: int = 20,
) -> Optional[GroupStabilityResult]:
    """Measure whether performance is consistent across domain groups (species, states...)."""
    counts = df[group_col].value_counts()
    valid_groups = counts[counts >= min_samples].index.tolist()
    if not valid_groups:
        return None

    filtered = df[df[group_col].isin(valid_groups)].copy()
    scores = _rmse_by_group(filtered, group_col=group_col, target_names=target_names)
    if len(scores) < 2:
        return None

    values = np.array(list(scores.values()), dtype=np.float64)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    cov = float(std_val / mean_val) if mean_val > 1e-8 else float("inf")

    return GroupStabilityResult(
        metric="rmse",
        group_scores={k: float(v) for k, v in scores.items()},
        coefficient_of_variation=cov,
        min_group_score=float(np.min(values)),
        is_learning=bool(cov < 0.35 and mean_val < 15.0),
    )


def evaluate_metadata_bias(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    id_column: str,
    features: Sequence[str],
    target_names: Sequence[str],
) -> Dict[str, MetadataBiasResult]:
    """Check residual correlation with metadata features (e.g. NDVI, height)."""
    merged = df.merge(metadata, on=id_column, how="left")
    residuals = {
        name: merged[name].to_numpy(dtype=np.float64) - merged[f"{name}_actual"].to_numpy(dtype=np.float64)
        for name in target_names
        if name in merged and f"{name}_actual" in merged
    }

    results: Dict[str, MetadataBiasResult] = {}
    for feature in features:
        if feature not in merged.columns:
            continue
        feature_values = merged[feature].to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(feature_values)
        correlations: Dict[str, float] = {}
        for name, resid in residuals.items():
            mask = valid_mask & np.isfinite(resid)
            if mask.sum() < 30:
                continue
            corr = float(np.corrcoef(feature_values[mask], resid[mask])[0, 1])
            correlations[name] = corr

        if correlations:
            max_abs_corr = float(np.max(np.abs(list(correlations.values()))))
            results[feature] = MetadataBiasResult(
                feature=feature,
                correlations=correlations,
                max_abs_correlation=max_abs_corr,
                is_learning=bool(max_abs_corr < 0.2),
            )
    return results


def run_domain_learning_assessment(
    *,
    oof_predictions: pd.DataFrame,
    target_names: Sequence[str],
    id_column: str,
    species_series: Optional[pd.Series] = None,
    state_series: Optional[pd.Series] = None,
    metadata_features: Optional[pd.DataFrame] = None,
    metadata_columns: Optional[Sequence[str]] = None,
) -> DomainLearningReport:
    tests_passed = 0
    tests_total = 0
    group_result: Optional[GroupStabilityResult] = None
    bias_results: Dict[str, MetadataBiasResult] = {}

    # Group stability by species / state.
    if species_series is not None:
        frame = oof_predictions.merge(
            species_series.to_frame().reset_index(),
            left_on=id_column,
            right_on=species_series.index.name or "index",
            how="left",
        )
        frame.rename(columns={species_series.name or species_series.index.name or "index": "species"}, inplace=True)
        result = evaluate_group_stability(
            frame,
            group_col="species",
            target_names=target_names,
        )
        if result is not None:
            group_result = result
            tests_total += 1
            if result.is_learning:
                tests_passed += 1

    if group_result is None and state_series is not None:
        frame = oof_predictions.merge(
            state_series.to_frame().reset_index(),
            left_on=id_column,
            right_on=state_series.index.name or "index",
            how="left",
        )
        frame.rename(columns={state_series.name or state_series.index.name or "index": "state"}, inplace=True)
        result = evaluate_group_stability(
            frame,
            group_col="state",
            target_names=target_names,
        )
        if result is not None:
            group_result = result
            tests_total += 1
            if result.is_learning:
                tests_passed += 1

    # Metadata bias tests.
    if metadata_features is not None and metadata_columns:
        bias_results = evaluate_metadata_bias(
            oof_predictions,
            metadata_features[[id_column, *metadata_columns]],
            id_column=id_column,
            features=metadata_columns,
            target_names=target_names,
        )
        if bias_results:
            tests_total += len(bias_results)
            tests_passed += sum(1 for res in bias_results.values() if res.is_learning)

    learning_score = float(tests_passed / tests_total) if tests_total else 0.0

    return DomainLearningReport(
        group_stability=group_result,
        metadata_bias=bias_results if bias_results else None,
        passed_tests=tests_passed,
        total_tests=tests_total,
        learning_score=learning_score,
    )
