"""Constraint-based repair utilities for biomass predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


TargetDict = Dict[str, float]
MetadataDict = Mapping[str, object]


@dataclass
class ConstraintConfig:
    tolerance: float = 0.5  # grams tolerance for component consistency
    max_total: float = 500.0
    max_daily_change: float = 20.0
    species_bounds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "perennial_ryegrass": {"max_clover_ratio": 0.30},
            "ryegrass_clover": {"max_clover_ratio": 0.65},
            "sub_clover": {"min_clover": 5.0, "max_clover_ratio": 0.80},
        }
    )


class BiomassConstraintProcessor:
    """Apply domain constraints to biomass predictions."""

    COMPONENTS = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g")
    TOTAL_KEY = "Dry_Total_g"

    def __init__(self, config: Optional[ConstraintConfig] = None) -> None:
        self.config = config or ConstraintConfig()

    def _apply_non_negative(self, prediction: TargetDict) -> TargetDict:
        for key in (*self.COMPONENTS, self.TOTAL_KEY):
            if key in prediction:
                prediction[key] = max(0.0, float(prediction[key]))
        return prediction

    def _apply_total_cap(self, prediction: TargetDict) -> TargetDict:
        if self.TOTAL_KEY in prediction:
            prediction[self.TOTAL_KEY] = float(
                min(prediction[self.TOTAL_KEY], self.config.max_total)
            )
        return prediction

    def _apply_composition(self, prediction: TargetDict, confidence: float) -> TargetDict:
        if self.TOTAL_KEY not in prediction:
            return prediction

        components = [prediction.get(component) for component in self.COMPONENTS]
        if any(value is None for value in components):
            return prediction

        component_sum = float(sum(components))  # type: ignore[arg-type]
        total_value = float(prediction[self.TOTAL_KEY])
        if abs(total_value - component_sum) <= self.config.tolerance:
            return prediction

        # Confidence in total vs components dictates repair direction
        if confidence >= 0.7 and component_sum > 0:
            scale = total_value / component_sum
            for component in self.COMPONENTS:
                prediction[component] = float(prediction[component]) * scale  # type: ignore[assignment]
        else:
            prediction[self.TOTAL_KEY] = component_sum
        return prediction

    def _apply_species_bounds(
        self, prediction: TargetDict, metadata: MetadataDict
    ) -> TargetDict:
        species_raw = metadata.get("species")
        if not species_raw or self.TOTAL_KEY not in prediction:
            return prediction

        species_key = str(species_raw).strip().lower().replace(" ", "_")
        bounds = self.config.species_bounds.get(species_key)
        if not bounds:
            return prediction

        total = float(prediction[self.TOTAL_KEY])
        clover_value = prediction.get("Dry_Clover_g")
        if clover_value is not None and "max_clover_ratio" in bounds:
            prediction["Dry_Clover_g"] = float(
                min(clover_value, total * bounds["max_clover_ratio"])
            )

        if clover_value is not None and "min_clover" in bounds:
            prediction["Dry_Clover_g"] = float(
                max(prediction["Dry_Clover_g"], bounds["min_clover"])
            )
        return prediction

    def _apply_temporal_smoothness(
        self, prediction: TargetDict, metadata: MetadataDict
    ) -> TargetDict:
        prev = metadata.get("previous_biomass")
        days_elapsed = metadata.get("days_elapsed")
        if not prev or days_elapsed is None:
            return prediction
        try:
            previous = dict(prev)  # type: ignore[arg-type]
            days = float(days_elapsed)
        except (TypeError, ValueError):
            return prediction

        delta = self.config.max_daily_change * max(days, 1.0)
        for key in (self.TOTAL_KEY, "Dry_Green_g"):
            if key in prediction and key in previous:
                low = float(previous[key]) - delta
                high = float(previous[key]) + delta
                prediction[key] = float(np.clip(prediction[key], low, high))
        return prediction

    def repair_vector(
        self,
        prediction: TargetDict,
        metadata: Optional[MetadataDict] = None,
        confidence: float = 0.5,
    ) -> TargetDict:
        repaired = dict(prediction)
        meta = metadata or {}

        repaired = self._apply_non_negative(repaired)
        repaired = self._apply_total_cap(repaired)
        repaired = self._apply_composition(repaired, confidence)
        repaired = self._apply_species_bounds(repaired, meta)
        repaired = self._apply_temporal_smoothness(repaired, meta)
        return repaired

    def repair_frame(
        self,
        predictions: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        id_column: str,
        confidence: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return repaired predictions without mutating inputs."""
        if id_column not in predictions.columns:
            raise ValueError(f"{id_column} must be present in predictions.")

        meta_index = metadata.set_index(id_column)
        confidence_index = (
            confidence.set_index(id_column) if confidence is not None else None
        )

        repaired_rows: List[TargetDict] = []
        for _, row in predictions.iterrows():
            identifier = row[id_column]
            meta_row = meta_index.loc[identifier].to_dict() if identifier in meta_index.index else {}
            conf_val = 0.5
            if confidence_index is not None and identifier in confidence_index.index:
                conf_series = confidence_index.loc[identifier]
                if isinstance(conf_series, pd.Series):
                    conf_val = float(conf_series.mean())
                else:
                    conf_val = float(np.mean(conf_series.values))

            repaired = self.repair_vector(
                {k: float(row[k]) for k in predictions.columns if k != id_column},
                metadata=meta_row,
                confidence=conf_val,
            )
            repaired[id_column] = identifier
            repaired_rows.append(repaired)

        return pd.DataFrame(repaired_rows, columns=predictions.columns)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

