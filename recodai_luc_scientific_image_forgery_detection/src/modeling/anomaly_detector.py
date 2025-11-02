"""Physics/statistics-driven anomaly detector for forgery heatmaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyDetectorConfig:
    contamination: float = 0.1
    random_state: int = 42


class PhysicsAnomalyDetector:
    """Isolation Forest wrapper for physics/statistical features."""

    def __init__(self, config: AnomalyDetectorConfig) -> None:
        self.config = config
        self.model = IsolationForest(
            contamination=config.contamination,
            random_state=config.random_state,
        )
        self.feature_names: List[str] = []

    def _to_matrix(self, features: Iterable[dict]) -> np.ndarray:
        features = list(features)
        if not features:
            raise ValueError("No features provided to anomaly detector")
        if not self.feature_names:
            self.feature_names = sorted(features[0].keys())
        matrix = np.array([[feat[name] for name in self.feature_names] for feat in features], dtype=np.float32)
        return matrix

    def fit(self, features: Iterable[dict]) -> None:
        matrix = self._to_matrix(features)
        self.model.fit(matrix)

    def score_samples(self, features: Iterable[dict]) -> np.ndarray:
        matrix = self._to_matrix(features)
        scores = -self.model.decision_function(matrix)
        return scores

    def predict(self, features: Iterable[dict]) -> np.ndarray:
        scores = self.score_samples(features)
        return (scores > 0).astype(np.float32)


__all__ = ["AnomalyDetectorConfig", "PhysicsAnomalyDetector"]


