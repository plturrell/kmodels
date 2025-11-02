"""Utility helpers to merge model predictions with physics-based scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class EnsembleConfig:
    weight_model: float = 0.7
    weight_physics: float = 0.3
    anomaly_threshold: float = 0.5


class ExplainableEnsemble:
    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config

    def fuse(
        self,
        class_logits: torch.Tensor,
        anomaly_scores: torch.Tensor,
        physics_breakdown: Dict[str, float],
    ) -> Dict[str, object]:
        if class_logits.dim() == 2:
            class_logits = class_logits.squeeze(0)
        probs = torch.softmax(class_logits, dim=0).cpu().numpy()
        anomaly_value = float(anomaly_scores.squeeze().item())
        weighted_score = (
            self.config.weight_model * probs
            + self.config.weight_physics * np.array([1 - anomaly_value, anomaly_value])
        )
        prediction = int(np.argmax(weighted_score))

        explanations: List[str] = []
        if anomaly_scores.item() > self.config.anomaly_threshold:
            explanations.append("Physics anomaly detected")
        sorted_features = sorted(physics_breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        for name, value in sorted_features:
            explanations.append(f"{name}={value:.3f}")

        return {
            "probabilities": weighted_score,
            "prediction": prediction,
            "explanations": explanations,
        }


__all__ = ["EnsembleConfig", "ExplainableEnsemble"]


