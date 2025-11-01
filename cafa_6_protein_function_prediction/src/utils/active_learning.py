"""
Active learning strategies for protein function annotation.

Implements uncertainty sampling and query-by-committee for selecting
proteins that would benefit most from manual annotation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..data import ProteinSample

LOGGER = logging.getLogger(__name__)


class ActiveLearningSelector:
    """Select samples for annotation using active learning strategies."""
    
    def __init__(self, strategy: str = "uncertainty"):
        """Initialize active learning selector.
        
        Args:
            strategy: Selection strategy ('uncertainty', 'margin', 'entropy', 'qbc')
        """
        self.strategy = strategy
    
    def select_samples(
        self,
        samples: Sequence[ProteinSample],
        predictions: np.ndarray,
        n_samples: int = 100,
        ensemble_predictions: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """Select samples for annotation.
        
        Args:
            samples: Pool of unlabeled samples
            predictions: Model predictions (n_samples, n_labels)
            n_samples: Number of samples to select
            ensemble_predictions: List of predictions from ensemble (for QBC)
        
        Returns:
            Tuple of (selected_indices, uncertainty_scores)
        """
        if self.strategy == "uncertainty":
            scores = self._uncertainty_sampling(predictions)
        elif self.strategy == "margin":
            scores = self._margin_sampling(predictions)
        elif self.strategy == "entropy":
            scores = self._entropy_sampling(predictions)
        elif self.strategy == "qbc":
            if ensemble_predictions is None:
                raise ValueError("ensemble_predictions required for QBC strategy")
            scores = self._query_by_committee(ensemble_predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Select top n_samples by score
        selected_indices = np.argsort(scores)[-n_samples:].tolist()
        
        LOGGER.info(f"Selected {len(selected_indices)} samples using {self.strategy} strategy")
        LOGGER.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        return selected_indices, scores
    
    def _uncertainty_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """Uncertainty sampling: select samples with predictions closest to 0.5.
        
        Args:
            predictions: Prediction probabilities (n_samples, n_labels)
        
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        # Distance from 0.5 for each prediction
        uncertainty = 1 - np.abs(predictions - 0.5) * 2
        
        # Average uncertainty across all labels
        scores = uncertainty.mean(axis=1)
        
        return scores
    
    def _margin_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """Margin sampling: select samples with smallest margin between top 2 predictions.
        
        Args:
            predictions: Prediction probabilities (n_samples, n_labels)
        
        Returns:
            Margin scores (higher = smaller margin = more uncertain)
        """
        # Sort predictions for each sample
        sorted_preds = np.sort(predictions, axis=1)
        
        # Margin between top 2 predictions
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]
        
        # Invert so higher score = more uncertain
        scores = 1 - margins
        
        return scores
    
    def _entropy_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """Entropy sampling: select samples with highest prediction entropy.
        
        Args:
            predictions: Prediction probabilities (n_samples, n_labels)
        
        Returns:
            Entropy scores (higher = more uncertain)
        """
        # Clip to avoid log(0)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        
        # Binary entropy for each label
        entropy = -(
            predictions * np.log2(predictions) +
            (1 - predictions) * np.log2(1 - predictions)
        )
        
        # Average entropy across labels
        scores = entropy.mean(axis=1)
        
        return scores
    
    def _query_by_committee(self, ensemble_predictions: List[np.ndarray]) -> np.ndarray:
        """Query by committee: select samples with highest disagreement among models.
        
        Args:
            ensemble_predictions: List of predictions from different models
        
        Returns:
            Disagreement scores (higher = more disagreement)
        """
        # Stack predictions: (n_models, n_samples, n_labels)
        predictions = np.array(ensemble_predictions)
        
        # Calculate variance across models for each sample and label
        variance = np.var(predictions, axis=0)
        
        # Average variance across labels
        scores = variance.mean(axis=1)
        
        return scores


def create_annotation_report(
    selected_samples: Sequence[ProteinSample],
    uncertainty_scores: np.ndarray,
    output_path: Optional[str] = None,
) -> str:
    """Create a report of selected samples for annotation.
    
    Args:
        selected_samples: Samples selected for annotation
        uncertainty_scores: Uncertainty scores for selected samples
        output_path: Optional path to save report
    
    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Active Learning Annotation Report")
    lines.append("=" * 80)
    lines.append(f"\nTotal samples selected: {len(selected_samples)}")
    lines.append(f"Average uncertainty: {uncertainty_scores.mean():.4f}")
    lines.append(f"Uncertainty range: [{uncertainty_scores.min():.4f}, {uncertainty_scores.max():.4f}]")
    lines.append("\nTop 10 samples by uncertainty:")
    lines.append("-" * 80)
    lines.append(f"{'Rank':<6} {'Accession':<15} {'Uncertainty':<12} {'Seq Length':<12}")
    lines.append("-" * 80)
    
    # Sort by uncertainty
    sorted_indices = np.argsort(uncertainty_scores)[::-1]
    
    for rank, idx in enumerate(sorted_indices[:10], 1):
        sample = selected_samples[idx]
        score = uncertainty_scores[idx]
        lines.append(f"{rank:<6} {sample.accession:<15} {score:<12.4f} {len(sample.sequence):<12}")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        LOGGER.info(f"Saved annotation report to {output_path}")
    
    return report


__all__ = [
    "ActiveLearningSelector",
    "create_annotation_report",
]

