"""Generate submission files for the CAFA 6 competition.

The submission format requires:
- One row per protein-GO term pair
- Columns: protein_id, GO_term, confidence_score
- Tab-separated values
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def create_submission_from_predictions(
    predictions: Mapping[str, Mapping[str, float]],
    output_path: Path,
    *,
    min_confidence: float = 0.01,
    max_terms_per_protein: Optional[int] = None,
) -> None:
    """Create a submission file from prediction dictionary.
    
    Args:
        predictions: Nested dict mapping protein_id -> {go_term: confidence}
        output_path: Path where submission TSV will be written
        min_confidence: Minimum confidence threshold (default: 0.01)
        max_terms_per_protein: Maximum GO terms per protein (default: None = unlimited)
    """
    rows = []
    for protein_id, terms in predictions.items():
        # Filter by minimum confidence
        filtered_terms = {term: conf for term, conf in terms.items() if conf >= min_confidence}
        
        # Sort by confidence descending
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        
        # Limit number of terms if specified
        if max_terms_per_protein:
            sorted_terms = sorted_terms[:max_terms_per_protein]
        
        for go_term, confidence in sorted_terms:
            rows.append({
                "protein_id": protein_id,
                "GO_term": go_term,
                "confidence": confidence,
            })
    
    if not rows:
        raise ValueError("No predictions to write. Check min_confidence threshold.")
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False, header=False)
    LOGGER.info("Wrote %d predictions to %s", len(rows), output_path)


def create_submission_from_classifier(
    classifier,
    label_binarizer,
    protein_ids: Sequence[str],
    features: np.ndarray,
    output_path: Path,
    *,
    min_confidence: float = 0.01,
    max_terms_per_protein: Optional[int] = None,
) -> None:
    """Create submission from sklearn classifier with probability estimates.
    
    Args:
        classifier: Trained sklearn classifier with predict_proba method
        label_binarizer: MultiLabelBinarizer used during training
        protein_ids: List of protein identifiers matching feature rows
        features: Feature matrix (n_samples, n_features)
        output_path: Path where submission TSV will be written
        min_confidence: Minimum confidence threshold
        max_terms_per_protein: Maximum GO terms per protein
    """
    if len(protein_ids) != features.shape[0]:
        raise ValueError(f"Mismatch: {len(protein_ids)} protein IDs but {features.shape[0]} feature rows")
    
    # Get probability predictions
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)
    elif hasattr(classifier, "decision_function"):
        # Convert decision function to probabilities using sigmoid
        scores = classifier.decision_function(features)
        probabilities = 1 / (1 + np.exp(-scores))
    else:
        raise ValueError("Classifier must have predict_proba or decision_function method")
    
    # Build predictions dictionary
    predictions: Dict[str, Dict[str, float]] = {}
    go_terms = label_binarizer.classes_
    
    for idx, protein_id in enumerate(protein_ids):
        predictions[protein_id] = {}
        for term_idx, go_term in enumerate(go_terms):
            confidence = float(probabilities[idx, term_idx])
            if confidence >= min_confidence:
                predictions[protein_id][go_term] = confidence
    
    create_submission_from_predictions(
        predictions,
        output_path,
        min_confidence=min_confidence,
        max_terms_per_protein=max_terms_per_protein,
    )


__all__ = [
    "create_submission_from_predictions",
    "create_submission_from_classifier",
]

