"""CAFA competition evaluation metrics.

Implements the official CAFA metrics:
- Fmax: Maximum F-score across all thresholds
- Smin: Minimum semantic distance
- Coverage: Fraction of proteins with at least one prediction

References:
- CAFA assessment: https://www.biofunctionprediction.org/cafa/
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def precision_recall_at_threshold(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, Set[str]],
    threshold: float,
) -> Tuple[float, float, int]:
    """Calculate precision and recall at a specific threshold.
    
    Args:
        predictions: Dict mapping protein_id -> {go_term: confidence}
        ground_truth: Dict mapping protein_id -> set of true GO terms
        threshold: Confidence threshold for predictions
    
    Returns:
        Tuple of (precision, recall, num_predictions)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    all_proteins = set(predictions.keys()) | set(ground_truth.keys())
    
    for protein_id in all_proteins:
        # Get predictions above threshold
        pred_terms = set()
        if protein_id in predictions:
            pred_terms = {
                term for term, conf in predictions[protein_id].items()
                if conf >= threshold
            }
        
        # Get ground truth
        true_terms = ground_truth.get(protein_id, set())
        
        # Calculate metrics
        true_positives += len(pred_terms & true_terms)
        false_positives += len(pred_terms - true_terms)
        false_negatives += len(true_terms - pred_terms)
    
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    return precision, recall, true_positives + false_positives


def calculate_fmax(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
) -> Tuple[float, float, float, float]:
    """Calculate Fmax (maximum F-score across all thresholds).
    
    Args:
        predictions: Dict mapping protein_id -> {go_term: confidence}
        ground_truth: Dict mapping protein_id -> set of true GO terms
        thresholds: List of thresholds to evaluate (default: 0.01 to 0.99 in steps of 0.01)
    
    Returns:
        Tuple of (fmax, best_threshold, precision_at_fmax, recall_at_fmax)
    """
    if thresholds is None:
        thresholds = [i / 100.0 for i in range(1, 100)]
    
    best_f1 = 0.0
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for threshold in thresholds:
        precision, recall, num_preds = precision_recall_at_threshold(
            predictions, ground_truth, threshold
        )
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # Update best
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    LOGGER.info(f"Fmax: {best_f1:.4f} at threshold {best_threshold:.2f} (P={best_precision:.4f}, R={best_recall:.4f})")
    
    return best_f1, best_threshold, best_precision, best_recall


def calculate_coverage(
    predictions: Dict[str, Dict[str, float]],
    threshold: float = 0.01,
) -> float:
    """Calculate coverage (fraction of proteins with at least one prediction).
    
    Args:
        predictions: Dict mapping protein_id -> {go_term: confidence}
        threshold: Minimum confidence threshold
    
    Returns:
        Coverage fraction (0.0 to 1.0)
    """
    proteins_with_predictions = 0
    
    for protein_id, terms in predictions.items():
        if any(conf >= threshold for conf in terms.values()):
            proteins_with_predictions += 1
    
    coverage = proteins_with_predictions / len(predictions) if predictions else 0.0
    return coverage


def calculate_semantic_distance(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, Set[str]],
    ontology: Optional[object] = None,
    threshold: float = 0.01,
    ic_calculator: Optional[object] = None,
    use_ic: bool = True,
) -> float:
    """Calculate semantic distance between predictions and ground truth.

    Uses information content-based distance when IC calculator is provided,
    otherwise falls back to Jaccard distance.

    Args:
        predictions: Dict mapping protein_id -> {go_term: confidence}
        ground_truth: Dict mapping protein_id -> set of true GO terms
        ontology: GO ontology object (optional, for hierarchy-aware distance)
        threshold: Confidence threshold for predictions
        ic_calculator: InformationContent object for IC-based distance
        use_ic: Whether to use IC-based distance (if available)

    Returns:
        Average semantic distance
    """
    distances = []

    all_proteins = set(predictions.keys()) | set(ground_truth.keys())

    for protein_id in all_proteins:
        # Get predictions above threshold
        pred_terms = set()
        if protein_id in predictions:
            pred_terms = {
                term for term, conf in predictions[protein_id].items()
                if conf >= threshold
            }

        # Get ground truth
        true_terms = ground_truth.get(protein_id, set())

        if not pred_terms and not true_terms:
            continue

        # Calculate distance
        if use_ic and ic_calculator is not None:
            # IC-based semantic distance (Jiang-Conrath)
            distance = _ic_based_distance(pred_terms, true_terms, ic_calculator)
        else:
            # Fallback: Jaccard distance
            if pred_terms or true_terms:
                intersection = len(pred_terms & true_terms)
                union = len(pred_terms | true_terms)
                jaccard_sim = intersection / union if union > 0 else 0.0
                distance = 1.0 - jaccard_sim
            else:
                distance = 0.0

        distances.append(distance)

    return np.mean(distances) if distances else 1.0


def _ic_based_distance(
    pred_terms: Set[str],
    true_terms: Set[str],
    ic_calculator,
) -> float:
    """Calculate IC-based semantic distance between two term sets.

    Uses best-match average (BMA) approach:
    For each true term, find the most similar predicted term.

    Args:
        pred_terms: Predicted GO terms
        true_terms: True GO terms
        ic_calculator: InformationContent object

    Returns:
        Semantic distance (0 = identical, higher = more different)
    """
    if not pred_terms and not true_terms:
        return 0.0
    if not pred_terms or not true_terms:
        return 1.0

    # Calculate best-match similarities
    similarities = []

    for true_term in true_terms:
        best_sim = 0.0
        for pred_term in pred_terms:
            sim = ic_calculator.semantic_similarity(true_term, pred_term, method="lin")
            best_sim = max(best_sim, sim)
        similarities.append(best_sim)

    # Average similarity
    avg_similarity = np.mean(similarities) if similarities else 0.0

    # Convert to distance
    return 1.0 - avg_similarity


def evaluate_cafa_metrics(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, Set[str]],
    ontology: Optional[object] = None,
    ic_calculator: Optional[object] = None,
) -> Dict[str, float]:
    """Calculate all CAFA metrics.

    Args:
        predictions: Dict mapping protein_id -> {go_term: confidence}
        ground_truth: Dict mapping protein_id -> set of true GO terms
        ontology: GO ontology object (optional)
        ic_calculator: InformationContent object for IC-based metrics (optional)

    Returns:
        Dictionary with all metrics
    """
    LOGGER.info("Calculating CAFA metrics...")

    # Fmax
    fmax, best_threshold, precision, recall = calculate_fmax(predictions, ground_truth)

    # Coverage
    coverage = calculate_coverage(predictions, threshold=0.01)

    # Semantic distance
    use_ic = ic_calculator is not None
    smin = calculate_semantic_distance(
        predictions,
        ground_truth,
        ontology,
        threshold=best_threshold,
        ic_calculator=ic_calculator,
        use_ic=use_ic,
    )

    metrics = {
        'fmax': fmax,
        'best_threshold': best_threshold,
        'precision_at_fmax': precision,
        'recall_at_fmax': recall,
        'coverage': coverage,
        'smin': smin,
        'smin_method': 'ic_based' if use_ic else 'jaccard',
    }

    LOGGER.info(f"CAFA Metrics: Fmax={fmax:.4f}, Coverage={coverage:.4f}, Smin={smin:.4f} ({metrics['smin_method']})")

    return metrics


__all__ = [
    "calculate_fmax",
    "calculate_coverage",
    "calculate_semantic_distance",
    "evaluate_cafa_metrics",
    "precision_recall_at_threshold",
    "_ic_based_distance",
]

