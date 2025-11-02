"""Command-line interface for blending model predictions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from ..data import load_go_terms_long_format
from ..utils.cafa_metrics import evaluate_cafa_metrics
from .ensemble import optimize_ensemble_weights

LOGGER = logging.getLogger(__name__)


def _load_predictions(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Predictions file {path} must contain a JSON object.")
    converted: Dict[str, Dict[str, float]] = {}
    for accession, terms in payload.items():
        converted[str(accession)] = {str(term): float(score) for term, score in terms.items()}
    return converted


def _align_entities(
    prediction_payloads: Sequence[Dict[str, Dict[str, float]]],
) -> tuple[List[str], List[str]]:
    """Return sorted lists of protein IDs and GO terms common to all predictions."""
    protein_ids = set(prediction_payloads[0].keys())
    go_terms = set()
    for payload in prediction_payloads:
        protein_ids &= set(payload.keys())
        for mapping in payload.values():
            go_terms.update(mapping.keys())
    if not protein_ids:
        raise RuntimeError("No overlapping protein IDs across provided prediction files.")
    if not go_terms:
        raise RuntimeError("No GO terms found in prediction files.")
    return sorted(protein_ids), sorted(go_terms)


def _predictions_to_matrix(
    payload: Dict[str, Dict[str, float]],
    protein_ids: Sequence[str],
    go_terms: Sequence[str],
) -> np.ndarray:
    matrix = np.zeros((len(protein_ids), len(go_terms)), dtype=np.float32)
    term_index = {term: idx for idx, term in enumerate(go_terms)}
    for row_idx, accession in enumerate(protein_ids):
        scores = payload.get(accession, {})
        for term, value in scores.items():
            col_idx = term_index.get(term)
            if col_idx is not None:
                matrix[row_idx, col_idx] = float(value)
    return matrix


def _combine_predictions(
    matrices: Sequence[np.ndarray],
    method: str,
    weights: Optional[Sequence[float]] = None,
) -> np.ndarray:
    stacked = np.stack(matrices, axis=0)
    if method == "average":
        return np.mean(stacked, axis=0)
    if method == "max":
        return np.max(stacked, axis=0)
    if method == "weighted":
        if weights is None:
            raise ValueError("Weights are required for the 'weighted' method.")
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape[0] != stacked.shape[0]:
            raise ValueError("Weights must match the number of prediction files.")
        weights = weights / weights.sum()
        weighted = stacked * weights.reshape(-1, 1, 1)
        return np.sum(weighted, axis=0)
    raise ValueError(f"Unsupported combine method: {method}")


def _matrix_to_predictions(
    matrix: np.ndarray,
    protein_ids: Sequence[str],
    go_terms: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for row_idx, accession in enumerate(protein_ids):
        payload[accession] = {
            go_terms[col_idx]: float(matrix[row_idx, col_idx]) for col_idx in range(matrix.shape[1])
        }
    return payload


def _load_ground_truth(annotations_path: Path, protein_ids: Sequence[str]) -> Dict[str, set[str]]:
    annotations = load_go_terms_long_format(annotations_path)
    ground_truth = {}
    for accession in protein_ids:
        terms = annotations.get(accession, [])
        ground_truth[accession] = set(terms)
    return ground_truth


def _matrix_from_ground_truth(
    ground_truth: Dict[str, set[str]],
    protein_ids: Sequence[str],
    go_terms: Sequence[str],
) -> np.ndarray:
    mlb = MultiLabelBinarizer(classes=go_terms)
    mlb.fit([go_terms])
    labels = [ground_truth.get(accession, set()) for accession in protein_ids]
    return mlb.transform(labels).astype(np.float32)


def _optimize_weights(
    matrices: Sequence[np.ndarray],
    ground_truth_matrix: np.ndarray,
    protein_ids: Sequence[str],
    go_terms: Sequence[str],
) -> np.ndarray:
    def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        predictions_dict = _matrix_to_predictions(y_pred, protein_ids, go_terms)
        truth_dict = {
            accession: {go_terms[col_idx] for col_idx, value in enumerate(row) if value > 0.5}
            for accession, row in zip(protein_ids, y_true)
        }
        metrics = evaluate_cafa_metrics(predictions_dict, truth_dict, ontology=None)
        return float(metrics.get("fmax", 0.0))

    return optimize_ensemble_weights(
        predictions_list=[matrix for matrix in matrices],
        ground_truth=ground_truth_matrix,
        metric_fn=metric_fn,
    )


def _evaluate_predictions(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, set[str]],
) -> Dict[str, float]:
    metrics = evaluate_cafa_metrics(predictions, ground_truth, ontology=None)
    return {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in metrics.items()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blend model predictions using ensemble strategies.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction JSON files to combine.",
    )
    parser.add_argument(
        "--method",
        choices=["average", "weighted", "max", "optimize"],
        default="average",
        help="Ensembling method.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Weights for the 'weighted' method. Must match the number of prediction files.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="GO-term annotations TSV (required for 'optimize' or when requesting evaluation).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compute CAFA metrics for the blended predictions (requires ground truth).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ensemble/ensemble_predictions.json"),
        help="Output file for combined predictions.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s - %(message)s")

    prediction_payloads = [_load_predictions(path.resolve()) for path in args.predictions]
    protein_ids, go_terms = _align_entities(prediction_payloads)
    matrices = [_predictions_to_matrix(payload, protein_ids, go_terms) for payload in prediction_payloads]

    result_matrix: np.ndarray
    if args.method == "optimize":
        if args.ground_truth is None:
            parser.error("The 'optimize' method requires --ground-truth.")
        ground_truth_matrix = _matrix_from_ground_truth(
            _load_ground_truth(args.ground_truth.resolve(), protein_ids),
            protein_ids,
            go_terms,
        )
        optimal_weights = _optimize_weights(matrices, ground_truth_matrix, protein_ids, go_terms)
        LOGGER.info("Optimised weights: %s", ", ".join(f"{w:.3f}" for w in optimal_weights))
        result_matrix = _combine_predictions(matrices, "weighted", weights=optimal_weights)
    elif args.method == "weighted":
        if args.weights is None:
            parser.error("--weights must be provided when method='weighted'.")
        result_matrix = _combine_predictions(matrices, "weighted", weights=args.weights)
    else:
        result_matrix = _combine_predictions(matrices, args.method)

    combined_predictions = _matrix_to_predictions(result_matrix, protein_ids, go_terms)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(combined_predictions, indent=2), encoding="utf-8")
    LOGGER.info("Saved blended predictions to %s", output_path)

    if args.evaluate:
        if args.ground_truth is None:
            parser.error("--evaluate requires --ground-truth to be provided.")
        ground_truth = _load_ground_truth(args.ground_truth.resolve(), protein_ids)
        metrics = _evaluate_predictions(combined_predictions, ground_truth)
        metrics_path = output_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        LOGGER.info("Saved ensemble evaluation metrics to %s", metrics_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
