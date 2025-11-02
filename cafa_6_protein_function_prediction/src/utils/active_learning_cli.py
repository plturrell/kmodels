"""Command-line interface for active learning sample selection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data import build_samples, load_go_terms_long_format, load_sequences_from_fasta
from ..data.dataset import ProteinSample
from .active_learning import ActiveLearningSelector, create_annotation_report

LOGGER = logging.getLogger(__name__)


def _load_predictions(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Predictions file {path} must contain a JSON object.")
    return {
        str(protein_id): {str(term): float(score) for term, score in terms.items()}
        for protein_id, terms in data.items()
    }


def _align_samples(
    samples: Sequence[ProteinSample],
    predictions: Dict[str, Dict[str, float]],
) -> tuple[List[ProteinSample], List[str]]:
    ordered_samples: List[ProteinSample] = []
    class_names: List[str] = []

    # Determine GO-term order from the first prediction
    for terms in predictions.values():
        class_names = sorted(terms.keys())
        break
    if not class_names:
        raise RuntimeError("No GO terms found in predictions.")

    for sample in samples:
        if sample.accession in predictions:
            ordered_samples.append(sample)
    if not ordered_samples:
        raise RuntimeError("No overlap between provided samples and prediction keys.")
    return ordered_samples, class_names


def _predictions_to_array(
    samples: Sequence[ProteinSample],
    class_names: Sequence[str],
    predictions: Dict[str, Dict[str, float]],
) -> np.ndarray:
    matrix = np.zeros((len(samples), len(class_names)), dtype=np.float32)
    term_to_idx = {term: idx for idx, term in enumerate(class_names)}
    for row_idx, sample in enumerate(samples):
        probs = predictions.get(sample.accession, {})
        for term, score in probs.items():
            col_idx = term_to_idx.get(term)
            if col_idx is not None:
                matrix[row_idx, col_idx] = float(score)
    return matrix


def _load_samples(fasta_path: Path, terms_path: Optional[Path]) -> List[ProteinSample]:
    sequences = load_sequences_from_fasta(fasta_path)
    if terms_path and terms_path.exists():
        annotations = load_go_terms_long_format(terms_path)
    else:
        annotations = {key: [] for key in sequences.keys()}
    samples = build_samples(sequences, annotations)
    if not samples:
        raise RuntimeError("No samples constructed from provided FASTA/annotation files.")
    return samples


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Active learning sample selection helper.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction JSON file(s). Provide multiple files for query-by-committee.",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/raw/cafa-6-protein-function-prediction/Test/test_sequences.fasta"),
        help="FASTA file containing candidate proteins.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional GO-term annotations TSV (if available).",
    )
    parser.add_argument(
        "--strategy",
        choices=["uncertainty", "margin", "entropy", "qbc"],
        default="uncertainty",
        help="Active learning strategy to use.",
    )
    parser.add_argument("--n-samples", type=int, default=50, help="Number of samples to select (default: 50).")
    parser.add_argument("--output", type=Path, default=Path("outputs/active_learning_report.txt"), help="Output report path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s - %(message)s")

    prediction_files = [path.resolve() for path in args.predictions]
    prediction_payloads = [_load_predictions(path) for path in prediction_files]

    samples = _load_samples(args.fasta.resolve(), args.annotations.resolve() if args.annotations else None)
    ordered_samples, class_names = _align_samples(samples, prediction_payloads[0])

    prediction_arrays = [
        _predictions_to_array(ordered_samples, class_names, payload) for payload in prediction_payloads
    ]

    if args.strategy != "qbc" and len(prediction_arrays) > 1:
        LOGGER.warning("Multiple prediction files provided but strategy is not 'qbc'; only the first will be used.")

    selector = ActiveLearningSelector(strategy=args.strategy)
    if args.strategy == "qbc":
        selected_indices, scores = selector.select_samples(
            ordered_samples,
            prediction_arrays[0],
            n_samples=args.n_samples,
            ensemble_predictions=prediction_arrays,
        )
    else:
        selected_indices, scores = selector.select_samples(
            ordered_samples,
            prediction_arrays[0],
            n_samples=args.n_samples,
        )

    selected_samples = [ordered_samples[idx] for idx in selected_indices]
    report_text = create_annotation_report(selected_samples, scores[selected_indices], output_path=str(args.output.resolve()))
    LOGGER.info("Generated active learning report with %d samples.", len(selected_samples))
    LOGGER.debug("Report preview:\n%s", report_text[:500])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
