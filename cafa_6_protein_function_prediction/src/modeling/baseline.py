"""Baseline multi-label classifier for the CAFA 6 competition.

This script demonstrates a minimal sequence → feature → classifier pipeline that
can be used as a sanity check once the official Kaggle data is available.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from ..data import (
    DEFAULT_DATA_ROOT,
    build_samples,
    load_go_terms_long_format,
    load_sequences_from_fasta,
)
from ..data.dataset import ProteinSample
from ..features import DEFAULT_MODEL_NAME, build_embedding_table, build_feature_table

LOGGER = logging.getLogger(__name__)


def _prepare_samples(fasta_path: Path, annotation_path: Path) -> Sequence[ProteinSample]:
    sequences = load_sequences_from_fasta(fasta_path)
    annotations = load_go_terms_long_format(annotation_path)
    samples = build_samples(sequences, annotations)
    return samples


def _train_val_split(
    samples: Sequence[ProteinSample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[Sequence[ProteinSample], Sequence[ProteinSample]]:
    if not samples:
        raise ValueError("No samples available for training.")
    indices = np.arange(len(samples))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


def _featurise(
    samples: Sequence[ProteinSample],
    *,
    include_embeddings: bool,
    embedding_kwargs: Optional[dict],
    embedding_table,
) -> np.ndarray:
    frame = build_feature_table(
        samples,
        include_embeddings=include_embeddings,
        embedding_kwargs=embedding_kwargs,
        embedding_table=embedding_table,
    )
    feature_columns = [col for col in frame.columns if col not in {"accession", "label_count"}]
    return frame[feature_columns].to_numpy(dtype=np.float32)


def _labels(samples: Sequence[ProteinSample], mlb: MultiLabelBinarizer) -> np.ndarray:
    return mlb.transform([sample.go_terms for sample in samples])


def run_training(
    *,
    fasta_path: Path,
    annotation_path: Path,
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_iter: int = 400,
    use_embeddings: bool = False,
    embedding_model: str = DEFAULT_MODEL_NAME,
    embedding_batch_size: int = 8,
    embedding_device: Optional[str] = None,
) -> dict[str, float]:
    samples = _prepare_samples(fasta_path, annotation_path)
    mlb = MultiLabelBinarizer()
    mlb.fit([sample.go_terms for sample in samples])
    if len(mlb.classes_) == 0:
        raise ValueError("No GO-term annotations found; cannot train baseline.")

    train_samples, val_samples = _train_val_split(samples, val_fraction=val_fraction, seed=seed)

    embedding_kwargs = None
    embedding_table = None
    if use_embeddings:
        embedding_kwargs = {
            "model_name": embedding_model,
            "batch_size": embedding_batch_size,
            "device": embedding_device,
        }
        embedding_table = build_embedding_table(
            samples,
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            device=embedding_device,
        )

    X_train = _featurise(
        train_samples,
        include_embeddings=use_embeddings,
        embedding_kwargs=embedding_kwargs,
        embedding_table=embedding_table,
    )
    X_val = _featurise(
        val_samples,
        include_embeddings=use_embeddings,
        embedding_kwargs=embedding_kwargs,
        embedding_table=embedding_table,
    )
    y_train = _labels(train_samples, mlb)
    y_val = _labels(val_samples, mlb)

    classifier = OneVsRestClassifier(
        LogisticRegression(max_iter=max_iter, solver="lbfgs"),
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)

    train_pred = classifier.predict(X_train)
    val_pred = classifier.predict(X_val)

    metrics = {
        "train_micro_f1": float(f1_score(y_train, train_pred, average="micro", zero_division=0)),
        "train_macro_f1": float(f1_score(y_train, train_pred, average="macro", zero_division=0)),
        "val_micro_f1": float(f1_score(y_val, val_pred, average="micro", zero_division=0)),
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
        "label_cardinality": float(np.mean([len(sample.go_terms) for sample in samples])),
        "num_labels": int(len(mlb.classes_)),
        "num_samples": int(len(samples)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    dump({"classifier": classifier, "label_binarizer": mlb}, output_dir / "baseline_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight baseline classifier.")
    parser.add_argument(
        "--fasta",
        type=Path,
        default=(DEFAULT_DATA_ROOT / "raw" / "cafa-6-protein-function-prediction" / "Train" / "train_sequences.fasta"),
        help="Path to the training FASTA file.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(DEFAULT_DATA_ROOT / "raw" / "cafa-6-protein-function-prediction" / "Train" / "train_terms.tsv"),
        help="Path to the GO-term annotation file (TSV).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "outputs" / "baseline",
        help="Directory where the trained model and metrics are saved.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the data used for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling the train/validation split.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=400,
        help="Maximum iterations for the logistic regression solver.",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Augment amino acid composition features with transformer embeddings.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL_NAME,
        help="Transformer model checkpoint to use when --use-embeddings is set.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="Batch size for embedding extraction (default: 8).",
    )
    parser.add_argument(
        "--embedding-device",
        help="Torch device string for embeddings (e.g., 'cuda', 'cpu'). Default auto-detects.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    try:
        metrics = run_training(
            fasta_path=args.fasta,
            annotation_path=args.annotations,
            output_dir=args.output_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_iter=args.max_iter,
            use_embeddings=args.use_embeddings,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_device=args.embedding_device,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Validation micro-F1: %.4f", metrics["val_micro_f1"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
