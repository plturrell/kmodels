"""Inference helpers for the CAFA 6 Lightning baseline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from ..config.training import AugmentationConfig, ExperimentConfig, OptimizerConfig
from ..data import build_samples, load_sequences_from_fasta
from ..features.embedding_cache import EmbeddingCache
from ..features.embeddings import embed_sequences
from ..features.fractal_features import FractalProteinFeatures, combine_features
from ..training.lightning_module import ProteinLightningModule
from ..utils.submission_generator import create_submission_from_predictions

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceArtifacts:
    config: ExperimentConfig
    class_names: Sequence[str]
    checkpoint_path: Path


def _as_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def _load_optimizer(payload: Mapping[str, object]) -> OptimizerConfig:
    return OptimizerConfig(**payload)


def _load_augmentation(payload: Mapping[str, object]) -> AugmentationConfig:
    return AugmentationConfig(**payload)


def _load_experiment_config(config_path: Path) -> ExperimentConfig:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["sequences_path"] = Path(data["sequences_path"])
    data["annotations_path"] = Path(data["annotations_path"])
    data["ontology_path"] = Path(data["ontology_path"])
    data["output_dir"] = Path(data["output_dir"])
    if data.get("embedding_cache_dir") is not None:
        data["embedding_cache_dir"] = Path(data["embedding_cache_dir"])
    data["hidden_dims"] = tuple(data.get("hidden_dims", ()))
    data["optimizer"] = _load_optimizer(data["optimizer"])
    data["augmentation"] = _load_augmentation(data["augmentation"])
    return ExperimentConfig(**data)


def load_inference_artifacts(run_dir: Path) -> InferenceArtifacts:
    run_dir = run_dir.resolve()
    config_path = run_dir / "config.json"
    class_names_path = run_dir / "class_names.json"
    summary_path = run_dir / "summary.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Missing class_names.json in {run_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {run_dir}")

    config = _load_experiment_config(config_path)
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    checkpoint = summary.get("best_checkpoint") or summary.get("checkpoint")
    if not checkpoint:
        raise ValueError("summary.json does not contain best_checkpoint")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (run_dir / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return InferenceArtifacts(config=config, class_names=class_names, checkpoint_path=checkpoint_path)


def _prepare_embeddings(
    samples,
    *,
    model_name: str,
    embedding_batch_size: int,
    device: Optional[str],
    augmentation: AugmentationConfig,
    cache: Optional[EmbeddingCache],
) -> np.ndarray:
    embeddings = embed_sequences(
        samples,
        model_name=model_name,
        batch_size=embedding_batch_size,
        device=device,
        return_array=True,
        cache=cache,
    )
    if augmentation.use_fractal_features:
        extractor = FractalProteinFeatures(max_iter=augmentation.fractal_max_iter)
        fractal = extractor.extract_batch(embeddings)
        embeddings = combine_features(embeddings, fractal)
    return embeddings.astype(np.float32, copy=False)


def _batched_indices(total: int, batch_size: int) -> Iterable[range]:
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        yield range(start, end)


def predict_sequences(
    artifacts: InferenceArtifacts,
    *,
    fasta_path: Path,
    batch_size: Optional[int] = None,
    embedding_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_embedding_cache: Optional[bool] = None,
    cache_dir: Optional[Path] = None,
    cache_use_memory: Optional[bool] = None,
    cache_use_disk: Optional[bool] = None,
) -> Dict[str, Dict[str, float]]:
    config = artifacts.config
    sequences = load_sequences_from_fasta(fasta_path)
    samples = build_samples(sequences)
    if not samples:
        raise ValueError(f"No sequences found in {fasta_path}")

    cache: Optional[EmbeddingCache] = None
    if use_embedding_cache is None:
        use_embedding_cache = config.use_embedding_cache
    if use_embedding_cache:
        cache = EmbeddingCache(
            cache_dir=cache_dir or config.embedding_cache_dir,
            use_memory_cache=config.embedding_cache_use_memory if cache_use_memory is None else cache_use_memory,
            use_disk_cache=config.embedding_cache_use_disk if cache_use_disk is None else cache_use_disk,
        )

    embeddings = _prepare_embeddings(
        samples,
        model_name=config.model_name,
        embedding_batch_size=embedding_batch_size or config.embedding_batch_size,
        device=device,
        augmentation=config.augmentation,
        cache=cache,
    )

    ckpt = torch.load(artifacts.checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    embedding_dim = int(hparams.get("embedding_dim", embeddings.shape[1]))
    hidden_dims = tuple(hparams.get("hidden_dims", tuple(config.hidden_dims)))
    dropout = float(hparams.get("dropout", config.dropout))

    module = ProteinLightningModule.load_from_checkpoint(
        artifacts.checkpoint_path,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        class_names=list(artifacts.class_names),
        optimizer_cfg=config.optimizer,
        val_accessions=[],
        val_ground_truth={},
        ontology=None,
    )
    module.eval()
    model = module.model.to(device or "cpu")

    predictions: Dict[str, Dict[str, float]] = {}
    tensor_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(tensor_device)

    inference_batch_size = batch_size or config.batch_size
    sample_order = [sample.accession for sample in samples]

    with torch.no_grad():
        for idx_range in _batched_indices(len(samples), inference_batch_size):
            batch_embeddings = torch.from_numpy(embeddings[idx_range.start : idx_range.stop]).to(tensor_device)
            logits = model(batch_embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()
            for accession, prob_vector in zip(sample_order[idx_range.start : idx_range.stop], probs):
                predictions[accession] = {
                    term: float(prob)
                    for term, prob in zip(artifacts.class_names, prob_vector.tolist())
                }

    return predictions


def generate_submission(
    artifacts: InferenceArtifacts,
    *,
    fasta_path: Path,
    output_path: Path,
    min_confidence: float = 0.01,
    max_terms_per_protein: Optional[int] = None,
    batch_size: Optional[int] = None,
    embedding_batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> Path:
    predictions = predict_sequences(
        artifacts,
        fasta_path=fasta_path,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        device=device,
    )
    create_submission_from_predictions(
        predictions,
        output_path,
        min_confidence=min_confidence,
        max_terms_per_protein=max_terms_per_protein,
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CAFA 6 submission from a Lightning run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to a completed Lightning run directory")
    parser.add_argument("--fasta", type=Path, required=True, help="Path to FASTA file with sequences to predict")
    parser.add_argument("--output", type=Path, required=True, help="Submission TSV output path")
    parser.add_argument("--min-confidence", type=float, default=0.01, help="Minimum confidence threshold")
    parser.add_argument("--max-terms", type=int, default=None, help="Maximum GO terms per protein")
    parser.add_argument("--batch-size", type=int, help="Inference batch size")
    parser.add_argument("--embedding-batch-size", type=int, help="Batch size for embedding extraction")
    parser.add_argument("--device", type=str, help="Device for embedding + inference (e.g. cpu, cuda)")
    parser.add_argument("--use-cache", action="store_true", help="Force embedding cache usage")
    parser.add_argument("--no-cache", action="store_true", help="Disable embedding cache usage")
    parser.add_argument("--cache-dir", type=Path, help="Embedding cache directory override")
    parser.add_argument("--cache-memory", action="store_true", help="Force in-memory cache")
    parser.add_argument("--cache-no-memory", action="store_true", help="Disable in-memory cache")
    parser.add_argument("--cache-disk", action="store_true", help="Force disk cache")
    parser.add_argument("--cache-no-disk", action="store_true", help="Disable disk cache")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (default: INFO)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    artifacts = load_inference_artifacts(args.run_dir)
    use_cache_override: Optional[bool] = None
    if args.use_cache and args.no_cache:
        raise ValueError("Cannot specify both --use-cache and --no-cache")
    if args.use_cache:
        use_cache_override = True
    elif args.no_cache:
        use_cache_override = False

    cache_memory_override: Optional[bool] = None
    if args.cache_memory and args.cache_no_memory:
        raise ValueError("Cannot combine --cache-memory and --cache-no-memory")
    if args.cache_memory:
        cache_memory_override = True
    elif args.cache_no_memory:
        cache_memory_override = False

    cache_disk_override: Optional[bool] = None
    if args.cache_disk and args.cache_no_disk:
        raise ValueError("Cannot combine --cache-disk and --cache-no-disk")
    if args.cache_disk:
        cache_disk_override = True
    elif args.cache_no_disk:
        cache_disk_override = False

    predictions = predict_sequences(
        artifacts,
        fasta_path=args.fasta,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        device=args.device,
        use_embedding_cache=use_cache_override,
        cache_dir=_as_path(args.cache_dir),
        cache_use_memory=cache_memory_override,
        cache_use_disk=cache_disk_override,
    )

    create_submission_from_predictions(
        predictions,
        args.output,
        min_confidence=args.min_confidence,
        max_terms_per_protein=args.max_terms,
    )

    LOGGER.info("Submission written to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


