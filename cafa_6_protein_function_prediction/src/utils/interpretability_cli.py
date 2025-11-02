"""Command-line helpers for model interpretability utilities.

Provides two sub-commands:

- ``attention``: generate attention heatmaps for Lightning runs that use the
  attention architecture.
- ``importance``: compute permutation feature importance for the logistic
  baseline model.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..config.training import ExperimentConfig
from ..data import (
    build_samples,
    load_go_terms_long_format,
    load_sequences_from_fasta,
)
from ..data.dataset import ProteinSample
from ..features import DEFAULT_MODEL_NAME, build_embedding_table, build_feature_table
from ..training.datamodule import ProteinDataModule
from ..training.inference import (
    InferenceArtifacts,
    load_inference_artifacts,
)
from ..training.lightning_module import ProteinLightningModule
from .interpretability import AttentionVisualizer, FeatureImportanceAnalyzer

LOGGER = logging.getLogger(__name__)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _select_indices(
    class_probs: np.ndarray,
    top_k: int,
    explicit: Optional[Sequence[int]] = None,
) -> List[int]:
    if explicit:
        return list(dict.fromkeys(explicit))  # Preserve order, drop duplicates
    if class_probs.size == 0:
        return []
    scores = class_probs.mean(axis=1)
    order = np.argsort(scores)
    selected = order[-top_k:]
    return selected.tolist()


def _load_module(artifacts: InferenceArtifacts, device: torch.device) -> ProteinLightningModule:
    ckpt = torch.load(artifacts.checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    cfg: ExperimentConfig = artifacts.config
    embedding_dim = int(hparams.get("embedding_dim", 0)) or None
    hidden_dims = tuple(hparams.get("hidden_dims", tuple(cfg.hidden_dims)))
    dropout = float(hparams.get("dropout", cfg.dropout))
    architecture = str(hparams.get("architecture", cfg.architecture)).lower()
    attention_heads = int(hparams.get("attention_heads", cfg.attention_heads))

    module = ProteinLightningModule.load_from_checkpoint(
        artifacts.checkpoint_path,
        embedding_dim=embedding_dim or 0,
        hidden_dims=hidden_dims,
        dropout=dropout,
        architecture=architecture,
        attention_heads=attention_heads,
        class_names=list(artifacts.class_names),
        optimizer_cfg=cfg.optimizer,
        val_accessions=[],
        val_ground_truth={},
        ontology=None,
    )
    module.eval()
    module.to(device)
    return module


def _gather_probabilities(
    module: ProteinLightningModule,
    datamodule: ProteinDataModule,
    device: torch.device,
) -> np.ndarray:
    """Return probabilities for the validation split."""
    dataloader = datamodule.val_dataloader()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for embeddings, _, _ in dataloader:
            logits = module(embeddings.to(device))
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.append(batch_probs)
    if not probs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def _generate_attention_reports(
    *,
    run_dir: Path,
    accessions: Optional[Sequence[str]],
    top_k: int,
    output_dir: Path,
    device: torch.device,
    save_distribution: bool,
) -> Path:
    artifacts = load_inference_artifacts(run_dir)
    cfg = artifacts.config
    if cfg.architecture.lower() != "attention":
        raise RuntimeError(
            "Attention visualisation requires a run trained with the 'attention' architecture."
        )

    datamodule = ProteinDataModule(cfg)
    datamodule.setup(stage="fit")

    module = _load_module(artifacts, device)

    probabilities = _gather_probabilities(module, datamodule, device)
    accession_to_index = {sample.accession: idx for idx, sample in enumerate(datamodule.val_samples)}
    explicit_indices: Optional[List[int]] = None
    if accessions:
        missing = [acc for acc in accessions if acc not in accession_to_index]
        if missing:
            raise KeyError(f"Accessions not found in validation split: {', '.join(missing)}")
        explicit_indices = [accession_to_index[acc] for acc in accessions]

    selected_indices = _select_indices(probabilities, top_k, explicit_indices)
    if not selected_indices:
        raise RuntimeError("No validation samples available for attention plotting.")

    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = AttentionVisualizer(output_dir=output_dir)

    summaries = []
    val_dataset = datamodule.val_dataset
    assert val_dataset is not None

    for idx in selected_indices:
        embedding_tensor, _, _ = val_dataset[idx]
        embedding_tensor = embedding_tensor.unsqueeze(0).to(device)
        logits, attention = module.extract_attention(embedding_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        attention_matrix = attention.squeeze(0).detach().cpu().numpy()

        sample = datamodule.val_samples[idx]
        accession = sample.accession
        heatmap_path = visualizer.plot_attention_heatmap(attention_matrix, accession, sequence=sample.sequence)
        distribution_path = None
        if save_distribution:
            distribution_path = visualizer.plot_attention_distribution(attention_matrix, accession)

        top_terms = sorted(
            zip(datamodule.classes, probs.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )[:10]
        summaries.append(
            {
                "accession": accession,
                "sequence_length": len(sample.sequence),
                "num_go_terms": len(sample.go_terms),
                "mean_probability": float(np.mean(probs)),
                "top_terms": [{"go_term": term, "probability": float(score)} for term, score in top_terms],
                "heatmap_path": str(heatmap_path),
                "distribution_path": str(distribution_path) if distribution_path else None,
            }
        )

    summary_path = output_dir / "attention_summary.json"
    summary_path.write_text(json.dumps({"run_dir": str(run_dir), "samples": summaries}, indent=2), encoding="utf-8")
    LOGGER.info("Saved attention summary to %s", summary_path)
    return summary_path


@dataclass
class ImportanceArgs:
    model_path: Path
    fasta_path: Path
    terms_path: Path
    output_dir: Path
    use_embeddings: bool
    embedding_model: Optional[str]
    embedding_batch_size: int
    embedding_device: Optional[str]
    max_samples: Optional[int]
    n_repeats: int
    top_k: int


def _load_baseline_artifacts(model_path: Path):
    from joblib import load

    payload = load(model_path)
    if "classifier" not in payload or "label_binarizer" not in payload:
        raise KeyError(f"Baseline model at {model_path} is missing expected keys.")
    return payload["classifier"], payload["label_binarizer"]


def _prepare_baseline_features(
    *,
    fasta_path: Path,
    terms_path: Path,
    use_embeddings: bool,
    embedding_model: Optional[str],
    embedding_batch_size: int,
    embedding_device: Optional[str],
    max_samples: Optional[int],
) -> tuple[np.ndarray, List[str], Sequence[ProteinSample]]:
    sequences = load_sequences_from_fasta(fasta_path)
    annotations = load_go_terms_long_format(terms_path)
    samples = build_samples(sequences, annotations)
    if max_samples:
        samples = samples[:max_samples]

    embedding_kwargs = None
    embedding_table = None
    if use_embeddings:
        model_name = embedding_model or DEFAULT_MODEL_NAME
        embedding_kwargs = {
            "model_name": model_name,
            "batch_size": embedding_batch_size,
            "device": embedding_device,
        }
        embedding_table = build_embedding_table(
            samples,
            model_name=model_name,
            batch_size=embedding_batch_size,
            device=embedding_device,
        )
    feature_table = build_feature_table(
        samples,
        include_embeddings=use_embeddings,
        embedding_kwargs=embedding_kwargs,
        embedding_table=embedding_table,
    )
    feature_columns = [col for col in feature_table.columns if col not in {"accession", "label_count"}]
    if not feature_columns:
        raise RuntimeError("Feature table is empty; cannot compute importance scores.")

    X = feature_table[feature_columns].to_numpy(dtype=np.float32)
    return X, feature_columns, samples


def _compute_feature_importance(args: ImportanceArgs) -> Path:
    classifier, label_binarizer = _load_baseline_artifacts(args.model_path)
    X, feature_names, samples = _prepare_baseline_features(
        fasta_path=args.fasta_path,
        terms_path=args.terms_path,
        use_embeddings=args.use_embeddings,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        max_samples=args.max_samples,
    )
    y = label_binarizer.transform([sample.go_terms for sample in samples])

    analyzer = FeatureImportanceAnalyzer(output_dir=args.output_dir)
    importance = analyzer.analyze_permutation_importance(
        classifier,
        X,
        y,
        feature_names=feature_names,
        n_repeats=args.n_repeats,
    )
    plot_path = analyzer.plot_feature_importance(importance, top_k=args.top_k)

    summary = {
        "model_path": str(args.model_path),
        "fasta": str(args.fasta_path),
        "annotations": str(args.terms_path),
        "n_samples": int(X.shape[0]),
        "top_features": [
            {"feature": name, "importance": float(score)}
            for name, score in list(importance.items())[: args.top_k]
        ],
        "plot_path": str(plot_path),
    }
    output_path = args.output_dir / "feature_importance.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved feature importance summary to %s", output_path)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model interpretability utilities.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    attention = subparsers.add_parser("attention", help="Generate attention heatmaps for Lightning runs.")
    attention.add_argument("--run-dir", type=Path, required=True, help="Path to a Lightning run directory.")
    attention.add_argument("--output-dir", type=Path, help="Directory to store attention artefacts.")
    attention.add_argument(
        "--accessions",
        nargs="+",
        help="Specific protein accessions from the validation split to visualise.",
    )
    attention.add_argument("--top-k", type=int, default=5, help="Number of validation samples to visualise.")
    attention.add_argument(
        "--device",
        default="auto",
        help="Device to run attention extraction on (default: auto-detect).",
    )
    attention.add_argument(
        "--save-distribution",
        action="store_true",
        help="Also save attention distribution plots alongside heatmaps.",
    )

    importance = subparsers.add_parser("importance", help="Permutation feature importance for the baseline model.")
    importance.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/baseline/baseline_model.joblib"),
        help="Path to the trained baseline joblib artifact.",
    )
    importance.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/raw/cafa-6-protein-function-prediction/Train/train_sequences.fasta"),
        help="Path to the FASTA file used for training.",
    )
    importance.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/cafa-6-protein-function-prediction/Train/train_terms.tsv"),
        help="Path to the GO-term annotations TSV.",
    )
    importance.add_argument("--output-dir", type=Path, default=Path("outputs/interpretability"), help="Output directory.")
    importance.add_argument("--use-embeddings", action="store_true", help="Include transformer embeddings in features.")
    importance.add_argument("--embedding-model", type=str, default=None, help="Hugging Face model name for embeddings.")
    importance.add_argument("--embedding-batch-size", type=int, default=8, help="Batch size for embedding extraction.")
    importance.add_argument("--embedding-device", type=str, default=None, help="Device for embedding extraction.")
    importance.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for speed.")
    importance.add_argument("--n-repeats", type=int, default=5, help="Permutation repeats (default: 5).")
    importance.add_argument("--top-k", type=int, default=20, help="Number of top features to include in report.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s - %(message)s")

    if args.mode == "attention":
        run_dir = args.run_dir.resolve()
        output_dir = (args.output_dir or (run_dir / "interpretability")).resolve()
        device = _resolve_device(args.device)
        _generate_attention_reports(
            run_dir=run_dir,
            accessions=args.accessions,
            top_k=args.top_k,
            output_dir=output_dir,
            device=device,
            save_distribution=args.save_distribution,
        )
        return 0

    if args.mode == "importance":
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        imp_args = ImportanceArgs(
            model_path=args.model_path.resolve(),
            fasta_path=args.fasta.resolve(),
            terms_path=args.annotations.resolve(),
            output_dir=output_dir,
            use_embeddings=args.use_embeddings,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            embedding_device=args.embedding_device,
            max_samples=args.max_samples,
            n_repeats=args.n_repeats,
            top_k=args.top_k,
        )
        _compute_feature_importance(imp_args)
        return 0

    parser.error(f"Unsupported mode: {args.mode}")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
