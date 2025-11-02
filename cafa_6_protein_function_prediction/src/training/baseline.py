"""Train the CAFA 6 Lightning baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from ..config import CONFIG_DIR, load_config
from ..config.training import ExperimentConfig
from ..training.datamodule import ProteinDataModule
from ..training.lightning_module import ProteinLightningModule
from ..utils.cafa_metrics import evaluate_cafa_metrics


COMPETITION_ROOT = Path(__file__).resolve().parents[2]


def _resolve_relative(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (COMPETITION_ROOT / path_obj).resolve()


def _resolve_data_file(base_dir: Path, filename: Optional[str | Path], current_path: Path) -> Path:
    base_dir = base_dir.resolve()
    if filename is None:
        if current_path.parent == base_dir:
            return current_path
        return (base_dir / current_path.name).resolve()
    file_path = Path(filename)
    if file_path.is_absolute():
        return file_path
    return (base_dir / file_path).resolve()


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _apply_config_mapping(cfg: ExperimentConfig, mapping: Dict[str, Any]) -> None:
    data_cfg = mapping.get("data", {}) or {}
    data_dir_path = cfg.sequences_path.parent
    if "data_dir" in data_cfg:
        resolved_dir = _resolve_relative(data_cfg["data_dir"])
        if resolved_dir is not None:
            data_dir_path = resolved_dir
    data_dir_path = data_dir_path.resolve()

    cfg.sequences_path = _resolve_data_file(data_dir_path, data_cfg.get("fasta_file"), cfg.sequences_path)
    cfg.annotations_path = _resolve_data_file(data_dir_path, data_cfg.get("terms_file"), cfg.annotations_path)
    if "go_obo_file" in data_cfg:
        ontology_path = _resolve_relative(data_cfg["go_obo_file"])
        if ontology_path is not None:
            cfg.ontology_path = ontology_path
    if "max_samples" in data_cfg:
        cfg.max_samples = data_cfg["max_samples"]
    if "min_go_terms" in data_cfg:
        cfg.min_go_terms = data_cfg["min_go_terms"]
    if "val_fraction" in data_cfg:
        cfg.val_fraction = data_cfg["val_fraction"]
    if "seed" in data_cfg:
        cfg.seed = data_cfg["seed"]
    if "filter_aspect" in data_cfg:
        cfg.filter_aspect = data_cfg["filter_aspect"]

    model_cfg = mapping.get("model", {}) or {}
    if "embedding_model" in model_cfg and model_cfg["embedding_model"]:
        cfg.model_name = str(model_cfg["embedding_model"])
    if "hidden_dims" in model_cfg and model_cfg["hidden_dims"]:
        cfg.hidden_dims = tuple(int(dim) for dim in model_cfg["hidden_dims"])
    if "architecture" in model_cfg and model_cfg["architecture"]:
        cfg.architecture = str(model_cfg["architecture"])
    if "attention_heads" in model_cfg and model_cfg["attention_heads"] is not None:
        cfg.attention_heads = int(model_cfg["attention_heads"])
    if "dropout" in model_cfg and model_cfg["dropout"] is not None:
        cfg.dropout = float(model_cfg["dropout"])
    if "use_fractal_features" in model_cfg:
        cfg.augmentation.use_fractal_features = bool(model_cfg["use_fractal_features"])
    if "fractal_max_iter" in model_cfg and model_cfg["fractal_max_iter"] is not None:
        cfg.augmentation.fractal_max_iter = int(model_cfg["fractal_max_iter"])

    training_cfg = mapping.get("training", {}) or {}
    if "learning_rate" in training_cfg and training_cfg["learning_rate"] is not None:
        cfg.optimizer.learning_rate = float(training_cfg["learning_rate"])
    if "batch_size" in training_cfg and training_cfg["batch_size"] is not None:
        cfg.batch_size = int(training_cfg["batch_size"])
    if "num_epochs" in training_cfg and training_cfg["num_epochs"] is not None:
        cfg.epochs = int(training_cfg["num_epochs"])
    if "gradient_clip" in training_cfg:
        cfg.optimizer.gradient_clip_val = training_cfg["gradient_clip"]
    if "embedding_batch_size" in training_cfg and training_cfg["embedding_batch_size"] is not None:
        cfg.embedding_batch_size = int(training_cfg["embedding_batch_size"])
    if "num_workers" in training_cfg and training_cfg["num_workers"] is not None:
        cfg.num_workers = int(training_cfg["num_workers"])
    if "device" in training_cfg and training_cfg["device"]:
        device_value = str(training_cfg["device"]).lower()
        if device_value in {"cuda", "gpu"}:
            cfg.accelerator = "gpu"
        elif device_value == "cpu":
            cfg.accelerator = "cpu"

    go_cfg = mapping.get("go_hierarchy", {}) or {}
    if "use_hierarchy" in go_cfg:
        cfg.use_go_hierarchy = bool(go_cfg["use_hierarchy"])

    cache_cfg = mapping.get("cache", {}) or {}
    if "use_embedding_cache" in cache_cfg:
        cfg.use_embedding_cache = bool(cache_cfg["use_embedding_cache"])
    if "cache_dir" in cache_cfg and cache_cfg["cache_dir"]:
        cache_dir = _resolve_relative(cache_cfg["cache_dir"])
        if cache_dir is not None:
            cfg.embedding_cache_dir = cache_dir
    if "use_memory_cache" in cache_cfg:
        cfg.embedding_cache_use_memory = bool(cache_cfg["use_memory_cache"])
    if "use_disk_cache" in cache_cfg:
        cfg.embedding_cache_use_disk = bool(cache_cfg["use_disk_cache"])

    output_cfg = mapping.get("output", {}) or {}
    if "output_dir" in output_cfg and output_cfg["output_dir"]:
        output_path = _resolve_relative(output_cfg["output_dir"])
        if output_path is not None:
            cfg.output_dir = output_path

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Lightning neural baseline for CAFA 6.")
    parser.add_argument("--config", type=str, help="Name of YAML config to load (without extension)")
    parser.add_argument("--config-dir", type=Path, help="Directory containing YAML configs")
    parser.add_argument("--sequences-path", type=Path)
    parser.add_argument("--annotations-path", type=Path)
    parser.add_argument("--ontology-path", type=Path)
    parser.add_argument("--output-dir", type=Path)

    parser.add_argument("--model-name")
    parser.add_argument("--hidden-dims", type=int, nargs="+")
    parser.add_argument("--architecture", choices=["mlp", "attention"])
    parser.add_argument("--attention-heads", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--embedding-batch-size", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--min-go-terms", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-embedding-cache", action="store_true")
    parser.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument("--embedding-cache-dir", type=Path)
    parser.add_argument("--no-cache-memory", action="store_true")
    parser.add_argument("--no-cache-disk", action="store_true")

    parser.add_argument("--use-go-hierarchy", action="store_true")
    parser.add_argument("--filter-aspect")
    parser.add_argument("--use-fractal-features", action="store_true")
    parser.add_argument("--fractal-max-iter", type=int)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--precision")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--limit-val-batches", type=int)

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--scheduler-patience", type=int)
    parser.add_argument("--scheduler-factor", type=float)
    parser.add_argument("--gradient-clip", type=float)

    parser.add_argument("--num-workers", type=int)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = ExperimentConfig()
    config_dir = _resolve_relative(args.config_dir) if args.config_dir else None
    if args.config:
        loaded_cfg = load_config(args.config, config_dir=config_dir)
        _apply_config_mapping(cfg, loaded_cfg.to_dict())

    sequences_path = _resolve_relative(args.sequences_path) if args.sequences_path else None
    annotations_path = _resolve_relative(args.annotations_path) if args.annotations_path else None
    ontology_path = _resolve_relative(args.ontology_path) if args.ontology_path else None
    output_dir = _resolve_relative(args.output_dir) if args.output_dir else None
    if sequences_path is not None:
        cfg.sequences_path = sequences_path
    if annotations_path is not None:
        cfg.annotations_path = annotations_path
    if ontology_path is not None:
        cfg.ontology_path = ontology_path
    if output_dir is not None:
        cfg.output_dir = output_dir

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.hidden_dims is not None:
        cfg.hidden_dims = tuple(args.hidden_dims)
    if args.architecture is not None:
        cfg.architecture = args.architecture
    if args.attention_heads is not None:
        cfg.attention_heads = args.attention_heads
    if args.dropout is not None:
        cfg.dropout = args.dropout

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.embedding_batch_size is not None:
        cfg.embedding_batch_size = args.embedding_batch_size
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.min_go_terms is not None:
        cfg.min_go_terms = args.min_go_terms
    if args.max_samples is not None and args.max_samples > 0:
        cfg.max_samples = args.max_samples
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    if args.use_go_hierarchy:
        cfg.use_go_hierarchy = True
    if args.filter_aspect:
        cfg.filter_aspect = args.filter_aspect
    if args.use_fractal_features:
        cfg.augmentation.use_fractal_features = True
    if args.fractal_max_iter is not None:
        cfg.augmentation.fractal_max_iter = args.fractal_max_iter

    if args.no_embedding_cache:
        cfg.use_embedding_cache = False
    elif args.use_embedding_cache:
        cfg.use_embedding_cache = True
    if args.embedding_cache_dir is not None:
        cache_dir = _resolve_relative(args.embedding_cache_dir)
        if cache_dir is not None:
            cfg.embedding_cache_dir = cache_dir
    if args.no_cache_memory:
        cfg.embedding_cache_use_memory = False
    if args.no_cache_disk:
        cfg.embedding_cache_use_disk = False

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.accelerator is not None:
        cfg.accelerator = args.accelerator
    if args.devices is not None:
        cfg.devices = args.devices
    if args.precision is not None:
        cfg.precision = args.precision
    if args.max_train_steps is not None:
        cfg.max_train_steps = args.max_train_steps
    if args.limit_val_batches is not None:
        cfg.limit_val_batches = args.limit_val_batches

    if args.learning_rate is not None:
        cfg.optimizer.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.optimizer.weight_decay = args.weight_decay
    if args.no_scheduler:
        cfg.optimizer.use_scheduler = False
    if args.scheduler_patience is not None:
        cfg.optimizer.scheduler_patience = args.scheduler_patience
    if args.scheduler_factor is not None:
        cfg.optimizer.scheduler_factor = args.scheduler_factor
    if args.gradient_clip is not None:
        cfg.optimizer.gradient_clip_val = args.gradient_clip

    return cfg


def run_experiment(cfg: ExperimentConfig) -> Path:
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = ProteinDataModule(cfg)
    datamodule.setup(stage="fit")

    if datamodule.embedding_dim is None or datamodule.num_labels is None:
        raise RuntimeError("Failed to infer embedding or label dimensions from data.")

    module = ProteinLightningModule(
        embedding_dim=datamodule.embedding_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        architecture=cfg.architecture,
        attention_heads=cfg.attention_heads,
        class_names=datamodule.classes,
        optimizer_cfg=cfg.optimizer,
        val_accessions=datamodule.val_accessions,
        val_ground_truth=datamodule.val_ground_truth,
        ontology=datamodule.ontology,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.output_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch{epoch:02d}-val{val_fmax:.4f}",
        monitor="val_fmax",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    trainer_kwargs = {
        "max_epochs": cfg.epochs,
        "default_root_dir": str(run_dir),
        "accelerator": cfg.accelerator,
        "devices": cfg.devices,
        "precision": cfg.precision,
        "log_every_n_steps": 25,
        "callbacks": [checkpoint_callback],
    }
    if cfg.optimizer.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = cfg.optimizer.gradient_clip_val
    if cfg.max_train_steps is not None:
        trainer_kwargs["max_steps"] = cfg.max_train_steps
    if cfg.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = cfg.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule)

    _write_json(run_dir / "history.json", module.history)
    _write_json(run_dir / "config.json", cfg.to_dict())

    best_checkpoint = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path

    best_module = ProteinLightningModule.load_from_checkpoint(
        best_checkpoint,
        embedding_dim=datamodule.embedding_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        architecture=cfg.architecture,
        attention_heads=cfg.attention_heads,
        class_names=datamodule.classes,
        optimizer_cfg=cfg.optimizer,
        val_accessions=datamodule.val_accessions,
        val_ground_truth=datamodule.val_ground_truth,
        ontology=datamodule.ontology,
    )
    best_module.eval()
    predictions = trainer.predict(best_module, dataloaders=datamodule.val_dataloader())

    indices = torch.cat([item["indices"] for item in predictions]).cpu().numpy()
    probabilities = torch.cat([item["probabilities"] for item in predictions]).cpu().numpy()
    order = np.argsort(indices)
    probabilities = probabilities[order]
    val_accessions = np.array(datamodule.val_accessions)[order]

    predictions_dict = {
        accession: {
            go_term: float(prob)
            for go_term, prob in zip(datamodule.classes, prob_vector)
        }
        for accession, prob_vector in zip(val_accessions, probabilities)
    }
    metrics = evaluate_cafa_metrics(predictions_dict, datamodule.val_ground_truth, datamodule.ontology)
    _write_json(run_dir / "cafa_metrics.json", metrics)

    converted_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (float, int)):
            converted_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            converted_metrics[key] = value.tolist()
        elif isinstance(value, np.floating):
            converted_metrics[key] = float(value)
        else:
            converted_metrics[key] = value

    summary = {
        "best_checkpoint": best_checkpoint,
        "epochs": cfg.epochs,
        **converted_metrics,
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "class_names.json", datamodule.classes)

    return run_dir
def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_args(argv)
    run_dir = run_experiment(cfg)
    print(f"Run artifacts stored in {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
