"""Lightning entrypoint for ECG image digitisation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ..config.training import AugmentationConfig, ExperimentConfig, OptimizerConfig
from ..modeling.baseline import BaselineModelConfig, build_baseline_model
from .datamodule import ECGDataModule
from .lightning_module import ECGLightningModule


def _timestamped_run_dir(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_model(
    cfg: ExperimentConfig, signal_length: int, signal_channels: int
) -> tuple[BaselineModelConfig, object]:
    model_cfg = BaselineModelConfig(
        signal_length=signal_length,
        signal_channels=signal_channels,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        use_refiner=cfg.use_refiner,
        refiner_channels=cfg.refiner_channels,
        refiner_layers=cfg.refiner_layers,
    )
    return model_cfg, build_baseline_model(model_cfg)


def run_experiment(config: ExperimentConfig) -> Path:
    pl.seed_everything(config.seed, workers=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _timestamped_run_dir(config.output_dir)

    datamodule = ECGDataModule(config)
    datamodule.setup(stage="fit")
    if not datamodule.signal_length:
        raise RuntimeError("Unable to infer ECG signal length from the training data.")

    signal_length = config.signal_length or datamodule.signal_length
    signal_channels = config.signal_channels or datamodule.signal_channels or 1
    model_cfg, model = _build_model(config, signal_length, signal_channels)

    lightning_module = ECGLightningModule(model=model, optimizer_cfg=config.optimizer)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch{epoch:02d}-val{val_mae:.4f}",
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer_kwargs = {
        "max_epochs": config.epochs,
        "default_root_dir": str(run_dir),
        "accelerator": config.accelerator,
        "devices": config.devices,
        "precision": config.precision,
        "log_every_n_steps": 25,
        "callbacks": [checkpoint_callback],
    }
    if config.optimizer.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = config.optimizer.gradient_clip_val
    if config.max_train_steps is not None:
        trainer_kwargs["max_steps"] = config.max_train_steps
    if config.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = config.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule=datamodule)

    best_checkpoint = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path

    _write_json(run_dir / "history.json", lightning_module.history)
    _write_json(
        run_dir / "config.json",
        {
            **config.to_dict(),
            "signal_length": signal_length,
            "signal_channels": signal_channels,
            "model": model_cfg.__dict__,
            "best_checkpoint": best_checkpoint,
        },
    )

    if lightning_module.history:
        best_entry = min(lightning_module.history, key=lambda entry: entry["val_mae"])
        summary = {
            "best_epoch": int(best_entry["epoch"]),
            "best_val_mae": float(best_entry["val_mae"]),
            "best_val_loss": float(best_entry["val_loss"]),
            "best_checkpoint": best_checkpoint,
        }
    else:
        summary = {"best_checkpoint": best_checkpoint}

    _write_json(run_dir / "summary.json", summary)
    return run_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train the Lightning ECG baseline.")
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--signal-dir", type=Path)
    parser.add_argument("--id-column")
    parser.add_argument("--image-column")
    parser.add_argument("--signal-column")
    parser.add_argument("--lead-column")
    parser.add_argument("--group-column")

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--precision")
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--limit-val-batches", type=int)

    parser.add_argument("--backbone")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-refiner", action="store_true")
    parser.add_argument("--refiner-channels", type=int)
    parser.add_argument("--refiner-layers", type=int)
    parser.add_argument("--signal-length", type=int)
    parser.add_argument("--preload-signals", action="store_true")

    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--scheduler-t-max", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--gradient-clip", type=float)

    parser.add_argument("--image-size", type=int)
    parser.add_argument("--random-rotation", type=float)
    parser.add_argument("--horizontal-flip", type=float)
    parser.add_argument("--vertical-flip", type=float)

    args = parser.parse_args(argv)

    defaults = ExperimentConfig()
    optimizer_cfg = OptimizerConfig(
        learning_rate=args.learning_rate if args.learning_rate is not None else defaults.optimizer.learning_rate,
        weight_decay=args.weight_decay if args.weight_decay is not None else defaults.optimizer.weight_decay,
        use_scheduler=not args.no_scheduler,
        scheduler_t_max=args.scheduler_t_max if args.scheduler_t_max is not None else defaults.optimizer.scheduler_t_max,
        warmup_epochs=args.warmup_epochs if args.warmup_epochs is not None else defaults.optimizer.warmup_epochs,
        gradient_clip_val=args.gradient_clip if args.gradient_clip is not None else defaults.optimizer.gradient_clip_val,
    )

    augmentation_cfg = AugmentationConfig(
        image_size=args.image_size if args.image_size is not None else defaults.augmentation.image_size,
        random_rotation=args.random_rotation if args.random_rotation is not None else defaults.augmentation.random_rotation,
        horizontal_flip=args.horizontal_flip if args.horizontal_flip is not None else defaults.augmentation.horizontal_flip,
        vertical_flip=args.vertical_flip if args.vertical_flip is not None else defaults.augmentation.vertical_flip,
    )

    return ExperimentConfig(
        train_csv=args.train_csv or defaults.train_csv,
        image_dir=args.image_dir or defaults.image_dir,
        signal_dir=args.signal_dir or defaults.signal_dir,
        id_column=args.id_column or defaults.id_column,
        image_column=args.image_column or defaults.image_column,
        signal_column=args.signal_column or defaults.signal_column,
        lead_column=args.lead_column if args.lead_column is not None else defaults.lead_column,
        group_column=args.group_column if args.group_column is not None else defaults.group_column,
        batch_size=args.batch_size if args.batch_size is not None else defaults.batch_size,
        val_fraction=args.val_fraction if args.val_fraction is not None else defaults.val_fraction,
        num_workers=args.num_workers if args.num_workers is not None else defaults.num_workers,
        seed=args.seed if args.seed is not None else defaults.seed,
        epochs=args.epochs if args.epochs is not None else defaults.epochs,
        accelerator=args.accelerator or defaults.accelerator,
        devices=args.devices if args.devices is not None else defaults.devices,
        precision=args.precision or defaults.precision,
        max_train_steps=args.max_train_steps if args.max_train_steps is not None else defaults.max_train_steps,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else defaults.limit_val_batches,
        backbone=args.backbone or defaults.backbone,
        hidden_dim=args.hidden_dim if args.hidden_dim is not None else defaults.hidden_dim,
        dropout=args.dropout if args.dropout is not None else defaults.dropout,
        pretrained=False if args.no_pretrained else defaults.pretrained,
        use_refiner=args.use_refiner or defaults.use_refiner,
        refiner_channels=args.refiner_channels if args.refiner_channels is not None else defaults.refiner_channels,
        refiner_layers=args.refiner_layers if args.refiner_layers is not None else defaults.refiner_layers,
        signal_length=args.signal_length if args.signal_length is not None else defaults.signal_length,
        preload_signals=args.preload_signals or defaults.preload_signals,
        output_dir=args.output_dir or defaults.output_dir,
        optimizer=optimizer_cfg,
        augmentation=augmentation_cfg,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    run_dir = run_experiment(config)
    print(f"Completed training run at {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


