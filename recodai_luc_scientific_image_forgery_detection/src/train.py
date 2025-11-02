"""Training entrypoint for the RecoDAI Lightning baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from .config.training import AugmentationConfig, ExperimentConfig, OptimizerConfig
from .modeling.baseline import VisionBaselineConfig, build_vision_baseline
from .modeling.dual_stream import DualStreamConfig, DualStreamForgeryModel
from .modeling.pretrained import PretrainedModelConfig, build_pretrained_model
from .modeling.utils import load_contrastive_encoder_weights
from .training.datamodule import ForgeryDataModule
from .training.lightning_module import ForgeryLightningModule


def _timestamped_run_dir(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> Path:
    pl.seed_everything(config.seed, workers=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _timestamped_run_dir(config.output_dir)

    datamodule = ForgeryDataModule(config)
    datamodule.setup(stage="fit")

    def _build_pretrained() -> nn.Module:
        encoder_weights = (
            None
            if config.contrastive_checkpoint is not None
            else (
                None
                if config.pretrained_weights is None
                or str(config.pretrained_weights).lower() == "none"
                else config.pretrained_weights
            )
        )
        pretrained_model = build_pretrained_model(
            architecture=config.pretrained_architecture,
            encoder_name=config.pretrained_encoder,
            encoder_weights=encoder_weights,
            num_classes=len(datamodule.class_names),
            segmentation_classes=1,
        )
        if config.contrastive_checkpoint is not None:
            load_contrastive_encoder_weights(pretrained_model, config.contrastive_checkpoint)
        if config.freeze_encoder:
            for param in pretrained_model.segmentation_model.encoder.parameters():
                param.requires_grad = False
        return pretrained_model

    if config.model_type == "pretrained":
        model = _build_pretrained()
    elif config.model_type == "dual_stream":
        spatial_model = _build_pretrained()
        dual_config = DualStreamConfig(
            spatial=PretrainedModelConfig(
                architecture=config.pretrained_architecture,
                encoder_name=config.pretrained_encoder,
                encoder_weights=None,
                num_classes=len(datamodule.class_names),
                segmentation_classes=1,
            ),
            num_classes=len(datamodule.class_names),
            frequency_channels=3,
        )
        model = DualStreamForgeryModel(dual_config, spatial_model=spatial_model)
    else:
        model = build_vision_baseline(
            VisionBaselineConfig(num_classes=len(datamodule.class_names))
        )
    lightning_module = ForgeryLightningModule(
        model,
        optimizer_cfg=config.optimizer,
        mask_loss_weight=config.mask_loss_weight,
        class_names=datamodule.class_names,
        class_weights=datamodule.class_weights,
        mask_loss=config.mask_loss,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch{epoch:02d}-val{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
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
    if config.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = config.gradient_clip_val
    if config.max_train_steps is not None:
        trainer_kwargs["max_steps"] = config.max_train_steps
    if config.limit_val_batches is not None:
        trainer_kwargs["limit_val_batches"] = config.limit_val_batches

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule=datamodule)

    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path

    _write_json(run_dir / "history.json", lightning_module.history)
    _write_json(run_dir / "config.json", config.to_dict())
    _write_json(
        run_dir / "summary.json",
        {
            "best_checkpoint": best_path,
            "epochs": config.epochs,
        },
    )

    (run_dir / "class_names.json").write_text(json.dumps(list(datamodule.class_names), indent=2))

    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the RecoDAI Lightning baseline.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mask-loss-weight", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--precision")
    parser.add_argument("--gradient-clip", type=float)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--model-type", choices=("baseline", "pretrained", "dual_stream"))
    parser.add_argument("--pretrained-architecture")
    parser.add_argument("--pretrained-encoder")
    parser.add_argument("--pretrained-weights")
    parser.add_argument("--contrastive-checkpoint", type=Path)
    parser.add_argument("--mask-loss", choices=("bce", "combined"))
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.set_defaults(use_class_weights=None)
    parser.add_argument("--freeze-encoder", action="store_true")

    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--scheduler-t-max", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--ema-decay", type=float)

    parser.add_argument("--aug-horizontal-flip", type=float)
    parser.add_argument("--aug-vertical-flip", type=float)
    parser.add_argument("--aug-rotation-prob", type=float)
    parser.add_argument("--aug-jitter-strength", type=float)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    parser = _build_parser()
    args = parser.parse_args(argv)

    default_opt = OptimizerConfig()
    optimizer_cfg = OptimizerConfig(
        learning_rate=args.learning_rate if args.learning_rate is not None else default_opt.learning_rate,
        weight_decay=args.weight_decay if args.weight_decay is not None else default_opt.weight_decay,
        use_scheduler=not args.no_scheduler,
        scheduler_t_max=args.scheduler_t_max if args.scheduler_t_max is not None else default_opt.scheduler_t_max,
        warmup_epochs=args.warmup_epochs if args.warmup_epochs is not None else default_opt.warmup_epochs,
        ema_decay=args.ema_decay if args.ema_decay is not None else default_opt.ema_decay,
    )

    default_aug = AugmentationConfig()
    augmentation_cfg = AugmentationConfig(
        horizontal_flip=args.aug_horizontal_flip
        if args.aug_horizontal_flip is not None
        else default_aug.horizontal_flip,
        vertical_flip=args.aug_vertical_flip
        if args.aug_vertical_flip is not None
        else default_aug.vertical_flip,
        rotation_probability=args.aug_rotation_prob
        if args.aug_rotation_prob is not None
        else default_aug.rotation_probability,
        jitter_strength=args.aug_jitter_strength
        if args.aug_jitter_strength is not None
        else default_aug.jitter_strength,
    )

    default_cfg = ExperimentConfig()
    model_type = args.model_type if args.model_type is not None else default_cfg.model_type
    pretrained_architecture = (
        args.pretrained_architecture if args.pretrained_architecture is not None else default_cfg.pretrained_architecture
    )
    pretrained_encoder = (
        args.pretrained_encoder if args.pretrained_encoder is not None else default_cfg.pretrained_encoder
    )
    pretrained_weights = (
        args.pretrained_weights if args.pretrained_weights is not None else default_cfg.pretrained_weights
    )
    use_class_weights = (
        default_cfg.use_class_weights if args.use_class_weights is None else args.use_class_weights
    )
    mask_loss = args.mask_loss if args.mask_loss is not None else default_cfg.mask_loss
    contrastive_checkpoint = (
        args.contrastive_checkpoint
        if args.contrastive_checkpoint is not None
        else default_cfg.contrastive_checkpoint
    )
    cfg = ExperimentConfig(
        data_root=args.data_root if args.data_root is not None else default_cfg.data_root,
        output_dir=args.output_dir if args.output_dir is not None else default_cfg.output_dir,
        epochs=args.epochs if args.epochs is not None else default_cfg.epochs,
        batch_size=args.batch_size if args.batch_size is not None else default_cfg.batch_size,
        val_fraction=args.val_fraction if args.val_fraction is not None else default_cfg.val_fraction,
        num_workers=args.num_workers if args.num_workers is not None else default_cfg.num_workers,
        seed=args.seed if args.seed is not None else default_cfg.seed,
        mask_loss_weight=args.mask_loss_weight
        if args.mask_loss_weight is not None
        else default_cfg.mask_loss_weight,
        max_train_samples=args.max_train_samples
        if args.max_train_samples is not None
        else default_cfg.max_train_samples,
        max_val_samples=args.max_val_samples
        if args.max_val_samples is not None
        else default_cfg.max_val_samples,
        accelerator=args.accelerator if args.accelerator is not None else default_cfg.accelerator,
        devices=args.devices if args.devices is not None else default_cfg.devices,
        precision=args.precision if args.precision is not None else default_cfg.precision,
        gradient_clip_val=args.gradient_clip if args.gradient_clip is not None else default_cfg.gradient_clip_val,
        max_train_steps=args.max_train_steps if args.max_train_steps is not None else default_cfg.max_train_steps,
        limit_val_batches=args.limit_val_batches
        if args.limit_val_batches is not None
        else default_cfg.limit_val_batches,
        optimizer=optimizer_cfg,
        augmentation=augmentation_cfg,
        model_type=model_type,
        pretrained_architecture=pretrained_architecture,
        pretrained_encoder=pretrained_encoder,
        pretrained_weights=pretrained_weights,
        use_class_weights=use_class_weights,
        mask_loss=mask_loss,
        contrastive_checkpoint=contrastive_checkpoint,
        freeze_encoder=args.freeze_encoder or default_cfg.freeze_encoder,
    )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_args(argv)
    run_dir = run_experiment(config)
    print(f"Run artifacts stored in {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
