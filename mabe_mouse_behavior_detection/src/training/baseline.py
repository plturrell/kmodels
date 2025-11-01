"""Lightning entrypoint for the MABe mouse behaviour baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ..config.training import TrainingConfig as ExperimentTrainingConfig
from ..config.training import AugmentationConfig, OptimizerConfig
from ..modeling import PoseBaselineConfig, build_pose_baseline
from .datamodule import BehaviorLightningDataModule, SequenceTransformConfig
from .lightning_module import MouseBehaviorLightningModule


def _timestamped_run_dir(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _create_datamodule(cfg: ExperimentTrainingConfig) -> BehaviorLightningDataModule:
    transform_cfg = SequenceTransformConfig(
        array_key=cfg.pose_array,
        sequence_length=cfg.sequence_length,
        time_axis=cfg.time_axis,
        center=cfg.center,
        standardize=cfg.standardize,
    )
    return BehaviorLightningDataModule(
        train_csv=cfg.train_csv,
        asset_root=cfg.asset_root,
        asset_column=cfg.asset_column,
        target_column=cfg.target_column,
        id_column=cfg.id_column,
        metadata_columns=cfg.metadata_columns,
        batch_size=cfg.batch_size,
        val_batch_size=cfg.val_batch_size,
        val_fraction=cfg.val_fraction,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        transform_cfg=transform_cfg,
        max_train_samples=cfg.max_train_samples,
        max_val_samples=cfg.max_val_samples,
    )


def _build_lightning_components(
    cfg: ExperimentTrainingConfig,
) -> tuple[BehaviorLightningDataModule, MouseBehaviorLightningModule, PoseBaselineConfig]:
    datamodule = _create_datamodule(cfg)
    datamodule.setup(stage="fit")

    if datamodule.input_dim is None:
        raise RuntimeError("Failed to infer input dimension from the training dataset.")

    model_spec = PoseBaselineConfig(
        input_dim=datamodule.input_dim,
        num_classes=len(datamodule.index_to_label),
    )
    model = build_pose_baseline(
        replace(
            model_spec,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
            pooling=cfg.pooling,
        )
    )

    lightning_module = MouseBehaviorLightningModule(
        model,
        optimizer_cfg=cfg.optimizer,
        class_names=[str(label) for label in datamodule.index_to_label],
    )
    return datamodule, lightning_module, model_spec


def _build_trainer(cfg: ExperimentTrainingConfig, run_dir: Path, checkpoint_callback: ModelCheckpoint) -> pl.Trainer:
    trainer_kwargs = {
        "max_epochs": cfg.epochs,
        "default_root_dir": str(run_dir),
        "accelerator": cfg.accelerator,
        "devices": cfg.devices,
        "precision": cfg.precision,
        "log_every_n_steps": 25,
        "callbacks": [checkpoint_callback],
    }
    if cfg.max_train_steps is not None:
        trainer_kwargs["max_steps"] = cfg.max_train_steps
    if cfg.max_val_steps is not None:
        trainer_kwargs["limit_val_batches"] = cfg.max_val_steps
    if cfg.optimizer.gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = cfg.optimizer.gradient_clip_val

    return pl.Trainer(**trainer_kwargs)


def _run_training(cfg: ExperimentTrainingConfig) -> tuple[Path, MouseBehaviorLightningModule, BehaviorLightningDataModule, ModelCheckpoint]:
    pl.seed_everything(cfg.seed, workers=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _timestamped_run_dir(cfg.output_dir)

    datamodule, lightning_module, _ = _build_lightning_components(cfg)

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

    trainer = _build_trainer(cfg, run_dir, checkpoint_callback)
    trainer.fit(lightning_module, datamodule=datamodule)

    return run_dir, lightning_module, datamodule, checkpoint_callback


def _log_artifacts(
    cfg: ExperimentTrainingConfig,
    run_dir: Path,
    lightning_module: MouseBehaviorLightningModule,
    checkpoint_callback: ModelCheckpoint,
) -> Path:
    history_path = run_dir / "history.json"
    _write_json(history_path, lightning_module.history)

    summary = {
        "best_val_acc": lightning_module.best_val_acc,
        "epochs": cfg.epochs,
        "monitor": "val_acc",
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "config.json", cfg.to_dict())

    if checkpoint_callback.best_model_path:
        return Path(checkpoint_callback.best_model_path)
    return Path(checkpoint_callback.last_model_path)


def _predict_and_write_submission(
    cfg: ExperimentTrainingConfig,
    datamodule: BehaviorLightningDataModule,
    checkpoint_path: Path,
    run_dir: Path,
) -> None:
    if cfg.test_csv is None:
        return

    test_dataset = datamodule.build_test_dataset(
        test_csv=cfg.test_csv,
        asset_root=cfg.test_asset_root,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.val_batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    trainer = pl.Trainer(accelerator=cfg.accelerator, devices=cfg.devices)
    predictions = trainer.predict(  # type: ignore[arg-type]
        model=MouseBehaviorLightningModule.load_from_checkpoint(  # type: ignore[attr-defined]
            str(checkpoint_path),
            map_location="cpu",
        ),
        dataloaders=test_loader,
    )

    id_column = cfg.id_output_column or cfg.id_column
    rows = []
    for batch in predictions:
        probs = batch["probabilities"].numpy()
        preds = batch["pred_class"].numpy()
        sample_ids = batch["sample_id"]
        for sample_id, pred_idx, prob_vec in zip(sample_ids, preds, probs):
            row = {
                id_column: str(sample_id),
                cfg.prediction_column: datamodule.index_to_label[pred_idx],
            }
            for class_idx, class_name in enumerate(datamodule.index_to_label):
                row[f"prob_{class_name}"] = float(prob_vec[class_idx])
            rows.append(row)

    submission_df = pd.DataFrame(rows)
    submission_path = cfg.submission_path or (run_dir / "submission.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    latest_path = cfg.output_dir / "latest_submission.csv"
    submission_df.to_csv(latest_path, index=False)


def parse_args(argv: Optional[Iterable[str]] = None) -> ExperimentTrainingConfig:
    parser = argparse.ArgumentParser(description="Train the Lightning-based MABe baseline.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--asset-column", default="pose_path")
    parser.add_argument("--target-column", default="behavior")
    parser.add_argument("--id-column", default="clip_id")
    parser.add_argument("--metadata-column", action="append", default=[])

    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--time-axis", type=int, default=0)
    parser.add_argument("--pose-array", default="keypoints")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--standardize", action="store_true")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-val-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--pooling", choices=["last", "mean"], default="last")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="32")

    parser.add_argument("--output-dir", type=Path, default=Path("competitions/mabe_mouse_behavior_detection/outputs/lightning_baseline"))
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)

    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--test-asset-root", type=Path)
    parser.add_argument("--submission-path", type=Path)
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--id-output-column")

    args = parser.parse_args(argv)

    optimizer_cfg = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip,
    )
    augmentation_cfg = AugmentationConfig()

    config = ExperimentTrainingConfig(
        train_csv=args.train_csv,
        asset_root=args.asset_root,
        asset_column=args.asset_column,
        target_column=args.target_column,
        id_column=args.id_column,
        metadata_columns=[col for col in (args.metadata_column or []) if col],
        sequence_length=args.sequence_length,
        time_axis=args.time_axis,
        pose_array=args.pose_array,
        center=not args.no_center,
        standardize=args.standardize,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        precision=args.precision,
        devices=args.devices,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        optimizer=optimizer_cfg,
        augmentation=augmentation_cfg,
    )

    if args.test_csv:
        config.test_csv = args.test_csv
        config.test_asset_root = args.test_asset_root
        config.submission_path = args.submission_path
    config.prediction_column = args.prediction_column
    if args.id_output_column:
        config.id_output_column = args.id_output_column

    return config


def main(argv: Optional[Iterable[str]] = None) -> int:
    cfg = parse_args(argv)
    run_dir, lightning_module, datamodule, checkpoint_callback = _run_training(cfg)
    best_checkpoint = _log_artifacts(cfg, run_dir, lightning_module, checkpoint_callback)
    _predict_and_write_submission(cfg, datamodule, best_checkpoint, run_dir)
    print(f"Run artifacts stored in {run_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
