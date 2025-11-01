"""K-fold cross-validation for ECG digitisation models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from ..config.training import AugmentationConfig, OptimizerConfig
from ..data.dataset import ECGDigitizationDataset, ECGSample
from ..features.transforms import build_eval_transform, build_train_transform
from .lightning_module import ECGLightningModule


def _infer_signal_shape(samples: Sequence[ECGSample]) -> Tuple[int, int]:
    for sample in samples:
        dataset = ECGDigitizationDataset([sample], preload_signals=True)
        batch = dataset[0]
        signal = batch.get("signal")
        if signal is None:
            continue
        if signal.ndim == 1:
            return signal.shape[0], 1
        if signal.ndim == 2:
            return signal.shape[1], signal.shape[0]
        raise RuntimeError(
            f"Unsupported signal tensor shape encountered: {tuple(signal.shape)}"
        )
    raise RuntimeError(
        "Unable to infer signal dimensions; ensure at least one sample includes a signal path."
    )


def _build_model(model_factory: Callable, signal_length: int, signal_channels: int):
    try:
        return model_factory(signal_length=signal_length, signal_channels=signal_channels)
    except TypeError:
        try:
            return model_factory(signal_length, signal_channels)
        except TypeError:
            return model_factory()


def _collate_batch(batch):
    images = torch.stack([item["image"] for item in batch])
    signal_tensors = [
        item.get("signal")
        for item in batch
        if isinstance(item.get("signal"), torch.Tensor)
    ]
    signals = torch.stack(signal_tensors) if len(signal_tensors) == len(batch) else None
    ids = [str(item.get("id", "")) for item in batch]
    leads = [item.get("lead") for item in batch]
    return {"image": images, "signal": signals, "id": ids, "lead": leads}


def _create_dataloaders(
    train_samples: Sequence[ECGSample],
    val_samples: Sequence[ECGSample],
    *,
    batch_size: int,
    num_workers: int,
    augmentation: AugmentationConfig,
    preload_signals: bool,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = build_train_transform(
        image_size=augmentation.image_size,
        hflip_prob=augmentation.horizontal_flip,
        vflip_prob=augmentation.vertical_flip,
        affine_degrees=augmentation.random_rotation,
    )
    val_tf = build_eval_transform(image_size=augmentation.image_size)

    train_dataset = ECGDigitizationDataset(
        train_samples,
        transforms=train_tf,
        preload_signals=preload_signals,
    )
    val_dataset = ECGDigitizationDataset(
        val_samples,
        transforms=val_tf,
        preload_signals=preload_signals,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_batch,
    )
    return train_loader, val_loader


def _build_splitter(
    samples: Sequence[ECGSample],
    *,
    n_folds: int,
    stratify_by: Optional[str],
    seed: int,
):
    if stratify_by == "lead":
        labels = np.array([sample.lead or "unknown" for sample in samples])
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return splitter.split(samples, encoded)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return splitter.split(samples)


def run_cross_validation(
    samples: List[ECGSample],
    model_factory: Callable,
    *,
    n_folds: int = 5,
    stratify_by: Optional[str] = "lead",
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 4,
    augmentation: Optional[AugmentationConfig] = None,
    optimizer_cfg: Optional[OptimizerConfig] = None,
    preload_signals: bool = False,
    accelerator: str = "auto",
    devices: int | Sequence[int] | str = 1,
    precision: str = "32",
) -> Dict[str, List[float]]:
    """Run k-fold cross-validation with the Lightning training stack."""

    if not samples:
        raise ValueError("No samples provided for cross-validation.")

    pl.seed_everything(seed, workers=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmentation = augmentation or AugmentationConfig()
    optimizer_cfg = optimizer_cfg or OptimizerConfig()

    signal_length, signal_channels = _infer_signal_shape(samples)

    fold_metrics: Dict[str, List[float]] = {"val_loss": [], "val_mae": []}
    fold_records = []

    splitter = _build_splitter(samples, n_folds=n_folds, stratify_by=stratify_by, seed=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter, start=1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx}/{n_folds}")
        print(f"{'=' * 60}")

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        print(f"Train: {len(train_samples)} samples")
        print(f"Val: {len(val_samples)} samples")

        train_loader, val_loader = _create_dataloaders(
            train_samples,
            val_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            augmentation=augmentation,
            preload_signals=preload_signals,
        )

        model = _build_model(model_factory, signal_length, signal_channels)
        lightning_module = ECGLightningModule(model, optimizer_cfg=optimizer_cfg)

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(fold_dir / "checkpoints"),
            filename="best",
            monitor="val_mae",
            mode="min",
            save_top_k=1,
            save_last=True,
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir=str(fold_dir),
            callbacks=[checkpoint_callback],
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            enable_progress_bar=True,
        )

        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        metrics_entry: Dict[str, float] = {
            "fold": float(fold_idx),
            "best_checkpoint": checkpoint_callback.best_model_path,
        }

        if lightning_module.history:
            last_metrics = lightning_module.history[-1]
            val_loss = float(last_metrics.get("val_loss", 0.0))
            val_mae = float(last_metrics.get("val_mae", 0.0))
            fold_metrics["val_loss"].append(val_loss)
            fold_metrics["val_mae"].append(val_mae)
            metrics_entry.update({
                "train_loss": float(last_metrics.get("train_loss", 0.0)),
                "train_mae": float(last_metrics.get("train_mae", 0.0)),
                "val_loss": val_loss,
                "val_mae": val_mae,
            })
        fold_records.append(metrics_entry)

    summary = {}
    for metric_name, values in fold_metrics.items():
        if values:
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))

    results_path = output_dir / "cv_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "fold_metrics": fold_metrics,
                "fold_records": fold_records,
                "summary": summary,
                "n_folds": n_folds,
                "signal_length": signal_length,
                "signal_channels": signal_channels,
            },
            fh,
            indent=2,
        )

    print(f"\n{'=' * 60}")
    print("Cross-Validation Summary")
    print(f"{'=' * 60}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    return fold_metrics


__all__ = ["run_cross_validation"]

