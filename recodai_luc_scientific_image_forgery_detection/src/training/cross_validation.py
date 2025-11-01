"""K-fold cross-validation for deep learning models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from ..config.training import ExperimentConfig
from ..data.dataset import Sample, load_samples, train_val_split
from ..training.datamodule import ForgeryDataModule
from ..training.lightning_module import ForgeryLightningModule


def run_cross_validation(
    config: ExperimentConfig,
    model_factory: Callable,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Run k-fold cross-validation.
    
    Args:
        config: Experiment configuration
        model_factory: Function that returns a new model instance
        n_folds: Number of folds
        seed: Random seed
    
    Returns:
        Dictionary of metrics per fold
    """
    pl.seed_everything(seed, workers=True)
    
    # Load all samples
    samples = load_samples(config.data_root)
    labels = np.array([s.label for s in samples])
    
    # Create stratified folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_metrics = {
        "val_acc": [],
        "val_dice": [],
        "val_iou": [],
        "val_loss": [],
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, labels), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{n_folds}")
        print(f"{'='*60}")
        
        # Split samples
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        
        # Create fold-specific config
        fold_config = ExperimentConfig(
            data_root=config.data_root,
            output_dir=config.output_dir / f"fold_{fold_idx}",
            epochs=config.epochs,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=seed + fold_idx,
            mask_loss_weight=config.mask_loss_weight,
            accelerator=config.accelerator,
            devices=config.devices,
            precision=config.precision,
            optimizer=config.optimizer,
            augmentation=config.augmentation,
        )
        
        # Create datamodule with fold-specific samples
        from torch.utils.data import DataLoader
        from ..data.dataset import ForgeryDataset
        from ..data.transforms import create_transforms
        
        train_transform = create_transforms(config.augmentation, is_training=True)
        val_transform = create_transforms(config.augmentation, is_training=False)
        
        train_dataset = ForgeryDataset(train_samples, transforms=train_transform)
        val_dataset = ForgeryDataset(val_samples, transforms=val_transform)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        
        # Create model
        model = model_factory()
        lightning_module = ForgeryLightningModule(
            model,
            optimizer_cfg=config.optimizer,
            mask_loss_weight=config.mask_loss_weight,
            class_names=("authentic", "forged"),
        )
        
        # Setup checkpoint callback
        checkpoint_dir = fold_config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
        )
        
        # Train
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            default_root_dir=str(fold_config.output_dir),
            accelerator=config.accelerator,
            devices=config.devices,
            precision=config.precision,
            callbacks=[checkpoint_callback],
            enable_progress_bar=True,
        )
        
        trainer.fit(lightning_module, train_loader, val_loader)
        
        # Store metrics
        if lightning_module.history:
            last_metrics = lightning_module.history[-1]
            fold_metrics["val_acc"].append(last_metrics["val_acc"])
            fold_metrics["val_dice"].append(last_metrics["val_dice"])
            fold_metrics["val_iou"].append(last_metrics["val_iou"])
            fold_metrics["val_loss"].append(last_metrics["val_loss"])
    
    # Calculate summary statistics
    summary = {}
    for metric_name, values in fold_metrics.items():
        summary[f"{metric_name}_mean"] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))
    
    # Save results
    results_path = config.output_dir / "cv_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            "fold_metrics": fold_metrics,
            "summary": summary,
            "n_folds": n_folds,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    return fold_metrics


__all__ = ["run_cross_validation"]

