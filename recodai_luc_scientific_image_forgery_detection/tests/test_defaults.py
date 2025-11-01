"""Sanity checks for default configuration and data access paths."""

from __future__ import annotations

from pathlib import Path

from src.config.training import ExperimentConfig
from src.training.datamodule import ForgeryDataModule


def test_default_paths_resolve() -> None:
    cfg = ExperimentConfig()
    assert cfg.data_root.exists(), f"Expected data root {cfg.data_root} to exist"
    # output directory may not be created yet, but it should live under a real parent
    assert cfg.output_dir.parent.exists(), f"Expected output parent {cfg.output_dir.parent} to exist"
    assert cfg.max_train_samples is None
    assert cfg.max_val_samples is None


def test_datamodule_uses_full_dataset(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        output_dir=tmp_path,
        max_train_samples=4,
        max_val_samples=2,
        batch_size=2,
    )
    dm = ForgeryDataModule(cfg)
    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert len(train_loader.dataset) <= 4
    assert len(val_loader.dataset) <= 2
