"""Advanced PyTorch training pipeline for interaction-aware trajectory models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from competitions.nfl_big_data_bowl_2026_prediction.src.data import (
    TrajectoryDataset,
    collate_trajectories,
    load_train_inputs,
    load_train_outputs,
)
from competitions.nfl_big_data_bowl_2026_prediction.src.modeling import (
    DiffusionConfig,
    RelationalTrajectoryModel,
    TrajectoryDiffusion,
)


def _concat_inputs() -> pd.DataFrame:
    inputs = load_train_inputs()
    useful_cols = [
        "game_id",
        "play_id",
        "frame_id",
        "nfl_id",
        "player_to_predict",
        "x",
        "y",
        "s",
        "a",
        "dir",
        "o",
        "absolute_yardline_number",
    ]
    extra = [col for col in inputs.columns if col not in useful_cols]
    feature_cols = useful_cols + extra
    return inputs[feature_cols].copy()


def _concat_outputs() -> pd.DataFrame:
    outputs = load_train_outputs()
    return outputs.copy()


def build_dataset(
    *,
    max_players: int = 22,
    max_time: int = 21,
    max_sequences: Optional[int] = None,
) -> TrajectoryDataset:
    inputs = _concat_inputs()
    outputs = _concat_outputs()
    numeric_columns = inputs.select_dtypes(include=["number"]).columns.tolist()
    excluded = {
        "game_id",
        "play_id",
        "frame_id",
        "nfl_id",
        "player_to_predict",
        "x",
        "y",
    }
    feature_columns = [col for col in numeric_columns if col not in excluded]
    return TrajectoryDataset(
        inputs,
        outputs,
        feature_columns=feature_columns,
        max_players=max_players,
        max_time=max_time,
        max_sequences=max_sequences,
    )


class RelationalLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        diffusion: Optional[TrajectoryDiffusion] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 2,
        total_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "diffusion"])
        self.model = model
        self.diffusion = diffusion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.mixup_alpha = 0.0
        self.position_jitter = 0.0

    def forward(self, batch):
        return self.model(batch["features"], batch["positions"], mask=batch["mask"])

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        torch.autograd.set_detect_anomaly(True)

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        torch.autograd.set_detect_anomaly(True)

    def _apply_augmentations(self, batch):
        if self.training and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            idx = torch.randperm(batch["features"].size(0), device=batch["features"].device)
            for key in ("features", "positions", "target"):
                batch[key] = lam * batch[key] + (1 - lam) * batch[key][idx]
        if self.training and self.position_jitter > 0:
            noise = torch.randn_like(batch["positions"]) * self.position_jitter
            batch["positions"] = batch["positions"] + noise
        return batch

    def training_step(self, batch, batch_idx):
        batch = self._apply_augmentations(batch)
        if self.diffusion is not None:
            loss, _ = self.diffusion(
                batch["features"],
                batch["positions"],
                target=batch["target"],
                mask=batch["mask"],
            )
        else:
            preds, _ = self.model(batch["features"], batch["positions"], mask=batch["mask"])
            if torch.isnan(preds).any():
                print(f'NaN predictions detected at batch {batch_idx}; stats:')
                print('features max', batch["features"].max().item(), 'min', batch["features"].min().item())
                print('positions max', batch["positions"].max().item(), 'min', batch["positions"].min().item())
                print('mask sum', batch["mask"].float().sum().item())
                print('target max', batch["target"].max().item(), 'min', batch["target"].min().item())
                nan_mask = torch.isnan(preds)
                print('nan entries', nan_mask.nonzero(as_tuple=False))
                print('pred stats pre', preds.nanmean().item() if torch.isnan(preds).any() else preds.mean().item())
                raise RuntimeError('NaN detected in predictions during training_step')
            mask = batch["mask"].float().mean(dim=2, keepdim=True)
            loss = torch.mean((preds * mask - batch["target"] * mask) ** 2)
            if torch.isnan(loss).any():
                print('NaN loss; preds stats', preds.max().item(), preds.min().item())
                raise RuntimeError('NaN loss encountered in training_step')
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, _ = self.model(batch["features"], batch["positions"], mask=batch["mask"])
        mask = batch["mask"].float().mean(dim=2, keepdim=True)
        loss = torch.mean((preds * mask - batch["target"] * mask) ** 2)
        mae = torch.mean(torch.abs(preds - batch["target"]))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        cosine = CosineAnnealingLR(optimizer, T_max=max(self.total_epochs - self.warmup_epochs, 1))
        if self.warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])
        else:
            scheduler = cosine
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    total_epochs: int = 50
    max_players: int = 22
    max_time: int = 21
    node_hidden_dim: int = 128
    gnn_layers: int = 3
    gnn_heads: int = 4
    temporal_layers: int = 2
    temporal_heads: int = 4
    dropout: float = 0.1
    use_diffusion: bool = False
    diffusion_timesteps: int = 128
    gradient_clip: float = 1.0
    precision: str = "bf16-mixed"
    num_workers: int = 4
    output_dir: Path = Path("competitions/nfl_big_data_bowl_2026_prediction/outputs/relational")
    max_sequences: Optional[int] = None
    accelerator: str = "auto"


def train_model(config: TrainingConfig) -> Path:
    dataset = build_dataset(
        max_players=config.max_players,
        max_time=config.max_time,
        max_sequences=config.max_sequences,
    )
    total_samples = len(dataset)
    val_size = max(int(0.1 * total_samples), 1)
    train_size = total_samples - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_trajectories,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_trajectories,
        drop_last=False,
    )

    node_input_dim = dataset.samples[0].features.shape[-1]
    model = RelationalTrajectoryModel(
        node_input_dim=node_input_dim,
        node_hidden_dim=config.node_hidden_dim,
        target_dim=2,
        gnn_layers=config.gnn_layers,
        gnn_heads=config.gnn_heads,
        temporal_layers=config.temporal_layers,
        temporal_heads=config.temporal_heads,
        dropout=config.dropout,
    )
    diffusion = None
    if config.use_diffusion:
        diffusion_cfg = DiffusionConfig(timesteps=config.diffusion_timesteps)
        diffusion = TrajectoryDiffusion(model, diffusion_cfg)

    lightning_module = RelationalLightningModule(
        model=model,
        diffusion=diffusion,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.total_epochs,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch{epoch:02d}-val{val_mae:.4f}",
            monitor="val_mae",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config.total_epochs >= 3:
        callbacks.append(StochasticWeightAveraging(swa_lrs=config.learning_rate * 0.1))

    trainer = pl.Trainer(
        max_epochs=config.total_epochs,
        default_root_dir=str(config.output_dir),
        accelerator=config.accelerator,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_path = Path(callbacks[0].best_model_path)
    config_path = config.output_dir / "training_config.json"
    config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()}
    config_path.write_text(json.dumps(config_dict, indent=2))
    return best_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train relational trajectory model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--total-epochs", type=int, default=50)
    parser.add_argument("--node-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--gnn-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-diffusion", action="store_true")
    parser.add_argument("--diffusion-timesteps", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=TrainingConfig.output_dir)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max-players", type=int, default=22)
    parser.add_argument("--max-time", type=int, default=21)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-sequences", type=int, help="Limit number of trajectory sequences for quick experiments")
    parser.add_argument("--accelerator", type=str, default="auto", help="Trainer accelerator setting (auto/cpu/mps/gpu)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Path:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.total_epochs,
        node_hidden_dim=args.node_hidden_dim,
        gnn_layers=args.gnn_layers,
        gnn_heads=args.gnn_heads,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        dropout=args.dropout,
        use_diffusion=args.use_diffusion,
        diffusion_timesteps=args.diffusion_timesteps,
        output_dir=args.output_dir,
        precision=args.precision,
        max_players=args.max_players,
        max_time=args.max_time,
        gradient_clip=args.gradient_clip,
        num_workers=args.num_workers,
        max_sequences=args.max_sequences,
        accelerator=args.accelerator,
    )
    best = train_model(config)
    print(f"Best checkpoint: {best}")
    return best


if __name__ == "__main__":  # pragma: no cover
    main()
