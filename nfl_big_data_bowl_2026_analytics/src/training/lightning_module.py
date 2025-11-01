"""Lightning module for the analytics baseline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config.experiment import OptimizerConfig
from ..utils.metrics import compute_metrics


class AnalyticsLightningModule(pl.LightningModule):
    """Wrap a regression network with training/eval utilities."""

    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: OptimizerConfig,
        target_names: Sequence[str],
        is_graph_module: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.target_names = list(target_names)
        self.is_graph_module = is_graph_module

        self.criterion = nn.SmoothL1Loss()
        self.history: List[Dict[str, float]] = []
        self.best_val_rmse: float = float("inf")

        self._epoch_train_loss: float = float("nan")
        self._train_loss_total: float = 0.0
        self._train_weight: int = 0
        self._val_predictions: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []

        self.save_hyperparameters(
            {
                "optimizer_cfg": asdict(self.optimizer_cfg),
                "target_names": self.target_names,
            }
        )

    def forward(self, batch) -> torch.Tensor:  # noqa: D401
        # For GNNs, the batch is the graph data object. For others, it's the feature tensor.
        return self.model(batch)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        self._train_loss_total = 0.0
        self._train_weight = 0

    def training_step(self, batch, batch_idx: int):
        if self.is_graph_module:
            targets = batch.y
            preds = self(batch)
        else:
            features, targets, _ = batch
            preds = self(features)
        
        loss = self.criterion(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=targets.size(0))
        self._train_loss_total += loss.detach().item() * targets.size(0)
        self._train_weight += targets.size(0)
        return loss

    def on_train_epoch_end(self) -> None:
        weight = max(self._train_weight, 1)
        self._epoch_train_loss = self._train_loss_total / weight
        self.log("train_epoch_loss", self._epoch_train_loss, prog_bar=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        self._val_predictions = []
        self._val_targets = []

    def validation_step(self, batch, batch_idx: int):
        if self.is_graph_module:
            targets = batch.y
            preds = self(batch)
        else:
            features, targets, _ = batch
            preds = self(features)

        loss = self.criterion(preds, targets)
        self._val_predictions.append(preds.detach().cpu())
        self._val_targets.append(targets.detach().cpu())
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, batch_size=targets.size(0))
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_predictions or not self._val_targets:
            return
        preds = torch.cat(self._val_predictions, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        metrics = compute_metrics(preds, targets, self.target_names)
        self.best_val_rmse = min(self.best_val_rmse, metrics["rmse"])

        train_loss_value = self._train_loss_total / max(self._train_weight, 1)
        self._epoch_train_loss = train_loss_value

        self.log("val_rmse", metrics["rmse"], prog_bar=True)
        self.log("val_mae", metrics["mae"], prog_bar=False)
        for name in self.target_names:
            mae_key = f"mae_{name}"
            rmse_key = f"rmse_{name}"
            if mae_key in metrics:
                self.log(f"val_{mae_key}", metrics[mae_key], prog_bar=False)
            if rmse_key in metrics:
                self.log(f"val_{rmse_key}", metrics[rmse_key], prog_bar=False)

        self.history.append(
            {
                "epoch": self.current_epoch + 1,
                "train_loss": train_loss_value,
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
            }
        )

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.learning_rate,
            weight_decay=self.optimizer_cfg.weight_decay,
            betas=self.optimizer_cfg.betas,
            eps=self.optimizer_cfg.eps,
        )

        if not self.optimizer_cfg.use_scheduler:
            return optimizer

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_cfg.cosine_t_max,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_rmse",
            },
        }
