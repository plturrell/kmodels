"""Lightning module wrapping the ECG regression baseline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..config.training import OptimizerConfig


class ECGLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer_cfg: OptimizerConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg

        self.history: List[Dict[str, float]] = []
        self._train_loss = 0.0
        self._train_mae = 0.0
        self._train_batches = 0
        self._train_epoch_loss = float("nan")
        self._train_epoch_mae = float("nan")
        self._val_outputs: List[Dict[str, torch.Tensor]] = []

        self.save_hyperparameters({"optimizer_cfg": asdict(self.optimizer_cfg)})

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(images)

    @staticmethod
    def _align_predictions(
        preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if targets.ndim == 3 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        if preds.ndim == 3 and preds.size(1) == 1:
            preds = preds.squeeze(1)

        if preds.shape != targets.shape:
            raise RuntimeError(
                "Prediction and target tensors must share the same shape, "
                f"but received {tuple(preds.shape)} vs {tuple(targets.shape)}."
            )
        return preds, targets

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        self._train_loss = 0.0
        self._train_mae = 0.0
        self._train_batches = 0

    def training_step(self, batch, batch_idx: int):
        preds = self(batch["image"])
        targets = batch["signal"]
        if targets is None:
            raise RuntimeError("Training batch missing signals.")
        preds, targets = self._align_predictions(preds, targets)

        loss = torch.nn.functional.mse_loss(preds, targets)
        mae = torch.mean(torch.abs(preds.detach() - targets.detach()))

        self._train_loss += loss.detach().item()
        self._train_mae += mae.item()
        self._train_batches += 1

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=targets.size(0))
        self.log("train_mae", mae, on_step=True, on_epoch=False, prog_bar=False, batch_size=targets.size(0))
        return loss

    def on_train_epoch_end(self) -> None:
        denom = max(self._train_batches, 1)
        self._train_epoch_loss = self._train_loss / denom
        self._train_epoch_mae = self._train_mae / denom
        self.log("train_epoch_loss", self._train_epoch_loss, prog_bar=False)
        self.log("train_epoch_mae", self._train_epoch_mae, prog_bar=False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def validation_step(self, batch, batch_idx: int):
        preds = self(batch["image"])
        targets = batch["signal"]
        if targets is None:
            return
        preds, targets = self._align_predictions(preds, targets)

        loss = torch.nn.functional.mse_loss(preds, targets)
        mae = torch.mean(torch.abs(preds - targets))

        self._val_outputs.append({"loss": loss.detach(), "mae": mae.detach()})
        self.log("val_step_loss", loss, on_step=True, on_epoch=False, batch_size=targets.size(0))

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return
        losses = torch.stack([item["loss"] for item in self._val_outputs]).mean().item()
        maes = torch.stack([item["mae"] for item in self._val_outputs]).mean().item()

        self.log("val_loss", losses, prog_bar=True)
        self.log("val_mae", maes, prog_bar=True)
        self.history.append(
            {
                "epoch": self.current_epoch + 1,
                "train_loss": self._train_epoch_loss,
                "train_mae": self._train_epoch_mae,
                "val_loss": losses,
                "val_mae": maes,
            }
        )

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.learning_rate,
            weight_decay=self.optimizer_cfg.weight_decay,
        )

        if not self.optimizer_cfg.use_scheduler:
            return optimizer

        cosine = CosineAnnealingLR(optimizer, T_max=self.optimizer_cfg.scheduler_t_max)
        if self.optimizer_cfg.warmup_epochs > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.optimizer_cfg.warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.optimizer_cfg.warmup_epochs],
            )
        else:
            scheduler = cosine

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


