"""Lightning module for the mouse behaviour baseline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from ..config.training import OptimizerConfig


class MouseBehaviorLightningModule(pl.LightningModule):
    """Wrap the GRU baseline with classification metrics and logging."""

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer_cfg: OptimizerConfig,
        class_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.class_names = list(class_names)
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_acc: float = 0.0
        self._train_loss_total: float = 0.0
        self._train_correct: int = 0
        self._train_weight: int = 0
        self._val_loss_total: float = 0.0
        self._val_correct: int = 0
        self._val_weight: int = 0
        self.history: List[Dict[str, float]] = []

        self.save_hyperparameters(
            {
                "optimizer_cfg": asdict(self.optimizer_cfg),
                "class_names": self.class_names,
            }
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(inputs, mask=mask)

    def on_train_epoch_start(self) -> None:
        self._train_loss_total = 0.0
        self._train_correct = 0
        self._train_weight = 0

    def training_step(self, batch, batch_idx: int):
        logits = self(batch["inputs"], mask=batch.get("mask"))
        targets = batch["target"]
        loss = self.criterion(logits, targets)

        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        batch_size = targets.size(0)

        self._train_loss_total += loss.detach().item() * batch_size
        self._train_correct += correct
        self._train_weight += batch_size

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        return loss

    def on_train_epoch_end(self) -> None:
        weight = max(self._train_weight, 1)
        avg_loss = self._train_loss_total / weight
        accuracy = self._train_correct / weight
        self.log("train_epoch_loss", avg_loss, prog_bar=False)
        self.log("train_epoch_acc", accuracy, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        self._val_loss_total = 0.0
        self._val_correct = 0
        self._val_weight = 0

    def validation_step(self, batch, batch_idx: int):
        logits = self(batch["inputs"], mask=batch.get("mask"))
        targets = batch["target"]
        loss = self.criterion(logits, targets)

        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        batch_size = targets.size(0)

        self._val_loss_total += loss.detach().item() * batch_size
        self._val_correct += correct
        self._val_weight += batch_size

        self.log("val_step_loss", loss, prog_bar=False, on_step=True, on_epoch=False, batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        weight = max(self._val_weight, 1)
        val_loss = self._val_loss_total / weight
        val_acc = self._val_correct / weight

        self.log("val_loss", val_loss, prog_bar=False)
        self.log("val_acc", val_acc, prog_bar=True)

        self.best_val_acc = max(self.best_val_acc, val_acc)
        self.history.append(
            {
                "epoch": self.current_epoch + 1,
                "train_loss": self._train_loss_total / max(self._train_weight, 1),
                "train_acc": self._train_correct / max(self._train_weight, 1),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    # ------------------------------------------------------------------
    # Optimiser configuration
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.learning_rate,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        return optimizer

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        logits = self(batch["inputs"], mask=batch.get("mask"))
        probabilities = F.softmax(logits, dim=1)
        preds = probabilities.argmax(dim=1)
        return {
            "sample_id": list(batch["sample_id"]),
            "pred_class": preds.detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
        }


