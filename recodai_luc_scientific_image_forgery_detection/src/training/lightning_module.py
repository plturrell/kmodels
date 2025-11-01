"""Lightning module orchestrating joint segmentation + classification."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..config.training import OptimizerConfig
from ..utils.metrics import compute_classification_metrics, compute_segmentation_metrics


class ForgeryLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer_cfg: OptimizerConfig,
        mask_loss_weight: float,
        class_names: Sequence[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.mask_loss_weight = mask_loss_weight
        self.class_names = list(class_names)

        self.cls_criterion = nn.CrossEntropyLoss()
        self.mask_criterion = nn.BCEWithLogitsLoss()

        self.history: List[Dict[str, float]] = []
        self._train_loss = 0.0
        self._train_weight = 0
        self._train_correct = 0
        self._val_outputs: List[Dict[str, torch.Tensor]] = []

        self.save_hyperparameters(
            {
                "optimizer_cfg": asdict(self.optimizer_cfg),
                "mask_loss_weight": self.mask_loss_weight,
                "class_names": self.class_names,
            }
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        return self.model(image)

    def on_train_epoch_start(self) -> None:
        self._train_loss = 0.0
        self._train_weight = 0
        self._train_correct = 0

    def training_step(self, batch, batch_idx: int):
        class_logits, mask_logits = self(batch["image"])
        targets = batch["label"]
        masks = batch["mask"]

        cls_loss = self.cls_criterion(class_logits, targets)
        mask_loss = self.mask_criterion(mask_logits, masks)
        loss = cls_loss + self.mask_loss_weight * mask_loss

        preds = class_logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        batch_size = targets.size(0)

        self._train_loss += loss.detach().item() * batch_size
        self._train_weight += batch_size
        self._train_correct += correct

        metrics_cls = compute_classification_metrics(class_logits.detach(), targets.detach())
        metrics_mask = compute_segmentation_metrics(mask_logits.detach(), masks.detach())

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("train_cls_loss", cls_loss, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log("train_mask_loss", mask_loss, on_step=True, on_epoch=False, batch_size=batch_size)
        self.log("train_acc", metrics_cls["accuracy"], on_step=True, on_epoch=False, prog_bar=False, batch_size=batch_size)
        self.log("train_dice", metrics_mask["dice"], on_step=True, on_epoch=False, batch_size=batch_size)
        return loss

    def on_train_epoch_end(self) -> None:
        weight = max(self._train_weight, 1)
        train_loss = self._train_loss / weight
        train_acc = self._train_correct / weight
        self.log("train_epoch_loss", train_loss, prog_bar=False)
        self.log("train_epoch_acc", train_acc, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        self._val_outputs = []

    def validation_step(self, batch, batch_idx: int):
        class_logits, mask_logits = self(batch["image"])
        targets = batch["label"]
        masks = batch["mask"]

        cls_loss = self.cls_criterion(class_logits, targets)
        mask_loss = self.mask_criterion(mask_logits, masks)
        loss = cls_loss + self.mask_loss_weight * mask_loss

        self._val_outputs.append(
            {
                "loss": loss.detach(),
                "cls_logits": class_logits.detach(),
                "mask_logits": mask_logits.detach(),
                "targets": targets.detach(),
                "masks": masks.detach(),
            }
        )

        self.log("val_step_loss", loss, on_step=True, on_epoch=False, batch_size=targets.size(0))

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return
        losses = torch.stack([out["loss"] for out in self._val_outputs])
        cls_logits = torch.cat([out["cls_logits"] for out in self._val_outputs], dim=0)
        mask_logits = torch.cat([out["mask_logits"] for out in self._val_outputs], dim=0)
        targets = torch.cat([out["targets"] for out in self._val_outputs], dim=0)
        masks = torch.cat([out["masks"] for out in self._val_outputs], dim=0)

        classification_metrics = compute_classification_metrics(cls_logits, targets)
        segmentation_metrics = compute_segmentation_metrics(mask_logits, masks)

        val_loss = losses.mean().item()
        val_acc = classification_metrics["accuracy"]

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_dice", segmentation_metrics["dice"], prog_bar=False)
        self.log("val_iou", segmentation_metrics["iou"], prog_bar=False)

        self.history.append(
            {
                "epoch": self.current_epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_dice": segmentation_metrics["dice"],
                "val_iou": segmentation_metrics["iou"],
            }
        )

    # ------------------------------------------------------------------
    # Optimiser / scheduler configuration
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

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        class_logits, mask_logits = self(batch["image"])
        class_prob = F.softmax(class_logits, dim=1)
        mask_prob = torch.sigmoid(mask_logits)
        return {
            "class_prob": class_prob.detach().cpu(),
            "mask_prob": mask_prob.detach().cpu(),
            "sample": batch.get("sample", None),
        }


__all__ = ["ForgeryLightningModule"]

