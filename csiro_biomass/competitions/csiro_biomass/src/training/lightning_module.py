from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
import numpy as np
import pandas as pd

from ..config.experiment import OptimizerConfig
from ..utils.metrics import compute_metrics

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class BiomassLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: OptimizerConfig,
        target_names: Sequence[str],
        huber_beta: float,
        train_sampler,
        *,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.0,
        save_oof: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.target_names = list(target_names)
        self.criterion = nn.SmoothL1Loss(beta=huber_beta)
        self.train_sampler = train_sampler
        self.mixup_alpha = float(mixup_alpha)
        self.mixup_prob = float(mixup_prob)
        self.save_oof = save_oof

        self.history: List[Dict[str, float]] = []
        self.best_val_rmse: float = float("inf")
        self._train_epoch_loss: float = float("nan")

        self.use_ema = optimizer_cfg.ema_decay is not None
        if self.use_ema:
            ema_decay = optimizer_cfg.ema_decay

            def ema_avg(averaged_param, model_param, num_averaged):
                return ema_decay * averaged_param + (1.0 - ema_decay) * model_param

            self.ema_model = AveragedModel(self.model, avg_fn=ema_avg)
        else:
            self.ema_model = None
        self._ema_backup_state: Optional[Dict[str, torch.Tensor]] = None

        self._val_loss_total: float = 0.0
        self._val_weight: int = 0
        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_ids: List[str] = []

        self._train_loss_total: float = 0.0
        self._train_weight_total: int = 0

        # Avoid storing large dataclasses in checkpoints.
        self.save_hyperparameters(
            {
                "target_names": self.target_names,
                "optimizer_cfg": asdict(self.optimizer_cfg),
                "huber_beta": huber_beta,
                "use_ema": self.use_ema,
                "mixup_alpha": self.mixup_alpha,
                "mixup_prob": self.mixup_prob,
            }
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.model(images, metadata)

    def on_train_start(self) -> None:
        if self.use_ema:
            self.ema_model.to(self.device)

    def on_train_epoch_start(self) -> None:
        if self.train_sampler is not None and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(self.current_epoch)
        self._train_loss_total = 0.0
        self._train_weight_total = 0

    def training_step(self, batch, batch_idx: int):
        images, targets, metadata, _ = batch
        if self.mixup_alpha > 0 and self.training and np.random.rand() < self.mixup_prob:
            mixed_images, targets_a, targets_b, lam = self._apply_mixup(images, targets)
            mean, _ = self.model(mixed_images, metadata)
            loss = mixup_criterion(self.criterion, mean, targets_a, targets_b, lam)
        else:
            mean, _ = self.model(images, metadata)
            loss = self.criterion(mean, targets)
        batch_size = targets.size(0)

        self._train_loss_total += loss.detach().cpu().item() * batch_size
        self._train_weight_total += batch_size
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
        return {"loss": loss, "batch_size": batch_size}

    def on_train_epoch_end(self) -> None:
        self._train_epoch_loss = self._train_loss_total / max(self._train_weight_total, 1)
        self.log("train_epoch_loss", self._train_epoch_loss, prog_bar=False, on_epoch=True)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.use_ema:
            self.ema_model.update_parameters(self.model)

    def on_validation_start(self) -> None:
        if self.use_ema and self.ema_model is not None:
            self._ema_backup_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema_model.module.state_dict())

    def on_validation_epoch_start(self) -> None:
        self._val_loss_total = 0.0
        self._val_weight = 0
        self._val_preds = []
        self._val_targets = []

    def validation_step(self, batch, batch_idx: int):
        images, targets, metadata, ids = batch
        mean, _ = self.model(images, metadata)
        loss = self.criterion(mean, targets)
        batch_size = targets.size(0)

        self._val_loss_total += loss.detach().cpu().item() * batch_size
        self._val_weight += batch_size
        self._val_preds.append(mean.detach().cpu())
        self._val_targets.append(targets.detach().cpu())
        self._val_ids.extend(ids)

        self.log("val_step_loss", loss, prog_bar=False, on_step=True, on_epoch=False, batch_size=batch_size)

    def on_validation_end(self) -> None:
        if self.use_ema and self._ema_backup_state is not None:
            self.model.load_state_dict(self._ema_backup_state)
            self._ema_backup_state = None

    def on_validation_epoch_end(self) -> None:
        val_loss = self._val_loss_total / max(self._val_weight, 1)
        preds = torch.cat(self._val_preds, dim=0) if self._val_preds else torch.empty(0)
        targets = torch.cat(self._val_targets, dim=0) if self._val_targets else torch.empty(0)
        metrics = compute_metrics(preds, targets, self.target_names)

        self.log("val_loss", val_loss, prog_bar=False, on_epoch=True)
        self.log("val_rmse", metrics["rmse"], prog_bar=True, on_epoch=True)
        self.log("val_mae", metrics["mae"], prog_bar=True, on_epoch=True)
        for name in self.target_names:
            mae_key = f"mae_{name}"
            rmse_key = f"rmse_{name}"
            if mae_key in metrics:
                self.log(f"val_{mae_key}", metrics[mae_key], prog_bar=False, on_epoch=True)
            if rmse_key in metrics:
                self.log(f"val_{rmse_key}", metrics[rmse_key], prog_bar=False, on_epoch=True)

        self.best_val_rmse = min(self.best_val_rmse, metrics["rmse"])
        history_entry = {
            "epoch": self.current_epoch + 1,
            "train_loss": self._train_epoch_loss,
            "val_loss": val_loss,
            "val_mae": metrics["mae"],
            "val_rmse": metrics["rmse"],
        }
        for name in self.target_names:
            for metric_type in ("mae", "rmse"):
                key = f"{metric_type}_{name}"
                if key in metrics:
                    history_entry[f"val_{key}"] = metrics[key]
        self.history.append(history_entry)

        if self.save_oof:
            log_dir = getattr(self.trainer, "log_dir", None) or getattr(self.trainer, "default_root_dir", None)
            if log_dir:
                log_path = Path(log_dir)
                log_path.mkdir(parents=True, exist_ok=True)
                oof_df = pd.DataFrame(preds.numpy(), columns=self.target_names)
                if targets.numel() and targets.ndim == 2:
                    targets_np = targets.numpy()
                    for idx, name in enumerate(self.target_names):
                        oof_df[f"{name}_actual"] = targets_np[:, idx]
                oof_df["identifier"] = self._val_ids
                oof_df.to_csv(log_path / "oof_predictions.csv", index=False)

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
            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.optimizer_cfg.warmup_epochs)
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
                "monitor": "val_rmse",
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        if self.use_ema and self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        if self.use_ema and self.ema_model is not None and "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])

    # ------------------------------------------------------------------
    # Mixup helper
    # ------------------------------------------------------------------
    def _apply_mixup(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        perm = torch.randperm(images.size(0), device=images.device)

        mixed_images = lam * images + (1 - lam) * images[perm]
        targets_a, targets_b = targets, targets[perm]

        return mixed_images, targets_a, targets_b, lam
