"""Lightning module wrapping the CAFA 6 neural baseline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from ..config.training import OptimizerConfig
from ..modeling.neural_baseline import ProteinFunctionPredictor
from ..utils.cafa_metrics import evaluate_cafa_metrics


class ProteinLightningModule(pl.LightningModule):
    """Train the multi-label protein classifier under Lightning."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        class_names: Sequence[str],
        optimizer_cfg: OptimizerConfig,
        val_accessions: Sequence[str],
        val_ground_truth: Dict[str, Sequence[str]],
        ontology,
    ) -> None:
        super().__init__()
        self.model = ProteinFunctionPredictor(
            embedding_dim=embedding_dim,
            num_labels=len(class_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.optimizer_cfg = optimizer_cfg
        self.class_names = list(class_names)
        self.val_accessions = list(val_accessions)
        self.val_ground_truth = {key: set(value) for key, value in val_ground_truth.items()}
        self.ontology = ontology

        self.criterion = nn.BCEWithLogitsLoss()
        self.history: List[Dict[str, float]] = []
        self.latest_metrics: Dict[str, float] = {}

        self._train_loss_total = 0.0
        self._train_batches = 0
        self._train_epoch_loss = float("nan")

        self._val_storage: List[Dict[str, torch.Tensor]] = []

        self.save_hyperparameters(
            {
                "embedding_dim": embedding_dim,
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
                "class_names": self.class_names,
                "optimizer_cfg": asdict(self.optimizer_cfg),
            }
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(embeddings)

    def on_train_epoch_start(self) -> None:
        self._train_loss_total = 0.0
        self._train_batches = 0

    def training_step(self, batch, batch_idx: int):
        embeddings, targets, _ = batch
        logits = self(embeddings)
        loss = self.criterion(logits, targets)

        self._train_loss_total += loss.detach().item()
        self._train_batches += 1

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=targets.size(0))
        return loss

    def on_train_epoch_end(self) -> None:
        denom = max(self._train_batches, 1)
        self._train_epoch_loss = self._train_loss_total / denom
        self.log("train_epoch_loss", self._train_epoch_loss, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        self._val_storage = []

    def validation_step(self, batch, batch_idx: int):
        embeddings, targets, indices = batch
        logits = self(embeddings)
        loss = self.criterion(logits, targets)

        self._val_storage.append(
            {
                "logits": logits.detach().cpu(),
                "targets": targets.detach().cpu(),
                "indices": indices.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )

        self.log("val_step_loss", loss, on_step=True, on_epoch=False, batch_size=targets.size(0))

    def on_validation_epoch_end(self) -> None:
        if not self._val_storage:
            return

        logits = torch.cat([item["logits"] for item in self._val_storage], dim=0)
        targets = torch.cat([item["targets"] for item in self._val_storage], dim=0)
        indices = torch.cat([item["indices"] for item in self._val_storage], dim=0)
        losses = torch.stack([item["loss"] for item in self._val_storage])

        order = torch.argsort(indices)
        logits = logits[order]
        targets = targets[order]

        val_loss = F.binary_cross_entropy_with_logits(logits, targets).item()
        probs = torch.sigmoid(logits).numpy()

        predictions: Dict[str, Dict[str, float]] = {}
        for accession, prob_vector in zip(self.val_accessions, probs):
            predictions[accession] = {
                go_term: float(prob)
                for go_term, prob in zip(self.class_names, prob_vector)
            }

        metrics = evaluate_cafa_metrics(
            predictions,
            {key: set(value) for key, value in self.val_ground_truth.items()},
            self.ontology,
        )

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_fmax", metrics.get("fmax", 0.0), prog_bar=True)
        self.log("val_coverage", metrics.get("coverage", 0.0), prog_bar=False)

        history_entry = {
            "epoch": float(self.current_epoch + 1),
            "train_loss": float(self._train_epoch_loss),
            "val_loss": float(val_loss),
            "val_fmax": float(metrics.get("fmax", 0.0)),
            "val_coverage": float(metrics.get("coverage", 0.0)),
        }
        self.history.append(history_entry)
        self.latest_metrics = {
            "val_loss": float(val_loss),
            "val_fmax": float(metrics.get("fmax", 0.0)),
            "val_coverage": float(metrics.get("coverage", 0.0)),
        }

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer_cfg.learning_rate,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        if not self.optimizer_cfg.use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.optimizer_cfg.scheduler_patience,
            factor=self.optimizer_cfg.scheduler_factor,
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        embeddings, _, indices = batch
        logits = self(embeddings)
        probabilities = torch.sigmoid(logits).detach().cpu()
        return {
            "indices": indices.detach().cpu(),
            "probabilities": probabilities,
        }


