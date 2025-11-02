"""Self-supervised contrastive pretraining module for forgery detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models


@dataclass
class ContrastiveConfig:
    backbone: str = "resnet50"
    projection_dim: int = 256
    hidden_dim: int = 2048
    temperature: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def _build_backbone(backbone: str) -> nn.Module:
    model_fn = getattr(models, backbone)
    model = model_fn(weights=None)
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    model.out_features = in_features
    return model


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class ContrastivePretrainModule(pl.LightningModule):
    """SimCLR-style NT-Xent training loop."""

    def __init__(self, config: ContrastiveConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.__dict__)
        self.config = config
        self.backbone = _build_backbone(config.backbone)
        self.projection_head = ProjectionHead(
            in_dim=self.backbone.out_features,
            hidden_dim=config.hidden_dim,
            out_dim=config.projection_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projections = self.projection_head(features)
        return nn.functional.normalize(projections, dim=1)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        z_i = self(batch["view1"])
        z_j = self(batch["view2"])
        loss = self.nt_xent_loss(z_i, z_j)
        self.log("train_contrastive_loss", loss, prog_bar=True, batch_size=z_i.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        max_epochs = self.trainer.max_epochs if self.trainer is not None else 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the NT-Xent loss between two batches of projections."""

        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)
        similarity = nn.functional.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2,
        )

        # Mask to remove similarity with self
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=similarity.device)
        logits = similarity / self.config.temperature
        logits = logits.masked_fill(mask, float("-inf"))

        # For each positive pair, the index of its counterpart
        targets = torch.arange(batch_size, device=similarity.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        loss = nn.functional.cross_entropy(logits, targets)
        return loss


__all__ = [
    "ContrastiveConfig",
    "ContrastivePretrainModule",
]


