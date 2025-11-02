"""Physics-guided neural architecture for interpretable forgery detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class PhysicsGuidedConfig:
    feature_dim: int
    num_classes: int = 2
    hidden_dim: int = 128
    mask_gate_bias: float = 0.5


class PhysicsGuidedForgeryModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        config: PhysicsGuidedConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config

        self.feature_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.class_fusion = nn.Linear(config.hidden_dim, config.num_classes)
        self.mask_gate = nn.Linear(config.hidden_dim, 1)

        nn.init.constant_(self.mask_gate.bias, config.mask_gate_bias)

    def forward(
        self,
        image: torch.Tensor,
        physics_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        class_logits, mask_logits = self.base_model(image)

        if physics_features is None:
            physics_features = torch.zeros(
                image.size(0),
                self.config.feature_dim,
                device=image.device,
                dtype=image.dtype,
            )

        physics_emb = self.feature_encoder(physics_features)
        class_adjustment = self.class_fusion(physics_emb)
        gated_logits = class_logits + class_adjustment

        gate = torch.sigmoid(self.mask_gate(physics_emb)).view(-1, 1, 1, 1)
        gated_mask = mask_logits * gate

        return gated_logits, gated_mask


__all__ = ["PhysicsGuidedConfig", "PhysicsGuidedForgeryModel"]


