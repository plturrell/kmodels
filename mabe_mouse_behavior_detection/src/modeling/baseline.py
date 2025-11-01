"""Baseline temporal model for mouse behaviour classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PoseBaselineConfig:
    """Configuration for the baseline sequence classifier."""

    input_dim: int
    num_classes: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    pooling: str = "last"  # allowed: "last", "mean"


class PoseSequenceClassifier(nn.Module):
    """GRU-based sequence model operating on flattened pose trajectories."""

    def __init__(self, config: PoseBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        direction_factor = 2 if config.bidirectional else 1
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_dim * direction_factor, config.num_classes)

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return classification logits for the supplied batch of sequences."""
        output, hidden = self.gru(sequence)
        if self.config.pooling == "mean":
            if mask is not None:
                # mask shape: (batch, seq_len)
                mask = mask.unsqueeze(-1)
                lengths = mask.sum(dim=1).clamp_min(1.0)
                pooled = (output * mask).sum(dim=1) / lengths
            else:
                pooled = output.mean(dim=1)
        else:  # "last"
            if self.config.bidirectional:
                forward_last = hidden[-2]
                backward_last = hidden[-1]
                pooled = torch.cat([forward_last, backward_last], dim=-1)
            else:
                pooled = hidden[-1]
        logits = self.head(self.dropout(pooled))
        return logits


def build_pose_baseline(config: PoseBaselineConfig) -> PoseSequenceClassifier:
    """Instantiate the baseline classifier from configuration."""
    return PoseSequenceClassifier(config)


__all__ = ["PoseBaselineConfig", "PoseSequenceClassifier", "build_pose_baseline"]

