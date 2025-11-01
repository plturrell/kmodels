"""Relational encoders for player interaction modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_deltas(positions: torch.Tensor) -> torch.Tensor:
    return positions.unsqueeze(2) - positions.unsqueeze(1)


def _pairwise_distances(deltas: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt((deltas**2).sum(-1) + eps)


class InteractionGNNLayer(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.scale = math.sqrt(hidden_dim // heads)
        self.query = nn.Linear(node_dim, hidden_dim, bias=False)
        self.key = nn.Linear(node_dim, hidden_dim, bias=False)
        self.value = nn.Linear(node_dim, hidden_dim, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_feats: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, num_nodes, _ = node_feats.shape
        q = self.query(node_feats).view(batch, num_nodes, self.heads, -1).transpose(1, 2)
        k = self.key(node_feats).view(batch, num_nodes, self.heads, -1).transpose(1, 2)
        v = self.value(node_feats).view(batch, num_nodes, self.heads, -1).transpose(1, 2)

        deltas = _pairwise_deltas(positions)
        distances = _pairwise_distances(deltas).unsqueeze(-1)
        direction = F.normalize(deltas, dim=-1, eps=1e-6)
        edge_geom = torch.cat([direction, distances], dim=-1)
        edge_bias = self.edge_mlp(edge_geom).view(batch, num_nodes, num_nodes, self.heads, -1)
        edge_bias = edge_bias.permute(0, 3, 1, 2, 4)

        attn_scores = torch.einsum('bhid,bhjd->bhij', q, k) / self.scale
        attn_scores = attn_scores + edge_bias.mean(-1)

        query_mask = None
        valid_key_rows = torch.ones_like(attn_scores[..., :1], dtype=torch.bool)
        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            finfo = torch.finfo(attn_scores.dtype)
            attn_scores = attn_scores.masked_fill(~key_mask, finfo.min)
            valid_key_rows = key_mask.any(dim=-1, keepdim=True)
            query_mask = mask.unsqueeze(1).unsqueeze(3)

        max_scores = attn_scores.max(dim=-1, keepdim=True).values
        max_scores = torch.where(valid_key_rows, max_scores, torch.zeros_like(max_scores))
        attn_scores = attn_scores - max_scores
        attn_scores = torch.where(valid_key_rows, attn_scores, torch.zeros_like(attn_scores))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.where(valid_key_rows, attn_weights, torch.zeros_like(attn_weights))
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)

        if query_mask is not None:
            attn_weights = attn_weights * query_mask.float()

        attn_weights = self.dropout(attn_weights)
        agg = torch.einsum('bhij,bhjd->bhid', attn_weights, v)
        agg = agg.transpose(1, 2).contiguous().view(batch, num_nodes, -1)
        agg = self.out_proj(agg)

        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()
            agg = agg * mask_float
            node_feats = node_feats * mask_float

        return self.norm(node_feats + agg)


@dataclass
class RelationalEncoderConfig:
    node_dim: int
    hidden_dim: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.1


class RelationalGNNEncoder(nn.Module):
    def __init__(self, config: RelationalEncoderConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                InteractionGNNLayer(
                    node_dim=config.node_dim,
                    hidden_dim=config.hidden_dim,
                    heads=config.heads,
                    dropout=config.dropout,
                )
                for _ in range(config.layers)
            ]
        )

    def forward(self, node_feats: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            node_feats = layer(node_feats, positions, mask=mask)
        return node_feats


class TemporalAggregation(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = nn.Parameter(torch.randn(1, 128, dim))

    def forward(self, sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = sequence.size(1)
        encoded = sequence + self.positional[:, :seq_len, :]
        return self.encoder(encoded, src_key_padding_mask=mask)


class RelationalTrajectoryModel(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        node_hidden_dim: int,
        target_dim: int = 2,
        *,
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_cfg = RelationalEncoderConfig(
            node_dim=node_hidden_dim,
            hidden_dim=node_hidden_dim,
            layers=gnn_layers,
            heads=gnn_heads,
            dropout=dropout,
        )
        self.node_proj = nn.Linear(node_input_dim, node_hidden_dim)
        self.relational = RelationalGNNEncoder(encoder_cfg)
        self.temporal = TemporalAggregation(node_hidden_dim, num_layers=temporal_layers, heads=temporal_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(node_hidden_dim),
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_hidden_dim, target_dim),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, _ = node_feats.shape
        node_feats = self.node_proj(node_feats)
        relational_outputs = []
        for t in range(T):
            relational = self.relational(
                node_feats[:, t],
                positions[:, t],
                mask=mask[:, t] if mask is not None else None,
            )
            relational_outputs.append(relational)
        relational_seq = torch.stack(relational_outputs, dim=1)
        pooled = relational_seq.mean(dim=2)
        temporal_encoded = self.temporal(pooled)
        predictions = self.head(temporal_encoded)
        return predictions, relational_seq
