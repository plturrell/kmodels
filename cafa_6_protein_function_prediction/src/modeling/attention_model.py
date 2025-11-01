"""
Attention-based neural network for protein function prediction.

Implements multi-head self-attention to capture important regions in embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention weights.
        
        Args:
            x: Input tensor (batch_size, embed_dim)
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = x.shape[0]
        
        # Add sequence dimension if not present
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        out = self.out_proj(out)
        
        # Return mean over sequence dimension
        out = out.mean(dim=1)  # (batch_size, embed_dim)
        attn_weights = attn.mean(dim=1).squeeze()  # (batch_size, seq_len, seq_len) or (seq_len, seq_len)
        
        return out, attn_weights


class AttentionProteinPredictor(nn.Module):
    """Attention-based protein function predictor."""
    
    def __init__(
        self,
        embedding_dim: int,
        num_labels: int,
        num_heads: int = 8,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        
        # Feed-forward layers
        layers = []
        in_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_labels))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """Forward pass.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Logits (batch_size, num_labels) or tuple with attention weights
        """
        # Apply attention
        x_attended, attn_weights = self.attention(x)
        
        # Feed-forward network
        logits = self.network(x_attended)
        
        if return_attention:
            return logits, attn_weights
        return logits


__all__ = [
    "MultiHeadAttention",
    "AttentionProteinPredictor",
]

