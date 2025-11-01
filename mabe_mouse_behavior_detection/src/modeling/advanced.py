"""Advanced temporal models for mouse behavior classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    """Configuration for Transformer-based classifier."""
    input_dim: int
    num_classes: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.2
    pooling: str = "cls"  # "cls", "mean", "max"


class TransformerSequenceClassifier(nn.Module):
    """Transformer encoder for sequence classification."""
    
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model, dropout=config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, config.num_classes)
        
        # CLS token for pooling
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
    
    def forward(self, sequence: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            sequence: (batch, seq_len, input_dim)
            mask: (batch, seq_len) boolean mask (True = valid)
        
        Returns:
            logits: (batch, num_classes)
        """
        batch_size = sequence.size(0)
        
        # Project input
        x = self.input_proj(sequence)  # (batch, seq_len, d_model)
        
        # Add CLS token if using cls pooling
        if self.config.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask (Transformer expects inverted mask)
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert: True = masked
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Pooling
        if self.config.pooling == "cls":
            pooled = x[:, 0]  # CLS token
        elif self.config.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
            else:
                pooled = x.mean(dim=1)
        elif self.config.pooling == "max":
            pooled = x.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")
        
        # Classification
        logits = self.head(self.dropout(pooled))
        return logits


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


@dataclass
class LSTMConfig:
    """Configuration for LSTM-based classifier."""
    input_dim: int
    num_classes: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    pooling: str = "last"


class LSTMSequenceClassifier(nn.Module):
    """LSTM-based sequence classifier."""
    
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
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
    
    def forward(self, sequence: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass."""
        output, (hidden, cell) = self.lstm(sequence)
        
        if self.config.pooling == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                pooled = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1.0)
            else:
                pooled = output.mean(dim=1)
        else:  # "last"
            if self.config.bidirectional:
                pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1)
            else:
                pooled = hidden[-1]
        
        logits = self.head(self.dropout(pooled))
        return logits


@dataclass
class TCNConfig:
    """Configuration for Temporal Convolutional Network."""
    input_dim: int
    num_classes: int
    num_channels: list[int] = None  # e.g., [64, 128, 256]
    kernel_size: int = 3
    dropout: float = 0.2

    def __post_init__(self):
        if self.num_channels is None:
            self.num_channels = [64, 128, 256]


class TemporalBlock(nn.Module):
    """Temporal convolutional block with residual connection."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal padding."""
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Remove future padding
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]  # Remove future padding
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNSequenceClassifier(nn.Module):
    """Temporal Convolutional Network for sequence classification."""

    def __init__(self, config: TCNConfig) -> None:
        super().__init__()
        self.config = config

        layers = []
        num_levels = len(config.num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = config.input_dim if i == 0 else config.num_channels[i - 1]
            out_channels = config.num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    dilation,
                    config.dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.num_channels[-1], config.num_classes)

    def forward(self, sequence: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            sequence: (batch, seq_len, input_dim)
            mask: (batch, seq_len) boolean mask

        Returns:
            logits: (batch, num_classes)
        """
        # TCN expects (batch, channels, seq_len)
        x = sequence.transpose(1, 2)

        # Apply TCN
        x = self.network(x)

        # Global pooling (batch, channels, seq_len) -> (batch, channels)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # (batch, 1, seq_len)
            x = (x * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2).clamp_min(1.0)
        else:
            x = x.mean(dim=2)

        # Classification
        logits = self.head(self.dropout(x))
        return logits


__all__ = [
    "TransformerConfig",
    "TransformerSequenceClassifier",
    "LSTMConfig",
    "LSTMSequenceClassifier",
    "TCNConfig",
    "TCNSequenceClassifier",
]

