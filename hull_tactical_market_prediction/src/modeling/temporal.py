"""Advanced temporal models for financial time-series prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    """Configuration for LSTM-based forecaster."""
    input_dim: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    output_dim: int = 1


class LSTMForecaster(nn.Module):
    """LSTM-based time-series forecaster for financial returns."""
    
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
        self.head = nn.Linear(config.hidden_dim * direction_factor, config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim) for single timestep
        
        Returns:
            predictions: (batch, output_dim)
        """
        # Handle single timestep input
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.config.bidirectional:
            # Concatenate forward and backward hidden states
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]
        
        # Prediction head
        return self.head(self.dropout(last_hidden))


@dataclass
class TransformerConfig:
    """Configuration for Transformer-based forecaster."""
    input_dim: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.2
    output_dim: int = 1


class TransformerForecaster(nn.Module):
    """Transformer encoder for financial time-series forecasting."""
    
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
        
        # Output head
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, config.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
        
        Returns:
            predictions: (batch, output_dim)
        """
        # Handle single timestep
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use last timestep for prediction
        last_output = x[:, -1, :]
        
        return self.head(self.dropout(last_output))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


__all__ = [
    "LSTMConfig",
    "LSTMForecaster",
    "TransformerConfig",
    "TransformerForecaster",
]

