"""
G-JEPA Model: Transformer-based predictor for masked step prediction.

Implements:
- Positional encoding + Transformer encoder
- Predicts ĥ_t at masked positions
- Computes MSE loss between ĥ_t and target h_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from ..geometry.scene_encoder import SceneEncoder


class GJEPA(nn.Module):
    """
    Geometric Joint-Embedding Predictive Architecture.
    
    Takes batched h_seq and mask_indices, predicts masked latents.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        """
        Initialize G-JEPA model.
        
        Args:
            latent_dim: Dimension of input latent vectors [D]
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Input projection: [D] → [hidden_dim]
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Predictor: [hidden_dim] → [D]
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(
        self,
        h_seq: torch.Tensor,
        mask_indices: List[torch.Tensor],
        lengths: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: predict masked latents.
        
        Implementation note: Masking is done via context pooling, not transformer attention.
        The transformer sees all positions, then we:
        1. Pool unmasked (and non-padding) positions to build context
        2. Predict each masked latent from that context
        
        Args:
            h_seq: Batched sequence of latents, shape [batch_size, T+1, D]
            mask_indices: List of mask index tensors, one per batch item
            lengths: Optional list of original sequence lengths (before padding)
            
        Returns:
            (predicted_latents, target_latents) where:
            - predicted_latents: Tensor of shape [total_masked, D]
            - target_latents: Tensor of shape [total_masked, D]
        """
        batch_size, seq_len, D = h_seq.shape
        
        # Project to hidden dimension
        x = self.input_proj(h_seq)  # [batch_size, T+1, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Encode with transformer (no masking in attention mechanism)
        # JEPA objective is implemented via context pooling below
        encoded = self.transformer(x)  # [batch_size, T+1, hidden_dim]
        
        # Extract predictions for masked positions
        predicted_latents = []
        target_latents = []
        
        for b, mask_idx in enumerate(mask_indices):
            if len(mask_idx) > 0:
                # Convert mask_idx to tensor if needed
                if isinstance(mask_idx, torch.Tensor):
                    mask_idx_tensor = mask_idx
                else:
                    mask_idx_tensor = torch.tensor(mask_idx, dtype=torch.long, device=x.device)
                
                # Determine valid length (exclude padding)
                valid_len = lengths[b] if lengths is not None else seq_len
                
                # Create vectorized boolean masks
                # True for positions that are masked (targets)
                mask_bool = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
                mask_bool[mask_idx_tensor] = True
                
                # True for positions that are padding
                padding_bool = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
                if valid_len < seq_len:
                    padding_bool[valid_len:] = True
                
                # Context comes from: valid AND unmasked positions
                context_bool = ~mask_bool & ~padding_bool
                unmasked_idx = context_bool.nonzero(as_tuple=False).squeeze(-1)
                
                if len(unmasked_idx) > 0:
                    # Use mean of unmasked, non-padding positions as context
                    context = encoded[b, unmasked_idx].mean(dim=0)  # [hidden_dim]
                else:
                    # Fallback: use mean of all valid (non-padding) positions
                    valid_idx = (~padding_bool).nonzero(as_tuple=False).squeeze(-1)
                    if len(valid_idx) > 0:
                        context = encoded[b, valid_idx].mean(dim=0)
                    else:
                        context = encoded[b].mean(dim=0)  # Last resort
                
                # Predict for each masked position
                for idx in mask_idx_tensor:
                    idx_int = idx.item() if isinstance(idx, torch.Tensor) else int(idx)
                    # Predict latent at this position
                    predicted = self.predictor(context)  # [D]
                    predicted_latents.append(predicted)
                    
                    # Target is the actual latent at this position
                    target = h_seq[b, idx_int]  # [D]
                    target_latents.append(target)
        
        if not predicted_latents:
            # No masked positions, return dummy tensors
            dummy = torch.zeros(1, self.latent_dim, device=h_seq.device)
            return dummy, dummy
        
        predicted = torch.stack(predicted_latents)  # [total_masked, D]
        targets = torch.stack(target_latents)  # [total_masked, D]
        
        return predicted, targets
    
    def compute_loss(
        self,
        h_seq: torch.Tensor,
        mask_indices: List[torch.Tensor],
        lengths: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss for masked prediction.
        
        Args:
            h_seq: Batched sequence of latents, shape [batch_size, T+1, D]
            mask_indices: List of mask index tensors
            lengths: Optional list of original sequence lengths (before padding)
            
        Returns:
            MSE loss scalar
        """
        predicted, targets = self.forward(h_seq, mask_indices, lengths)
        
        # MSE loss
        loss = F.mse_loss(predicted, targets)
        
        return loss


def create_gjepa_model(
    latent_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    max_seq_len: int = 100,
) -> GJEPA:
    """
    Factory function to create G-JEPA model.
    
    Args:
        latent_dim: Latent dimension [D]
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
        
    Returns:
        Initialized GJEPA model
    """
    return GJEPA(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    )

