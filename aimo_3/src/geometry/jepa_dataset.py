"""
Geometry JEPA Dataset for training.

Provides GeometryJEPADataset(traces, encoder) → (h_seq, mask_indices)
suitable for JEPA loss computation.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import random

from .scene_sequence import SceneTrace
from .scene_encoder import SceneEncoder


class GeometryJEPADataset(Dataset):
    """
    Dataset for G-JEPA training.
    
    Takes traces and encoder, returns (h_seq, mask_indices) suitable for JEPA loss.
    """
    
    def __init__(
        self,
        traces: List[SceneTrace],
        encoder: SceneEncoder,
        mask_ratio: float = 0.3,
        min_mask: int = 1,
        max_mask: Optional[int] = None,
        train_encoder: bool = True,
        cache_latents: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            traces: List of SceneTrace objects
            encoder: SceneEncoder to encode scenes
            mask_ratio: Ratio of steps to mask (0.0-1.0)
            min_mask: Minimum number of steps to mask
            max_mask: Maximum number of steps to mask (None = no limit)
            train_encoder: If True, encoder gradients flow through dataset (joint training).
                          If False, encoding is done with torch.no_grad() (frozen encoder).
            cache_latents: If True, precompute all latent sequences once in __init__.
                          Trades memory for speed. Automatically sets train_encoder=False.
        """
        self.traces = traces
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.train_encoder = train_encoder and not cache_latents  # Caching implies frozen encoder
        self.cache_latents = cache_latents
        self._cached_latents: Optional[List[torch.Tensor]] = None
        
        # Precompute latents if caching enabled
        if self.cache_latents:
            print(f"Precomputing latent sequences for {len(traces)} traces...")
            self._cached_latents = []
            with torch.no_grad():
                for i, trace in enumerate(traces):
                    if (i + 1) % 100 == 0:
                        print(f"  Encoded {i + 1}/{len(traces)} traces")
                    h_seq = self.encoder.encode_trace(trace)
                    self._cached_latents.append(h_seq.cpu())  # Store on CPU to save GPU memory
            print(f"✓ Precomputed {len(self._cached_latents)} latent sequences")
    
    def __len__(self) -> int:
        return len(self.traces)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item: (h_seq, mask_indices).
        
        Args:
            idx: Index of trace
            
        Returns:
            (h_seq, mask_indices) where:
            - h_seq: Tensor of shape [T+1, D] with encoded latent vectors
            - mask_indices: Tensor of shape [num_masked] with indices to mask
        """
        # Get latent sequence (from cache or encode on-the-fly)
        if self._cached_latents is not None:
            h_seq = self._cached_latents[idx]
        else:
            trace = self.traces[idx]
            # Encode trace to sequence of latent vectors
            if self.train_encoder:
                # Joint training: allow gradients to flow into encoder
                h_seq = self.encoder.encode_trace(trace)
            else:
                # Frozen encoder: no gradients
                with torch.no_grad():
                    h_seq = self.encoder.encode_trace(trace)
        
        # Generate mask indices
        T = h_seq.shape[0]  # Sequence length
        num_to_mask = max(
            self.min_mask,
            min(
                int(T * self.mask_ratio),
                self.max_mask if self.max_mask else T
            )
        )
        
        # Randomly select indices to mask
        if T > 0:
            mask_indices = torch.tensor(
                random.sample(range(T), min(num_to_mask, T)),
                dtype=torch.long
            )
        else:
            mask_indices = torch.tensor([], dtype=torch.long)
        
        return h_seq, mask_indices
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of (h_seq, mask_indices) tuples
            
        Returns:
            (batched_h_seq, list_of_mask_indices, lengths) where:
            - batched_h_seq: Padded tensor of shape [batch_size, max_T+1, D]
            - list_of_mask_indices: List of mask index tensors (variable length)
            - lengths: List of original sequence lengths (before padding)
        """
        h_seqs = [item[0] for item in batch]
        mask_indices_list = [item[1] for item in batch]
        
        # Record original lengths before padding
        lengths = [seq.shape[0] for seq in h_seqs]
        
        # Pad sequences to same length
        max_len = max(lengths)
        D = h_seqs[0].shape[1]
        
        padded_seqs = []
        for seq in h_seqs:
            if seq.shape[0] < max_len:
                padding = torch.zeros(max_len - seq.shape[0], D)
                padded = torch.cat([seq, padding], dim=0)
            else:
                padded = seq
            padded_seqs.append(padded)
        
        batched_h_seq = torch.stack(padded_seqs)  # Shape: [batch_size, max_T+1, D]
        
        return batched_h_seq, mask_indices_list, lengths
