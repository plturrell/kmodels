from __future__ import annotations

"""Minimal dataset utilities for G-JEPA training on geometry traces.

This module provides two key pieces:

- A utility function that turns a GeometryTrace into a *sequence* of
  latent vectors using the GeometryStateEncoder.
- A small PyTorch Dataset that yields (h_seq, mask_indices) pairs suitable
  for training a JEPA-style predictor over proof trajectories.

The design is intentionally simple and non-opinionated about batching or
model details; higher-level training code can decide how to collate
variable-length sequences and how to plug them into a Transformer-based
predictor.
"""

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..data.geometry_schema import GeometryTrace
from .gjepa_encoder import GeometryStateEncoder


def encode_trace_to_sequence(
    encoder: GeometryStateEncoder,
    trace: GeometryTrace,
    device: torch.device | None = None,
) -> Tensor:
    """Encode a GeometryTrace into a [T+1, D] tensor of latent vectors.

    This is a thin wrapper around ``GeometryStateEncoder.encode_trace`` that
    returns a detached CPU tensor by default, which is convenient when used
    inside Dataset.__getitem__.
    """

    with torch.no_grad():
        h_seq = encoder.encode_trace(trace, device=device)
    # For simplicity, keep everything on CPU for the Dataset; callers can
    # move batches to GPU in the training loop.
    return h_seq.cpu()


class GeometryJEPADataset(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset yielding (h_seq, mask_indices) pairs for JEPA-style training.

    Each item corresponds to one GeometryTrace:

    - ``h_seq``: float32 tensor of shape [T, D] where T is the number of
      states (typically number of theorem applications + 1).
    - ``mask_indices``: bool tensor of shape [T], indicating which time
      steps should be treated as *masked* (targets) by the JEPA loss.

    Masking strategy is deliberately simple:

    - With probability ``future_mask_prob``, we mask a *single* future step
      chosen uniformly from the latter half of the sequence.
    - Otherwise, we apply independent Bernoulli masking with rate
      ``random_mask_fraction`` and ensure at least one position is masked.
    """

    def __init__(
        self,
        traces: Sequence[GeometryTrace],
        encoder: GeometryStateEncoder,
        random_mask_fraction: float = 0.15,
        future_mask_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self._traces = list(traces)
        self._encoder = encoder
        self.random_mask_fraction = float(random_mask_fraction)
        self.future_mask_prob = float(future_mask_prob)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._traces)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        trace = self._traces[idx]
        h_seq = encode_trace_to_sequence(self._encoder, trace)
        T = h_seq.shape[0]

        if T == 0:
            # Degenerate case: no states. Return empty tensors.
            return h_seq, torch.zeros(0, dtype=torch.bool)

        mask = torch.zeros(T, dtype=torch.bool)

        # Decide whether to use "future" masking or random masking.
        if torch.rand(()) < self.future_mask_prob and T > 1:
            # Mask a single step from the latter half of the trajectory.
            start = max(1, T // 2)
            k = torch.randint(start, T, ()).item()
            mask[k] = True
        else:
            # Random masking with a minimum of 1 masked position.
            num_mask = max(1, int(round(self.random_mask_fraction * T)))
            perm = torch.randperm(T)
            mask[perm[:num_mask]] = True

        return h_seq, mask

