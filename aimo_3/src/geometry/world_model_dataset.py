"""Dataset for training the action-conditioned geometry world model.

This dataset pairs encoded state sequences with action sequences derived
from solver proof traces, suitable for training ActionConditionedGJEPA
in next-step dynamics mode.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from .scene_sequence import SceneTrace
from .scene_encoder import SceneEncoder
from ..modeling.action_encoder import TheoremType


class GeometryWorldModelDataset(Dataset):
    """Dataset yielding (state_sequence, action_sequence) pairs.

    Each item corresponds to one SceneTrace:

    - h_seq: float32 tensor of shape [T, D] with encoded state latents
    - actions: int64 tensor of shape [T] with action indices (TheoremType)

    The world model then learns p(h_{t+1} | h_t, a_t) via next-step loss.
    """

    def __init__(
        self,
        traces: List[SceneTrace],
        encoder: SceneEncoder,
        train_encoder: bool = False,
        cache_latents: bool = True,
    ) -> None:
        super().__init__()
        self.traces = traces
        self.encoder = encoder
        self.train_encoder = train_encoder and not cache_latents
        self.cache_latents = cache_latents
        self._cached_latents: Optional[List[torch.Tensor]] = None
        self._cached_actions: Optional[List[torch.Tensor]] = None

        if self.cache_latents:
            self._precompute_latents_and_actions()

    def _precompute_latents_and_actions(self) -> None:
        """Precompute latent sequences and actions for all traces (CPU)."""
        self.encoder.eval()
        device = next(self.encoder.parameters()).device
        latents: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []

        with torch.no_grad():
            for trace in self.traces:
                h_seq, a_seq = self._encode_single_trace(trace, device=device)
                latents.append(h_seq.cpu())
                actions.append(a_seq.cpu())

        self._cached_latents = latents
        self._cached_actions = actions

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.traces)

    @staticmethod
    def _map_theorem_name_to_action(name: str) -> int:
        """Map theorem name string to a TheoremType id.

        This is a heuristic mapping based on substring matches. Unknown names
        are mapped to UNKNOWN_ACTION.
        """
        n = (name or "").lower()

        if "pythagorean" in n:
            return TheoremType.PYTHAGOREAN.value
        if "anglesum" in n or "angle_sum" in n or "angle sum" in n:
            return TheoremType.TRIANGLE_SUM.value
        if n in {"sss", "congruenttriangles", "congruent_triangles"}:
            return TheoremType.CONGRUENT_TRIANGLES.value

        # Fallback for everything else
        return TheoremType.UNKNOWN_ACTION.value

    def _encode_single_trace(
        self,
        trace: SceneTrace,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single trace to (h_seq, actions)."""
        if device is None:
            device = next(self.encoder.parameters()).device

        states = trace.states
        T = len(states)

        if T == 0:
            # Degenerate: no states
            h_seq = torch.zeros(1, self.encoder.output_dim, device=device)
            actions = torch.tensor(
                [TheoremType.NO_ACTION.value],
                dtype=torch.long,
                device=device,
            )
            return h_seq, actions

        # Encode states
        latents: List[torch.Tensor] = []
        for state in states:
            if self.train_encoder:
                latent = self.encoder.encode_state(state)
            else:
                with torch.no_grad():
                    latent = self.encoder.encode_state(state)
            latents.append(latent.to(device))
        h_seq = torch.stack(latents, dim=0)  # [T, D]

        # Derive actions from state histories
        action_ids: List[int] = []
        for t in range(T):
            if t + 1 < T:
                # Action that led from state_t to state_{t+1}
                next_state = states[t + 1]
                if next_state.history:
                    entry = next_state.history[-1]
                    theorem_name = entry.split("(", 1)[0]
                else:
                    theorem_name = ""
                action_ids.append(self._map_theorem_name_to_action(theorem_name))
            else:
                # Last state: no further action
                action_ids.append(TheoremType.NO_ACTION.value)

        actions = torch.tensor(action_ids, dtype=torch.long, device=device)
        return h_seq, actions

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if self._cached_latents is not None and self._cached_actions is not None:
            return self._cached_latents[idx], self._cached_actions[idx]

        trace = self.traces[idx]
        device = next(self.encoder.parameters()).device
        return self._encode_single_trace(trace, device=device)

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Collate variable-length (h_seq, actions) into padded batches.

        Returns:
            batched_h_seq: [B, T_max, D]
            batched_actions: [B, T_max]
            lengths: list of original lengths
        """
        h_seqs = [item[0] for item in batch]
        action_seqs = [item[1] for item in batch]

        lengths = [seq.shape[0] for seq in h_seqs]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            raise ValueError("Empty batch in GeometryWorldModelDataset.collate_fn")

        D = h_seqs[0].shape[1]
        device = h_seqs[0].device

        padded_h: List[torch.Tensor] = []
        padded_actions: List[torch.Tensor] = []

        for h_seq, a_seq in zip(h_seqs, action_seqs):
            pad_len = max_len - h_seq.shape[0]
            if pad_len > 0:
                h_padding = torch.zeros(pad_len, D, device=device)
                a_padding = torch.full(
                    (pad_len,),
                    TheoremType.NO_ACTION.value,
                    dtype=torch.long,
                    device=device,
                )
                h_seq = torch.cat([h_seq.to(device), h_padding], dim=0)
                a_seq = torch.cat([a_seq.to(device), a_padding], dim=0)
            else:
                h_seq = h_seq.to(device)
                a_seq = a_seq.to(device)

            padded_h.append(h_seq)
            padded_actions.append(a_seq)

        batched_h = torch.stack(padded_h, dim=0)
        batched_actions = torch.stack(padded_actions, dim=0)
        return batched_h, batched_actions, lengths
