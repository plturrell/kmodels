"""Utility helpers for assembling modeling components."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def _extract_backbone_state_dict(state_dict: dict) -> dict:
    prefix = "backbone."
    extracted = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    return extracted


def load_contrastive_encoder_weights(model, checkpoint_path: Path) -> Tuple[list, list]:
    """Load encoder weights from a contrastive pretraining checkpoint.

    Args:
        model: Instance of :class:`PretrainedForgeryModel` (or compatible) whose
            ``segmentation_model.encoder`` attribute should receive the weights.
        checkpoint_path: Path to a Lightning checkpoint produced by
            ``train_contrastive``.

    Returns:
        A tuple of (missing_keys, unexpected_keys) reported by
        ``load_state_dict`` for logging/debugging.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = _extract_backbone_state_dict(state_dict)

    encoder = getattr(model, "segmentation_model", model).encoder
    load_result = encoder.load_state_dict(encoder_state, strict=False)
    if isinstance(load_result, tuple):
        missing, unexpected = load_result
    else:
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))

    if missing:
        print(
            f"[contrastive] encoder missing {len(missing)} parameters after load: {missing[:5]}"
        )
    if unexpected:
        print(
            f"[contrastive] encoder unexpected {len(unexpected)} parameters: {unexpected[:5]}"
        )
    return missing, unexpected


__all__ = ["load_contrastive_encoder_weights"]


