from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    diff = preds - targets
    metrics: Dict[str, float] = {
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt((diff**2).mean()).item(),
    }
    if target_names is not None:
        for idx, name in enumerate(target_names):
            col_diff = diff[:, idx]
            metrics[f"mae_{name}"] = col_diff.abs().mean().item()
            metrics[f"rmse_{name}"] = torch.sqrt((col_diff**2).mean()).item()
    return metrics

