"""Lightweight tests for interpretability CLI helpers."""

import numpy as np

from src.utils.interpretability_cli import _select_indices


def test_select_indices_prefers_high_scores():
    probs = np.array([[0.1, 0.2], [0.8, 0.9], [0.5, 0.4]])
    selected = _select_indices(probs, top_k=1)
    assert selected == [1]

    explicit = _select_indices(probs, top_k=1, explicit=[2, 2, 0])
    assert explicit == [2, 0]
