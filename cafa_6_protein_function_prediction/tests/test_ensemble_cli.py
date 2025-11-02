"""Tests for ensemble CLI helper routines."""

import numpy as np
import pytest

from src.modeling.ensemble_cli import _combine_predictions, _matrix_to_predictions


def test_weighted_combination_matches_manual_sum():
    matrices = [
        np.array([[0.2, 0.5], [0.6, 0.4]], dtype=np.float32),
        np.array([[0.1, 0.7], [0.8, 0.9]], dtype=np.float32),
    ]
    weights = [0.3, 0.7]
    combined = _combine_predictions(matrices, "weighted", weights)
    manual = weights[0] * matrices[0] + weights[1] * matrices[1]
    np.testing.assert_allclose(combined, manual)


def test_matrix_roundtrip_to_predictions():
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    proteins = ["P1", "P2"]
    go_terms = ["GO:1", "GO:2"]
    payload = _matrix_to_predictions(matrix, proteins, go_terms)
    assert payload["P1"]["GO:1"] == pytest.approx(0.1)
    assert payload["P2"]["GO:2"] == pytest.approx(0.4)
