"""Tests for active learning CLI helper logic."""

import numpy as np

from src.utils.active_learning_cli import _predictions_to_array, _align_samples


class DummySample:
    def __init__(self, accession: str):
        self.accession = accession


def test_predictions_to_array_respects_order():
    samples = [DummySample("P1"), DummySample("P2")]  # order matters
    predictions = {
        "P1": {"GO:A": 0.4, "GO:B": 0.6},
        "P2": {"GO:A": 0.1, "GO:B": 0.2},
    }
    aligned, classes = _align_samples([
        DummySample("P1"),
        DummySample("P2"),
    ], predictions)
    assert [sample.accession for sample in aligned] == ["P1", "P2"]
    assert classes == ["GO:A", "GO:B"]

    matrix = _predictions_to_array(samples, classes, predictions)
    np.testing.assert_allclose(matrix, np.array([[0.4, 0.6], [0.1, 0.2]], dtype=np.float32))
