"""Feature engineering helpers for amino acid sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.dataset import AMINO_ACIDS, ProteinSample

AA_ALPHABET = tuple(sorted(AMINO_ACIDS))


@dataclass(frozen=True)
class CompositionFeatures:
    """Collection of numeric descriptors derived from a protein sequence."""

    accession: str
    length: int
    composition: np.ndarray

    @property
    def normalised(self) -> np.ndarray:
        """Return the amino acid composition normalised to sum to 1."""
        if self.length == 0:
            return self.composition
        return self.composition / self.length


def _count_amino_acids(sequence: str, alphabet: Sequence[str]) -> np.ndarray:
    counts = np.zeros(len(alphabet), dtype=np.float32)
    for residue in sequence.upper():
        try:
            idx = alphabet.index(residue)
        except ValueError:
            continue
        counts[idx] += 1.0
    return counts


def extract_composition(sample: ProteinSample, *, alphabet: Sequence[str] = AA_ALPHABET) -> CompositionFeatures:
    """Compute amino acid counts for a ``ProteinSample``."""
    counts = _count_amino_acids(sample.sequence, alphabet)
    return CompositionFeatures(accession=sample.accession, length=len(sample.sequence), composition=counts)


def samples_to_dataframe(
    samples: Sequence[ProteinSample],
    *,
    alphabet: Sequence[str] = AA_ALPHABET,
    include_fractional: bool = True,
) -> pd.DataFrame:
    """Convert samples into a feature table.

    ``include_fractional`` toggles whether normalised fractions accompany the
    raw count features.
    """
    records: List[Mapping[str, float]] = []
    for sample in samples:
        features = extract_composition(sample, alphabet=alphabet)
        record: dict[str, float] = {"accession": sample.accession, "length": float(features.length)}
        for idx, residue in enumerate(alphabet):
            record[f"count_{residue}"] = float(features.composition[idx])
        if include_fractional:
            fractions = features.normalised
            for idx, residue in enumerate(alphabet):
                record[f"frac_{residue}"] = float(fractions[idx])
        record["label_count"] = float(len(sample.go_terms))
        records.append(record)
    return pd.DataFrame.from_records(records)


__all__ = [
    "AA_ALPHABET",
    "CompositionFeatures",
    "extract_composition",
    "samples_to_dataframe",
]
