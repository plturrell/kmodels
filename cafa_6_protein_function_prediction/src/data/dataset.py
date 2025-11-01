"""Dataset utilities for the CAFA 6 Protein Function Prediction competition.

The official dataset ships protein sequences in FASTA format along with GO-term
annotations in TSV/CSV companions. The helpers here aim to keep loading logic
lightweight and dependency-free while providing a familiar interface shared by
other competition workspaces in this repository.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # Optional dependency: only import torch when available.
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    Dataset = object  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class ProteinSample:
    """Container encapsulating a protein sequence and associated metadata."""

    accession: str
    sequence: str
    go_terms: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        sequence = self.sequence.replace("\n", "").strip()
        if not sequence:
            raise ValueError(f"Sequence for {self.accession} is empty.")
        object.__setattr__(self, "sequence", sequence)
        invalid = {residue for residue in sequence.upper() if residue not in AMINO_ACIDS}
        if invalid:
            object.__setattr__(self, "metadata", {**self.metadata, "contains_non_standard": "true"})


def _iter_fasta_records(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield ``(header, sequence)`` tuples from a FASTA file."""
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    header: Optional[str] = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                    chunks.clear()
                header = line[1:].strip()
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def load_sequences_from_fasta(path: Path, extract_uniprot_id: bool = True) -> Dict[str, str]:
    """Return a mapping of sequence accession → amino acid string.

    Args:
        path: Path to FASTA file
        extract_uniprot_id: If True, extract UniProt accession from headers like
                           'sp|Q5W0B1|OBI1_HUMAN' -> 'Q5W0B1'
    """
    records = {}
    for header, sequence in _iter_fasta_records(path):
        if not header:
            raise ValueError(f"Encountered record without header in {path}.")
        accession = header.split()[0]

        # Extract UniProt accession from pipe-delimited format
        if extract_uniprot_id and "|" in accession:
            parts = accession.split("|")
            if len(parts) >= 2:
                accession = parts[1]  # Extract middle part (e.g., Q5W0B1)

        if accession in records:
            raise ValueError(f"Duplicate accession '{accession}' detected in {path}.")
        records[accession] = sequence
    if not records:
        raise RuntimeError(f"No sequences parsed from {path}.")
    return records


def load_go_terms(
    path: Path,
    *,
    id_column: str = "protein_id",
    term_column: str = "go_terms",
    delimiter: Optional[str] = None,
    term_separator: str = " ",
) -> Dict[str, Tuple[str, ...]]:
    """Load GO annotations from a CSV/TSV file.

    ``term_separator`` controls how the GO-term string is split (default: space
    separated list). Empty annotations produce empty tuples.

    DEPRECATED: Use load_go_terms_long_format for CAFA 6 data format.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    annotations: Dict[str, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader_kwargs = {"delimiter": delimiter} if delimiter else {}
        reader = csv.DictReader(handle, **reader_kwargs)
        for row in reader:
            if row.get(id_column) in (None, ""):
                raise KeyError(f"Row missing identifier column '{id_column}'.")
            accession = str(row[id_column])
            raw_terms = str(row.get(term_column, ""))
            terms = tuple(sorted({term.strip() for term in raw_terms.split(term_separator) if term.strip()}))
            annotations[accession] = terms
    return annotations


def load_go_terms_long_format(
    path: Path,
    *,
    id_column: str = "EntryID",
    term_column: str = "term",
    aspect_column: Optional[str] = "aspect",
    delimiter: str = "\t",
    filter_aspect: Optional[str] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Load GO annotations from a long-format TSV file (one row per term).

    This is the format used in CAFA 6 competition data where each protein-term
    pair is on a separate row.

    Args:
        path: Path to the TSV file
        id_column: Column name for protein identifiers (default: "EntryID")
        term_column: Column name for GO terms (default: "term")
        aspect_column: Column name for GO aspect (C/F/P), optional
        delimiter: Field delimiter (default: tab)
        filter_aspect: If provided, only include terms with this aspect (C/F/P)

    Returns:
        Dictionary mapping protein accession to tuple of GO terms
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    annotations: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            if row.get(id_column) in (None, ""):
                raise KeyError(f"Row missing identifier column '{id_column}'.")
            accession = str(row[id_column])
            term = str(row.get(term_column, "")).strip()

            if not term or not term.startswith("GO:"):
                continue  # Skip empty or non-GO terms

            # Filter by aspect if requested
            if filter_aspect and aspect_column:
                aspect = str(row.get(aspect_column, "")).strip()
                if aspect != filter_aspect:
                    continue

            if accession not in annotations:
                annotations[accession] = []
            annotations[accession].append(term)

    # Convert to tuples and sort
    return {k: tuple(sorted(set(v))) for k, v in annotations.items()}


def load_split_from_json(path: Path) -> Dict[str, str]:
    """Read a JSON mapping ``accession → split`` if one ships with the data."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, MutableMapping):
        raise ValueError("Split file must be a JSON object mapping accession to split name.")
    return {str(key): str(value) for key, value in payload.items()}


def build_samples(
    sequence_map: Mapping[str, str],
    annotations: Optional[Mapping[str, Sequence[str]]] = None,
    metadata: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> List[ProteinSample]:
    """Combine sequences, annotations, and optional metadata into samples."""
    samples: List[ProteinSample] = []
    annotations = annotations or {}
    metadata = metadata or {}
    for accession, sequence in sequence_map.items():
        terms = tuple(sorted(map(str, annotations.get(accession, ()))))
        meta = dict(metadata.get(accession, {}))
        samples.append(ProteinSample(accession=accession, sequence=sequence, go_terms=terms, metadata=meta))
    return samples


def train_val_split(
    samples: Sequence[ProteinSample],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[ProteinSample], List[ProteinSample]]:
    """Split samples into train/validation folds.

    Stratification groups proteins by the count of annotated GO terms to maintain
    label density across folds.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    samples = list(samples)
    if not samples:
        raise ValueError("No samples provided for splitting.")

    rng = random.Random(seed)
    if stratify:
        buckets: Dict[int, List[ProteinSample]] = {}
        for sample in samples:
            buckets.setdefault(len(sample.go_terms), []).append(sample)
        train: List[ProteinSample] = []
        val: List[ProteinSample] = []
        for bucket in buckets.values():
            rng.shuffle(bucket)
            split_idx = max(1, int(len(bucket) * (1 - val_fraction))) if len(bucket) > 1 else int(len(bucket) * (1 - val_fraction))
            train.extend(bucket[:split_idx])
            val.extend(bucket[split_idx:])
    else:
        shuffled = samples[:]
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - val_fraction))
        train = shuffled[:split_idx]
        val = shuffled[split_idx:]

    return train, val


class ProteinDataset(Dataset):
    """Minimal torch-compatible dataset wrapper around ``ProteinSample`` items."""

    def __init__(self, samples: Sequence[ProteinSample]):
        if Dataset is object:  # pragma: no cover - guard when torch missing
            raise ImportError("torch is required to use ProteinDataset.")
        self._samples = list(samples)
        if not self._samples:
            raise ValueError("ProteinDataset received an empty sample list.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ProteinSample:
        return self._samples[idx]


def create_dataloaders(
    train_samples: Sequence[ProteinSample],
    val_samples: Sequence[ProteinSample],
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Instantiate PyTorch dataloaders for train/validation splits."""
    if DataLoader is None:  # pragma: no cover - guard when torch missing
        raise ImportError("torch is required to create dataloaders.")

    train_ds = ProteinDataset(train_samples)
    val_ds = ProteinDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


__all__ = [
    "AMINO_ACIDS",
    "ProteinDataset",
    "ProteinSample",
    "build_samples",
    "create_dataloaders",
    "load_go_terms",
    "load_go_terms_long_format",
    "load_sequences_from_fasta",
    "load_split_from_json",
    "train_val_split",
]
