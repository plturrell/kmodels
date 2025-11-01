"""Feature engineering utilities for CAFA 6."""

from typing import Optional

import pandas as pd

from .amino_acid import (  # noqa: E402
    AA_ALPHABET,
    CompositionFeatures,
    extract_composition,
    samples_to_dataframe,
)
from .embeddings import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    EmbeddingResult,
    build_embedding_table,
    embed_sequences,
    embeddings_to_dataframe,
)


def build_feature_table(
    samples,
    *,
    include_fractional: bool = True,
    alphabet=AA_ALPHABET,
    include_embeddings: bool = False,
    embedding_kwargs: Optional[dict] = None,
    embedding_table: Optional[pd.DataFrame] = None,
):
    """Combine composition features with optional transformer embeddings."""
    table = samples_to_dataframe(samples, include_fractional=include_fractional, alphabet=alphabet)
    if include_embeddings:
        emb_table = embedding_table
        if emb_table is None:
            kwargs = embedding_kwargs or {}
            emb_table = build_embedding_table(samples, **kwargs)
        table = table.merge(emb_table, on="accession", how="left")
    return table

__all__ = [
    "AA_ALPHABET",
    "CompositionFeatures",
    "DEFAULT_MODEL_NAME",
    "EmbeddingResult",
    "build_embedding_table",
    "build_feature_table",
    "embed_sequences",
    "extract_composition",
    "embeddings_to_dataframe",
    "samples_to_dataframe",
]
