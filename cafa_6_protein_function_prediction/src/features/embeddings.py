"""Sequence embedding helpers using Hugging Face transformer models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.dataset import ProteinSample

try:  # Optional heavy dependencies imported lazily.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


@dataclass(frozen=True)
class EmbeddingResult:
    """Mean-pooled embedding for a single protein sequence."""

    accession: str
    vector: np.ndarray


def _ensure_dependencies() -> None:
    if torch is None:
        raise ImportError(
            "torch is required for transformer embeddings. Install it via `pip install torch`."
        )
    if AutoTokenizer is None or AutoModel is None:
        raise ImportError(
            "transformers is required for embedding extraction. Install it via `pip install transformers`."
        )


def _resolve_device(device: Optional[str]) -> "torch.device":
    assert torch is not None  # For type checkers.
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_name: str, device: Optional[str]) -> Tuple["torch.nn.Module", "torch.device", object]:
    _ensure_dependencies()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    torch_device = _resolve_device(device)
    model.to(torch_device)
    model.eval()
    return model, torch_device, tokenizer


def _batch_iterable(items: Sequence[ProteinSample], batch_size: int) -> Iterable[Sequence[ProteinSample]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_sequences(
    samples: Sequence[ProteinSample],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
    device: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    return_array: bool = False,
    cache: Optional[object] = None,
) -> List[EmbeddingResult] | np.ndarray:
    """Return mean-pooled transformer embeddings for the given samples.

    Args:
        samples: Sequence of protein samples to embed
        model_name: Hugging Face model name
        batch_size: Batch size for inference
        device: Device to use (cuda/cpu)
        progress: Optional progress callback
        return_array: If True, return np.ndarray instead of List[EmbeddingResult]
        cache: Optional EmbeddingCache object for caching

    Returns:
        List of EmbeddingResult objects or np.ndarray of shape (n_samples, embedding_dim)
    """
    if not samples:
        return np.array([]) if return_array else []

    # Check cache if available
    if cache is not None:
        cached_embeddings, missing_indices = cache.get_batch(samples, model_name)
        if not missing_indices:
            # All embeddings cached
            LOGGER.info("All %d embeddings found in cache", len(samples))
            results = [
                EmbeddingResult(accession=sample.accession, vector=emb)
                for sample, emb in zip(samples, cached_embeddings)
            ]
            if return_array:
                return np.array([r.vector for r in results])
            return results
        else:
            LOGGER.info("Found %d/%d embeddings in cache, computing %d",
                       len(samples) - len(missing_indices), len(samples), len(missing_indices))
    else:
        cached_embeddings = [None] * len(samples)
        missing_indices = list(range(len(samples)))

    # Load model only if needed
    model, torch_device, tokenizer = _load_model(model_name, device)
    assert torch is not None

    results: List[EmbeddingResult] = [None] * len(samples)
    total = len(samples)

    # Compute missing embeddings
    samples_to_compute = [samples[i] for i in missing_indices]
    inference_ctx = getattr(torch, "inference_mode", torch.no_grad)

    computed_idx = 0
    with inference_ctx():
        for idx, batch in enumerate(_batch_iterable(samples_to_compute, batch_size)):
            tokenised = tokenizer(
                [sample.sequence for sample in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            tokenised = {key: value.to(torch_device) for key, value in tokenised.items()}
            outputs = model(**tokenised)
            hidden = outputs.last_hidden_state
            mask = tokenised["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = (summed / lengths).cpu().numpy().astype(np.float32)

            for sample, vector in zip(batch, pooled):
                result = EmbeddingResult(accession=sample.accession, vector=vector)
                original_idx = missing_indices[computed_idx]
                results[original_idx] = result

                # Cache the result
                if cache is not None:
                    cache.put(sample, model_name, vector)

                computed_idx += 1

            if progress is not None:
                progress(min((idx + 1) * batch_size, len(samples_to_compute)), len(samples_to_compute))

    # Fill in cached results
    for idx, cached_emb in enumerate(cached_embeddings):
        if cached_emb is not None and results[idx] is None:
            results[idx] = EmbeddingResult(accession=samples[idx].accession, vector=cached_emb)

    if return_array:
        return np.array([r.vector for r in results])
    return results


def embeddings_to_dataframe(embeddings: Sequence[EmbeddingResult]) -> pd.DataFrame:
    """Convert embedding results into a wide pandas DataFrame."""
    if not embeddings:
        return pd.DataFrame(columns=["accession"])
    width = len(embeddings[0].vector)
    records = []
    for item in embeddings:
        record = {"accession": item.accession}
        for idx, value in enumerate(item.vector):
            record[f"emb_{idx}"] = float(value)
        records.append(record)
    df = pd.DataFrame.from_records(records)
    expected_columns = {f"emb_{idx}" for idx in range(width)} | {"accession"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Embedding DataFrame missing expected columns: {sorted(missing)}")
    return df


def build_embedding_table(
    samples: Sequence[ProteinSample],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
    device: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """Convenience helper that combines embedding extraction and DataFrame creation."""
    embeddings = embed_sequences(
        samples,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        progress=progress,
    )
    return embeddings_to_dataframe(embeddings)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "EmbeddingResult",
    "build_embedding_table",
    "embed_sequences",
    "embeddings_to_dataframe",
]
