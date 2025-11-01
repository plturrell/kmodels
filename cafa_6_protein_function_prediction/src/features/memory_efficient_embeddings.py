"""
Memory-efficient embedding generation using batch processing and memory mapping.

Handles large datasets that don't fit in memory by processing in chunks
and storing results in memory-mapped arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..data.dataset import ProteinSample

LOGGER = logging.getLogger(__name__)


class MemoryMappedEmbeddings:
    """Memory-mapped storage for embeddings."""
    
    def __init__(
        self,
        output_path: Path,
        n_samples: int,
        embedding_dim: int,
        dtype: np.dtype = np.float32,
        mode: str = 'w+',
    ):
        """Initialize memory-mapped embedding storage.
        
        Args:
            output_path: Path to memory-mapped file
            n_samples: Number of samples
            embedding_dim: Embedding dimension
            dtype: Data type for embeddings
            mode: File mode ('w+' for write, 'r' for read)
        """
        self.output_path = output_path
        self.n_samples = n_samples
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        
        # Create memory-mapped array
        self.embeddings = np.memmap(
            output_path,
            dtype=dtype,
            mode=mode,
            shape=(n_samples, embedding_dim),
        )
        
        LOGGER.info(f"Created memory-mapped embeddings: {output_path}")
        LOGGER.info(f"Shape: {self.embeddings.shape}, dtype: {dtype}")
    
    def write_batch(self, start_idx: int, batch_embeddings: np.ndarray):
        """Write a batch of embeddings.
        
        Args:
            start_idx: Starting index for batch
            batch_embeddings: Batch of embeddings to write
        """
        end_idx = start_idx + len(batch_embeddings)
        self.embeddings[start_idx:end_idx] = batch_embeddings
        self.embeddings.flush()  # Ensure written to disk
    
    def read_batch(self, start_idx: int, batch_size: int) -> np.ndarray:
        """Read a batch of embeddings.
        
        Args:
            start_idx: Starting index
            batch_size: Number of embeddings to read
        
        Returns:
            Batch of embeddings
        """
        end_idx = min(start_idx + batch_size, self.n_samples)
        return np.array(self.embeddings[start_idx:end_idx])
    
    def close(self):
        """Close memory-mapped file."""
        del self.embeddings
    
    @classmethod
    def load(cls, path: Path) -> MemoryMappedEmbeddings:
        """Load existing memory-mapped embeddings.
        
        Args:
            path: Path to memory-mapped file
        
        Returns:
            MemoryMappedEmbeddings instance
        """
        # Read shape from file
        temp_mmap = np.memmap(path, dtype=np.float32, mode='r')
        
        # Infer shape (assuming 2D)
        # This is a simplification - in practice, store metadata separately
        LOGGER.warning("Loading memmap without metadata - shape inference may be incorrect")
        
        return cls(
            output_path=path,
            n_samples=temp_mmap.shape[0] if temp_mmap.ndim > 0 else 0,
            embedding_dim=temp_mmap.shape[1] if temp_mmap.ndim > 1 else 0,
            mode='r',
        )


def generate_embeddings_chunked(
    samples: Sequence[ProteinSample],
    embed_fn,
    output_path: Path,
    chunk_size: int = 1000,
    embedding_dim: Optional[int] = None,
) -> MemoryMappedEmbeddings:
    """Generate embeddings in chunks to avoid memory issues.
    
    Args:
        samples: Sequence of protein samples
        embed_fn: Function that takes samples and returns embeddings
        output_path: Path to save memory-mapped embeddings
        chunk_size: Number of samples per chunk
        embedding_dim: Embedding dimension (auto-detected if None)
    
    Returns:
        MemoryMappedEmbeddings instance
    """
    n_samples = len(samples)
    
    # Detect embedding dimension if not provided
    if embedding_dim is None:
        LOGGER.info("Detecting embedding dimension...")
        test_embedding = embed_fn([samples[0]])
        embedding_dim = test_embedding.shape[1]
        LOGGER.info(f"Detected embedding dimension: {embedding_dim}")
    
    # Create memory-mapped storage
    mmap_embeddings = MemoryMappedEmbeddings(
        output_path=output_path,
        n_samples=n_samples,
        embedding_dim=embedding_dim,
    )
    
    # Process in chunks
    LOGGER.info(f"Processing {n_samples} samples in chunks of {chunk_size}")
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk_samples = samples[start_idx:end_idx]
        
        LOGGER.info(f"Processing chunk {start_idx}-{end_idx} ({len(chunk_samples)} samples)")
        
        # Generate embeddings for chunk
        chunk_embeddings = embed_fn(chunk_samples)
        
        # Write to memory-mapped file
        mmap_embeddings.write_batch(start_idx, chunk_embeddings)
        
        # Progress
        progress = (end_idx / n_samples) * 100
        LOGGER.info(f"Progress: {progress:.1f}%")
    
    LOGGER.info("Completed embedding generation")
    return mmap_embeddings


__all__ = [
    "MemoryMappedEmbeddings",
    "generate_embeddings_chunked",
]

