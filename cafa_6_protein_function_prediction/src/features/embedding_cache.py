"""
Embedding cache to avoid recomputation of expensive transformer embeddings.

Supports both memory and disk-based caching with automatic invalidation.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ..data.dataset import ProteinSample

LOGGER = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for protein embeddings."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_memory_cache: bool = True,
        use_disk_cache: bool = True,
    ):
        """Initialize embedding cache.
        
        Args:
            cache_dir: Directory for disk cache (default: data/processed/embeddings)
            use_memory_cache: Whether to use in-memory cache
            use_disk_cache: Whether to use disk cache
        """
        self.cache_dir = cache_dir or Path("data/processed/embeddings")
        self.use_memory_cache = use_memory_cache
        self.use_disk_cache = use_disk_cache
        
        if self.use_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._memory_cache: Dict[str, np.ndarray] = {}
    
    def _get_cache_key(
        self,
        sample: ProteinSample,
        model_name: str,
    ) -> str:
        """Generate cache key for a sample and model.
        
        Args:
            sample: Protein sample
            model_name: Model name
        
        Returns:
            Cache key (hash of sequence + model)
        """
        # Create deterministic hash from sequence and model
        content = f"{sample.sequence}|{model_name}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_disk_path(self, cache_key: str) -> Path:
        """Get disk path for cache key."""
        # Use subdirectories to avoid too many files in one directory
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.npy"
    
    def get(
        self,
        sample: ProteinSample,
        model_name: str,
    ) -> Optional[np.ndarray]:
        """Get cached embedding for a sample.
        
        Args:
            sample: Protein sample
            model_name: Model name
        
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(sample, model_name)
        
        # Try memory cache first
        if self.use_memory_cache and cache_key in self._memory_cache:
            LOGGER.debug(f"Memory cache hit for {sample.accession}")
            return self._memory_cache[cache_key]
        
        # Try disk cache
        if self.use_disk_cache:
            disk_path = self._get_disk_path(cache_key)
            if disk_path.exists():
                LOGGER.debug(f"Disk cache hit for {sample.accession}")
                embedding = np.load(disk_path)
                
                # Populate memory cache
                if self.use_memory_cache:
                    self._memory_cache[cache_key] = embedding
                
                return embedding
        
        return None
    
    def put(
        self,
        sample: ProteinSample,
        model_name: str,
        embedding: np.ndarray,
    ) -> None:
        """Store embedding in cache.
        
        Args:
            sample: Protein sample
            model_name: Model name
            embedding: Embedding vector
        """
        cache_key = self._get_cache_key(sample, model_name)
        
        # Store in memory cache
        if self.use_memory_cache:
            self._memory_cache[cache_key] = embedding
        
        # Store in disk cache
        if self.use_disk_cache:
            disk_path = self._get_disk_path(cache_key)
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(disk_path, embedding)
            LOGGER.debug(f"Cached embedding for {sample.accession} to disk")
    
    def get_batch(
        self,
        samples: Sequence[ProteinSample],
        model_name: str,
    ) -> tuple[np.ndarray, list[int]]:
        """Get cached embeddings for a batch of samples.
        
        Args:
            samples: Sequence of protein samples
            model_name: Model name
        
        Returns:
            Tuple of (cached_embeddings, missing_indices)
            - cached_embeddings: Array of cached embeddings (may have None values)
            - missing_indices: Indices of samples that need to be computed
        """
        embeddings = []
        missing_indices = []
        
        for idx, sample in enumerate(samples):
            embedding = self.get(sample, model_name)
            if embedding is None:
                missing_indices.append(idx)
                embeddings.append(None)
            else:
                embeddings.append(embedding)
        
        return embeddings, missing_indices
    
    def clear_memory(self) -> None:
        """Clear memory cache."""
        self._memory_cache.clear()
        LOGGER.info("Cleared memory cache")
    
    def clear_disk(self) -> None:
        """Clear disk cache."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Cleared disk cache")


__all__ = [
    "EmbeddingCache",
]

