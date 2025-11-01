"""Walk-forward cross-validation for time-series models."""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


def walk_forward_split(
    n_samples: int,
    n_splits: int = 5,
    test_size: int = None,
    gap: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward cross-validation split for time-series.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of splits
        test_size: Size of test set (if None, computed from n_splits)
        gap: Embargo period between train and test (to avoid leakage)
    
    Yields:
        Tuples of (train_indices, test_indices)
    """
    if test_size is None:
        test_size = n_samples // (n_splits + 1)
    
    for i in range(n_splits):
        # Test set: sliding window
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
        
        # Train set: all data before test (with gap)
        train_end = test_start - gap
        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)
        
        yield train_indices, test_indices


def expanding_window_split(
    n_samples: int,
    n_splits: int = 5,
    min_train_size: int = None,
    test_size: int = None,
    gap: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Expanding window cross-validation (anchored walk-forward).
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of splits
        min_train_size: Minimum training set size
        test_size: Size of test set
        gap: Embargo period
    
    Yields:
        Tuples of (train_indices, test_indices)
    """
    if test_size is None:
        test_size = n_samples // (n_splits + 1)
    
    if min_train_size is None:
        min_train_size = test_size
    
    for i in range(n_splits):
        test_start = min_train_size + i * test_size
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
        
        # Train set: from start to test (with gap)
        train_end = test_start - gap
        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)
        
        yield train_indices, test_indices


def purged_kfold_split(
    n_samples: int,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Purged K-Fold for time-series (from Advances in Financial ML).
    
    Removes samples close to test set to avoid leakage.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of splits
        embargo_pct: Percentage of samples to embargo after test set
    
    Yields:
        Tuples of (train_indices, test_indices)
    """
    test_size = n_samples // n_splits
    embargo_size = int(n_samples * embargo_pct)
    
    for i in range(n_splits):
        # Test set
        test_start = i * test_size
        test_end = test_start + test_size
        test_indices = np.arange(test_start, min(test_end, n_samples))
        
        # Train set: exclude test and embargo periods
        train_indices = []
        
        # Before test set
        if test_start > 0:
            train_indices.extend(range(0, test_start))
        
        # After test set (with embargo)
        embargo_end = min(test_end + embargo_size, n_samples)
        if embargo_end < n_samples:
            train_indices.extend(range(embargo_end, n_samples))
        
        train_indices = np.array(train_indices)
        
        yield train_indices, test_indices


class TimeSeriesCV:
    """Time-series cross-validation wrapper."""
    
    def __init__(
        self,
        n_splits: int = 5,
        method: str = "walk_forward",
        test_size: int = None,
        gap: int = 0,
        embargo_pct: float = 0.01,
    ):
        """Initialize time-series CV.
        
        Args:
            n_splits: Number of splits
            method: 'walk_forward', 'expanding_window', or 'purged_kfold'
            test_size: Size of test set
            gap: Embargo period (for walk_forward and expanding_window)
            embargo_pct: Embargo percentage (for purged_kfold)
        """
        self.n_splits = n_splits
        self.method = method
        self.test_size = test_size
        self.gap = gap
        self.embargo_pct = embargo_pct
    
    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits.
        
        Args:
            X: Features (array-like or DataFrame)
            y: Target (optional)
            groups: Group labels (optional)
        
        Yields:
            Tuples of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if self.method == "walk_forward":
            yield from walk_forward_split(n_samples, self.n_splits, self.test_size, self.gap)
        elif self.method == "expanding_window":
            yield from expanding_window_split(n_samples, self.n_splits, None, self.test_size, self.gap)
        elif self.method == "purged_kfold":
            yield from purged_kfold_split(n_samples, self.n_splits, self.embargo_pct)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


__all__ = [
    "walk_forward_split",
    "expanding_window_split",
    "purged_kfold_split",
    "TimeSeriesCV",
]

