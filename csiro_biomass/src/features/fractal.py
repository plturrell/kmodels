"""Fractal mathematics and information theory utilities for biomass analysis."""

import numpy as np
import pywt
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from pathlib import Path

def compute_fractal_dimension(image: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate box-counting fractal dimension of an image.
    
    Args:
        image: 2D grayscale image array
        threshold: Binarization threshold
        
    Returns:
        Fractal dimension (D ∈ [1.0, 2.0])
    """
    # Ensure image is 2D
    if image.ndim > 2:
        image = image.mean(axis=-1) if image.shape[-1] in [3, 4] else image[0]
    
    # Binarize the image
    binary = image > threshold
    
    # Box counting at multiple scales
    scales = 2 ** np.arange(1, 8)  # Scales from 2 to 128
    counts = []
    
    for scale in scales:
        if binary.shape[0] < scale or binary.shape[1] < scale:
            break
        # Downsample by averaging
        scaled = binary[::scale, ::scale]
        if scaled.size == 0:
            continue
        # Count non-empty boxes
        box_count = np.sum(scaled > 0)
        counts.append(box_count)
    
    if len(counts) < 3:
        return 1.5  # Default for insufficient data
    
    # Fit power law: N(ε) ∝ ε^(-D)
    log_scales = np.log(scales[:len(counts)])
    log_counts = np.log(np.array(counts) + 1e-8)
    
    # Fractal dimension is negative slope
    slope, _, r_value, _, _ = stats.linregress(log_scales, log_counts)
    fractal_dim = -slope
    
    return max(1.0, min(2.0, fractal_dim))  # Clamp to reasonable range

class WaveletFractalFeatures:
    """Multi-scale fractal analysis using wavelet transforms."""
    
    def __init__(self, wavelet: str = 'db4', levels: int = 3):
        self.wavelet = wavelet
        self.levels = levels
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract multi-scale fractal and energy features.
        """
        if image.ndim > 2:
            image = image.mean(axis=-1)
        
        try:
            coeffs = pywt.wavedec2(image, self.wavelet, level=self.levels)
        except ValueError:
            return np.array([]) # Image too small for specified level

        features = []
        # Process the final approximation coefficients (coarsest scale)
        final_approx = coeffs[0]
        features.extend([compute_fractal_dimension(final_approx), np.mean(final_approx), np.std(final_approx)])

        # Process the detail coefficients at each level
        for i, detail_coeffs in enumerate(coeffs[1:]):
            cH, cV, cD = detail_coeffs
            # Energy features for this level
            detail_energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
            total_energy = np.sum(final_approx**2) + detail_energy
            features.append(detail_energy / (total_energy + 1e-8))

            # Statistical features for this level
            features.extend([np.std(cH), np.std(cV), np.std(cD)])
        
        return np.array(features)

class InformationAwareFusion(nn.Module):
    """
    Information-theoretic feature fusion based on mutual information estimates.
    """
    
    def __init__(self, img_dim: int, meta_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.img_dim = img_dim
        self.meta_dim = meta_dim
        
        # Learnable information weights
        self.info_weights = nn.Parameter(torch.ones(2))
        
        # Feature compression with information bottleneck
        self.feature_compression = nn.Sequential(
            nn.Linear(img_dim + meta_dim, hidden_dim),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2))
        )
        
    def forward(self, img_feat: torch.Tensor, meta_feat: torch.Tensor) -> torch.Tensor:
        # This is the corrected interpretation: use weights to scale features before concatenation.
        weights = torch.softmax(self.info_weights, dim=0)
        
        # Information-weighted concatenation
        fused = torch.cat([img_feat * weights[0], meta_feat * weights[1]], dim=1)
        
        # Information-preserving compression
        return self.feature_compression(fused)

def estimate_mutual_info(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Estimate mutual information between features and targets.
    
    Args:
        features: (n_samples, n_features)
        targets: (n_samples, n_targets)
        
    Returns:
        Mutual information scores for each feature-target pair
    """
    mi_scores = np.zeros((features.shape[1], targets.shape[1]))
    
    for i in range(features.shape[1]):
        for j in range(targets.shape[1]):
            mi_scores[i, j] = mutual_info_regression(
                features[:, i:i+1], targets[:, j], 
                random_state=42
            )[0]
    
    return mi_scores

class FractalCurriculum:
    """Curriculum learning based on fractal complexity progression."""
    
    def __init__(self, stages: List[Tuple[float, float]] = None):
        self.stages = stages or [
            (1.2, 1.4),   # Stage 1: Low complexity (dense, homogeneous)
            (1.4, 1.6),   # Stage 2: Medium complexity
            (1.6, 1.8)    # Stage 3: High complexity (sparse, complex)
        ]
        
    def get_stage_samples(self, epoch: int, dataframe: pd.DataFrame, image_column: str) -> List[int]:
        """Get sample indices for the current curriculum stage."""
        target_fd_range = self.stages[min(epoch // 10, len(self.stages) - 1)]
        
        if 'fractal_dimension' not in dataframe.columns:
            return dataframe.index.tolist()

        selected_indices = dataframe[
            (dataframe['fractal_dimension'] >= target_fd_range[0]) & 
            (dataframe['fractal_dimension'] <= target_fd_range[1])
        ].index.tolist()
        
        return selected_indices if selected_indices else dataframe.index.tolist()