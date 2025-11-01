"""Signal post-processing utilities for ECG waveforms."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d


def remove_baseline_wander(
    signal: np.ndarray,
    sampling_rate: float = 500.0,
    cutoff: float = 0.5,
) -> np.ndarray:
    """Remove baseline wander using high-pass filter.
    
    Args:
        signal: Input signal (1D or 2D)
        sampling_rate: Sampling rate in Hz
        cutoff: Cutoff frequency in Hz
    
    Returns:
        Baseline-corrected signal
    """
    nyquist = sampling_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    # Design high-pass Butterworth filter
    b, a = scipy_signal.butter(4, normalized_cutoff, btype='high')
    
    if signal.ndim == 1:
        return scipy_signal.filtfilt(b, a, signal)
    else:
        # Apply to each channel
        return np.array([scipy_signal.filtfilt(b, a, ch) for ch in signal])


def smooth_signal(
    signal: np.ndarray,
    method: str = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
    sigma: float = 2.0,
) -> np.ndarray:
    """Smooth signal using various methods.
    
    Args:
        signal: Input signal
        method: Smoothing method ('savgol', 'gaussian', 'moving_average')
        window_length: Window length for Savitzky-Golay or moving average
        polyorder: Polynomial order for Savitzky-Golay
        sigma: Standard deviation for Gaussian smoothing
    
    Returns:
        Smoothed signal
    """
    if method == "savgol":
        # Savitzky-Golay filter
        if signal.ndim == 1:
            return scipy_signal.savgol_filter(signal, window_length, polyorder)
        else:
            return np.array([
                scipy_signal.savgol_filter(ch, window_length, polyorder)
                for ch in signal
            ])
    
    elif method == "gaussian":
        # Gaussian smoothing
        if signal.ndim == 1:
            return gaussian_filter1d(signal, sigma)
        else:
            return np.array([gaussian_filter1d(ch, sigma) for ch in signal])
    
    elif method == "moving_average":
        # Moving average
        kernel = np.ones(window_length) / window_length
        if signal.ndim == 1:
            return np.convolve(signal, kernel, mode='same')
        else:
            return np.array([np.convolve(ch, kernel, mode='same') for ch in signal])
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def denoise_signal(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    threshold_mode: str = "soft",
) -> np.ndarray:
    """Denoise signal using wavelet transform.
    
    Args:
        signal: Input signal
        wavelet: Wavelet type
        level: Decomposition level
        threshold_mode: Thresholding mode ('soft' or 'hard')
    
    Returns:
        Denoised signal
    """
    try:
        import pywt
    except ImportError:
        print("Warning: pywt not installed, skipping wavelet denoising")
        return signal
    
    if signal.ndim == 1:
        # Wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Calculate threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply threshold to detail coefficients
        coeffs[1:] = [pywt.threshold(c, threshold, mode=threshold_mode) for c in coeffs[1:]]
        
        # Reconstruct
        return pywt.waverec(coeffs, wavelet)
    else:
        return np.array([denoise_signal(ch, wavelet, level, threshold_mode) for ch in signal])


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
    clip_std: float = 3.0,
) -> np.ndarray:
    """Normalize signal.
    
    Args:
        signal: Input signal
        method: Normalization method ('zscore', 'minmax', 'robust')
        clip_std: Number of standard deviations for clipping (zscore only)
    
    Returns:
        Normalized signal
    """
    if method == "zscore":
        mean = np.mean(signal)
        std = np.std(signal)
        normalized = (signal - mean) / (std + 1e-8)
        # Clip outliers
        normalized = np.clip(normalized, -clip_std, clip_std)
        return normalized
    
    elif method == "minmax":
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "robust":
        # Use median and IQR for robustness
        median = np.median(signal)
        q75, q25 = np.percentile(signal, [75, 25])
        iqr = q75 - q25
        return (signal - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def validate_signal(
    signal: np.ndarray,
    *,
    check_nan: bool = True,
    check_inf: bool = True,
    check_range: Optional[tuple[float, float]] = None,
) -> bool:
    """Validate signal quality.
    
    Args:
        signal: Input signal
        check_nan: Check for NaN values
        check_inf: Check for infinite values
        check_range: Expected value range (min, max)
    
    Returns:
        True if signal is valid
    """
    if check_nan and np.any(np.isnan(signal)):
        return False
    
    if check_inf and np.any(np.isinf(signal)):
        return False
    
    if check_range is not None:
        min_val, max_val = check_range
        if np.any(signal < min_val) or np.any(signal > max_val):
            return False
    
    return True


__all__ = [
    "remove_baseline_wander",
    "smooth_signal",
    "denoise_signal",
    "normalize_signal",
    "validate_signal",
]

