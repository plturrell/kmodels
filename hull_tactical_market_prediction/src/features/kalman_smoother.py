"""Kalman smoothing utilities for target denoising in financial time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class KalmanSmootherConfig:
    """Configuration for Kalman smoother."""
    
    process_noise: float = 0.01  # Process noise variance (Q)
    observation_noise: Optional[float] = None  # Observation noise variance (R), None = learned
    initial_state: Optional[float] = None  # Initial state estimate, None = use first observation
    initial_uncertainty: float = 1.0  # Initial state uncertainty (P0)


def smooth_targets(
    targets: pd.Series | np.ndarray,
    config: Optional[KalmanSmootherConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Kalman smoother to denoise target values.
    
    Args:
        targets: Time series of target values
        config: Kalman smoother configuration
        
    Returns:
        Tuple of (smoothed_targets, uncertainties)
        - smoothed_targets: Denoised target values
        - uncertainties: Estimated uncertainty for each point
    """
    if config is None:
        config = KalmanSmootherConfig()
    
    if isinstance(targets, pd.Series):
        values = targets.values
    else:
        values = np.asarray(targets).flatten()
    
    n = len(values)
    if n == 0:
        return np.array([]), np.array([])
    
    # Remove NaN values for computation, but preserve structure
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return values.copy(), np.full_like(values, np.nan)
    
    valid_values = values[valid_mask]
    
    # Initialize state and covariance
    if config.initial_state is None:
        x = valid_values[0]  # Initial state = first observation
    else:
        x = config.initial_state
    
    P = config.initial_uncertainty  # Initial uncertainty
    
    # Process and observation noise
    Q = config.process_noise
    if config.observation_noise is None:
        # Learn observation noise from data variance
        R = np.var(np.diff(valid_values)) * 0.1 if len(valid_values) > 1 else 0.1
    else:
        R = config.observation_noise
    
    # Kalman filter (forward pass)
    filtered_states = np.full(n, np.nan)
    filtered_uncertainties = np.full(n, np.nan)
    
    valid_idx = 0
    for i in range(n):
        if valid_mask[i]:
            # Prediction step
            # x_pred = x (no dynamics, assuming constant state with process noise)
            P_pred = P + Q
            
            # Update step
            y = valid_values[valid_idx] - x  # Innovation
            S = P_pred + R  # Innovation covariance
            K = P_pred / S  # Kalman gain
            
            x = x + K * y  # Updated state estimate
            P = (1 - K) * P_pred  # Updated uncertainty
            
            filtered_states[i] = x
            filtered_uncertainties[i] = P
            valid_idx += 1
        else:
            # For NaN values, propagate prediction but don't update
            P = P + Q  # Uncertainty increases
            filtered_states[i] = x
            filtered_uncertainties[i] = P
    
    # Kalman smoother (backward pass)
    smoothed_states = filtered_states.copy()
    smoothed_uncertainties = filtered_uncertainties.copy()
    
    # Start from second-to-last and work backwards
    for i in range(n - 2, -1, -1):
        if valid_mask[i] and valid_mask[i + 1]:
            # Smoothing step
            # C = P[i] / (P[i] + Q)  # Smoothing gain
            # But for constant state model, we use a simple weighted average
            C = filtered_uncertainties[i] / (filtered_uncertainties[i] + Q)
            smoothed_states[i] = filtered_states[i] + C * (smoothed_states[i + 1] - filtered_states[i])
            smoothed_uncertainties[i] = filtered_uncertainties[i] - C * (filtered_uncertainties[i] - smoothed_uncertainties[i + 1])
        elif valid_mask[i]:
            smoothed_states[i] = filtered_states[i]
            smoothed_uncertainties[i] = filtered_uncertainties[i]
    
    return smoothed_states, smoothed_uncertainties


def apply_kalman_smoothing_to_dataframe(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[KalmanSmootherConfig] = None,
) -> pd.DataFrame:
    """Apply Kalman smoothing to target column in DataFrame.
    
    Args:
        df: DataFrame with time series data
        target_column: Name of target column to smooth
        config: Kalman smoother configuration
        
    Returns:
        DataFrame with additional columns:
        - {target_column}_smoothed: Smoothed target values
        - {target_column}_uncertainty: Uncertainty estimates
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    targets = df[target_column]
    smoothed, uncertainties = smooth_targets(targets, config)
    
    result = df.copy()
    result[f"{target_column}_smoothed"] = smoothed
    result[f"{target_column}_uncertainty"] = uncertainties
    
    return result

