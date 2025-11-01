"""Advanced metrics for trajectory prediction evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def final_displacement_error(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> float:
    """Final Displacement Error (FDE): Euclidean distance at final position.
    
    Args:
        pred: Predicted coordinates (N, 2) or (N, T, 2)
        target: Target coordinates (N, 2) or (N, T, 2)
    
    Returns:
        Mean FDE across all samples
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # If 3D (batch, time, coords), take last timestep
    if pred.ndim == 3:
        pred = pred[:, -1, :]
    if target.ndim == 3:
        target = target[:, -1, :]
    
    # Euclidean distance
    distances = np.sqrt(np.sum((pred - target) ** 2, axis=-1))
    return float(np.mean(distances))


def average_displacement_error(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> float:
    """Average Displacement Error (ADE): Mean Euclidean distance over trajectory.
    
    Args:
        pred: Predicted coordinates (N, 2) or (N, T, 2)
        target: Target coordinates (N, 2) or (N, T, 2)
    
    Returns:
        Mean ADE across all samples
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # If 2D, add time dimension
    if pred.ndim == 2:
        pred = pred[:, np.newaxis, :]
    if target.ndim == 2:
        target = target[:, np.newaxis, :]
    
    # Euclidean distance at each timestep
    distances = np.sqrt(np.sum((pred - target) ** 2, axis=-1))
    return float(np.mean(distances))


def miss_rate(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 2.0,
) -> float:
    """Miss Rate: Percentage of predictions beyond threshold distance.
    
    Args:
        pred: Predicted coordinates (N, 2) or (N, T, 2)
        target: Target coordinates (N, 2) or (N, T, 2)
        threshold: Distance threshold in yards
    
    Returns:
        Miss rate (0.0 to 1.0)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # If 3D, take last timestep
    if pred.ndim == 3:
        pred = pred[:, -1, :]
    if target.ndim == 3:
        target = target[:, -1, :]
    
    distances = np.sqrt(np.sum((pred - target) ** 2, axis=-1))
    return float(np.mean(distances > threshold))


def direction_accuracy(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold_degrees: float = 15.0,
) -> float:
    """Direction Accuracy: Percentage of predictions within angular threshold.
    
    Args:
        pred: Predicted coordinates (N, 2)
        target: Target coordinates (N, 2)
        threshold_degrees: Angular threshold in degrees
    
    Returns:
        Direction accuracy (0.0 to 1.0)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # If 3D, take last timestep
    if pred.ndim == 3:
        pred = pred[:, -1, :]
    if target.ndim == 3:
        target = target[:, -1, :]
    
    # Compute angles
    pred_angles = np.arctan2(pred[:, 1], pred[:, 0])
    target_angles = np.arctan2(target[:, 1], target[:, 0])
    
    # Angular difference
    angle_diff = np.abs(pred_angles - target_angles)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # Wrap around
    
    # Convert to degrees
    angle_diff_deg = np.degrees(angle_diff)
    
    return float(np.mean(angle_diff_deg <= threshold_degrees))


def speed_consistency(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> float:
    """Speed Consistency: Mean absolute difference in predicted vs. actual speed.
    
    Args:
        pred: Predicted coordinates (N, T, 2)
        target: Target coordinates (N, T, 2)
    
    Returns:
        Mean speed difference
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # If 2D, can't compute speed
    if pred.ndim == 2 or target.ndim == 2:
        return 0.0
    
    # Compute velocities
    pred_vel = np.diff(pred, axis=1)
    target_vel = np.diff(target, axis=1)
    
    # Compute speeds
    pred_speed = np.sqrt(np.sum(pred_vel ** 2, axis=-1))
    target_speed = np.sqrt(np.sum(target_vel ** 2, axis=-1))
    
    return float(np.mean(np.abs(pred_speed - target_speed)))


def compute_all_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> Dict[str, float]:
    """Compute all trajectory metrics.
    
    Args:
        pred: Predicted coordinates
        target: Target coordinates
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        "fde": final_displacement_error(pred, target),
        "ade": average_displacement_error(pred, target),
        "miss_rate_2yd": miss_rate(pred, target, threshold=2.0),
        "miss_rate_5yd": miss_rate(pred, target, threshold=5.0),
        "direction_acc_15deg": direction_accuracy(pred, target, threshold_degrees=15.0),
        "direction_acc_30deg": direction_accuracy(pred, target, threshold_degrees=30.0),
    }
    
    # Add speed consistency if temporal data
    if isinstance(pred, np.ndarray) and pred.ndim == 3:
        metrics["speed_consistency"] = speed_consistency(pred, target)
    
    return metrics


__all__ = [
    "final_displacement_error",
    "average_displacement_error",
    "miss_rate",
    "direction_accuracy",
    "speed_consistency",
    "compute_all_metrics",
]

