"""Expose configuration dataclasses for the ECG digitisation workspace."""

from .training import AugmentationConfig, ExperimentConfig, OptimizerConfig

__all__ = ["AugmentationConfig", "ExperimentConfig", "OptimizerConfig"]


