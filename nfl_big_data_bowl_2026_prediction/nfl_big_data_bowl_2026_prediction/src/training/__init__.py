"""Training entry points for NFL Big Data Bowl models."""

from .torch_pipeline import TrainingConfig, main, train_model

__all__ = ["TrainingConfig", "main", "train_model"]
