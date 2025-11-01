"""Model definitions for MABe mouse behavior detection."""

from .baseline import PoseBaselineConfig, PoseSequenceClassifier, build_pose_baseline

__all__ = ["PoseBaselineConfig", "PoseSequenceClassifier", "build_pose_baseline"]

