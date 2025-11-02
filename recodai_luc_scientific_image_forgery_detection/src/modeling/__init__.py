"""Model registry for the forgery detection workspace."""

from .baseline import build_vision_baseline, ForgeryBaseline, VisionBaselineConfig
from .contrastive_pretrain import ContrastiveConfig, ContrastivePretrainModule
from .dual_stream import DualStreamConfig, DualStreamForgeryModel, build_dual_stream_model
from .physics_guided import PhysicsGuidedConfig, PhysicsGuidedForgeryModel
from .pretrained import (
    ENCODER_CONFIGS,
    PretrainedForgeryModel,
    PretrainedModelConfig,
    build_pretrained_model,
)
from .utils import load_contrastive_encoder_weights

__all__ = [
    "build_vision_baseline",
    "ForgeryBaseline",
    "VisionBaselineConfig",
    "ContrastiveConfig",
    "ContrastivePretrainModule",
    "DualStreamConfig",
    "DualStreamForgeryModel",
    "build_dual_stream_model",
    "PhysicsGuidedConfig",
    "PhysicsGuidedForgeryModel",
    "ENCODER_CONFIGS",
    "PretrainedForgeryModel",
    "PretrainedModelConfig",
    "build_pretrained_model",
    "load_contrastive_encoder_weights",
]


