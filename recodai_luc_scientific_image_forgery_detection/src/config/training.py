"""Configuration structures inspired by the CSIRO biomass workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional


_BASE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _BASE_DIR / "data" / "raw" / "recodai-luc-scientific-image-forgery-detection"
_DEFAULT_OUTPUT_DIR = _BASE_DIR / "outputs" / "baseline"


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    scheduler_t_max: int = 10
    warmup_epochs: int = 0
    ema_decay: Optional[float] = 0.99


@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.25
    rotation_probability: float = 0.2
    jitter_strength: float = 0.1


@dataclass
class ExperimentConfig:
    data_root: Path = _DEFAULT_DATA_ROOT
    output_dir: Path = _DEFAULT_OUTPUT_DIR
    epochs: int = 5
    batch_size: int = 4
    val_fraction: float = 0.2
    num_workers: int = 0
    seed: int = 42
    mask_loss_weight: float = 1.0
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    gradient_clip_val: Optional[float] = 1.0
    max_train_steps: Optional[int] = None
    limit_val_batches: Optional[int] = None
    model_type: str = "baseline"
    pretrained_architecture: str = "Unet"
    pretrained_encoder: str = "resnet34"
    pretrained_weights: Optional[str] = "imagenet"
    use_class_weights: bool = True
    mask_loss: str = "combined"
    contrastive_checkpoint: Optional[Path] = None
    freeze_encoder: bool = False

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def to_dict(self) -> Dict[str, object]:
        cfg = asdict(self)
        cfg["data_root"] = str(self.data_root)
        cfg["output_dir"] = str(self.output_dir)
        if self.contrastive_checkpoint is not None:
            cfg["contrastive_checkpoint"] = str(self.contrastive_checkpoint)
        return cfg
