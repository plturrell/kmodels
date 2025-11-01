"""Training configuration for the ECG digitisation Lightning pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from ..data.dataset import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_METADATA_CSV,
    DEFAULT_SIGNAL_DIR,
)


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    scheduler_t_max: int = 10
    warmup_epochs: int = 0
    gradient_clip_val: Optional[float] = 1.0


@dataclass
class AugmentationConfig:
    image_size: int = 512
    random_rotation: float = 3.0
    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0


@dataclass
class ExperimentConfig:
    train_csv: Path = DEFAULT_METADATA_CSV
    image_dir: Path = DEFAULT_IMAGE_DIR
    signal_dir: Optional[Path] = DEFAULT_SIGNAL_DIR
    id_column: str = "ecg_id"
    image_column: str = "image"
    signal_column: str = "signal"
    lead_column: Optional[str] = "lead"
    group_column: Optional[str] = None

    batch_size: int = 16
    val_fraction: float = 0.2
    num_workers: int = 4
    seed: int = 42

    epochs: int = 20
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    max_train_steps: Optional[int] = None
    limit_val_batches: Optional[int] = None

    backbone: str = "resnet18"
    hidden_dim: int = 512
    dropout: float = 0.1
    pretrained: bool = True
    use_refiner: bool = False
    refiner_channels: int = 128
    refiner_layers: int = 3
    signal_length: Optional[int] = None
    signal_channels: Optional[int] = None
    preload_signals: bool = False

    output_dir: Path = Path(
        "competitions/physionet_ecg_image_digitization/outputs/lightning_baseline"
    )

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["train_csv"] = str(self.train_csv)
        payload["image_dir"] = str(self.image_dir)
        payload["signal_dir"] = str(self.signal_dir) if self.signal_dir else None
        payload["output_dir"] = str(self.output_dir)
        return payload


