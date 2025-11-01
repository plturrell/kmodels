"""Training configuration inspired by the CSIRO Lightning scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_val: Optional[float] = 1.0


@dataclass
class AugmentationConfig:
    horizontal_flip: float = 0.5
    temporal_jitter: int = 8
    spatial_noise: float = 0.02


@dataclass
class TrainingConfig:
    train_csv: Path
    asset_root: Optional[Path] = None
    asset_column: str = "pose_path"
    target_column: str = "behavior"
    id_column: str = "clip_id"
    metadata_columns: list[str] = field(default_factory=list)

    sequence_length: int = 128
    time_axis: int = 0
    pose_array: str = "keypoints"
    center: bool = True
    standardize: bool = False

    batch_size: int = 64
    val_batch_size: Optional[int] = None
    val_fraction: float = 0.2
    seed: int = 42

    epochs: int = 30
    max_train_steps: Optional[int] = None
    max_val_steps: Optional[int] = None

    num_workers: int = 4
    accelerator: str = "auto"
    precision: str = "32"
    devices: int = 1

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    pooling: str = "last"

    output_dir: Path = Path(
        "competitions/mabe_mouse_behavior_detection/outputs/lightning_baseline"
    )

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    test_csv: Optional[Path] = None
    test_asset_root: Optional[Path] = None
    submission_path: Optional[Path] = None
    prediction_column: str = "prediction"
    id_output_column: Optional[str] = None

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["train_csv"] = str(self.train_csv)
        payload["asset_root"] = (
            str(self.asset_root) if self.asset_root is not None else None
        )
        payload["output_dir"] = str(self.output_dir)
        return payload


