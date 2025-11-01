from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class BackboneConfig:
    name: str = "efficientnet_b3"
    pretrained: bool = True
    dropout: float = 0.3


@dataclass
class FusionConfig:
    tabular_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    fusion_dropout: float = 0.25
    use_layernorm: bool = True
    fusion_type: str = "mlp"  # "mlp" or "perceiver"
    perceiver_latents: int = 32
    perceiver_layers: int = 3
    perceiver_heads: int = 4
    perceiver_dropout: float = 0.1


@dataclass
class CurriculumStageConfig:
    fractal_range: Optional[Tuple[float, float]] = None
    biomass_range: Optional[Tuple[float, float]] = None
    epochs: int = 0


@dataclass
class CurriculumConfig:
    enable: bool = True
    target_column: str = "Dry_Total_g"
    stages: List[CurriculumStageConfig] = field(
        default_factory=lambda: [
            CurriculumStageConfig(fractal_range=(0.0, 1.4), biomass_range=(0.0, 150.0), epochs=5),
            CurriculumStageConfig(fractal_range=(1.3, 1.6), biomass_range=(150.0, 350.0), epochs=5),
            CurriculumStageConfig(fractal_range=(1.5, 2.1), biomass_range=(350.0, 1000.0), epochs=5),
        ]
    )


@dataclass
class AugmentationConfig:
    image_size: int = 352
    random_resized_crop: Dict[str, List[float]] = field(
        default_factory=lambda: {"scale": [0.5, 1.0], "ratio": [0.8, 1.2]}
    )
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.3
    affine_scale: List[float] = field(default_factory=lambda: [0.85, 1.15])
    affine_translate: float = 0.08
    affine_rotate: float = 30.0
    brightness_contrast: float = 0.35
    hue_shift: float = 20.0
    blur_probability: float = 0.3
    jpeg_probability: float = 0.2
    coarse_dropout_probability: float = 0.4
    coarse_dropout_max: float = 0.18
    policy: str = "standard"  # "standard", "randaugment", "trivialaugment"
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    trivial_magnitude_bins: int = 31


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    scheduler_t_max: int = 15
    ema_decay: Optional[float] = 0.999
    warmup_epochs: int = 1


@dataclass
class RegularizationConfig:
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.5


@dataclass
class SnapshotConfig:
    num_snapshots: int = 0


@dataclass
class ExperimentConfig:
    train_csv: Path
    image_dir: Path
    output_dir: Path = Path("competitions/csiro_biomass/outputs/baseline")
    test_csv: Optional[Path] = None
    sample_submission: Optional[Path] = None
    fractal_csv: Optional[Path] = None
    use_metadata: bool = True

    image_column: Optional[str] = None
    id_column: str = "sample_id"
    target_name_column: str = "target_name"
    target_value_column: str = "target"

    batch_size: int = 32
    epochs: int = 15
    val_fraction: float = 0.2
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    constraint_tolerance: float = 0.5

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    save_oof: bool = False

    def to_dict(self) -> Dict:
        cfg = asdict(self)
        cfg["train_csv"] = str(self.train_csv)
        cfg["image_dir"] = str(self.image_dir)
        cfg["output_dir"] = str(self.output_dir)
        if self.test_csv is not None:
            cfg["test_csv"] = str(self.test_csv)
        if self.sample_submission is not None:
            cfg["sample_submission"] = str(self.sample_submission)
        if self.fractal_csv is not None:
            cfg["fractal_csv"] = str(self.fractal_csv)
        cfg["use_metadata"] = self.use_metadata
        return cfg
