"""Training configuration dataclasses for the analytics competition.

The shapes are intentionally inspired by the CSIRO biomass pipeline so we can
reuse tooling such as config serialization or automated search drivers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetConfig:
    """Location and schema hints for the competition bundle."""

    data_root: Path = Path("competitions/nfl_big_data_bowl_2026_analytics/data/raw")
    bundle_dirname: str = "nfl-big-data-bowl-2026-analytics"
    train_subdir: str = "train"
    supplementary_filename: str = "supplementary_data.csv"
    input_prefix: str = "input"
    output_prefix: str = "output"
    id_columns: List[str] = field(
        default_factory=lambda: ["game_id", "play_id", "nfl_id", "frame_id"]
    )
    target_columns: List[str] = field(default_factory=lambda: ["target_x", "target_y"])
    fold_column: Optional[str] = "game_id"
    include_supplementary: bool = True

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        return payload


@dataclass
class ModelConfig:
    """Generic sequence regressor hyperparameters."""

    architecture: str = "mlp"
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    n_heads: int = 4
    sequence_length: int = 64
    use_metadata: bool = True
    num_layers: int = 4
    latent_dim: int = 256
    num_latents: int = 8

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class FeatureConfig:
    """Feature toggles inspired by relational/temporal research ideas."""

    use_pairwise_distance: bool = False
    use_game_clock_seconds: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Optimiser / scheduler options."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    use_scheduler: bool = True
    cosine_t_max: int = 20
    warmup_steps: int = 500

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["betas"] = list(self.betas)
        return payload


@dataclass
class TrainingConfig:
    """Top-level experiment configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    experiment_root: Path = Path("competitions/nfl_big_data_bowl_2026_analytics/outputs/baseline")
    run_name: Optional[str] = None

    batch_size: int = 256
    num_workers: int = 8
    persistent_workers: bool = False
    epochs: int = 30
    gradient_clip_norm: float = 1.0
    seed: int = 42
    device: str = "mps"
    val_fraction: float = 0.2
    snapshot_count: int = 0
    early_stopping_patience: Optional[int] = None

    log_interval: int = 100
    eval_interval: int = 1

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["dataset"] = self.dataset.to_dict()
        payload["model"] = self.model.to_dict()
        payload["features"] = self.features.to_dict()
        payload["optimizer"] = self.optimizer.to_dict()
        payload["experiment_root"] = str(self.experiment_root)
        payload["persistent_workers"] = self.persistent_workers
        payload["early_stopping_patience"] = self.early_stopping_patience
        return payload
