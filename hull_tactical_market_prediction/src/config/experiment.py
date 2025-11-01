"""Experiment configuration for tabular neural solvers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    model_type: str = "mlp"
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.2
    activation: str = "gelu"
    batch_norm: bool = True
    perceiver_latent_dim: int = 128
    perceiver_num_latents: int = 16
    perceiver_layers: int = 6
    perceiver_heads: int = 4
    perceiver_dropout: float = 0.1
    perceiver_ff_mult: int = 4


@dataclass
class LossConfig:
    mode: str = "mse"
    sharpe_risk_free: float = 0.0
    sharpe_lambda: float = 1.0
    mse_weight: float = 0.5


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    gradient_clip_val: float = 1.0


@dataclass
class TrainerConfig:
    epochs: int = 40
    batch_size: int = 512
    num_workers: int = 4
    val_fraction: float = 0.12
    time_series_holdout: bool = True
    accelerator: str = "auto"
    devices: Optional[int] = None
    precision: str = "32-true"
    deterministic: bool = True


@dataclass
class FeatureConfig:
    drop_columns: List[str] = field(default_factory=list)
    max_nan_ratio: float = 0.5
    drop_constant: bool = True
    lag_steps: List[int] = field(default_factory=lambda: [1, 5, 21])
    rolling_windows: List[int] = field(default_factory=lambda: [5, 21])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std"])


@dataclass
class ExperimentConfig:
    train_csv: Path
    output_dir: Path = Path("competitions/hull_tactical_market_prediction/outputs/tabular_nn")
    test_csv: Optional[Path] = None
    sample_submission: Optional[Path] = None

    target_column: str = "forward_returns"
    id_column: str = "date_id"
    submission_column: str = "prediction"

    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["train_csv"] = str(self.train_csv)
        if self.test_csv is not None:
            data["test_csv"] = str(self.test_csv)
        if self.sample_submission is not None:
            data["sample_submission"] = str(self.sample_submission)
        data["output_dir"] = str(self.output_dir)
        return data
