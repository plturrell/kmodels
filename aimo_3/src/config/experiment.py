"""Experiment configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""

    model_type: str = "hybrid"  # llm, symbolic, hybrid
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    use_code_generation: bool = True
    use_llm_reasoning: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: Path = Path("data/raw")
    train_split: float = 0.9
    max_seq_length: int = 2048
    preprocessing: Dict[str, bool] = field(default_factory=lambda: {
        "normalize_latex": True,
        "extract_math": True,
    })


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_name: str = "baseline"
    output_dir: Path = Path("outputs")
    seed: int = 42
    device: str = "cuda"  # cuda, cpu, mps

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "device": self.device,
            "model": {
                "model_type": self.model.model_type,
                "model_name": self.model.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "use_code_generation": self.model.use_code_generation,
                "use_llm_reasoning": self.model.use_llm_reasoning,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "num_epochs": self.training.num_epochs,
                "warmup_steps": self.training.warmup_steps,
                "weight_decay": self.training.weight_decay,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "max_grad_norm": self.training.max_grad_norm,
            },
            "data": {
                "data_dir": str(self.data.data_dir),
                "train_split": self.data.train_split,
                "max_seq_length": self.data.max_seq_length,
                "preprocessing": self.data.preprocessing,
            },
        }

