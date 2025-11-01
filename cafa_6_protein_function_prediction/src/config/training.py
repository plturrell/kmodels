"""Training configuration for the CAFA 6 Lightning baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "data" / "raw" / "cafa-6-protein-function-prediction"
_TRAIN_DIR = _DATA_ROOT / "Train"


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    gradient_clip_val: Optional[float] = 1.0


@dataclass
class AugmentationConfig:
    use_fractal_features: bool = False
    fractal_max_iter: int = 100


@dataclass
class ExperimentConfig:
    sequences_path: Path = _TRAIN_DIR / "train_sequences.fasta"
    annotations_path: Path = _TRAIN_DIR / "train_terms.tsv"
    ontology_path: Path = _DATA_ROOT / "go-basic.obo"
    output_dir: Path = _REPO_ROOT / "outputs" / "lightning_baseline"

    model_name: str = "facebook/esm2_t6_8M_UR50D"
    hidden_dims: Sequence[int] = (512, 256)
    dropout: float = 0.3

    batch_size: int = 32
    num_workers: int = 0
    embedding_batch_size: int = 8
    val_fraction: float = 0.2
    min_go_terms: int = 3
    max_samples: Optional[int] = None
    seed: int = 42
    use_go_hierarchy: bool = False
    filter_aspect: Optional[str] = None

    use_embedding_cache: bool = False
    embedding_cache_dir: Optional[Path] = None
    embedding_cache_use_memory: bool = True
    embedding_cache_use_disk: bool = True

    epochs: int = 50
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    max_train_steps: Optional[int] = None
    limit_val_batches: Optional[int] = None

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["sequences_path"] = str(self.sequences_path)
        payload["annotations_path"] = str(self.annotations_path)
        payload["ontology_path"] = str(self.ontology_path)
        payload["output_dir"] = str(self.output_dir)
        payload["hidden_dims"] = list(self.hidden_dims)
        payload["embedding_cache_dir"] = (
            str(self.embedding_cache_dir) if self.embedding_cache_dir is not None else None
        )
        return payload


