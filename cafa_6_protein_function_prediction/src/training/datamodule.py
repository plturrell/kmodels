"""Lightning data module for CAFA 6 protein function prediction."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

from ..config.training import AugmentationConfig, ExperimentConfig
from ..data import build_samples, load_go_terms_long_format, load_sequences_from_fasta, train_val_split
from ..data.dataset import ProteinSample
from ..features.embedding_cache import EmbeddingCache
from ..features.embeddings import EmbeddingResult, embed_sequences
from ..features.fractal_features import FractalProteinFeatures, combine_features
from ..data.go_ontology import parse_obo_file


def _samples_to_matrix(results: Sequence[EmbeddingResult]) -> np.ndarray:
    return np.stack([item.vector for item in results], axis=0).astype(np.float32)


def _apply_fractal_features(
    embeddings: np.ndarray,
    augmentation: AugmentationConfig,
) -> np.ndarray:
    if not augmentation.use_fractal_features:
        return embeddings
    extractor = FractalProteinFeatures(max_iter=augmentation.fractal_max_iter)
    fractal = extractor.extract_batch(embeddings)
    return combine_features(embeddings, fractal)


class ProteinDataModule(pl.LightningDataModule):
    """Prepare train/validation tensors and metadata for Lightning."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config

        self.train_samples: List[ProteinSample] = []
        self.val_samples: List[ProteinSample] = []
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None

        self.mlb: Optional[MultiLabelBinarizer] = None
        self.classes: List[str] = []
        self.val_accessions: List[str] = []
        self.val_ground_truth: Dict[str, Sequence[str]] = {}
        self.ontology = None

        self.embedding_dim: Optional[int] = None
        self.num_labels: Optional[int] = None
        self.embedding_cache: Optional[EmbeddingCache] = None

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        sequences = load_sequences_from_fasta(self.config.sequences_path)
        annotations = load_go_terms_long_format(
            self.config.annotations_path,
            filter_aspect=self.config.filter_aspect,
        )
        samples = build_samples(sequences, annotations)

        if self.config.min_go_terms > 0:
            samples = [s for s in samples if len(s.go_terms) >= self.config.min_go_terms]

        if self.config.max_samples:
            samples = samples[: self.config.max_samples]

        ontology = None
        if self.config.use_go_hierarchy and self.config.ontology_path.exists():
            ontology = parse_obo_file(self.config.ontology_path)
            propagated: List[ProteinSample] = []
            for sample in samples:
                propagated_terms = ontology.propagate_annotations(set(sample.go_terms))
                propagated.append(
                    replace(sample, go_terms=tuple(sorted(propagated_terms)))
                )
            samples = propagated
        self.ontology = ontology

        train_samples, val_samples = train_val_split(
            samples,
            val_fraction=self.config.val_fraction,
            seed=self.config.seed,
        )
        if not train_samples or not val_samples:
            raise RuntimeError("Train/validation split produced no samples.")

        if self.config.use_embedding_cache and self.embedding_cache is None:
            self.embedding_cache = EmbeddingCache(
                cache_dir=self.config.embedding_cache_dir,
                use_memory_cache=self.config.embedding_cache_use_memory,
                use_disk_cache=self.config.embedding_cache_use_disk,
            )

        train_embeddings = _samples_to_matrix(
            embed_sequences(
                train_samples,
                model_name=self.config.model_name,
                batch_size=self.config.embedding_batch_size,
                cache=self.embedding_cache,
            )
        )
        val_embeddings = _samples_to_matrix(
            embed_sequences(
                val_samples,
                model_name=self.config.model_name,
                batch_size=self.config.embedding_batch_size,
                cache=self.embedding_cache,
            )
        )

        train_embeddings = _apply_fractal_features(train_embeddings, self.config.augmentation)
        val_embeddings = _apply_fractal_features(val_embeddings, self.config.augmentation)

        mlb = MultiLabelBinarizer()
        train_labels = mlb.fit_transform([sample.go_terms for sample in train_samples])
        val_labels = mlb.transform([sample.go_terms for sample in val_samples])
        if len(mlb.classes_) == 0:
            raise RuntimeError("No GO-term annotations found; cannot train model.")

        train_tensor = torch.from_numpy(train_embeddings).float()
        val_tensor = torch.from_numpy(val_embeddings).float()
        train_target = torch.from_numpy(train_labels.astype(np.float32)).float()
        val_target = torch.from_numpy(val_labels.astype(np.float32)).float()

        train_indices = torch.arange(train_tensor.shape[0], dtype=torch.long)
        val_indices = torch.arange(val_tensor.shape[0], dtype=torch.long)

        self.train_dataset = TensorDataset(train_tensor, train_target, train_indices)
        self.val_dataset = TensorDataset(val_tensor, val_target, val_indices)

        self.train_samples = list(train_samples)
        self.val_samples = list(val_samples)
        self.mlb = mlb
        self.classes = list(mlb.classes_)
        self.val_accessions = [sample.accession for sample in val_samples]
        self.val_ground_truth = {
            sample.accession: set(sample.go_terms) for sample in val_samples
        }
        self.embedding_dim = train_tensor.shape[1]
        self.num_labels = len(self.classes)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule has not been setup.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule has not been setup.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


