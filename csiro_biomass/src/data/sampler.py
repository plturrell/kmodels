"""Custom PyTorch samplers for curriculum learning."""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

LOGGER = logging.getLogger(__name__)


class CurriculumStage:
    """Representation of a curriculum stage."""

    def __init__(
        self,
        *,
        fractal_range: Optional[Tuple[float, float]] = None,
        biomass_range: Optional[Tuple[float, float]] = None,
        epochs: Optional[int] = None,
    ) -> None:
        self.fractal_range = fractal_range
        self.biomass_range = biomass_range
        self.epochs = max(int(epochs) if epochs is not None else 0, 0)

    @classmethod
    def from_config(cls, config) -> "CurriculumStage":
        if isinstance(config, CurriculumStage):
            return config
        if isinstance(config, dict):
            fractal_range = config.get("fractal_range")
            biomass_range = config.get("biomass_range")
            return cls(
                fractal_range=tuple(fractal_range) if fractal_range is not None else None,
                biomass_range=tuple(biomass_range) if biomass_range is not None else None,
                epochs=config.get("epochs"),
            )
        raise TypeError(f"Unsupported curriculum stage config: {type(config)!r}")


class FractalCurriculumSampler(Sampler[List[int]]):
    """Samples data based on a curriculum of increasing complexity."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        batch_size: int,
        shuffle: bool = True,
        *,
        stages: Optional[Sequence] = None,
        total_epochs: int = 15,
        biomass_column: Optional[str] = None,
    ):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.biomass_column = biomass_column

        default_stages: List[CurriculumStage] = [
            CurriculumStage(fractal_range=(0.0, 1.4), biomass_range=None, epochs=0),
            CurriculumStage(fractal_range=(1.4, 1.6), biomass_range=None, epochs=0),
            CurriculumStage(fractal_range=(1.6, 2.1), biomass_range=None, epochs=0),
        ]

        if stages:
            parsed_stages = [CurriculumStage.from_config(stage) for stage in stages]
        else:
            parsed_stages = default_stages

        total_epochs = max(int(total_epochs), 1)
        specified = sum(stage.epochs for stage in parsed_stages)
        unspecified = [stage for stage in parsed_stages if stage.epochs == 0]
        if unspecified:
            remaining = max(total_epochs - specified, 0)
            per_stage = max(1, math.floor(remaining / len(unspecified))) if remaining > 0 else max(1, math.floor(total_epochs / len(parsed_stages)))
            for stage in unspecified:
                stage.epochs = per_stage
        else:
            if specified != total_epochs and specified > 0:
                scale = total_epochs / specified
                for stage in parsed_stages:
                    stage.epochs = max(1, int(round(stage.epochs * scale)))

        self.stages = parsed_stages
        self.stage_boundaries: List[int] = []
        cumulative = 0
        for stage in self.stages:
            stage.epochs = max(stage.epochs, 1)
            cumulative += stage.epochs
            self.stage_boundaries.append(cumulative)

        self._validate_columns()

    def _validate_columns(self) -> None:
        if self.biomass_column and self.biomass_column not in self.dataframe.columns:
            LOGGER.warning(
                "Biomass column '%s' not found in dataframe; curriculum will ignore biomass filters.",
                self.biomass_column,
            )
            self.biomass_column = None

    def set_epoch(self, epoch: int):
        """Set the current epoch for the sampler."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        stage = self._current_stage()
        mask = self._stage_mask(stage)

        indices = self.dataframe.index[mask].tolist()
        if not indices:
            indices = self.dataframe.index.tolist()

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self) -> int:
        stage = self._current_stage()
        mask = self._stage_mask(stage)
        num_samples = int(mask.sum())
        if num_samples == 0:
            num_samples = len(self.dataframe)
        return (num_samples + self.batch_size - 1) // self.batch_size

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _current_stage(self) -> CurriculumStage:
        for idx, boundary in enumerate(self.stage_boundaries):
            if self.epoch < boundary:
                return self.stages[idx]
        return self.stages[-1]

    def _stage_mask(self, stage: CurriculumStage) -> np.ndarray:
        mask = np.ones(len(self.dataframe), dtype=bool)
        if stage.fractal_range and "fractal_dimension" in self.dataframe.columns:
            min_fd, max_fd = stage.fractal_range
            mask &= (self.dataframe["fractal_dimension"] >= min_fd) & (
                self.dataframe["fractal_dimension"] < max_fd
            )

        if (
            stage.biomass_range
            and self.biomass_column
            and self.biomass_column in self.dataframe.columns
        ):
            min_bio, max_bio = stage.biomass_range
            mask &= (self.dataframe[self.biomass_column] >= min_bio) & (
                self.dataframe[self.biomass_column] < max_bio
            )

        return mask
