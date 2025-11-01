"""Skeleton MCTS driver for adaptive hyperparameter search.

This is intentionally lightweight: it samples candidate ExperimentConfig objects,
executes short probe runs, and records rewards so future searches can reuse the
best settings. Full MCTS expansion/backup logic can be plugged in later.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

from ..config.experiment import AugmentationConfig, BackboneConfig, ExperimentConfig, OptimizerConfig
from ..train import run_experiment


class CandidatePool:
    """Maintains the best discovered configurations and simple expansion logic."""

    def __init__(self, max_candidates: int = 5) -> None:
        self.max_candidates = max_candidates
        self.candidates: List[Dict[str, object]] = []

    def add(self, config: ExperimentConfig, reward: float, run_dir: Path) -> None:
        self.candidates.append({"config": config, "reward": reward, "run_dir": str(run_dir)})
        self.candidates.sort(key=lambda item: item["reward"], reverse=True)
        if len(self.candidates) > self.max_candidates:
            self.candidates.pop()

    def best(self) -> Optional[ExperimentConfig]:
        if not self.candidates:
            return None
        return self.candidates[0]["config"]


class SimpleMCTS:
    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.pool = CandidatePool()

    def _mutate_config(self, base: ExperimentConfig) -> ExperimentConfig:
        """Create a neighbouring configuration by mutating augmentation/backbone."""
        backbones = ["efficientnet_b3", "efficientnet_b0", "convnext_tiny", "resnet34"]
        candidates = [b for b in backbones if b != base.backbone.name]
        new_backbone = self.rng.choice(candidates)
        backbone_cfg = replace(base.backbone, name=new_backbone)

        aug_cfg = replace(
            base.augmentation,
            image_size=self.rng.choice([320, 352, 380]),
        )

        opt_cfg = replace(
            base.optimizer,
            learning_rate=self.rng.choice([5e-4, 1e-3, 2e-3]),
        )

        return replace(
            base,
            backbone=backbone_cfg,
            augmentation=aug_cfg,
            optimizer=opt_cfg,
        )

    def propose(self, base_config: ExperimentConfig, n: int = 4) -> List[ExperimentConfig]:
        proposals = []
        for _ in range(n):
            proposals.append(self._mutate_config(base_config))
        return proposals

    def evaluate(self, config: ExperimentConfig, probe_epochs: int = 3) -> Dict[str, object]:
        probe_config = replace(config, epochs=probe_epochs)
        result = run_experiment(probe_config)
        reward = -result.get("val_rmse", float("inf"))
        self.pool.add(config, reward, Path(result["run_dir"]))
        return {"result": result, "reward": reward}

    def run_search(
        self,
        seed_config: ExperimentConfig,
        rounds: int = 3,
        probe_epochs: int = 3,
    ) -> ExperimentConfig:
        current_best = seed_config
        for _ in range(rounds):
            for proposal in self.propose(current_best):
                outcome = self.evaluate(proposal, probe_epochs=probe_epochs)
                if outcome["reward"] > -float("inf"):
                    current_best = proposal
        best_config = self.pool.best() or seed_config
        return best_config

    def export(self, path: Path) -> None:
        payload = [
            {
                "reward": item["reward"],
                "run_dir": item["run_dir"],
                "config": item["config"].to_dict(),
            }
            for item in self.pool.candidates
        ]
        path.write_text(json.dumps(payload, indent=2))
