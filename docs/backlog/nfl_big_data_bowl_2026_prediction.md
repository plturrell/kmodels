# NFL Big Data Bowl 2026 Prediction Backlog

## Objective
- Produce high-quality player trajectory predictions using Liquid Networks, GNNs, and diffusion models while maintaining reproducible local experimentation aligned with Kaggle submissions.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py` (mirrors Kaggle bundles) populating `data/raw/`; leaderboard sync lives in `src/utils/leaderboard.py`.
- Feature engineering: `src/features/` builds trajectory datasets and feature transforms feeding multiple model families.
- Modeling: `train_liquid.py` plus `src/modeling` modules for Liquid Networks, GNNs, and diffusion-based models; cross-validation logic in `src/training`.
- Evaluation: Advanced submission generation via `src/utils/advanced_submission.py`, metrics helpers throughout `src/utils/`.
- Tooling: Requirements pinned via `requirements.txt`, but no automated tests or smoke configs yet; outputs stored under `outputs/`.

## Open Questions
- What minimal dataset slices can we use to validate Liquid/GNN pipelines without requiring the full Kaggle bundle?
- How should we compare Liquid vs GNN vs Diffusion runs in a consistent validation framework?
- Which parts of the analytics track tooling can be shared to reduce duplication (metrics, preprocessing, Optuna sweeps)?

## Backlog

### P0 – Immediate
- Create pytest-based schema checks for trajectory CSVs and integrate them into CI (currently no automated validation scripts).
- Introduce a smoke configuration for `train_liquid.py` that trains for 1 epoch on a handful of samples to ensure the training loop stays functional.
- Add dataset version manifests and checksum verification to the download helper to detect upstream bundle changes early.

### P1 – Mid-term
- Document modelling pathways (Liquid, GNN, Diffusion) in `docs/` with guidance on resource requirements and best-practice hyperparameters.
- Build evaluation scripts that compare ADE/FDE/miss-rate across model families and persist summaries in `outputs/metrics_history.json`.
- Factor shared preprocessing steps into reusable modules with the analytics track to avoid divergent feature definitions.

### P2 – Long-term
- Explore knowledge distillation or ensemble techniques combining Liquid, GNN, and diffusion predictions for leaderboard robustness.
- Investigate real-time inference optimisation (quantisation, TorchScript) for on-field automation scenarios.
- Add automated hyperparameter search (Optuna/Ray Tune) covering architecture-specific knobs with stored study artifacts.

## Quick Wins
- Publish a quickstart notebook demonstrating how to load a trained Liquid checkpoint and generate a submission via `src/utils/advanced_submission.py`.
- Add CLI examples to the README showing how to run cross-validation (`src/training/cross_validation.py`) for each model family.

