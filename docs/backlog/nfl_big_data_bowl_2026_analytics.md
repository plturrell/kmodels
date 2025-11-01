# NFL Big Data Bowl 2026 Analytics Backlog

## Objective
- Analyse route trajectories and enriched metadata for the analytics track, leveraging Lightning training loops, advanced metrics, and reusable CSIRO-inspired infrastructure.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py` and `data/raw/nfl-big-data-bowl-2026-analytics/...` manage Kaggle assets; `test_data_validation.py` prints schema checks.
- Feature engineering: `src/features/` creates kinematic and relational features, including distance-to-landing toggles described in `README.md`.
- Modeling: `src/modeling/` includes residual MLPs, Perceiver heads, and GAT variants; `src/train.py` orchestrates Lightning training with snapshot ensembling.
- Evaluation: Advanced metrics available via `src/utils/advanced_metrics.py`, plus submission validation in `src/utils/validation.py`.
- Tooling: `docs/csiro_reuse_notes.md` documents cross-project reuse; `src/optimize.py` enables Optuna sweeps; outputs stored under `outputs/baseline/`.

## Open Questions
- How can we keep large CSV bundles manageable (chunked loading, caching) while maintaining reproducible preprocessing?
- What guardrails ensure GAT/graph dependencies (e.g., `torch_geometric`) stay consistent across environments?
- Which metrics (FDE, ADE, miss rate) best correlate with leaderboard performance, and how should they be tracked automatically?

## Backlog

### P0 – Immediate
- Convert `test_data_validation.py` into pytest-based schema checks and wire them into CI for early detection of organiser updates.
- Add a small fixture dataset + config enabling `python -m ...src.train --epochs 1 --weeks 2023_01` smoke tests to validate the Lightning stack.
- Implement dataset manifest/checksum logging in `src/data/download.py` so large Kaggle bundles are versioned and verified.

### P1 – Mid-term
- Document the residual MLP vs Perceiver vs GAT trade-offs in `docs/` with recommended hyperparameters and compute budgets.
- Build evaluation scripts that aggregate FDE/ADE/miss-rate trends per week and surface them in `outputs/baseline/metrics_history.json`.
- Package Optuna/Hyperparameter sweeps with saved study summaries to track which features/architectures outperform baselines.

### P2 – Long-term
- Explore integrating player tracking heatmaps or graph-based features requiring preprocessing pipelines shared with the prediction track.
- Investigate distributed training or mixed precision setups for faster experimentation on larger sequences.
- Research sequential ensembling (e.g., stacking analytics + prediction models) to reuse learned representations across contests.

## Quick Wins
- Add shell aliases or a simple `Makefile` to run download, train, and evaluate commands from the README automatically.
- Provide a notebook template summarising advanced metrics from `src/utils/advanced_metrics.py` for quick inspection before submission.

