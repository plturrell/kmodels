# Hull Tactical Market Prediction Backlog

## Objective
- Forecast forward returns for Hull Tactical using enriched technical features, walk-forward validation, and both gradient boosting and neural pipelines reproducible outside Kaggle.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py` and `data/raw/` mirror Kaggle releases; `test_data_validation.py` prints schema checks.
- Feature engineering: `src/features/` builds lag/rolling indicators, RSI/MACD/Bollinger features, and metadata enrichment.
- Modeling: `src/train.py` (GBM-style baseline), `src/train_nn.py`, `src/train_tsmixer.py`, and `src/experimental/` adapters for Google Research TSMixer and Perceiver variants.
- Evaluation: Financial metrics in `src/utils/metrics.py`, cross-validation helpers inside training scripts, and outputs stored in `outputs/` (baseline, tabular_nn, tsmixer).
- Tooling: `scripts/run_next_steps.sh`, configuration flags in CLI entrypoints, but limited formal testing/automation.

## Open Questions
- How can we guarantee leakage-free feature pipelines when organisers update the schema or add new macro columns?
- What monitoring is needed to compare Sharpe/Sortino trajectories across different model families before submission?
- How should we prioritise GPU vs CPU experimentation when tabular nets and TSMixer have divergent resource requirements?

## Backlog

### P0 – Immediate
- Rewrite `test_data_validation.py` as pytest cases under `tests/` with assertions for column groups to integrate into CI.
- Add a lightweight regression dataset + config to run `python -m ...src.train --epochs 1` in under 2 minutes, verifying walk-forward splitting still works after code changes.
- Implement guard rails in `src/data/download.py` that compute hashes and store dataset version metadata in `outputs/dataset_manifest.json` for reproducible experiments.

### P1 – Mid-term
- Build a consolidated experiment tracker (e.g., JSONL or SQLite log) capturing Sharpe, drawdown, and configuration for each run in `outputs/`.
- Document the neural vs GBM playbooks (feature options, regularisation knobs) under `docs/` so newcomers can navigate the CLI surface quickly.
- Develop calibration notebooks comparing risk-adjusted metrics vs RMSE to inform which submissions are leaderboard-ready.

### P2 – Long-term
- Prototype macro regime detection features (e.g., clustering on `M*` columns) and integrate them via `src/features/` for conditional modeling.
- Explore GPU-accelerated LightGBM/CatBoost training and add wrappers in `src/modeling/` for larger-scale experimentation.
- Investigate multi-objective optimisation (Sharpe + turnover) using Optuna or internal search loops for combined metric gains.

## Quick Wins
- Add `make download-data`, `make train-baseline`, etc., to simplify README instructions.
- Generate automated EDA notebooks in `notebooks/` summarising missingness and feature drift before training.

