# CSIRO Image2Biomass Backlog

## Objective
- Predict biomass composition from canopy imagery and metadata with reliable, repeatable local training pipelines that mirror the Kaggle competition workflow.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py`, curated `data/raw/` mirrors Kaggle bundles, and `test_data_validation.py` provides manual checks.
- Feature engineering: `src/features/metadata.py`, curriculum sampler logic, and fractal feature hooks embedded in `src/train.py`.
- Modeling: `src/modeling/baseline.py`, Perceiver fusion options, snapshot ensembling, and Lightning orchestration within `src/train.py`/`src/training`.
- Evaluation: Constraint repair in `src/postprocess/constraints.py`, metrics under `src/utils/metrics.py`, and ablation tooling via `src/ablation_runner.py`.
- Tooling: MCTS search (`src/search/mcts.py`), pseudo-labelling helper, numerous outputs in `outputs/`, and `QUICK_START.md` for newcomers.

## Open Questions
- How can we continuously validate that metadata pipelines remain consistent when organisers refresh contextual files?
- What automation is required to keep TTA, EMA, and snapshot ensembling reproducible across cross-validation folds?
- Which pieces of the CSIRO stack should be abstracted for reuse in other competitions without duplicating maintenance effort?

## Backlog

### P0 – Immediate
- Convert `test_data_validation.py` into pytest cases under `tests/` so dataset schema checks run in CI rather than only via print statements.
- Add a deterministic smoke configuration (e.g., `configs/smoke.yaml`) that executes `src/train.py` for 1 epoch on a reduced dataset to confirm the Lightning pipeline still launches on each code change.
- Implement metadata version manifest + checksum logging in `src/data/download.py` to detect organiser updates before training on stale CSVs.

### P1 – Mid-term
- Refactor augmentation/controller options in `src/train.py` into dataclasses (mirroring `src/config/experiment.py`) to simplify experiment diffs.
- Document the search workflow (MCTS + ablation runner) under `docs/` with examples of promoting probe configs into full training runs.
- Build automated evaluation notebooks/scripts that track RMSE improvements when toggling `--fusion-type perceiver` and constraint repair.

### P2 – Long-term
- Explore integrating hyperspectral or vegetation indices as additional modalities by extending `src/data/dataset.py` and fusion heads.
- Investigate ONNX/TensorRT export validation by adding regression tests around `src/export.py` for different hardware targets.
- Research active curriculum scheduling using live leaderboard deltas to reweight underperforming biomass classes.

## Quick Wins
- Add `Makefile` or shell aliases invoking common commands from `README.md` (download, train, export) to reduce manual typing.
- Publish a short guide in `docs/` explaining how to interpret `outputs/baseline/run-*/validation_constraint_metrics.json` for constraint compliance.

