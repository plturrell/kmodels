# CAFA 6 Protein Function Prediction Backlog

## Objective
- Deliver high-quality protein function predictions for CAFA 6 using ESM-2 embeddings, fractal features, and Lightning-based training pipelines.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py`, `src/data/go_ontology.py`, `src/data/augmentation.py` provide Kaggle sync, GO ontology propagation, and augmentation utilities.
- Feature engineering: `src/features/embedding_cache.py`, `src/features/fractal_benchmark.py`, and related helpers cache transformer embeddings and benchmark fractal descriptors.
- Modeling: `train_neural.py`, `src/modeling/attention_model.py`, `src/modeling/ensemble.py`, and `src/training/cross_validation.py` cover Lightning baselines, attention heads, ensembles, and stratified CV.
- Evaluation: `benchmark_fractal.py`, `src/utils/information_content.py`, and `tests/test_features.py` implement CAFA-aligned metrics, IC scoring, and feature sanity checks.
- Tooling: `configs/*.yaml`, `tests/test_data.py`, `IMPROVEMENTS.md`, and the `outputs/` directory capture experiment configs, pytest coverage, and historical runs.

## Open Questions
- How do we guard GO ontology parsing against upstream schema changes before competition milestones?
- Which combination of transformer embeddings + fractal features yields the most stable leaderboard gains across cross-validation folds?
- What level of automation is needed to keep embedding caches and benchmark reports reproducible for new data drops?

## Backlog

### P0 – Immediate
- Add checksum and file-size validation to `src/data/download.py` so Kaggle downloads are verified before extraction (prevents silent corruption in `data/raw`).
- Extend `tests/test_data.py` with a regression fixture that loads a trimmed GO annotation/sample FASTA bundle to ensure `src/data/go_ontology.py` propagation stays aligned with the latest release.

### P1 – Mid-term
- Expose the active learning loop in `src/utils/active_learning.py` as a CLI (`python -m ...`) and document sampling strategies so uncertain proteins can be queued automatically.
- Produce experiment playbooks under `docs/` describing how to combine `configs/fractal.yaml` with `benchmark_fractal.py` and Lightning training for reproducible faction-specific runs.
- Introduce validation curves/threshold calibration reporting in `src/utils/information_content.py` to monitor metric sensitivity before leaderboard submissions.

### P2 – Long-term
- Research hybrid structural featurisation (e.g., AlphaFold pocket embeddings) and prototype integration alongside current `src/features/*` pipelines.
- Investigate multi-GPU or distributed training checkpoints within `train_neural.py` to shorten turnaround time for full CV sweeps.
- Explore semi-supervised label propagation using the ontology graph to pre-score unlabeled sequences before active learning loops.

## Quick Wins
- Backfill `docs/` with a short “Run Checklist” linking README steps, config presets, and benchmark commands.
- Publish a canonical notebook in `notebooks/` illustrating how to inspect embedding caches (`src/features/embedding_cache.py`) for reuse across experiments.

