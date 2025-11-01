# MABe Mouse Behavior Detection Backlog

## Objective
- Detect mouse behaviours from pose sequences by reimplementing and extending GRU/Transformer/TCN baselines with Lightning training loops suitable for local development.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py` (Kaggle mirror) and large pose `.npz` assets under `data/raw/`.
- Feature engineering: Pose augmentation and preprocessing utilities in `src/data/processing.py` and temporal augmentation hooks in `src/data/augmentation.py`.
- Modeling: Lightning baselines in `src/training/baseline.py`, advanced architectures within `src/modeling/advanced.py`, and ensemble helpers in `src/modeling`.
- Evaluation: Metrics and leaderboard helper scripts inside `src/utils/`, with outputs captured under `outputs/baseline/`.
- Tooling: No formal pytest suite; documentation limited to the README; `docs/` is currently empty.

## Open Questions
- How can we create small deterministic fixtures to exercise the Lightning training loop without requiring the full Kaggle dataset?
- Which augmentation strategies provide the best boost per behaviour class, and how should we measure that beyond simple accuracy/F1?
- What infrastructure is needed to manage the heavy `.parquet` pose files (caching, streaming, GPU vs CPU pipelines)?

## Backlog

### P0 – Immediate
- Create synthetic pose fixtures (similar to README smoke script) under `tests/` and add pytest coverage for `src/training/baseline.py` and `src/modeling/advanced.py` forward passes.
- Implement a minimal `smoke.yaml`/CLI flag that limits sequence length and dataset size so CI can run a 1-epoch Lightning check.
- Add checksum/version logging to the Kaggle download helper to ensure large parquet bundles are validated before training.

### P1 – Mid-term
- Document architectural knobs (Transformer/TCN/GRU) and recommended hyperparameters under `docs/behavior_modeling.md` for onboarding.
- Build evaluation notebooks calculating per-class F1 and confusion matrices using outputs stored in `outputs/`.
- Introduce configurable data loaders that stream from disk to reduce RAM pressure when experimenting on laptops.

### P2 – Long-term
- Explore self-supervised pretraining (e.g., contrastive sequence models) on unlabeled pose data before fine-tuning the classifier.
- Investigate real-time inference optimization (TorchScript/ONNX) using `src/utils/submission.py` pipelines.
- Add active learning or curriculum sampling modules to prioritise misclassified behaviours during retraining.

## Quick Wins
- Populate the empty `docs/` directory with a quickstart mirroring the README plus tips for large-file handling.
- Provide a `notebooks/` template showing how to visualise pose sequences and augmentations for debugging.

