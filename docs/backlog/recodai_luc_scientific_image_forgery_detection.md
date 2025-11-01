# RecoDAI LUC Scientific Image Forgery Detection Backlog

## Objective
- Detect scientific image forgeries via joint segmentation + classification models, advanced augmentations, and ensemble tooling aligned with Kaggle submissions.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py` mirrors Kaggle assets; dataset inspection lives in `src/data/summary.py` and stores outputs under `outputs/dataset_overview.json`.
- Feature engineering: `src/data/transforms.py` applies Albumentations pipelines; tabular feature extraction at `src/features/tabular.py`.
- Modeling: `src/modeling/baseline.py`, `src/modeling/pretrained.py`, and `src/modeling/ensemble.py` support Lightning baselines, pretrained encoders, and ensembling.
- Evaluation: `src/utils/submission.py` handles TTA submissions; metrics tracked by `src/utils/metrics.py`.
- Tooling: Lightning training orchestrated via `src/train.py`, cross-validation through `src/training/cross_validation.py`, and documentation cross-linking to CSIRO patterns in `docs/csiro_audit.md`.

## Open Questions
- How do we guarantee segmentation masks and images stay aligned when new data arrives or augmentations change?
- What automated benchmarks compare segmentation + classification metrics per fold to justify ensemble choices?
- Which aspects of the CSIRO pipeline should be further modularised for reuse without duplicating maintenance?

## Backlog

### P0 – Immediate
- Create pytest fixtures for a miniature image/mask dataset to validate `src/modeling/baseline.py` forward passes and `src/utils/submission.py` formatting.
- Add dataset manifest + checksum logging to the download helper to detect refreshed Kaggle bundles early.
- Implement integration tests ensuring `src/data/transforms.py` maintains mask alignment after augmentation (e.g., random crops/rotations).

### P1 – Mid-term
- Document segmentation vs classification experiment setups in `docs/` with guidance on when to enable TTA and ensemble weighting.
- Build evaluation scripts summarising Dice/IoU and classification accuracy per fold, storing aggregates in `outputs/metrics_summary.json`.
- Provide recipes for combining tabular features with image backbones, including sample configs or CLI flags.

### P2 – Long-term
- Explore diffusion or generative adversarial approaches to detect subtle forgeries and integrate them into the ensemble pipeline.
- Investigate active learning strategies that prioritise uncertain masks for manual inspection.
- Add hyperparameter optimisation workflows (Optuna/Ray) targeting augmentation policies and loss weighting.

## Quick Wins
- Publish a short “how-to” note describing the dataset summary command (`src/data/summary.py`) and how to interpret its output JSON.
- Add CLI wrappers or scripts to regenerate submissions from stored predictions without re-running full inference.

