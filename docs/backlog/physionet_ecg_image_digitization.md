# PhysioNet ECG Image Digitization Backlog

## Objective
- Digitise ECG waveforms from images by training Lightning models with advanced loss functions, post-processing, and submission tooling tuned for the PhysioNet competition.

## Current Toolkit Snapshot
- Data ingestion: `src/data/download.py` pulls Kaggle bundles into `data/raw/`; synthetic smoke scripts exist in the README.
- Feature engineering: `src/features/signal_processing.py` implements baseline correction, smoothing, and wavelet denoising.
- Modeling: `src/modeling/baseline.py` and related configs drive Lightning training; advanced losses appear in `src/modeling/losses.py`.
- Evaluation: Submission helper `src/utils/submission.py`, metrics and cross-validation harness inside `src/training` modules.
- Tooling: README smoke scripts, but no automated pytest coverage or manifests; outputs captured in `outputs/`.

## Open Questions
- How do we routinely verify that paired image/signal assets remain aligned after organiser updates?
- What automation is needed to benchmark different loss configurations and report reconstruction fidelity?
- How can we streamline large-image preprocessing to keep local iteration times reasonable?

## Backlog

### P0 – Immediate
- Build pytest fixtures mirroring the README synthetic dataset and add tests for `src/training/baseline.py` to ensure the pipeline runs end-to-end.
- Add checksum logging and manifest generation to the download helper to detect corrupted or refreshed PhysioNet assets.
- Implement automated validation comparing predicted vs ground-truth signals (MSE + signal metrics) for smoke runs, persisting results in `outputs/`.

### P1 – Mid-term
- Document recommended augmentation/loss combinations in `docs/` with guidance on when to enable frequency-domain losses.
- Provide notebooks/scripts for visualising reconstruction quality and post-processing effects using `src/features/signal_processing.py`.
- Introduce configurable dataclasses/Hydra configs to simplify switching between loss blends and model sizes.

### P2 – Long-term
- Explore diffusion-based or transformer models for sequence reconstruction as alternatives to the baseline CNN.
- Investigate mixed-precision training and GPU acceleration to reduce runtime for high-resolution inputs.
- Add automated hyperparameter search support (Optuna) focusing on loss-weight schedules and TTA variants.

## Quick Wins
- Move the README smoke script into `notebooks/` or `examples/` with comments explaining each step.
- Create a short troubleshooting guide covering common Kaggle download issues and large-file management tips.

