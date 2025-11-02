# CSIRO Image2Biomass Enhancements

This starter kit ships with a handful of practical boosters layered on top of the plain baseline. Each feature is implemented inside `competitions.csiro_biomass.src` and can be toggled through the training CLI.

## Test Time Augmentation

Module: `competitions.csiro_biomass.src.utils.tta`

- `TTAWrapper` wraps a trained PyTorch module and evaluates horizontal/vertical flips, then averages predictions.
- `MultiCropTTA` crops the image multiple times (centre + random crops) and aggregates the outputs.

Enable it by wrapping the `best_model.pt` checkpoint during inference.

## Advanced Loss Options

Module: `competitions.csiro_biomass.src.modeling.loss`

- `GaussianNLLLoss` enables uncertainty-aware training (mean + log variance).
- `PhysicsInformedLoss` demonstrates how to plug in domain heuristics.

You can swap the loss inside custom trainers; the Lightning pipeline exposes Smooth L1 with EMA by default.

## Stacking Ensemble Utilities

Module: `competitions.csiro_biomass.src.utils.ensemble`

- Helpers to average, rank-average, or meta-learn from multiple prediction CSVs.
- Used by `run_cv_ensemble.py` to blend fold outputs.

## Pseudo-Labeling Pipeline

Script: `competitions/csiro_biomass/src/pseudo_label.py`

- Loads a trained checkpoint, scores the public test set, filters high-confidence predictions, and appends them to `train.csv` to form a pseudo-labelled dataset.

## Perceiver Fusion Head

Module: `competitions.csiro_biomass.src.modeling.baseline`

- Toggle with `--fusion-type perceiver` to replace the standard MLP fusion between image embeddings and metadata.
- Configurable via `--perceiver-latents`, `--perceiver-layers`, and `--perceiver-heads`.

## Fractal Curriculum Sampler

Module: `competitions.csiro_biomass.src.data.sampler`

- `FractalCurriculumSampler` orders training samples by estimated fractal complexity (or biomass range) to ease optimisation.
- Controlled by `--curriculum-target` and curriculum stage settings within the CLI.

## Constraint-Aware Post-Processing

Modules:

- `competitions.csiro_biomass.src.postprocess.constraints.BiomassConstraintProcessor`
- `competitions.csiro_biomass.src.modeling.postprocessing.AdvancedBiomassConstraintProcessor`

They repair predictions to keep component totals consistent and enforce species-aware bounds before submission.

## Snapshot Ensembling and EMA

- Configure `--snapshot-count` to average the best Lightning checkpoints.
- EMA is enabled by default (see `OptimizerConfig.ema_decay`).

## Monte-Carlo Tree Search Scaffolding

Module: `competitions.csiro_biomass.src.search.mcts`

- `SimpleMCTS` mutates the `ExperimentConfig` and launches probe runs to explore alternative hyperparameters.

---

Use `python -m competitions.csiro_biomass.src.train --help` to discover the CLI flags for each feature.

