# CSIRO Biomass Pipeline Audit

## Key Modules

- `competitions/csiro_biomass/src/train.py`
  - Central experiment runner orchestrating k-fold training, curriculum sampling, Albumentations pipelines, and Lightning orchestration.
  - Provides utilities for metadata normalisation, category encoding, and fractal feature injection.
- `competitions/csiro_biomass/src/training/lightning_module.py`
  - Wraps models in `BiomassLightningModule`, handling SmoothL1 objective, EMA tracking, metric logging, and LR scheduling.
- `competitions/csiro_biomass/src/modeling/baseline.py`
  - Defines `ModelSpec`, fusion networks, and backbone conversion helpers for joint image/tabular regression.
- `competitions/csiro_biomass/src/data/dataset.py`
  - Supplies dataset classes returning (image, target, metadata, id) tuples plus paired sampling for physics-informed constraints.
- `competitions/csiro_biomass/src/utils/metrics.py`
  - Implements RMSE/MAE scoring utilities shared across training and validation.

## Reusable Patterns

1. **Configuration via dataclasses**  
   Experiment pieces (augmentations, fusion, optimizer) are codified as dataclasses, enabling type-safe configs that serialize cleanly.
2. **LightningModule abstraction**  
   Centralises loss computation, metric logging, and EMA weight averaging for any model that emits predictions compatible with the criterion.
3. **Backbone + metadata fusion**  
   Pluggable `ModelSpec` converts torchvision backbones and fuses tabular features through learned gates, a pattern reusable when both image and tabular features exist.
4. **Curriculum sampling hooks**  
   Custom sampler (`FractalCurriculumSampler`) is injected to the LightningModule, allowing epoch-dependent sampling without altering the loop.
5. **Post-processing pipeline**  
   Constraint processors encapsulate rule-based adjustments after model inference.

## Applicability to Forgery Detection

- **Direct fit**
  - Dataclass config pattern for organizing augmentation/model/training settings.
  - Lightning-style training wrapper to decouple optimization from data/model code.
  - Metric utilities (extend to classification + segmentation metrics).
- **Partial fit**
  - Backbone fusion: could adapt to support multi-head classification + segmentation outputs with optional metadata (e.g., artifact scores).
  - Curriculum sampler concept: adjust to prioritise forged samples with masks early in training.
- **Out of scope (for now)**
  - Fractal feature engineering and biomass-specific constraints.
  - Paired temporal dataset logic.

## Proposed Reuse Plan

1. Introduce a slim Lightning-style trainer tailored for joint segmentation/classification (leveraging the EMA + scheduler pattern).
2. Refactor baseline config into dataclass-driven configuration to match CSIRO structure.
3. Extend metrics utilities to compute Dice/IoU alongside classification accuracy, following CSIRO logging structure.
