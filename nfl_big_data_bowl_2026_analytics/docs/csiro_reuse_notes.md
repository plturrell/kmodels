# CSIRO Pipeline Audit

The CSIRO Image2Biomass workspace contains several reusable pieces we can port into the NFL analytics project.
Key components and their locations:

- **Configuration dataclasses** – encode backbone, augmentation, optimiser, and dataset handles (`competitions/csiro_biomass/src/config/experiment.py:5`).  These inspired the new `nfl_big_data_bowl_2026_analytics/src/config/experiment.py`.
- **Data assembly & metadata enrichment** – pivots multi-target CSVs, injects fractal/metadata features, and normalises splits (`competitions/csiro_biomass/src/train.py:358`).  The pattern of returning `(table, target_names, metadata_columns, stats, category_maps)` maps cleanly to assembling weekly NFL inputs/outputs.
- **Augmentation builder** – wraps Albumentations transforms to deliver paired train/val pipelines (`competitions/csiro_biomass/src/train.py:448`).  While NFL analytics is tabular, the abstraction guides how to centralise preprocessing later.
- **Dataset abstractions** – image dataset plus metadata-aware loader and curriculum-aware sampler support (`competitions/csiro_biomass/src/data/dataset.py:25`, `competitions/csiro_biomass/src/data/sampler.py:17`).  The sampler idea can become a drive/drive sequencing sampler for NFL frames.
- **LightningModule** – centralises loss, EMA averaging, logging, and scheduler wiring (`competitions/csiro_biomass/src/training/lightning_module.py:17`).  We can reuse the structure with a sequence model instead of EfficientNet.
- **Constraint processors** – enforce domain rules and provide evaluation deltas (`competitions/csiro_biomass/src/postprocess/constraints.py:16`, `competitions/csiro_biomass/src/modeling/postprocessing.py:10`).  Similar repair hooks can clip unrealistic trajectory predictions.
- **Checkpoint averaging & CV tooling** – snapshot ensembling and multi-fold helper functions ready to repurpose (`competitions/csiro_biomass/src/train.py:1473`, `competitions/csiro_biomass/src/train.py:1516`).
- **Search scaffold** – lightweight MCTS driver to mutate configs and record probe scores (`competitions/csiro_biomass/src/search/mcts.py:11`).  Works unchanged once the analytics runner exposes a compatible interface.

## Immediate Follow-Up

1. **Config surface** – Added `src/config/experiment.py` with dataset/model/optimiser dataclasses so downstream tooling can mirror `ExperimentConfig`.
2. **Shared metrics** – Ported `compute_metrics` into `src/utils/metrics.py`, matching the Lightning logging API.
3. **Documentation** – This note captures module pointers to speed up future porting (e.g., building a Lightning module that imports `TrainingConfig`).
4. **Training CLI** – Implemented `src/train.py` with Lightning scaffolding, cross-validation helpers, and checkpoint averaging, reusing the CSIRO patterns.
5. **Feature flags** – Added distance-to-landing and game-clock feature toggles to mirror relational/temporal ideas from graph nets and standard DeepMind perception stacks.
6. **Architectures** – Introduced residual MLP and Perceiver-style latent attention heads selectable via CLI to experiment with deeper models inspired by DeepMind research.

## Suggested Next Steps

1. Implement a `LightningModule` skeleton that consumes `TrainingConfig` (reuse EMA + logging behavior from CSIRO).
2. Draft a weekly dataloader that mirrors `prepare_training_dataframe` for the NFL CSV layout, including metadata normalisation and sequence batching.
3. Port the checkpoint averaging helper and start wiring a baseline trainer, enabling future integration with the MCTS search scaffold.
