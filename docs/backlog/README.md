# Competition Backlog Overview

This directory houses per-project backlog roadmaps for the eight active Kaggle workspaces. Each backlog captures the project objective, current toolkit snapshot, open questions, and a prioritised task list (P0/P1/P2) plus quick wins.

## Backlog Index
- [CAFA 6 Protein Function Prediction](cafa_6_protein_function_prediction.md)
- [CSIRO Image2Biomass](csiro_biomass.md)
- [Hull Tactical Market Prediction](hull_tactical_market_prediction.md)
- [MABe Mouse Behavior Detection](mabe_mouse_behavior_detection.md)
- [NFL Big Data Bowl 2026 Analytics](nfl_big_data_bowl_2026_analytics.md)
- [NFL Big Data Bowl 2026 Prediction](nfl_big_data_bowl_2026_prediction.md)
- [PhysioNet ECG Image Digitization](physionet_ecg_image_digitization.md)
- [RecoDAI LUC Scientific Image Forgery Detection](recodai_luc_scientific_image_forgery_detection.md)

## Cross-Project Themes
- **Data integrity first**: Every project needs automated schema checks (convert existing validation scripts into pytest) and dataset manifests with checksums inside respective `src/data/download.py` modules.
- **Smaller smoke configs**: Introducing deterministic, reduced-size training configs enables CI or pre-commit smoke tests for Lightning/GBM pipelines across competitions.
- **Documentation gaps**: Most projects rely on rich READMEs but lack deeper docs; the backlogs call for playbooks explaining architecture choices, augmentation strategies, and experiment workflows.
- **Metrics + tracking**: Establishing run logs (JSON/CSV) and metric summaries in `outputs/` will help compare experiments over time, especially for financial (Sharpe/Sortino) and trajectory (ADE/FDE) metrics.
- **Search & reuse**: Several workspaces borrow ideas from CSIRO (dataclasses, EMA, search). Centralising reusable components or documenting reuse pathways avoids divergence.

## Recommended Sequencing
1. **Stabilise foundations (Weeks 1-2)**
   - Implement dataset manifests + checksum verification across all download helpers.
   - Convert existing `test_data_validation.py` scripts into pytest suites and add synthetic fixtures for projects lacking tests.
   - Add smoke configs enabling quick end-to-end runs (1 epoch / small sample) for Lightning trainers.
2. **Improve observability (Weeks 3-4)**
   - Build metric tracking utilities and evaluation summaries saved alongside `outputs/`.
   - Document core workflows in `docs/` (training playbooks, augmentation guides, architecture comparisons).
   - Wire leaderboard or validation scripts into repeatable notebooks/scripts for rapid iteration.
3. **Advance modelling & search (Weeks 5+)**
   - Pursue project-specific P1/P2 items: new modalities (CSIRO), hybrid features (CAFA), self-supervised pose pretraining (MABe), trajectory ensembling (NFL prediction), etc.
   - Integrate automated hyperparameter search or Optuna studies where called out.
   - Explore cross-project synergies (e.g., sharing metrics utilities between analytics/prediction tracks).

## Next Steps
- Review each project backlog to assign owners and timelines for the P0 items.
- Stand up CI (GitHub Actions or local pre-commit) to execute the new smoke tests and schema checks once implemented.
- Iterate on documentation as tasks complete so the backlogs stay current and actionable.

