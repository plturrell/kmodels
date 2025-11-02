"""Analysis utilities for assessing learning behaviour in CSIRO biomass models."""

from .learning_assessment import (
    LearningAssessmentResult,
    aggregate_learning_score,
    compute_compression_test,
    compute_generalization_gap,
    compute_ensemble_diversity,
    run_learning_assessment,
)
from .domain_learning_tests import (
    DomainLearningReport,
    GroupStabilityResult,
    MetadataBiasResult,
    evaluate_group_stability,
    evaluate_metadata_bias,
    run_domain_learning_assessment,
)

__all__ = [
    "LearningAssessmentResult",
    "aggregate_learning_score",
    "compute_compression_test",
    "compute_generalization_gap",
    "compute_ensemble_diversity",
    "run_learning_assessment",
    "DomainLearningReport",
    "GroupStabilityResult",
    "MetadataBiasResult",
    "evaluate_group_stability",
    "evaluate_metadata_bias",
    "run_domain_learning_assessment",
]
