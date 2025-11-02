"""Adaptive ensemble with regime-aware weighting for market prediction."""

from .adaptive_ensemble import AdaptiveEnsemble, ModelDiagnostics
from .regime_detection import RegimeDetector
from .meta_features import MetaFeatureBuilder
from .dynamic_stacker import DynamicStacker
from .performance_monitor import PerformanceMonitor
from .competition_pipeline import CompetitionAdaptor
from .optimization_utils import (
    optimize_regime_thresholds,
    calculate_dynamic_blend_ratio,
    analyze_regime_specialization,
    get_fast_adaptation_config,
    get_stable_adaptation_config,
)
from .learning_assessment import (
    comprehensive_learning_assessment,
    generalization_gap_test,
    feature_stability_test,
    ensemble_diversity_test,
    kolmogorov_complexity_test,
    ood_generalization_test,
)
from .financial_learning_tests import FinancialLearningAssessment
from .extreme_stress_tests import ExtremeStressTester
from .causal_learning_tests import CausalLearningTester
from .adversarial_robustness_tests import AdversarialRobustnessTester
from .validation import (
    validate_submission,
    analyze_model_diversity,
    calculate_submission_confidence,
    calculate_regime_stability,
)
from .competition_utils import (
    apply_volatility_scaling,
    regime_performance_breakdown,
    analyze_leaderboard_feedback,
    create_enhanced_metadata,
)
from .advanced_position_sizing import AdvancedPositionSizer
from .signal_enhancement import SignalEnhancer
from .advanced_risk_management import AdvancedRiskManager
from .sharpe_optimization import SharpeOptimizer

__all__ = [
    "AdaptiveEnsemble",
    "ModelDiagnostics",
    "RegimeDetector",
    "MetaFeatureBuilder",
    "DynamicStacker",
    "PerformanceMonitor",
    "CompetitionAdaptor",
    "optimize_regime_thresholds",
    "calculate_dynamic_blend_ratio",
    "analyze_regime_specialization",
    "get_fast_adaptation_config",
    "get_stable_adaptation_config",
    "validate_submission",
    "analyze_model_diversity",
    "calculate_submission_confidence",
    "calculate_regime_stability",
    "apply_volatility_scaling",
    "regime_performance_breakdown",
    "analyze_leaderboard_feedback",
    "create_enhanced_metadata",
    "comprehensive_learning_assessment",
    "generalization_gap_test",
    "feature_stability_test",
    "ensemble_diversity_test",
    "kolmogorov_complexity_test",
    "ood_generalization_test",
    "FinancialLearningAssessment",
    "ExtremeStressTester",
    "CausalLearningTester",
    "AdversarialRobustnessTester",
    "AdvancedPositionSizer",
    "SignalEnhancer",
    "AdvancedRiskManager",
    "SharpeOptimizer",
]

