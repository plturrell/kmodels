"""Extreme Learning Assessment - 3x Harder Testing Framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ensemble.extreme_stress_tests import ExtremeStressTester
from ..ensemble.causal_learning_tests import CausalLearningTester
from ..ensemble.adversarial_robustness_tests import AdversarialRobustnessTester


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Extreme Learning Assessment (3x Harder) - Tests that separate genuine learning from pattern matching'
    )
    parser.add_argument(
        '--train-csv',
        type=Path,
        required=True,
        help='Training CSV file',
    )
    parser.add_argument(
        '--oof-prediction',
        type=Path,
        action='append',
        required=True,
        help='OOF prediction files (repeat for multiple models)',
    )
    parser.add_argument(
        '--target-column',
        default='forward_returns',
        help='Target column name',
    )
    parser.add_argument(
        '--id-column',
        default='date_id',
        help='ID column name',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/extreme_assessment'),
        help='Output directory for assessment results',
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=0.99,
        help='Confidence level for statistical tests (default: 0.99)',
    )
    parser.add_argument(
        '--attack-budget',
        type=float,
        default=0.15,
        help='Adversarial attack budget (default: 0.15)',
    )
    return parser


def run_extreme_assessment(
    model_predict_fn,
    data: Dict,
    predictions: Dict[str, pd.Series],
    config: Dict,
) -> Dict:
    """Run the extreme 3x harder learning assessment."""
    
    print("🚀 STARTING EXTREME LEARNING ASSESSMENT (3x HARDER)")
    print("=" * 70)
    print()
    
    results = {}
    
    # Initialize testers
    stress_tester = ExtremeStressTester(confidence_level=config.get('confidence_level', 0.99))
    causal_tester = CausalLearningTester(alpha=0.01)  # Stricter alpha
    adversarial_tester = AdversarialRobustnessTester(
        attack_budget=config.get('attack_budget', 0.15)
    )
    
    # 1. Extreme Stress Tests
    print("🧪 Running Extreme Stress Tests...")
    shock_scenarios = [
        {
            'name': 'volatility_explosion',
            'type': 'volatility_explosion',
            'returns': data['y'],
        },
        {'name': 'flash_crash', 'type': 'flash_crash', 'returns': data['y']},
    ]
    
    results['stress_tests'] = stress_tester.adversarial_regime_shock_test(
        predictions, data['y'], shock_scenarios
    )
    print(f"   ✅ Stress resilience: {results['stress_tests']['overall_resilience']:.3f}")
    print()
    
    # 2. Multi-Scale Consistency
    print("⏰ Testing Multi-Scale Consistency...")
    if len(predictions) > 0:
        first_model_preds = list(predictions.values())[0]
        results['multi_scale'] = stress_tester.multi_scale_consistency_test(
            first_model_preds, data['y']
        )
        print(f"   ✅ Multi-scale consistency: {results['multi_scale']['multi_scale_consistency']:.3f}")
    else:
        results['multi_scale'] = {'multi_scale_consistency': 0.0, 'passed_multi_scale': False}
    print()
    
    # 3. Information Theoretic Learning
    print("📊 Running Information Theoretic Tests...")
    if len(predictions) > 0 and 'X' in data:
        first_model_preds = list(predictions.values())[0]
        results['information_theory'] = stress_tester.information_theoretic_learning_test(
            first_model_preds, data['y'], data.get('X')
        )
        print(f"   ✅ Learning efficiency: {results['information_theory']['learning_efficiency_score']:.3f}")
    else:
        results['information_theory'] = {
            'learning_efficiency_score': 0.0,
            'efficient_learner': False,
        }
    print()
    
    # 4. Causal Learning Tests
    print("🔍 Testing Causal Discovery...")
    
    if 'X' in data and len(data['X']) > 0:
        # Create synthetic environment labels for ICP test
        n_samples = len(data['X'])
        np.random.seed(42)  # For reproducibility
        environment_labels = pd.Series(
            np.random.choice(['env1', 'env2', 'env3'], n_samples),
            index=data['X'].index,
        )
        
        results['causal_learning'] = causal_tester.invariant_causal_prediction_test(
            data['X'], data['y'], environment_labels
        )
        print(f"   ✅ Causal feature ratio: {results['causal_learning']['causal_feature_ratio']:.3f}")
        
        # Intervention response test
        print("   🔬 Testing intervention responses...")
        results['intervention_test'] = causal_tester.intervention_response_test(
            model_predict_fn, data['X'], data['y']
        )
        print(f"   ✅ Causal consistency: {results['intervention_test']['causal_consistency_score']:.3f}")
    else:
        results['causal_learning'] = {'causal_feature_ratio': 0.0, 'has_causal_features': False}
        results['intervention_test'] = {'causal_consistency_score': 0.0, 'causally_consistent': False}
    print()
    
    # 5. Adversarial Robustness
    print("🛡️ Testing Adversarial Robustness...")
    
    if 'X' in data:
        # Calculate feature importance for targeted attacks (simplified: use variance)
        feature_importance = {i: float(data['X'].iloc[:, i].var()) for i in range(min(50, data['X'].shape[1]))}
        
        results['adversarial_robustness'] = adversarial_tester.worst_case_feature_perturbation_test(
            model_predict_fn, data['X'], data['y'], feature_importance
        )
        print(f"   ✅ Adversarial robustness: {results['adversarial_robustness']['overall_robustness']:.3f}")
        
        # Temporal adversarial attacks
        print("   ⏱️ Testing temporal adversarial attacks...")
        results['temporal_robustness'] = adversarial_tester.temporal_adversarial_attack_test(
            model_predict_fn, data['X'], data['y']
        )
        print(f"   ✅ Temporal robustness: {results['temporal_robustness']['temporal_robustness']:.3f}")
    else:
        results['adversarial_robustness'] = {'overall_robustness': 0.0, 'adversarially_robust': False}
        results['temporal_robustness'] = {'temporal_robustness': 0.0, 'temporally_robust': False}
    print()
    
    # Calculate Extreme Learning Score
    extreme_scores = []
    
    # Stress test score (normalize to 0-1 range - cap at reasonable max)
    stress_score = results['stress_tests']['overall_resilience']
    stress_score_normalized = min(1.0, stress_score / 10.0)  # Cap at 10x for normalization
    extreme_scores.append(stress_score_normalized)
    
    # Multi-scale score
    extreme_scores.append(results['multi_scale']['multi_scale_consistency'])
    
    # Information theory score
    extreme_scores.append(results['information_theory']['learning_efficiency_score'])
    
    # Causal learning score (average of ICP and intervention)
    causal_score = (
        results['causal_learning']['causal_feature_ratio'] * 0.5
        + results['intervention_test']['causal_consistency_score'] * 0.5
    )
    extreme_scores.append(causal_score)
    
    # Adversarial robustness score (average of feature and temporal)
    adv_score = (
        results['adversarial_robustness']['overall_robustness'] * 0.5
        + results['temporal_robustness']['temporal_robustness'] * 0.5
    )
    extreme_scores.append(adv_score)
    
    extreme_learning_score = float(np.mean(extreme_scores))
    
    # Final verdict
    if extreme_learning_score > 0.8:
        verdict = "🎉 EXTREME LEARNING CONFIRMED - GENUINE INTELLIGENCE"
        confidence = "VERY HIGH"
    elif extreme_learning_score > 0.7:
        verdict = "✅ STRONG LEARNING EVIDENCE - PRODUCTION READY"
        confidence = "HIGH"
    elif extreme_learning_score > 0.6:
        verdict = "⚠️ MODERATE LEARNING - NEEDS MONITORING"
        confidence = "MODERATE"
    else:
        verdict = "❌ WEAK LEARNING - LIKELY PATTERN MATCHING"
        confidence = "LOW"
    
    results['extreme_learning_score'] = extreme_learning_score
    results['verdict'] = verdict
    results['confidence'] = confidence
    results['extreme_scores_breakdown'] = {
        'stress_resilience': extreme_scores[0],
        'multi_scale': extreme_scores[1],
        'information_efficiency': extreme_scores[2],
        'causal_discovery': extreme_scores[3],
        'adversarial_robustness': extreme_scores[4],
    }
    
    return results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args()
    
    # Load training data
    print("📁 Loading data...")
    train_df = pd.read_csv(args.train_csv)
    feature_cols = [
        c
        for c in train_df.columns
        if c not in [args.id_column, args.target_column, "risk_free_rate", "market_forward_excess_returns"]
    ]
    
    # Load OOF predictions
    predictions_dict = {}
    for oof_file in args.oof_prediction:
        oof_df = pd.read_csv(oof_file)
        merged = train_df.merge(oof_df, on=args.id_column)
        
        model_name = Path(oof_file).parent.parent.name if len(args.oof_prediction) > 1 else 'model'
        
        if f"{args.target_column}_oof" in merged.columns:
            predictions_dict[model_name] = merged[f"{args.target_column}_oof"]
    
    # Create simple model predict function (average ensemble)
    def model_predict_fn(X: pd.DataFrame) -> pd.Series:
        """Simple prediction function using average of models."""
        if len(predictions_dict) == 0:
            return pd.Series(0.0, index=X.index[:len(X)])
        # Return average predictions aligned with X
        avg_pred = pd.concat(list(predictions_dict.values()), axis=1).mean(axis=1)
        # Align with X index (sample if needed)
        if len(avg_pred) >= len(X):
            return avg_pred.iloc[:len(X)].values
        else:
            # Repeat if needed
            n_repeats = (len(X) // len(avg_pred)) + 1
            repeated = pd.concat([avg_pred] * n_repeats)
            return repeated.iloc[:len(X)].values
    
    # Prepare data
    data = {
        'X': train_df[feature_cols] if feature_cols else None,
        'y': train_df[args.target_column],
    }
    
    config = {
        'confidence_level': args.confidence_level,
        'attack_budget': args.attack_budget,
    }
    
    # Run extreme assessment
    results = run_extreme_assessment(model_predict_fn, data, predictions_dict, config)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    results_file = args.output_dir / 'extreme_assessment_results.json'
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    # Print summary
    print()
    print("=" * 70)
    print("📊 EXTREME LEARNING ASSESSMENT RESULTS")
    print("=" * 70)
    print()
    print(f"🎯 Extreme Learning Score: {results['extreme_learning_score']:.3f}")
    print(f"📈 Verdict: {results['verdict']}")
    print(f"🎲 Confidence: {results['confidence']}")
    print()
    print("Breakdown:")
    for test, score in results['extreme_scores_breakdown'].items():
        print(f"  {test}: {score:.3f}")
    print()
    print(f"📁 Results saved to: {results_file}")
    print()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

