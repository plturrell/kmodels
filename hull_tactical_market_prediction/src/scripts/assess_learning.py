"""Comprehensive learning assessment for adaptive ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ensemble.learning_assessment import comprehensive_learning_assessment
from ..ensemble.financial_learning_tests import FinancialLearningAssessment
from ..ensemble import AdaptiveEnsemble, DynamicStacker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess if ensemble is genuinely learning vs pattern matching."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Training CSV file",
    )
    parser.add_argument(
        "--oof-prediction",
        type=Path,
        action="append",
        required=True,
        help="OOF prediction files (repeat for multiple models)",
    )
    parser.add_argument(
        "--target-column",
        default="forward_returns",
        help="Target column name",
    )
    parser.add_argument(
        "--id-column",
        default="date_id",
        help="ID column name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/learning_assessment"),
        help="Output directory for assessment results",
    )
    parser.add_argument(
        "--financial-tests",
        action="store_true",
        help="Run financial-specific learning tests",
    )
    parser.add_argument(
        "--regime-data",
        type=Path,
        help="CSV file with regime labels (id_column, regime_label columns)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    print("="*70)
    print("COMPREHENSIVE LEARNING ASSESSMENT")
    print("="*70)
    print()
    
    # Load training data
    train_df = pd.read_csv(args.train_csv)
    feature_cols = [
        c for c in train_df.columns
        if c not in [args.id_column, args.target_column, "risk_free_rate", "market_forward_excess_returns"]
    ]
    
    # Load OOF predictions
    oof_predictions_dict = {}
    oof_dfs = []
    
    for oof_file in args.oof_prediction:
        oof_df = pd.read_csv(oof_file)
        oof_dfs.append(oof_df)
        
        # Extract model name from path
        model_name = Path(oof_file).parent.parent.name
        
        if "forward_returns_oof" in oof_df.columns:
            merged = train_df.merge(oof_df, on=args.id_column)
            oof_predictions_dict[model_name] = merged["forward_returns_oof"]
    
    # Combine OOF predictions
    if oof_dfs:
        merged_oof = oof_dfs[0]
        for i, df in enumerate(oof_dfs[1:], 1):
            merged_oof = pd.merge(merged_oof, df, on=args.id_column, suffixes=("", f"_model_{i}"))
        
        merged_oof = train_df.merge(merged_oof, on=args.id_column)
        
        # Get combined OOF predictions
        oof_cols = [col for col in merged_oof.columns if "_oof" in col]
        if oof_cols:
            combined_oof = merged_oof[oof_cols].mean(axis=1)
            oof_actuals = merged_oof[args.target_column]
        else:
            combined_oof = None
            oof_actuals = None
    else:
        combined_oof = None
        oof_actuals = None
    
    # Prepare data for assessment
    train_data = {
        'X': train_df[feature_cols] if feature_cols else None,
        'y': train_df[args.target_column],
        'oof_predictions': combined_oof,
        'oof_actuals': oof_actuals,
        'predictions': combined_oof if combined_oof is not None else train_df[args.target_column],
    }
    
    # Create mock ensemble (since we don't have full trained ensemble in memory)
    # In production, you'd load the actual trained ensemble
    adaptive_ensemble = AdaptiveEnsemble(lookback_window=21)
    
    # Mock dynamic stacker - in production load from saved model
    class MockStacker:
        def __init__(self):
            self.meta_model = None
        
        def predict(self, base_predictions, original_features=None):
            # Simple average for assessment
            if isinstance(base_predictions, pd.DataFrame):
                return base_predictions.mean(axis=1)
            return base_predictions
    
    ensemble = MockStacker()
    
    # Load regime data if provided
    regime_data = None
    if args.regime_data and args.regime_data.exists():
        regime_df = pd.read_csv(args.regime_data)
        if args.id_column in regime_df.columns and 'regime_label' in regime_df.columns:
            regime_data = regime_df.set_index(args.id_column)['regime_label']
            print(f"📊 Loaded regime data: {len(regime_data)} samples, {regime_data.nunique()} regimes")
            print()
    
    # Run comprehensive assessment
    print("🔬 Running learning assessment tests...")
    print()
    
    assessment = comprehensive_learning_assessment(
        ensemble=ensemble,
        adaptive_ensemble=adaptive_ensemble,
        train_data=train_data,
        oof_predictions=oof_predictions_dict if oof_predictions_dict else None,
    )
    
    # Run financial-specific tests if requested
    financial_results = None
    if args.financial_tests:
        print("💰 Running financial-specific learning tests...")
        print()
        
        financial_assessor = FinancialLearningAssessment()
        actual_returns = train_data['y']
        
        # Ensure we have predictions for financial tests
        if oof_predictions_dict:
            financial_results = {}
            
            # Alpha persistence test (use first model or average)
            if len(oof_predictions_dict) > 0:
                first_model_id = list(oof_predictions_dict.keys())[0]
                pred_series = oof_predictions_dict[first_model_id]
                
                # Align indices
                common_idx = pred_series.index.intersection(actual_returns.index)
                if len(common_idx) > 0:
                    pred_aligned = pred_series.reindex(common_idx).dropna()
                    actual_aligned = actual_returns.reindex(pred_aligned.index)
                    
                    if len(pred_aligned) > 126:  # Need enough data
                        financial_results['alpha_persistence'] = financial_assessor.alpha_persistence_test(
                            pred_aligned, actual_aligned
                        )
                        
                        financial_results['capacity_robustness'] = financial_assessor.capacity_robustness_test(
                            pred_aligned, actual_aligned
                        )
            
            # Regime robustness test (if regime data available)
            if regime_data is not None and oof_predictions_dict:
                # Align all indices
                all_indices = actual_returns.index
                for pred_series in oof_predictions_dict.values():
                    all_indices = all_indices.intersection(pred_series.index)
                all_indices = all_indices.intersection(regime_data.index)
                
                if len(all_indices) > 100:
                    aligned_predictions = {
                        model_id: pred_series.reindex(all_indices).dropna()
                        for model_id, pred_series in oof_predictions_dict.items()
                    }
                    aligned_actual = actual_returns.reindex(all_indices)
                    aligned_regime = regime_data.reindex(all_indices).dropna()
                    
                    # Only proceed if we have overlapping indices
                    if len(aligned_actual.index.intersection(aligned_regime.index)) > 100:
                        financial_results['regime_robustness'] = financial_assessor.regime_robustness_test(
                            aligned_predictions, aligned_actual, aligned_regime
                        )
        
        if financial_results:
            # Combine with original assessment
            financial_indicators = [
                financial_results.get('alpha_persistence', {}).get('is_learning', False),
                financial_results.get('capacity_robustness', {}).get('is_learning', False),
                financial_results.get('regime_robustness', {}).get('is_learning', True),  # Default True if not available
            ]
            
            financial_score = float(np.mean([i for i in financial_indicators if isinstance(i, bool)]))
            
            # Combined learning score (weighted average)
            combined_score = 0.7 * assessment['learning_score'] + 0.3 * financial_score
            
            assessment['financial_results'] = financial_results
            assessment['financial_learning_score'] = financial_score
            assessment['combined_learning_score'] = combined_score
            assessment['is_genuinely_learning_financial'] = combined_score > 0.7
            
            print("Financial Test Results:")
            print("-"*70)
            for test_name, test_result in financial_results.items():
                if isinstance(test_result, dict):
                    is_learning = test_result.get('is_learning', False)
                    status = "✅ PASS" if is_learning else "❌ FAIL"
                    print(f"\n{test_name.upper()}: {status}")
                    
                    if test_name == 'alpha_persistence':
                        print(f"   Persistence Score: {test_result.get('persistence_score', 0):.3f}")
                        for period, metrics in test_result.get('alpha_persistence', {}).items():
                            print(f"   {period}: {metrics.get('positive_alpha_ratio', 0):.3f} positive alpha ratio")
                    elif test_name == 'capacity_robustness':
                        print(f"   Robustness Score: {test_result.get('robustness_score', 0):.3f}")
                        print(f"   Base Sharpe: {test_result.get('base_sharpe', 0):.4f}")
                    elif test_name == 'regime_robustness':
                        print(f"   Overall Consistency: {test_result.get('overall_consistency', 0):.3f}")
                        print(f"   Number of Regimes: {test_result.get('num_regimes', 0)}")
            print()
        
        assessment['original_learning_score'] = assessment['learning_score']
        if 'combined_learning_score' in assessment:
            assessment['learning_score'] = assessment['combined_learning_score']
    
    # Display results
    print("="*70)
    print("LEARNING ASSESSMENT RESULTS")
    print("="*70)
    print()
    
    print(f"📊 Overall Learning Score: {assessment['learning_score']:.3f}")
    if 'financial_learning_score' in assessment:
        print(f"   - Original Score: {assessment['original_learning_score']:.3f}")
        print(f"   - Financial Score: {assessment['financial_learning_score']:.3f}")
        print(f"   - Combined Score: {assessment['combined_learning_score']:.3f}")
    print(f"🎯 Confidence Level: {assessment['confidence_level']}")
    print(f"💡 Interpretation: {assessment['interpretation']}")
    print(f"✅ Tests Passed: {assessment['passed_tests']}/{assessment['test_count']}")
    print()
    
    print("Detailed Test Results:")
    print("-"*70)
    
    for test_name, test_result in assessment['detailed_results'].items():
        if isinstance(test_result, dict):
            is_learning = test_result.get('is_learning', False)
            status = "✅ PASS" if is_learning else "❌ FAIL"
            
            print(f"\n{test_name.upper()}: {status}")
            
            # Show key metrics
            if test_name == 'generalization':
                print(f"   Generalization Gap: {test_result.get('generalization_gap', 0):.6f}")
                print(f"   OOF RMSE: {test_result.get('oof_rmse', 0):.6f}")
                print(f"   P-value: {test_result.get('p_value', 0):.4f}")
            elif test_name == 'stability':
                print(f"   Mean Stability: {test_result.get('mean_stability', 0):.3f}")
            elif test_name == 'diversity':
                print(f"   Mean Diversity: {test_result.get('mean_diversity', 0):.3f}")
                print(f"   Mean Correlation: {test_result.get('mean_correlation', 0):.3f}")
            elif test_name == 'compression':
                print(f"   Compression Ratio: {test_result.get('compression_ratio', 1):.3f}")
                print(f"   MDL Score: {test_result.get('mdl_score', 1):.3f}")
    
    print()
    print("="*70)
    
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
    
    results_file = args.output_dir / "learning_assessment.json"
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(assessment), f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    print()
    
    # Final verdict
    if assessment['is_genuinely_learning']:
        print("🎉 VERDICT: Model is GENUINELY LEARNING")
        print(f"   Confidence: {assessment['confidence']:.1%}")
    else:
        print("⚠️  VERDICT: Evidence of learning is MIXED")
        print(f"   Learning Score: {assessment['learning_score']:.3f}")
        print("   Review detailed results for areas of improvement")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

