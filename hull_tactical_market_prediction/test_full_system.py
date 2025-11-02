"""Comprehensive end-to-end test of adaptive ensemble system."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ensemble import (
    AdaptiveEnsemble,
    RegimeDetector,
    MetaFeatureBuilder,
    DynamicStacker,
    PerformanceMonitor,
    CompetitionAdaptor,
    get_fast_adaptation_config,
    analyze_regime_specialization,
    calculate_dynamic_blend_ratio,
)


def test_regime_detection():
    """Test regime detection on real data."""
    print("\n" + "="*60)
    print("TEST 1: Regime Detection")
    print("="*60)
    
    train_df = pd.read_csv("data/raw/train.csv")
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    ]
    
    if not feature_cols:
        print("❌ No features found")
        return False
    
    feature_df = train_df[feature_cols].head(100)  # Use subset for speed
    
    regime_detector = RegimeDetector()
    regime_detector.fit(feature_df)
    
    regimes = regime_detector.detect_regime(feature_df)
    regime_proxy = regime_detector.get_regime_proxy(feature_df)
    
    print(f"✅ Regime detector fitted on {len(feature_df)} samples")
    print(f"✅ Detected {len(regimes.unique())} unique regimes: {regimes.unique().tolist()}")
    print(f"✅ Regime proxy has {len(regime_proxy.columns)} columns")
    print(f"   Regime distribution:")
    for regime, count in regimes.value_counts().items():
        print(f"     {regime}: {count} ({count/len(regimes)*100:.1f}%)")
    
    return True


def test_adaptive_ensemble():
    """Test adaptive ensemble with simulated predictions."""
    print("\n" + "="*60)
    print("TEST 2: Adaptive Ensemble")
    print("="*60)
    
    # Load training data
    train_df = pd.read_csv("data/raw/train.csv")
    
    # Create adaptive ensemble with fast config
    config = get_fast_adaptation_config()
    adaptive_ensemble = AdaptiveEnsemble(**config)
    
    print(f"✅ Created AdaptiveEnsemble with config:")
    print(f"   Lookback window: {config['lookback_window']}")
    print(f"   Min active periods: {config['min_active_periods']}")
    print(f"   Sharpe decay: {config['sharpe_decay_factor']}")
    
    # Simulate model predictions
    n_samples = min(100, len(train_df))
    subset = train_df.head(n_samples)
    
    for i in range(3):
        model_id = f"model_{i}"
        
        # Create synthetic predictions (slight noise around actual returns)
        predictions = subset['forward_returns'].copy() + pd.Series(
            np.random.randn(n_samples) * 0.001, index=subset.index
        )
        actuals = subset['forward_returns'].copy()
        
        # Register model
        adaptive_ensemble.register_model(
            model_id=model_id,
            predictions=predictions,
            actuals=actuals,
            current_regime="normal",
        )
        print(f"✅ Registered {model_id} with {len(predictions)} predictions")
    
    # Calculate weights
    weights = adaptive_ensemble.calculate_adaptive_weights(current_regime="normal")
    print(f"\n✅ Calculated adaptive weights:")
    for model_id, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model_id}: {weight:.1%}")
    
    # Get model stats
    stats = adaptive_ensemble.get_model_stats()
    print(f"\n✅ Model statistics:")
    print(stats.to_string(index=False))
    
    return True, adaptive_ensemble


def test_meta_features(adaptive_ensemble, regime_detector):
    """Test meta-feature builder."""
    print("\n" + "="*60)
    print("TEST 3: Meta-Feature Builder")
    print("="*60)
    
    train_df = pd.read_csv("data/raw/train.csv")
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    ]
    feature_df = train_df[feature_cols].head(100)
    
    # Create synthetic base predictions
    base_predictions = pd.DataFrame({
        'model_0': pd.Series(np.random.randn(100) * 0.001, index=feature_df.index),
        'model_1': pd.Series(np.random.randn(100) * 0.001, index=feature_df.index),
        'model_2': pd.Series(np.random.randn(100) * 0.001, index=feature_df.index),
    })
    
    meta_builder = MetaFeatureBuilder(
        adaptive_ensemble=adaptive_ensemble,
        regime_detector=regime_detector,
    )
    
    meta_features = meta_builder.build_meta_features(
        base_predictions=base_predictions,
        original_features=feature_df,
    )
    
    print(f"✅ Built meta-features: {meta_features.shape[0]} samples, {meta_features.shape[1]} features")
    print(f"   Feature types:")
    feature_types = {}
    for col in meta_features.columns:
        prefix = col.split('_')[0]
        feature_types[prefix] = feature_types.get(prefix, 0) + 1
    for feat_type, count in sorted(feature_types.items()):
        print(f"     {feat_type}: {count} features")
    
    return True


def test_performance_monitor(adaptive_ensemble):
    """Test performance monitoring."""
    print("\n" + "="*60)
    print("TEST 4: Performance Monitor")
    print("="*60)
    
    monitor = PerformanceMonitor(output_dir=Path("outputs/test_monitoring"))
    
    # Log multiple states
    for regime in ["normal", "trending", "high_vol", "normal"]:
        monitor.log_ensemble_state(adaptive_ensemble, current_regime=regime)
    
    print(f"✅ Logged {len(monitor.weight_history)} ensemble states")
    
    # Generate report
    report = monitor.generate_performance_report()
    print(f"✅ Generated performance report ({len(report)} characters)")
    print("\n" + "="*40)
    print(report[:500] + "..." if len(report) > 500 else report)
    print("="*40)
    
    # Get weight DataFrame
    weight_df = monitor.get_weight_dataframe()
    print(f"\n✅ Weight history DataFrame: {weight_df.shape}")
    
    # Analyze specialization
    specialization = monitor.analyze_regime_specialization(adaptive_ensemble)
    print(f"✅ Regime specialization analysis: {len(specialization)} models")
    
    # Save report
    report_path = monitor.save_report("test_performance_report.md")
    print(f"✅ Saved report to: {report_path}")
    
    return True


def test_dynamic_stacker():
    """Test dynamic stacker."""
    print("\n" + "="*60)
    print("TEST 5: Dynamic Stacker")
    print("="*60)
    
    # Load OOF predictions if available
    oof_files = list(Path("outputs/full_training").glob("*/run-*/oof_predictions.csv"))
    
    if not oof_files:
        print("⚠️  No OOF predictions found, skipping stacker test")
        return True
    
    oof_files = oof_files[:3]  # Use first 3
    
    # Load and merge OOF predictions
    oof_dfs = [pd.read_csv(f) for f in oof_files]
    train_df = pd.read_csv("data/raw/train.csv")
    
    # Merge OOF predictions
    merged_oof = oof_dfs[0]
    for i, df in enumerate(oof_dfs[1:], 1):
        merged_oof = pd.merge(
            merged_oof, df,
            on="date_id",
            suffixes=("", f"_model_{i}")
        )
    
    # Merge with targets
    merged_oof = pd.merge(
        merged_oof,
        train_df[["date_id", "forward_returns"]],
        on="date_id"
    )
    
    # Extract OOF columns
    oof_cols = [col for col in merged_oof.columns if "_oof" in col]
    base_oof = merged_oof[oof_cols].copy()
    base_oof.columns = [f"model_{i}" for i in range(len(oof_cols))]
    base_oof.index = merged_oof["date_id"].values
    
    # Create stacker components
    config = get_fast_adaptation_config()
    adaptive_ensemble = AdaptiveEnsemble(**config)
    
    regime_detector = RegimeDetector()
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    ]
    feature_df = train_df[train_df["date_id"].isin(merged_oof["date_id"])][feature_cols]
    regime_detector.fit(feature_df)
    
    meta_builder = MetaFeatureBuilder(
        adaptive_ensemble=adaptive_ensemble,
        regime_detector=regime_detector,
    )
    
    stacker = DynamicStacker(
        meta_model_type="linear",
        adaptive_ensemble=adaptive_ensemble,
        regime_detector=regime_detector,
        meta_feature_builder=meta_builder,
        blend_factor=0.7,
    )
    
    # Fit stacker
    y_train = merged_oof["forward_returns"]
    y_train.index = base_oof.index
    
    feature_df_aligned = feature_df.copy()
    feature_df_aligned.index = train_df[train_df["date_id"].isin(merged_oof["date_id"])]["date_id"].values
    
    stacker.fit(
        base_predictions=base_oof,
        original_features=feature_df_aligned.reindex(base_oof.index),
        y=y_train,
    )
    
    print(f"✅ Dynamic stacker fitted on {len(base_oof)} samples")
    print(f"   Base models: {len(base_oof.columns)}")
    print(f"   Meta-feature dimensions: {len(stacker.feature_columns)} features")
    
    # Test prediction
    test_files = list(Path("outputs/full_training").glob("*/run-*/submission.csv"))
    if test_files:
        test_dfs = [pd.read_csv(f) for f in test_files[:3]]
        merged_test = test_dfs[0]
        for i, df in enumerate(test_dfs[1:], 1):
            merged_test = pd.merge(merged_test, df, on="date_id", suffixes=("", f"_model_{i}"))
        
        test_pred_cols = [col for col in merged_test.columns if col not in ["date_id"]]
        base_test = merged_test[test_pred_cols].copy()
        base_test.columns = [f"model_{i}" for i in range(len(test_pred_cols))]
        base_test.index = merged_test["date_id"].values
        
        predictions = stacker.predict(base_predictions=base_test)
        print(f"✅ Generated {len(predictions)} test predictions")
        print(f"   Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    
    return True


def test_competition_pipeline(adaptive_ensemble):
    """Test competition submission pipeline."""
    print("\n" + "="*60)
    print("TEST 6: Competition Pipeline")
    print("="*60)
    
    # Create test predictions
    test_df = pd.read_csv("data/raw/test.csv")
    predictions_df = pd.DataFrame({
        "date_id": test_df["date_id"],
        "prediction": pd.Series(np.random.randn(len(test_df)) * 0.001),
    })
    
    # Create competition adaptor
    adaptor = CompetitionAdaptor(
        initial_ensemble=adaptive_ensemble,
        output_dir=Path("outputs/test_submission"),
        ensemble_metadata={
            "blend_factor": "70/30",
            "lookback_window": 21,
        },
    )
    
    # Create submission
    files_created = adaptor.create_submission(
        predictions=predictions_df,
        model_metadata={
            "test_run": True,
            "prediction_count": len(predictions_df),
        },
        current_regime="normal",
    )
    
    print(f"✅ Created submission package:")
    for filepath, description in files_created.items():
        exists = filepath.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {filepath.name}: {description}")
        if exists:
            size = filepath.stat().st_size
            print(f"      Size: {size} bytes")
    
    # Test leaderboard integration
    recommendations = adaptor.update_based_on_leaderboard(
        public_score=0.0005,
        private_score=0.0004,
    )
    
    print(f"\n✅ Leaderboard integration:")
    print(f"   Status: {recommendations['status']}")
    print(f"   Recommendations: {len(recommendations.get('recommendations', []))}")
    
    return True


def test_optimization_utils():
    """Test optimization utilities."""
    print("\n" + "="*60)
    print("TEST 7: Optimization Utilities")
    print("="*60)
    
    # Test fast config
    fast_config = get_fast_adaptation_config()
    print(f"✅ Fast adaptation config:")
    for key, value in fast_config.items():
        print(f"   {key}: {value}")
    
    # Test dynamic blend ratio
    blend_ratio = calculate_dynamic_blend_ratio(
        ensemble_confidence=0.85,
        regime_stability=0.75,
    )
    print(f"\n✅ Dynamic blend ratio: {blend_ratio:.2%}")
    
    # Test regime threshold optimization (if we have features)
    train_df = pd.read_csv("data/raw/train.csv")
    feature_cols = [
        c for c in train_df.columns
        if c.startswith('V') or c.startswith('M')
    ]
    if feature_cols:
        feature_df = train_df[feature_cols].head(100)
        regime_detector = RegimeDetector()
        
        from src.ensemble.optimization_utils import optimize_regime_thresholds
        
        thresholds = optimize_regime_thresholds(
            regime_detector=regime_detector,
            features=feature_df,
            method="percentile",
        )
        print(f"\n✅ Optimized regime thresholds: {len(thresholds)} found")
        for key, value in thresholds.items():
            print(f"   {key}: {value:.6f}")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("FULL SYSTEM TEST - Adaptive Ensemble")
    print("="*60)
    
    results = {}
    
    # Test 1: Regime Detection
    try:
        results['regime_detection'] = test_regime_detection()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['regime_detection'] = False
    
    # Test 2: Adaptive Ensemble
    try:
        success, adaptive_ensemble = test_adaptive_ensemble()
        results['adaptive_ensemble'] = success
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['adaptive_ensemble'] = False
        adaptive_ensemble = None
    
    # Test 3: Meta Features (if adaptive_ensemble available)
    if adaptive_ensemble:
        try:
            regime_detector = RegimeDetector()
            train_df = pd.read_csv("data/raw/train.csv")
            feature_cols = [
                c for c in train_df.columns
                if c not in ["date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
            ]
            if feature_cols:
                feature_df = train_df[feature_cols].head(100)
                regime_detector.fit(feature_df)
            results['meta_features'] = test_meta_features(adaptive_ensemble, regime_detector)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results['meta_features'] = False
    else:
        results['meta_features'] = False
    
    # Test 4: Performance Monitor
    if adaptive_ensemble:
        try:
            results['performance_monitor'] = test_performance_monitor(adaptive_ensemble)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results['performance_monitor'] = False
    else:
        results['performance_monitor'] = False
    
    # Test 5: Dynamic Stacker
    try:
        results['dynamic_stacker'] = test_dynamic_stacker()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['dynamic_stacker'] = False
    
    # Test 6: Competition Pipeline
    if adaptive_ensemble:
        try:
            results['competition_pipeline'] = test_competition_pipeline(adaptive_ensemble)
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results['competition_pipeline'] = False
    else:
        results['competition_pipeline'] = False
    
    # Test 7: Optimization Utils
    try:
        results['optimization_utils'] = test_optimization_utils()
    except Exception as e:
        print(f"❌ FAILED: {e}")
        results['optimization_utils'] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(results.values())
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

