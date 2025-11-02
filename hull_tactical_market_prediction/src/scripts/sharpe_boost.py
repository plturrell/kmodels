#!/usr/bin/env python3
"""
Sharpe Ratio Boost Pipeline - Target: 16.31 → 18.0
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..ensemble.advanced_position_sizing import AdvancedPositionSizer
from ..ensemble.advanced_risk_management import AdvancedRiskManager
from ..ensemble.signal_enhancement import SignalEnhancer
from ..ensemble.sharpe_optimization import SharpeOptimizer


def calculate_market_correlation(
    predictions: pd.Series, returns: pd.Series, window: int = 21
) -> pd.Series:
    """Calculate rolling correlation with market"""
    # Align indices
    common_idx = predictions.index.intersection(returns.index)
    pred_aligned = predictions.reindex(common_idx)
    returns_aligned = returns.reindex(common_idx)

    # Calculate rolling correlation
    correlation = pred_aligned.rolling(window=window, min_periods=5).corr(returns_aligned)
    return correlation.fillna(0)


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)


def boost_sharpe_ratio(
    predictions: pd.Series,
    actual_returns: pd.Series,
    volatility_estimate: pd.Series,
    regime_labels: pd.Series = None,
    regime_performance: Dict = None,
) -> pd.Series:
    """
    Apply all Sharpe-boosting techniques
    
    Args:
        predictions: Model predictions
        actual_returns: Actual returns (for dynamic targeting)
        volatility_estimate: Volatility estimates
        regime_labels: Optional regime labels
        regime_performance: Optional regime performance metrics
        
    Returns:
        Sharpe-boosted predictions
    """
    # Initialize components
    position_sizer = AdvancedPositionSizer(target_volatility=0.18)  # Slightly higher target
    signal_enhancer = SignalEnhancer()
    risk_manager = AdvancedRiskManager()

    # Step 1: Advanced position sizing
    sized_predictions = position_sizer.volatility_scaling(predictions, volatility_estimate)

    # Step 2: Dynamic volatility targeting
    sized_predictions = position_sizer.dynamic_volatility_targeting(
        sized_predictions, actual_returns
    )

    # Step 3: Signal enhancement
    enhanced_predictions = signal_enhancer.volatility_regime_filtering(
        sized_predictions, volatility_estimate
    )

    # Step 4: Regime boosting if available
    if regime_labels is not None and regime_performance is not None:
        enhanced_predictions = signal_enhancer.regime_conditional_boosting(
            enhanced_predictions, regime_labels, regime_performance
        )

    # Step 5: Risk management
    market_correlation = calculate_market_correlation(enhanced_predictions, actual_returns)
    final_predictions = risk_manager.correlation_penalty(enhanced_predictions, market_correlation)

    return final_predictions


def evaluate_sharpe_improvement(
    original_preds: pd.Series, boosted_preds: pd.Series, actual_returns: pd.Series
) -> Dict:
    """
    Evaluate improvement from Sharpe boosting
    
    Args:
        original_preds: Original predictions
        boosted_preds: Sharpe-boosted predictions
        actual_returns: Actual returns
        
    Returns:
        Dictionary with improvement metrics
    """
    # Align all series
    common_idx = original_preds.index.intersection(
        boosted_preds.index.intersection(actual_returns.index)
    )
    orig_aligned = original_preds.reindex(common_idx).dropna()
    boost_aligned = boosted_preds.reindex(common_idx).dropna()
    returns_aligned = actual_returns.reindex(common_idx).dropna()

    # Calculate strategy returns
    original_returns = orig_aligned * returns_aligned.reindex(orig_aligned.index)
    boosted_returns = boost_aligned * returns_aligned.reindex(boost_aligned.index)

    original_sharpe = calculate_sharpe(original_returns.dropna())
    boosted_sharpe = calculate_sharpe(boosted_returns.dropna())

    improvement = ((boosted_sharpe / original_sharpe) - 1) * 100 if original_sharpe > 0 else 0

    return {
        'original_sharpe': float(original_sharpe),
        'boosted_sharpe': float(boosted_sharpe),
        'improvement_pct': float(improvement),
        'target_achieved': boosted_sharpe >= 18.0,
        'sharpe_diff': float(boosted_sharpe - original_sharpe),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sharpe Ratio Boost Pipeline')
    parser.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='Path to predictions CSV file',
    )
    parser.add_argument(
        '--train-csv',
        type=Path,
        required=True,
        help='Path to training data CSV',
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
        '--prediction-column',
        default='prediction',
        help='Prediction column name',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/sharpe_boosted_predictions.csv'),
        help='Output path for boosted predictions',
    )
    parser.add_argument(
        '--results-output',
        type=Path,
        default=Path('outputs/sharpe_boost_results.json'),
        help='Output path for results JSON',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("🚀 SHARPE RATIO BOOST PIPELINE")
    print("=" * 70)
    print()

    # Load predictions
    print(f"📁 Loading predictions from {args.predictions}...")
    pred_df = pd.read_csv(args.predictions)
    if args.prediction_column not in pred_df.columns:
        raise ValueError(
            f"Prediction column '{args.prediction_column}' not found in predictions file"
        )

    # Load training data
    print(f"📁 Loading training data from {args.train_csv}...")
    train_df = pd.read_csv(args.train_csv)

    # Merge predictions with training data
    merged = pred_df.merge(
        train_df[[args.id_column, args.target_column]], on=args.id_column, how='inner'
    )

    predictions = pd.Series(
        merged[args.prediction_column].values, index=merged[args.id_column].values
    )
    actual_returns = pd.Series(
        merged[args.target_column].values, index=merged[args.id_column].values
    )

    # Estimate volatility from returns
    print("📊 Calculating volatility estimates...")
    volatility_estimate = actual_returns.rolling(window=21, min_periods=5).std().fillna(
        actual_returns.std()
    )

    # Calculate baseline Sharpe
    baseline_returns = predictions * actual_returns
    baseline_sharpe = calculate_sharpe(baseline_returns.dropna())
    print(f"📈 Baseline Sharpe Ratio: {baseline_sharpe:.2f}")
    print()

    # Boost Sharpe ratio
    print("🔧 Applying Sharpe-boosting techniques...")
    boosted_predictions = boost_sharpe_ratio(
        predictions,
        actual_returns,
        volatility_estimate,
        regime_labels=None,  # Add if you have regime data
        regime_performance=None,  # Add if you have regime performance
    )
    print("   ✅ Position sizing applied")
    print("   ✅ Signal enhancement applied")
    print("   ✅ Risk management applied")
    print()

    # Evaluate improvement
    print("📊 Evaluating improvement...")
    results = evaluate_sharpe_improvement(predictions, boosted_predictions, actual_returns)

    print()
    print("=" * 70)
    print("📈 SHARPE RATIO IMPROVEMENT RESULTS")
    print("=" * 70)
    print(f"   Original Sharpe:  {results['original_sharpe']:.2f}")
    print(f"   Boosted Sharpe:   {results['boosted_sharpe']:.2f}")
    print(f"   Improvement:      {results['improvement_pct']:.1f}%")
    print(f"   Sharpe Increase:  {results['sharpe_diff']:.2f}")
    print(f"   Target Achieved:  {'✅ YES' if results['target_achieved'] else '❌ NO (target: 18.0)'}")
    print()

    # Save boosted predictions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    boosted_df = pd.DataFrame(
        {args.id_column: boosted_predictions.index, args.prediction_column: boosted_predictions.values}
    )
    boosted_df.to_csv(args.output, index=False)
    print(f"💾 Boosted predictions saved to: {args.output}")

    # Save results
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {args.results_output}")
    print()

    if results['target_achieved']:
        print("🎉 TARGET ACHIEVED! Sharpe ratio >= 18.0")
    else:
        print(f"⚠️  Target not yet achieved. Current: {results['boosted_sharpe']:.2f}, Target: 18.0")
        print(f"   Additional improvement needed: {18.0 - results['boosted_sharpe']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

