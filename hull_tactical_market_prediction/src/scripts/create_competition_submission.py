"""Create comprehensive competition submission with all metadata and reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..ensemble import AdaptiveEnsemble, CompetitionAdaptor, get_fast_adaptation_config
from ..ensemble import RegimeDetector, DynamicStacker, MetaFeatureBuilder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create competition submission with full metadata and reports."
    )
    parser.add_argument(
        "--ensemble-predictions",
        type=Path,
        required=True,
        help="Path to ensemble predictions CSV (output from train_stacker.py)",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Path to training CSV (for regime analysis)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/competition_submission"),
        help="Output directory for submission package",
    )
    parser.add_argument(
        "--adaptation-speed",
        choices=["fast", "stable", "default"],
        default="default",
        help="Adaptation speed: fast (21-day), stable (126-day), or default (63-day)",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        help="Override lookback window (if not using preset adaptation speed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Load predictions
    predictions_df = pd.read_csv(args.ensemble_predictions)
    
    if "prediction" not in predictions_df.columns:
        raise ValueError("Predictions CSV must have 'prediction' column")
    
    # Prepare submission DataFrame
    if "date_id" not in predictions_df.columns:
        # Assume first column is ID
        id_col = predictions_df.columns[0]
        submission_df = pd.DataFrame({
            "date_id": predictions_df[id_col],
            "prediction": predictions_df["prediction"],
        })
    else:
        submission_df = predictions_df[["date_id", "prediction"]].copy()
    
    # Get adaptation config
    if args.adaptation_speed == "fast":
        config = get_fast_adaptation_config()
        lookback = config["lookback_window"]
    elif args.adaptation_speed == "stable":
        from ..ensemble.optimization_utils import get_stable_adaptation_config
        config = get_stable_adaptation_config()
        lookback = config["lookback_window"]
    else:
        lookback = args.lookback_window or 63
        config = {"lookback_window": lookback}
    
    # Initialize adaptive ensemble (for analysis)
    adaptive_ensemble = AdaptiveEnsemble(
        lookback_window=lookback,
        min_active_periods=max(10, lookback // 3),
    )
    
    # Initialize regime detector and fit on training data
    train_df = pd.read_csv(args.train_csv)
    regime_detector = RegimeDetector()
    
    feature_cols = [
        c for c in train_df.columns
        if c not in ["date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"]
    ]
    if feature_cols:
        feature_df = train_df[feature_cols]
        regime_detector.fit(feature_df)
        
        # Detect current regime (use last row as proxy)
        current_regime = regime_detector.detect_regime(feature_df.iloc[-10:]).iloc[-1]
    else:
        current_regime = "normal"
    
    # Create competition adaptor
    metadata = {
        "blend_factor": "70% meta-model, 30% adaptive",
        "lookback_window": lookback,
        "adaptation_speed": args.adaptation_speed,
        "base_models": 3,  # From our training
    }
    
    adaptor = CompetitionAdaptor(
        initial_ensemble=adaptive_ensemble,
        output_dir=args.output_dir,
        ensemble_metadata=metadata,
    )
    
    # Create comprehensive submission
    print("Creating competition submission package...")
    files_created = adaptor.create_submission(
        predictions=submission_df,
        model_metadata={
            "prediction_count": len(submission_df),
            "prediction_mean": float(submission_df["prediction"].mean()),
            "prediction_std": float(submission_df["prediction"].std()),
        },
        current_regime=current_regime,
    )
    
    print("\n✅ Submission package created!")
    print("\nFiles generated:")
    for filepath, description in files_created.items():
        print(f"  - {filepath.name}: {description}")
    
    print(f"\n📦 Main submission file: {files_created.popitem()[0]}")
    print(f"📊 Full package in: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

