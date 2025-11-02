"""Train a stacking ensemble from the OOF predictions of multiple models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from .ensemble import AdaptiveEnsemble, DynamicStacker, MetaFeatureBuilder, RegimeDetector
from .training.walk_forward_cv import TimeSeriesCV


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a stacking ensemble.")
    parser.add_argument(
        "--oof-prediction",
        action="append",
        help="Path to an out-of-fold prediction CSV. Repeat for multiple models. (Required if not using checkpoints)",
    )
    parser.add_argument(
        "--test-prediction",
        action="append",
        required=True,
        help="Path to a test prediction (submission) CSV. Repeat for multiple models.",
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to the original training CSV.")
    parser.add_argument("--target-column", default="forward_returns", help="Target column name.")
    parser.add_argument("--id-column", default="date_id", help="ID column name.")
    parser.add_argument("--output-path", type=Path, default="submission_stacked.csv", help="Path for the final submission file.")
    
    # Adaptive ensemble arguments
    parser.add_argument(
        "--use-adaptive-weights",
        action="store_true",
        help="Enable adaptive ensemble weighting based on rolling Sharpe and regime detection",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=63,
        help="Rolling window for performance metrics (default: 63)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        help="Path to checkpoint directory. Repeat for multiple models. Can be combined with --oof-prediction.",
    )
    parser.add_argument(
        "--meta-model",
        choices=["linear", "gbm"],
        default="linear",
        help="Meta-model type: linear (default) or gbm",
    )
    parser.add_argument(
        "--blend-factor",
        type=float,
        default=0.7,
        help="Blend factor for meta-model vs adaptive weights (0-1, default: 0.7)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate arguments
    if args.oof_prediction is None and args.checkpoint_dir is None:
        raise ValueError("Either --oof-prediction or --checkpoint-dir must be provided.")
    
    if args.oof_prediction is not None and len(args.oof_prediction) != len(args.test_prediction):
        raise ValueError("The number of OOF and test prediction files must be the same.")
    
    if args.checkpoint_dir is not None and len(args.checkpoint_dir) != len(args.test_prediction):
        raise ValueError("The number of checkpoint directories and test prediction files must be the same.")

    # Load original training data
    train_df = pd.read_csv(args.train_csv)
    
    # Handle adaptive weights mode
    if args.use_adaptive_weights:
        # Initialize adaptive ensemble components
        adaptive_ensemble = AdaptiveEnsemble(
            lookback_window=args.lookback_window,
            min_active_periods=max(21, args.lookback_window // 3),
        )
        
        regime_detector = RegimeDetector()
        
        # Fit regime detector on training data
        # Extract feature columns (exclude id and target)
        feature_cols = [col for col in train_df.columns 
                       if col not in [args.id_column, args.target_column]]
        if len(feature_cols) > 0:
            feature_df = train_df[feature_cols]
            regime_detector.fit(feature_df)
        
        meta_feature_builder = MetaFeatureBuilder(
            adaptive_ensemble=adaptive_ensemble,
            regime_detector=regime_detector,
        )
        
        # Initialize dynamic stacker
        stacker = DynamicStacker(
            meta_model_type=args.meta_model,
            adaptive_ensemble=adaptive_ensemble,
            regime_detector=regime_detector,
            meta_feature_builder=meta_feature_builder,
            blend_factor=args.blend_factor,
        )
        
        # Load OOF predictions from CSV or generate from checkpoints
        if args.oof_prediction is not None:
            # Load from CSV files (existing behavior)
            oof_dfs = [pd.read_csv(p) for p in args.oof_prediction]
            merged_oof = oof_dfs[0]
            for i, df in enumerate(oof_dfs[1:]):
                merged_oof = pd.merge(
                    merged_oof,
                    df,
                    on=args.id_column,
                    suffixes=("", f"_model_{i + 1}"),
                )
            
            # Merge with targets
            merged_oof = pd.merge(
                merged_oof,
                train_df[[args.id_column, args.target_column]],
                on=args.id_column
            )
            
            # Extract OOF prediction columns
            oof_cols = [col for col in merged_oof.columns if "_oof" in col]
            base_oof_predictions = merged_oof[oof_cols].copy()
            base_oof_predictions.columns = [f"model_{i}" for i in range(len(oof_cols))]
            
            # Extract original features for meta-features
            feature_df_oof = train_df[train_df[args.id_column].isin(merged_oof[args.id_column])]
            feature_df_oof = feature_df_oof.set_index(args.id_column)
            feature_df_oof = feature_df_oof[feature_cols]
            
            # Align indices
            base_oof_predictions.index = merged_oof[args.id_column].values
            y_train = merged_oof[args.target_column].values
            y_train_series = pd.Series(y_train, index=base_oof_predictions.index)
            
            # Fit stacker
            stacker.fit(
                base_predictions=base_oof_predictions,
                original_features=feature_df_oof.reindex(base_oof_predictions.index),
                y=y_train_series,
            )
        else:
            # TODO: Load from checkpoints and generate OOF predictions
            # This would require model factory and feature preparation
            # For now, raise an error
            raise NotImplementedError(
                "Checkpoint loading with adaptive weights requires OOF predictions. "
                "Please provide --oof-prediction files or implement checkpoint OOF generation."
            )
        
        # Load and merge test predictions
        test_dfs = [pd.read_csv(p) for p in args.test_prediction]
        merged_test = test_dfs[0]
        for i, df in enumerate(test_dfs[1:]):
            merged_test = pd.merge(
                merged_test,
                df,
                on=args.id_column,
                suffixes=("", f"_model_{i + 1}"),
            )
        
        # Extract test prediction columns (remove _oof suffix logic)
        test_pred_cols = [col for col in merged_test.columns 
                         if col not in [args.id_column]]
        base_test_predictions = merged_test[test_pred_cols].copy()
        base_test_predictions.columns = [f"model_{i}" for i in range(len(test_pred_cols))]
        base_test_predictions.index = merged_test[args.id_column].values
        
        # Extract original features for test (if available)
        # Note: For test set, we may not have all features, so use training distribution
        feature_df_test = None
        if len(feature_cols) > 0:
            # Try to get features from test CSV if it has them
            test_feature_cols = [col for col in merged_test.columns if col in feature_cols]
            if len(test_feature_cols) > 0:
                feature_df_test = merged_test[test_feature_cols].copy()
                feature_df_test.index = merged_test[args.id_column].values
        
        # Generate predictions
        final_preds = stacker.predict(
            base_predictions=base_test_predictions,
            original_features=feature_df_test,
        )
        
        # Create submission
        submission = pd.DataFrame({
            args.id_column: merged_test[args.id_column],
            "prediction": final_preds.values
        })
        submission.to_csv(args.output_path, index=False)
        
        print(f"Dynamic stacking complete. Final submission saved to {args.output_path}")
        
        # Print adaptive weights if available
        if adaptive_ensemble is not None:
            weights = adaptive_ensemble.calculate_adaptive_weights()
            if weights:
                print("\nAdaptive ensemble weights:")
                for model_id, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {model_id}: {weight:.4f}")
        
        return 0
    
    else:
        # Original behavior (backward compatible)
        if args.oof_prediction is None:
            raise ValueError("--oof-prediction is required when not using --use-adaptive-weights")
        
        # Load and merge OOF predictions
        oof_dfs = [pd.read_csv(p) for p in args.oof_prediction]
        merged_oof = oof_dfs[0]
        for i, df in enumerate(oof_dfs[1:]):
            merged_oof = pd.merge(
                merged_oof,
                df,
                on=args.id_column,
                suffixes=("", f"_model_{i + 1}"),
            )

        # Load original training data to get the true target
        merged_oof = pd.merge(merged_oof, train_df[[args.id_column, args.target_column]], on=args.id_column)

        # Prepare training data for the meta-model
        feature_cols = [col for col in merged_oof.columns if "_oof" in col]
        X_train = merged_oof[feature_cols]
        y_train = merged_oof[args.target_column]

        # Train the meta-model
        meta_model = LinearRegression()
        meta_model.fit(X_train, y_train)

        # Load and merge test predictions
        test_dfs = [pd.read_csv(p) for p in args.test_prediction]
        merged_test = test_dfs[0]
        for i, df in enumerate(test_dfs[1:]):
            merged_test = pd.merge(
                merged_test,
                df,
                on=args.id_column,
                suffixes=("", f"_model_{i + 1}"),
            )

        # Prepare test data for the meta-model
        X_test = merged_test[[col.replace("_oof", "") for col in feature_cols]]

        # Generate final predictions
        final_preds = meta_model.predict(X_test)

        # Create submission file
        submission = pd.DataFrame({args.id_column: merged_test[args.id_column], "prediction": final_preds})
        submission.to_csv(args.output_path, index=False)

        print(f"Stacking complete. Final submission saved to {args.output_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
