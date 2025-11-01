"""Train a stacking ensemble from the OOF predictions of multiple models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a stacking ensemble.")
    parser.add_argument(
        "--oof-prediction",
        action="append",
        required=True,
        help="Path to an out-of-fold prediction CSV. Repeat for multiple models.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.oof_prediction) != len(args.test_prediction):
        raise ValueError("The number of OOF and test prediction files must be the same.")

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
    train_df = pd.read_csv(args.train_csv)
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
