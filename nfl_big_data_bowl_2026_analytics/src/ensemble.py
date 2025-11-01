"""Create an ensemble from multiple prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-files", type=Path, nargs='+', required=True,
                        help="A list of CSV files containing predictions.")
    parser.add_argument("--output-path", type=Path, required=True,
                        help="The path to save the final ensembled submission file.")
    args = parser.parse_args()

    if not args.prediction_files:
        print("No prediction files provided.")
        return 1

    # Load and average the predictions
    ensembled_preds = None
    id_cols = None
    for file_path in args.prediction_files:
        pred_df = pd.read_csv(file_path)
        if id_cols is None:
            id_cols = [col for col in pred_df.columns if 'target' not in col]
        
        if ensembled_preds is None:
            ensembled_preds = pred_df.drop(columns=id_cols).copy()
        else:
            ensembled_preds += pred_df.drop(columns=id_cols)

    ensembled_preds /= len(args.prediction_files)

    # Create the final submission file
    final_df = pd.concat([pd.read_csv(args.prediction_files[0])[id_cols], ensembled_preds], axis=1)
    final_df.to_csv(args.output_path, index=False)
    print(f"Ensembled submission saved to {args.output_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
