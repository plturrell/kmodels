"""Generate predictions from a trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .config.experiment import ModelConfig, DatasetConfig, FeatureConfig
from .data import load_training_dataframe, extract_numeric_features, AnalyticsDataset
from .features.player_kinematics import add_player_kinematics, add_distance_to_ball
from .features.relational import add_relational_features
from .modeling import build_baseline_model
from .train import _create_graph_data
from torch_geometric.loader import DataLoader as GraphDataLoader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load config and metadata
    with (args.run_dir / "config.json").open("r") as f:
        config_dict = json.load(f)
    with (args.run_dir / "feature_metadata.json").open("r") as f:
        metadata = json.load(f)

    model_cfg = ModelConfig(**config_dict["model"])
    dataset_cfg = DatasetConfig(**config_dict["dataset"])
    feature_cfg = FeatureConfig(**config_dict["features"])

    # Load model
    input_dim = len(metadata["feature_columns"])
    output_dim = len(metadata["target_columns"])
    model = build_baseline_model(input_dim, output_dim, model_cfg)
    model.load_state_dict(torch.load(args.run_dir / "best_model.pt", map_location=args.device))
    model.to(args.device)
    model.eval()

    # Load and prepare data
    df = pd.read_csv(args.data_path)
    df = add_player_kinematics(df)
    df = add_distance_to_ball(df)
    df = add_relational_features(df)
    
    bundle = extract_numeric_features(df, dataset_cfg)
    dataset = AnalyticsDataset(
        bundle,
        feature_mean=metadata["feature_mean"],
        feature_std=metadata["feature_std"],
    )

    # Create loader
    is_graph_model = model_cfg.architecture == 'gat'
    if is_graph_model:
        graphs = _create_graph_data(dataset)
        loader = GraphDataLoader(graphs, batch_size=1024, shuffle=False)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    # Predict
    predictions = []
    with torch.no_grad():
        for batch in loader:
            if is_graph_model:
                preds = model(batch.to(args.device))
            else:
                features, _, _ = batch
                preds = model(features.to(args.device))
            predictions.append(preds.cpu())

    # Save predictions
    pred_df = pd.DataFrame(torch.cat(predictions).numpy(), columns=metadata["target_columns"])
    output_df = pd.concat([df[metadata["id_columns"]], pred_df], axis=1)
    output_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
