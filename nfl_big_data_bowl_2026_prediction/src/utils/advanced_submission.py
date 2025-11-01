"""Submission generation for advanced models (Liquid, GNN, Diffusion)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.loaders import load_submission_format, load_test_input
from ..data.trajectory_dataset import TrajectoryDataset, collate_trajectories


def generate_predictions_from_model(
    model: torch.nn.Module,
    test_input: pd.DataFrame,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    feature_columns: list[str] = None,
) -> pd.DataFrame:
    """Generate predictions using a trained model.
    
    Args:
        model: Trained model (Liquid, GNN, etc.)
        test_input: Test input DataFrame
        device: Device for inference
        batch_size: Batch size
        feature_columns: Feature columns to use
    
    Returns:
        DataFrame with predictions
    """
    model = model.to(device)
    model.eval()
    
    if feature_columns is None:
        feature_columns = ["s", "a", "player_height", "player_weight"]
    
    # Create dummy output frame (no targets for test)
    dummy_output = test_input.copy()
    
    # Create dataset
    dataset = TrajectoryDataset(
        test_input,
        dummy_output,
        feature_columns=feature_columns,
        normalise_xy=False,  # Don't normalize for test
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
    )
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating predictions"):
            features = batch["features"].to(device)
            positions = batch["positions"].to(device)
            mask = batch["mask"].to(device)
            
            # Forward pass
            preds = model(features, positions, mask)
            
            # Extract predictions (last timestep)
            preds_np = preds[:, -1, :].cpu().numpy()
            
            # Store with metadata
            for i, sample in enumerate(batch["metadata"]):
                game_id = sample.metadata["game_id"]
                play_id = sample.metadata["play_id"]
                
                # Find target player
                target_mask = test_input[
                    (test_input["game_id"] == game_id) &
                    (test_input["play_id"] == play_id) &
                    (test_input["player_to_predict"] == True)
                ]
                
                if not target_mask.empty:
                    nfl_id = target_mask.iloc[0]["nfl_id"]
                    predictions.append({
                        "game_id": game_id,
                        "play_id": play_id,
                        "nfl_id": nfl_id,
                        "pred_x": preds_np[i, 0],
                        "pred_y": preds_np[i, 1],
                    })
    
    return pd.DataFrame(predictions)


def create_submission_from_predictions(
    predictions: pd.DataFrame,
    test_input: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Create Kaggle submission from predictions.
    
    Args:
        predictions: DataFrame with predictions
        test_input: Test input DataFrame
        output_path: Path to save submission
    
    Returns:
        Submission DataFrame
    """
    submission_index = load_submission_format()
    
    # Fallback: use last observed position
    fallback = (
        test_input[test_input["player_to_predict"]]
        .sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"], as_index=False)[["x", "y"]]
        .last()
        .rename(columns={"x": "fallback_x", "y": "fallback_y"})
    )
    
    # Merge predictions with submission index
    submission = submission_index.merge(
        predictions, on=["game_id", "play_id", "nfl_id"], how="left"
    ).merge(
        fallback, on=["game_id", "play_id", "nfl_id"], how="left"
    )
    
    # Fill missing predictions with fallback
    submission["x"] = submission["pred_x"].fillna(submission["fallback_x"])
    submission["y"] = submission["pred_y"].fillna(submission["fallback_y"])
    
    # Final submission format
    submission = submission[["id", "x", "y"]]
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    print(f"Total predictions: {len(submission)}")
    
    return submission


def generate_submission_from_checkpoint(
    checkpoint_path: Path,
    model_factory,
    output_path: Path,
    *,
    device: Optional[str] = None,
    batch_size: int = 32,
    feature_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load model from checkpoint and generate submission.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_factory: Function that returns a new model instance
        output_path: Path to save submission
        device: Device to use (auto-detect if None)
        batch_size: Batch size for inference
        feature_columns: Feature columns to use
    
    Returns:
        Submission DataFrame
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Create model
    model = model_factory()
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Load test data
    print("Loading test data...")
    test_input = load_test_input()
    
    # Generate predictions
    predictions = generate_predictions_from_model(
        model,
        test_input,
        device=device,
        batch_size=batch_size,
        feature_columns=feature_columns,
    )
    
    # Create submission
    return create_submission_from_predictions(predictions, test_input, output_path)


__all__ = [
    "generate_predictions_from_model",
    "create_submission_from_predictions",
    "generate_submission_from_checkpoint",
]

