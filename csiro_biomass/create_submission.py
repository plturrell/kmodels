#!/usr/bin/env python
"""Create Kaggle submission for CSIRO Image2Biomass competition.

This script generates predictions on the test set and creates a submission file
in the correct format for Kaggle.

Example usage:
    # From single model
    python create_submission.py --checkpoint outputs/baseline/best_model.ckpt
    
    # From ensemble of models
    python create_submission.py --checkpoints outputs/run1/best_model.ckpt outputs/run2/best_model.ckpt
    
    # From cross-validation
    python create_submission.py --cv_dir outputs/cross_validation
"""

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import BiomassDataset
from src.modeling.baseline import MultiModalBiomassModel
from src.utils.ensemble import simple_average_ensemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: str = "cuda") -> MultiModalBiomassModel:
    """Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    LOGGER.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use default config
        config = Config()
    
    # Create model
    model = MultiModalBiomassModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def predict_test_set(
    model: MultiModalBiomassModel,
    test_csv: Path,
    image_dir: Path,
    config: Config,
    device: str = "cuda",
) -> pd.DataFrame:
    """Generate predictions on test set.
    
    Args:
        model: Trained model
        test_csv: Path to test.csv
        image_dir: Directory containing test images
        config: Model configuration
        device: Device for inference
    
    Returns:
        DataFrame with predictions
    """
    LOGGER.info("Loading test dataset...")
    
    # Create test dataset
    test_dataset = BiomassDataset(
        csv_path=test_csv,
        image_dir=image_dir,
        config=config,
        is_training=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    LOGGER.info(f"Generating predictions for {len(test_dataset)} samples...")
    
    # Generate predictions
    predictions = []
    sample_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            
            # Get predictions (mean only, ignore variance)
            outputs = model(images, metadata)
            if isinstance(outputs, tuple):
                preds = outputs[0]  # Mean predictions
            else:
                preds = outputs
            
            predictions.append(preds.cpu().numpy())
            sample_ids.extend(batch['sample_id'])
    
    # Concatenate predictions
    import numpy as np
    predictions = np.concatenate(predictions, axis=0)
    
    # Create DataFrame in submission format
    df_test = pd.read_csv(test_csv)
    
    # Match predictions to sample IDs
    submission_rows = []
    for idx, sample_id in enumerate(sample_ids):
        # Each image has multiple targets
        for target_idx, target_name in enumerate(config.target_names):
            submission_rows.append({
                'sample_id': f"{sample_id}__{target_name}",
                'target': predictions[idx, target_idx],
            })
    
    submission_df = pd.DataFrame(submission_rows)
    
    LOGGER.info(f"Generated {len(submission_df)} predictions")
    
    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Create Kaggle submission")
    
    # Model options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to single model checkpoint",
    )
    group.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="Paths to multiple checkpoints for ensemble",
    )
    group.add_argument(
        "--cv_dir",
        type=Path,
        help="Cross-validation directory (will ensemble all folds)",
    )
    
    # Data options
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("csiro_biomass_extract"),
        help="Data directory",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("submission.csv"),
        help="Output submission file path",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    LOGGER.info("=" * 80)
    LOGGER.info("CSIRO Image2Biomass - Submission Creation")
    LOGGER.info("=" * 80)
    
    # Determine checkpoints to use
    checkpoints = []
    if args.checkpoint:
        checkpoints = [args.checkpoint]
    elif args.checkpoints:
        checkpoints = args.checkpoints
    elif args.cv_dir:
        # Find all fold checkpoints
        for fold_dir in sorted(args.cv_dir.glob("fold_*")):
            ckpt = fold_dir / "best_model.ckpt"
            if ckpt.exists():
                checkpoints.append(ckpt)
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {args.cv_dir}")
    
    LOGGER.info(f"Using {len(checkpoints)} model(s)")
    
    # Test data paths
    test_csv = args.data_dir / "test.csv"
    test_image_dir = args.data_dir / "test"
    
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not test_image_dir.exists():
        raise FileNotFoundError(f"Test image directory not found: {test_image_dir}")
    
    # Generate predictions for each model
    config = Config()  # Use default config
    prediction_dfs = []
    
    for i, checkpoint_path in enumerate(checkpoints):
        LOGGER.info(f"\nModel {i+1}/{len(checkpoints)}")
        
        model = load_model(checkpoint_path, device=args.device)
        pred_df = predict_test_set(model, test_csv, test_image_dir, config, args.device)
        prediction_dfs.append(pred_df)
    
    # Ensemble if multiple models
    if len(prediction_dfs) > 1:
        LOGGER.info(f"\nEnsembling {len(prediction_dfs)} models...")
        
        # Save individual predictions
        temp_files = []
        for i, df in enumerate(prediction_dfs):
            temp_path = args.output_path.parent / f"temp_pred_{i}.csv"
            df.to_csv(temp_path, index=False)
            temp_files.append(temp_path)
        
        # Create ensemble
        submission_df = simple_average_ensemble(
            temp_files,
            args.output_path,
        )
        
        # Clean up temp files
        for temp_file in temp_files:
            temp_file.unlink()
    else:
        submission_df = prediction_dfs[0]
        submission_df.to_csv(args.output_path, index=False)
    
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Submission Created!")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Submission file: {args.output_path}")
    LOGGER.info(f"Number of predictions: {len(submission_df)}")
    LOGGER.info("\nTo submit to Kaggle:")
    LOGGER.info(f"  kaggle competitions submit -c csiro-biomass -f {args.output_path} -m 'My submission'")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

