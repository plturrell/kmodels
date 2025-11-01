"""Submission generation for PhysioNet ECG Image Digitization competition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import ECGDigitizationDataset, ECGSample, discover_samples
from ..features.transforms import build_eval_transform


def generate_predictions(
    model: torch.nn.Module,
    test_samples: List[ECGSample],
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 16,
    image_size: int = 512,
    use_tta: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate predictions for test samples.
    
    Args:
        model: Trained model
        test_samples: List of test samples
        device: Device for inference
        batch_size: Batch size
        image_size: Image size for transforms
        use_tta: Whether to use test-time augmentation
    
    Returns:
        Dictionary mapping ecg_id to predicted signal
    """
    model = model.to(device)
    model.eval()
    
    transform = build_eval_transform(image_size=image_size)
    dataset = ECGDigitizationDataset(test_samples, transforms=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating predictions"):
            images = batch["image"].to(device)
            ids = batch["id"]
            
            if use_tta:
                # Original + horizontal flip
                preds_orig = model(images)
                preds_flip = model(torch.flip(images, dims=[3]))
                preds = (preds_orig + preds_flip) / 2
            else:
                preds = model(images)
            
            # Store predictions
            preds_np = preds.cpu().numpy()
            for i, ecg_id in enumerate(ids):
                predictions[ecg_id] = preds_np[i]
    
    return predictions


def create_submission(
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    *,
    format: str = "csv",
) -> pd.DataFrame:
    """Create submission file from predictions.
    
    Args:
        predictions: Dictionary of predictions per ECG ID
        output_path: Path to save submission
        format: Output format ('csv' or 'npy')
    
    Returns:
        Submission DataFrame (if CSV format)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        # Create CSV with one row per ECG. Store the shape to preserve
        # multi-channel predictions and serialise higher-rank tensors as JSON.
        rows = []
        for ecg_id, signal in predictions.items():
            array = np.asarray(signal)
            shape_str = "x".join(str(dim) for dim in array.shape)
            if array.ndim <= 1:
                payload = ",".join(map(str, array.reshape(-1)))
            else:
                payload = json.dumps(array.tolist())
            rows.append({
                "ecg_id": ecg_id,
                "signal": payload,
                "signal_shape": shape_str,
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values("ecg_id")
        df.to_csv(output_path, index=False)
        
        print(f"Saved submission to {output_path}")
        print(f"Total predictions: {len(df)}")
        
        return df
    
    elif format == "npy":
        # Save as numpy arrays (one file per ECG)
        output_dir = output_path.parent / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for ecg_id, signal in predictions.items():
            np.save(output_dir / f"{ecg_id}.npy", signal)
        
        print(f"Saved {len(predictions)} predictions to {output_dir}")
        return None
    
    else:
        raise ValueError(f"Unknown format: {format}")


def load_checkpoint_and_predict(
    checkpoint_path: Path,
    test_image_dir: Path,
    output_path: Path,
    *,
    model_factory,
    device: Optional[str] = None,
    batch_size: int = 16,
    use_tta: bool = False,
    format: str = "csv",
) -> Optional[pd.DataFrame]:
    """Load model from checkpoint and generate submission.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_image_dir: Directory containing test images
        output_path: Path to save submission
        model_factory: Function that returns a new model instance
        device: Device to use (auto-detect if None)
        batch_size: Batch size for inference
        use_tta: Whether to use test-time augmentation
        format: Output format
    
    Returns:
        Submission DataFrame (if CSV format)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = model_factory()
    
    # Load weights
    if "state_dict" in checkpoint:
        # Lightning checkpoint
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    # Discover test samples
    print(f"Discovering test samples in {test_image_dir}")
    test_samples = discover_samples(test_image_dir, signal_dir=None)
    print(f"Found {len(test_samples)} test samples")
    
    # Generate predictions
    predictions = generate_predictions(
        model,
        test_samples,
        device=device,
        batch_size=batch_size,
        use_tta=use_tta,
    )
    
    # Create submission
    return create_submission(predictions, output_path, format=format)


__all__ = [
    "generate_predictions",
    "create_submission",
    "load_checkpoint_and_predict",
]

