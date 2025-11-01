"""Enhanced submission generation with TTA and ensemble support."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..data.dataset import DEFAULT_DATA_ROOT


LABEL_MAP = {0: "authentic", 1: "forged"}


def _load_image(path: Path, size: Sequence[int] = (512, 648)) -> torch.Tensor:
    with Image.open(path) as img:
        image = img.convert("RGB")
        array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
    if size is not None and tensor.shape[1:] != tuple(size):
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=tuple(size), mode="bilinear", align_corners=False
        ).squeeze(0)
    return tensor


def predict_labels(
    model: torch.nn.Module,
    image_dir: Path,
    device: torch.device,
    use_tta: bool = False,
    return_probs: bool = False,
) -> pd.DataFrame:
    """Generate predictions for test images.

    Args:
        model: Trained model
        image_dir: Directory containing test images
        device: Device for inference
        use_tta: Whether to use test-time augmentation
        return_probs: Whether to return probabilities

    Returns:
        DataFrame with predictions
    """
    records = []
    image_paths = sorted(image_dir.glob("*.png"))

    for path in tqdm(image_paths, desc="Predicting"):
        tensor = _load_image(path)

        if use_tta:
            # Test-time augmentation
            logits_list = []

            # Original
            with torch.no_grad():
                out = model(tensor.unsqueeze(0).to(device))
                if isinstance(out, tuple):
                    logits = out[0]  # (class_logits, mask_logits)
                else:
                    logits = out["class_logits"]
                logits_list.append(logits)

            # Horizontal flip
            with torch.no_grad():
                flipped = torch.flip(tensor, dims=[2])
                out = model(flipped.unsqueeze(0).to(device))
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out["class_logits"]
                logits_list.append(logits)

            # Vertical flip
            with torch.no_grad():
                flipped = torch.flip(tensor, dims=[1])
                out = model(flipped.unsqueeze(0).to(device))
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out["class_logits"]
                logits_list.append(logits)

            # Average logits
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            # Single prediction
            with torch.no_grad():
                out = model(tensor.unsqueeze(0).to(device))
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out["class_logits"]

        probs = F.softmax(logits, dim=1)
        label_idx = logits.argmax(dim=1).item()

        case_id = int(path.stem)
        record = {
            "case_id": case_id,
            "annotation": LABEL_MAP[label_idx],
        }

        if return_probs:
            record["prob_authentic"] = probs[0, 0].item()
            record["prob_forged"] = probs[0, 1].item()

        records.append(record)

    return pd.DataFrame(records)


def save_submission(df: pd.DataFrame, output_path: Path) -> Path:
    df = df.sort_values("case_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission for the forgery detection competition")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output submission CSV path")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Data root directory")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--use-tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--return-probs", action="store_true", help="Include probabilities in output")
    return parser.parse_args()


def main(argv=None) -> int:
    args = parse_args()
    run_dir = args.run_dir

    # Find checkpoint
    state_path = None
    checkpoint_candidates = [
        "checkpoints/last.ckpt",
        "checkpoints/*.ckpt",
        "baseline_model.pt",
        "best_model.pt",
    ]

    for pattern in checkpoint_candidates:
        if "*" in pattern:
            matches = list(run_dir.glob(pattern))
            if matches:
                state_path = matches[0]
                break
        else:
            candidate_path = run_dir / pattern
            if candidate_path.exists():
                state_path = candidate_path
                break

    if state_path is None:
        raise FileNotFoundError(f"Could not locate model checkpoint in {run_dir}")

    print(f"Loading checkpoint from {state_path}")

    device = torch.device(args.device)

    # Load model (Lightning checkpoint or raw state dict)
    from ..modeling.baseline import VisionBaselineConfig, build_vision_baseline

    model = build_vision_baseline(VisionBaselineConfig()).to(device)

    checkpoint = torch.load(state_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    model.eval()

    image_dir = args.data_root / "test_images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {image_dir}")

    print(f"Generating predictions for images in {image_dir}")
    predictions = predict_labels(
        model,
        image_dir,
        device,
        use_tta=args.use_tta,
        return_probs=args.return_probs,
    )

    save_submission(predictions, args.output)
    print(f"Submission written to {args.output}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted forged: {(predictions['annotation'] == 'forged').sum()}")
    print(f"Predicted authentic: {(predictions['annotation'] == 'authentic').sum()}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
