"""Generate pseudo-labels from a trained checkpoint for semi-supervised loops."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.training import AugmentationConfig
from ..data.dataset import ForgeryDataset, load_samples
from ..data.transforms import create_transforms
from ..training.lightning_module import ForgeryLightningModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate pseudo labels for unlabeled data")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing train/test images")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pseudo_labels"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--devices", type=str, default="auto")
    return parser


def _select_samples(data_root: Path, split: str) -> List:
    samples = load_samples(data_root)
    if split == "train":
        return samples
    # For test split we assume a mirror directory structure without labels
    test_dir = data_root / "test_images"
    test_samples: List = []
    if test_dir.exists():
        for image_path in sorted(test_dir.glob("*.png")):
            test_samples.append(type("Anon", (), {"image_path": image_path, "label": 0, "mask_path": None}))
    return test_samples


def _write_csv(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = _select_samples(args.data_root, args.split)
    if not samples:
        raise FileNotFoundError("No samples found for pseudo-labelling")

    transforms = create_transforms(AugmentationConfig(), is_training=False)
    dataset = ForgeryDataset(samples, transforms=transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    module = ForgeryLightningModule.load_from_checkpoint(str(args.checkpoint), map_location=device)
    module.eval()
    module.to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            case_ids = batch.get("case_id")
            outputs = module(images)
            class_logits, mask_logits = outputs
            class_prob = torch.softmax(class_logits, dim=1).cpu()
            mask_prob = torch.sigmoid(mask_logits).cpu()

            for idx in range(class_prob.size(0)):
                probs = class_prob[idx]
                max_prob, pred_label = probs.max(dim=0)
                confident = float(max_prob) >= args.min_confidence
                if isinstance(case_ids, (list, tuple)):
                    case_id = case_ids[idx]
                else:
                    case_id = case_ids

                if confident:
                    mask = (mask_prob[idx, 0] > args.mask_threshold).numpy().astype("uint8")
                    npy_path = mask_dir / f"{case_id}.npy"
                    np.save(npy_path, mask)
                else:
                    npy_path = None

                rows.append(
                    {
                        "case_id": case_id,
                        "prob_authentic": float(probs[0]),
                        "prob_forged": float(probs[1]) if probs.numel() > 1 else float(probs[0]),
                        "pseudo_label": int(pred_label.item()),
                        "confidence": float(max_prob),
                        "mask_path": str(npy_path) if npy_path else "",
                    }
                )

    _write_csv(rows, args.output_dir / "pseudo_labels.csv")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


