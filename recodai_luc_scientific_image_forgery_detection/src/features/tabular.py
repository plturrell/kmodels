"""Image-level tabular feature extraction for quick gradient boosting baselines."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

_BASE_DIR = Path(__file__).resolve().parents[2]

DATA_ROOT = _BASE_DIR / "data" / "raw" / "recodai-luc-scientific-image-forgery-detection"

LABEL_MAP = {"authentic": 0, "forged": 1}


@dataclass
class FeatureRow:
    case_id: str
    label: int
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    max_r: float
    max_g: float
    max_b: float
    min_r: float
    min_g: float
    min_b: float
    ratio_rg: float
    ratio_gb: float
    ratio_rb: float
    intensity_mean: float
    intensity_std: float
    intensity_skew: float
    edge_density: float
    mask_coverage: float


def _safe_ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def _calc_skew(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = values.mean()
    std = values.std()
    if std == 0:
        return 0.0
    centered = values - mean
    return float(np.mean((centered / std) ** 3))


def _sobel_edge_density(image: np.ndarray) -> float:
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image
    gray = gray / 255.0
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = _convolve2d(gray, kernel_x)
    gy = _convolve2d(gray, kernel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    threshold = 0.2
    edge_pixels = np.sum(magnitude > threshold)
    total_pixels = magnitude.size
    if total_pixels == 0:
        return 0.0
    return float(edge_pixels / total_pixels)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ih, iw = image.shape
    out_h = ih - kh + 1
    out_w = iw - kw + 1
    if out_h <= 0 or out_w <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = float(np.sum(image[i : i + kh, j : j + kw] * kernel))
    return out


def extract_features(
    data_root: Path,
    max_images: Optional[int] = None,
) -> List[FeatureRow]:
    features: List[FeatureRow] = []
    for label_name, label_idx in LABEL_MAP.items():
        image_dir = data_root / "train_images" / label_name
        mask_dir = data_root / "train_masks"
        image_paths = sorted(image_dir.glob("*.png"))
        if max_images is not None:
            image_paths = image_paths[:max_images]
        for image_path in image_paths:
            with Image.open(image_path) as img:
                arr = np.array(img.convert("RGB"), dtype=np.float32)
            means = arr.mean(axis=(0, 1))
            stds = arr.std(axis=(0, 1))
            maxs = arr.max(axis=(0, 1))
            mins = arr.min(axis=(0, 1))
            intensity = arr.mean(axis=2)
            intensity_mean = float(intensity.mean())
            intensity_std = float(intensity.std())
            intensity_skew = _calc_skew(intensity.flatten())
            edge_density = _sobel_edge_density(arr)
            feature = FeatureRow(
                case_id=image_path.stem,
                label=label_idx,
                mean_r=float(means[0]),
                mean_g=float(means[1]),
                mean_b=float(means[2]),
                std_r=float(stds[0]),
                std_g=float(stds[1]),
                std_b=float(stds[2]),
                max_r=float(maxs[0]),
                max_g=float(maxs[1]),
                max_b=float(maxs[2]),
                min_r=float(mins[0]),
                min_g=float(mins[1]),
                min_b=float(mins[2]),
                ratio_rg=_safe_ratio(means[0], means[1]),
                ratio_gb=_safe_ratio(means[1], means[2]),
                ratio_rb=_safe_ratio(means[0], means[2]),
                intensity_mean=intensity_mean,
                intensity_std=intensity_std,
                intensity_skew=intensity_skew,
                edge_density=edge_density,
                mask_coverage=_mask_coverage(mask_dir, image_path.stem),
            )
            features.append(feature)
    return features


def _mask_coverage(mask_dir: Path, case_id: str) -> float:
    mask_path = mask_dir / f"{case_id}.npy"
    if not mask_path.exists():
        return 0.0
    mask = np.load(mask_path).astype(np.float32)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    total = mask.size
    if total == 0:
        return 0.0
    return float(mask.sum() / total)


def write_csv(rows: Iterable[FeatureRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output_path.write_text("")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract tabular features from images for scikit-learn baselines."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path to extracted dataset root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_BASE_DIR / "outputs" / "features" / "tabular_features.csv",
        help="Where to write the feature CSV.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional cap on images per class for quick smoke tests.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    rows = extract_features(args.data_root, args.max_images)
    if not rows:
        print("No images processed; check --data-root.")
        return 1
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
