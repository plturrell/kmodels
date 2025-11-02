"""Spatial-domain forensic descriptors for scientific image forgery detection.

The heuristics implemented here are inspired by classical multimedia forensic
literature. They intentionally trade absolute precision for tractability so
that they can be run at scale for Kaggle-style competitions while still
providing interpretable signals:

- CFA pattern analysis approximates the periodic structure of Bayer filters.
- PRNU captures sensor-specific noise residuals using a denoising residual.
- Local Binary Pattern (LBP) histograms expose texture anomalies.
- Illumination direction heuristics look for inconsistent lighting cues.

These descriptors complement the pure frequency-domain cues implemented in
``frequency_analysis.py`` and are designed to plug seamlessly into the
tabular baseline or any downstream ensembling strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class ForensicFeatures:
    case_id: str

    cfa_periodicity: float
    cfa_horizontal_strength: float
    cfa_vertical_strength: float

    prnu_energy: float
    prnu_correlation: float

    lbp_uniformity: float
    lbp_entropy: float

    illumination_consistency: float
    illumination_direction_variance: float


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.float32)


def _cfa_features(image: np.ndarray) -> Tuple[float, float, float]:
    """Estimate CFA periodicity via autocorrelation along axes."""

    green_channel = image[:, :, 1]
    mean_removed = green_channel - green_channel.mean()

    autocorr = cv2.matchTemplate(mean_removed, mean_removed, method=cv2.TM_CCORR_NORMED)
    center = autocorr[autocorr.shape[0] // 2, autocorr.shape[1] // 2]

    # Shifts corresponding to Bayer pattern periodicity (2 pixels)
    shift = 2
    if mean_removed.shape[1] > shift:
        horizontal = np.corrcoef(mean_removed[:, :-shift].flatten(), mean_removed[:, shift:].flatten())[0, 1]
    else:
        horizontal = 0.0
    if mean_removed.shape[0] > shift:
        vertical = np.corrcoef(mean_removed[:-shift, :].flatten(), mean_removed[shift:, :].flatten())[0, 1]
    else:
        vertical = 0.0

    periodicity = float(center)
    return periodicity, float(horizontal), float(vertical)


def _wavelet_denoise(channel: np.ndarray, ksize: int = 3) -> np.ndarray:
    return channel - cv2.GaussianBlur(channel, (ksize, ksize), 0)


def _prnu_features(image: np.ndarray) -> Tuple[float, float]:
    """Compute PRNU-inspired residual energy and self-correlation."""

    residual = np.stack([_wavelet_denoise(image[:, :, c]) for c in range(3)], axis=2)
    prnu_energy = float(np.mean(residual ** 2))

    # Correlate residual with green channel as proxy for sensor fingerprint
    green = image[:, :, 1]
    residual_gray = residual.mean(axis=2)
    numerator = np.sum((residual_gray - residual_gray.mean()) * (green - green.mean()))
    denominator = np.sqrt(
        np.sum((residual_gray - residual_gray.mean()) ** 2)
        * np.sum((green - green.mean()) ** 2)
        + 1e-6
    )
    prnu_corr = float(numerator / denominator)
    return prnu_energy, prnu_corr


def _lbp_histogram(image: np.ndarray, radius: int = 1, neighbors: int = 8) -> Tuple[float, float]:
    """Compute simplified rotationally invariant LBP histogram statistics."""

    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gray = gray / 255.0
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    for n in range(neighbors):
        theta = 2 * np.pi * n / neighbors
        y = radius * np.sin(theta)
        x = radius * np.cos(theta)
        y0 = np.clip(np.arange(h)[:, None] + y, 0, h - 1)
        x0 = np.clip(np.arange(w)[None, :] + x, 0, w - 1)
        sampled = gray[y0.astype(int), x0.astype(int)]
        lbp |= ((sampled >= gray).astype(np.uint8) << n)

    hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 255), density=True)
    hist = hist + 1e-6
    uniformity = float(np.sum(hist ** 2))
    entropy = float(-np.sum(hist * np.log(hist)))
    return uniformity, entropy


def _illumination_features(image: np.ndarray, sample_points: int = 64) -> Tuple[float, float]:
    """Estimate illumination consistency by comparing gradient fields."""

    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gray = gray / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    normalised = np.stack([gx / magnitude, gy / magnitude], axis=2)

    h, w = gray.shape
    rng = np.random.default_rng(seed=42)
    ys = rng.integers(0, h, size=sample_points)
    xs = rng.integers(0, w, size=sample_points)
    sampled = normalised[ys, xs]

    mean_direction = sampled.mean(axis=0)
    deviations = sampled - mean_direction
    variance = float(np.mean(np.linalg.norm(deviations, axis=1)))

    consistency = float(np.linalg.norm(mean_direction))
    return consistency, variance


def extract_forensic_features(image_path: Path) -> ForensicFeatures:
    """Return spatial forensic descriptors for a given RGB image."""

    image = _load_rgb(image_path)

    cfa_periodicity, cfa_h, cfa_v = _cfa_features(image)
    prnu_energy, prnu_corr = _prnu_features(image)
    lbp_uniformity, lbp_entropy = _lbp_histogram(image)
    illum_consistency, illum_var = _illumination_features(image)

    return ForensicFeatures(
        case_id=image_path.stem,
        cfa_periodicity=float(cfa_periodicity),
        cfa_horizontal_strength=float(cfa_h),
        cfa_vertical_strength=float(cfa_v),
        prnu_energy=float(prnu_energy),
        prnu_correlation=float(prnu_corr),
        lbp_uniformity=float(lbp_uniformity),
        lbp_entropy=float(lbp_entropy),
        illumination_consistency=float(illum_consistency),
        illumination_direction_variance=float(illum_var),
    )


def batch_extract_forensic_features(image_paths: Iterable[Path]) -> List[ForensicFeatures]:
    rows: List[ForensicFeatures] = []
    for path in image_paths:
        try:
            rows.append(extract_forensic_features(path))
        except Exception as exc:  # pragma: no cover - user insight
            print(f"[forensic_features] Failed to process {path}: {exc}")
    return rows


def forensic_features_to_dict(features: ForensicFeatures) -> Dict[str, float]:
    return asdict(features)


__all__ = [
    "ForensicFeatures",
    "batch_extract_forensic_features",
    "extract_forensic_features",
    "forensic_features_to_dict",
]


