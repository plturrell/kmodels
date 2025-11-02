"""Frequency domain feature extraction for scientific image forgery detection.

This module focuses on classical multimedia forensics cues such as JPEG
compression artifacts, discrete cosine transform (DCT) statistics and
error-level analysis (ELA). The resulting descriptors can be combined with
the existing tabular baseline or fed into deep models (e.g. the dual-stream
architecture introduced alongside this feature set).

The implementation intentionally favours readability and deterministic
behaviour over micro-optimisations so that the features can double as
explainable diagnostics when analysing suspicious samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class FrequencyFeatures:
    """Container holding frequency-domain statistics for a single image."""

    case_id: str

    # DCT-based descriptors
    dct_mean_low: float
    dct_mean_high: float
    dct_energy_ratio: float
    dct_high_std: float

    # Error Level Analysis
    ela_mean: float
    ela_std: float
    ela_max: float

    # JPEG/grid artefacts
    jpeg_grid_strength: float
    double_compression_score: float

    # Benford's law compliance for DCT coefficients
    benford_deviation: float

    # Noise statistics in frequency bands
    high_freq_noise: float
    mid_freq_noise: float


# ---------------------------------------------------------------------------
# Utility helpers


def _load_image_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"), dtype=np.float32)


def _block_view(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Return a view of ``arr`` split into non-overlapping blocks."""

    h, w = arr.shape
    h_crop = h - (h % block_size)
    w_crop = w - (w % block_size)
    arr = arr[:h_crop, :w_crop]
    new_shape = (h_crop // block_size, w_crop // block_size, block_size, block_size)
    return arr.reshape(new_shape)


def _compute_block_dct(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def _dct_energy_stats(image: np.ndarray, block_size: int = 8) -> Tuple[float, float, float, float]:
    blocks = _block_view(image, block_size)
    # Shape: (h_blocks, w_blocks, block_size, block_size)
    flat_blocks = blocks.reshape(-1, block_size, block_size)

    low_band: List[float] = []
    high_band: List[float] = []

    for blk in flat_blocks:
        coeffs = _compute_block_dct(blk)
        low_coeff = coeffs[0, 0]
        # Define high frequency mask omitting DC and first row/column
        high_coeffs = coeffs[1:, 1:].flatten()
        low_band.append(abs(low_coeff))
        high_band.extend(np.abs(high_coeffs))

    low_band_arr = np.array(low_band, dtype=np.float32)
    high_band_arr = np.array(high_band, dtype=np.float32)

    dct_mean_low = float(low_band_arr.mean()) if low_band_arr.size else 0.0
    dct_mean_high = float(high_band_arr.mean()) if high_band_arr.size else 0.0
    dct_energy_ratio = float(
        (high_band_arr.sum() + 1e-6) / (low_band_arr.sum() + 1e-6)
    )
    dct_high_std = float(high_band_arr.std()) if high_band_arr.size else 0.0

    return dct_mean_low, dct_mean_high, dct_energy_ratio, dct_high_std


def _error_level_analysis(image: np.ndarray, quality: int = 90) -> Tuple[float, float, float]:
    """Perform Error Level Analysis by recompressing the image as JPEG."""

    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    success, encoded = cv2.imencode(".jpg", image_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return 0.0, 0.0, 0.0
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ela = np.abs(image - decoded)
    return float(ela.mean()), float(ela.std()), float(ela.max())


def _jpeg_grid_strength(image: np.ndarray, block_size: int = 8) -> float:
    """Estimate JPEG grid artefacts via block boundary energy."""

    h, w = image.shape
    vertical_edges = []
    horizontal_edges = []

    for x in range(block_size, w, block_size):
        col_diff = np.abs(image[:, x : x + 1].astype(np.float32) - image[:, x - 1 : x].astype(np.float32))
        vertical_edges.append(col_diff.mean())

    for y in range(block_size, h, block_size):
        row_diff = np.abs(image[y : y + 1, :].astype(np.float32) - image[y - 1 : y, :].astype(np.float32))
        horizontal_edges.append(row_diff.mean())

    if not vertical_edges and not horizontal_edges:
        return 0.0

    return float(np.mean(vertical_edges + horizontal_edges))


def _double_compression_score(image: np.ndarray, block_size: int = 8) -> float:
    """Heuristic score for JPEG double compression via DCT histogram periodicity."""

    blocks = _block_view(image, block_size)
    flat_blocks = blocks.reshape(-1, block_size, block_size)
    hist_accum: List[np.ndarray] = []

    for blk in flat_blocks:
        coeffs = _compute_block_dct(blk)
        ac_coeffs = coeffs.flatten()[1:]
        if ac_coeffs.size == 0:
            continue
        hist, _ = np.histogram(ac_coeffs, bins=100, range=(-50, 50), density=True)
        hist_accum.append(hist)

    if not hist_accum:
        return 0.0

    hist_stack = np.stack(hist_accum, axis=0)
    spectrum = np.fft.rfft(hist_stack, axis=1)
    magnitude = np.abs(spectrum).mean(axis=0)
    # Skip the DC component; higher peaks indicate periodicity (double compression)
    if magnitude.size <= 2:
        return 0.0
    return float(magnitude[1:].max() / (magnitude[1:].mean() + 1e-6))


def _benford_deviation(image: np.ndarray, block_size: int = 8) -> float:
    """Compute deviation from Benford's law for DCT coefficients."""

    blocks = _block_view(image, block_size)
    flat_blocks = blocks.reshape(-1, block_size, block_size)
    coeffs_list: List[int] = []

    for blk in flat_blocks:
        coeffs = _compute_block_dct(blk)
        ac_coeffs = np.abs(coeffs.flatten()[1:])
        first_digits = (ac_coeffs // 10 ** np.floor(np.log10(ac_coeffs + 1e-6))).astype(np.int32)
        valid = first_digits[(first_digits >= 1) & (first_digits <= 9)]
        coeffs_list.extend(valid.tolist())

    if not coeffs_list:
        return 0.0

    observed, _ = np.histogram(coeffs_list, bins=np.arange(1, 11), density=True)
    benford = np.array([np.log10(1 + 1 / d) for d in range(1, 10)], dtype=np.float32)
    return float(np.linalg.norm(observed - benford, ord=1))


def _frequency_noise(image: np.ndarray, low_cut: float = 0.1, high_cut: float = 0.3) -> Tuple[float, float]:
    """Estimate noise power in specified frequency bands using FFT."""

    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = image.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    norm_radius = radius / radius.max()

    mid_band = magnitude[(norm_radius >= low_cut) & (norm_radius < high_cut)]
    high_band = magnitude[norm_radius >= high_cut]

    high_noise = float(np.mean(high_band)) if high_band.size else 0.0
    mid_noise = float(np.mean(mid_band)) if mid_band.size else 0.0
    return high_noise, mid_noise


# ---------------------------------------------------------------------------
# Public API


def extract_frequency_features(image_path: Path) -> FrequencyFeatures:
    """Extract frequency-based forensic descriptors for a single image."""

    image = _load_image_grayscale(image_path)

    dct_mean_low, dct_mean_high, dct_ratio, dct_std = _dct_energy_stats(image)
    ela_mean, ela_std, ela_max = _error_level_analysis(image)
    jpeg_grid = _jpeg_grid_strength(image)
    double_comp = _double_compression_score(image)
    benford = _benford_deviation(image)
    high_noise, mid_noise = _frequency_noise(image)

    return FrequencyFeatures(
        case_id=image_path.stem,
        dct_mean_low=dct_mean_low,
        dct_mean_high=dct_mean_high,
        dct_energy_ratio=dct_ratio,
        dct_high_std=dct_std,
        ela_mean=ela_mean,
        ela_std=ela_std,
        ela_max=ela_max,
        jpeg_grid_strength=jpeg_grid,
        double_compression_score=double_comp,
        benford_deviation=benford,
        high_freq_noise=high_noise,
        mid_freq_noise=mid_noise,
    )


def batch_extract_frequency_features(
    image_paths: Iterable[Path],
) -> List[FrequencyFeatures]:
    """Return frequency descriptors for a collection of image paths."""

    features: List[FrequencyFeatures] = []
    for path in image_paths:
        try:
            features.append(extract_frequency_features(path))
        except Exception as exc:  # pragma: no cover - diagnostic print for users
            print(f"[frequency_analysis] Failed to process {path}: {exc}")
    return features


def frequency_features_to_dict(freq: FrequencyFeatures) -> Dict[str, float]:
    """Convert :class:`FrequencyFeatures` to a dict suitable for DataFrames."""

    return asdict(freq)


def write_frequency_features(
    rows: Iterable[FrequencyFeatures],
    output_path: Path,
) -> None:
    """Persist frequency features as a CSV file."""

    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output_path.write_text("")
        return

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


__all__ = [
    "FrequencyFeatures",
    "batch_extract_frequency_features",
    "extract_frequency_features",
    "frequency_features_to_dict",
    "write_frequency_features",
]


