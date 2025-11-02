"""Statistical anomaly features complementing physics-based cues."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
from PIL import Image


@dataclass
class StatisticalAnomalyFeatures:
    case_id: str
    noise_level_variance: float
    sensor_pattern_deviation: float
    compression_inconsistency: float
    resampling_detected: float
    histogram_kl_divergence: float
    color_channel_ks: float


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.float32)


def _noise_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    patch_size = 32
    h, w = gray.shape
    variances = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            variances.append(np.var(patch))
    return float(np.var(variances)) if variances else 0.0


def _sensor_pattern_deviation(image: np.ndarray) -> float:
    green = image[:, :, 1]
    highpass = green - cv2.GaussianBlur(green, (5, 5), 0)
    rows = highpass.mean(axis=1)
    spectrum = np.fft.rfft(rows)
    magnitudes = np.abs(spectrum)
    return float(np.std(magnitudes))


def _compression_inconsistency(image: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    y_channel = ycrcb[:, :, 0]
    hist, _ = np.histogram(y_channel.flatten(), bins=64, range=(0, 255))
    hist = hist / (hist.sum() + 1e-6)
    block_hist = []
    for y in range(0, y_channel.shape[0] - 8, 8):
        for x in range(0, y_channel.shape[1] - 8, 8):
            block = y_channel[y : y + 8, x : x + 8]
            h_block, _ = np.histogram(block.flatten(), bins=64, range=(0, 255))
            h_block = h_block / (h_block.sum() + 1e-6)
            block_hist.append(h_block)
    if not block_hist:
        return 0.0
    block_hist = np.stack(block_hist, axis=0)
    divergence = np.mean(np.sum(block_hist * np.log((block_hist + 1e-6) / (hist + 1e-6)), axis=1))
    return float(divergence)


def _resampling_artifacts(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fft = np.fft.fft2(gray)
    magnitude = np.abs(fft)
    rows = magnitude.mean(axis=1)
    peaks = np.sort(rows)[-10:]
    return float(peaks.mean() / (rows.mean() + 1e-6))


def _histogram_kl(image: np.ndarray) -> float:
    r, g, b = cv2.split(image)
    hist_r, _ = np.histogram(r.flatten(), bins=128, range=(0, 255), density=True)
    hist_g, _ = np.histogram(g.flatten(), bins=128, range=(0, 255), density=True)
    return float(np.sum(hist_r * np.log((hist_r + 1e-6) / (hist_g + 1e-6))))


def _color_channel_ks(image: np.ndarray) -> float:
    r, g, b = cv2.split(image)
    r_sorted = np.sort(r.flatten())
    g_sorted = np.sort(g.flatten())
    ks_stat = np.max(np.abs(np.linspace(0, 1, r_sorted.size) - np.linspace(0, 1, g_sorted.size)))
    return float(ks_stat)


def extract_statistical_features(image_path: Path) -> StatisticalAnomalyFeatures:
    image = _load_rgb(image_path)

    noise_var = _noise_variance(image)
    sensor_dev = _sensor_pattern_deviation(image)
    compression = _compression_inconsistency(image)
    resampling = _resampling_artifacts(image)
    hist_kl = _histogram_kl(image)
    color_ks = _color_channel_ks(image)

    return StatisticalAnomalyFeatures(
        case_id=image_path.stem,
        noise_level_variance=noise_var,
        sensor_pattern_deviation=sensor_dev,
        compression_inconsistency=compression,
        resampling_detected=resampling,
        histogram_kl_divergence=hist_kl,
        color_channel_ks=color_ks,
    )


def batch_extract_statistical_features(image_paths: Iterable[Path]) -> List[StatisticalAnomalyFeatures]:
    rows: List[StatisticalAnomalyFeatures] = []
    for path in image_paths:
        try:
            rows.append(extract_statistical_features(path))
        except Exception as exc:  # pragma: no cover
            print(f"[statistical_anomalies] Failed to process {path}: {exc}")
    return rows


def statistical_features_to_dict(features: StatisticalAnomalyFeatures) -> Dict[str, float]:
    return asdict(features)


__all__ = [
    "StatisticalAnomalyFeatures",
    "batch_extract_statistical_features",
    "extract_statistical_features",
    "statistical_features_to_dict",
]


