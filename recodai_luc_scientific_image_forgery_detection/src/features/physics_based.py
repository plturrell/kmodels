"""Approximate physics-based descriptors for scientific image forensics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
from PIL import Image


@dataclass
class PhysicsFeatures:
    case_id: str
    illumination_angle_variance: float
    illumination_color_temperature: float
    shadow_alignment_score: float
    vanishing_point_dispersion: float
    scale_consistency: float
    radial_distortion: float
    chromatic_aberration: float
    focus_measure: float


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.float32)


def _illumination_metrics(image: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_norm = gray / (gray.max() + 1e-6)
    highlight_mask = gray_norm > 0.9
    dark_mask = gray_norm < 0.2

    gx = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)
    angles = np.arctan2(gy, gx)

    highlight_angles = angles[highlight_mask]
    dark_angles = angles[dark_mask]

    angle_var = float(np.var(highlight_angles)) if highlight_angles.size else 0.0

    highlights = image[highlight_mask]
    if highlights.size:
        avg_rgb = highlights.mean(axis=0)
        color_temp = float(avg_rgb[0] / (avg_rgb[2] + 1e-6))
    else:
        color_temp = 1.0

    if highlight_angles.size and dark_angles.size:
        diff = np.abs(np.unwrap(highlight_angles) - np.unwrap(dark_angles[: highlight_angles.size]))
        shadow_align = float(np.mean(np.cos(diff)))
    else:
        shadow_align = 0.0

    return angle_var, color_temp, shadow_align


def _vanishing_point_dispersion(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None or len(lines) < 4:
        return 0.0

    angles = np.array([theta for _, theta in lines[:, 0]])
    return float(np.var(np.cos(angles)))


def _scale_consistency(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(gray.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 0])
    if areas.size < 2:
        return 0.0
    return float(np.std(np.log(areas + 1e-6)))


def _radial_distortion(image: np.ndarray) -> float:
    h, w, _ = image.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.indices((h, w))
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    normalized = gray / (gray.max() + 1e-6)
    radial_profile = normalized.mean(axis=1)
    coeffs = np.polyfit(radius.mean(axis=1), radial_profile, deg=2)
    return float(abs(coeffs[0]))


def _chromatic_aberration(image: np.ndarray) -> float:
    b, g, r = cv2.split(image)
    edges_b = cv2.Canny(b.astype(np.uint8), 50, 150)
    edges_r = cv2.Canny(r.astype(np.uint8), 50, 150)
    diff = cv2.absdiff(edges_b, edges_r)
    return float(diff.mean() / 255.0)


def _focus_measure(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(fm))


def extract_physics_features(image_path: Path) -> PhysicsFeatures:
    image = _load_rgb(image_path)

    illumination_var, color_temp, shadow_align = _illumination_metrics(image)
    vanishing_dispersion = _vanishing_point_dispersion(image)
    scale_consistency = _scale_consistency(image)
    radial_dist = _radial_distortion(image)
    chrom_ab = _chromatic_aberration(image)
    focus = _focus_measure(image)

    return PhysicsFeatures(
        case_id=image_path.stem,
        illumination_angle_variance=illumination_var,
        illumination_color_temperature=color_temp,
        shadow_alignment_score=shadow_align,
        vanishing_point_dispersion=vanishing_dispersion,
        scale_consistency=scale_consistency,
        radial_distortion=radial_dist,
        chromatic_aberration=chrom_ab,
        focus_measure=focus,
    )


def batch_extract_physics_features(image_paths: Iterable[Path]) -> List[PhysicsFeatures]:
    rows: List[PhysicsFeatures] = []
    for path in image_paths:
        try:
            rows.append(extract_physics_features(path))
        except Exception as exc:  # pragma: no cover
            print(f"[physics_based] Failed to process {path}: {exc}")
    return rows


def physics_features_to_dict(features: PhysicsFeatures) -> Dict[str, float]:
    return asdict(features)


__all__ = [
    "PhysicsFeatures",
    "batch_extract_physics_features",
    "extract_physics_features",
    "physics_features_to_dict",
]


