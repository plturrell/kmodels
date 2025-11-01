"""Quick dataset summary utilities for the LUC forgery detection challenge."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

from PIL import Image  # type: ignore
import numpy as np


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "recodai-luc-scientific-image-forgery-detection"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "dataset_overview.json"


@dataclass
class ImageStats:
    label_counts: Dict[str, int]
    sampled_image_shapes: List[Tuple[int, int]]
    channel_histogram: Dict[int, int]
    average_width: float
    average_height: float


@dataclass
class MaskStats:
    total_masks: int
    sample_shape: Tuple[int, ...]
    mask_dtype: str
    value_min: int
    value_max: int


def _iter_images(image_dir: Path) -> Iterable[Path]:
    for path in sorted(image_dir.glob("*.png")):
        yield path


def collect_image_stats(
    train_dir: Path, max_samples_per_class: int
) -> ImageStats:
    label_counts: Dict[str, int] = {}
    sampled_shapes: List[Tuple[int, int]] = []
    channel_hist = Counter()
    width_samples: List[int] = []
    height_samples: List[int] = []

    for label_dir in sorted(train_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        image_paths = list(_iter_images(label_dir))
        label_counts[label_dir.name] = len(image_paths)
        for path in image_paths[:max_samples_per_class]:
            with Image.open(path) as img:
                width, height = img.size
                sampled_shapes.append((width, height))
                channel_hist[len(img.getbands())] += 1
                width_samples.append(width)
                height_samples.append(height)

    average_width = mean(width_samples) if width_samples else 0.0
    average_height = mean(height_samples) if height_samples else 0.0

    return ImageStats(
        label_counts=dict(label_counts),
        sampled_image_shapes=sampled_shapes,
        channel_histogram=dict(channel_hist),
        average_width=average_width,
        average_height=average_height,
    )


def collect_mask_stats(mask_dir: Path) -> MaskStats:
    mask_paths = sorted(mask_dir.glob("*.npy"))
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    sample_mask = np.load(mask_paths[0])
    return MaskStats(
        total_masks=len(mask_paths),
        sample_shape=tuple(sample_mask.shape),
        mask_dtype=str(sample_mask.dtype),
        value_min=int(sample_mask.min()),
        value_max=int(sample_mask.max()),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a quick dataset overview and persist it as JSON."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path to the extracted competition data directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the JSON summary.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=200,
        help="How many images to inspect per class for shape/channel stats.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    data_root: Path = args.data_root
    train_dir = data_root / "train_images"
    mask_dir = data_root / "train_masks"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    LOGGER.info("Collecting image statistics from %s", train_dir)
    image_stats = collect_image_stats(
        train_dir=train_dir, max_samples_per_class=args.max_samples_per_class
    )

    LOGGER.info("Collecting mask statistics from %s", mask_dir)
    mask_stats = collect_mask_stats(mask_dir=mask_dir)

    summary = {
        "images": asdict(image_stats),
        "masks": asdict(mask_stats),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Wrote dataset summary to %s", args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
