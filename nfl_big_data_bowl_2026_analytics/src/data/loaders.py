"""Helpers to load NFL Big Data Bowl 2026 Analytics datasets into pandas."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_DOWNLOAD_ROOT = Path("competitions/nfl_big_data_bowl_2026_analytics/data/raw")
DEFAULT_BUNDLE_DIRNAME = "nfl-big-data-bowl-2026-analytics"
_WEEK_PATTERN = re.compile(r"(?:input|output)_(\d{4})_w(\d{2})", re.IGNORECASE)
_SUPPLEMENTARY_FILE = "supplementary_data.csv"


def _resolve_bundle_root(base_dir: Optional[Path | str]) -> Path:
    """Return the directory that contains the ``train`` folder."""
    base = Path(base_dir) if base_dir is not None else DEFAULT_DOWNLOAD_ROOT
    if not base.exists():
        raise FileNotFoundError(
            f"Data directory {base} not found. Run the download script first."
        )

    if (base / "train").is_dir():
        return base

    default_candidate = base / DEFAULT_BUNDLE_DIRNAME
    if (default_candidate / "train").is_dir():
        return default_candidate
    if default_candidate.is_dir():
        nested = [
            child for child in default_candidate.iterdir() if (child / "train").is_dir()
        ]
        if len(nested) == 1:
            return nested[0]

    candidates = [child for child in base.iterdir() if (child / "train").is_dir()]
    # Check one level deeper if nothing found.
    if not candidates:
        for child in base.iterdir():
            if child.is_dir():
                candidates.extend(
                    grandchild
                    for grandchild in child.iterdir()
                    if (grandchild / "train").is_dir()
                )

    if len(candidates) == 1:
        return candidates[0]

    pretty = ", ".join(str(c) for c in candidates) or "<none>"
    raise FileNotFoundError(
        "Could not determine analytics bundle root. "
        f"Tried {default_candidate}. Discovered candidates: {pretty}"
    )


def _format_week(season: int, week: int) -> str:
    return f"{season}_w{week:02d}"


def _week_filepath(
    prefix: str,
    season: int,
    week: int,
    base_dir: Optional[Path | str] = None,
) -> Path:
    week_tag = _format_week(season, week)
    bundle_root = _resolve_bundle_root(base_dir)
    path = bundle_root / "train" / f"{prefix}_{week_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not locate file {path}")
    return path


def available_train_weeks(base_dir: Optional[Path | str] = None) -> List[Tuple[int, int]]:
    """Return sorted list of (season, week) pairs with training data."""
    bundle_root = _resolve_bundle_root(base_dir)
    train_dir = bundle_root / "train"
    weeks = set()
    for path in train_dir.glob("input_*.csv"):
        match = _WEEK_PATTERN.search(path.stem)
        if match:
            weeks.add((int(match.group(1)), int(match.group(2))))
    if not weeks:
        raise FileNotFoundError(
            f"No training weeks discovered under {train_dir}. Check your data download."
        )
    return sorted(weeks)


def load_train_week_input(
    season: int,
    week: int,
    base_dir: Optional[Path | str] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a single weekly training input CSV."""
    path = _week_filepath("input", season, week, base_dir)
    return pd.read_csv(path, **read_csv_kwargs)


def load_train_week_output(
    season: int,
    week: int,
    base_dir: Optional[Path | str] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load the matching weekly training output CSV."""
    path = _week_filepath("output", season, week, base_dir)
    return pd.read_csv(path, **read_csv_kwargs)


def load_train_week_pair(
    season: int,
    week: int,
    base_dir: Optional[Path | str] = None,
    input_kwargs: Optional[dict] = None,
    output_kwargs: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch input/output DataFrames for a single (season, week)."""
    input_kwargs = input_kwargs or {}
    output_kwargs = output_kwargs or {}
    inputs = load_train_week_input(season, week, base_dir=base_dir, **input_kwargs)
    outputs = load_train_week_output(season, week, base_dir=base_dir, **output_kwargs)
    return inputs, outputs


def load_train_inputs(
    weeks: Optional[Sequence[Tuple[int, int]]] = None,
    base_dir: Optional[Path | str] = None,
    concatenate: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame | List[pd.DataFrame]:
    """Load multiple weekly input files (all weeks by default)."""
    if weeks is None:
        weeks = available_train_weeks(base_dir=base_dir)
    frames = [
        load_train_week_input(season, week, base_dir=base_dir, **read_csv_kwargs)
        for season, week in weeks
    ]
    if concatenate:
        return pd.concat(frames, ignore_index=True)
    return frames


def load_train_outputs(
    weeks: Optional[Sequence[Tuple[int, int]]] = None,
    base_dir: Optional[Path | str] = None,
    concatenate: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame | List[pd.DataFrame]:
    """Load multiple weekly output files (all weeks by default)."""
    if weeks is None:
        weeks = available_train_weeks(base_dir=base_dir)
    frames = [
        load_train_week_output(season, week, base_dir=base_dir, **read_csv_kwargs)
        for season, week in weeks
    ]
    if concatenate:
        return pd.concat(frames, ignore_index=True)
    return frames


def load_supplementary(
    base_dir: Optional[Path | str] = None, **read_csv_kwargs
) -> pd.DataFrame:
    """Load the supplemental analytics summary table."""
    bundle_root = _resolve_bundle_root(base_dir)
    path = bundle_root / _SUPPLEMENTARY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Could not locate file {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def resolve_bundle_root(base_dir: Optional[Path | str] = None) -> Path:
    """Expose the resolved bundle root for downstream tooling."""
    return _resolve_bundle_root(base_dir)


__all__ = [
    "DEFAULT_DOWNLOAD_ROOT",
    "DEFAULT_BUNDLE_DIRNAME",
    "resolve_bundle_root",
    "available_train_weeks",
    "load_train_inputs",
    "load_train_outputs",
    "load_train_week_input",
    "load_train_week_output",
    "load_train_week_pair",
    "load_supplementary",
]
