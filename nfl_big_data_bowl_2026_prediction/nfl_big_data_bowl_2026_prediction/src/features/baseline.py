"""Baseline feature engineering utilities for NFL Big Data Bowl 2026."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

from competitions.nfl_big_data_bowl_2026_prediction.src.data import (
    DEFAULT_DATA_ROOT,
    available_train_weeks,
    load_train_week_pair,
)


FEATURE_KEYS = ["game_id", "play_id", "nfl_id"]


def _ensure_weeks(
    weeks: Optional[Sequence[Tuple[int, int]]], base_dir: Path | str | None
) -> Sequence[Tuple[int, int]]:
    return (
        weeks
        if weeks is not None
        else available_train_weeks(base_dir=base_dir)
    )


def compute_player_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """Create per-player summary statistics from trajectory inputs."""
    grouped = inputs.sort_values("frame_id").groupby(FEATURE_KEYS, as_index=False)
    features = grouped.agg(
        player_to_predict=("player_to_predict", "max"),
        player_height=("player_height", "first"),
        player_weight=("player_weight", "first"),
        player_position=("player_position", "first"),
        player_role=("player_role", "first"),
        play_direction=("play_direction", "first"),
        absolute_yardline_number=("absolute_yardline_number", "first"),
        frames=("frame_id", "nunique"),
        start_x=("x", "first"),
        start_y=("y", "first"),
        end_x=("x", "last"),
        end_y=("y", "last"),
        mean_speed=("s", "mean"),
        max_speed=("s", "max"),
        mean_acceleration=("a", "mean"),
        max_acceleration=("a", "max"),
    )
    features["displacement_x"] = features["end_x"] - features["start_x"]
    features["displacement_y"] = features["end_y"] - features["start_y"]
    return features


def compute_targets(outputs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate output trajectories into final coordinate targets."""
    grouped = outputs.sort_values("frame_id").groupby(FEATURE_KEYS, as_index=False)
    targets = grouped.agg(
        target_final_x=("x", "last"),
        target_final_y=("y", "last"),
    )
    return targets


def build_week_features(
    season: int,
    week: int,
    base_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Construct features and targets for a single week."""
    inputs, outputs = load_train_week_pair(season, week, base_dir=base_dir)
    features = compute_player_features(inputs)
    targets = compute_targets(outputs)
    merged = features.merge(targets, on=FEATURE_KEYS, how="left", validate="1:1")
    merged.insert(0, "season", season)
    merged.insert(1, "week", week)
    return merged


def build_features(
    weeks: Optional[Sequence[Tuple[int, int]]] = None,
    base_dir: Path | str | None = None,
    concatenate: bool = True,
) -> pd.DataFrame | Iterable[pd.DataFrame]:
    """Generate features for the specified weeks (all weeks by default)."""
    weeks_to_process = _ensure_weeks(weeks, base_dir)
    frames = [
        build_week_features(season, week, base_dir=base_dir)
        for season, week in weeks_to_process
    ]
    if concatenate:
        return pd.concat(frames, ignore_index=True)
    return frames


def _write_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    """Persist dataframe to disk, preferring parquet with a CSV fallback."""
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(output_path, index=False)
            return output_path
        except ImportError:
            output_path = output_path.with_suffix(".csv")
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
        return output_path
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def save_features(
    output_path: Path,
    weeks: Optional[Sequence[Tuple[int, int]]] = None,
    base_dir: Path | str | None = None,
) -> Path:
    """Materialize engineered features to disk (parquet preferred, CSV fallback)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_features(weeks=weeks, base_dir=base_dir, concatenate=True)
    return _write_dataframe(df, output_path)
