"""Feature engineering for player kinematics with graceful fallbacks."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _ensure_copy(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy()


def add_player_kinematics(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add speed/acceleration/orientation features when derivatives exist."""
    tracking_df = _ensure_copy(tracking_df)
    required = {"s_x", "s_y", "a_x", "a_y", "o"}
    missing = required - set(tracking_df.columns)
    if missing:
        LOGGER.warning("Skipping kinematics features; missing columns: %s", sorted(missing))
        tracking_df["speed"] = tracking_df.get("speed", 0.0)
        tracking_df["acceleration"] = tracking_df.get("acceleration", 0.0)
        tracking_df["orientation_rad"] = tracking_df.get("orientation_rad", 0.0)
        return tracking_df

    tracking_df["speed"] = np.sqrt(tracking_df["s_x"] ** 2 + tracking_df["s_y"] ** 2)
    tracking_df["acceleration"] = np.sqrt(tracking_df["a_x"] ** 2 + tracking_df["a_y"] ** 2)
    tracking_df["orientation_rad"] = np.deg2rad(tracking_df["o"])
    return tracking_df


def add_distance_to_ball(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Add distance-to-ball feature when football positions are available."""
    tracking_df = _ensure_copy(tracking_df)
    required = {"player_id", "game_play_id", "x", "y"}
    missing = required - set(tracking_df.columns)
    if missing:
        LOGGER.warning("Skipping distance-to-ball feature; missing columns: %s", sorted(missing))
        tracking_df["distance_to_ball"] = tracking_df.get("distance_to_ball", np.nan)
        return tracking_df

    ball_position = (
        tracking_df.loc[tracking_df["player_id"] == "football", ["game_play_id", "x", "y"]]
        .rename(columns={"x": "ball_x", "y": "ball_y"})
    )
    tracking_df = tracking_df.merge(ball_position, on="game_play_id", how="left")
    tracking_df["distance_to_ball"] = np.sqrt(
        (tracking_df["x"] - tracking_df["ball_x"]) ** 2
        + (tracking_df["y"] - tracking_df["ball_y"]) ** 2
    )
    tracking_df = tracking_df.drop(columns=["ball_x", "ball_y"])
    return tracking_df
