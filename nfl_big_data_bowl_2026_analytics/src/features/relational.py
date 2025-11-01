"""Feature engineering for relational player and field features."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


LOGGER = logging.getLogger(__name__)


def add_relational_features(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Adds pairwise spacing and field-boundary features when metadata permits."""
    required = {"game_play_id", "team", "x", "y", "play_direction"}
    missing = required - set(tracking_df.columns)
    tracking_df = tracking_df.copy()

    if missing:
        LOGGER.warning("Skipping relational features; missing columns: %s", sorted(missing))
        for column in (
            "distance_to_nearest_opponent",
            "distance_to_nearest_teammate",
            "distance_to_sideline",
            "distance_to_endzone",
        ):
            tracking_df[column] = tracking_df.get(column, np.nan)
        return tracking_df

    enriched_rows = []
    for _, play_df in tracking_df.groupby("game_play_id"):
        play_df = play_df.copy()

        offense_df = play_df[play_df["team"] == "offense"].copy()
        defense_df = play_df[play_df["team"] == "defense"].copy()

        if not offense_df.empty and not defense_df.empty:
            off_def_dist = cdist(offense_df[["x", "y"]], defense_df[["x", "y"]])
            offense_df.loc[:, "distance_to_nearest_opponent"] = off_def_dist.min(axis=1)
            defense_df.loc[:, "distance_to_nearest_opponent"] = off_def_dist.min(axis=0)

            if len(offense_df) > 1:
                off_off = cdist(offense_df[["x", "y"]], offense_df[["x", "y"]])
                np.fill_diagonal(off_off, np.inf)
                offense_df.loc[:, "distance_to_nearest_teammate"] = off_off.min(axis=1)
            else:
                offense_df.loc[:, "distance_to_nearest_teammate"] = np.nan

            if len(defense_df) > 1:
                def_def = cdist(defense_df[["x", "y"]], defense_df[["x", "y"]])
                np.fill_diagonal(def_def, np.inf)
                defense_df.loc[:, "distance_to_nearest_teammate"] = def_def.min(axis=1)
            else:
                defense_df.loc[:, "distance_to_nearest_teammate"] = np.nan

            play_df = pd.concat([offense_df, defense_df], ignore_index=True)
        else:
            play_df.loc[:, "distance_to_nearest_opponent"] = np.nan
            play_df.loc[:, "distance_to_nearest_teammate"] = np.nan

        play_df.loc[:, "distance_to_sideline"] = np.minimum(play_df["y"], 53.3 - play_df["y"])

        play_direction = play_df["play_direction"].iloc[0]
        if play_direction == "left":
            play_df.loc[:, "distance_to_endzone"] = play_df["x"]
        else:
            play_df.loc[:, "distance_to_endzone"] = 120 - play_df["x"]

        enriched_rows.append(play_df)

    return pd.concat(enriched_rows, ignore_index=True)
