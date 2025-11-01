"""Sequence dataset helpers for trajectory modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TrajectorySample:
    features: torch.Tensor
    positions: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    metadata: Dict[str, int]


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        target_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_player_flag: str = "player_to_predict",
        player_id_column: str = "nfl_id",
        spatial_columns: Tuple[str, str] = ("x", "y"),
        max_players: int = 22,
        max_time: int = 21,
        normalise_xy: bool = True,
        max_sequences: Optional[int] = None,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.target_player_flag = target_player_flag
        self.player_id_column = player_id_column
        self.spatial_columns = spatial_columns
        self.max_players = max_players
        self.max_time = max_time
        self.normalise_xy = normalise_xy

        grouped = frame.groupby(["game_id", "play_id"])
        labels = target_frame.set_index(["game_id", "play_id", player_id_column, "frame_id"])
        self.samples: List[TrajectorySample] = []

        for (game_id, play_id), group in grouped:
            players = group[player_id_column].unique()
            time_steps = sorted(group["frame_id"].unique())
            if not players.size or not time_steps:
                continue

            feat_tensor = torch.zeros(max_time, max_players, len(self.feature_columns), dtype=torch.float32)
            pos_tensor = torch.zeros(max_time, max_players, 2, dtype=torch.float32)
            mask_tensor = torch.zeros(max_time, max_players, dtype=torch.bool)
            target_tensor = torch.zeros(max_time, 2, dtype=torch.float32)

            player_to_index: Dict[int, int] = {}
            for idx, player in enumerate(players[: max_players]):
                player_to_index[int(player)] = idx

            for t_idx, frame_id in enumerate(time_steps[: max_time]):
                frame_slice = group[group["frame_id"] == frame_id]
                for _, row in frame_slice.iterrows():
                    player = int(row[player_id_column])
                    slot = player_to_index.get(player)
                    if slot is None:
                        if len(player_to_index) >= max_players:
                            continue
                        slot = len(player_to_index)
                        player_to_index[player] = slot
                    mask_tensor[t_idx, slot] = True
                    numeric_values = row.loc[self.feature_columns].astype("float32").fillna(0.0)
                    feat_tensor[t_idx, slot] = torch.from_numpy(
                        numeric_values.to_numpy(copy=False)
                    )
                    pos_tensor[t_idx, slot] = torch.tensor([row[spatial_columns[0]], row[spatial_columns[1]]], dtype=torch.float32)

                target_rows = frame_slice[frame_slice[self.target_player_flag]]
                if not target_rows.empty:
                    player = int(target_rows.iloc[0][player_id_column])
                    key = (game_id, play_id, player, frame_id)
                    if key in labels.index:
                        label_row = labels.loc[key]
                        target_tensor[t_idx] = torch.tensor([label_row["x"], label_row["y"]], dtype=torch.float32)

            if not mask_tensor.any():
                continue

            if normalise_xy:
                valid_positions = pos_tensor[mask_tensor]
                mean = valid_positions.mean(dim=0, keepdim=True)
                std = valid_positions.std(dim=0, keepdim=True).clamp_min(1.0)
                pos_tensor[mask_tensor] = (pos_tensor[mask_tensor] - mean) / std
                target_tensor = (target_tensor - mean.squeeze(0)) / std.squeeze(0)

            metadata = {"game_id": int(game_id), "play_id": int(play_id)}
            self.samples.append(
                TrajectorySample(
                    features=feat_tensor,
                    positions=pos_tensor,
                    mask=mask_tensor,
                    target=target_tensor,
                    metadata=metadata,
                )
            )
            if max_sequences is not None and len(self.samples) >= max_sequences:
                break
        if max_sequences is not None and len(self.samples) >= max_sequences:
            return

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrajectorySample:
        return self.samples[index]


def collate_trajectories(batch: Sequence[TrajectorySample]):
    features = torch.stack([sample.features for sample in batch], dim=0)
    positions = torch.stack([sample.positions for sample in batch], dim=0)
    mask = torch.stack([sample.mask for sample in batch], dim=0)
    targets = torch.stack([sample.target for sample in batch], dim=0)
    return {
        "features": features,
        "positions": positions,
        "mask": mask,
        "target": targets,
        "metadata": batch,
    }
