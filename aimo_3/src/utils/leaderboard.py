"""Leaderboard tracking utilities."""

import json
from pathlib import Path
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi


def check_leaderboard(
    competition: str = "aimo3",
    team_name: Optional[str] = None,
    top_n: int = 20,
) -> None:
    """
    Check competition leaderboard.

    Args:
        competition: Competition slug
        team_name: Your team name (optional)
        top_n: Number of top entries to display
    """
    api = KaggleApi()
    api.authenticate()

    print(f"Leaderboard for {competition}:")
    print("-" * 80)

    # Get leaderboard
    leaderboard = api.competition_view_leaderboard(competition)

    # Display top entries
    for i, entry in enumerate(leaderboard[:top_n], 1):
        team = entry.get("teamName", "Unknown")
        score = entry.get("score", "N/A")
        print(f"{i:3d}. {team:40s} {score}")

    # Highlight your team if specified
    if team_name:
        for entry in leaderboard:
            if entry.get("teamName") == team_name:
                rank = entry.get("rank", "N/A")
                score = entry.get("score", "N/A")
                print(f"\nYour team '{team_name}': Rank {rank}, Score {score}")
                break

