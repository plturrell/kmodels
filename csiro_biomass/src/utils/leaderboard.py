"""Utilities for comparing local results against the Kaggle leaderboard."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kaggle.api.kaggle_api_extended import KaggleApi


def _load_local_score(metrics_path: Optional[Path]) -> Optional[float]:
    if metrics_path is None:
        return None
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r") as f:
        payload = json.load(f)
    # Support a variety of shapes (dict of scalars, run summary list, constraint metrics).
    if isinstance(payload, dict):
        for key in ("val_rmse", "rmse", "score"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        if "validation_summary" in payload and isinstance(payload["validation_summary"], dict):
            repaired = payload["validation_summary"].get("repaired")
            if isinstance(repaired, dict):
                score = repaired.get("rmse")
                if isinstance(score, (int, float)):
                    return float(score)
    if isinstance(payload, list) and payload:
        # Assume list of history entries with val_rmse
        rmse_values: List[float] = []
        for entry in payload:
            if isinstance(entry, dict):
                value = entry.get("val_rmse")
                if isinstance(value, (int, float)):
                    rmse_values.append(float(value))
        if rmse_values:
            return float(min(rmse_values))
    return None


def _summarise_leaderboard(
    rows: List[Dict[str, Any]],
    *,
    top_k: int,
    team_name: Optional[str],
) -> Tuple[List[Tuple[int, str, float]], Optional[Tuple[int, str, float]]]:
    def _get(row: Any, key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    summary: List[Tuple[int, str, float]] = []
    team_entry: Optional[Tuple[int, str, float]] = None

    for row in rows[:top_k]:
        team = _get(row, "teamName", _get(row, "team_name", "unknown"))
        rank = int(_get(row, "rankDisplay", _get(row, "teamRank", 0)))
        score = float(_get(row, "score", 0.0))
        summary.append((rank, team, score))
        if team_name and team.lower() == team_name.lower():
            team_entry = (rank, team, score)

    if team_entry is None and team_name:
        for row in rows:
            team = _get(row, "teamName", _get(row, "team_name", "unknown"))
            if team.lower() == team_name.lower():
                rank = int(_get(row, "rankDisplay", _get(row, "teamRank", 0)))
                score = float(_get(row, "score", 0.0))
                team_entry = (rank, team, score)
                break

    return summary, team_entry


def compare_leaderboard(
    *,
    competition: str,
    top_k: int = 5,
    team: Optional[str] = None,
    metrics_file: Optional[Path] = None,
    local_rmse: Optional[float] = None,
    print_distribution: bool = False,
) -> Dict[str, Any]:
    api = KaggleApi()
    api.authenticate()

    leaderboard = api.competition_leaderboard_view(competition)
    if isinstance(leaderboard, list):
        rows = leaderboard
    else:
        rows = leaderboard.get("submissions") or leaderboard.get("leaderboard") or []
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Unable to fetch leaderboard rows. Check competition slug and API credentials.")

    top_entries, team_entry = _summarise_leaderboard(
        rows,
        top_k=top_k,
        team_name=team,
    )

    local_score = _load_local_score(metrics_file) if metrics_file else None
    if local_score is None and local_rmse is not None:
        local_score = float(local_rmse)

    distribution = None
    if print_distribution:
        scores = [float(getattr(row, "score", getattr(row, "publicScore", 0.0))) if not isinstance(row, dict) else float(row.get("score", 0.0)) for row in rows]
        if scores:
            distribution = {
                "count": len(scores),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std": statistics.pstdev(scores),
            }

    return {
        "competition": competition,
        "top_entries": top_entries,
        "team_entry": team_entry,
        "local_score": local_score,
        "distribution": distribution,
    }


def print_leaderboard_summary(summary: Dict[str, Any]) -> None:
    competition = summary["competition"]
    top_entries = summary["top_entries"]
    team_entry = summary["team_entry"]
    local_score = summary["local_score"]
    distribution = summary["distribution"]

    print(f"Competition: {competition}")
    print("Top leaderboard scores:")
    for rank, team, score in top_entries:
        print(f"  #{rank:<3} {team:<30} {score:.5f}")

    if team_entry:
        rank, team, score = team_entry
        print(f"\nYour team ({team}) best score: {score:.5f} (rank #{rank})")

    if local_score is not None:
        best_public = top_entries[0][2]
        delta_best = local_score - best_public
        print(f"\nLocal RMSE: {local_score:.5f}")
        print(f"  Δ to public leader ({best_public:.5f}): {delta_best:+.5f}")
        if team_entry:
            delta_team = local_score - team_entry[2]
            print(f"  Δ to your best submission ({team_entry[2]:.5f}): {delta_team:+.5f}")

    if distribution:
        print("\nLeaderboard distribution:")
        print(f"  Count: {distribution['count']}")
        print(f"  Mean : {distribution['mean']:.5f}")
        print(f"  Median: {distribution['median']:.5f}")
        print(f"  Std  : {distribution['std']:.5f}")


def check_leaderboard(args: argparse.Namespace) -> int:
    summary = compare_leaderboard(
        competition=args.competition,
        top_k=args.top_k,
        team=args.team,
        metrics_file=Path(args.metrics_file) if args.metrics_file else None,
        local_rmse=args.local_rmse,
        print_distribution=args.print_distribution,
    )
    print_leaderboard_summary(summary)
    if args.team and summary["team_entry"] is None:
        print(f"\nTeam '{args.team}' not found on the public leaderboard.")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare local results against the Kaggle leaderboard."
    )
    parser.add_argument("--competition", default="csiro-biomass", help="Kaggle competition slug.")
    parser.add_argument("--team", type=str, default=None, help="Filter to your team name.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top leaderboard rows to display.")
    parser.add_argument("--metrics-file", type=str, default=None, help="Path to a JSON metrics file (run_summary.json or validation summary).")
    parser.add_argument("--local-rmse", type=float, default=None, help="Local validation score to compare (overrides metrics file).")
    parser.add_argument("--print-distribution", action="store_true", help="Print mean/median/std of public leaderboard scores.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return check_leaderboard(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
