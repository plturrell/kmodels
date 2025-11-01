"""Compare local scores with Kaggle leaderboard standings."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a score against Kaggle leaderboard.")
    parser.add_argument(
        "--competition",
        default="hull-tactical-market-prediction",
        help="Competition slug (default: hull-tactical-market-prediction).",
    )
    parser.add_argument(
        "--score",
        type=float,
        help="Local validation or public score to compare. If omitted, only leaderboard stats are shown.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=10,
        help="Number of top leaderboard rows to retrieve (default: 10).",
    )
    return parser


@dataclass
class LeaderboardEntry:
    team_id: int
    team_name: str
    score: float
    submission_date: str


def fetch_leaderboard(competition: str, max_entries: int) -> list[LeaderboardEntry]:
    api = KaggleApi()
    api.authenticate()
    entries = api.competition_leaderboard_view(competition)
    simplified: list[LeaderboardEntry] = []
    for entry in entries[:max_entries]:
        simplified.append(
            LeaderboardEntry(
                team_id=entry.team_id,
                team_name=entry.team_name,
                score=float(entry.score),
                submission_date=str(entry.submission_date),
            )
        )
    return simplified


def compare_score(local_score: float, leaderboard: list[LeaderboardEntry]) -> dict:
    delta_best = local_score - leaderboard[0].score
    rank = None
    for idx, row in enumerate(leaderboard, start=1):
        if local_score >= row.score:
            rank = idx
            break
    info = {
        "local_score": local_score,
        "leader_top": leaderboard[0].score,
        "difference_from_top": delta_best,
        "estimated_rank_within_chunk": rank,
    }
    return info


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    leaderboard = fetch_leaderboard(args.competition, args.max_entries)
    output = {"leaderboard": [asdict(entry) for entry in leaderboard]}
    if args.score is not None and leaderboard:
        output["comparison"] = compare_score(args.score, leaderboard)

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
