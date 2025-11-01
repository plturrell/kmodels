"""Compare our Kaggle submissions against the public leaderboard."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _authenticate():
    try:
        from kaggle import api  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import Kaggle. Install it via `pip install kaggle`."
        ) from exc

    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to authenticate with Kaggle. Ensure ~/.kaggle/kaggle.json exists."
        ) from exc
    return api


def _parse_score(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lstrip("=")
        if not text:
            return None
        text = text.replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _default_username() -> Optional[str]:
    config_path = Path.home() / ".kaggle" / "kaggle.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text())
        return payload.get("username")
    except Exception:
        return None


def _get_submission_attr(submission, name: str):
    value = getattr(submission, name, None)
    if value is None and isinstance(submission, dict):
        value = submission.get(name)
    return value


@dataclass
class LeaderboardSnapshot:
    top_entries: List[Dict]
    my_entry: Optional[Dict]
    my_best_submission: Optional[Dict]


def fetch_leaderboard_snapshot(
    competition: str,
    username: Optional[str],
    page_size: int = 200,
) -> LeaderboardSnapshot:
    api = _authenticate()
    raw_entries = list(api.competition_leaderboard_view(competition))
    entries: List[Dict] = []
    for item in raw_entries:
        if hasattr(item, 'to_dict'):
            entries.append(item.to_dict())
        elif isinstance(item, dict):
            entries.append(item)
        else:
            entries.append(dict(getattr(item, '__dict__', {})))

    my_entry: Optional[Dict] = None
    username = (username or _default_username() or '').lower()
    if username:
        for entry in entries:
            team_name = (entry.get('teamName') or entry.get('team_name') or '').lower()
            if team_name == username:
                my_entry = entry
                break

    submissions = api.competition_submissions(competition)
    best_submission: Optional[Dict] = None
    for submission in submissions:
        score = _parse_score(_get_submission_attr(submission, "publicScore"))
        if score is None:
            continue
        if best_submission is None or score > best_submission['score']:
            best_submission = {
                'ref': _get_submission_attr(submission, 'ref'),
                'description': _get_submission_attr(submission, 'description'),
                'date': _get_submission_attr(submission, 'date'),
                'score': score,
            }

    return LeaderboardSnapshot(
        top_entries=entries,
        my_entry=my_entry,
        my_best_submission=best_submission,
    )


def _format_entry(entry: Dict) -> str:
    score = _parse_score(entry.get('score')) or _parse_score(
        entry.get('scoreDisplay')
    )
    team = entry.get('teamName') or entry.get('team_name')
    rank = entry.get('rank')
    return f"#{rank} {team} — {score}"

def _format_entry(entry: Dict, fallback_rank: Optional[int]) -> str:
    score = _parse_score(entry.get("score")) or _parse_score(
        entry.get("scoreDisplay")
    )
    team = entry.get("teamName") or entry.get("team_name")
    rank = entry.get("rank") or fallback_rank
    return f"#{rank} {team} — {score}"


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare your Kaggle score against the competition leaderboard."
    )
    parser.add_argument(
        "--competition",
        default="recodai-luc-scientific-image-forgery-detection",
        help="Competition slug.",
    )
    parser.add_argument(
        "--username",
        help="Kaggle username (defaults to ~/.kaggle/kaggle.json).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top leaderboard entries to display (default: 5).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    snapshot = fetch_leaderboard_snapshot(
        competition=args.competition,
        username=args.username,
        page_size=max(args.top, 50),
    )

    top_display = snapshot.top_entries[: args.top]
    print(f"Fetched {len(snapshot.top_entries)} leaderboard rows (top {args.top} shown)")
    if not top_display:
        print("No leaderboard entries returned.")
        return 0

    for idx, entry in enumerate(top_display, start=1):
        print("  " + _format_entry(entry, idx))

    top_score = _parse_score(top_display[0].get("score")) or _parse_score(
        top_display[0].get("scoreDisplay")
    )

    if snapshot.my_entry is not None:
        my_score = _parse_score(snapshot.my_entry.get("score")) or _parse_score(
            snapshot.my_entry.get("scoreDisplay")
        )
        print()
        print("Your leaderboard entry:")
        print("  " + _format_entry(snapshot.my_entry, None))
        if top_score is not None and my_score is not None:
            gap = top_score - my_score
            print(f"  Gap to #1: {gap:.6f} (assuming higher is better)")
    else:
        print()
        print("You do not appear in the fetched leaderboard page.")

    if snapshot.my_best_submission:
        best = snapshot.my_best_submission
        print()
        print("Best public submission:")
        print(
            f"  Ref: {best['ref']} | Score: {best['score']:.6f} | Date: {best.get('date')}"
        )
        if best.get("description"):
            print(f"  Notes: {best['description']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
