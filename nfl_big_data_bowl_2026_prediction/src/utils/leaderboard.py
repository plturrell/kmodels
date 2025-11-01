"""Fetch and summarise the Kaggle leaderboard for the competition."""

from __future__ import annotations

import argparse
import io
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _authenticate():
    """Return an authenticated Kaggle API client."""
    try:
        from kaggle import api  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Kaggle package not found. Install it via `pip install kaggle`."
        ) from exc

    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to authenticate with Kaggle. Ensure ~/.kaggle/kaggle.json exists and is valid."
        ) from exc
    return api


@dataclass
class LeaderboardEntry:
    rank: int
    team_id: int
    team_name: str
    score: float
    submission_count: int
    last_submission: str
    members: List[str]


def _parse_leaderboard_csv(data: Iterable[str]) -> List[LeaderboardEntry]:
    df = pd.read_csv(io.StringIO("".join(data)), encoding="utf-8-sig")
    entries: List[LeaderboardEntry] = []
    for _, row in df.iterrows():
        members = [name.strip() for name in str(row["TeamMemberUserNames"]).split(",") if name]
        entries.append(
            LeaderboardEntry(
                rank=int(row["Rank"]),
                team_id=int(row["TeamId"]),
                team_name=str(row["TeamName"]),
                score=float(row["Score"]),
                submission_count=int(row["SubmissionCount"]),
                last_submission=str(row["LastSubmissionDate"]),
                members=members,
            )
        )
    return entries


def fetch_leaderboard(competition: str) -> List[LeaderboardEntry]:
    api = _authenticate()
    with tempfile.TemporaryDirectory() as tmp_dir:
        api.competition_leaderboard_download(competition, path=tmp_dir)
        zip_path = next(Path(tmp_dir).glob("*.zip"))
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = next(name for name in zf.namelist() if name.endswith(".csv"))
            with zf.open(csv_name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8-sig")
                data = text.readlines()
    entries = _parse_leaderboard_csv(data)
    return entries


def locate_user(entries: List[LeaderboardEntry], username: str) -> Optional[LeaderboardEntry]:
    username = username.lower()
    for entry in entries:
        if any(member.lower() == username for member in entry.members):
            return entry
    return None


def read_local_metrics(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    for key in ("lb_score", "public_score", "mae_overall", "score"):
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    return None


def summarise(entries: List[LeaderboardEntry], top_n: int, user: Optional[str], local_score: Optional[float]) -> str:
    lines = []
    lines.append("Top leaderboard snapshot:")
    header = f"{'Rank':>4}  {'Team':<30}  {'Score':>8}  Submissions"
    lines.append(header)
    lines.append("-" * len(header))
    for entry in entries[:top_n]:
        lines.append(f"{entry.rank:>4}  {entry.team_name:<30.30}  {entry.score:>8.3f}  {entry.submission_count:>3}")

    if user:
        found = locate_user(entries, user)
        if found:
            lines.append("")
            lines.append(
                f"Your team '{found.team_name}' is currently #{found.rank} with score {found.score:.3f} (last submit {found.last_submission})."
            )
        else:
            lines.append("")
            lines.append(f"No leaderboard entry found for username '{user}'.")

    if local_score is not None:
        best = entries[0].score if entries else None
        lines.append("")
        lines.append(f"Local validation score: {local_score:.3f}")
        if best is not None:
            delta = local_score - best
            lines.append(f"Difference vs current leader: {delta:+.3f}")
    return "\n".join(lines)


def detect_default_username() -> Optional[str]:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        try:
            payload = json.loads(kaggle_json.read_text(encoding="utf-8"))
            return payload.get("username")
        except Exception:  # noqa: BLE001
            return None
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check competition leaderboard standings.")
    parser.add_argument(
        "--competition",
        default="nfl-big-data-bowl-2026-prediction",
        help="Kaggle competition slug.",
    )
    parser.add_argument(
        "--username",
        default=detect_default_username(),
        help="Kaggle username to locate in the leaderboard (default: read from kaggle.json).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top leaderboard entries to display (default: 10).",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Optional local metrics JSON file to show alongside leaderboard scores.",
    )
    parser.add_argument(
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    try:
        entries = fetch_leaderboard(args.competition)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("%s", exc)
        return 1

    local_score = read_local_metrics(args.metrics_file)
    summary = summarise(entries, top_n=args.top, user=args.username, local_score=local_score)
    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
