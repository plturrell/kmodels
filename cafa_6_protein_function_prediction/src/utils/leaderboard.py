"""Fetch and summarise the Kaggle leaderboard for CAFA 6."""

from __future__ import annotations

import argparse
import io
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from ..data import DEFAULT_COMPETITION

LOGGER = logging.getLogger(__name__)


def authenticate():
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
            "Failed to authenticate with Kaggle. Ensure ~/.kaggle/kaggle.json exists and you have joined the competition."
        ) from exc
    return api


@dataclass
class LeaderboardEntry:
    rank: int
    team_name: str
    score: float
    submissions: int
    last_submission: str
    members: List[str]


def _parse_entries(rows: Iterable[str]) -> List[LeaderboardEntry]:
    df = pd.read_csv(io.StringIO("".join(rows)), encoding="utf-8-sig")
    entries: List[LeaderboardEntry] = []
    for _, row in df.iterrows():
        members = [part.strip() for part in str(row["TeamMemberUserNames"]).split(",") if part and part.strip()]
        entries.append(
            LeaderboardEntry(
                rank=int(row["Rank"]),
                team_name=str(row["TeamName"]),
                score=float(row["Score"]),
                submissions=int(row["SubmissionCount"]),
                last_submission=str(row["LastSubmissionDate"]),
                members=members,
            )
        )
    return entries


def fetch_leaderboard(competition: str = DEFAULT_COMPETITION) -> List[LeaderboardEntry]:
    api = authenticate()
    with tempfile.TemporaryDirectory() as tmp_dir:
        api.competition_leaderboard_download(competition, path=tmp_dir)
        zip_path = next(Path(tmp_dir).glob("*.zip"))
        with zipfile.ZipFile(zip_path, "r") as archive:
            csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
            with archive.open(csv_name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8-sig")
                rows = text.readlines()
    return _parse_entries(rows)


def find_user(entries: Sequence[LeaderboardEntry], username: str) -> Optional[LeaderboardEntry]:
    username = username.lower()
    for entry in entries:
        if any(member.lower() == username for member in entry.members):
            return entry
    return None


def read_local_score(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("lb_score", "public_score", "f1", "map", "score"):
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    return None


def format_summary(
    entries: Sequence[LeaderboardEntry],
    *,
    top_n: int,
    username: Optional[str],
    local_score: Optional[float],
) -> str:
    lines: List[str] = []
    lines.append("Top leaderboard snapshot:")
    header = f"{'Rank':>4}  {'Team':<30}  {'Score':>10}  Submissions"
    lines.append(header)
    lines.append("-" * len(header))
    for entry in entries[:top_n]:
        lines.append(f"{entry.rank:>4}  {entry.team_name:<30.30}  {entry.score:>10.4f}  {entry.submissions:>3}")

    if username:
        located = find_user(entries, username)
        lines.append("")
        if located:
            lines.append(
                f"Your team '{located.team_name}' is #{located.rank} with score {located.score:.4f} (last submit {located.last_submission})."
            )
        else:
            lines.append(f"No leaderboard entry found for username '{username}'.")

    if local_score is not None:
        leader = entries[0].score if entries else None
        lines.append("")
        lines.append(f"Local validation score: {local_score:.4f}")
        if leader is not None:
            delta = local_score - leader
            lines.append(f"Gap to leader: {delta:+.4f}")
    return "\n".join(lines)


def detect_default_username() -> Optional[str]:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return None
    try:
        payload = json.loads(kaggle_json.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload.get("username")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect public leaderboard standings.")
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Kaggle competition slug (default: cafa-6-protein-function-prediction).",
    )
    parser.add_argument(
        "--username",
        default=detect_default_username(),
        help="Kaggle username to highlight (default: derived from ~/.kaggle/kaggle.json).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of leaderboard rows to display (default: 10).",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Optional JSON file with a local validation score to display.",
    )
    parser.add_argument(
        "-v",
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

    summary = format_summary(
        entries,
        top_n=args.top,
        username=args.username,
        local_score=read_local_score(args.metrics_file) if args.metrics_file else None,
    )
    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
