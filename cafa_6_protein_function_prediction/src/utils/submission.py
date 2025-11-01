"""Kaggle submission helper for the CAFA 6 competition."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

from ..data import DEFAULT_COMPETITION
from .leaderboard import authenticate

LOGGER = logging.getLogger(__name__)


def submit_file(
    submission_path: Path,
    *,
    competition: str = DEFAULT_COMPETITION,
    message: str = "Auto submission",
    quiet: bool = False,
) -> None:
    api = authenticate()
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_path}")
    api.competition_submit(str(submission_path), message, competition, quiet=quiet)


def build_submission_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a prediction file to Kaggle.")
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the submission CSV to upload.",
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help="Kaggle competition slug (default: cafa-6-protein-function-prediction).",
    )
    parser.add_argument(
        "--message",
        default="CAFA 6 submission",
        help="Submission message shown on Kaggle (default: 'CAFA 6 submission').",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass the quiet flag to the Kaggle API submit call.",
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
    parser = build_submission_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    try:
        submit_file(
            submission_path=args.file,
            competition=args.competition,
            message=args.message,
            quiet=args.quiet,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("%s", exc)
        return 1

    LOGGER.info("Submitted %s to %s", args.file, args.competition)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
