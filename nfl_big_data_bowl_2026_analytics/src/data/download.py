"""Utilities to fetch the NFL Big Data Bowl 2026 Analytics dataset locally.

This mirrors the Kaggle CLI so you can grab the competition bundle straight
into this project structure.

Example:

    python -m competitions.nfl_big_data_bowl_2026_analytics.src.data.download \
        --download-dir competitions/nfl_big_data_bowl_2026_analytics/data/raw --extract

You need the Kaggle package installed (``pip install kaggle``) and a personal
API token stored at ``~/.kaggle/kaggle.json`` before the script can
authenticate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

from competitions.nfl_big_data_bowl_2026_prediction.src.data import download as _base


LOGGER = logging.getLogger(__name__)

DEFAULT_COMPETITION = "nfl-big-data-bowl-2026-analytics"
DEFAULT_DOWNLOAD_DIR = Path("competitions/nfl_big_data_bowl_2026_analytics/data/raw")

# Reuse the shared Kaggle download implementation.
download_competition = _base.download_competition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download data for the NFL Big Data Bowl 2026 Analytics Kaggle competition."
        )
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Competition slug (default: {DEFAULT_COMPETITION}).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=(
            "Directory to save downloads "
            f"(default: {DEFAULT_DOWNLOAD_DIR})."
        ),
    )
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="Specific file(s) to retrieve. Repeat for multiple entries.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract any downloaded zip archives after download completes.",
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    try:
        download_competition(
            competition=args.competition,
            download_dir=args.download_dir,
            files=args.files,
            extract=args.extract,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
