"""Utilities to fetch the NFL Big Data Bowl 2026 Prediction dataset locally.

This mirrors the Kaggle CLI so you can grab the competition bundle straight
into this project structure.

Example:

    python -m competitions.nfl_big_data_bowl_2026_prediction.src.data.download \
        --download-dir data/nfl_big_data_bowl_2026_prediction/raw --extract

You need the Kaggle package installed (``pip install kaggle``) and a personal
API token stored at ``~/.kaggle/kaggle.json`` before the script can
authenticate.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional


LOGGER = logging.getLogger(__name__)


def _authenticate():
    """Return an authenticated Kaggle API client or raise a helpful error."""
    try:
        from kaggle import api  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import Kaggle. Install it via `pip install kaggle` and "
            "ensure your API credentials are stored in ~/.kaggle/kaggle.json."
        ) from exc

    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to authenticate with Kaggle. Confirm that your API token is "
            "present at ~/.kaggle/kaggle.json and that it has correct permissions."
        ) from exc
    return api


def _extract_zip(archive: Path, extract_dir: Path) -> None:
    """Extract a zip archive into the given directory."""
    LOGGER.info("Extracting %s -> %s", archive.name, extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_dir)


def download_competition(
    competition: str,
    download_dir: Path,
    files: Optional[Iterable[str]] = None,
    extract: bool = False,
) -> None:
    """Download competition files to ``download_dir``."""
    api = _authenticate()
    download_dir.mkdir(parents=True, exist_ok=True)

    if files:
        file_list = list(files)
        LOGGER.info("Downloading %d files from %s", len(file_list), competition)
        for filename in file_list:
            LOGGER.info("Fetching %s", filename)
            api.competition_download_file(
                competition, filename, path=str(download_dir)
            )
    else:
        LOGGER.info("Downloading all files from %s", competition)
        api.competition_download_files(competition, path=str(download_dir))

    if extract:
        archives = list(download_dir.glob("*.zip"))
        if not archives:
            LOGGER.warning("No zip archives found to extract.")
        for archive in archives:
            target_dir = download_dir / archive.stem
            _extract_zip(archive, target_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download data for a Kaggle competition."
    )
    parser.add_argument(
        "--competition",
        default="nfl-big-data-bowl-2026-prediction",
        help="Competition slug (default: nfl-big-data-bowl-2026-prediction).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/nfl_big_data_bowl_2026_prediction/raw"),
        help=(
            "Directory to save downloads "
            "(default: data/nfl_big_data_bowl_2026_prediction/raw)."
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
