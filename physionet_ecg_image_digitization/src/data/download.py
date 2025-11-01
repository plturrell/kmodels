"""Utilities to fetch the PhysioNet ECG Image Digitization dataset locally.

Usage example:

    python -m competitions.physionet_ecg_image_digitization.src.data.download \
        --extract

You need Kaggle credentials stored at ``~/.kaggle/kaggle.json`` and the Kaggle
CLI package installed (``pip install kaggle``) before running this module.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional

from . import DEFAULT_COMPETITION, raw_data_dir

LOGGER = logging.getLogger(__name__)


def _authenticate():
    """Return an authenticated Kaggle API client or raise an informative error."""
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
            "Failed to authenticate with Kaggle. Confirm ~/.kaggle/kaggle.json "
            "exists, has permissions 0600, and you have joined the competition."
        ) from exc
    return api


def _extract_zip(archive: Path, extract_dir: Path) -> None:
    """Extract a zip archive into the given directory."""
    LOGGER.info("Extracting %s -> %s", archive.name, extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_dir)


def download_competition(
    competition: str = DEFAULT_COMPETITION,
    download_dir: Optional[Path] = None,
    files: Optional[Iterable[str]] = None,
    extract: bool = False,
) -> None:
    """Download competition files to ``download_dir``."""
    api = _authenticate()
    target_dir = download_dir or raw_data_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    if files:
        file_list = list(files)
        LOGGER.info("Downloading %d files from %s", len(file_list), competition)
        for filename in file_list:
            LOGGER.info("Fetching %s", filename)
            api.competition_download_file(competition, filename, path=str(target_dir))
    else:
        LOGGER.info("Downloading all files from %s", competition)
        api.competition_download_files(competition, path=str(target_dir))

    if extract:
        archives = list(target_dir.glob("*.zip"))
        if not archives:
            LOGGER.warning("No zip archives found to extract.")
        for archive in archives:
            destination = target_dir / archive.stem
            _extract_zip(archive, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download data for the PhysioNet ECG Image Digitization competition."
    )
    parser.add_argument(
        "--competition",
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug (default: {DEFAULT_COMPETITION}).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=raw_data_dir(),
        help=(
            "Directory to save downloads "
            f"(default: {raw_data_dir().as_posix()})."
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
