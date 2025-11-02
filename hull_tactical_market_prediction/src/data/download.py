"""Helpers to pull the Hull Tactical Market Prediction dataset locally.

This mirrors the Kaggle CLI so you can fetch the competition bundle without
leaving your own environment.

Example:

    python -m competitions.hull_tactical_market_prediction.src.data.download \\
        --download-dir competitions/hull_tactical_market_prediction/data/raw --extract

You need the Kaggle package installed (`pip install kaggle`) and a personal API
token stored at `~/.kaggle/kaggle.json` before the script can authenticate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from datetime import datetime
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


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _generate_manifest(
    download_dir: Path,
    competition: str,
    output_path: Optional[Path] = None,
) -> dict:
    """Generate a dataset manifest with file hashes and metadata.
    
    Args:
        download_dir: Directory containing downloaded/extracted files
        competition: Competition slug
        output_path: Optional path to save manifest JSON. If None, uses outputs/dataset_manifest.json
        
    Returns:
        Dictionary containing manifest data
    """
    manifest = {
        "competition": competition,
        "download_timestamp": datetime.utcnow().isoformat(),
        "files": [],
    }
    
    # Collect all files (excluding hidden files and the manifest itself)
    files = []
    for item in download_dir.rglob("*"):
        if item.is_file() and not item.name.startswith("."):
            # Skip manifest files
            if item.name == "dataset_manifest.json":
                continue
            files.append(item)
    
    for file_path in sorted(files):
        try:
            file_size = file_path.stat().st_size
            file_hash = _compute_file_hash(file_path)
            relative_path = file_path.relative_to(download_dir)
            manifest["files"].append({
                "path": str(relative_path),
                "size_bytes": file_size,
                "sha256": file_hash,
            })
        except (OSError, IOError) as exc:
            LOGGER.warning("Failed to process file %s: %s", file_path, exc)
    
    manifest["file_count"] = len(manifest["files"])
    
    if output_path is None:
        # Default to outputs/dataset_manifest.json relative to download_dir
        outputs_dir = download_dir.parent.parent / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / "dataset_manifest.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    LOGGER.info("Generated dataset manifest with %d files at %s", manifest["file_count"], output_path)
    return manifest


def download_competition(
    competition: str,
    download_dir: Path,
    files: Optional[Iterable[str]] = None,
    extract: bool = False,
    generate_manifest: bool = True,
) -> None:
    """Download competition files to ``download_dir``.
    
    Args:
        competition: Competition slug
        download_dir: Directory to save downloads
        files: Optional list of specific files to download
        extract: Whether to extract zip archives
        generate_manifest: Whether to generate dataset manifest with checksums
    """
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

    if generate_manifest:
        _generate_manifest(download_dir, competition)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download data for a Kaggle competition."
    )
    parser.add_argument(
        "--competition",
        default="hull-tactical-market-prediction",
        help="Competition slug (default: hull-tactical-market-prediction).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/data/raw"),
        help=(
            "Directory to save downloads "
            "(default: competitions/hull_tactical_market_prediction/data/raw)."
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
        "--no-manifest",
        action="store_true",
        help="Skip generating dataset manifest with checksums.",
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
            generate_manifest=not args.no_manifest,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
