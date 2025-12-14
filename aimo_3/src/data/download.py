"""Download competition data from Kaggle."""

import argparse
import os
from pathlib import Path
from typing import Optional

from kaggle.api.kaggle_api_extended import KaggleApi


_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _safe_project_path(path: Path) -> Path:
    """
    Resolve a user-provided path and ensure it stays within the project root.
    """
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = (_PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"Refusing to write outside project root: {candidate}")
    return candidate


def download_data(
    competition: str = "aimo3",
    download_dir: Optional[Path] = None,
    extract: bool = True,
    file: Optional[str] = None,
) -> Path:
    """
    Download competition data from Kaggle.

    Args:
        competition: Competition slug (default: "aimo3")
        download_dir: Directory to download to (default: data/raw/)
        extract: Whether to extract zip files
        file: Specific file to download (optional)

    Returns:
        Path to downloaded data directory
    """
    if download_dir is None:
        download_dir = _PROJECT_ROOT / "data" / "raw"

    download_dir = _safe_project_path(Path(download_dir))
    download_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading data for competition: {competition}")
    print(f"Target directory: {download_dir}")

    if file:
        api.competition_download_file(
            competition=competition,
            file_name=file,
            path=str(download_dir),
        )
    else:
        api.competition_download_files(
            competition=competition,
            path=str(download_dir),
            unzip=extract,
        )

    print(f"Download complete. Files in: {download_dir}")
    return download_dir


def main():
    """CLI entry point for data download."""
    parser = argparse.ArgumentParser(description="Download AIMO 3 competition data")
    parser.add_argument(
        "--competition",
        default="aimo3",
        help="Competition slug (default: aimo3)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Directory to download to (default: data/raw/)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        default=True,
        help="Extract zip files (default: True)",
    )
    parser.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        help="Don't extract zip files",
    )
    parser.add_argument(
        "--file",
        help="Specific file to download (optional)",
    )

    args = parser.parse_args()
    download_data(
        competition=args.competition,
        download_dir=args.download_dir,
        extract=args.extract,
        file=args.file,
    )


if __name__ == "__main__":
    main()

