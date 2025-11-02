#!/usr/bin/env python3
"""Aggregate model outputs into a Kaggle-ready submission.

This helper mirrors the workflow described in the README: after you have
trained one or more runs with

    python -m competitions.csiro_biomass.src.train ...

you can point the script at the resulting run directory (or individual
checkpoints) and it will collect the corresponding `submission.csv` files.

For multiple runs/folds, the script performs a simple average using the
ensemble utilities that ship with the starter kit.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from competitions.csiro_biomass.src.utils.ensemble import simple_average_ensemble

LOGGER = logging.getLogger(__name__)


def _resolve_run_dir(path: Path) -> Path:
    """Return the directory that should contain submission artifacts."""
    path = path.expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def _candidate_submission_files(run_dir: Path) -> List[Path]:
    """Return candidate submission files for a single run directory."""
    candidates = [
        run_dir / "submission.csv",
        run_dir / "latest_submission.csv",
    ]
    if run_dir.name.startswith("fold-"):
        # Nested CV fold – allow the file to live one directory deeper.
        nested = sorted(run_dir.glob("**/submission.csv"))
        candidates.extend(nested)
    return [path for path in candidates if path.exists()]


def _collect_submission_files(directories: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for run_dir in directories:
        run_dir = _resolve_run_dir(run_dir)
        candidates = _candidate_submission_files(run_dir)
        if not candidates:
            raise FileNotFoundError(
                f"No submission CSV discovered in '{run_dir}'. "
                "Ensure the training run was executed with --test-csv / --sample-submission."
            )
        files.append(candidates[0])
    return files


def _submission_from_cv_dir(cv_dir: Path) -> List[Path]:
    """Collect submission files from cross-validation folds."""
    cv_dir = cv_dir.expanduser().resolve()
    if not cv_dir.exists():
        raise FileNotFoundError(f"Cross-validation directory does not exist: {cv_dir}")
    fold_dirs = sorted(path for path in cv_dir.glob("fold-*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {cv_dir}")
    return _collect_submission_files(fold_dirs)


def copy_single_submission(submission_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(submission_path, output_path)
    return output_path


def average_submissions(
    submission_files: Sequence[Path],
    output_path: Path,
    weights: Optional[Sequence[float]] = None,
) -> Path:
    LOGGER.info("Ensembling %d submissions", len(submission_files))
    return simple_average_ensemble(submission_files, output_path, weights=weights)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Kaggle submission from trained runs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path, help="Path to a single best_model checkpoint (parent run dir is used).")
    group.add_argument("--checkpoints", type=Path, nargs="+", help="Paths to multiple checkpoints to ensemble.")
    group.add_argument("--run-dir", type=Path, help="Training run directory that already contains submission.csv.")
    group.add_argument("--run-dirs", type=Path, nargs="+", help="Multiple run directories to ensemble.")
    group.add_argument("--cv-dir", type=Path, help="Cross-validation output directory (averages all fold submissions).")

    parser.add_argument("--output-path", type=Path, default=Path("submission.csv"), help="Destination for the final submission.")
    parser.add_argument("--weights", type=float, nargs="+", help="Optional weights when averaging multiple submissions.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    if args.checkpoint:
        submission_files = _collect_submission_files([args.checkpoint])
    elif args.checkpoints:
        submission_files = _collect_submission_files(args.checkpoints)
    elif args.run_dir:
        submission_files = _collect_submission_files([args.run_dir])
    elif args.run_dirs:
        submission_files = _collect_submission_files(args.run_dirs)
    elif args.cv_dir:
        submission_files = _submission_from_cv_dir(args.cv_dir)
    else:
        raise RuntimeError("No valid input provided.")

    output_path = args.output_path.expanduser().resolve()
    submission_files = [path.expanduser().resolve() for path in submission_files]

    LOGGER.info("Collected submission files: %s", ", ".join(str(path) for path in submission_files))

    if len(submission_files) == 1:
        copy_single_submission(submission_files[0], output_path)
        LOGGER.info("Copied submission to %s", output_path)
    else:
        if args.weights and len(args.weights) != len(submission_files):
            raise ValueError("Number of weights must match the number of submissions.")
        average_submissions(submission_files, output_path, weights=args.weights)
        LOGGER.info("Averaged submission written to %s", output_path)

    # Provide a quick sanity check by printing the number of rows / NaNs.
    df = pd.read_csv(output_path)
    LOGGER.info("Final submission shape: %s, missing values: %d", df.shape, int(df.isnull().sum().sum()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

