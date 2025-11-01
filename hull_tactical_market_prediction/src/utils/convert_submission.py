"""Convert CSV submissions into the parquet format required by Kaggle."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a submission CSV into parquet format."
    )
    default_input = Path(
        "competitions/hull_tactical_market_prediction/outputs/baseline/latest_submission.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Submission CSV to convert (default: {default_input}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination parquet path. Defaults to replacing the input suffix with .parquet.",
    )
    parser.add_argument(
        "--compression",
        choices=("snappy", "gzip", "brotli", "lz4", "zstd", "none"),
        default="snappy",
        help="Parquet compression codec (default: snappy).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity.",
    )
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def convert_submission(
    input_csv: Path, output_parquet: Path, compression: Optional[str]
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Submission CSV not found: {input_csv}")
    LOGGER.info("Loading submission from %s", input_csv)
    df = pd.read_csv(input_csv)
    parquet_kwargs: dict[str, object] = {"index": False}
    if compression and compression != "none":
        parquet_kwargs["compression"] = compression
    LOGGER.info("Writing %d rows to %s", len(df), output_parquet)
    try:
        df.to_parquet(output_parquet, **parquet_kwargs)
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Parquet support requires `pyarrow` or `fastparquet`. Install via `pip install pyarrow`."
        ) from exc


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.verbose)

    output = args.output or args.input.with_suffix(".parquet")
    compression = None if args.compression == "none" else args.compression

    try:
        convert_submission(args.input, output, compression)
    except Exception as exc:  # pragma: no cover - CLI wrapper
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
