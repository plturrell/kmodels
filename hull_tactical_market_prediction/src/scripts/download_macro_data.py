"""CLI to download macroeconomic data from the FRED API."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from fredapi import Fred

LOGGER = logging.getLogger(__name__)

# Key macroeconomic series from FRED
SERIES_MAP = {
    "DFF": "fed_funds_rate",
    "VIXCLS": "vix",
    "T10Y2Y": "t10y2y_spread",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment_rate",
    "GDP": "gdp",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download macroeconomic data from FRED.")
    parser.add_argument("--api-key", required=True, help="FRED API key.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/data/processed/macro_data.csv"),
        help="Path to save the processed macro data CSV.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    try:
        fred = Fred(api_key=args.api_key)
    except ValueError as e:
        LOGGER.error("Failed to initialize FRED API: %s", e)
        return 1

    all_series = []
    for series_id, name in SERIES_MAP.items():
        LOGGER.info("Downloading series: %s (%s)", name, series_id)
        try:
            series = fred.get_series(series_id).rename(name)
            all_series.append(series)
        except Exception as e:
            LOGGER.error("Failed to download series %s: %s", series_id, e)
            return 1

    df = pd.concat(all_series, axis=1)

    # Forward-fill to create daily series from lower-frequency data (e.g., monthly CPI)
    df = df.resample("D").ffill()

    # Ensure the output directory exists
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.output_path)
    LOGGER.info("Saved macro data to %s", args.output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
