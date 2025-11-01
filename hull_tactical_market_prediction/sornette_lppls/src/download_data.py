"""Download historical market data for LPPLS analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yfinance as yf

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download historical market data.")
    parser.add_argument(
        "--ticker",
        default="^GSPC",
        help="Ticker symbol to download (default: ^GSPC for S&P 500).",
    )
    parser.add_argument("--start-date", default="1995-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", default="2003-01-01", help="End date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/sornette_lppls/data/sp500_dotcom_bubble.csv"),
        help="Path to save the data CSV.",
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

    LOGGER.info(
        "Downloading data for %s from %s to %s",
        args.ticker,
        args.start_date,
        args.end_date,
    )

    try:
        data = yf.download(args.ticker, start=args.start_date, end=args.end_date)
    except Exception as e:
        LOGGER.error("Failed to download data: %s", e)
        return 1

    if data.empty:
        LOGGER.error("No data downloaded. Check the ticker and date range.")
        return 1

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_path)
    LOGGER.info("Saved data to %s", args.output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
