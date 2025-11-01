"""Analyze the Hull Tactical dataset for micro-bubbles using the LPPLS model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lppls import fit_lppls, lppls_model

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze micro-bubbles in a time series.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Path to the training CSV containing returns data.",
    )
    parser.add_argument(
        "--target-column",
        default="forward_returns",
        help="The column containing the returns to construct the price series from.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity."
    )
    return parser


def construct_price_series(returns: pd.Series, initial_price: float = 100.0) -> pd.Series:
    """Construct a price series from a series of returns."""
    return (1 + returns).cumprod() * initial_price


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    LOGGER.info("Loading data from %s", args.train_csv)
    try:
        df = pd.read_csv(args.train_csv)
    except FileNotFoundError:
        LOGGER.error("Training data not found at %s", args.train_csv)
        return 1

    if args.target_column not in df.columns:
        LOGGER.error("Target column '%s' not found in the data.", args.target_column)
        return 1

    # Create a date column from the integer date_id
    # Assuming date_id starts from a base date. This can be adjusted.
    base_date = pd.to_datetime('2000-01-01')
    df['date'] = df['date_id'].apply(lambda x: base_date + pd.Timedelta(days=x))

    df = df.sort_values("date").reset_index(drop=True)
    price_series = construct_price_series(df[args.target_column])

    LOGGER.info("Constructed price series with %d data points.", len(price_series))

    # --- Bubble Hunter Logic ---
    window_size = 90  # Days to look for a bubble
    growth_threshold = 0.15  # 15% growth to be considered a bubble

    found_bubbles = 0
    for i in range(len(price_series) - window_size):
        window = price_series.iloc[i : i + window_size]
        start_price = window.iloc[0]
        end_price = window.iloc[-1]

        # Check for significant growth
        if (end_price - start_price) / start_price > growth_threshold:
            LOGGER.info("Potential bubble detected starting at index %d", i)
            found_bubbles += 1

            # Attempt to fit the LPPLS model to this window
            fit_result = fit_lppls(window.to_numpy())

            if fit_result and fit_result["params"]:
                params = fit_result["params"]
                tc = params["tc"]
                critical_index = i + int(tc)
                if 0 <= critical_index < len(df):
                    critical_date = df['date'].iloc[critical_index]
                    LOGGER.info(
                        "  LPPLS fit successful! Predicted critical date: %s",
                        critical_date.strftime("%Y-%m-%d"),
                    )
                    print(f"  Fit parameters: m={params['m']:.2f}, omega={params['omega']:.2f}")

                    # --- Visualization ---
                    plt.figure(figsize=(10, 6))
                    t_window = np.arange(len(window))
                    plt.plot(t_window, window.to_numpy(), label="Price")
                    
                    # Generate the fitted curve
                    fitted_curve = lppls_model(t_window, **params)
                    plt.plot(t_window, np.exp(fitted_curve), label="LPPLS Fit", color="red")
                    
                    # Mark the critical time
                    plt.axvline(x=params['tc'], color='green', linestyle='--', label=f"Critical Time (tc={params['tc']:.2f})")
                    
                    plt.title(f"LPPLS Fit for Bubble starting at index {i}")
                    plt.xlabel("Time (days from start of window)")
                    plt.ylabel("Price")
                    plt.legend()
                    plt.grid(True)
                    
                    plot_path = Path("competitions/hull_tactical_market_prediction/sornette_lppls/plots")
                    plot_path.mkdir(exist_ok=True)
                    plt.savefig(plot_path / f"bubble_{i}.png")
                    plt.close()
                    LOGGER.info("  Saved plot to %s", plot_path / f"bubble_{i}.png")

                else:
                    LOGGER.warning("  Fit successful, but critical time is out of bounds.")
            else:
                LOGGER.info("  Could not fit LPPLS model to this window.")

    if found_bubbles == 0:
        LOGGER.info("No potential micro-bubbles were detected with the current settings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
