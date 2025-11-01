"""Quick exploratory script for the NFL Big Data Bowl 2026 dataset.

Run with ``python -m competitions.nfl_big_data_bowl_2026_prediction.notebooks.eda_overview``.
"""

from __future__ import annotations

from competitions.nfl_big_data_bowl_2026_prediction.src.data import (
    DEFAULT_DATA_ROOT,
    available_train_weeks,
    load_train_week_pair,
)


def main() -> None:
    print(f"Data root: {DEFAULT_DATA_ROOT.resolve()}")
    weeks = available_train_weeks()
    print(f"Discovered {len(weeks)} training weeks. First five: {weeks[:5]}")
    print(f"Last week: {weeks[-1]}")

    season, week = weeks[-1]
    inputs, outputs = load_train_week_pair(season, week)
    print("\nInput sample:")
    print(inputs.head())
    print("\nOutput sample:")
    print(outputs.head())

    input_summary = inputs[["s", "a"]].describe()
    output_summary = outputs[["x", "y"]].describe()

    print("\nSpeed / acceleration summary:")
    print(input_summary)
    print("\nTarget position summary:")
    print(output_summary)


if __name__ == "__main__":
    main()
