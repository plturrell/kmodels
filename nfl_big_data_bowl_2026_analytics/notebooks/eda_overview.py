"""Quick exploratory script for the NFL Big Data Bowl 2026 Analytics dataset.

Run with ``python -m competitions.nfl_big_data_bowl_2026_analytics.notebooks.eda_overview``.
"""

from __future__ import annotations

from competitions.nfl_big_data_bowl_2026_analytics.src.data import (
    available_train_weeks,
    load_supplementary,
    load_train_week_pair,
    resolve_bundle_root,
)


def main() -> None:
    bundle_root = resolve_bundle_root()
    print(f"Bundle root: {bundle_root.resolve()}")

    weeks = available_train_weeks()
    print(f"Discovered {len(weeks)} training weeks. First five: {weeks[:5]}")

    season, week = weeks[0]
    inputs, outputs = load_train_week_pair(season, week)
    print(f"\nSample week: season {season}, week {week}")
    print(f"Inputs shape: {inputs.shape}; Outputs shape: {outputs.shape}")
    print("\nInput preview:")
    print(inputs.head())
    print("\nOutput preview:")
    print(outputs.head())

    supplementary = load_supplementary()
    print(
        f"\nSupplementary table rows: {len(supplementary)} "
        f"columns: {len(supplementary.columns)}"
    )
    print("Supplementary preview:")
    print(supplementary.head())


if __name__ == "__main__":
    main()

