from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def build_submission_dataframe(
    predictions: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    id_columns: Sequence[str],
    target_columns: Sequence[str],
) -> pd.DataFrame:
    """Return a dataframe with the required id + target columns."""
    id_columns = list(id_columns)
    target_columns = list(target_columns)

    ids = test_df[id_columns].drop_duplicates()
    merged = ids.merge(
        predictions[id_columns + target_columns],
        on=id_columns,
        how="left",
    )
    return merged[id_columns + target_columns]


def save_submission_csv(
    predictions: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    id_columns: Sequence[str],
    target_columns: Sequence[str],
    output_path: Path,
    sample_submission: Optional[pd.DataFrame] = None,
) -> Path:
    submission_df = build_submission_dataframe(
        predictions,
        test_df=test_df,
        id_columns=id_columns,
        target_columns=target_columns,
    )

    if sample_submission is not None:
        expected_columns = list(sample_submission.columns)
        for column in expected_columns:
            if column not in submission_df.columns:
                submission_df[column] = sample_submission[column]
        submission_df = submission_df[expected_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path
