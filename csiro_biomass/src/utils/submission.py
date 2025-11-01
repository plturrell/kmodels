from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def build_submission_dataframe(
    predictions: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    image_column: str,
    target_names: Sequence[str],
    target_name_column: str,
    id_column: str,
    target_value_column: str = "target",
) -> pd.DataFrame:
    melted = predictions.melt(
        id_vars=[image_column],
        value_vars=list(target_names),
        var_name=target_name_column,
        value_name=target_value_column,
    )

    id_lookup = (
        test_df[[id_column, image_column, target_name_column]]
        .drop_duplicates()
    )
    submission_df = id_lookup.merge(
        melted,
        on=[image_column, target_name_column],
        how="left",
    )
    return submission_df[[id_column, target_name_column, target_value_column]]


def save_submission_csv(
    predictions_df: pd.DataFrame,
    *,
    test_df: pd.DataFrame,
    image_column: str,
    target_names: Sequence[str],
    target_name_column: str,
    id_column: str,
    target_value_column: str,
    output_path: Path,
    sample_submission: Optional[pd.DataFrame] = None,
) -> Path:
    submission_df = build_submission_dataframe(
        predictions_df,
        test_df=test_df,
        image_column=image_column,
        target_names=target_names,
        target_name_column=target_name_column,
        id_column=id_column,
        target_value_column=target_value_column,
    )
    if sample_submission is not None:
        submission_df = sample_submission[[id_column]].merge(
            submission_df[[id_column, target_value_column]],
            on=id_column,
            how="left",
        )
    else:
        submission_df = submission_df[[id_column, target_value_column]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    return output_path
