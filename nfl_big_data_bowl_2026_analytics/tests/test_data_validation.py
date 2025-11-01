"""Pytest-based data integrity checks for the NFL Big Data Bowl 2026 Analytics competition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "nfl-big-data-bowl-2026-analytics"
    / "114239_nfl_competition_files_published_analytics_final"
)
SUPPLEMENTARY_CSV = DATA_DIR / "supplementary_data.csv"
TRAIN_DIR = DATA_DIR / "train"


if not SUPPLEMENTARY_CSV.exists():  # pragma: no cover - optional data bundle
    pytest.skip("NFL analytics dataset not available in repository", allow_module_level=True)


@pytest.fixture(scope="module")
def supplementary_df() -> pd.DataFrame:
    return pd.read_csv(SUPPLEMENTARY_CSV, low_memory=False)


def test_required_paths_exist():
    assert DATA_DIR.exists(), f"Missing data directory at {DATA_DIR}"
    assert SUPPLEMENTARY_CSV.exists(), f"Missing supplementary data at {SUPPLEMENTARY_CSV}"
    assert TRAIN_DIR.exists(), f"Missing train directory at {TRAIN_DIR}"


def test_supplementary_basic_shape(supplementary_df: pd.DataFrame):
    assert len(supplementary_df) > 0, "Supplementary data is unexpectedly empty"
    assert len(supplementary_df.columns) > 0, "Supplementary data missing columns"


def test_no_fully_missing_columns(supplementary_df: pd.DataFrame):
    fully_missing = supplementary_df.isnull().all()
    missing_columns = list(fully_missing[fully_missing].index)
    assert not missing_columns, f"Columns entirely missing values: {missing_columns}"


def test_no_duplicate_rows(supplementary_df: pd.DataFrame):
    duplicates = int(supplementary_df.duplicated().sum())
    assert duplicates == 0, f"Found {duplicates} duplicate rows in supplementary data"


def test_numeric_columns_are_finite(supplementary_df: pd.DataFrame):
    numeric_cols = supplementary_df.select_dtypes(include=[np.number])
    if numeric_cols.empty:
        pytest.skip("No numeric columns detected in supplementary data")
    assert np.isfinite(numeric_cols.to_numpy(dtype=float)).all(), (
        "Numeric columns contain non-finite values"
    )


def test_categorical_column_cardinality(supplementary_df: pd.DataFrame):
    categorical_cols = supplementary_df.select_dtypes(include=["object", "category"])
    if categorical_cols.empty:
        pytest.skip("No categorical columns detected in supplementary data")
    high_cardinality = {
        col: categorical_cols[col].nunique(dropna=False)
        for col in categorical_cols.columns
    }
    # Ensure at least one categorical column provides meaningful variety
    assert any(count > 1 for count in high_cardinality.values()), (
        "All categorical columns are constant"
    )


def test_train_directory_contains_csvs():
    csv_files = list(TRAIN_DIR.glob("*.csv"))
    assert csv_files, "No CSV files found within the train directory"

