"""Pytest-based data integrity checks for the Hull Tactical Market Prediction competition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"


if not TRAIN_CSV.exists():  # pragma: no cover - optional data bundle
    pytest.skip("Hull Tactical dataset not available in repository", allow_module_level=True)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def test_required_files_exist():
    assert TRAIN_CSV.exists(), f"Missing training CSV at {TRAIN_CSV}"
    assert TEST_CSV.exists(), f"Missing test CSV at {TEST_CSV}"


def test_required_columns_present(train_df: pd.DataFrame):
    required = {
        "date_id",
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
    }
    assert required.issubset(train_df.columns), (
        f"Missing required columns: {required - set(train_df.columns)}"
    )


def test_required_columns_non_null(train_df: pd.DataFrame):
    required = [
        "date_id",
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
    ]
    missing_counts = train_df[required].isnull().sum()
    assert (missing_counts == 0).all(), (
        f"Required columns contain nulls: {missing_counts[missing_counts > 0].to_dict()}"
    )


def test_feature_groups_detected(train_df: pd.DataFrame):
    prefixes = ["D", "E", "I", "M", "P", "S", "V"]
    for prefix in prefixes:
        cols = [c for c in train_df.columns if c.startswith(prefix) and c != "date_id"]
        assert cols, f"No features detected for prefix '{prefix}'"


def test_target_columns_finite(train_df: pd.DataFrame):
    target_cols = [
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
    ]
    for column in target_cols:
        values = train_df[column].dropna()
        assert np.isfinite(values).all(), f"Non-finite values found in {column}"


def test_date_id_monotonic(train_df: pd.DataFrame):
    diffs = train_df["date_id"].diff().dropna()
    assert (diffs >= 0).all(), "date_id column must be monotonically increasing"


def test_no_duplicate_date_ids(train_df: pd.DataFrame):
    duplicates = train_df["date_id"].duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate date_id entries"


def test_test_csv_columns(test_df: pd.DataFrame):
    required = {"date_id"}
    assert required.issubset(test_df.columns), (
        f"Missing columns in test.csv: {required - set(test_df.columns)}"
    )


def test_train_and_test_feature_parity(train_df: pd.DataFrame, test_df: pd.DataFrame):
    target_cols = {
        "forward_returns",
        "risk_free_rate",
        "market_forward_excess_returns",
    }
    train_features = set(train_df.columns) - target_cols
    test_features = set(test_df.columns)
    assert train_features == test_features, (
        f"Feature mismatch between train and test: missing={train_features - test_features}, "
        f"extra={test_features - train_features}"
    )


def test_constant_features_absent(train_df: pd.DataFrame):
    feature_cols = [
        c
        for c in train_df.columns
        if c
        not in {
            "date_id",
            "forward_returns",
            "risk_free_rate",
            "market_forward_excess_returns",
        }
    ]
    constants = [c for c in feature_cols if train_df[c].nunique(dropna=False) <= 1]
    assert not constants, f"Constant features detected: {constants[:5]}"


def test_sample_submission_columns_if_present(test_df: pd.DataFrame):
    if not SAMPLE_SUBMISSION.exists():
        pytest.skip("sample_submission.csv not provided")

    submission = pd.read_csv(SAMPLE_SUBMISSION)
    required = {"date_id", "forward_returns"}
    assert required.issubset(submission.columns), (
        f"Sample submission missing columns: {required - set(submission.columns)}"
    )

    assert len(submission) == len(test_df), (
        "Sample submission length does not match test set"
    )

