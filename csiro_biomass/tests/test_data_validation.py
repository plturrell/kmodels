"""Pytest-based data integrity checks for the CSIRO Image2Biomass competition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "csiro_biomass_extract"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
TRAIN_IMAGE_DIR = DATA_DIR / "train"
TEST_IMAGE_DIR = DATA_DIR / "test"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"


if not TRAIN_CSV.exists():  # pragma: no cover - optional data bundle
    pytest.skip("CSIRO dataset not available in repository", allow_module_level=True)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    return pd.read_csv(TEST_CSV)


def test_required_files_exist():
    assert TRAIN_CSV.exists(), f"Missing training CSV at {TRAIN_CSV}"
    assert TEST_CSV.exists(), f"Missing test CSV at {TEST_CSV}"
    assert TRAIN_IMAGE_DIR.exists(), f"Missing training images at {TRAIN_IMAGE_DIR}"
    assert TEST_IMAGE_DIR.exists(), f"Missing test images at {TEST_IMAGE_DIR}"
    assert SAMPLE_SUBMISSION.exists(), f"Missing sample submission at {SAMPLE_SUBMISSION}"


def test_train_columns_present(train_df: pd.DataFrame):
    required_columns = [
        "sample_id",
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "target_name",
        "target",
    ]
    missing = [col for col in required_columns if col not in train_df.columns]
    assert not missing, f"Missing expected columns: {missing}"


def test_target_names(train_df: pd.DataFrame):
    expected_targets = {
        "Dry_Clover_g",
        "Dry_Dead_g",
        "Dry_Green_g",
        "Dry_Total_g",
        "GDM_g",
    }
    actual_targets = set(train_df["target_name"].unique())
    assert actual_targets == expected_targets, (
        f"Target mismatch: expected {expected_targets}, got {actual_targets}"
    )


def test_no_missing_values(train_df: pd.DataFrame):
    missing_total = int(train_df.isnull().sum().sum())
    assert missing_total == 0, f"Found {missing_total} missing values in training data"


def test_target_values_non_negative(train_df: pd.DataFrame):
    assert (train_df["target"] >= 0).all(), "Target column contains negative values"


def test_biomass_composition_consistency(train_df: pd.DataFrame):
    pivot = (
        train_df.pivot_table(
            index="image_path",
            columns="target_name",
            values="target",
        )
        .reset_index()
        .dropna()
    )

    required = {"Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g"}
    if not required.issubset(pivot.columns):
        pytest.skip("Pivoted targets missing expected columns")

    computed_total = (
        pivot["Dry_Clover_g"] + pivot["Dry_Dead_g"] + pivot["Dry_Green_g"]
    )
    error = pivot["Dry_Total_g"] - computed_total
    assert np.allclose(error.abs().max(), 0, atol=0.1), (
        f"Biomass composition exceeds tolerance (max error {error.abs().max():.4f})"
    )


def test_sampled_train_images_exist_and_valid():
    image_paths = sorted((TRAIN_IMAGE_DIR).glob("**/*.jpg"))
    sample_size = min(25, len(image_paths))
    assert sample_size > 0, "No training images found"

    for image_path in image_paths[:sample_size]:
        assert image_path.exists(), f"Missing image file {image_path}"
        with Image.open(image_path) as img:
            assert img.mode in {"RGB", "L"}, (
                f"Unexpected image mode {img.mode} for {image_path.name}"
            )


def test_ndvi_range(train_df: pd.DataFrame):
    ndvi_min = train_df["Pre_GSHH_NDVI"].min()
    ndvi_max = train_df["Pre_GSHH_NDVI"].max()
    assert -1 <= ndvi_min <= 1, f"NDVI min outside expected range: {ndvi_min}"
    assert -1 <= ndvi_max <= 1, f"NDVI max outside expected range: {ndvi_max}"


def test_height_non_negative(train_df: pd.DataFrame):
    assert (train_df["Height_Ave_cm"] >= 0).all(), "Negative height values found"


def test_test_csv_structure(test_df: pd.DataFrame):
    required = {"sample_id", "image_path"}
    assert required.issubset(test_df.columns), (
        f"Missing columns from test.csv: {required - set(test_df.columns)}"
    )


def test_sample_submission_alignment(test_df: pd.DataFrame):
    submission = pd.read_csv(SAMPLE_SUBMISSION)
    required_cols = {"sample_id", "target"}
    assert required_cols.issubset(submission.columns), (
        f"Sample submission missing columns: {required_cols - set(submission.columns)}"
    )

    expected_ids = set(test_df["sample_id"].unique())
    submission_ids = set(submission["sample_id"].unique())
    assert expected_ids == submission_ids, "Sample submission IDs do not match test set"

