"""Submission validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def validate_submission_format(
    submission: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """Validate submission format.
    
    Args:
        submission: Submission DataFrame
        required_columns: Required column names
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if required_columns is None:
        required_columns = ["game_id", "play_id", "nfl_id", "target_x", "target_y"]
    
    # Check required columns
    missing_cols = set(required_columns) - set(submission.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for duplicates
    if not errors:
        id_cols = [c for c in required_columns if c != "target_x" and c != "target_y"]
        duplicates = submission.duplicated(subset=id_cols)
        if duplicates.any():
            errors.append(f"Found {duplicates.sum()} duplicate rows")
    
    # Check for missing values
    if not errors:
        for col in required_columns:
            if submission[col].isna().any():
                errors.append(f"Column '{col}' contains NaN values")
    
    return len(errors) == 0, errors


def validate_prediction_ranges(
    submission: pd.DataFrame,
    x_range: Tuple[float, float] = (0.0, 120.0),
    y_range: Tuple[float, float] = (0.0, 53.3),
) -> Tuple[bool, List[str]]:
    """Validate prediction value ranges.
    
    Args:
        submission: Submission DataFrame
        x_range: Valid range for x coordinates (yards)
        y_range: Valid range for y coordinates (yards)
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check x range
    if "target_x" in submission.columns:
        x_min, x_max = x_range
        out_of_range_x = (submission["target_x"] < x_min) | (submission["target_x"] > x_max)
        if out_of_range_x.any():
            errors.append(
                f"Found {out_of_range_x.sum()} predictions with target_x outside "
                f"valid range [{x_min}, {x_max}]"
            )
    
    # Check y range
    if "target_y" in submission.columns:
        y_min, y_max = y_range
        out_of_range_y = (submission["target_y"] < y_min) | (submission["target_y"] > y_max)
        if out_of_range_y.any():
            errors.append(
                f"Found {out_of_range_y.sum()} predictions with target_y outside "
                f"valid range [{y_min}, {y_max}]"
            )
    
    # Check for infinite values
    if "target_x" in submission.columns:
        if np.isinf(submission["target_x"]).any():
            errors.append("Found infinite values in target_x")
    
    if "target_y" in submission.columns:
        if np.isinf(submission["target_y"]).any():
            errors.append("Found infinite values in target_y")
    
    return len(errors) == 0, errors


def validate_submission_completeness(
    submission: pd.DataFrame,
    expected_ids: Optional[pd.DataFrame] = None,
) -> Tuple[bool, List[str]]:
    """Validate submission completeness against expected IDs.
    
    Args:
        submission: Submission DataFrame
        expected_ids: DataFrame with expected (game_id, play_id, nfl_id)
    
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if expected_ids is None:
        return True, []
    
    id_cols = ["game_id", "play_id", "nfl_id"]
    
    # Check for missing predictions
    expected_set = set(map(tuple, expected_ids[id_cols].values))
    submission_set = set(map(tuple, submission[id_cols].values))
    
    missing = expected_set - submission_set
    if missing:
        errors.append(f"Missing {len(missing)} expected predictions")
    
    # Check for extra predictions
    extra = submission_set - expected_set
    if extra:
        errors.append(f"Found {len(extra)} unexpected predictions")
    
    return len(errors) == 0, errors


def validate_submission(
    submission: pd.DataFrame,
    expected_ids: Optional[pd.DataFrame] = None,
    x_range: Tuple[float, float] = (0.0, 120.0),
    y_range: Tuple[float, float] = (0.0, 53.3),
) -> Tuple[bool, List[str]]:
    """Comprehensive submission validation.
    
    Args:
        submission: Submission DataFrame
        expected_ids: Expected IDs (optional)
        x_range: Valid x coordinate range
        y_range: Valid y coordinate range
    
    Returns:
        Tuple of (is_valid, all_error_messages)
    """
    all_errors = []
    
    # Format validation
    is_valid, errors = validate_submission_format(submission)
    all_errors.extend(errors)
    
    # Range validation
    is_valid, errors = validate_prediction_ranges(submission, x_range, y_range)
    all_errors.extend(errors)
    
    # Completeness validation
    if expected_ids is not None:
        is_valid, errors = validate_submission_completeness(submission, expected_ids)
        all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors


def validate_and_save_submission(
    submission: pd.DataFrame,
    output_path: Path,
    expected_ids: Optional[pd.DataFrame] = None,
    strict: bool = True,
) -> bool:
    """Validate and save submission with error reporting.
    
    Args:
        submission: Submission DataFrame
        output_path: Path to save submission
        expected_ids: Expected IDs (optional)
        strict: If True, raise error on validation failure
    
    Returns:
        True if validation passed
    """
    is_valid, errors = validate_submission(submission, expected_ids)
    
    if not is_valid:
        print("Submission validation failed:")
        for error in errors:
            print(f"  - {error}")
        
        if strict:
            raise ValueError("Submission validation failed")
        else:
            print("Saving submission anyway (strict=False)")
    else:
        print("Submission validation passed!")
    
    # Save submission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return is_valid


__all__ = [
    "validate_submission_format",
    "validate_prediction_ranges",
    "validate_submission_completeness",
    "validate_submission",
    "validate_and_save_submission",
]

