"""Baseline modeling pipeline for NFL Big Data Bowl 2026."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from competitions.nfl_big_data_bowl_2026_prediction.src.data import (
    load_submission_format,
    load_test_input,
)
from competitions.nfl_big_data_bowl_2026_prediction.src.features import (
    compute_player_features,
    save_features,
)

FEATURES_DIR = Path(
    "competitions/nfl_big_data_bowl_2026_prediction/data/processed"
)
FEATURES_BASENAME = "baseline_features"
MODEL_PATH = Path(
    "competitions/nfl_big_data_bowl_2026_prediction/outputs/baseline_random_forest.joblib"
)
METRICS_PATH = Path(
    "competitions/nfl_big_data_bowl_2026_prediction/outputs/baseline_metrics.json"
)
SUBMISSION_PATH = Path(
    "competitions/nfl_big_data_bowl_2026_prediction/outputs/baseline_submission.csv"
)
DROP_COLUMNS = {
    "target_final_x",
    "target_final_y",
    "season",
    "week",
    "game_id",
    "play_id",
    "nfl_id",
    "player_to_predict",
}


def _locate_feature_artifact(preferred_suffix: str = ".parquet") -> Path:
    """Return path to existing features or trigger creation."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    candidate = FEATURES_DIR / f"{FEATURES_BASENAME}{preferred_suffix}"
    if candidate.exists():
        return candidate
    csv_candidate = FEATURES_DIR / f"{FEATURES_BASENAME}.csv"
    if csv_candidate.exists():
        return csv_candidate
    return save_features(candidate)


def _read_features(path: Path) -> pd.DataFrame:
    """Load engineered features based on the file suffix."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature format: {path.suffix}")


def _select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier/target columns before modeling."""
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    return df.drop(columns=drop_cols)


def _prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter rows with targets and split features/labels."""
    df = df.copy()
    df["player_to_predict"] = df["player_to_predict"].astype(bool)
    df = df[df["player_to_predict"]]
    df = df.dropna(subset=["target_final_x", "target_final_y"])
    y = df[["target_final_x", "target_final_y"]]
    X = _select_model_features(df)
    return X, y


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer | str:
    """Construct preprocessing transformer based on column types."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    return ColumnTransformer(transformers) if transformers else "passthrough"


def train_baseline(
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    feature_suffix: str = ".parquet",
) -> dict[str, float]:
    """Train a RandomForest baseline and persist artifacts."""
    feature_path = _locate_feature_artifact(preferred_suffix=feature_suffix)
    features = _read_features(feature_path)
    X, y = _prepare_training_frame(features)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = _build_preprocessor(X_train)
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    max_depth=max_depth,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)

    mae_vector = mean_absolute_error(y_valid, preds, multioutput="raw_values")
    mae_overall = mean_absolute_error(y_valid, preds)

    metrics = {
        "feature_path": str(feature_path),
        "mae_overall": float(mae_overall),
        "mae_x": float(mae_vector[0]),
        "mae_y": float(mae_vector[1]),
        "n_train_samples": int(len(X_train)),
        "n_valid_samples": int(len(X_valid)),
        "test_size": float(test_size),
        "n_estimators": int(n_estimators),
        "max_depth": None if max_depth is None else int(max_depth),
        "random_state": int(random_state),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


def _prepare_inference_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return identifier columns and model-ready features for inference."""
    df = df.copy()
    df["player_to_predict"] = df["player_to_predict"].astype(bool)
    df = df[df["player_to_predict"]]
    identifiers = df[["game_id", "play_id", "nfl_id"]].reset_index(drop=True)
    X = _select_model_features(df).reset_index(drop=True)
    return identifiers, X


def generate_submission(
    model_path: Path = MODEL_PATH,
    output_path: Path = SUBMISSION_PATH,
) -> Path:
    """Generate a Kaggle-formatted submission using the trained model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    test_input = load_test_input()
    submission_index = load_submission_format()

    player_features = compute_player_features(test_input)
    identifiers, X_test = _prepare_inference_features(player_features)

    preds = model.predict(X_test)
    pred_df = identifiers.copy()
    pred_df["pred_x"] = preds[:, 0]
    pred_df["pred_y"] = preds[:, 1]

    fallback = (
        test_input[test_input["player_to_predict"]]
        .sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"], as_index=False)[["x", "y"]]
        .last()
        .rename(columns={"x": "fallback_x", "y": "fallback_y"})
    )

    submission = submission_index.merge(
        pred_df, on=["game_id", "play_id", "nfl_id"], how="left"
    ).merge(
        fallback, on=["game_id", "play_id", "nfl_id"], how="left"
    )

    submission["x"] = submission["pred_x"].fillna(submission["fallback_x"])
    submission["y"] = submission["pred_y"].fillna(submission["fallback_y"])
    submission = submission[["id", "x", "y"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline model and optionally submit")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-estimators", type=int, default=300, help="RandomForest trees")
    parser.add_argument("--max-depth", type=int, default=None, help="RandomForest max depth")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Where to store the trained model")
    parser.add_argument("--metrics-path", type=Path, default=METRICS_PATH, help="Where to store metrics JSON")
    parser.add_argument("--feature-suffix", choices=[".parquet", ".csv"], default=".parquet", help="Feature artifact suffix preference")
    parser.add_argument("--generate-submission", action="store_true", help="Generate submission after training")
    parser.add_argument("--submission-path", type=Path, default=SUBMISSION_PATH, help="Output path for submission CSV")
    parser.add_argument("--compare-leaderboard", action="store_true", help="Fetch live leaderboard snapshot after training")
    parser.add_argument("--competition", type=str, default="nfl-big-data-bowl-2026-prediction", help="Kaggle competition slug for leaderboard lookup")
    parser.add_argument("--leaderboard-top", type=int, default=10, help="Number of top leaderboard rows to display")
    parser.add_argument("--leaderboard-username", type=str, help="Override Kaggle username to highlight on leaderboard")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict[str, float]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    metrics = train_baseline(
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        feature_suffix=args.feature_suffix,
    )

    print("Baseline training complete. Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"Model saved to {args.model_path}")
    print(f"Metrics saved to {args.metrics_path}")

    if args.generate_submission:
        submission_path = generate_submission(model_path=args.model_path, output_path=args.submission_path)
        print(f"Submission saved to {submission_path}")

    if args.compare_leaderboard:
        try:
            from competitions.nfl_big_data_bowl_2026_prediction.src.utils import leaderboard as lb

            username = args.leaderboard_username or lb.detect_default_username()
            entries = lb.fetch_leaderboard(args.competition)
            local_score = metrics.get("mae_overall")
            summary = lb.summarise(
                entries,
                top_n=args.leaderboard_top,
                user=username,
                local_score=local_score,
            )
            print("\n" + summary)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to fetch leaderboard: {exc}")

    return metrics


if __name__ == "__main__":  # pragma: no cover
    main()
