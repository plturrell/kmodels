"""Gradient boosting baseline over tabular image statistics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


_BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = _BASE_DIR / "outputs" / "features" / "tabular_features.csv"
DEFAULT_OUTPUT_DIR = _BASE_DIR / "outputs" / "tabular_baseline"


@dataclass
class TabularConfig:
    features_csv: Path = DEFAULT_FEATURES
    output_dir: Path = DEFAULT_OUTPUT_DIR
    n_splits: int = 5
    random_state: int = 42


def _build_pipeline(random_state: int) -> Pipeline:
    classifier = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=0.0,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", classifier),
        ]
    )


def run_cv(config: TabularConfig) -> pd.DataFrame:
    df = pd.read_csv(config.features_csv)
    feature_cols = [col for col in df.columns if col not in ("case_id", "label")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    skf = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    rows: List[dict] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        pipeline = _build_pipeline(config.random_state + fold)
        pipeline.fit(X[train_idx], y[train_idx])
        preds = pipeline.predict(X[val_idx])
        probas = pipeline.predict_proba(X[val_idx])[:, 1]
        rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y[val_idx], preds),
                "f1": f1_score(y[val_idx], preds),
                "roc_auc": roc_auc_score(y[val_idx], probas),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a gradient boosting baseline on tabular features."
    )
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TabularConfig(
        features_csv=args.features_csv,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        random_state=args.seed,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = run_cv(config)
    metrics_path = config.output_dir / "cv_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary = metrics_df.mean(numeric_only=True).to_dict()
    (config.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    cfg = asdict(config)
    cfg["features_csv"] = str(cfg["features_csv"])
    cfg["output_dir"] = str(cfg["output_dir"])
    (config.output_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    print("Cross-validation metrics:")
    print(metrics_df)
    print("Averages:", summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
