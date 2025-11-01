from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib


def export_model(model_path: Path, output_dir: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = output_dir / "tabular_model.joblib"
    model = joblib.load(model_path)
    joblib.dump(model, exported)
    metadata = {
        "source_model": str(model_path),
        "export_model": str(exported),
        "model_type": type(model).__name__,
    }
    with (output_dir / "export_metadata.json").open("w") as fp:
        json.dump(metadata, fp, indent=2)
    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hull Tactical baseline model")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model joblib")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for exported artifacts")
    return parser.parse_args()


def main(argv=None) -> int:
    args = parse_args()
    exported = export_model(args.model_path, args.output_dir)
    print(f"Exported model to {exported}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
