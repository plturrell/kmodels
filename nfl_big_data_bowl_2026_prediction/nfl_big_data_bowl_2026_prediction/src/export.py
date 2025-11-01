from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib

from .modeling import baseline


def export_model(
    model_path: Path,
    output_dir: Path,
) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_model = output_dir / "baseline_model.joblib"
    model = joblib.load(model_path)
    joblib.dump(model, output_model)

    metadata = {
        "source_model": str(model_path),
        "export_model": str(output_model),
        "model_type": type(model).__name__,
    }
    with (output_dir / "export_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
    return output_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline random-forest model")
    parser.add_argument("--model-path", type=Path, default=baseline.MODEL_PATH, help="Path to trained model (joblib)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store exported model")
    return parser.parse_args()


def main(argv=None) -> int:
    args = parse_args()
    output_path = export_model(args.model_path, args.output_dir)
    print(f"Exported model to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
