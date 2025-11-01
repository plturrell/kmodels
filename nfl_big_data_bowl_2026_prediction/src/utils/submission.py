from __future__ import annotations

import argparse
from pathlib import Path

from ..modeling import baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission for NFL Big Data Bowl 2026 Prediction")
    parser.add_argument("--model-path", type=Path, default=baseline.MODEL_PATH, help="Path to the trained model joblib file")
    parser.add_argument("--output", type=Path, default=baseline.SUBMISSION_PATH, help="Destination for the submission CSV")
    return parser.parse_args()


def main(argv=None) -> int:
    args = parse_args()
    output_path = baseline.generate_submission(model_path=args.model_path, output_path=args.output)
    print(f"Submission saved to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
