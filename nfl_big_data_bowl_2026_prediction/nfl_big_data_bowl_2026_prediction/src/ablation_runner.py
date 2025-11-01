from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_ABLATIONS: Dict[str, Sequence[str]] = {
    "baseline": [],
    "shallow_forest": ["--n-estimators", "100", "--max-depth", "6"],
    "deep_forest": ["--n-estimators", "500", "--max-depth", "18"],
    "small_split": ["--test-size", "0.1"],
}


def build_base_cli(args: argparse.Namespace) -> List[str]:
    cli: List[str] = [
        "--test-size",
        str(args.test_size),
        "--random-state",
        str(args.random_state),
        "--model-path",
        str(args.model_path),
        "--metrics-path",
        str(args.metrics_path),
        "--feature-suffix",
        args.feature_suffix,
    ]
    if args.generate_submission:
        cli.append("--generate-submission")
        cli.extend(["--submission-path", str(args.submission_path)])
    if args.compare_leaderboard:
        cli.append("--compare-leaderboard")
    return cli


def run_ablation(
    name: str,
    base_cli: List[str],
    ablation_args: Sequence[str],
    extra_args: Sequence[str],
) -> None:
    command = [
        sys.executable,
        "-m",
        "competitions.nfl_big_data_bowl_2026_prediction.src.modeling.baseline",
        *base_cli,
        *ablation_args,
        *extra_args,
    ]
    print(f"[ablation] {name}:", " ".join(command))
    subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation suite for NFL prediction baseline")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=baseline.MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=baseline.METRICS_PATH)
    parser.add_argument("--submission-path", type=Path, default=baseline.SUBMISSION_PATH)
    parser.add_argument("--feature-suffix", choices=[".parquet", ".csv"], default=".parquet")
    parser.add_argument("--generate-submission", action="store_true")
    parser.add_argument("--compare-leaderboard", action="store_true")
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(DEFAULT_ABLATIONS.keys()),
        help="Which ablations to run",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to baseline trainer",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [name for name in args.ablations if name not in DEFAULT_ABLATIONS]
    if missing:
        raise ValueError(f"Unknown ablations: {missing}")

    base_cli = build_base_cli(args)
    extra_args = [arg for arg in args.extra_args if arg != "--"]

    for name in args.ablations:
        run_ablation(name, base_cli, DEFAULT_ABLATIONS[name], extra_args)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
