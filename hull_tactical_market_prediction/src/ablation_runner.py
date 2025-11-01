from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_ABLATIONS: Dict[str, Sequence[str]] = {
    "baseline": [],
    "less_lags": ["--lag-step", "1", "--lag-step", "5"],
    "more_lags": ["--lag-step", "1", "--lag-step", "5", "--lag-step", "10", "--lag-step", "20"],
    "no_rolling": ["--no-rolling-features"],
}


def build_base_cli(args: argparse.Namespace) -> List[str]:
    cli: List[str] = [
        "--train-csv",
        str(args.train_csv),
        "--test-csv",
        str(args.test_csv),
        "--sample-submission",
        str(args.sample_submission),
        "--target-column",
        args.target_column,
        "--id-column",
        args.id_column,
        "--submission-column",
        args.submission_column,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.save_model:
        cli.append("--save-model")
    return cli


def run_ablation(name: str, base_cli: List[str], extra_args: Sequence[str]) -> None:
    command = [
        sys.executable,
        "-m",
        "competitions.hull_tactical_market_prediction.src.train",
        *base_cli,
        *extra_args,
    ]
    print(f"[ablation] {name}:", " ".join(command))
    subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablations for Hull Tactical baseline")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--submission-column", default="prediction")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(DEFAULT_ABLATIONS.keys()),
        help="Which ablations to run",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    missing = [name for name in args.ablations if name not in DEFAULT_ABLATIONS]
    if missing:
        raise ValueError(f"Unknown ablations: {missing}")

    base_cli = build_base_cli(args)
    for name in args.ablations:
        run_ablation(name, base_cli, DEFAULT_ABLATIONS[name])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
