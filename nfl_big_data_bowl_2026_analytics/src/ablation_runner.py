from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_ABLATIONS: Dict[str, Sequence[str]] = {
    "baseline": [],
    "short_epochs": ["--epochs", "10"],
    "no_scheduler": ["--no-scheduler"],
    "smaller_model": ["--hidden-dims", "256,128"],
}


def build_base_cli(args: argparse.Namespace) -> List[str]:
    cli: List[str] = [
        "--data-root",
        str(args.data_root),
        "--bundle-dirname",
        args.bundle_dirname,
        "--output-dir",
        str(args.output_dir),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.snapshot_count is not None:
        cli.extend(["--snapshot-count", str(args.snapshot_count)])
    if args.val_fraction is not None:
        cli.extend(["--val-fraction", str(args.val_fraction)])
    return cli


def run_ablation(
    name: str,
    base_cli: List[str],
    ablation_args: Sequence[str],
    extra_args: Sequence[str],
    output_dir: Path,
) -> None:
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "competitions.nfl_big_data_bowl_2026_analytics.src.train",
        *base_cli,
        "--output-dir",
        str(run_dir),
        *ablation_args,
        *extra_args,
    ]
    print(f"[ablation] {name}:", " ".join(command))
    subprocess.run(command, check=True)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation suite for the analytics competition")
    parser.add_argument("--data-root", default="competitions/nfl_big_data_bowl_2026_analytics/data/raw")
    parser.add_argument("--bundle-dirname", default="nfl-big-data-bowl-2026-analytics")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-count", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--ablations",
        nargs="*",
        default=list(DEFAULT_ABLATIONS.keys()),
        help="Which ablations to execute",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments forwarded to train.py",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [name for name in args.ablations if name not in DEFAULT_ABLATIONS]
    if unknown:
        raise ValueError(f"Unknown ablations requested: {unknown}")

    base_cli = build_base_cli(args)
    extra_args = [arg for arg in args.extra_args if arg != "--"]

    for name in args.ablations:
        run_ablation(
            name,
            base_cli,
            DEFAULT_ABLATIONS[name],
            extra_args,
            args.output_dir,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
