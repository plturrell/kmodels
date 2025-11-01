from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

DEFAULT_ABLATIONS: Dict[str, Sequence[str]] = {
    "image_only": ["--no-metadata", "--no-curriculum", "--ema-decay", "0"],
    "no_ema": ["--ema-decay", "0"],
    "no_curriculum": ["--no-curriculum"],
}


def build_base_cli(args: argparse.Namespace) -> List[str]:
    cli: List[str] = [
        "--train-csv",
        str(args.train_csv),
        "--image-dir",
        str(args.image_dir),
        "--output-dir",
        str(args.output_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--image-size",
        str(args.image_size),
        "--model",
        args.model,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.test_csv:
        cli.extend(["--test-csv", str(args.test_csv)])
    if args.sample_submission:
        cli.extend(["--sample-submission", str(args.sample_submission)])
    if args.fractal_csv:
        cli.extend(["--fractal-csv", str(args.fractal_csv)])
    if args.snapshot_count is not None:
        cli.extend(["--snapshot-count", str(args.snapshot_count)])
    if args.n_folds > 1:
        cli.extend(["--n-folds", str(args.n_folds)])
        if args.cv_group_column:
            cli.extend(["--cv-group-column", args.cv_group_column])
    return cli


def run_ablation(
    ablation_name: str,
    base_cli: List[str],
    ablation_args: Sequence[str],
    extra_args: Sequence[str],
    base_output_dir: Path,
) -> None:
    output_dir = base_output_dir / ablation_name
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "competitions.csiro_biomass.src.train",
        *base_cli,
        "--output-dir",
        str(output_dir),
        *ablation_args,
        *extra_args,
    ]

    print(f"[ablation] Running '{ablation_name}':", " ".join(command))
    subprocess.run(command, check=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard ablation suites for CSIRO pipeline")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--fractal-csv", type=Path)
    parser.add_argument("--model", default="efficientnet_b3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-count", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--cv-group-column", type=str)
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="*",
        default=list(DEFAULT_ABLATIONS.keys()),
        help="Subset of ablations to run (default: all)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded to the training command",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    missing = [name for name in args.ablations if name not in DEFAULT_ABLATIONS]
    if missing:
        raise ValueError(f"Unknown ablations requested: {missing}")

    base_cli = build_base_cli(args)
    extra_args = [arg for arg in args.extra_args if arg != "--"]
    for ablation in args.ablations:
        ablation_args = DEFAULT_ABLATIONS[ablation]
        run_ablation(ablation, base_cli, ablation_args, extra_args, args.output_dir)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
