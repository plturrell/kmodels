from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

_BASE_DIR = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = _BASE_DIR / "data" / "raw" / "recodai-luc-scientific-image-forgery-detection"

DEFAULT_ABLATIONS: Dict[str, Sequence[str]] = {
    "baseline": [],
    "no_ema": ["--ema-decay", "0"],
    "no_scheduler": ["--no-scheduler"],
    "small_dataset": ["--max-train-samples", "512", "--max-val-samples", "256"],
}


def build_base_cli(args: argparse.Namespace) -> List[str]:
    cli: List[str] = [
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--mask-loss-weight",
        str(args.mask_loss_weight),
        "--val-fraction",
        str(args.val_fraction),
        "--seed",
        str(args.seed),
        "--data-root",
        str(args.data_root),
    ]
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
        "competitions.recodai_luc_scientific_image_forgery_detection.src.train",
        *base_cli,
        "--output-dir",
        str(run_dir),
        *ablation_args,
        *extra_args,
    ]
    print(f"[ablation] {name}:", " ".join(command))
    subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation suite for the forgery detection baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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
