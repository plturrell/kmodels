"""Run two-phase Perceiver training: MSE warm-up followed by Sharpe fine-tuning."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_command(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def two_phase_training(args: argparse.Namespace) -> None:
    base_cmd = [sys.executable, "-m", "competitions.hull_tactical_market_prediction.src.train_nn"]

    common_flags = [
        "--model", "perceiver",
        "--train-csv", str(args.train_csv),
        "--test-csv", str(args.test_csv),
        "--target-column", args.target_column,
        "--id-column", args.id_column,
        "--perceiver-layers", str(args.perceiver_layers),
        "--perceiver-num-latents", str(args.perceiver_num_latents),
        "--perceiver-latent-dim", str(args.perceiver_latent_dim),
        "--perceiver-heads", str(args.perceiver_heads),
        "--perceiver-dropout", str(args.perceiver_dropout),
        "--perceiver-ff-mult", str(args.perceiver_ff_mult),
        "--max-nan-ratio", str(args.max_nan_ratio),
        "--output-dir", str(args.output_dir),
        "--batch-size", str(args.batch_size),
    ]

    # Phase 1: MSE warm-up
    phase1_dir = args.output_dir / "perceiver-two-phase-mse"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase1_cmd = base_cmd + common_flags + [
        "--run-name", "perceiver-two-phase-mse",
        "--epochs", str(args.mse_epochs),
        "--learning-rate", str(args.mse_learning_rate),
        "--loss-mode", "mse",
    ]
    print("=== Phase 1: MSE Warm-up ===")
    _run_command(phase1_cmd)

    checkpoint_path = phase1_dir / "model_state.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected checkpoint at {checkpoint_path}. Ensure train_nn saves model_state.pt in the run directory."
        )

    # Phase 2: Sharpe fine-tuning
    phase2_dir = args.output_dir / "perceiver-two-phase-sharpe"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    phase2_cmd = base_cmd + common_flags + [
        "--run-name", "perceiver-two-phase-sharpe",
        "--epochs", str(args.sharpe_epochs),
        "--learning-rate", str(args.sharpe_learning_rate),
        "--loss-mode", "sharpe_mse",
        "--loss-mse-weight", str(args.sharpe_mse_weight),
        "--sharpe-lambda", str(args.sharpe_lambda),
        "--checkpoint", str(checkpoint_path),
    ]
    print("=== Phase 2: Sharpe Fine-tuning ===")
    _run_command(phase2_cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-phase Perceiver training pipeline.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/data/raw/train.csv"),
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("competitions/hull_tactical_market_prediction/data/raw/test.csv"),
    )
    parser.add_argument("--target-column", default="forward_returns")
    parser.add_argument("--id-column", default="date_id")
    parser.add_argument("--output-dir", type=Path, default=Path("competitions/hull_tactical_market_prediction/outputs/tabular_nn"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-nan-ratio", type=float, default=0.5)

    # Perceiver architecture
    parser.add_argument("--perceiver-layers", type=int, default=4)
    parser.add_argument("--perceiver-num-latents", type=int, default=24)
    parser.add_argument("--perceiver-latent-dim", type=int, default=96)
    parser.add_argument("--perceiver-heads", type=int, default=6)
    parser.add_argument("--perceiver-dropout", type=float, default=0.1)
    parser.add_argument("--perceiver-ff-mult", type=int, default=4)

    # Phase hyperparameters
    parser.add_argument("--mse-epochs", type=int, default=20)
    parser.add_argument("--mse-learning-rate", type=float, default=3e-4)
    parser.add_argument("--sharpe-epochs", type=int, default=10)
    parser.add_argument("--sharpe-learning-rate", type=float, default=1e-4)
    parser.add_argument("--sharpe-mse-weight", type=float, default=0.9)
    parser.add_argument("--sharpe-lambda", type=float, default=0.1)

    # aliases for user-provided flags
    parser.add_argument("--phase1-epochs", type=int)
    parser.add_argument("--phase2-epochs", type=int)
    parser.add_argument("--phase1-lr", type=float)
    parser.add_argument("--phase2-lr", type=float)
    parser.add_argument("--mse-weight", type=float)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase1_epochs is not None:
        args.mse_epochs = args.phase1_epochs
    if args.phase2_epochs is not None:
        args.sharpe_epochs = args.phase2_epochs
    if args.phase1_lr is not None:
        args.mse_learning_rate = args.phase1_lr
    if args.phase2_lr is not None:
        args.sharpe_learning_rate = args.phase2_lr
    if args.mse_weight is not None:
        args.sharpe_mse_weight = args.mse_weight
    two_phase_training(args)


if __name__ == "__main__":
    main()
