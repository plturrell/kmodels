"""Baseline training script."""

import argparse
from pathlib import Path

from ..config import load_config
from ..data.loader import load_problems
from .trainer import AIMOTrainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train baseline model for AIMO 3")
    parser.add_argument(
        "--config",
        default="baseline",
        help="Configuration name (default: baseline)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: outputs/baseline)",
    )
    parser.add_argument(
        "--model-name",
        help="Model name (default: from config)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config_dict = config.to_dict()

    # Override with CLI args
    if args.data_dir:
        config_dict["data"]["data_dir"] = str(args.data_dir)
    if args.output_dir:
        config_dict["output_dir"] = str(args.output_dir)
    if args.model_name:
        config_dict["model"]["model_name"] = args.model_name

    # Load data
    data_dir = Path(config_dict["data"]["data_dir"])
    problems = load_problems(data_dir)

    # Split train/val
    train_split = config_dict["data"]["train_split"]
    split_idx = int(len(problems) * train_split)
    train_problems = problems[:split_idx]
    val_problems = problems[split_idx:] if split_idx < len(problems) else []

    # Initialize trainer
    output_dir = Path(config_dict.get("output_dir", "outputs/baseline"))
    trainer = AIMOTrainer(
        model_name=config_dict["model"]["model_name"],
        output_dir=output_dir,
        config=config_dict.get("training", {}),
    )

    # Train
    trainer.train(train_problems, val_problems)
    trainer.save_config()

    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()

