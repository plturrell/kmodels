"""Baseline training script."""

from pathlib import Path
from typing import Optional

from ..config import load_config
from ..data.loader import load_problems
from .trainer import AIMOTrainer


def main(config_name: str = "baseline", model_name: Optional[str] = None) -> None:
    """Main training entry point (safe defaults, no CLI paths)."""
    project_root = Path(__file__).parent.parent.parent
    config = load_config(config_name)
    config_dict = config.to_dict()

    if model_name:
        config_dict["model"]["model_name"] = model_name

    # Load data
    data_dir_name = Path(str(config_dict["data"]["data_dir"])).name
    data_dir = (project_root / "data" / data_dir_name).resolve()
    problems = load_problems(data_dir)

    # Split train/val
    train_split = config_dict["data"]["train_split"]
    split_idx = int(len(problems) * train_split)
    train_problems = problems[:split_idx]
    val_problems = problems[split_idx:] if split_idx < len(problems) else []

    # Initialize trainer
    output_dir_name = Path(str(config_dict.get("output_dir", "baseline"))).name
    output_dir = (project_root / "outputs" / output_dir_name).resolve()
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

