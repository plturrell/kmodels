#!/usr/bin/env python3
"""
Train G-JEPA model for geometry proof heuristic guidance.

Usage:
    python scripts/train_gjepa.py --num_problems 1000 --num_epochs 10
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.geometry.gjepa_trainer import train_gjepa
from src.modeling.gjepa_model import GJEPA
from src.geometry.scene_encoder import SceneEncoder


_OUTPUTS_DIR = project_root / "outputs"


def _safe_outputs_dir(path: Path) -> Path:
    """Constrain outputs to a directory name under aimo_3/outputs/."""
    return (_OUTPUTS_DIR / Path(path).name).resolve()


def _safe_checkpoint(path: Path) -> Path:
    """Constrain checkpoints to a file name under aimo_3/outputs/."""
    return (_OUTPUTS_DIR / Path(path).name).resolve()


def main():
    parser = argparse.ArgumentParser(description="Train G-JEPA model")
    parser.add_argument(
        "--num_problems",
        type=int,
        default=1000,
        help="Number of problems to generate for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    args = parser.parse_args()
    output_dir = _safe_outputs_dir(Path("gjepa"))
    checkpoint = None
    
    print("=" * 60)
    print("Training G-JEPA Model")
    print("=" * 60)
    print(f"Number of problems: {args.num_problems}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Train model
    model, encoder = train_gjepa(
        num_problems=args.num_problems,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=output_dir,
        checkpoint_path=checkpoint,
    )
    
    print("\n✓ Training complete!")
    print(f"Model and encoder saved to {output_dir / 'gjepa_final.pt'}")


if __name__ == "__main__":
    main()

