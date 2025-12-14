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
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/gjepa"),
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training G-JEPA Model")
    print("=" * 60)
    print(f"Number of problems: {args.num_problems}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Train model
    model, encoder = train_gjepa(
        num_problems=args.num_problems,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
    )
    
    print("\n✓ Training complete!")
    print(f"Model and encoder saved to {args.output_dir / 'gjepa_final.pt'}")


if __name__ == "__main__":
    main()

