"""Entry point for self-supervised contrastive pretraining."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .data.contrastive_dataset import ContrastivePairDataset
from .modeling.contrastive_pretrain import ContrastiveConfig, ContrastivePretrainModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Contrastive pretraining for forgery detection")
    parser.add_argument("--data-root", type=Path, required=True, help="Path to extracted Kaggle dataset")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contrastive"))
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = ContrastivePairDataset(data_root=args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    config = ContrastiveConfig(
        learning_rate=args.learning_rate,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
    )
    module = ContrastivePretrainModule(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=str(args.output_dir),
        logger=True,
    )
    trainer.fit(module, train_dataloaders=loader)
    trainer.save_checkpoint(str(args.output_dir / "contrastive.ckpt"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


