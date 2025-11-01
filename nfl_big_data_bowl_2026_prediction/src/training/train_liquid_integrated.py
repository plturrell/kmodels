"""Integrated training script for Liquid Neural Network with real NFL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ..data.loaders import load_train_inputs, load_train_outputs
from ..data.trajectory_dataset import TrajectoryDataset, collate_trajectories
from ..modeling.liquid_network import LiquidNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Train Liquid Network on NFL data")
    
    # Data
    parser.add_argument("--data-dir", type=Path, default=None,
                       help="Data directory (uses default if not specified)")
    parser.add_argument("--max-sequences", type=int, default=None,
                       help="Limit number of sequences for faster training")
    
    # Model
    parser.add_argument("--input-dim", type=int, default=10,
                       help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=128,
                       help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of liquid layers")
    parser.add_argument("--output-dim", type=int, default=2,
                       help="Output dimension (x, y)")
    parser.add_argument("--dt", type=float, default=0.1,
                       help="Integration time step")
    parser.add_argument("--integrator", type=str, default="euler",
                       choices=["euler", "rk4"],
                       help="Integration method")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split fraction")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device")
    
    # Output
    parser.add_argument("--output-dir", type=Path, 
                       default=Path("outputs/liquid"),
                       help="Output directory")
    
    return parser.parse_args()


def load_nfl_data(data_dir: Optional[Path], max_sequences: Optional[int]):
    """Load NFL trajectory data."""
    print("Loading NFL training data...")
    
    # Load input and output trajectories
    inputs = load_train_inputs(base_dir=data_dir, concatenate=True)
    outputs = load_train_outputs(base_dir=data_dir, concatenate=True)
    
    print(f"Loaded {len(inputs)} input frames, {len(outputs)} output frames")
    
    # Define feature columns (numeric only)
    feature_columns = ["s", "a", "player_height", "player_weight"]
    
    # Create dataset
    dataset = TrajectoryDataset(
        inputs,
        outputs,
        feature_columns=feature_columns,
        max_sequences=max_sequences,
    )
    
    print(f"Created dataset with {len(dataset)} sequences")
    
    return dataset


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(loader, desc="Training"):
        # Move to device
        features = batch["features"].to(device)
        positions = batch["positions"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["target"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(features, positions, mask)
        
        # Compute loss (only on valid timesteps)
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            features = batch["features"].to(device)
            positions = batch["positions"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["target"].to(device)
            
            predictions = model(features, positions, mask)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    dataset = load_nfl_data(args.data_dir, args.max_sequences)
    
    # Split into train/val
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
    )
    
    # Create model
    model = LiquidNetwork(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        dt=args.dt,
        integrator=args.integrator,
        dropout=args.dropout,
    ).to(args.device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_loss = float('inf')
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss = validate(model, val_loader, criterion, args.device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    # Save final model and history
    torch.save(model.state_dict(), args.output_dir / "final_model.pt")
    
    with open(args.output_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()

