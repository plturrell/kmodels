"""Neural network baseline for CAFA 6 protein function prediction.

This module implements a deep learning baseline using ESM-2 protein embeddings
and a multi-layer neural network for multi-label classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..data import ProteinSample
from ..features.embeddings import embed_sequences

LOGGER = logging.getLogger(__name__)


class ProteinFunctionPredictor(nn.Module):
    """Multi-layer neural network for protein function prediction.
    
    Architecture:
        Input (embedding_dim) 
        → FC + BatchNorm + ReLU + Dropout
        → FC + BatchNorm + ReLU + Dropout
        → FC (num_labels)
        → Sigmoid
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_labels: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        
        layers = []
        in_dim = embedding_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_labels))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
        
        Returns:
            Logits (batch_size, num_labels)
        """
        return self.network(x)


def train_neural_baseline(
    train_samples: Sequence[ProteinSample],
    val_samples: Sequence[ProteinSample],
    *,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    hidden_dims: Sequence[int] = (512, 256),
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    num_epochs: int = 50,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    use_fractal_features: bool = False,
    fractal_max_iter: int = 100,
) -> dict:
    """Train neural network baseline using ESM-2 embeddings.
    
    Args:
        train_samples: Training protein samples
        val_samples: Validation protein samples
        model_name: ESM-2 model name
        hidden_dims: Hidden layer dimensions
        dropout: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of training epochs
        device: Device to use (cuda/cpu)
        output_dir: Directory to save model
    
    Returns:
        Dictionary with training metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    LOGGER.info("Generating ESM-2 embeddings for training set...")
    train_embeddings = embed_sequences(
        train_samples,
        model_name=model_name,
        batch_size=8,
        device=device,
        return_array=True,
    )

    LOGGER.info("Generating ESM-2 embeddings for validation set...")
    val_embeddings = embed_sequences(
        val_samples,
        model_name=model_name,
        batch_size=8,
        device=device,
        return_array=True,
    )

    # Add fractal features if requested
    if use_fractal_features:
        LOGGER.info("Extracting fractal features (expected +5-10% Fmax)...")
        from ..features.fractal_features import FractalProteinFeatures, combine_features

        fractal_extractor = FractalProteinFeatures(max_iter=fractal_max_iter)

        LOGGER.info("Extracting fractal features for training set...")
        train_fractal = fractal_extractor.extract_batch(train_embeddings)

        LOGGER.info("Extracting fractal features for validation set...")
        val_fractal = fractal_extractor.extract_batch(val_embeddings)

        # Combine original and fractal features
        train_embeddings = combine_features(train_embeddings, train_fractal)
        val_embeddings = combine_features(val_embeddings, val_fractal)

        LOGGER.info(f"Combined embedding dimension: {train_embeddings.shape[1]}")

    # Prepare labels
    LOGGER.info("Preparing labels...")
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([s.go_terms for s in train_samples])
    val_labels = mlb.transform([s.go_terms for s in val_samples])
    
    num_labels = len(mlb.classes_)
    embedding_dim = train_embeddings.shape[1]
    
    LOGGER.info(f"Training set: {len(train_samples)} samples")
    LOGGER.info(f"Validation set: {len(val_samples)} samples")
    LOGGER.info(f"Number of GO terms: {num_labels}")
    LOGGER.info(f"Embedding dimension: {embedding_dim}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_embeddings),
        torch.FloatTensor(train_labels),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_embeddings),
        torch.FloatTensor(val_labels),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ProteinFunctionPredictor(
        embedding_dim=embedding_dim,
        num_labels=num_labels,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    LOGGER.info("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        LOGGER.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': best_model_state,
            'mlb': mlb,
            'config': {
                'embedding_dim': embedding_dim,
                'num_labels': num_labels,
                'hidden_dims': hidden_dims,
                'dropout': dropout,
            }
        }, output_dir / "neural_baseline.pt")
        LOGGER.info(f"Model saved to {output_dir / 'neural_baseline.pt'}")
    
    return {
        'best_val_loss': best_val_loss,
        'num_labels': num_labels,
        'model': model,
        'mlb': mlb,
    }


__all__ = [
    "ProteinFunctionPredictor",
    "train_neural_baseline",
]

