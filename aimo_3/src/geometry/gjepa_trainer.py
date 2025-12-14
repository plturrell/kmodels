"""
Training infrastructure for G-JEPA.

Generates synthetic proof sequences and trains the G-JEPA model
using masked prediction objective.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Union
import random
from pathlib import Path
import json

from .solver import GeometrySolver
from .state import State
from .generator import ProblemGenerator
from .scene_sequence import SceneTrace, build_scene_sequence
from .scene_encoder import SceneEncoder
from .jepa_dataset import GeometryJEPADataset
from ..modeling.gjepa_model import GJEPA, create_gjepa_model


PathType = Union[str, Path]


# ProofSequenceDataset replaced by GeometryJEPADataset


def generate_training_traces(
    num_problems: int = 1000,
    solver: Optional[GeometrySolver] = None,
    generator: Optional[ProblemGenerator] = None,
) -> List[SceneTrace]:
    """
    Generate synthetic proof traces for training.
    
    Args:
        num_problems: Number of problems to generate
        solver: Geometry solver (creates new if None)
        generator: Problem generator (creates new if None)
        
    Returns:
        List of SceneTrace objects
    """
    if solver is None:
        solver = GeometrySolver(max_search_iterations=500, max_depth=30)
    
    if generator is None:
        generator = ProblemGenerator(seed=42)
    
    traces = []
    
    print(f"Generating {num_problems} proof traces...")
    
    for i in range(num_problems):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_problems} traces")
        
        try:
            # Generate a problem
            problem_type = random.choice(["triangle", "circle", "coordinate"])
            difficulty = random.choice(["easy", "medium", "hard"])
            
            if problem_type == "triangle":
                triangle_type = random.choice(["right", "equilateral", "isosceles", "scalene"])
                problem, _ = generator.generate_triangle_problem(triangle_type, difficulty)
            elif problem_type == "circle":
                circle_type = random.choice(["inscribed", "circumscribed", "tangent", "chord"])
                problem, _ = generator.generate_circle_problem(circle_type, difficulty)
            else:
                coord_type = random.choice(["distance", "midpoint", "slope", "area"])
                problem, _ = generator.generate_coordinate_problem(coord_type, difficulty)
            
            # Solve problem and extract real proof sequence
            try:
                solver.solve(problem, problem_id=f"problem_{i}")
                proof_states = solver.get_proof_states()
                
                # If solver didn't find proof or states are empty, create minimal trace
                if not proof_states or len(proof_states) < 2:
                    initial_graph = solver.parser.parse(problem)
                    initial_state = State(graph=initial_graph)
                    proof_states = [initial_state]
            except Exception as e:
                # Fallback: create minimal trace
                initial_graph = solver.parser.parse(problem)
                initial_state = State(graph=initial_graph)
                proof_states = [initial_state]
            
            # Build scene trace
            trace = build_scene_sequence(
                proof_states=proof_states,
                trace_id=f"trace_{i}",
                problem_id=f"problem_{i}",
            )
            traces.append(trace)
            
        except Exception as e:
            # Skip problematic problems
            continue
    
    print(f"Generated {len(traces)} valid proof traces")
    return traces


class GJEPATrainer:
    """Trainer for G-JEPA model."""
    
    def __init__(
        self,
        model: GJEPA,
        encoder: SceneEncoder,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: G-JEPA model
            encoder: SceneEncoder for encoding scenes
            learning_rate: Learning rate
            device: Device to train on (auto-detect if None)
        """
        self.model = model
        self.encoder = encoder
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model.to(device)
        self.encoder.to(device)
        
        self.optimizer = optim.Adam(
            list(model.parameters()) + list(encoder.parameters()),
            lr=learning_rate,
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Data loader
            
        Returns:
            Average loss
        """
        self.model.train()
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        for h_seq, mask_indices, lengths in dataloader:
            # Move to device
            h_seq = h_seq.to(self.device)
            mask_indices = [idx.to(self.device) for idx in mask_indices]
            # lengths stay as Python list of ints
            
            # Forward pass
            loss = self.model.compute_loss(h_seq, mask_indices, lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
        val_loader: Optional[DataLoader] = None,
    ) -> List[float]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs
            val_loader: Optional validation data loader
            
        Returns:
            List of training losses per epoch
        """
        losses = []
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            losses.append(train_loss)
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"  Val Loss: {val_loss:.4f}")
        
        return losses
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        self.encoder.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for h_seq, mask_indices, lengths in dataloader:
                h_seq = h_seq.to(self.device)
                mask_indices = [idx.to(self.device) for idx in mask_indices]
                
                loss = self.model.compute_loss(h_seq, mask_indices, lengths)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_gjepa(
    num_problems: int = 1000,
    num_epochs: int = 10,
    batch_size: int = 32,
    output_dir: PathType = Path("outputs/gjepa"),
    checkpoint_path: Optional[PathType] = None,
    latent_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    device: Optional[torch.device] = None,
    train_encoder: bool = True,
    cache_latents: bool = False,
) -> Tuple[GJEPA, SceneEncoder]:
    """
    Train G-JEPA model.
    
    Args:
        num_problems: Number of problems to generate
        num_epochs: Number of training epochs
        batch_size: Batch size
        output_dir: Output directory
        checkpoint_path: Optional path to load checkpoint
        latent_dim: Latent dimension [D]
        hidden_dim: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        device: Device to train on (auto-detect if None)
        train_encoder: If True, jointly train encoder + predictor.
                      If False, freeze encoder during training.
        cache_latents: If True, precompute all latent sequences before training.
                      Trades memory for speed. Sets train_encoder=False automatically.
        
    Returns:
        (trained_model, encoder) tuple
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize encoder and model
    encoder = SceneEncoder(output_dim=latent_dim)
    model = create_gjepa_model(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    
    # Move to device
    encoder.to(device)
    model.to(device)
    
    # Generate training data
    print("Generating training traces...")
    traces = generate_training_traces(num_problems=num_problems)
    
    # Create dataset and dataloader
    dataset = GeometryJEPADataset(
        traces=traces,
        encoder=encoder,
        train_encoder=train_encoder,
        cache_latents=cache_latents,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=GeometryJEPADataset.collate_fn,
    )
    
    # Initialize trainer (will handle device internally)
    trainer = GJEPATrainer(model, encoder, device=device)
    
    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
    
    # Train
    losses = trainer.train(dataloader, num_epochs=num_epochs)
    
    # Save final model
    final_path = output_dir / "gjepa_final.pt"
    trainer.save_checkpoint(final_path)
    print(f"Model saved to {final_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({'losses': losses}, f)
    
    return model, encoder

