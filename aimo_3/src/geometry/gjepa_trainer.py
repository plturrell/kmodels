"""
Training infrastructure for G-JEPA.

Generates synthetic proof sequences and trains the G-JEPA model
using masked prediction objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import random
from pathlib import Path
import json

from .solver import GeometrySolver
from .state import State
from .generator import ProblemGenerator
from .scene_sequence import SceneTrace, build_scene_sequence
from .scene_encoder import SceneEncoder
from .jepa_dataset import GeometryJEPADataset
from .world_model_dataset import GeometryWorldModelDataset
from ..modeling.gjepa_model import GJEPA, ActionConditionedGJEPA, create_gjepa_model


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
        model: ActionConditionedGJEPA,
        encoder: SceneEncoder,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
        use_distributional: bool = True,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Action-conditioned G-JEPA world model
            encoder: SceneEncoder for encoding scenes
            learning_rate: Learning rate
            device: Device to train on (auto-detect if None)
            use_distributional: Use distributional loss
        """
        self.model = model
        self.encoder = encoder
        self.use_distributional = bool(use_distributional)
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
        
        for batch_data in dataloader:
            if len(batch_data) == 4:  # Action-conditioned
                h_seq, actions, mask_indices, lengths = batch_data
                # Move to device
                h_seq = h_seq.to(self.device)
                actions = actions.to(self.device)
            else:  # Legacy
                h_seq, mask_indices, lengths = batch_data
                h_seq = h_seq.to(self.device)
                actions = None
            
            mask_indices = [idx.to(self.device) for idx in mask_indices]
            # lengths stay as Python list of ints
            
            # Forward pass
            if self.use_distributional:
                loss = self.model.compute_loss(h_seq, actions, mask_indices, lengths)
            else:
                loss = self.model.compute_loss(h_seq, actions, mask_indices, lengths)
            
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
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model with comprehensive metrics."""
        self.model.eval()
        self.encoder.eval()
        total_loss = 0.0
        rollout_errors = {1: 0.0, 3: 0.0, 5: 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in dataloader:
                if len(batch_data) == 4:
                    h_seq, actions, mask_indices, lengths = batch_data
                    h_seq = h_seq.to(self.device)
                    actions = actions.to(self.device)
                else:
                    h_seq, mask_indices, lengths = batch_data
                    h_seq = h_seq.to(self.device)
                    actions = None
                
                mask_indices = [idx.to(self.device) for idx in mask_indices]
                
                # Single-step loss
                loss = self.model.compute_loss(h_seq, actions, mask_indices, lengths)
                total_loss += loss.item()
                
                # Multi-step rollout validation (if action-conditioned)
                if actions is not None and hasattr(self.model, 'rollout'):
                    for horizon in [1, 3, 5]:
                        if h_seq.size(1) > horizon:
                            initial = h_seq[:, 0]
                            ground_truth = h_seq[:, 1:horizon+1]
                            rollout_actions = actions[:, :horizon]
                            
                            predicted = self.model.rollout(
                                initial_state=initial,
                                actions=rollout_actions,
                                horizon=horizon,
                            )
                            if isinstance(predicted, tuple):
                                predicted = predicted[0]
                            
                            error = F.mse_loss(predicted, ground_truth)
                            rollout_errors[horizon] += error.item()
                
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'rollout_1step': rollout_errors[1] / num_batches if num_batches > 0 else 0.0,
            'rollout_3step': rollout_errors[3] / num_batches if num_batches > 0 else 0.0,
            'rollout_5step': rollout_errors[5] / num_batches if num_batches > 0 else 0.0,
        }
        
        return metrics
    
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


class WorldModelTrainer:
    """Trainer for action-conditioned world model (ActionConditionedGJEPA)."""

    def __init__(
        self,
        model: ActionConditionedGJEPA,
        encoder: SceneEncoder,
        learning_rate: float = 1e-4,
        device: Optional[torch.device] = None,
    ) -> None:
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

        # Optionally train encoder jointly; can be changed by freezing encoder params
        self.optimizer = optim.Adam(
            list(model.parameters()) + list(encoder.parameters()),
            lr=learning_rate,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        self.encoder.train()
        total_loss = 0.0
        num_batches = 0

        for h_seq, actions, lengths in dataloader:
            h_seq = h_seq.to(self.device)
            actions = actions.to(self.device)

            loss = self.model.compute_loss(h_seq, actions=actions, lengths=lengths)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 10,
    ) -> List[float]:
        losses: List[float] = []
        for epoch in range(num_epochs):
            print(f"WorldModel Epoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader)
            losses.append(train_loss)
            print(f"  WorldModel Train Loss: {train_loss:.4f}")
        return losses

    def save_checkpoint(self, path: Path) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "encoder_state_dict": self.encoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "encoder_state_dict" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


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
) -> Tuple[Union[ActionConditionedGJEPA, GJEPA], SceneEncoder]:
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
    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            trainer.load_checkpoint(checkpoint_file)
    
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


def train_world_model(
    num_problems: int = 1000,
    num_epochs: int = 10,
    batch_size: int = 32,
    output_dir: PathType = Path("outputs/world_model"),
    checkpoint_path: Optional[PathType] = None,
    latent_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    device: Optional[torch.device] = None,
    train_encoder: bool = True,
    cache_latents: bool = True,
    use_liquid: bool = False,
) -> Tuple[ActionConditionedGJEPA, SceneEncoder]:
    """Train the action-conditioned G-JEPA world model on (state, action) traces."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize encoder and world model
    encoder = SceneEncoder(output_dim=latent_dim)
    model = create_gjepa_model(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        action_conditioned=True,
        use_distributional=True,
        use_ensemble=False,
        use_liquid=use_liquid,
    )

    encoder.to(device)
    model.to(device)

    # Generate training traces
    print("Generating training traces for world model...")
    traces = generate_training_traces(num_problems=num_problems)

    # Build world-model dataset
    dataset = GeometryWorldModelDataset(
        traces=traces,
        encoder=encoder,
        train_encoder=train_encoder,
        cache_latents=cache_latents,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=GeometryWorldModelDataset.collate_fn,
    )

    trainer = WorldModelTrainer(model, encoder, device=device)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            trainer.load_checkpoint(checkpoint_path)

    losses = trainer.train(dataloader, num_epochs=num_epochs)

    final_path = output_dir / "world_model_final.pt"
    trainer.save_checkpoint(final_path)
    print(f"World model saved to {final_path}")

    history_path = output_dir / "world_model_history.json"
    with open(history_path, "w") as f:
        json.dump({"losses": losses}, f)

    return model, encoder
