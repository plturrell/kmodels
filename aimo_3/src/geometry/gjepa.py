"""
Geometric Joint-Embedding Predictive Architecture (G-JEPA).

Inspired by Meta's V-JEPA, this module learns latent dynamics of geometric proofs
through self-supervised learning on proof sequences. It acts as a heuristic guide
for the MCTS search by predicting promising proof steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass

from .scene_graph import GeometricSceneGraph
from .state import State
from .theorems import Theorem


@dataclass
class ProofStep:
    """A single step in a proof sequence."""
    state: State
    theorem_applied: Optional[str] = None
    match: Optional[Dict[str, str]] = None
    step_index: int = 0


class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder for geometric scene graphs.
    
    Encodes a GeometricSceneGraph into a fixed-dimensional latent vector.
    """
    
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
    ):
        """
        Initialize graph encoder.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            output_dim: Output latent dimension
            num_layers: Number of GNN layers
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Graph-level pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, scene_graph: GeometricSceneGraph) -> torch.Tensor:
        """
        Encode scene graph to latent vector.
        
        Args:
            scene_graph: GeometricSceneGraph to encode
            
        Returns:
            Latent vector of shape (output_dim,)
        """
        # Extract graph structure
        nodes = list(scene_graph.graph.nodes())
        
        if len(nodes) == 0:
            # Empty graph -> zero vector
            return torch.zeros(self.output_dim)
        
        # Create node features
        node_features = self._extract_node_features(nodes, scene_graph)
        node_embeddings = self.node_proj(node_features)
        
        # Message passing (simplified GNN)
        for layer in self.gnn_layers:
            # Aggregate neighbor messages
            messages = torch.zeros_like(node_embeddings)
            for i, node in enumerate(nodes):
                neighbors = list(scene_graph.graph.neighbors(node))
                if neighbors:
                    neighbor_indices = [nodes.index(n) for n in neighbors if n in nodes and n in nodes]
                    if neighbor_indices:
                        neighbor_embs = node_embeddings[neighbor_indices]
                        messages[i] = neighbor_embs.mean(dim=0)
            
            # Update node embeddings (residual connection)
            node_embeddings = node_embeddings + layer(messages)
            node_embeddings = F.relu(node_embeddings)
        
        # Graph-level pooling (mean pooling)
        graph_embedding = node_embeddings.mean(dim=0)
        
        # Project to output dimension
        latent = self.pool(graph_embedding)
        
        return latent
    
    def _extract_node_features(self, nodes: List, scene_graph: GeometricSceneGraph) -> torch.Tensor:
        """
        Extract features from graph nodes.
        
        Args:
            nodes: List of nodes
            scene_graph: Scene graph
            
        Returns:
            Node feature tensor of shape (num_nodes, node_dim)
        """
        features = []
        for node in nodes:
            # Extract features based on node type
            if hasattr(node, 'x') and hasattr(node, 'y'):
                # Point: coordinates
                feat = torch.tensor([float(node.x) if hasattr(node, 'x') else 0.0,
                                    float(node.y) if hasattr(node, 'y') else 0.0,
                                    1.0, 0.0, 0.0])  # [x, y, is_point, is_line, is_circle]
            elif hasattr(node, 'point1') and hasattr(node, 'point2'):
                # Line: endpoints
                feat = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
            elif hasattr(node, 'center') and hasattr(node, 'radius'):
                # Circle: center and radius
                feat = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            else:
                # Unknown type
                feat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Pad to node_dim
            if len(feat) < self.node_dim:
                feat = torch.cat([feat, torch.zeros(self.node_dim - len(feat))])
            features.append(feat[:self.node_dim])
        
        if not features:
            return torch.zeros(1, self.node_dim)
        
        return torch.stack(features)


class ContextEncoder(nn.Module):
    """
    Transformer-based context encoder for proof sequences.
    
    Processes sequence of latent vectors from unmasked proof steps.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 100,
    ):
        """
        Initialize context encoder.
        
        Args:
            latent_dim: Dimension of input latent vectors
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
    def forward(self, latent_sequence: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sequence of latent vectors.
        
        Args:
            latent_sequence: Tensor of shape (batch_size, seq_len, latent_dim)
            mask: Optional mask tensor for padding
            
        Returns:
            Context tensor of shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, _ = latent_sequence.shape
        
        # Project to hidden dimension
        x = self.input_proj(latent_sequence)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create padding mask if needed
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=~mask)
        
        # Pool to single context vector (mean pooling)
        context = x.mean(dim=1)
        
        return context


class Predictor(nn.Module):
    """
    Predictor network for masked step prediction.
    
    Takes context from unmasked steps and predicts latent vector of masked step.
    """
    
    def __init__(
        self,
        context_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        """
        Initialize predictor.
        
        Args:
            context_dim: Dimension of context vector
            hidden_dim: Hidden dimension
            output_dim: Output latent dimension (should match encoder output)
        """
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict latent vector from context.
        
        Args:
            context: Context vector of shape (batch_size, context_dim)
            
        Returns:
            Predicted latent vector of shape (batch_size, output_dim)
        """
        return self.predictor(context)


class GJEPA(nn.Module):
    """
    Geometric Joint-Embedding Predictive Architecture.
    
    Learns latent dynamics of geometric proofs through self-supervised learning.
    """
    
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        latent_dim: int = 256,
        context_dim: int = 512,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 100,
    ):
        """
        Initialize G-JEPA.
        
        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            latent_dim: Latent space dimension
            context_dim: Context dimension
            hidden_dim: Hidden dimension
            num_gnn_layers: Number of GNN layers
            num_transformer_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Encoder: Scene graph -> latent vector
        self.encoder = GraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=num_gnn_layers,
        )
        
        # Context encoder: Sequence of latents -> context
        self.context_encoder = ContextEncoder(
            latent_dim=latent_dim,
            hidden_dim=context_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )
        
        # Predictor: Context -> predicted latent
        self.predictor = Predictor(
            context_dim=context_dim,
            hidden_dim=latent_dim,
            output_dim=latent_dim,
        )
        
        self.latent_dim = latent_dim
    
    def encode_state(self, state: State) -> torch.Tensor:
        """
        Encode a proof state to latent vector.
        
        Args:
            state: Proof state
            
        Returns:
            Latent vector
        """
        return self.encoder(state.graph)
    
    def predict_next_latent(
        self,
        proof_sequence: List[State],
        mask_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Predict latent vector of masked steps given context.
        
        Args:
            proof_sequence: Sequence of proof states
            mask_indices: Indices of masked steps (if None, predicts next step)
            
        Returns:
            Predicted latent vector(s)
        """
        # Encode all states
        latents = torch.stack([self.encode_state(state) for state in proof_sequence])
        
        # Create mask (unmasked = True)
        if mask_indices is None:
            # Predict next step (mask last)
            mask = torch.ones(len(proof_sequence) - 1, dtype=torch.bool)
            context_latents = latents[:-1].unsqueeze(0)
        else:
            # Mask specified indices
            mask = torch.ones(len(proof_sequence), dtype=torch.bool)
            mask[mask_indices] = False
            context_latents = latents[mask].unsqueeze(0)
        
        # Encode context
        context = self.context_encoder(context_latents)
        
        # Predict masked latent
        predicted = self.predictor(context)
        
        return predicted
    
    def compute_heuristic_score(
        self,
        current_state: State,
        candidate_state: State,
        goal_state: Optional[State] = None,
    ) -> float:
        """
        Compute heuristic score for candidate state.
        
        Higher score = more promising candidate.
        
        Args:
            current_state: Current proof state
            candidate_state: Candidate next state
            goal_state: Optional goal state
            
        Returns:
            Heuristic score (higher is better)
        """
        self.eval()  # Ensure in eval mode
        with torch.no_grad():
            # Encode states
            current_latent = self.encode_state(current_state)
            candidate_latent = self.encode_state(candidate_state)
            
            # Predict ideal next step from current state
            predicted_latent = self.predict_next_latent([current_state])
            
            # Compute similarity between candidate and predicted
            similarity = F.cosine_similarity(
                candidate_latent.unsqueeze(0),
                predicted_latent,
                dim=1,
            ).item()
            
            # If goal state provided, also compute similarity to goal
            goal_score = 0.0
            if goal_state is not None:
                goal_latent = self.encode_state(goal_state)
                goal_similarity = F.cosine_similarity(
                    candidate_latent.unsqueeze(0),
                    goal_latent.unsqueeze(0),
                    dim=1,
                ).item()
                goal_score = goal_similarity * 0.3  # Weight goal similarity
            
            # Combined score
            score = similarity + goal_score
            
            return score
    
    def forward(
        self,
        proof_sequences: List[List[State]],
        mask_indices: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            proof_sequences: Batch of proof sequences
            mask_indices: Batch of mask indices for each sequence
            
        Returns:
            (predicted_latents, target_latents) for loss computation
        """
        batch_size = len(proof_sequences)
        predicted_latents = []
        target_latents = []
        
        for seq, masks in zip(proof_sequences, mask_indices):
            # Encode all states
            latents = torch.stack([self.encode_state(state) for state in seq])
            
            # Get context from unmasked steps
            unmasked_indices = [i for i in range(len(seq)) if i not in masks]
            if not unmasked_indices:
                continue
            
            context_latents = latents[unmasked_indices].unsqueeze(0)
            context = self.context_encoder(context_latents)
            
            # Predict masked latents
            for mask_idx in masks:
                predicted = self.predictor(context)
                predicted_latents.append(predicted)
                target_latents.append(latents[mask_idx])
        
        if not predicted_latents:
            # Return dummy tensors if no valid predictions
            dummy = torch.zeros(1, self.latent_dim)
            return dummy, dummy
        
        predicted = torch.cat(predicted_latents, dim=0)
        targets = torch.stack(target_latents)
        
        return predicted, targets

