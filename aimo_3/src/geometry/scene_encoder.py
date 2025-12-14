"""
Scene encoder for G-JEPA: encodes geometric scenes to latent vectors.

Provides:
- encode_state(scene_or_trace) → [D]
- encode_trace(trace) → [T+1, D]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Optional, Any

from .scene_graph import GeometricSceneGraph
from .state import State
from .scene_sequence import SceneTrace

from .state import State
from .scene_sequence import SceneTrace
from ..modeling.graph_attention_encoder import GraphAttentionEncoder


class GNNSceneEncoder(nn.Module):
    """
    Standard GNN scene encoder (legacy).
    
    Encodes geometric scene graphs to fixed-dimensional latent vectors.
    """
    
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
    ):
        """
        Initialize scene encoder.
        
        Args:
            node_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            output_dim: Output latent dimension [D]
            num_layers: Number of GNN layers
        """
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
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

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert value to float, falling back to default on None or invalid inputs."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def forward(self, scene_graph: GeometricSceneGraph) -> torch.Tensor:
        """
        Encode scene graph to latent vector.
        
        Args:
            scene_graph: GeometricSceneGraph to encode
            
        Returns:
            Latent vector of shape [output_dim]
        """
        # Extract graph structure
        device = next(self.parameters()).device
        nodes = list(scene_graph.graph.nodes())
        
        if len(nodes) == 0:
            # Empty graph -> zero vector
            return torch.zeros(self.output_dim, device=device)
        
        # Extract node features
        node_features = self._extract_node_features(nodes, scene_graph)
        node_embeddings = self.node_proj(node_features)
        
        # Create node-to-index mapping for O(1) lookup
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Message passing
        for layer in self.gnn_layers:
            # Aggregate neighbor messages
            messages = torch.zeros_like(node_embeddings)
            for i, node in enumerate(nodes):
                neighbors = list(scene_graph.graph.neighbors(node))
                if neighbors:
                    neighbor_indices = []
                    for n in neighbors:
                        idx = node_to_idx.get(n)
                        if idx is not None and idx < len(node_embeddings):
                            neighbor_indices.append(idx)
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
    
    def _extract_node_features(
        self,
        nodes: List,
        scene_graph: GeometricSceneGraph,
    ) -> torch.Tensor:
        """
        Extract features from graph nodes.
        
        Args:
            nodes: List of nodes
            scene_graph: Scene graph
            
        Returns:
            Node feature tensor of shape (num_nodes, node_dim)
        """
        device = next(self.parameters()).device
        features = []
        for node in nodes:
            # Get node data from graph
            node_data = scene_graph.graph.nodes.get(node, {})
            node_type = node_data.get('type', 'Unknown')
            primitive = node_data.get('primitive')
            
            # Extract features based on node type
            if node_type == 'Point' and primitive:
                raw_x = getattr(primitive, 'x', 0.0) if hasattr(primitive, 'x') else 0.0
                raw_y = getattr(primitive, 'y', 0.0) if hasattr(primitive, 'y') else 0.0
                x = self._safe_float(raw_x)
                y = self._safe_float(raw_y)
                feat = torch.tensor([x, y, 1.0, 0.0, 0.0], device=device)
            elif node_type == 'Line' and primitive:
                p1 = getattr(primitive, 'point1', None)
                p2 = getattr(primitive, 'point2', None)
                if p1 and p2:
                    raw_x1 = getattr(p1, 'x', 0.0) if hasattr(p1, 'x') else 0.0
                    raw_y1 = getattr(p1, 'y', 0.0) if hasattr(p1, 'y') else 0.0
                    raw_x2 = getattr(p2, 'x', 0.0) if hasattr(p2, 'x') else 0.0
                    raw_y2 = getattr(p2, 'y', 0.0) if hasattr(p2, 'y') else 0.0
                    x1 = self._safe_float(raw_x1)
                    y1 = self._safe_float(raw_y1)
                    x2 = self._safe_float(raw_x2)
                    y2 = self._safe_float(raw_y2)
                    feat = torch.tensor([(x1+x2)/2, (y1+y2)/2, 0.0, 1.0, 0.0], device=device)
                else:
                    feat = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], device=device)
            elif node_type == 'Circle' and primitive:
                center = getattr(primitive, 'center', None)
                radius = getattr(primitive, 'radius', 0.0) if hasattr(primitive, 'radius') else 0.0
                if center:
                    raw_x = getattr(center, 'x', 0.0) if hasattr(center, 'x') else 0.0
                    raw_y = getattr(center, 'y', 0.0) if hasattr(center, 'y') else 0.0
                    x = self._safe_float(raw_x)
                    y = self._safe_float(raw_y)
                    r = self._safe_float(radius)
                    feat = torch.tensor([x, y, r, 0.0, 1.0], device=device)
                else:
                    r = self._safe_float(radius)
                    feat = torch.tensor([0.0, 0.0, r, 0.0, 1.0], device=device)
            else:
                feat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)
            
            # Pad to node_dim
            if len(feat) < self.node_dim:
                feat = torch.cat([feat, torch.zeros(self.node_dim - len(feat), device=device)])
            features.append(feat[:self.node_dim])
        
        if not features:
            return torch.zeros(1, self.node_dim, device=device)
        
        return torch.stack(features)
    
    def encode_state(
        self,
        scene_or_trace: Union[GeometricSceneGraph, State, SceneTrace],
    ) -> torch.Tensor:
        """
        Encode a scene, state, or trace to latent vector.
        
        Args:
            scene_or_trace: GeometricSceneGraph, State, or SceneTrace
            
        Returns:
            Latent vector of shape [D]
        """
        device = next(self.parameters()).device
        if isinstance(scene_or_trace, SceneTrace):
            # For trace, encode the last scene
            if len(scene_or_trace.scenes) > 0:
                return self.forward(scene_or_trace.scenes[-1])
            else:
                return torch.zeros(self.output_dim, device=device)
        elif isinstance(scene_or_trace, State):
            # For state, encode its graph
            return self.forward(scene_or_trace.graph)
        elif isinstance(scene_or_trace, GeometricSceneGraph):
            # Direct scene graph
            return self.forward(scene_or_trace)
        else:
            raise TypeError(f"Unsupported type: {type(scene_or_trace)}")
    
    def encode_trace(self, trace: SceneTrace) -> torch.Tensor:
        """
        Encode entire trace to sequence of latent vectors.
        
        Args:
            trace: SceneTrace to encode
            
        Returns:
            Tensor sequences
        """
        latents = [self.encode_state(scene) for scene in trace.scenes]
        if not latents:
            return torch.zeros(1, self.output_dim).to(next(self.parameters()).device)
        return torch.stack(latents)


# Alias for backward compatibility
SceneEncoder = GNNSceneEncoder


def create_scene_encoder(
    encoder_type: str = 'gat',
    node_dim: int = 64,
    hidden_dim: int = 256,
    output_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    **kwargs,
) -> Union[GNNSceneEncoder, GraphAttentionEncoder]:
    """
    Factory function to create scene encoder.
    
    Args:
        encoder_type: 'gnn' or 'gat'
        node_dim: Input node feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output latent dimension
        num_layers: Number of layers
        num_heads: Number of attention heads (GAT only)
        dropout: Dropout rate
        **kwargs: Additional arguments
        
    Returns:
        Scene encoder instance
    """
    if encoder_type.lower() == 'gat':
        return GraphAttentionEncoder(
            node_feature_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs
        )
    else:
        # Legacy GNN
        return GNNSceneEncoder(
            node_dim=node_dim if encoder_type == 'gnn' else 9,  # Legacy default
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
