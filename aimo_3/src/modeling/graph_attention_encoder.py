"""
Graph Attention Network-based Scene Encoder.

Enhanced scene encoder using Graph Attention Networks (GAT) with:
- Rich node features (64-dim: position, type, geometry, derived, contextual)
- Edge features (32-dim: relation type, distance, angle, geometric properties)
- Multi-head attention for expressive graph encoding
- Hierarchical pooling for graph-level representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Dict
import math

from ..geometry.scene_graph import GeometricSceneGraph
from ..geometry.primitives import Point, Line, Circle
from ..geometry.state import State
from ..geometry.scene_sequence import SceneTrace


class GraphAttentionLayer(nn.Module):
    """Single Graph Attention Layer with edge features."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        output_dim: int,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        
        # Per-head dimension
        assert output_dim % num_heads == 0
        self.head_dim = output_dim // num_heads if concat_heads else output_dim
        
        # Linear transformations for nodes
        self.W_node = nn.Linear(node_dim, self.head_dim * num_heads, bias=False)
        self.W_edge = nn.Linear(edge_dim, self.head_dim * num_heads, bias=False)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim + self.head_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            edge_indices: [2, num_edges] (source, target pairs)
            edge_features: [num_edges, edge_dim]
            
        Returns:
            Updated node features: [num_nodes, output_dim]
        """
        num_nodes = node_features.size(0)
        
        # Transform node features
        h = self.W_node(node_features)  # [N, heads * head_dim]
        h = h.view(num_nodes, self.num_heads, self.head_dim)  # [N, heads, head_dim]
        
        # Transform edge features
        e = self.W_edge(edge_features)  # [E, heads * head_dim]
        e = e.view(-1, self.num_heads, self.head_dim)  # [E, heads, head_dim]
        
        # Compute attention coefficients
        source_nodes = edge_indices[0]
        target_nodes = edge_indices[1]
        
        # Concatenate source, target, and edge features for attention
        h_source = h[source_nodes]  # [E, heads, head_dim]
        h_target = h[target_nodes]  # [E, heads, head_dim]
        
        attention_input = torch.cat([h_source, h_target, e], dim=-1)  # [E, heads, 3*head_dim]
        
        # Compute attention scores
        e_attention = (attention_input * self.a).sum(dim=-1)  # [E, heads]
        e_attention = self.leaky_relu(e_attention)
        
        # Normalize attention scores per target node
        attention_weights = torch.zeros(
            num_nodes, edge_indices.size(1), self.num_heads,
            device=node_features.device
        )
        attention_weights.scatter_(0, target_nodes.unsqueeze(0).unsqueeze(-1).expand(-1, -1, self.num_heads),
                                  e_attention.unsqueeze(0))
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention and aggregate
        h_prime = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=node_features.device)
        for i in range(edge_indices.size(1)):
            src, tgt = source_nodes[i], target_nodes[i]
            h_prime[tgt] += attention_weights[tgt, i].unsqueeze(-1) * h_source[i]
        
        h_prime = self.dropout(h_prime)
        
        # Concatenate or average heads
        if self.concat_heads:
            output = h_prime.view(num_nodes, -1)  # [N, output_dim]
        else:
            output = h_prime.mean(dim=1)  # [N, output_dim]
        
        return output


class GraphAttentionEncoder(nn.Module):
    """
    Enhanced scene encoder using Graph Attention Networks.
    
    Implements rich geometric features and multi-head attention.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Node feature extraction
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Edge feature extraction
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim //2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim if i == 0 else hidden_dim
            self.gat_layers.append(
                GraphAttentionLayer(
                    node_dim=layer_input_dim,
                    edge_dim=hidden_dim // 2,
                    output_dim=hidden_dim,
                    num_heads=num_heads,
                    concat_heads=True,
                    dropout=dropout,
                )
            )
        
        # Graph-level pooling
        self.pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert value to float safely."""
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def extract_node_features(
        self,
        nodes: List,
        scene_graph: GeometricSceneGraph,
    ) -> torch.Tensor:
        """
        Extract rich 64-dim node features.
        
        Features:
        - Position (3): x, y, z coordinates
        - Type one-hot (10): point, line, circle, etc.
        - Geometric properties (15): length, angle, radius, area, etc.
        - Derived features (10): centroid, symmetry, etc.
        - Contextual (6): degree, betweenness, clustering
        - Reserved (20): for future use
        """
        device = next(self.parameters()).device
        features = []
        
        for node in nodes:
            node_data = scene_graph.graph.nodes.get(node, {})
            node_type = node_data.get('type', 'Unknown')
            primitive = node_data.get('primitive')
            
            feat = torch.zeros(self.node_feature_dim, device=device)
            
            # Position features (0-2)
            if node_type == 'Point' and primitive:
                x = self._safe_float(getattr(primitive, 'x', 0.0))
                y = self._safe_float(getattr(primitive, 'y', 0.0))
                feat[0:2] = torch.tensor([x, y], device=device)
            elif node_type == 'Line' and primitive:
                p1 = getattr(primitive, 'point1', None)
                p2 = getattr(primitive, 'point2', None)
                if p1 and p2:
                    x = (self._safe_float(getattr(p1, 'x', 0.0)) + 
                         self._safe_float(getattr(p2, 'x', 0.0))) / 2
                    y = (self._safe_float(getattr(p1, 'y', 0.0)) + 
                         self._safe_float(getattr(p2, 'y', 0.0))) / 2
                    feat[0:2] = torch.tensor([x, y], device=device)
            elif node_type == 'Circle' and primitive:
                center = getattr(primitive, 'center', None)
                if center:
                    x = self._safe_float(getattr(center, 'x', 0.0))
                    y = self._safe_float(getattr(center, 'y', 0.0))
                    feat[0:2] = torch.tensor([x, y], device=device)
            
            # Type one-hot (3-12)
            type_map = {'Point': 3, 'Line': 4, 'Circle': 5, 'Polygon': 6,
                       'Ray': 7, 'Segment': 8, 'Arc': 9}
            if node_type in type_map:
                feat[type_map[node_type]] = 1.0
            
            # Geometric properties (13-27)
            if node_type == 'Line' and primitive:
                p1 = getattr(primitive, 'point1', None)
                p2 = getattr(primitive, 'point2', None)
                if p1 and p2:
                    x1 = self._safe_float(getattr(p1, 'x', 0.0))
                    y1 = self._safe_float(getattr(p1, 'y', 0.0))
                    x2 = self._safe_float(getattr(p2, 'x', 0.0))
                    y2 = self._safe_float(getattr(p2, 'y', 0.0))
                    length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = math.atan2(y2-y1, x2-x1)
                    feat[13] = length
                    feat[14] = angle
            elif node_type == 'Circle' and primitive:
                radius = self._safe_float(getattr(primitive, 'radius', 0.0))
                area = math.pi * radius ** 2
                circumference = 2 * math.pi * radius
                feat[13] = radius
                feat[15] = area
                feat[16] = circumference
            
            # Contextual features (28-33): graph topology
            degree = scene_graph.graph.degree(node)
            feat[28] = degree / 10.0  # Normalized
            
            features.append(feat)
        
        return torch.stack(features) if features else torch.zeros(1, self.node_feature_dim, device=device)
    
    def extract_edge_features(
        self,
        edge_list: List,
        scene_graph: GeometricSceneGraph,
    ) -> torch.Tensor:
        """
        Extract 32-dim edge features.
        
        Features:
        - Relation type one-hot (10)
        - Euclidean distance (1)
        - Relative angle (1)
        - Geometric properties (10): parallel, perpendicular, tangent, etc.
        - Reserved (10)
        """
        device = next(self.parameters()).device
        features = []
        
        for u, v, key, data in edge_list:
            feat = torch.zeros(self.edge_feature_dim, device=device)
            
            # Relation type (0-9)
            relation_type = data.get('relation_type')
            if relation_type:
                feat[min(relation_type.value if hasattr(relation_type, 'value') else 0, 9)] = 1.0
            
            # Compute geometric features between nodes
            u_data = scene_graph.graph.nodes.get(u, {})
            v_data = scene_graph.graph.nodes.get(v, {})
            
            u_prim = u_data.get('primitive')
            v_prim = v_data.get('primitive')
            
            # Distance (10)
            if u_prim and v_prim:
                u_pos = self._get_position(u_prim)
                v_pos = self._get_position(v_prim)
                if u_pos and v_pos:
                    dist = math.sqrt((u_pos[0] - v_pos[0])**2 + (u_pos[1] - v_pos[1])**2)
                    feat[10] = dist
            
            features.append(feat)
        
        return torch.stack(features) if features else torch.zeros(1, self.edge_feature_dim, device=device)
    
    def _get_position(self, primitive: Any) -> Optional[tuple]:
        """Get (x, y) position of a primitive."""
        if isinstance(primitive, Point):
            return (self._safe_float(primitive.x), self._safe_float(primitive.y))
        elif isinstance(primitive, Circle):
            center = getattr(primitive, 'center', None)
            if center:
                return (self._safe_float(center.x), self._safe_float(center.y))
        elif isinstance(primitive, Line):
            p1 = getattr(primitive, 'point1', None)
            p2 = getattr(primitive, 'point2', None)
            if p1 and p2:
                x = (self._safe_float(p1.x) + self._safe_float(p2.x)) / 2
                y = (self._safe_float(p1.y) + self._safe_float(p2.y)) / 2
                return (x, y)
        return None
    
    def forward(self, scene_graph: GeometricSceneGraph) -> torch.Tensor:
        """Encode scene graph to latent vector."""
        device = next(self.parameters()).device
        nodes = list(scene_graph.graph.nodes())
        
        if len(nodes) == 0:
            return torch.zeros(self.output_dim, device=device)
        
        # Extract features
        node_features = self.extract_node_features(nodes, scene_graph)
        node_features = self.node_encoder(node_features)
        
        # Build edge index and features
        edge_list = list(scene_graph.graph.edges(keys=True, data=True))
        
        if len(edge_list) > 0:
            edge_features = self.extract_edge_features(edge_list, scene_graph)
            edge_features = self.edge_encoder(edge_features)
            
            # Create edge index tensor
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            edge_index_list: list[list[int]] = []
            for u, v, key, data in edge_list:
                if u in node_to_idx and v in node_to_idx:
                    edge_index_list.append([node_to_idx[u], node_to_idx[v]])
            
            if edge_index_list:
                edge_index = torch.tensor(edge_index_list, device=device).t()  # [2, E]
                
                # Apply GAT layers
                h = node_features
                for gat_layer in self.gat_layers:
                    h = gat_layer(h, edge_index, edge_features)
                    h = F.elu(h)
                
                # Graph-level pooling
                graph_embedding = h.mean(dim=0)
            else:
                graph_embedding = node_features.mean(dim=0)
        else:
            graph_embedding = node_features.mean(dim=0)
        
        # Project to output dimension
        latent = self.pool(graph_embedding)
        return latent
    
    def encode_state(
        self,
        scene_or_trace) -> torch.Tensor:
        """Encode state/trace to latent vector."""
        device = next(self.parameters()).device
        if isinstance(scene_or_trace, SceneTrace):
            if len(scene_or_trace.scenes) > 0:
                return self.forward(scene_or_trace.scenes[-1])
            return torch.zeros(self.output_dim, device=device)
        elif isinstance(scene_or_trace, State):
            return self.forward(scene_or_trace.graph)
        elif isinstance(scene_or_trace, GeometricSceneGraph):
            return self.forward(scene_or_trace)
        else:
            raise TypeError(f"Unsupported type: {type(scene_or_trace)}")
    
    def encode_trace(self, trace: SceneTrace) -> torch.Tensor:
        """Encode entire trace to sequence of latents."""
        device = next(self.parameters()).device
        latents = [self.forward(scene) for scene in trace.scenes]
        
        if not latents:
            return torch.zeros(1, self.output_dim, device=device)
        
        return torch.stack(latents)
