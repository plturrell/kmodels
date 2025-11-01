"""Graph Neural Network models for the NFL Big Data Bowl competition."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATPlayerTracker(torch.nn.Module):
    """A Graph Attention Network to model player interactions and predict landing spot."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=n_heads, concat=True)
        self.conv2 = GATv2Conv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=True)
        self.conv3 = GATv2Conv(hidden_dim * n_heads, hidden_dim, heads=1, concat=False)

        self.output_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.elu(x)

        # Aggregate node embeddings to get a graph-level representation
        graph_embedding = global_mean_pool(x, batch)

        # Final prediction
        output = self.output_mlp(graph_embedding)
        return output
