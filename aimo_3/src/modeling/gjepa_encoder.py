from __future__ import annotations

"""GeometryStateEncoder: encodes GeometryScene / GeometryTrace into a latent vector.

This is a *first concrete draft* of the encoder for the proposed G-JEPA module.
It is intentionally simple and self-contained:

- It consumes the existing Pydantic geometry schema in aimo_3/src/data/geometry_schema.py.
- It performs vanilla message passing over the scene graph using plain PyTorch.
- It exposes a single public method, `encode_state`, which accepts either a
  GeometryScene or a GeometryTrace and returns a fixed-dimensional embedding.

The goal is to provide a clear, hackable starting point that we can later extend
with richer features (angles, numeric props, theorem history, etc.) and plug
into a JEPA-style predictor.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from ..data.geometry_schema import GeoEdge, GeoNode, GeometryScene, GeometryTrace


class _MessagePassingLayer(nn.Module):
    """Single message-passing block over a simple directed graph.

    For each edge i -> j with relation type r, we compute a message using the
    source and target node states and a learned relation embedding, aggregate
    messages at each destination node, and apply a residual MLP update.
    """

    def __init__(self, hidden_dim: int, rel_emb_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + rel_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_states: torch.Tensor,  # [N, D]
        edge_index: torch.Tensor,  # [2, E] (src, dst)
        edge_rel_emb: torch.Tensor,  # [E, R]
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            # No edges: return states unchanged.
            return node_states

        src, dst = edge_index
        src_h = node_states[src]
        dst_h = node_states[dst]

        m_input = torch.cat([src_h, dst_h, edge_rel_emb], dim=-1)
        messages = self.edge_mlp(m_input)  # [E, D]

        # Aggregate messages by destination node (sum).
        agg = torch.zeros_like(node_states)
        agg.index_add_(0, dst, messages)

        updated = node_states + self.node_mlp(agg)
        return self.norm(updated)


class GeometryStateEncoder(nn.Module):
    """Encode a GeometryScene / GeometryTrace into a fixed-size latent vector.

    This class is deliberately lightweight:

    - Nodes are embedded from their (string) `type` and optional `label`.
    - Edges are embedded from their `relation` string.
    - We run several rounds of message passing, then mean-pool over nodes.

    Later we can enrich this with numeric features (known lengths/angles),
    theorem history, or attention pooling. For now, it gives a stable
    implementation that can be called from a G-JEPA training loop.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        rel_emb_dim: int = 64,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # String-to-index vocabularies are stored as Python dicts on the module;
        # they are grown dynamically the first time new types/relations appear.
        self._node_type_to_id: Dict[str, int] = {}
        self._label_to_id: Dict[str, int] = {}
        self._rel_to_id: Dict[str, int] = {}

        # Embeddings (we keep them relatively small and grow vocab as needed).
        self.node_type_emb = nn.Embedding(32, hidden_dim)
        self.label_emb = nn.Embedding(64, hidden_dim)
        self.rel_emb = nn.Embedding(32, rel_emb_dim)

        # Initial node projection combines type + label embeddings.
        self.node_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList(
            [_MessagePassingLayer(hidden_dim, rel_emb_dim) for _ in range(num_layers)]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_state(
        self,
        scene_or_trace: GeometryScene | GeometryTrace,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode a scene or trace into a single latent vector [D].

        If a GeometryTrace is passed, we encode its *finalScene*.
        """

        if isinstance(scene_or_trace, GeometryTrace):
            scene = scene_or_trace.finalScene
        else:
            scene = scene_or_trace

        return self._encode_scene(scene, device=device)

    @torch.no_grad()
    def encode_trace(
        self,
        trace: GeometryTrace,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode an entire GeometryTrace into a sequence of latent vectors.

        We reconstruct the intermediate scenes after each theorem application
        (S_0, S_1, ..., S_T) and encode each into a latent vector. The result
        has shape [T+1, D], where T = len(trace.theoremApplications).
        """

        scenes = build_scene_sequence(trace)
        if not scenes:
            return torch.zeros((0, self.hidden_dim), device=device)

        vecs: List[torch.Tensor] = []
        for scene in scenes:
            vecs.append(self._encode_scene(scene, device=device))

        return torch.stack(vecs, dim=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_or_add(self, table: Dict[str, int], key: str) -> int:
        idx = table.get(key)
        if idx is None:
            idx = len(table)
            table[key] = idx
        return idx

    def _build_node_states(self, scene: GeometryScene) -> torch.Tensor:
        node_type_ids: List[int] = []
        label_ids: List[int] = []

        for node in scene.nodes:
            t_id = self._get_or_add(self._node_type_to_id, node.type)
            node_type_ids.append(t_id)

            if node.label is not None:
                l_id = self._get_or_add(self._label_to_id, node.label)
            else:
                l_id = 0  # shared "null" label bucket
            label_ids.append(l_id)

        if not node_type_ids:
            return torch.zeros((0, self.hidden_dim))

        node_type_ids_t = torch.tensor(node_type_ids, dtype=torch.long)
        label_ids_t = torch.tensor(label_ids, dtype=torch.long)

        type_emb = self.node_type_emb(node_type_ids_t)
        label_emb = self.label_emb(label_ids_t)
        node_feat = torch.cat([type_emb, label_emb], dim=-1)
        return self.node_proj(node_feat)

    def _build_edge_tensors(self, scene: GeometryScene) -> Tuple[torch.Tensor, torch.Tensor]:
        if not scene.edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            rel_ids_t = torch.zeros((0,), dtype=torch.long)
            return edge_index, rel_ids_t

        src_ids: List[int] = []
        dst_ids: List[int] = []
        rel_ids: List[int] = []

        # Map node ids to integer indices in the node list.
        node_index: Dict[str, int] = {n.id: i for i, n in enumerate(scene.nodes)}

        for edge in scene.edges:
            if edge.source not in node_index or edge.target not in node_index:
                continue
            src_ids.append(node_index[edge.source])
            dst_ids.append(node_index[edge.target])
            rel_ids.append(self._get_or_add(self._rel_to_id, edge.relation))

        if not src_ids:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            rel_ids_t = torch.zeros((0,), dtype=torch.long)
            return edge_index, rel_ids_t

        edge_index = torch.stack(
            [torch.tensor(src_ids, dtype=torch.long), torch.tensor(dst_ids, dtype=torch.long)],
            dim=0,
        )
        rel_ids_t = torch.tensor(rel_ids, dtype=torch.long)
        return edge_index, rel_ids_t

    def _encode_scene(
        self,
        scene: GeometryScene,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Internal helper: encode a single GeometryScene to [D]."""

        edge_index, edge_rel_ids = self._build_edge_tensors(scene)
        node_states = self._build_node_states(scene)

        if device is not None:
            node_states = node_states.to(device)
            edge_index = edge_index.to(device)
            edge_rel_ids = edge_rel_ids.to(device)

        if edge_rel_ids.numel() > 0:
            edge_rel_emb = self.rel_emb(edge_rel_ids)
        else:
            edge_rel_emb = torch.zeros(
                (0, self.rel_emb.embedding_dim), device=node_states.device
            )

        for layer in self.layers:
            node_states = layer(node_states, edge_index, edge_rel_emb)

        if node_states.shape[0] == 0:
            return torch.zeros(self.hidden_dim, device=node_states.device)
        return node_states.mean(dim=0)


def build_scene_sequence(trace: GeometryTrace) -> List[GeometryScene]:
    """Reconstruct the sequence of scenes S_0, ..., S_T from a GeometryTrace.

    - S_0 is trace.initialScene.
    - Each subsequent S_t applies the addedNodes/addedEdges of the t-th
      theorem application on top of S_{t-1}.
    """

    scenes: List[GeometryScene] = []
    cur_nodes = list(trace.initialScene.nodes)
    cur_edges = list(trace.initialScene.edges)
    scenes.append(GeometryScene(nodes=list(cur_nodes), edges=list(cur_edges)))

    for app in trace.theoremApplications:
        if app.addedNodes:
            cur_nodes.extend(app.addedNodes)
        if app.addedEdges:
            cur_edges.extend(app.addedEdges)
        scenes.append(GeometryScene(nodes=list(cur_nodes), edges=list(cur_edges)))

    return scenes

