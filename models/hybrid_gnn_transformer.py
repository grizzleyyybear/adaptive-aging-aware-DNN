import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class HybridGNNTransformer(nn.Module):
    """
    Hybrid Architecture capturing spatial hardware topology (GNN)
    and temporal/context interactions (Transformer).
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        gnn_layers: int = 3,
        gat_heads: int = 4,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
        seq_len: int = 10,
        components: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.components = set(components or ("gcn", "gat", "transformer"))

        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(max(gnn_layers, 1))]
        )
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(max(gnn_layers, 1))]
        )

        self.gat = GATConv(hidden_dim, hidden_dim // gat_heads, heads=gat_heads)
        self.gat_bn = nn.BatchNorm1d(hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=max(seq_len, 512))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def encode_graph(self, x, edge_index, edge_attr=None, batch=None):
        h = F.relu(self.input_proj(x))

        if "gcn" in self.components:
            for gcn, bn in zip(self.gcn_layers, self.bn_layers):
                h_new = F.relu(bn(gcn(h, edge_index)))
                h = h + h_new

        if "gat" in self.components:
            try:
                if edge_attr is not None and edge_attr.shape[0] > 0:
                    h_gat = self.gat(h, edge_index, edge_attr=edge_attr)
                else:
                    h_gat = self.gat(h, edge_index)
            except TypeError:
                h_gat = self.gat(h, edge_index)
            h = self.gat_bn(h_gat) + h

        if "transformer" in self.components:
            if batch is None:
                dense_h = h.unsqueeze(0)
                mask = torch.ones((1, h.size(0)), dtype=torch.bool, device=h.device)
            else:
                dense_h, mask = to_dense_batch(h, batch)

            dense_h = self.pos_encoding(dense_h)
            transformed = self.transformer(dense_h, src_key_padding_mask=~mask)
            h = h + transformed[mask]

        return h

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass. Returns [N, 1] aging predictions.
        Compatible with single graphs and PyG Batch objects.
        """
        h = self.encode_graph(x, edge_index, edge_attr, batch)
        return self.head(h)
