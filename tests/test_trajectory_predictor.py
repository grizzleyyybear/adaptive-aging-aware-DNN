import pytest
import torch

from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor


def test_trajectory_predictor_shapes():
    encoder = HybridGNNTransformer(
        node_feature_dim=8,
        hidden_dim=32,
        gat_heads=4,
        transformer_heads=4,
        transformer_layers=1,
        seq_len=4,
    )
    model = TrajectoryPredictor(gnn_encoder=encoder, hidden_dim=32, horizon=4)

    x = torch.rand(6, 8)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 0, 2], [1, 2, 3, 4, 5, 0, 2, 4]],
        dtype=torch.long,
    )
    edge_attr = torch.rand(edge_index.shape[1], 2)
    batch = torch.zeros(6, dtype=torch.long)

    pred = model(x, edge_index, edge_attr=edge_attr, batch=batch)
    target = torch.rand(6, 4)
    loss = model.trajectory_loss(pred, target)

    assert pred.shape == (6, 4)
    assert torch.all((pred >= 0.0) & (pred <= 1.0))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
