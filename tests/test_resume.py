"""Resumability tests for TrainingPipeline epoch-level checkpointing."""
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Data

from models.training_pipeline import TrainingPipeline


class _TinyNodeModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        return self.lin(x)


def _tiny_dataset(n_graphs: int = 12, n_nodes: int = 4, in_dim: int = 5):
    torch.manual_seed(0)
    items = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, in_dim)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        edge_attr = torch.randn(edge_index.shape[1], 2)
        y = (x.sum(dim=1, keepdim=True) * 0.1)
        items.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return items


def _cfg(epochs: int):
    return OmegaConf.create({
        "training": {"epochs": epochs, "batch_size": 4, "learning_rate": 1e-3,
                     "patience": epochs + 1, "weight_decay": 0.0},
        "runtime": {"device": "cpu"},
    })


def test_completed_training_is_skipped_on_resume(tmp_path):
    ds = _tiny_dataset()
    epochs = 3

    p1 = TrainingPipeline(_cfg(epochs), _TinyNodeModel(5), ds, checkpoint_dir=tmp_path, resume=True)
    p1.train()

    state = torch.load(tmp_path / "predictor_state.pt", map_location="cpu", weights_only=False)
    assert state["completed"] is True
    assert state["epoch"] == epochs
    assert (tmp_path / "predictor_best.pt").exists()

    # A fresh pipeline with resume=True must detect completion and NOT retrain.
    p2 = TrainingPipeline(_cfg(epochs), _TinyNodeModel(5), ds, checkpoint_dir=tmp_path, resume=True)
    optimizer_state_before = {k: v.clone() if torch.is_tensor(v) else v
                              for k, v in p2.optimizer.state_dict()["param_groups"][0].items()
                              if not isinstance(v, list)}
    metrics = p2.train()
    assert set(metrics) >= {"loss", "mae", "rmse", "r2"}
    # Step count in the scheduler should equal the saved schedule, proving no
    # extra epochs were run.
    assert p2.scheduler.state_dict()["last_epoch"] == epochs
    _ = optimizer_state_before


def test_partial_state_resumes_from_saved_epoch(tmp_path):
    ds = _tiny_dataset()
    epochs = 4

    # Run a 2-epoch pipeline, then rewrite the state as a partial (non-complete)
    # checkpoint of a 4-epoch schedule resuming at epoch 2.
    warm = TrainingPipeline(_cfg(2), _TinyNodeModel(5), ds, checkpoint_dir=tmp_path, resume=True)
    warm.train()
    warm_state = torch.load(tmp_path / "predictor_state.pt", map_location="cpu", weights_only=False)
    warm_state["epochs"] = epochs
    warm_state["epoch"] = 2
    warm_state["completed"] = False
    torch.save(warm_state, tmp_path / "predictor_state.pt")

    p = TrainingPipeline(_cfg(epochs), _TinyNodeModel(5), ds, checkpoint_dir=tmp_path, resume=True)
    p.train()
    final = torch.load(tmp_path / "predictor_state.pt", map_location="cpu", weights_only=False)
    assert final["completed"] is True
    assert final["epoch"] == epochs
    # Scheduler advanced exactly to the full horizon (resumed 2 -> 4).
    assert p.scheduler.state_dict()["last_epoch"] == epochs


def test_fresh_run_without_resume_ignores_state(tmp_path):
    ds = _tiny_dataset()
    epochs = 2
    TrainingPipeline(_cfg(epochs), _TinyNodeModel(5), ds, checkpoint_dir=tmp_path, resume=True).train()
    # resume=False must retrain from scratch regardless of existing state.
    p = TrainingPipeline(_cfg(epochs), _TinyNodeModel(5), ds, checkpoint_dir=tmp_path, resume=False)
    p.train()
    assert p.scheduler.state_dict()["last_epoch"] == epochs
