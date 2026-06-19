import numpy as np
from omegaconf import OmegaConf

from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.training_pipeline import TrainingPipeline
from graph.graph_dataset import AgingDataset


def _build_cfg(epochs=1, horizon=4):
    return OmegaConf.create({
        "seed": 7,
        "accelerator": {
            "pe_array": [4, 4],
            "pe_array_rows": 4,
            "pe_array_cols": 4,
            "mac_clusters": 16,
            "sram_banks": 8,
            "noc_routers": 4,
            "num_layers": 4,
        },
        "model": {"hidden_dim": 32, "gat_heads": 2, "transformer_layers": 1,
                   "transformer_heads": 2, "prediction_horizon": horizon},
        "training": {"epochs": epochs, "batch_size": 8, "learning_rate": 1e-3,
                      "patience": 2, "weight_decay": 1e-5, "low_vram_batch_size": 8},
        "workloads": [],
        "runtime": {"device": "cpu"},
    })


def test_predictor_extended_metrics(tmp_path):
    cfg = _build_cfg()
    dataset = AgingDataset(root=str(tmp_path), split="train", size=24, cfg=cfg, seed=7)
    sample = dataset[0]
    model = HybridGNNTransformer(
        node_feature_dim=int(sample.x.shape[1]),
        hidden_dim=32, gat_heads=2,
        transformer_layers=1, transformer_heads=2, seq_len=cfg.model.prediction_horizon,
    )
    metrics = TrainingPipeline(cfg, model, dataset, checkpoint_dir=tmp_path / "ckpts").train()

    assert "spearman" in metrics
    assert -1.0 <= metrics["spearman"] <= 1.0
    assert "baseline_mean_mae" in metrics
    assert "skill_vs_mean" in metrics


def test_trajectory_per_step_metrics(tmp_path):
    cfg = _build_cfg(horizon=4)
    dataset = AgingDataset(root=str(tmp_path), split="train", size=24, cfg=cfg, seed=11)
    sample = dataset[0]
    from models.trajectory_predictor import TrajectoryPredictor
    base = HybridGNNTransformer(
        node_feature_dim=int(sample.x.shape[1]),
        hidden_dim=32, gat_heads=2,
        transformer_layers=1, transformer_heads=2, seq_len=cfg.model.prediction_horizon,
    )
    traj = TrajectoryPredictor(gnn_encoder=base, hidden_dim=32, horizon=cfg.model.prediction_horizon, gamma=0.95)
    metrics = TrainingPipeline(cfg, traj, dataset, checkpoint_dir=tmp_path / "ckpts").train()

    assert "per_step_mae" in metrics
    assert len(metrics["per_step_mae"]) == cfg.model.prediction_horizon
    assert len(metrics["per_step_r2"]) == cfg.model.prediction_horizon
    assert "baseline_persistence_mae" in metrics
