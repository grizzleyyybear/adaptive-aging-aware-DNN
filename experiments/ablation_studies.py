from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

from features.feature_builder import FeatureBuilder
from graph.accelerator_graph import AcceleratorGraph
from graph.graph_dataset import AgingDataset
from models.hybrid_gnn_transformer import HybridGNNTransformer
from simulator.timeloop_runner import TimeloopRunner
from simulator.workload_runner import WorkloadRunner
from utils.device import dataloader_kwargs
from utils.runtime_eval import cfg_get, compute_predictor_ttf, get_workload_names, load_pretrained_predictor, simulate_mapping


ABLATION_CONFIGS = [
    ("GCN Only", ("gcn",)),
    ("GCN + GAT", ("gcn", "gat")),
    ("GCN + Transformer", ("gcn", "transformer")),
    ("Full Model", ("gcn", "gat", "transformer")),
]


def _evaluate_predictor(model: HybridGNNTransformer, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    preds = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view_as(batch.y)
            preds.append(pred.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0).reshape(-1)
    label_arr = np.concatenate(labels, axis=0).reshape(-1)
    return float(r2_score(label_arr, pred_arr)), float(mean_absolute_error(label_arr, pred_arr))


def _component_model(cfg: Any, components: tuple[str, ...], device: torch.device, state_dict: dict[str, Any]) -> HybridGNNTransformer:
    model_cfg = cfg_get(cfg, "model", cfg)
    model = HybridGNNTransformer(
        node_feature_dim=8,
        hidden_dim=int(cfg_get(model_cfg, "hidden_dim", 256)),
        gat_heads=int(cfg_get(model_cfg, "gat_heads", 4)),
        transformer_layers=int(cfg_get(model_cfg, "transformer_layers", 2)),
        transformer_heads=int(cfg_get(model_cfg, "transformer_heads", 4)),
        seq_len=1,
        components=components,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def run_ablation_studies(cfg, all_components: dict) -> pd.DataFrame:
    predictor, device, checkpoint = load_pretrained_predictor(cfg)
    state_dict = predictor.state_dict()

    dataset_root = str(cfg_get(cfg_get(cfg, "dataset", {}), "root", "data"))
    test_size = int(cfg_get(cfg_get(cfg, "ablation", {}), "test_size", 5000))
    test_ds = AgingDataset(root=dataset_root, split="test", size=test_size, cfg=cfg, seed=int(cfg_get(cfg, "seed", 44)))
    loader = DataLoader(
        test_ds,
        batch_size=int(cfg_get(cfg_get(cfg, "training", {}), "batch_size", 64)),
        shuffle=False,
        **dataloader_kwargs(device),
    )

    accel_cfg = cfg_get(cfg, "accelerator", cfg)
    simulator = TimeloopRunner(accel_cfg)
    graph = AcceleratorGraph(accel_cfg)
    graph.build()
    feature_builder = FeatureBuilder(accel_cfg)
    workload_runner = WorkloadRunner(cfg_get(cfg, "workloads", None))
    workloads = get_workload_names(cfg, workload_runner)
    failure_threshold = float(cfg_get(cfg_get(cfg, "planning", {}), "failure_threshold", 0.8))
    num_clusters = int(cfg_get(accel_cfg, "mac_clusters", cfg_get(accel_cfg, "num_mac_clusters", 16)))

    rows = []
    mean_peaks = []
    for config_name, components in ABLATION_CONFIGS:
        model = _component_model(cfg, components, device, state_dict)
        r2, mae = _evaluate_predictor(model, loader, device)

        peaks = []
        ttfs = []
        for workload_name in workloads:
            layers = workload_runner.get_workload_layers(workload_name)
            mapping = np.arange(len(layers), dtype=np.int32) % max(num_clusters, 1)
            metrics = simulate_mapping(
                simulator=simulator,
                feature_builder=feature_builder,
                graph=graph,
                layers=layers,
                mapping=mapping,
                workload_name=workload_name,
                predictor=model,
                device=device,
            )
            peaks.append(metrics["peak_aging"])
            ttfs.append(
                compute_predictor_ttf(
                    simulator=simulator,
                    feature_builder=feature_builder,
                    graph=graph,
                    predictor=model,
                    layers=layers,
                    mapping=mapping,
                    workload_name=workload_name,
                    failure_threshold=failure_threshold,
                    device=device,
                )
            )

        mean_peak = float(np.mean(peaks)) if peaks else 0.0
        mean_peaks.append(mean_peak)
        rows.append(
            {
                "Config": config_name,
                "R2": float(r2),
                "MAE": float(mae),
                "TTF (Yrs)": float(np.mean(ttfs)) if ttfs else 0.0,
                "Mean Peak Aging": mean_peak,
            }
        )

    baseline_peak = max(mean_peaks[0], 1e-6) if mean_peaks else 1e-6
    for row in rows:
        row["Peak Reduction"] = float(100.0 * (baseline_peak - row["Mean Peak Aging"]) / baseline_peak)
        row.pop("Mean Peak Aging", None)

    df = pd.DataFrame(rows, columns=["Config", "R2", "MAE", "TTF (Yrs)", "Peak Reduction"])
    output_dir = Path(cfg_get(cfg, "output_dir", "outputs")) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "ablation_results.csv", index=False)
    return df
