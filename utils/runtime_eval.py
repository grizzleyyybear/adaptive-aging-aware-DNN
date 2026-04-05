from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from aging_models.aging_label_generator import AgingLabelGenerator
from features.feature_builder import FeatureBuilder
from graph.accelerator_graph import AcceleratorGraph
from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor
from simulator.workload_runner import WorkloadRunner
from utils.device import configure_torch_runtime, resolve_device


REFERENCE_STRESS_TIME_S = 360_000.0
MAX_TTF_TIME_S = 10_000_000.0


def cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    if hasattr(container, key):
        return getattr(container, key)
    try:
        return container.get(key, default)
    except AttributeError:
        return default


def normalize_mapping(mapping: Any, num_layers: int, num_clusters: int) -> np.ndarray:
    num_layers = max(int(num_layers), 1)
    num_clusters = max(int(num_clusters), 1)

    if mapping is None:
        return np.arange(num_layers, dtype=np.int32) % num_clusters

    mapping_arr = np.asarray(mapping, dtype=np.int32).reshape(-1)
    if mapping_arr.size == 0:
        return np.zeros(num_layers, dtype=np.int32)

    if mapping_arr.size < num_layers:
        padding = np.arange(mapping_arr.size, num_layers, dtype=np.int32) % num_clusters
        mapping_arr = np.concatenate([mapping_arr, padding], axis=0)

    return np.clip(mapping_arr[:num_layers], 0, num_clusters - 1).astype(np.int32)


def activity_dict_from_result(result: Any) -> dict[str, np.ndarray]:
    return {
        "switching_activity": np.asarray(result.switching_activity, dtype=np.float32),
        "mac_utilization": np.asarray(result.mac_utilization, dtype=np.float32),
        "sram_access_rate": np.asarray(result.sram_access_rate, dtype=np.float32),
        "noc_traffic": np.asarray(result.noc_traffic, dtype=np.float32),
    }


def build_graph(accel_cfg: Any) -> AcceleratorGraph:
    graph = AcceleratorGraph(accel_cfg)
    graph.build()
    return graph


def _num_clusters_from_simulator(simulator: Any) -> int:
    cfg = getattr(simulator, "cfg", {})
    return max(
        int(
            cfg_get(
                cfg,
                "mac_clusters",
                cfg_get(cfg, "num_mac_clusters", getattr(simulator, "num_mac_clusters", 1)),
            )
        ),
        1,
    )


def get_workload_names(cfg: Any | None = None, workload_runner: WorkloadRunner | None = None) -> list[str]:
    if workload_runner is not None and getattr(workload_runner, "available_workloads", None):
        return list(workload_runner.available_workloads)

    workloads_cfg = cfg_get(cfg, "workloads", [])
    names: list[str] = []
    if isinstance(workloads_cfg, Iterable) and not isinstance(workloads_cfg, (str, bytes, dict)):
        for item in workloads_cfg:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = cfg_get(item, "name", None)
            if name:
                names.append(str(name))

    return names or WorkloadRunner(None).available_workloads


def get_model_device(model: torch.nn.Module | None) -> torch.device:
    if model is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_runtime_device(request: str | None = None) -> torch.device:
    device = resolve_device(request or "auto")
    configure_torch_runtime(device)
    return device


def build_node_features(
    feature_builder: FeatureBuilder,
    result: Any,
    workload_name: str,
    stress_time_s: float,
) -> torch.Tensor:
    return feature_builder.build_node_features(
        activity_dict=activity_dict_from_result(result),
        workload_name=workload_name,
        latency=result.total_latency_cycles,
        energy=result.total_energy_pj,
        stress_time_s=stress_time_s,
    )


def _graph_to_device(graph_data: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    edge_attr = graph_data.edge_attr
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    batch = getattr(graph_data, "batch", None)
    if batch is not None:
        batch = batch.to(device)
    return graph_data.x.to(device), graph_data.edge_index.to(device), edge_attr, batch


def run_predictor_inference(
    predictor: HybridGNNTransformer,
    graph: AcceleratorGraph,
    node_features: torch.Tensor,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or get_model_device(predictor)
    graph_data = graph.to_pyg(node_features)
    x, edge_index, edge_attr, batch = _graph_to_device(graph_data, device)
    predictor.eval()
    with torch.no_grad():
        pred = predictor(x, edge_index, edge_attr, batch)
    return pred.detach().cpu().view(-1).numpy()


def run_trajectory_inference(
    trajectory_predictor: TrajectoryPredictor,
    graph: AcceleratorGraph,
    node_features: torch.Tensor,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or get_model_device(trajectory_predictor)
    graph_data = graph.to_pyg(node_features)
    x, edge_index, edge_attr, batch = _graph_to_device(graph_data, device)
    trajectory_predictor.eval()
    with torch.no_grad():
        pred = trajectory_predictor(x, edge_index, edge_attr, batch)
    return pred.detach().cpu().numpy()


def simulate_mapping(
    simulator: Any,
    feature_builder: FeatureBuilder,
    graph: AcceleratorGraph,
    layers: Sequence[dict[str, Any]],
    mapping: Any,
    workload_name: str,
    stress_time_s: float = REFERENCE_STRESS_TIME_S,
    predictor: HybridGNNTransformer | None = None,
    trajectory_predictor: TrajectoryPredictor | None = None,
    aging_generator: AgingLabelGenerator | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    mapping_arr = normalize_mapping(mapping, len(layers), _num_clusters_from_simulator(simulator))
    result = simulator.run_workload(list(layers), mapping_arr)
    activity = activity_dict_from_result(result)
    node_features = build_node_features(feature_builder, result, workload_name, stress_time_s)

    aging_scores: np.ndarray | None = None
    if predictor is not None:
        try:
            aging_scores = run_predictor_inference(predictor, graph, node_features, device=device)
        except Exception:
            aging_scores = None

    if aging_scores is None and aging_generator is not None:
        aging_scores = aging_generator.compute_aging_score(activity, stress_time_s)

    if aging_scores is None:
        aging_scores = np.zeros(node_features.shape[0], dtype=np.float32)

    trajectory_scores: np.ndarray | None = None
    if trajectory_predictor is not None:
        try:
            trajectory_scores = run_trajectory_inference(trajectory_predictor, graph, node_features, device=device)
        except Exception:
            trajectory_scores = None

    return {
        "mapping": mapping_arr,
        "result": result,
        "activity": activity,
        "node_features": node_features,
        "aging_scores": np.asarray(aging_scores, dtype=np.float32),
        "trajectory_scores": None if trajectory_scores is None else np.asarray(trajectory_scores, dtype=np.float32),
        "peak_aging": float(np.max(aging_scores)) if len(aging_scores) else 0.0,
        "aging_variance": float(np.var(aging_scores)) if len(aging_scores) else 0.0,
        "latency_cycles": float(result.total_latency_cycles),
        "energy_pj": float(result.total_energy_pj),
        "latency_norm": float(result.total_latency_cycles / 1e8),
        "energy_norm": float(result.total_energy_pj / 1e9),
        "stress_time_s": float(stress_time_s),
        "workload_name": workload_name,
    }


def compute_physics_ttf(
    simulator: Any,
    aging_generator: AgingLabelGenerator,
    layers: Sequence[dict[str, Any]],
    mapping: Any,
    failure_threshold: float = 0.8,
    max_time_s: float = MAX_TTF_TIME_S,
    n_iter: int = 30,
) -> float:
    mapping_arr = normalize_mapping(mapping, len(layers), _num_clusters_from_simulator(simulator))
    result = simulator.run_workload(list(layers), mapping_arr)
    activity = activity_dict_from_result(result)

    lo = 0.0
    hi = float(max_time_s)
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        peak = float(np.max(aging_generator.compute_aging_score(activity, mid)))
        if peak >= failure_threshold:
            hi = mid
        else:
            lo = mid

    return lo / 3600.0 / 8760.0


def compute_predictor_ttf(
    simulator: Any,
    feature_builder: FeatureBuilder,
    graph: AcceleratorGraph,
    predictor: HybridGNNTransformer,
    layers: Sequence[dict[str, Any]],
    mapping: Any,
    workload_name: str,
    failure_threshold: float = 0.8,
    max_time_s: float = MAX_TTF_TIME_S,
    n_iter: int = 30,
    device: torch.device | None = None,
) -> float:
    lo = 0.0
    hi = float(max_time_s)
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        metrics = simulate_mapping(
            simulator=simulator,
            feature_builder=feature_builder,
            graph=graph,
            layers=layers,
            mapping=mapping,
            workload_name=workload_name,
            stress_time_s=mid,
            predictor=predictor,
            device=device,
        )
        if metrics["peak_aging"] >= failure_threshold:
            hi = mid
        else:
            lo = mid

    return lo / 3600.0 / 8760.0


def find_existing_checkpoint(paths: Sequence[str | Path]) -> Path | None:
    for path in paths:
        if path is None:
            continue
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def load_pretrained_predictor(
    cfg: Any,
    checkpoint_candidates: Sequence[str | Path] | None = None,
    device_request: str | None = None,
) -> tuple[HybridGNNTransformer, torch.device, Path]:
    device = resolve_runtime_device(device_request)
    model_cfg = cfg_get(cfg, "model", cfg)
    checkpoint = find_existing_checkpoint(
        checkpoint_candidates
        or [
            "outputs/models/hybrid_gnn_transformer.pt",
            "outputs/best_predictor.pt",
            "checkpoints/predictor_best.pt",
        ]
    )
    if checkpoint is None:
        raise FileNotFoundError("Could not find a predictor checkpoint in the expected output paths.")

    predictor = HybridGNNTransformer(
        node_feature_dim=8,
        hidden_dim=int(cfg_get(model_cfg, "hidden_dim", 256)),
        gat_heads=int(cfg_get(model_cfg, "gat_heads", 4)),
        transformer_layers=int(cfg_get(model_cfg, "transformer_layers", 2)),
        transformer_heads=int(cfg_get(model_cfg, "transformer_heads", 4)),
        seq_len=1,
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    predictor.load_state_dict(state_dict)
    predictor.eval()
    return predictor, device, checkpoint


def load_pretrained_trajectory(
    cfg: Any,
    predictor: HybridGNNTransformer,
    checkpoint_candidates: Sequence[str | Path] | None = None,
    device: torch.device | None = None,
) -> tuple[TrajectoryPredictor, Path]:
    device = device or get_model_device(predictor)
    model_cfg = cfg_get(cfg, "model", cfg)
    checkpoint = find_existing_checkpoint(
        checkpoint_candidates
        or [
            "outputs/models/trajectory_predictor.pt",
            "outputs/best_trajectory.pt",
            "checkpoints/trajectory_best.pt",
        ]
    )
    if checkpoint is None:
        raise FileNotFoundError("Could not find a trajectory checkpoint in the expected output paths.")

    encoder = copy.deepcopy(predictor).to(device)
    trajectory = TrajectoryPredictor(
        gnn_encoder=encoder,
        hidden_dim=int(cfg_get(model_cfg, "hidden_dim", 256)),
        horizon=int(cfg_get(model_cfg, "prediction_horizon", 10)),
        gamma=float(cfg_get(cfg_get(cfg, "training", {}), "discount_factor", 0.95)),
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    trajectory.load_state_dict(state_dict)
    trajectory.eval()
    return trajectory, checkpoint
