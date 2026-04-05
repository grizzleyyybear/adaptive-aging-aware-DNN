from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from aging_models.aging_label_generator import AgingLabelGenerator
from simulator.workload_runner import WorkloadRunner
from utils.runtime_eval import REFERENCE_STRESS_TIME_S, cfg_get, compute_physics_ttf, normalize_mapping


@dataclass
class BaselineResult:
    name: str
    ttf: float
    peak_aging: float
    latency: float
    energy: float
    per_workload: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ttf_years": float(self.ttf),
            "peak_aging": float(self.peak_aging),
            "latency_cycles": float(self.latency),
            "energy_pj": float(self.energy),
            "per_workload": self.per_workload,
        }


def _workload_names(cfg: Any, workload_stream: list[str] | None) -> list[str]:
    if workload_stream:
        return list(workload_stream)
    runner = WorkloadRunner(cfg_get(cfg, "workloads", None))
    return list(runner.available_workloads)


def _layer_load(layer: dict[str, Any]) -> float:
    layer_type = str(layer.get("type", "conv2d")).lower()
    if layer_type == "matmul":
        return float(layer.get("M", 1) * layer.get("K", 1) * layer.get("N", 1))
    return float(
        layer.get("N", 1)
        * layer.get("C", 1)
        * layer.get("K", 1)
        * layer.get("R", 1)
        * layer.get("S", 1)
        * layer.get("P", 1)
        * layer.get("Q", 1)
    )


def _evaluate_mapping(
    simulator: Any,
    aging_gen: AgingLabelGenerator,
    workload_name: str,
    layers: list[dict[str, Any]],
    mapping: np.ndarray,
    failure_threshold: float,
) -> dict[str, float]:
    mapping = normalize_mapping(mapping, len(layers), getattr(simulator, "num_mac_clusters", 1))
    result = simulator.run_workload(layers, mapping)
    activity = {
        "switching_activity": result.switching_activity,
        "mac_utilization": result.mac_utilization,
        "sram_access_rate": result.sram_access_rate,
        "noc_traffic": result.noc_traffic,
    }
    aging_scores = aging_gen.compute_aging_score(activity, REFERENCE_STRESS_TIME_S)
    ttf_years = compute_physics_ttf(
        simulator=simulator,
        aging_generator=aging_gen,
        layers=layers,
        mapping=mapping,
        failure_threshold=failure_threshold,
    )
    return {
        "peak_aging": float(np.max(aging_scores)),
        "latency_cycles": float(result.total_latency_cycles),
        "energy_pj": float(result.total_energy_pj),
        "ttf_years": float(ttf_years),
        "mapping": mapping.astype(int).tolist(),
    }


def _aggregate_result(name: str, metrics_by_workload: dict[str, dict[str, float]]) -> BaselineResult:
    peaks = [item["peak_aging"] for item in metrics_by_workload.values()]
    latencies = [item["latency_cycles"] for item in metrics_by_workload.values()]
    energies = [item["energy_pj"] for item in metrics_by_workload.values()]
    ttfs = [item["ttf_years"] for item in metrics_by_workload.values()]

    return BaselineResult(
        name=name,
        ttf=float(np.mean(ttfs)) if ttfs else 0.0,
        peak_aging=float(np.mean(peaks)) if peaks else 0.0,
        latency=float(np.mean(latencies)) if latencies else 0.0,
        energy=float(np.mean(energies)) if energies else 0.0,
        per_workload=metrics_by_workload,
    )


def _run_mapping_strategy(
    name: str,
    simulator: Any,
    workload_stream: list[str] | None,
    cfg: Any,
    mapping_builder: Callable[[str, list[dict[str, Any]]], np.ndarray],
) -> BaselineResult:
    workloads = _workload_names(cfg, workload_stream)
    runner = WorkloadRunner(cfg_get(cfg, "workloads", None))
    aging_gen = AgingLabelGenerator(cfg=cfg)
    failure_threshold = float(cfg_get(cfg_get(cfg, "planning", cfg), "failure_threshold", 0.8))

    results: dict[str, dict[str, float]] = {}
    for workload_name in workloads:
        layers = runner.get_workload_layers(workload_name)
        mapping = mapping_builder(workload_name, layers)
        results[workload_name] = _evaluate_mapping(simulator, aging_gen, workload_name, layers, mapping, failure_threshold)

    return _aggregate_result(name, results)


def run_static_mapping(simulator, graph, workload_stream, cfg) -> BaselineResult:
    num_clusters = max(int(getattr(simulator, "num_mac_clusters", 1)), 1)
    return _run_mapping_strategy(
        "Static",
        simulator,
        workload_stream,
        cfg,
        lambda _workload_name, layers: np.zeros(len(layers), dtype=np.int32) % num_clusters,
    )


def run_random_mapping(simulator, graph, workload_stream, cfg, seed) -> BaselineResult:
    rng = np.random.default_rng(seed)
    num_clusters = max(int(getattr(simulator, "num_mac_clusters", 1)), 1)
    return _run_mapping_strategy(
        "Random",
        simulator,
        workload_stream,
        cfg,
        lambda _workload_name, layers: rng.integers(0, num_clusters, size=len(layers), endpoint=False, dtype=np.int32),
    )


def run_round_robin(simulator, graph, workload_stream, cfg) -> BaselineResult:
    num_clusters = max(int(getattr(simulator, "num_mac_clusters", 1)), 1)
    return _run_mapping_strategy(
        "Round-Robin",
        simulator,
        workload_stream,
        cfg,
        lambda _workload_name, layers: np.arange(len(layers), dtype=np.int32) % num_clusters,
    )


def run_thermal_balancing(simulator, graph, workload_stream, cfg) -> BaselineResult:
    num_clusters = max(int(getattr(simulator, "num_mac_clusters", 1)), 1)

    def build_mapping(_workload_name: str, layers: list[dict[str, Any]]) -> np.ndarray:
        cluster_loads = np.zeros(num_clusters, dtype=np.float64)
        mapping = np.zeros(len(layers), dtype=np.int32)
        for idx, layer in enumerate(layers):
            target_cluster = int(np.argmin(cluster_loads))
            mapping[idx] = target_cluster
            cluster_loads[target_cluster] += _layer_load(layer)
        return mapping

    return _run_mapping_strategy("Thermal-Balancing", simulator, workload_stream, cfg, build_mapping)


def run_simulated_annealing(simulator, graph, workload_stream, cfg) -> BaselineResult:
    workloads = _workload_names(cfg, workload_stream)
    runner = WorkloadRunner(cfg_get(cfg, "workloads", None))
    aging_gen = AgingLabelGenerator(cfg=cfg)
    failure_threshold = float(cfg_get(cfg_get(cfg, "planning", cfg), "failure_threshold", 0.8))
    num_clusters = max(int(getattr(simulator, "num_mac_clusters", 1)), 1)
    rng = np.random.default_rng(int(cfg_get(cfg, "seed", 42)))

    results: dict[str, dict[str, float]] = {}
    for workload_name in workloads:
        layers = runner.get_workload_layers(workload_name)
        current = rng.integers(0, num_clusters, size=len(layers), endpoint=False, dtype=np.int32)
        current_metrics = _evaluate_mapping(simulator, aging_gen, workload_name, layers, current, failure_threshold)
        best_mapping = current.copy()
        best_metrics = dict(current_metrics)

        temperature = 1.0
        cooling = math.pow(0.01 / temperature, 1.0 / 100.0)

        for _ in range(100):
            proposal = current.copy()
            if len(proposal) > 0:
                layer_idx = int(rng.integers(0, len(proposal)))
                proposal[layer_idx] = int(rng.integers(0, num_clusters))

            proposal_metrics = _evaluate_mapping(simulator, aging_gen, workload_name, layers, proposal, failure_threshold)
            delta = proposal_metrics["peak_aging"] - current_metrics["peak_aging"]

            accept = delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-6))
            if accept:
                current = proposal
                current_metrics = proposal_metrics

            if proposal_metrics["peak_aging"] < best_metrics["peak_aging"]:
                best_mapping = proposal.copy()
                best_metrics = dict(proposal_metrics)

            temperature *= cooling

        best_metrics["mapping"] = best_mapping.astype(int).tolist()
        results[workload_name] = best_metrics

    return _aggregate_result("SA", results)


def run_all_baselines(cfg, simulator, graph) -> dict:
    results = {
        "Static": run_static_mapping(simulator, graph, [], cfg),
        "Random": run_random_mapping(simulator, graph, [], cfg, 42),
        "Round-Robin": run_round_robin(simulator, graph, [], cfg),
        "Thermal-Balancing": run_thermal_balancing(simulator, graph, [], cfg),
        "SA": run_simulated_annealing(simulator, graph, [], cfg),
    }

    output_dir = Path(cfg_get(cfg, "output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    details_path = output_dir / "baseline_details.json"
    with open(details_path, "w", encoding="utf-8") as handle:
        json.dump({name: result.to_dict() for name, result in results.items()}, handle, indent=2)

    summary = {name: result.ttf for name, result in results.items()}
    summary_path = output_dir / "baseline_comparison.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
