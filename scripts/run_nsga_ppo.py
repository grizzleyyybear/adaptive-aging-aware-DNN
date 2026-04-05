from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from aging_models.aging_label_generator import AgingLabelGenerator
from experiments.baseline_experiments import run_all_baselines
from features.feature_builder import FeatureBuilder
from graph.accelerator_graph import AcceleratorGraph
from optimization.nsga2_optimizer import NSGA2Optimizer
from planning.lifetime_planner import LifetimePlanner
from rl.environment import AgingControlEnv
from rl.policy_network import ActorCritic
from rl.trainer import PPOTrainer
from simulator.timeloop_runner import TimeloopRunner
from simulator.workload_runner import WorkloadRunner
from utils.device import describe_device
from utils.runtime_eval import compute_physics_ttf, compute_predictor_ttf, get_workload_names, load_pretrained_predictor, load_pretrained_trajectory, normalize_mapping, simulate_mapping


def load_cfg():
    accel = OmegaConf.load(REPO_ROOT / "configs/accelerator.yaml")
    workloads = OmegaConf.load(REPO_ROOT / "configs/workloads.yaml")
    training = OmegaConf.load(REPO_ROOT / "configs/training.yaml")
    experiments = OmegaConf.load(REPO_ROOT / "configs/experiments.yaml")
    return OmegaConf.merge(experiments, accel, workloads, training)


def choose_best_pareto_solution(front):
    if not front:
        return None
    return min(front, key=lambda sol: (sol.peak_aging, sol.latency, sol.energy))


_TTF_MAX_TIME_S = 200_000_000.0  # ~6.34 years; HCI model saturates at ~100M s for typical activity
# The combined aging score maxes at 0.4+0.4+~0=0.8 (NBTI+HCI+TDDB≈0).
# Use 0.75 as the failure threshold so the score can strictly exceed it,
# allowing the binary search to converge.  Different mappings produce different
# activity levels → different TTF values in the range ~0.6–2.4 years.
_TTF_FAILURE_THRESHOLD = 0.75


def _compute_ttf_physics(aging_gen: AgingLabelGenerator, activity: dict, failure_threshold: float = _TTF_FAILURE_THRESHOLD) -> float:
    """Binary search for time when peak physics-based aging score exceeds threshold."""
    lo, hi = 0.0, _TTF_MAX_TIME_S
    for _ in range(40):
        mid = (lo + hi) / 2.0
        scores = aging_gen.compute_aging_score(activity, mid)
        if float(np.max(scores)) >= failure_threshold:
            hi = mid
        else:
            lo = mid
    return lo / (3600 * 8760)


def evaluate_mapping(
    simulator: TimeloopRunner,
    feature_builder: FeatureBuilder,
    graph: AcceleratorGraph,
    predictor,
    planner: LifetimePlanner,
    workload_name: str,
    layers: list[dict[str, Any]],
    mapping: np.ndarray,
    device,
    aging_gen: AgingLabelGenerator | None = None,
) -> dict[str, Any]:
    metrics = simulate_mapping(
        simulator=simulator,
        feature_builder=feature_builder,
        graph=graph,
        layers=layers,
        mapping=mapping,
        workload_name=workload_name,
        predictor=predictor,
        device=device,
    )
    # Use physics-based TTF (per-mapping activity, large search window) instead of the
    # predictor-based version which converges to a fixed value when the model is not
    # well-calibrated for the full stress-time range.
    if aging_gen is not None:
        ttf_years = _compute_ttf_physics(aging_gen, metrics["activity"])
    else:
        ttf_years = compute_physics_ttf(
            simulator=simulator,
            aging_generator=planner.aging_generator if planner.aging_generator is not None else aging_gen,
            layers=layers,
            mapping=mapping,
            max_time_s=_TTF_MAX_TIME_S,
            n_iter=40,
        ) if (planner.aging_generator is not None) else planner.estimate_failure_time(layers, mapping, workload_name)
    return {
        "mapping": normalize_mapping(mapping, len(layers), simulator.num_mac_clusters).astype(int).tolist(),
        "peak_aging": float(metrics["peak_aging"]),
        "latency_cycles": float(metrics["latency_cycles"]),
        "energy_pj": float(metrics["energy_pj"]),
        "ttf_years": float(ttf_years),
    }


def evaluate_ppo_policy(
    env: AgingControlEnv,
    policy: ActorCritic,
    simulator: TimeloopRunner,
    feature_builder: FeatureBuilder,
    graph: AcceleratorGraph,
    predictor,
    planner: LifetimePlanner,
    workload_name: str,
    layers: list[dict[str, Any]],
    device,
    aging_gen: AgingLabelGenerator | None = None,
) -> dict[str, Any]:
    obs, _ = env.reset(options={"fixed_workload": workload_name})
    final_info: dict[str, Any] = {}

    while True:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.act_deterministic(obs_tensor)
        obs, _reward, terminated, truncated, info = env.step(int(action.item()))
        final_info = info
        if terminated or truncated:
            break

    final_mapping = np.asarray(final_info.get("mapping", np.arange(len(layers)) % simulator.num_mac_clusters), dtype=np.int32)
    return evaluate_mapping(
        simulator=simulator,
        feature_builder=feature_builder,
        graph=graph,
        predictor=predictor,
        planner=planner,
        workload_name=workload_name,
        layers=layers,
        mapping=final_mapping,
        device=device,
        aging_gen=aging_gen,
    )


def main() -> None:
    cfg = load_cfg()
    output_root = REPO_ROOT / cfg.get("output_dir", "outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "models").mkdir(parents=True, exist_ok=True)

    predictor, device, predictor_ckpt = load_pretrained_predictor(
        cfg,
        checkpoint_candidates=[REPO_ROOT / "outputs/best_predictor.pt", REPO_ROOT / "outputs/models/hybrid_gnn_transformer.pt"],
        device_request=cfg.runtime.device,
    )
    trajectory_predictor, trajectory_ckpt = load_pretrained_trajectory(
        cfg,
        predictor,
        checkpoint_candidates=[REPO_ROOT / "outputs/best_trajectory.pt", REPO_ROOT / "outputs/models/trajectory_predictor.pt"],
        device=device,
    )

    print(f"[Setup] Predictor checkpoint: {predictor_ckpt}")
    print(f"[Setup] Trajectory checkpoint: {trajectory_ckpt}")
    print(f"[Setup] Inference device: {describe_device(device)}")

    accel_cfg = cfg.accelerator
    simulator = TimeloopRunner(accel_cfg)
    workload_runner = WorkloadRunner(cfg.workloads)
    feature_builder = FeatureBuilder(accel_cfg)
    aging_gen = AgingLabelGenerator(cfg=cfg)
    graph = AcceleratorGraph(accel_cfg)
    graph.build()

    planner = LifetimePlanner(graph, cfg.planning)
    planner.attach_runtime(
        predictor=predictor,
        simulator=simulator,
        feature_builder=feature_builder,
        aging_generator=aging_gen,
        device=device,
    )

    workload_names = get_workload_names(cfg, workload_runner)
    nsga_cfg = OmegaConf.create(
        {
            "pop_size": max(int(cfg_get(cfg.nsga2, "pop_size", 20)), 20),
            "population_size": max(int(cfg_get(cfg.nsga2, "population_size", 20)), 20),
            "n_gen": max(int(cfg_get(cfg.nsga2, "n_gen", cfg_get(cfg.nsga2, "generations", 10))), 10),
            "crossover_prob": float(cfg_get(cfg.nsga2, "crossover_prob", 0.9)),
            "mutation_prob": float(cfg_get(cfg.nsga2, "mutation_prob", 0.1)),
            "seed": int(cfg.get("seed", 42)),
        }
    )

    optimizer = NSGA2Optimizer(accel_cfg, simulator, predictor, nsga_cfg)
    pareto_summary = {}

    print("[NSGA-II] Starting Pareto optimization across workloads")
    for workload_name in workload_names:
        layers = workload_runner.get_workload_layers(workload_name)
        # Use all-to-one as the seed so NSGA-II has a concrete suboptimal baseline to improve on.
        # The diverse sampling in _build_sampling also adds round-robin and random seeds.
        initial_mapping = np.zeros(len(layers), dtype=np.int32)
        print(f"[NSGA-II] Optimizing {workload_name} with {len(layers)} layers")
        front = optimizer.run(initial_mapping=initial_mapping, n_gen=int(nsga_cfg.n_gen), workload_name=workload_name)
        pareto_summary[workload_name] = [solution.to_dict() for solution in front]
        print(f"[NSGA-II] {workload_name}: {len(front)} Pareto solutions")

    optimizer.save_pareto_solutions(output_root / "pareto_mappings.json")

    print("[PPO] Building environment and policy")
    env = AgingControlEnv(simulator, planner, workload_runner, aging_gen, graph, predictor, trajectory_predictor, feature_builder, cfg)
    policy = ActorCritic(obs_dim=env.observation_space.shape[0], action_dim=5)
    ppo_cfg = OmegaConf.create(
        {
            "total_timesteps": 5120,
            "n_iterations": 40,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": int(cfg_get(cfg.ppo, "n_epochs", cfg_get(cfg.ppo, "epochs", 4))),
            "gamma": float(cfg_get(cfg.ppo, "gamma", 0.99)),
            "gae_lambda": float(cfg_get(cfg.ppo, "gae_lambda", 0.95)),
            "clip_range": 0.2,
            "ent_coef": float(cfg_get(cfg.ppo, "entropy_coeff", 0.01)),
            "vf_coef": float(cfg_get(cfg.ppo, "value_loss_coeff", 0.5)),
            "learning_rate": float(cfg_get(cfg.ppo, "learning_rate", 3e-4)),
            "device": cfg.runtime.device,
        }
    )
    trainer = PPOTrainer(env, policy, ppo_cfg)
    rl_metrics = trainer.train()
    torch.save(policy.state_dict(), output_root / "models" / "rl_policy_final.pt")

    print("[Eval] Comparing initial mapping, NSGA-II best, and PPO policy")
    summary_rows = []
    method_metrics = {"Initial": [], "NSGA-II": [], "PPO": []}

    for workload_name in workload_names:
        layers = workload_runner.get_workload_layers(workload_name)
        # "Initial" = all-to-one (naive/worst-case baseline): all layers on cluster 0,
        # high latency (sequential within the cluster), high aging on cluster 0.
        # NSGA-II and PPO should improve over this baseline.
        initial_mapping = np.zeros(len(layers), dtype=np.int32)
        initial_metrics = evaluate_mapping(simulator, feature_builder, graph, predictor, planner, workload_name, layers, initial_mapping, device, aging_gen=aging_gen)

        front = optimizer.pareto_history.get(workload_name, [])
        best_solution = choose_best_pareto_solution(front)
        if best_solution is None:
            nsga_metrics = initial_metrics
        else:
            nsga_metrics = evaluate_mapping(
                simulator,
                feature_builder,
                graph,
                predictor,
                planner,
                workload_name,
                layers,
                best_solution.mapping,
                device,
                aging_gen=aging_gen,
            )

        ppo_metrics = evaluate_ppo_policy(env, policy, simulator, feature_builder, graph, predictor, planner, workload_name, layers, device, aging_gen=aging_gen)

        for method_name, metrics in [("Initial", initial_metrics), ("NSGA-II", nsga_metrics), ("PPO", ppo_metrics)]:
            summary_rows.append(
                {
                    "Workload": workload_name,
                    "Method": method_name,
                    "Peak Aging": metrics["peak_aging"],
                    "Latency Cycles": metrics["latency_cycles"],
                    "Energy (pJ)": metrics["energy_pj"],
                    "TTF (Yrs)": metrics["ttf_years"],
                }
            )
            method_metrics[method_name].append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "nsga_ppo_summary.csv", index=False)

    overall = {}
    initial_ttfs = np.array([item["ttf_years"] for item in method_metrics["Initial"]], dtype=np.float64)
    initial_latencies = np.array([item["latency_cycles"] for item in method_metrics["Initial"]], dtype=np.float64)
    for method_name in ("NSGA-II", "PPO"):
        ttfs = np.array([item["ttf_years"] for item in method_metrics[method_name]], dtype=np.float64)
        latencies = np.array([item["latency_cycles"] for item in method_metrics[method_name]], dtype=np.float64)
        overall[method_name] = {
            "mean_lifetime_extension": float(np.mean(ttfs - initial_ttfs)),
            "mean_latency_overhead": float(np.mean((latencies - initial_latencies) / np.maximum(initial_latencies, 1e-6))),
        }

    baseline_ttf = run_all_baselines(cfg, simulator, graph)
    results_payload = {
        "pareto": pareto_summary,
        "ppo_training": rl_metrics,
        "evaluation": summary_rows,
        "overall": overall,
        "baseline_ttf": baseline_ttf,
    }
    with open(output_root / "nsga_ppo_results.json", "w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)

    print(summary_df.to_string(index=False))
    print("\nOverall:")
    for method_name, metrics in overall.items():
        print(
            f"  {method_name}: mean_lifetime_extension={metrics['mean_lifetime_extension']:.4f} yrs | "
            f"mean_latency_overhead={metrics['mean_latency_overhead']:.4f}"
        )


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


if __name__ == "__main__":
    main()
