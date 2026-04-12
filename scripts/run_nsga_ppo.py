from __future__ import annotations

import json
import shutil
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
from features.feature_builder import FeatureBuilder
from graph.accelerator_graph import AcceleratorGraph
from optimization.nsga2_optimizer import NSGA2Optimizer, ParetoSolution
from planning.lifetime_planner import LifetimePlanner
from rl.environment import AgingControlEnv
from rl.policy_network import ActorCritic
from rl.trainer import PPOTrainer
from simulator.timeloop_runner import TimeloopRunner
from simulator.workload_runner import WorkloadRunner
from utils.device import describe_device
from utils.runtime_eval import (
    cfg_get,
    get_workload_names,
    load_pretrained_predictor,
    load_pretrained_trajectory,
    normalize_mapping,
    simulate_mapping,
)


def load_cfg():
    accel = OmegaConf.load(REPO_ROOT / "configs/accelerator.yaml")
    workloads = OmegaConf.load(REPO_ROOT / "configs/workloads.yaml")
    training = OmegaConf.load(REPO_ROOT / "configs/training.yaml")
    experiments = OmegaConf.load(REPO_ROOT / "configs/experiments.yaml")
    cfg = OmegaConf.merge(experiments, accel, workloads, training)
    cfg.model.prediction_horizon = 10
    cfg.runtime.device = "cuda"
    return cfg


def choose_best_pareto_solution(front: list[dict[str, Any]] | list[ParetoSolution]):
    if not front:
        return None
    if isinstance(front[0], ParetoSolution):
        return min(front, key=lambda sol: (sol.peak_aging, sol.latency, sol.energy))
    return min(front, key=lambda sol: (sol["peak_aging"], sol["latency_cycles"], sol["energy_pj"]))


_TTF_MAX_TIME_S = 200_000_000.0
_TTF_FAILURE_THRESHOLD = 0.75


def _compute_ttf_physics(
    aging_gen: AgingLabelGenerator,
    activity: dict,
    failure_threshold: float = _TTF_FAILURE_THRESHOLD,
) -> float:
    lo, hi = 0.0, _TTF_MAX_TIME_S
    for _ in range(40):
        mid = (lo + hi) / 2.0
        scores = aging_gen.compute_aging_score(activity, mid)
        if float(np.max(scores)) >= failure_threshold:
            hi = mid
        else:
            lo = mid
    return lo / (3600.0 * 8760.0)


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
    if aging_gen is not None:
        ttf_years = _compute_ttf_physics(aging_gen, metrics["activity"])
    else:
        ttf_years = planner.estimate_failure_time(layers, mapping, workload_name)

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

    final_mapping = np.asarray(
        final_info.get("mapping", np.arange(len(layers)) % simulator.num_mac_clusters),
        dtype=np.int32,
    )
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


def _copy_checkpoint_if_present(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _load_training_metrics() -> dict[str, Any]:
    metrics_path = REPO_ROOT / "outputs" / "real_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _build_final_metrics(
    training_metrics: dict[str, Any],
    nsga2_metrics: dict[str, Any],
    rl_metrics: dict[str, Any],
) -> dict[str, Any]:
    rewards = [float(x) for x in rl_metrics.get("reward", [])]
    predictor_metrics = training_metrics.get("predictor", {})
    trajectory_metrics = training_metrics.get("trajectory", {})
    return {
        "mode": "FULL",
        "dataset_size": 40000,
        "predictor": {
            "r2": float(predictor_metrics.get("r2", 0.0)),
            "mae": float(predictor_metrics.get("mae", 0.0)),
            "rmse": float(predictor_metrics.get("rmse", 0.0)),
        },
        "trajectory": {
            "r2": float(trajectory_metrics.get("r2", 0.0)),
            "mae": float(trajectory_metrics.get("mae", 0.0)),
            "rmse": float(trajectory_metrics.get("rmse", 0.0)),
            "per_step_r2": [float(x) for x in trajectory_metrics.get("per_step_r2", [])],
        },
        "nsga2": nsga2_metrics,
        "ppo": {
            "rewards": rewards,
            "first": float(rewards[0]) if rewards else 0.0,
            "last": float(rewards[-1]) if rewards else 0.0,
            "best": float(max(rewards)) if rewards else 0.0,
            "mean": float(np.mean(rewards)) if rewards else 0.0,
        },
    }


def main() -> None:
    cfg = load_cfg()
    output_root = REPO_ROOT / cfg.get("output_dir", "outputs")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "models").mkdir(parents=True, exist_ok=True)

    predictor, device, predictor_ckpt = load_pretrained_predictor(
        cfg,
        checkpoint_candidates=[
            REPO_ROOT / "outputs/best_predictor.pt",
            REPO_ROOT / "outputs/models/hybrid_gnn_transformer.pt",
        ],
        device_request="cuda",
    )
    if device.type != "cuda":
        raise RuntimeError("scripts/run_nsga_ppo.py requires CUDA for predictor inference.")

    trajectory_predictor, trajectory_ckpt = load_pretrained_trajectory(
        cfg,
        predictor,
        checkpoint_candidates=[
            REPO_ROOT / "outputs/best_trajectory.pt",
            REPO_ROOT / "outputs/models/trajectory_predictor.pt",
        ],
        device=device,
    )

    print(f"[Setup] Predictor checkpoint : {predictor_ckpt}")
    print(f"[Setup] Trajectory checkpoint: {trajectory_ckpt}")
    print(f"[Setup] Inference device    : {describe_device(device)}")

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
    nsga2_cfg_section = cfg_get(cfg, "nsga2", {})
    nsga_cfg = OmegaConf.create(
        {
            "pop_size": max(int(cfg_get(nsga2_cfg_section, "pop_size", 20)), 50),
            "population_size": max(int(cfg_get(nsga2_cfg_section, "population_size", 20)), 50),
            "n_gen": max(int(cfg_get(nsga2_cfg_section, "n_gen", cfg_get(nsga2_cfg_section, "generations", 10))), 30),
            "crossover_prob": float(cfg_get(nsga2_cfg_section, "crossover_prob", 0.9)),
            "mutation_prob": float(cfg_get(nsga2_cfg_section, "mutation_prob", 0.1)),
            "seed": int(cfg.get("seed", 42)),
            "balance_weight": float(cfg_get(nsga2_cfg_section, "balance_weight", 0.3)),
            "convergence_patience": int(cfg_get(nsga2_cfg_section, "convergence_patience", 15)),
            "convergence_min_improvement": float(cfg_get(nsga2_cfg_section, "convergence_min_improvement", 1e-6)),
            "verbose": False,
        }
    )

    optimizer = NSGA2Optimizer(accel_cfg, simulator, predictor, nsga_cfg)
    pareto_payload: dict[str, list[dict[str, Any]]] = {}
    pareto_mappings_payload: dict[str, list[list[int]]] = {}
    pareto_objectives_payload: dict[str, list[dict[str, Any]]] = {}

    print("[NSGA-II] Starting Pareto optimization across workloads")
    print(
        f"[NSGA-II] pop_size={int(nsga_cfg.pop_size)}  generations={int(nsga_cfg.n_gen)}  "
        f"workloads={len(workload_names)}"
    )
    for workload_name in workload_names:
        layers = workload_runner.get_workload_layers(workload_name)
        initial_mapping = np.zeros(len(layers), dtype=np.int32)
        print(f"[NSGA-II] Optimizing {workload_name} with {len(layers)} layers")
        front = optimizer.run(
            initial_mapping=initial_mapping,
            n_gen=int(nsga_cfg.n_gen),
            workload_name=workload_name,
        )
        raw_front = [
            evaluate_mapping(
                simulator,
                feature_builder,
                graph,
                predictor,
                planner,
                workload_name,
                layers,
                sol.mapping,
                device,
                aging_gen=aging_gen,
            )
            for sol in front
        ]
        raw_front.sort(key=lambda item: (item["peak_aging"], item["latency_cycles"], item["energy_pj"]))

        pareto_payload[workload_name] = raw_front
        pareto_mappings_payload[workload_name] = [item["mapping"] for item in raw_front]
        pareto_objectives_payload[workload_name] = [
            {
                "peak_aging": item["peak_aging"],
                "latency_cycles": item["latency_cycles"],
                "energy_pj": item["energy_pj"],
                "ttf_years": item["ttf_years"],
            }
            for item in raw_front
        ]
        print(f"[NSGA-II] {workload_name}: {len(raw_front)} Pareto solutions")

    with open(output_root / "pareto_mappings.json", "w", encoding="utf-8") as handle:
        json.dump(pareto_mappings_payload, handle, indent=2)
    with open(output_root / "pareto_objectives.json", "w", encoding="utf-8") as handle:
        json.dump(pareto_objectives_payload, handle, indent=2)

    print("[PPO] Building environment and policy")
    env = AgingControlEnv(
        simulator,
        planner,
        workload_runner,
        aging_gen,
        graph,
        predictor,
        trajectory_predictor,
        feature_builder,
        cfg,
    )
    policy = ActorCritic(obs_dim=env.observation_space.shape[0], action_dim=5).to(device)
    ppo_cfg_section = cfg_get(cfg, "ppo", {})
    ppo_iterations = 80
    ppo_steps = 128
    ppo_cfg = OmegaConf.create(
        {
            "total_timesteps": ppo_iterations * ppo_steps,
            "n_iterations": ppo_iterations,
            "n_steps": ppo_steps,
            "batch_size": 64,
            "n_epochs": int(cfg_get(ppo_cfg_section, "n_epochs", cfg_get(ppo_cfg_section, "epochs", 4))),
            "gamma": float(cfg_get(ppo_cfg_section, "gamma", 0.99)),
            "gae_lambda": float(cfg_get(ppo_cfg_section, "gae_lambda", 0.95)),
            "clip_range": 0.2,
            "ent_coef": float(cfg_get(ppo_cfg_section, "entropy_coeff", 0.01)),
            "vf_coef": float(cfg_get(ppo_cfg_section, "value_loss_coeff", 0.5)),
            "learning_rate": float(cfg_get(ppo_cfg_section, "learning_rate", 3e-4)),
            "device": "cuda",
            "eval_interval": 5,
            "eval_episodes": 5,
        }
    )
    print(f"[PPO] iterations={ppo_iterations}  steps_per_iter={ppo_steps}  device=cuda")
    trainer = PPOTrainer(env, policy, ppo_cfg)
    rl_metrics = trainer.train()
    torch.save(policy.state_dict(), output_root / "models" / "rl_policy_final.pt")
    _copy_checkpoint_if_present(REPO_ROOT / "checkpoints" / "rl_policy_best.pt", output_root / "models" / "rl_policy_best.pt")

    print("[Eval] Comparing initial mapping, NSGA-II best, and PPO policy")
    summary_rows = []
    method_metrics = {"Initial": [], "NSGA-II": [], "PPO": []}
    nsga2_metrics = {}

    for workload_name in workload_names:
        layers = workload_runner.get_workload_layers(workload_name)
        initial_mapping = np.zeros(len(layers), dtype=np.int32)
        initial_metrics = evaluate_mapping(
            simulator,
            feature_builder,
            graph,
            predictor,
            planner,
            workload_name,
            layers,
            initial_mapping,
            device,
            aging_gen=aging_gen,
        )

        raw_front = pareto_payload.get(workload_name, [])
        best_solution = choose_best_pareto_solution(raw_front)
        nsga_metrics = best_solution if best_solution is not None else initial_metrics

        ppo_metrics = evaluate_ppo_policy(
            env,
            policy,
            simulator,
            feature_builder,
            graph,
            predictor,
            planner,
            workload_name,
            layers,
            device,
            aging_gen=aging_gen,
        )

        init_peak = float(initial_metrics["peak_aging"])
        best_peak = float(nsga_metrics["peak_aging"])
        reduction = 100.0 * (init_peak - best_peak) / max(init_peak, 1e-12)
        nsga2_metrics[workload_name] = {
            "count": int(len(raw_front)),
            "init": init_peak,
            "best": best_peak,
            "reduction": float(reduction),
        }

        for method_name, metrics in [("Initial", initial_metrics), ("NSGA-II", nsga_metrics), ("PPO", ppo_metrics)]:
            summary_rows.append(
                {
                    "Workload": workload_name,
                    "Method": method_name,
                    "Peak Aging": float(metrics["peak_aging"]),
                    "Latency Cycles": float(metrics["latency_cycles"]),
                    "Energy (pJ)": float(metrics["energy_pj"]),
                    "TTF (Yrs)": float(metrics["ttf_years"]),
                }
            )
            method_metrics[method_name].append(metrics)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "nsga_ppo_summary.csv", index=False)
    with open(output_root / "evaluation_table.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    overall = {}
    initial_ttfs = np.array([item["ttf_years"] for item in method_metrics["Initial"]], dtype=np.float64)
    initial_latencies = np.array([item["latency_cycles"] for item in method_metrics["Initial"]], dtype=np.float64)
    for method_name in ("NSGA-II", "PPO"):
        ttfs = np.array([item["ttf_years"] for item in method_metrics[method_name]], dtype=np.float64)
        latencies = np.array([item["latency_cycles"] for item in method_metrics[method_name]], dtype=np.float64)
        overall[method_name] = {
            "mean_ttf_years": float(np.mean(ttfs)),
            "mean_lifetime_extension": float(np.mean(ttfs - initial_ttfs)),
            "mean_latency_overhead": float(
                np.mean((latencies - initial_latencies) / np.maximum(initial_latencies, 1e-6))
            ),
        }

    results_payload = {
        "pareto": pareto_payload,
        "ppo_training": rl_metrics,
        "evaluation": summary_rows,
        "overall": overall,
    }
    with open(output_root / "nsga_ppo_results.json", "w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)

    training_metrics = _load_training_metrics()
    final_metrics = _build_final_metrics(training_metrics, nsga2_metrics, rl_metrics)
    with open(output_root / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    print(summary_df.to_string(index=False))
    print("\n[Summary] Overall comparison")
    for method_name, metrics in overall.items():
        print(
            f"  {method_name}: mean_ttf={metrics['mean_ttf_years']:.4f} yrs | "
            f"extension={metrics['mean_lifetime_extension']:.4f} yrs | "
            f"latency_overhead={metrics['mean_latency_overhead']:.4f}"
        )

    rewards = final_metrics["ppo"]["rewards"]
    if rewards:
        print(
            f"[PPO] rewards: first={final_metrics['ppo']['first']:+.4f}  "
            f"last={final_metrics['ppo']['last']:+.4f}  "
            f"best={final_metrics['ppo']['best']:+.4f}  "
            f"mean={final_metrics['ppo']['mean']:+.4f}"
        )
    print(f"[Done] Saved outputs to {output_root}")


if __name__ == "__main__":
    main()
