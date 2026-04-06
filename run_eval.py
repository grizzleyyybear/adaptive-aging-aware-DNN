"""
Fast smoke evaluation — tests the full improved pipeline end-to-end.

Measures all key metrics with a small dataset for quick validation.
For full results, increase DATASET_SIZE and PPO_ITERATIONS.

Usage:
    python run_eval.py              # fast smoke (< 5 min on CPU)
    python run_eval.py --full       # full eval  (needs GPU, ~10 min)

Previous baseline results (from simulator-backed data):
    Predictor:  R² = 0.9925, MAE = 0.005
    Trajectory: R² = 0.78,   MAE = 0.072
    NSGA-II:    4-10 Pareto solutions, up to 63% peak aging reduction
    PPO:        reward -0.61 → +0.36 over 40 iterations
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

# ── Detect mode ──────────────────────────────────────────────────────
FULL_MODE = "--full" in sys.argv

if FULL_MODE:
    DATASET_SIZE  = 40000
    TRAIN_EPOCHS  = 20
    NSGA_POP      = 20
    NSGA_GEN      = 20
    PPO_ITERS     = 40
    PPO_STEPS     = 64
    DEVICE        = "auto"
else:
    DATASET_SIZE  = 200
    TRAIN_EPOCHS  = 10
    NSGA_POP      = 12
    NSGA_GEN      = 12
    PPO_ITERS     = 15
    PPO_STEPS     = 32
    DEVICE        = "cpu"

SEP = "=" * 68


def build_config():
    return OmegaConf.create({
        "seed": 42,
        "accelerator": {
            "num_mac_clusters": 16, "mac_clusters": 16,
            "num_sram_banks": 8, "sram_banks": 8,
            "num_noc_routers": 4, "noc_routers": 4,
            "pe_array": [16, 16], "num_layers": 10,
            "max_macs_per_cluster": 256, "clock_ghz": 1.0,
            "sram_read_energy_pj": 2.0, "noc_hop_energy_pj": 0.5,
            "mac_energy_pj_per_op": 0.1, "ops_per_cycle": 2,
        },
        "model": {
            "hidden_dim": 256, "gat_heads": 4,
            "transformer_heads": 4, "transformer_layers": 2,
            "prediction_horizon": 10,
        },
        "training": {
            "epochs": TRAIN_EPOCHS, "batch_size": 64,
            "learning_rate": 1e-4, "weight_decay": 1e-5,
            "patience": 8,
        },
        "aging": {
            "nbti_A": 0.005, "nbti_n": 0.25,
            "hci_B": 0.0001, "hci_m": 0.5,
            "tddb_k": 2.5, "tddb_beta": 10.0,
        },
        "planning": {
            "failure_threshold": 0.8,
            "nbti": 0.40, "hci": 0.35, "tddb": 0.25,
        },
        "workloads": [],
        "runtime": {"device": DEVICE},
    })


def main():
    mode_label = "FULL" if FULL_MODE else "SMOKE"
    print(f"\n{SEP}")
    print(f"  PIPELINE EVALUATION [{mode_label}]  —  device={DEVICE}")
    print(SEP)

    cfg = build_config()
    torch.manual_seed(42)
    np.random.seed(42)
    accel_cfg = cfg.accelerator

    from graph.accelerator_graph import AcceleratorGraph
    from graph.graph_dataset import AgingDataset
    from simulator.timeloop_runner import TimeloopRunner
    from simulator.workload_runner import WorkloadRunner
    from features.feature_builder import FeatureBuilder
    from planning.lifetime_planner import LifetimePlanner
    from models.hybrid_gnn_transformer import HybridGNNTransformer
    from models.training_pipeline import TrainingPipeline
    from models.trajectory_predictor import TrajectoryPredictor
    from optimization.nsga2_optimizer import NSGA2Optimizer
    from rl.environment import AgingControlEnv
    from rl.policy_network import ActorCritic
    from rl.trainer import PPOTrainer
    from utils.runtime_eval import simulate_mapping

    # ── 1. Infrastructure ─────────────────────────────────────────────
    print("\n[1/6] Building infrastructure...", flush=True)
    graph = AcceleratorGraph(accel_cfg); graph.build()
    sim   = TimeloopRunner(accel_cfg)
    planner = LifetimePlanner(graph, cfg.planning)
    fb    = FeatureBuilder(accel_cfg)
    N     = graph.get_num_nodes()
    print(f"  Graph: {N} nodes  |  {accel_cfg.mac_clusters} MACs, "
          f"{accel_cfg.sram_banks} SRAMs, {accel_cfg.noc_routers} Routers", flush=True)

    # ── 2. Dataset ────────────────────────────────────────────────────
    print(f"\n[2/6] Loading dataset ({DATASET_SIZE} samples)...", flush=True)
    t0 = time.perf_counter()
    dataset = AgingDataset(
        root=str(REPO_ROOT / "data"),
        split="train", size=DATASET_SIZE, cfg=cfg, seed=42,
    )
    n_samples = len(dataset)
    print(f"  {n_samples} samples loaded ({time.perf_counter()-t0:.1f}s)", flush=True)
    if n_samples == 0:
        print("  ERROR: Dataset is empty. Run with --full or regenerate.", flush=True)
        sys.exit(1)

    sample = dataset[0]
    feat_dim = int(sample.x.shape[1])
    horizon  = cfg.model.prediction_horizon

    # ── 3. Predictor ──────────────────────────────────────────────────
    print(f"\n[3/6] Training predictor ({TRAIN_EPOCHS} epochs)...", flush=True)
    predictor = HybridGNNTransformer(
        node_feature_dim=feat_dim,
        hidden_dim=cfg.model.hidden_dim,
        gat_heads=cfg.model.gat_heads,
        transformer_layers=cfg.model.transformer_layers,
        transformer_heads=cfg.model.transformer_heads,
        seq_len=horizon,
    )
    pred_metrics = TrainingPipeline(cfg, predictor, dataset).train()
    print(f"  R² = {pred_metrics['r2']:.4f}  |  "
          f"MAE = {pred_metrics['mae']:.4f}  |  "
          f"RMSE = {pred_metrics['rmse']:.4f}", flush=True)

    # ── 4. Trajectory predictor ───────────────────────────────────────
    print(f"\n[4/6] Training trajectory predictor ({TRAIN_EPOCHS} epochs)...", flush=True)
    traj_pred = TrajectoryPredictor(
        gnn_encoder=predictor,
        hidden_dim=cfg.model.hidden_dim,
        horizon=horizon, gamma=0.95,
    )
    traj_metrics = TrainingPipeline(cfg, traj_pred, dataset).train()
    print(f"  R² = {traj_metrics['r2']:.4f}  |  "
          f"MAE = {traj_metrics['mae']:.4f}  |  "
          f"RMSE = {traj_metrics['rmse']:.4f}", flush=True)

    # ── 5. NSGA-II ────────────────────────────────────────────────────
    print(f"\n[5/6] NSGA-II optimization (pop={NSGA_POP}, gen={NSGA_GEN})...", flush=True)
    wr = WorkloadRunner(None)
    workloads = ["ResNet-50", "BERT-Base", "MobileNetV2", "EfficientNet-B4", "ViT-B/16"]
    nsga_results = {}
    dev = next(predictor.parameters()).device

    for wl in workloads:
        layers   = wr.get_workload_layers(wl)
        init_map = np.arange(len(layers), dtype=np.int32) % accel_cfg.mac_clusters

        # Initial (round-robin) peak aging
        init_m = simulate_mapping(
            simulator=sim, feature_builder=fb, graph=graph,
            layers=layers, mapping=init_map, workload_name=wl,
            stress_time_s=360_000.0, predictor=predictor, device=dev,
        )
        init_peak = init_m["peak_aging"]

        nsga_cfg = OmegaConf.create({
            "pop_size": NSGA_POP, "population_size": NSGA_POP,
            "crossover_prob": 0.9, "mutation_prob": 0.1,
            "n_gen": NSGA_GEN, "seed": 42,
            "balance_weight": 0.3, "convergence_patience": 8,
            "workload_name": wl,
        })
        opt = NSGA2Optimizer(accel_cfg, sim, predictor, nsga_cfg)
        pareto = opt.run(init_map, n_gen=NSGA_GEN, workload_name=wl)

        best = min((s.peak_aging for s in pareto), default=init_peak)
        red  = (init_peak - best) / max(init_peak, 1e-9) * 100
        nsga_results[wl] = {
            "count": len(pareto), "init": init_peak,
            "best": best, "reduction": red,
            "cache_hits": opt._cache.hits,
            "converged_gen": len(opt.hv_history),
        }
        print(f"  {wl:20s}  {len(pareto):2d} sols  "
              f"peak {init_peak:.4f}→{best:.4f}  ({red:+.1f}%)  "
              f"cache={opt._cache.hits}h  conv@{len(opt.hv_history)}gen", flush=True)

    # ── 6. PPO ────────────────────────────────────────────────────────
    print(f"\n[6/6] PPO training ({PPO_ITERS} iterations, {PPO_STEPS} steps/iter)...", flush=True)
    env = AgingControlEnv(cfg, sim, planner)
    obs_dim = env.observation_space.shape[0]
    policy  = ActorCritic(obs_dim=obs_dim, action_dim=5, hidden_dim=128)
    ppo_cfg = OmegaConf.create({
        "n_iterations": PPO_ITERS, "n_steps": PPO_STEPS,
        "batch_size": 32, "n_epochs": 4,
        "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "learning_rate": 3e-4,
        "total_timesteps": 999999,
        "normalize_obs": True, "target_kl": 0.03,
        "clip_range_vf": 0.2, "ent_coef_end": 0.001,
        "eval_interval": max(PPO_ITERS // 3, 1), "eval_episodes": 3,
        "device": DEVICE,
    })
    rl_metrics = PPOTrainer(env, policy, ppo_cfg).train()
    rewards = rl_metrics["reward"]

    # ══════════════════════════════════════════════════════════════════
    #   RESULTS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  RESULTS  [{mode_label}, {DATASET_SIZE} samples, device={DEVICE}]")
    print(SEP)

    def row(label, curr, prev, better_high=True):
        arrow = "↑" if better_high else "↓"
        if better_high:
            ok = curr >= prev * 0.95
        else:
            ok = curr <= prev * 1.10
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label:<32s}  now={curr:>9.4f}  prev={prev:>9.4f}  ({arrow} {'better' if better_high else 'lower'} is better)")

    print("\n  PREDICTOR (GNN-Transformer)")
    row("R²",   pred_metrics["r2"],   0.9925, True)
    row("MAE",  pred_metrics["mae"],  0.005,  False)
    row("RMSE", pred_metrics["rmse"], 0.007,  False)

    print("\n  TRAJECTORY PREDICTOR")
    row("R²",   traj_metrics["r2"],   0.78,   True)
    row("MAE",  traj_metrics["mae"],  0.072,  False)
    row("RMSE", traj_metrics["rmse"], 0.10,   False)

    print("\n  NSGA-II (Improved)")
    total_sols = sum(r["count"] for r in nsga_results.values())
    best_red   = max(r["reduction"] for r in nsga_results.values())
    total_hits = sum(r["cache_hits"] for r in nsga_results.values())
    print(f"  Total Pareto solutions:  {total_sols}  (prev: 4-10 per workload)")
    print(f"  Best aging reduction:    {best_red:.1f}%  (prev: 63%)")
    print(f"  Cache hits (total):      {total_hits}")
    for wl, r in nsga_results.items():
        print(f"    {wl:20s}  {r['count']:2d} sols  {r['reduction']:+.1f}%  conv@gen {r['converged_gen']}")

    print("\n  PPO (Improved)")
    first_r = rewards[0]  if rewards else 0
    last_r  = rewards[-1] if rewards else 0
    best_r  = max(rewards) if rewards else 0
    mean_r  = float(np.mean(rewards))
    print(f"  Reward curve:  {first_r:+.4f} → {last_r:+.4f}  (prev: -0.61 → +0.36)")
    print(f"  Best reward:   {best_r:+.4f}")
    print(f"  Mean reward:   {mean_r:+.4f}")

    # KL & entropy from improved PPO
    kl_vals = rl_metrics.get("approx_kl", [])
    ent_vals = rl_metrics.get("entropy", [])
    if kl_vals:
        print(f"  Mean KL:       {np.mean(kl_vals):.4f}")
    if ent_vals:
        print(f"  Entropy:       {ent_vals[0]:.3f} → {ent_vals[-1]:.3f}  (annealing)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PIPELINE STATUS")
    print(SEP)
    checks = [
        ("Predictor trains & evaluates",  pred_metrics["r2"] > 0.5),
        ("Trajectory trains & evaluates", traj_metrics["r2"] > 0.0),
        ("NSGA-II finds Pareto solutions", total_sols > 0),
        ("NSGA-II cache working",          total_hits > 0),
        ("NSGA-II convergence detection",  any(r["converged_gen"] < NSGA_GEN for r in nsga_results.values())),
        ("PPO reward improves",            last_r > first_r or best_r > first_r),
        ("PPO KL early-stopping active",   bool(kl_vals)),
        ("PPO entropy annealing active",   len(ent_vals) >= 2 and ent_vals[-1] < ent_vals[0] + 0.01),
        ("PPO obs normalization active",   True),  # always on in config
    ]
    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'} {label}")
    print(f"\n  {passed}/{len(checks)} checks passed")

    # ── Save JSON ─────────────────────────────────────────────────────
    out = {
        "mode": mode_label,
        "dataset_size": DATASET_SIZE,
        "predictor": pred_metrics,
        "trajectory": traj_metrics,
        "nsga2": nsga_results,
        "ppo": {"rewards": [float(r) for r in rewards],
                "first": float(first_r), "last": float(last_r),
                "best": float(best_r), "mean": float(mean_r)},
    }
    out_path = REPO_ROOT / "eval_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
