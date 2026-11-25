"""Generate the paper's supplemental figures from measured data only.

Every number plotted here comes from the released checkpoints, the analytical
simulator, or eval_results.json. No training is performed.

    py -3.11 scripts/measure_paper_figures.py

Writes paper/assets/fig_*.{pdf,png} and paper/tables/measured.json.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ASSETS = REPO_ROOT / "paper" / "assets"
TABLES = REPO_ROOT / "paper" / "tables"

import torch_geometric  # noqa: F401  must precede matplotlib on Windows (pyarrow init fails with WinError 6714 otherwise)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(ASSETS / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def _r2(t, p):
    ss = float(((t - t.mean()) ** 2).sum())
    return 1.0 - float(((t - p) ** 2).sum()) / ss if ss > 1e-12 else float("nan")


def main() -> int:
    results = json.loads((REPO_ROOT / "eval_results.json").read_text(encoding="utf-8"))
    sys.argv = [sys.argv[0], "--quality", "--dataset-size", str(results["dataset_size"])]
    import torch
    from torch_geometric.loader import DataLoader
    import run_eval
    from graph.accelerator_graph import AcceleratorGraph
    from graph.graph_dataset import AgingDataset
    from models.hybrid_gnn_transformer import HybridGNNTransformer
    from models.trajectory_predictor import TrajectoryPredictor
    from optimization.nsga2_optimizer import NSGA2Optimizer
    from simulator.timeloop_runner import TimeloopRunner
    from simulator.workload_runner import WorkloadRunner
    from features.feature_builder import FeatureBuilder
    from utils.runtime_eval import simulate_mapping
    from omegaconf import OmegaConf

    torch.manual_seed(42)
    np.random.seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = run_eval.build_config(epochs=1, patience=1)
    m, acc = cfg.model, cfg.accelerator
    measured = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"}

    dataset = AgingDataset(root=str(REPO_ROOT / "data"), split="train", size=int(results["dataset_size"]), cfg=cfg, seed=42)
    feat_dim = int(dataset[0].x.shape[1])

    def encoder():
        return HybridGNNTransformer(node_feature_dim=feat_dim, hidden_dim=m.hidden_dim, gat_heads=m.gat_heads,
                                    transformer_layers=m.transformer_layers, transformer_heads=m.transformer_heads,
                                    seq_len=m.prediction_horizon)

    pred = encoder()
    pred.load_state_dict(torch.load(REPO_ROOT / "checkpoints/predictor_best.pt", map_location="cpu"))
    traj = TrajectoryPredictor(gnn_encoder=encoder(), hidden_dim=m.hidden_dim, horizon=m.prediction_horizon, gamma=0.95)
    traj.load_state_dict(torch.load(REPO_ROOT / "checkpoints/trajectory_best.pt", map_location="cpu"))
    pred.to(dev).eval()
    traj.to(dev).eval()

    # Same 80/10/10 split and seed as models/training_pipeline.py.
    n = len(dataset)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    _, _, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n - n_train - n_val], generator=torch.Generator().manual_seed(42))

    # ── Trajectory accuracy per step and per workload ─────────────────────
    workloads = list(WorkloadRunner(cfg.workloads).available_workloads)
    preds, targets, wl_of_node = [], [], []
    with torch.no_grad():
        for b in DataLoader(test_set, batch_size=64, shuffle=False):
            b = b.to(dev)
            preds.append(traj(b.x, b.edge_index, b.edge_attr, b.batch).cpu().numpy())
            targets.append(b.y_trajectory.cpu().numpy())
            wl_graph = b.workload_emb.view(b.num_graphs, -1).argmax(dim=1).cpu().numpy()
            wl_of_node.append(wl_graph[b.batch.cpu().numpy()])
    P, T, W = np.concatenate(preds), np.concatenate(targets), np.concatenate(wl_of_node)
    K = P.shape[1]
    steps = np.arange(1, K + 1)
    per_step_r2 = [_r2(T[:, k], P[:, k]) for k in range(K)]
    per_step_mae = [float(np.abs(T[:, k] - P[:, k]).mean()) for k in range(K)]
    per_wl = {}
    for i, wl in enumerate(workloads):
        sel = W == i
        if sel.sum() > 1:
            per_wl[wl] = {"nodes": int(sel.sum()), "graphs": int(sel.sum() // dataset[0].num_nodes),
                          "per_step_r2": [_r2(T[sel, k], P[sel, k]) for k in range(K)]}
    measured["trajectory_test"] = {"test_graphs": len(test_set), "per_step_r2": per_step_r2,
                                   "per_step_mae": per_step_mae, "per_workload": per_wl}

    fig, ax1 = plt.subplots(figsize=(6.5, 3.8))
    ax1.plot(steps, per_step_r2, "o-", color="#2a6fbb", label="$R^2$")
    ax1.set_xlabel("Forecast step $k$"); ax1.set_ylabel("$R^2$", color="#2a6fbb"); ax1.set_xticks(steps)
    ax2 = ax1.twinx()
    ax2.plot(steps, per_step_mae, "s--", color="#c0392b", label="MAE")
    ax2.set_ylabel("MAE", color="#c0392b"); ax2.grid(False)
    ax1.set_title(f"Trajectory head, held-out test set ({len(test_set)} graphs)")
    _save(fig, "fig_trajectory_per_step")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for wl, v in per_wl.items():
        ax.plot(steps, v["per_step_r2"], marker="o", ms=3.5, label=f"{wl} (n={v['graphs']})")
    ax.set_xlabel("Forecast step $k$"); ax.set_ylabel("$R^2$"); ax.set_xticks(steps)
    ax.set_title("Per-workload trajectory $R^2$ (held-out test set)")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig_per_workload_trajectory_r2")

    # ── Model size (latency is not measured here: it needs an idle machine) ─
    graph = AcceleratorGraph(acc); graph.build()
    measured["model_size"] = {
        name: {"params": sum(p.numel() for p in model.parameters()),
               "fp32_mb": sum(p.numel() for p in model.parameters()) * 4 / 2**20}
        for name, model in (("Predictor", pred), ("Predictor + trajectory", traj))}

    # ── NSGA-II, all workloads, with the evaluated predictor ──────────────
    # Reduction is measured on the *true* simulated peak of each mapping, not on
    # the optimizer's f1 (which adds a 0.3*Var penalty). Baseline: all layers on
    # cluster 0 ("static_zero"), the same reference run_eval.py uses.
    sim, fb, wr = TimeloopRunner(acc), FeatureBuilder(acc), WorkloadRunner(cfg.workloads)
    pop, gens = 20, 20  # settings of the promoted run (README)
    nsga_out, fronts = {}, {}
    for wl in workloads:
        layers = wr.get_workload_layers(wl)
        init_map = np.zeros(len(layers), dtype=np.int32)
        opt = NSGA2Optimizer(acc, sim, pred, OmegaConf.create({
            "pop_size": pop, "population_size": pop, "crossover_prob": 0.9, "mutation_prob": 0.1,
            "n_gen": gens, "seed": 42, "balance_weight": 0.3, "convergence_patience": 8, "workload_name": wl}))
        pareto = opt.run(init_map, n_gen=gens, workload_name=wl)
        run = lambda mp, wl=wl, layers=layers: simulate_mapping(
            simulator=sim, feature_builder=fb, graph=graph, layers=layers, mapping=mp,
            workload_name=wl, stress_time_s=360_000.0, predictor=pred, device=dev)
        before = run(init_map)
        sims = [run(s.mapping) for s in pareto]
        best_i = int(np.argmin([s["peak_aging"] for s in sims]))
        after = sims[best_i]
        red = (before["peak_aging"] - after["peak_aging"]) / before["peak_aging"] * 100
        nsga_out[wl] = {"layers": len(layers), "pareto_count": len(pareto), "generations": len(opt.hv_history),
                        "cache_hits": int(opt._cache.hits), "init_peak": before["peak_aging"],
                        "best_true_peak": after["peak_aging"], "reduction_pct": red,
                        "best_latency_norm": pareto[best_i].latency, "best_energy_norm": pareto[best_i].energy}
        fronts[wl] = (pareto, before, after)
        print(f"  NSGA-II {wl}: {len(pareto)} sols, true peak {before['peak_aging']:.4f} -> {after['peak_aging']:.4f} ({red:.2f}%)")
    measured["nsga2"] = nsga_out
    wl = "BERT-Base"
    pareto, before, after = fronts[wl]

    F = np.array([[s.peak_aging, s.latency, s.energy] for s in pareto])
    fig = plt.figure(figsize=(7, 5.2))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(F[:, 0], F[:, 1], F[:, 2], c=F[:, 0], cmap="viridis", s=30, depthshade=False)
    ax.set_xlabel("$f_1$ peak aging + balance", fontsize=8, labelpad=8)
    ax.set_ylabel("$f_2$ latency ($10^8$ cycles)", fontsize=8, labelpad=8)
    ax.set_zlabel("$f_3$ energy ($10^9$ pJ)", fontsize=8, labelpad=10)
    ax.tick_params(labelsize=7)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12, label="$f_1$")
    ax.set_title(f"{wl} Pareto front ({len(pareto)} non-dominated mappings)", fontsize=10)
    _save(fig, "fig_pareto_front_3d")

    nm = int(acc.mac_clusters); side = int(round(nm ** 0.5))
    hb = before["aging_scores"][:nm].reshape(side, side)
    ha = after["aging_scores"][:nm].reshape(side, side)
    vmax = float(max(hb.max(), ha.max()))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    for ax, h, d, title in ((axes[0], hb, before, "Static (all layers on cluster 0)"),
                            (axes[1], ha, after, "NSGA-II min-peak mapping")):
        title = f"{title}\nMAC max {h.max():.3f}\nall-node peak {d['peak_aging']:.3f}"
        im = ax.imshow(h, cmap="inferno", vmin=0, vmax=vmax)
        for (i, j), v in np.ndenumerate(h):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if v < 0.6 * vmax else "k")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="Predicted aging score")
    fig.suptitle(f"{wl}: predicted per-MAC-cluster aging", fontsize=10)
    _save(fig, "fig_aging_heatmap_before_after")
    measured["heatmap_bert"] = {"before_mac": hb.tolist(), "after_mac": ha.tolist()}

    # ── PPO vs baselines on one scale: total reward per 10-step episode ───
    # eval_results.json stores PPO as episode *sums* but the baselines as
    # per-step *means*; re-run the (non-learned) baselines as episode sums.
    from planning.lifetime_planner import LifetimePlanner
    from rl.environment import AgingControlEnv
    # The released PPO policy was trained in run_eval.py's 16x16-PE environment (the dataset and
    # predictor use 4x4; see run_eval.PE_ARRAY), so fixed policies are compared in that environment.
    cfg16 = OmegaConf.merge(cfg, {"accelerator": {"pe_array": [16, 16], "pe_array_rows": 16, "pe_array_cols": 16}})
    env = AgingControlEnv(cfg16, TimeloopRunner(cfg16.accelerator), LifetimePlanner(graph, cfg.planning))
    rng = np.random.default_rng(123)
    policies = {
        "Random": lambda e: int(rng.integers(0, e.action_space.n)),
        "Threshold": lambda e: 1 if float(np.max(e.aging_vector)) >= 0.5 * e.failure_threshold else 0,
        "Round-robin": lambda e: 2,
        "Hold (no-op)": lambda e: 0,
    }
    n_ep = 20
    base = {}
    for name, fn in policies.items():
        totals = []
        for ep in range(n_ep):
            env.reset(seed=1000 + ep)
            total, done = 0.0, False
            while not done:
                _, r, term, trunc, _ = env.step(fn(env))
                total += float(r); done = bool(term or trunc)
            totals.append(total)
        base[name] = {"mean": float(np.mean(totals)), "std": float(np.std(totals)), "episodes": n_ep}
    ppo = results["ppo"]
    measured["ppo_vs_baselines_episode_sum"] = {"baselines": base, "ppo_training": {
        k: ppo[k] for k in ("first", "last", "mean", "best")}, "episode_steps": int(env.max_steps)}
    labels = list(base) + ["PPO train\n(mean)", "PPO train\n(best iter.)"]
    vals = [base[k]["mean"] for k in base] + [ppo["mean"], ppo["best"]]
    errs = [base[k]["std"] for k in base] + [0, 0]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bs = ax.bar(labels, vals, yerr=errs, capsize=3, color=["#95a5a6"] * len(base) + ["#2a6fbb", "#1b4f8a"])
    ax.bar_label(bs, fmt="%.3f", fontsize=8, padding=2)
    ax.set_ylabel(f"Total reward per {env.max_steps}-step episode")
    ax.set_title(f"Runtime policies, same reward and scale (baselines: {n_ep} seeded episodes, ±1 s.d.)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    _save(fig, "fig_ppo_baseline_comparison")

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "measured.json").write_text(json.dumps(measured, indent=2, default=float), encoding="utf-8")
    print(f"  wrote {TABLES / 'measured.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
