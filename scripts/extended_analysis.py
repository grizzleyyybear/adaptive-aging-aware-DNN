"""Robustness analyses that need no training.

Uses only the released checkpoints, the simulator and the physics label model:
  1. inference latency (GPU / CPU, single graph and batch of 64)
  2. bootstrap 95% CIs for test metrics, and per-node-type errors
  3. mapping study: NSGA-II over 5 optimizer seeds, a same-budget random search,
     round-robin, a larger NSGA-II budget, and NSGA-II run directly on the physics
     model; every chosen mapping is scored by both the predictor and the physics model
  4. runtime policies: every constant action and a greedy one-step oracle

    py -3.11 scripts/extended_analysis.py          # released checkpoints, ~30 min on an RTX 3050 laptop
    py -3.11 scripts/extended_analysis.py --ckpt-dir runs/var0.1_s42 --aging-variation 0.1 --tag var0.1

Writes paper/tables/extended{_TAG}.json and paper/assets/fig_mapping_study{_TAG}.{pdf,png}.
Run on an idle machine: section 1 is a timing measurement.
"""
import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ASSETS = REPO_ROOT / "paper" / "assets"
TABLES = REPO_ROOT / "paper" / "tables"

import torch_geometric  # noqa: F401  must precede matplotlib on Windows (WinError 6714)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

NSGA_SEEDS = [42, 1, 2, 3, 4]
REF_T = 360_000.0
NODE_TYPES = {"MAC": slice(0, 16), "SRAM": slice(16, 24), "Router": slice(24, 28)}


def _timeit(fn, cuda, n=300, warm=50):
    import torch
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1e3
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def _ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--dataset-size", type=int, default=None, help="default: eval_results.json")
    ap.add_argument("--aging-variation", type=float, default=0.0, help="must match the training run")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--pe-array", default="4,4", help="must match the training run (run_eval --pe-array)")
    ap.add_argument("--latency-only", action="store_true", help="only re-measure latency and update the JSON")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    ckpt = Path(args.ckpt_dir)
    size = args.dataset_size or json.loads((REPO_ROOT / "eval_results.json").read_text(encoding="utf-8"))["dataset_size"]
    sys.argv = [sys.argv[0], "--quality", "--dataset-size", str(size), "--aging-variation", str(args.aging_variation), "--pe-array", args.pe_array]
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
    from omegaconf import OmegaConf
    import run_eval
    from aging_models.aging_label_generator import AgingLabelGenerator
    from features.feature_builder import FeatureBuilder
    from graph.accelerator_graph import AcceleratorGraph
    from graph.graph_dataset import AgingDataset
    from models.hybrid_gnn_transformer import HybridGNNTransformer
    from models.trajectory_predictor import TrajectoryPredictor
    from optimization.nsga2_optimizer import NSGA2Optimizer
    from planning.lifetime_planner import LifetimePlanner
    from rl.environment import AgingControlEnv
    from simulator.timeloop_runner import TimeloopRunner
    from simulator.workload_runner import WorkloadRunner
    from utils.runtime_eval import simulate_mapping

    torch.manual_seed(42)
    np.random.seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = run_eval.build_config(epochs=1, patience=1)
    m, acc = cfg.model, cfg.accelerator
    out = {"ckpt_dir": str(ckpt), "aging_variation": args.aging_variation, "dataset_size": int(size), "pe_array": args.pe_array}

    dataset = AgingDataset(root=str(REPO_ROOT / "data"), split="train", size=int(size), cfg=cfg, seed=42)
    feat_dim = int(dataset[0].x.shape[1])

    def encoder():
        return HybridGNNTransformer(node_feature_dim=feat_dim, hidden_dim=m.hidden_dim, gat_heads=m.gat_heads,
                                    transformer_layers=m.transformer_layers, transformer_heads=m.transformer_heads,
                                    seq_len=m.prediction_horizon)

    pred = encoder()
    pred.load_state_dict(torch.load(ckpt / "predictor_best.pt", map_location="cpu"))
    traj = TrajectoryPredictor(gnn_encoder=encoder(), hidden_dim=m.hidden_dim, horizon=m.prediction_horizon, gamma=0.95)
    traj.load_state_dict(torch.load(ckpt / "trajectory_best.pt", map_location="cpu"))
    pred.eval(), traj.eval()

    n = len(dataset)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    _, _, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n - n_train - n_val], generator=torch.Generator().manual_seed(42))

    # ── 1. Inference latency ───────────────────────────────────────────────
    print("[1/4] latency", flush=True)
    one = test_set[0]
    batch64 = Batch.from_data_list([test_set[i] for i in range(64)])
    lat = {"cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
    for dname in (["cuda"] if dev.type == "cuda" else []) + ["cpu"]:
        d = torch.device(dname)
        for mname, model in (("predictor", pred), ("predictor+trajectory", traj)):
            model.to(d)
            for bname, b in (("1 graph", one), ("64 graphs", batch64)):
                b = b.to(d)
                bb = getattr(b, "batch", None)
                with torch.no_grad():
                    r = _timeit(lambda: model(b.x, b.edge_index, b.edge_attr, bb), cuda=dname == "cuda",
                                n=100 if (dname == "cpu" and bname == "64 graphs") else 300)
                lat[f"{dname} | {mname} | {bname}"] = r
                print(f"  {dname:4s} {mname:22s} {bname:9s} median {r['median_ms']:.2f} ms  p95 {r['p95_ms']:.2f} ms")
    out["latency"] = lat
    pred.to(dev), traj.to(dev)
    if args.latency_only:  # re-time on an idle machine without redoing sections 2-4
        path = TABLES / f"extended{sfx}.json"
        prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        prev["latency"] = lat
        path.write_text(json.dumps(prev, indent=2, default=float), encoding="utf-8")
        print(f"updated latency in {path}")
        return 0

    # ── 2. Bootstrap CIs and per-node-type errors ──────────────────────────
    print("[2/4] bootstrap", flush=True)
    P, T, Pt, Tt = [], [], [], []
    with torch.no_grad():
        for b in DataLoader(test_set, batch_size=1, shuffle=False):
            b = b.to(dev)
            P.append(pred(b.x, b.edge_index, b.edge_attr, b.batch).view(-1).cpu().numpy())
            T.append(b.y.view(-1).cpu().numpy())
            Pt.append(traj(b.x, b.edge_index, b.edge_attr, b.batch).cpu().numpy())
            Tt.append(b.y_trajectory.cpu().numpy())
    P, T, Pt, Tt = map(np.stack, (P, T, Pt, Tt))  # [G, N] and [G, N, K]
    G = P.shape[0]
    rng = np.random.default_rng(0)
    boot = {"predictor": {"r2": [], "mae": []}, "trajectory": {"r2": [], "mae": []}}
    for _ in range(2000):
        idx = rng.integers(0, G, G)
        p, t = P[idx].ravel(), T[idx].ravel()
        boot["predictor"]["r2"].append(r2_score(t, p)); boot["predictor"]["mae"].append(mean_absolute_error(t, p))
        pt, tt = Pt[idx].reshape(-1, Pt.shape[2]), Tt[idx].reshape(-1, Tt.shape[2])
        boot["trajectory"]["r2"].append(r2_score(tt, pt)); boot["trajectory"]["mae"].append(mean_absolute_error(tt, pt))
    out["bootstrap_95ci"] = {
        "resamples": 2000, "unit": "test graph", "test_graphs": int(G),
        "predictor": {"r2": r2_score(T.ravel(), P.ravel()), "r2_ci": _ci(boot["predictor"]["r2"]),
                      "mae": mean_absolute_error(T.ravel(), P.ravel()), "mae_ci": _ci(boot["predictor"]["mae"])},
        "trajectory": {"r2": r2_score(Tt.reshape(-1, Tt.shape[2]), Pt.reshape(-1, Pt.shape[2])),
                       "r2_ci": _ci(boot["trajectory"]["r2"]),
                       "mae": mean_absolute_error(Tt.ravel(), Pt.ravel()), "mae_ci": _ci(boot["trajectory"]["mae"])},
    }
    out["per_node_type"] = {
        k: {"nodes": int(P[:, s].size), "target_mean": float(T[:, s].mean()), "target_std": float(T[:, s].std()),
            "mae": float(mean_absolute_error(T[:, s].ravel(), P[:, s].ravel())),
            "r2": float(r2_score(T[:, s].ravel(), P[:, s].ravel())),
            "traj_mae": float(mean_absolute_error(Tt[:, s].ravel(), Pt[:, s].ravel()))}
        for k, s in NODE_TYPES.items()}
    print(json.dumps(out["bootstrap_95ci"], indent=1, default=float))
    print(json.dumps(out["per_node_type"], indent=1))

    # ── 3. Mapping study ────────────────────────────────────────────────────
    print("[3/4] mapping study", flush=True)
    graph = AcceleratorGraph(acc); graph.build()
    sim, fb, wr = TimeloopRunner(acc), FeatureBuilder(acc), WorkloadRunner(cfg.workloads)
    ag = AgingLabelGenerator(cfg=cfg)
    nc = int(acc.mac_clusters)

    def nsga(wl, seed, pop, gens, surrogate=True):
        opt = NSGA2Optimizer(acc, sim, pred if surrogate else None, OmegaConf.create({
            "pop_size": pop, "population_size": pop, "crossover_prob": 0.9, "mutation_prob": 0.1, "n_gen": gens,
            "seed": seed, "balance_weight": 0.3, "convergence_patience": 8, "workload_name": wl}),
            aging_generator=None if surrogate else ag)
        layers = wr.get_workload_layers(wl)
        return opt.run(np.zeros(len(layers), dtype=np.int32), n_gen=gens, workload_name=wl), opt._cache.misses

    study, fidelity_all = {}, []
    for wl in wr.available_workloads:
        layers = wr.get_workload_layers(wl)
        L = len(layers)

        def score(mp, wl=wl, layers=layers):
            kw = dict(simulator=sim, feature_builder=fb, graph=graph, layers=layers, mapping=mp,
                      workload_name=wl, stress_time_s=REF_T)
            a = simulate_mapping(predictor=pred, device=dev, **kw)
            b = simulate_mapping(aging_generator=ag, **kw)
            return {"pred": a["peak_aging"], "phys": b["peak_aging"],
                    "latency_norm": a["latency_norm"], "energy_norm": a["energy_norm"]}

        def pick(mappings, key):
            scored = [score(mp) for mp in mappings]
            return scored[int(np.argmin([s[key] for s in scored]))]

        row = {"layers": L, "static_zero": score(np.zeros(L, dtype=int)),
               "round_robin": score(np.arange(L) % nc),
               "nsga_surrogate": [], "random_search": [], "nsga_physics": []}
        for seed in NSGA_SEEDS:
            pareto, evals = nsga(wl, seed, 20, 20)
            row["nsga_surrogate"].append({"seed": seed, "evals": evals,
                                          **pick([s.mapping for s in pareto], "pred")})
            rs = np.random.default_rng(seed).integers(0, nc, size=(evals, L))
            rs_scores = [score(mp) for mp in rs]
            fidelity_all += [(wl, s["pred"], s["phys"]) for s in rs_scores]
            row["random_search"].append({"seed": seed, "evals": evals,
                                         **rs_scores[int(np.argmin([s["pred"] for s in rs_scores]))]})
            pareto, evals = nsga(wl, seed, 20, 20, surrogate=False)
            row["nsga_physics"].append({"seed": seed, "evals": evals, **pick([s.mapping for s in pareto], "phys")})
        pareto, evals = nsga(wl, 42, 60, 50)
        row["nsga_surrogate_large"] = {"pop": 60, "gens": 50, "evals": evals, **pick([s.mapping for s in pareto], "pred")}

        base = row["static_zero"]
        for k in ("nsga_surrogate", "random_search", "nsga_physics"):
            for key in ("pred", "phys"):
                v = np.array([r[key] for r in row[k]])
                red = (base[key] - v) / base[key] * 100
                row[f"{k}_{key}_reduction"] = {"mean": float(red.mean()), "std": float(red.std()), "min": float(red.min()), "max": float(red.max())}
        for key in ("pred", "phys"):
            row[f"round_robin_{key}_reduction"] = float((base[key] - row["round_robin"][key]) / base[key] * 100)
            row[f"nsga_surrogate_large_{key}_reduction"] = float((base[key] - row["nsga_surrogate_large"][key]) / base[key] * 100)
        study[wl] = row
        print(f"  {wl:16s} phys red%: RR {row['round_robin_phys_reduction']:.1f} | "
              f"NSGA-sur {row['nsga_surrogate_phys_reduction']['mean']:.1f}±{row['nsga_surrogate_phys_reduction']['std']:.1f} | "
              f"rand {row['random_search_phys_reduction']['mean']:.1f}±{row['random_search_phys_reduction']['std']:.1f} | "
              f"NSGA-phys {row['nsga_physics_phys_reduction']['mean']:.1f}±{row['nsga_physics_phys_reduction']['std']:.1f} | "
              f"NSGA-large {row['nsga_surrogate_large_phys_reduction']:.1f}", flush=True)
    out["mapping_study"] = {"stress_time_s": REF_T, "nsga_seeds": NSGA_SEEDS, "per_workload": study}
    fp = np.array([f[1] for f in fidelity_all]); fq = np.array([f[2] for f in fidelity_all])
    out["surrogate_fidelity"] = {
        "mappings": int(len(fp)), "spearman_peak": float(spearmanr(fp, fq)[0]),
        "mae_peak": float(np.abs(fp - fq).mean()), "bias_pred_minus_phys": float((fp - fq).mean()),
        "per_workload_spearman": {wl: float(spearmanr([f[1] for f in fidelity_all if f[0] == wl],
                                                      [f[2] for f in fidelity_all if f[0] == wl])[0])
                                  for wl in study}}
    print(json.dumps(out["surrogate_fidelity"], indent=1))

    wls = list(study)
    short = [w.split("-")[0] if not w.startswith("ViT") else "ViT" for w in wls]
    series = [("Round-robin", lambda r: (r["round_robin_phys_reduction"], 0)),
              ("Random search", lambda r: (r["random_search_phys_reduction"]["mean"], r["random_search_phys_reduction"]["std"])),
              ("NSGA-II (surrogate)", lambda r: (r["nsga_surrogate_phys_reduction"]["mean"], r["nsga_surrogate_phys_reduction"]["std"])),
              ("NSGA-II (physics)", lambda r: (r["nsga_physics_phys_reduction"]["mean"], r["nsga_physics_phys_reduction"]["std"]))]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
    w = 0.2
    for i, (name, f) in enumerate(series):
        vals = [f(study[wl]) for wl in wls]
        ax.bar(np.arange(len(wls)) + (i - 1.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=2, label=name)
    ax.set_xticks(np.arange(len(wls))); ax.set_xticklabels(short)
    ax.set_ylabel("Physics peak-aging reduction (%)\nvs. all layers on cluster 0")
    ax.set_title(f"Mappings scored by the physics model ({len(NSGA_SEEDS)} seeds, ±1 s.d.)", fontsize=9)
    ax.legend(fontsize=7.5, ncol=2)
    ax2.scatter(fq, fp, s=4, alpha=0.35)
    lo, hi = float(min(fp.min(), fq.min())), float(max(fp.max(), fq.max()))
    ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax2.set_xlabel("Physics-model peak aging"); ax2.set_ylabel("Predicted peak aging")
    ax2.set_title(f"Surrogate fidelity on {len(fp)} random mappings\n(Spearman $\\rho$ = {out['surrogate_fidelity']['spearman_peak']:.3f})", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ASSETS / f"fig_mapping_study{sfx}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── 4. Runtime policies: constant actions and greedy one-step oracle ──
    print("[4/4] runtime policies", flush=True)
    # The released PPO policy was trained in run_eval.py's 16x16-PE environment (the dataset and
    # predictor use 4x4; see run_eval.PE_ARRAY), so policies are compared in that environment.
    cfg16 = OmegaConf.merge(cfg, {"accelerator": {"pe_array": [16, 16], "pe_array_rows": 16, "pe_array_cols": 16}})
    env = AgingControlEnv(cfg16, TimeloopRunner(cfg16.accelerator), LifetimePlanner(graph, cfg.planning))

    def greedy(e):
        best, best_r = 0, -np.inf
        for a in range(e.action_space.n):
            r = copy.deepcopy(e).step(a)[1]
            if r > best_r:
                best, best_r = a, r
        return best

    names = {0: "Hold", 1: "Hottest-to-coolest", 2: "Rotate (round-robin)", 3: "Move one layer by 8", 4: "Planner rebalance"}
    policies = {names[a]: (lambda e, a=a: a) for a in range(5)}
    policies["Greedy one-step oracle"] = greedy
    pol = {}
    for name, fn in policies.items():
        totals = []
        for ep in range(20):
            env.reset(seed=1000 + ep)
            total, done = 0.0, False
            while not done:
                _, r, term, trunc, _ = env.step(fn(env))
                total += float(r); done = bool(term or trunc)
            totals.append(total)
        pol[name] = {"mean": float(np.mean(totals)), "std": float(np.std(totals))}
        print(f"  {name:24s} {pol[name]['mean']:.3f} ± {pol[name]['std']:.3f}", flush=True)
    out["runtime_policies_episode_sum"] = {"episodes": 20, "reset_seeds": "1000+ep", "policies": pol}

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"extended{sfx}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"wrote {TABLES / f'extended{sfx}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
