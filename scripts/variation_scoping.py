"""Does aging-aware mapping beat round-robin once components differ? Physics model only, no training.

Without process variation every MAC cluster ages identically, so spreading layers
(round-robin) is already optimal and NSGA-II has nothing to find. This script adds
per-node lognormal variation (fixed per simulated chip) to the physics label model
and compares round-robin with NSGA-II run directly on that model.

    py -3.11 scripts/variation_scoping.py                    # released 4x4 PE configuration, ~5 min
    py -3.11 scripts/variation_scoping.py --pe-array 16,16   # intended configuration (TRAINING_PLAN E0)

Writes paper/tables/variation_scoping.json (variation_scoping_pe16x16.json for --pe-array 16,16).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
SIGMAS = [0.0, 0.1, 0.2]
CHIPS = [42, 43, 44]  # variation_seed: one simulated die each


def main() -> int:
    pe = sys.argv[sys.argv.index("--pe-array") + 1] if "--pe-array" in sys.argv else "4,4"
    sys.argv = [sys.argv[0], "--quality", "--pe-array", pe]
    import numpy as np
    from omegaconf import OmegaConf
    import run_eval
    from aging_models.aging_label_generator import AgingLabelGenerator
    from features.feature_builder import FeatureBuilder
    from graph.accelerator_graph import AcceleratorGraph
    from optimization.nsga2_optimizer import NSGA2Optimizer
    from simulator.timeloop_runner import TimeloopRunner
    from simulator.workload_runner import WorkloadRunner
    from utils.runtime_eval import simulate_mapping

    cfg = run_eval.build_config(epochs=1, patience=1)
    acc = cfg.accelerator
    graph = AcceleratorGraph(acc); graph.build()
    sim, fb, wr = TimeloopRunner(acc), FeatureBuilder(acc), WorkloadRunner(cfg.workloads)
    nc = int(acc.mac_clusters)
    out = {"pe_array": pe, "stress_time_s": 360_000.0, "nsga": {"pop": 20, "gens": 20, "seed": 42}, "results": []}

    # Where is the physics peak without variation? 200 uniformly random mappings per workload.
    ag0 = AgingLabelGenerator(cfg=cfg)
    rng = np.random.default_rng(0)
    floor = {}
    for wl in wr.available_workloads:
        layers = wr.get_workload_layers(wl)
        scores = lambda mp: simulate_mapping(simulator=sim, feature_builder=fb, graph=graph, layers=layers, mapping=mp,
                                             workload_name=wl, stress_time_s=360_000.0, aging_generator=ag0)["aging_scores"]
        st, rr = scores(np.zeros(len(layers), dtype=int)), scores(np.arange(len(layers)) % nc)
        rand = [scores(rng.integers(0, nc, len(layers))) for _ in range(200)]
        peaks = np.array([s.max() for s in rand])
        floor[wl] = {"static_peak": float(st.max()), "static_peak_node": int(st.argmax()),
                     "rr_peak": float(rr.max()), "rr_peak_node": int(rr.argmax()), "rr_mac_max": float(rr[:nc].max()),
                     "random_peak_std": float(peaks.std()), "random_peak_min": float(peaks.min()), "random_peak_max": float(peaks.max()),
                     "random_peak_on_router_frac": float(np.mean([int(s.argmax()) >= graph.get_num_nodes() - 4 for s in rand]))}
        print(f"floor {wl:16s} {floor[wl]}", flush=True)
    out["peak_location_no_variation"] = floor

    for sigma in SIGMAS:
        for chip in (CHIPS if sigma > 0 else CHIPS[:1]):
            c = OmegaConf.merge(cfg, {"aging": {"stochastic_variation": sigma > 0, "variation_sigma": sigma, "variation_seed": chip}})
            ag = AgingLabelGenerator(cfg=c)
            for wl in wr.available_workloads:
                layers = wr.get_workload_layers(wl)
                L = len(layers)
                peak = lambda mp: simulate_mapping(simulator=sim, feature_builder=fb, graph=graph, layers=layers, mapping=mp,
                                                   workload_name=wl, stress_time_s=360_000.0, aging_generator=ag)["peak_aging"]
                opt = NSGA2Optimizer(acc, sim, None, OmegaConf.create({
                    "pop_size": 20, "population_size": 20, "crossover_prob": 0.9, "mutation_prob": 0.1, "n_gen": 20,
                    "seed": 42, "balance_weight": 0.3, "convergence_patience": 8, "workload_name": wl}), aging_generator=ag)
                pareto = opt.run(np.zeros(L, dtype=np.int32), n_gen=20, workload_name=wl)
                static, rr = peak(np.zeros(L, dtype=int)), peak(np.arange(L) % nc)
                best = min(peak(s.mapping) for s in pareto)
                row = {"sigma": sigma, "chip": chip, "workload": wl, "static": static, "round_robin": rr, "nsga_physics": best,
                       "nsga_vs_rr_pct": (rr - best) / rr * 100}
                out["results"].append(row)
                print(f"sigma={sigma:.1f} chip={chip} {wl:16s} static {static:.4f}  RR {rr:.4f}  NSGA {best:.4f}  "
                      f"NSGA vs RR {row['nsga_vs_rr_pct']:+.2f}%", flush=True)

    summary = {}
    for sigma in SIGMAS:
        v = [r["nsga_vs_rr_pct"] for r in out["results"] if r["sigma"] == sigma]
        summary[str(sigma)] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
        print(f"sigma={sigma}: NSGA-II (physics) vs round-robin {summary[str(sigma)]['mean']:+.2f}% ± {summary[str(sigma)]['std']:.2f}")
    out["summary_nsga_vs_rr_pct"] = summary
    path = REPO_ROOT / "paper" / "tables" / ("variation_scoping.json" if pe == "4,4" else f"variation_scoping_pe{pe.replace(',', 'x')}.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
