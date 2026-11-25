"""Evaluate a trained PPO policy against fixed policies on the same seeded episodes.

No training. Loads rl_policy_{best,final}.pt and its *_obs_norm.pt from --ckpt-dir,
acts greedily, and scores total reward per episode over --episodes seeded resets
(seeds 1000+ep, the same as scripts/measure_paper_figures.py).

    py -3.11 scripts/eval_ppo_policy.py --ckpt-dir runs/ppo_s1
    py -3.11 scripts/eval_ppo_policy.py --ckpt-dir runs/ppo_pred_s1 --use-predictor   # env built as in --ppo-use-predictor
"""
import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--which", choices=["best", "final"], default="best")
    ap.add_argument("--hidden", type=int, default=128, help="must match --ppo-hidden of the training run")
    ap.add_argument("--use-predictor", action="store_true", help="must match --ppo-use-predictor of the training run")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--pe-array", default="4,4", help="must match run_eval --pe-array of the training run")
    ap.add_argument("--aging-variation", default="0.0", help="must match run_eval --aging-variation of the training run")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()
    ckpt = Path(args.ckpt_dir)

    sys.argv = [sys.argv[0], "--quality", "--pe-array", args.pe_array, "--aging-variation", args.aging_variation]
    import numpy as np
    import torch
    import run_eval
    from graph.accelerator_graph import AcceleratorGraph
    from planning.lifetime_planner import LifetimePlanner
    from rl.environment import AgingControlEnv
    from rl.policy_network import ActorCritic
    from simulator.timeloop_runner import TimeloopRunner

    cfg = run_eval.build_config(epochs=1, patience=1)
    graph = AcceleratorGraph(cfg.accelerator); graph.build()
    sim = TimeloopRunner(cfg.accelerator)
    planner = LifetimePlanner(graph, cfg.planning)
    if args.use_predictor:
        from aging_models.aging_label_generator import AgingLabelGenerator
        from features.feature_builder import FeatureBuilder
        from models.hybrid_gnn_transformer import HybridGNNTransformer
        from models.trajectory_predictor import TrajectoryPredictor
        from simulator.workload_runner import WorkloadRunner
        m = cfg.model
        enc = lambda: HybridGNNTransformer(node_feature_dim=8, hidden_dim=m.hidden_dim, gat_heads=m.gat_heads,
                                           transformer_layers=m.transformer_layers, transformer_heads=m.transformer_heads,
                                           seq_len=m.prediction_horizon)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred = enc(); pred.load_state_dict(torch.load(ckpt / "predictor_best.pt", map_location="cpu"))
        traj = TrajectoryPredictor(gnn_encoder=enc(), hidden_dim=m.hidden_dim, horizon=m.prediction_horizon, gamma=0.95)
        traj.load_state_dict(torch.load(ckpt / "trajectory_best.pt", map_location="cpu"))
        pred.to(dev).eval(); traj.to(dev).eval()
        env = AgingControlEnv(sim, planner, WorkloadRunner(cfg.workloads), AgingLabelGenerator(cfg=cfg),
                              graph, pred, traj, FeatureBuilder(cfg.accelerator), cfg)
    else:
        env = AgingControlEnv(cfg, sim, planner)

    policy = ActorCritic(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.n, hidden_dim=args.hidden)
    policy.load_state_dict(torch.load(ckpt / f"rl_policy_{args.which}.pt", map_location="cpu"))
    policy.eval()
    norm_path = ckpt / f"rl_policy_{args.which}_obs_norm.pt"
    if not norm_path.exists():
        raise SystemExit(f"{norm_path} missing: this checkpoint predates obs-norm saving and cannot be evaluated faithfully.")
    norm = torch.load(norm_path, map_location="cpu", weights_only=False)
    mean, std = np.asarray(norm["mean"], np.float32), np.sqrt(np.asarray(norm["var"], np.float32) + 1e-8)

    def ppo(e):
        obs = torch.as_tensor((e._get_obs() - mean) / std, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return int(policy.get_action(obs, deterministic=True)[0].item())

    def greedy(e):
        return int(np.argmax([copy.deepcopy(e).step(a)[1] for a in range(e.action_space.n)]))

    rng = np.random.default_rng(123)
    policies = {"PPO": ppo, "Random": lambda e: int(rng.integers(0, e.action_space.n)),
                "Hold": lambda e: 0, "Rotate (round-robin)": lambda e: 2, "Greedy one-step oracle": greedy}
    res = {}
    for name, fn in policies.items():
        totals = []
        for ep in range(args.episodes):
            env.reset(seed=1000 + ep)
            total, done = 0.0, False
            while not done:
                _, r, term, trunc, _ = env.step(fn(env))
                total += float(r); done = bool(term or trunc)
            totals.append(total)
        res[name] = {"mean": float(np.mean(totals)), "std": float(np.std(totals))}
        print(f"{name:24s} {res[name]['mean']:.3f} ± {res[name]['std']:.3f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"ckpt_dir": str(ckpt), "which": args.which, "use_predictor": args.use_predictor,
                                              "episodes": args.episodes, "policies": res}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
