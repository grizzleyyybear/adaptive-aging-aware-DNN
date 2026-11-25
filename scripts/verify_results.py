"""Re-evaluate the released checkpoints and check them against eval_results.json.

No training: rebuilds the same seeded dataset and 80/10/10 split used by
run_eval.py, loads checkpoints/*_best.pt, and recomputes test-set metrics.

    py -3.11 scripts/verify_results.py            # uses eval_results.json settings
    py -3.11 scripts/verify_results.py --tol 1e-3
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

KEYS = ["mae", "rmse", "r2", "spearman", "baseline_mean_mae", "skill_vs_mean"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO_ROOT / "eval_results.json"))
    ap.add_argument("--ckpt-dir", default=str(REPO_ROOT / "checkpoints"))
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    ap.add_argument("--tol", type=float, default=2e-3, help="max abs diff per metric")
    ap.add_argument("--pe-array", default="4,4", help="must match run_eval --pe-array of the run")
    ap.add_argument("--aging-variation", default="0.0", help="must match run_eval --aging-variation of the run")
    args = ap.parse_args()

    reported = json.loads(Path(args.results).read_text(encoding="utf-8"))
    size = int(reported["dataset_size"])
    # run_eval reads its settings from argv at import time.
    sys.argv = [sys.argv[0], "--quality", "--dataset-size", str(size), "--pe-array", args.pe_array,
                "--aging-variation", args.aging_variation]
    import torch
    import run_eval
    from graph.graph_dataset import AgingDataset
    from models.hybrid_gnn_transformer import HybridGNNTransformer
    from models.trajectory_predictor import TrajectoryPredictor
    from models.training_pipeline import TrainingPipeline

    torch.manual_seed(42)
    cfg = run_eval.build_config(epochs=1, patience=1)
    dataset = AgingDataset(root=args.data_root, split="train", size=size, cfg=cfg, seed=42)
    feat_dim = int(dataset[0].x.shape[1])
    m = cfg.model

    def encoder():
        return HybridGNNTransformer(
            node_feature_dim=feat_dim, hidden_dim=m.hidden_dim, gat_heads=m.gat_heads,
            transformer_layers=m.transformer_layers, transformer_heads=m.transformer_heads,
            seq_len=m.prediction_horizon,
        )

    pred = encoder()
    traj = TrajectoryPredictor(gnn_encoder=encoder(), hidden_dim=m.hidden_dim, horizon=m.prediction_horizon, gamma=0.95)
    ckpt = Path(args.ckpt_dir)
    pred.load_state_dict(torch.load(ckpt / "predictor_best.pt", map_location="cpu"))
    traj.load_state_dict(torch.load(ckpt / "trajectory_best.pt", map_location="cpu"))

    failures = 0
    for name, model in (("predictor", pred), ("trajectory", traj)):
        got = TrainingPipeline(cfg, model, dataset, checkpoint_dir=None, resume=False).evaluate("test")
        print(f"\n[{name}]  {'metric':<20}{'reported':>12}{'recomputed':>12}{'diff':>10}")
        for k in KEYS:
            if k not in reported[name]:
                continue
            diff = abs(float(got[k]) - float(reported[name][k]))
            ok = diff <= args.tol
            failures += not ok
            print(f"  {'' if ok else 'FAIL '}{k:<20}{reported[name][k]:>12.4f}{got[k]:>12.4f}{diff:>10.2e}")

    print("\nALL METRICS REPRODUCED" if failures == 0 else f"\n{failures} METRIC(S) DO NOT MATCH")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
