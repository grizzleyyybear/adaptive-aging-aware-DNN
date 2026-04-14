"""
Generate the paper figures from real evaluation artifacts.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import Normalize
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from graph.accelerator_graph import AcceleratorGraph
from graph.graph_dataset import AgingDataset
from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor
from utils.device import dataloader_kwargs
from utils.runtime_eval import load_pretrained_predictor, load_pretrained_trajectory


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "figure.dpi": 300,
    "font.family": "serif",
    "savefig.bbox": "tight",
})

COLOR_INITIAL = "#888888"
COLOR_NSGA = "#4682B4"
COLOR_PPO = "#2E8B57"
WORKLOAD_COLORS = {
    "ResNet-50": "coral",
    "BERT-Base": "orange",
    "MobileNetV2": "teal",
    "EfficientNet-B4": "royalblue",
    "ViT-B/16": "purple",
}
WORKLOAD_ORDER = ["ResNet-50", "BERT-Base", "MobileNetV2", "EfficientNet-B4", "ViT-B/16"]
PLOTS_DIR = REPO_ROOT / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_cfg():
    accel = OmegaConf.load(REPO_ROOT / "configs/accelerator.yaml")
    workloads = OmegaConf.load(REPO_ROOT / "configs/workloads.yaml")
    training = OmegaConf.load(REPO_ROOT / "configs/training.yaml")
    experiments = OmegaConf.load(REPO_ROOT / "configs/experiments.yaml")
    cfg = OmegaConf.merge(experiments, accel, workloads, training)
    cfg.model.prediction_horizon = 10
    cfg.runtime.device = "cuda"
    return cfg


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return path


def load_json(path: Path):
    require_file(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def savefig(fig, filename: str, generated: list[Path]):
    path = PLOTS_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    generated.append(path)
    print(f"  saved: {path.relative_to(REPO_ROOT)}")


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    if len(values) < window:
        return np.full_like(values, np.mean(values), dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def flatten_metrics(pred, true):
    pred_arr = np.asarray(pred).reshape(-1)
    true_arr = np.asarray(true).reshape(-1)
    ss_res = np.sum((pred_arr - true_arr) ** 2)
    ss_tot = np.sum((true_arr - np.mean(true_arr)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    return float(r2), mae, rmse


def load_artifacts():
    cfg = load_cfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("scripts/generate_paper_outputs.py requires CUDA for inference.")

    predictor, _, _ = load_pretrained_predictor(cfg, device_request="cuda")
    trajectory, _ = load_pretrained_trajectory(cfg, predictor, device=device)

    test_ds = AgingDataset(root="./data", split="test", size=5000, cfg=cfg, seed=44)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, **dataloader_kwargs(device))
    num_nodes = int(test_ds[0].num_nodes)
    graph_workloads = []
    for idx in range(len(test_ds)):
        graph_workloads.append(WORKLOAD_ORDER[int(test_ds[idx].workload_emb.argmax().item())])
    node_workloads = np.repeat(np.array(graph_workloads, dtype=object), num_nodes)

    metrics = load_json(REPO_ROOT / "outputs" / "metrics.json")
    training_history = load_json(REPO_ROOT / "outputs" / "training_history.json")
    nsga_ppo_results = load_json(REPO_ROOT / "outputs" / "nsga_ppo_results.json")
    pareto_objectives = load_json(REPO_ROOT / "outputs" / "pareto_objectives.json")
    eval_df = pd.DataFrame(nsga_ppo_results["evaluation"])

    return {
        "cfg": cfg,
        "device": device,
        "predictor": predictor,
        "trajectory": trajectory,
        "test_ds": test_ds,
        "test_loader": loader,
        "num_nodes": num_nodes,
        "graph_workloads": graph_workloads,
        "node_workloads": node_workloads,
        "metrics": metrics,
        "training_history": training_history,
        "results": nsga_ppo_results,
        "pareto": pareto_objectives,
        "eval_df": eval_df,
    }


def collect_test_predictions(artifacts):
    predictor = artifacts["predictor"]
    trajectory = artifacts["trajectory"]
    device = artifacts["device"]

    predictor.eval()
    trajectory.eval()
    pred_nodes = []
    true_nodes = []
    pred_traj = []
    true_traj = []
    with torch.no_grad():
        for batch in artifacts["test_loader"]:
            batch = batch.to(device)
            node_pred = predictor(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view_as(batch.y)
            traj_pred = trajectory(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            pred_nodes.append(node_pred.detach().cpu().numpy())
            true_nodes.append(batch.y.detach().cpu().numpy())
            pred_traj.append(traj_pred.detach().cpu().numpy())
            true_traj.append(batch.y_trajectory.detach().cpu().numpy())

    predictor_pred = np.concatenate(pred_nodes, axis=0).reshape(-1)
    predictor_true = np.concatenate(true_nodes, axis=0).reshape(-1)
    trajectory_pred = np.concatenate(pred_traj, axis=0)
    trajectory_true = np.concatenate(true_traj, axis=0)

    return {
        "predictor_pred": predictor_pred,
        "predictor_true": predictor_true,
        "trajectory_pred": trajectory_pred,
        "trajectory_true": trajectory_true,
    }


def evaluate_ablation_models(artifacts):
    device = artifacts["device"]
    state_dict = artifacts["predictor"].state_dict()
    cfg = artifacts["cfg"]
    model_cfg = cfg.model
    configs = [
        ("GCN only", ("gcn",)),
        ("GCN+GAT", ("gcn", "gat")),
        ("GCN+Trans", ("gcn", "transformer")),
        ("Full model", ("gcn", "gat", "transformer")),
    ]
    rows = []
    for name, components in configs:
        model = HybridGNNTransformer(
            node_feature_dim=8,
            hidden_dim=int(model_cfg.hidden_dim),
            gat_heads=int(model_cfg.gat_heads),
            transformer_layers=int(model_cfg.transformer_layers),
            transformer_heads=int(model_cfg.transformer_heads),
            seq_len=1,
            components=components,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for batch in artifacts["test_loader"]:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view_as(batch.y)
                preds.append(pred.detach().cpu().numpy())
                labels.append(batch.y.detach().cpu().numpy())

        pred_arr = np.concatenate(preds, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        r2, mae, _ = flatten_metrics(pred_arr, label_arr)
        rows.append({"Config": name, "R2": r2, "MAE": mae})
    return pd.DataFrame(rows)


def per_step_r2(trajectory_pred: np.ndarray, trajectory_true: np.ndarray) -> list[float]:
    values = []
    for step in range(trajectory_true.shape[1]):
        pred = trajectory_pred[:, step]
        true = trajectory_true[:, step]
        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        values.append(float(1.0 - ss_res / max(ss_tot, 1e-12)))
    return values


def workload_sample_indices(graph_workloads: list[str]) -> dict[str, int]:
    picked = {}
    for idx, workload in enumerate(graph_workloads):
        if workload not in picked:
            picked[workload] = idx
        if len(picked) == len(WORKLOAD_ORDER):
            break
    return picked


def slice_graph_values(values: np.ndarray, graph_idx: int, num_nodes: int) -> np.ndarray:
    start = graph_idx * num_nodes
    end = start + num_nodes
    return values[start:end]


def build_positions(acc_graph: AcceleratorGraph) -> dict[int, tuple[float, float]]:
    positions = {}
    mac_side = int(math.ceil(math.sqrt(acc_graph.mac_clusters)))
    for node_id, info in acc_graph.node_info.items():
        node_type = info["type"]
        local_idx = int(info["local_idx"])
        if node_type == "mac":
            row = local_idx // mac_side
            col = local_idx % mac_side
            positions[node_id] = (float(col), float(-row))
        elif node_type == "sram":
            positions[node_id] = (float(local_idx) * 0.9 - 0.5, 1.8)
        else:
            positions[node_id] = (float(local_idx) * 1.4 - 0.5, -mac_side - 1.0)
    return positions


def method_metric_table(eval_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = eval_df.pivot(index="Workload", columns="Method", values=metric).reset_index()
    return table.set_index("Workload").loc[WORKLOAD_ORDER].reset_index()


def fig1_prediction_scatter(predictions, generated):
    print("[Fig 1] Prediction scatter")
    pred = predictions["predictor_pred"]
    true = predictions["predictor_true"]
    r2, mae, _ = flatten_metrics(pred, true)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    hb = ax.hexbin(true, pred, gridsize=60, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")
    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.2)
    ax.set_xlabel("Ground-truth aging score")
    ax.set_ylabel("Predicted aging score")
    ax.text(
        0.04,
        0.96,
        f"$R^2$ = {r2:.4f}\nMAE = {mae:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    savefig(fig, "fig1_prediction_scatter.pdf", generated)


def fig2_ablation_bars(ablation_df: pd.DataFrame, generated):
    print("[Fig 2] Ablation bars")
    x = np.arange(len(ablation_df))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax2 = ax1.twinx()
    bars_r2 = ax1.bar(x - width / 2, ablation_df["R2"], width, color=COLOR_NSGA, label="$R^2$")
    bars_mae = ax2.bar(x + width / 2, ablation_df["MAE"], width, color="indianred", label="MAE")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ablation_df["Config"], rotation=15, ha="right")
    ax1.set_ylabel("$R^2$")
    ax2.set_ylabel("MAE")
    ax1.set_ylim(0.0, 1.02)
    ax2.set_ylim(0.0, max(ablation_df["MAE"]) * 1.25)
    for bar, value in zip(bars_r2, ablation_df["R2"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, value in zip(bars_mae, ablation_df["MAE"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    savefig(fig, "fig2_ablation_bars.pdf", generated)


def fig3_trajectory_horizon(step_r2: list[float], overall_r2: float, generated):
    print("[Fig 3] Trajectory horizon")
    steps = np.arange(1, len(step_r2) + 1)
    step_r2_arr = np.array(step_r2)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(steps, step_r2_arr, marker="o", color="purple", linewidth=2.0)
    ax.axhline(overall_r2, linestyle="--", color="gray", linewidth=1.2)
    ax.fill_between(steps, step_r2_arr, overall_r2, where=step_r2_arr < overall_r2, color="lightgray", alpha=0.4)
    ax.set_xlabel("Prediction horizon $k$")
    ax.set_ylabel("$R^2$ score")
    ax.set_xticks(steps)
    ax.set_ylim(min(step_r2_arr.min(), overall_r2) - 0.05, 1.0)
    savefig(fig, "fig3_trajectory_horizon.pdf", generated)


def fig4_pareto_front(eval_df: pd.DataFrame, pareto: dict, generated):
    print("[Fig 4] Pareto front")
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

    initials = eval_df[eval_df["Method"] == "Initial"].set_index("Workload")
    for workload in WORKLOAD_ORDER:
        color = WORKLOAD_COLORS[workload]
        points = pareto.get(workload, [])
        if points:
            ax.scatter(
                [p["latency_cycles"] for p in points],
                [p["peak_aging"] for p in points],
                s=38,
                color=color,
                alpha=0.8,
                label=workload,
            )
        init_row = initials.loc[workload]
        ax.scatter(init_row["Latency Cycles"], init_row["Peak Aging"], marker="x", s=80, linewidths=2.0, color=color)
    ax.set_xlabel("Latency (cycles)")
    ax.set_ylabel("Peak aging score")
    ax.legend(loc="best", ncol=2, fontsize=9)
    savefig(fig, "fig4_pareto_front.pdf", generated)


def fig5_ppo_reward(rewards: list[float], generated):
    print("[Fig 5] PPO reward")
    reward_arr = np.array(rewards, dtype=np.float64)
    smoothed = moving_average(reward_arr, 5)
    x = np.arange(1, len(reward_arr) + 1)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(x, reward_arr, color=COLOR_PPO, alpha=0.35, linewidth=1.2, label="Raw reward")
    ax.plot(x, smoothed, color=COLOR_PPO, linewidth=2.2, label="Smoothed (w=5)")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.annotate(f"start = {reward_arr[0]:+.3f}", xy=(1, reward_arr[0]), xytext=(4, reward_arr[0] + 0.1),
                arrowprops={"arrowstyle": "->", "color": "0.3"}, fontsize=9)
    ax.annotate(f"end = {reward_arr[-1]:+.3f}", xy=(x[-1], reward_arr[-1]), xytext=(max(1, x[-1] - 16), reward_arr[-1] + 0.1),
                arrowprops={"arrowstyle": "->", "color": "0.3"}, fontsize=9)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode reward")
    ax.legend(loc="best")
    savefig(fig, "fig5_ppo_reward.pdf", generated)


def fig6_aging_heatmap(predictions, artifacts, generated):
    print("[Fig 6] Aging heatmap")
    sample_idx = workload_sample_indices(artifacts["graph_workloads"])
    num_nodes = artifacts["num_nodes"]

    rows = []
    for workload in WORKLOAD_ORDER:
        idx = sample_idx[workload]
        rows.append((
            workload,
            slice_graph_values(predictions["predictor_true"], idx, num_nodes),
            slice_graph_values(predictions["predictor_pred"], idx, num_nodes),
        ))

    values = np.concatenate([item[1] for item in rows] + [item[2] for item in rows])
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 1.4 * len(rows) + 0.8))
    for row_idx, (workload, truth, pred) in enumerate(rows):
        for col_idx, matrix in enumerate([truth, pred]):
            ax = axes[row_idx, col_idx]
            ax.imshow(matrix.reshape(4, 7), cmap="hot", norm=norm, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title("Ground truth" if col_idx == 0 else "Predicted")
            if col_idx == 0:
                ax.set_ylabel(workload, rotation=0, labelpad=48, va="center")
    savefig(fig, "fig6_aging_heatmap.pdf", generated)


def fig7_method_comparison(eval_df: pd.DataFrame, generated):
    print("[Fig 7] Method comparison")
    table = method_metric_table(eval_df, "Peak Aging")
    x = np.arange(len(table))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    initial_vals = table["Initial"].to_numpy()
    nsga_vals = table["NSGA-II"].to_numpy()
    ppo_vals = table["PPO"].to_numpy()

    bars_init = ax.bar(x - width, initial_vals, width, color=COLOR_INITIAL, label="Initial")
    bars_nsga = ax.bar(x, nsga_vals, width, color=COLOR_NSGA, label="NSGA-II")
    bars_ppo = ax.bar(x + width, ppo_vals, width, color=COLOR_PPO, label="PPO")

    for bar, init, val in zip(bars_nsga, initial_vals, nsga_vals):
        reduction = 100.0 * (init - val) / max(init, 1e-12)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"-{reduction:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, init, val in zip(bars_ppo, initial_vals, ppo_vals):
        reduction = 100.0 * (init - val) / max(init, 1e-12)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"-{reduction:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Peak aging score")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Workload"], rotation=15, ha="right")
    ax.legend(loc="best")
    savefig(fig, "fig7_method_comparison.pdf", generated)


def fig8_physics_curves(generated):
    print("[Fig 8] Physics curves")
    hours = np.linspace(0.0, 500.0, 500)
    acts = [0.3, 0.6, 0.9]
    A, n_exp = 0.005, 0.25
    B, m_exp = 0.0001, 0.5
    eta, beta = 200.0, 2.5

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    for act in acts:
        axes[0].plot(hours, A * np.power(np.clip(act * hours, 1e-9, None), n_exp), label=f"act={act:.1f}")
        axes[1].plot(hours, B * np.power(act, m_exp) * np.sqrt(hours), label=f"act={act:.1f}")
        axes[2].plot(hours, 1.0 - np.exp(-np.power(hours / eta, beta)), label=f"act={act:.1f}")

    axes[0].set_title("NBTI")
    axes[1].set_title("HCI")
    axes[2].set_title("TDDB")
    for ax in axes:
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Degradation")
        ax.legend(loc="best", fontsize=8)
    savefig(fig, "fig8_physics_curves.pdf", generated)


def fig9_lifetime_per_workload(eval_df: pd.DataFrame, generated):
    print("[Fig 9] Lifetime per workload")
    table = method_metric_table(eval_df, "TTF (Yrs)")
    x = np.arange(len(table))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    initial_vals = table["Initial"].to_numpy()
    nsga_vals = table["NSGA-II"].to_numpy()
    ppo_vals = table["PPO"].to_numpy()

    bars_init = ax.bar(x - width, initial_vals, width, color=COLOR_INITIAL, label="Initial")
    bars_nsga = ax.bar(x, nsga_vals, width, color=COLOR_NSGA, label="NSGA-II")
    bars_ppo = ax.bar(x + width, ppo_vals, width, color=COLOR_PPO, label="PPO")

    for bar, init, val in zip(bars_nsga, initial_vals, nsga_vals):
        improvement = 100.0 * (val - init) / max(init, 1e-12)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"+{improvement:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar, init, val in zip(bars_ppo, initial_vals, ppo_vals):
        improvement = 100.0 * (val - init) / max(init, 1e-12)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"+{improvement:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("TTF (years)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Workload"], rotation=15, ha="right")
    ax.legend(loc="best")
    savefig(fig, "fig9_lifetime_per_workload.pdf", generated)


def fig10_latency_reduction(eval_df: pd.DataFrame, generated):
    print("[Fig 10] Latency reduction")
    table = method_metric_table(eval_df, "Latency Cycles")
    x = np.arange(len(table))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    init_vals = table["Initial"].to_numpy()
    nsga_vals = table["NSGA-II"].to_numpy()
    bars_init = ax.bar(x - width / 2, init_vals, width, color=COLOR_INITIAL, label="Initial")
    bars_nsga = ax.bar(x + width / 2, nsga_vals, width, color=COLOR_NSGA, label="NSGA-II")

    for bar, init, val in zip(bars_nsga, init_vals, nsga_vals):
        reduction = 100.0 * (init - val) / max(init, 1e-12)
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(nsga_vals) * 0.01, f"-{reduction:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Latency (cycles)")
    ax.set_xticks(x)
    ax.set_xticklabels(table["Workload"], rotation=15, ha="right")
    ax.legend(loc="best")
    savefig(fig, "fig10_latency_reduction.pdf", generated)


def fig11_lifetime_summary(eval_df: pd.DataFrame, generated):
    print("[Fig 11] Lifetime summary")
    grouped = eval_df.groupby("Method")["TTF (Yrs)"].mean()
    labels = ["Static", "NSGA-II Best", "PPO Policy"]
    values = np.array([grouped["Initial"], grouped["NSGA-II"], grouped["PPO"]], dtype=np.float64)
    base = values[0]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    bars = ax.bar(labels, values, color=[COLOR_INITIAL, COLOR_NSGA, COLOR_PPO])
    for bar, val in zip(bars[1:], values[1:]):
        improvement = 100.0 * (val - base) / max(base, 1e-12)
        ax.text(bar.get_x() + bar.get_width() / 2, val + values.max() * 0.02, f"+{improvement:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("Mean TTF (years)")
    savefig(fig, "fig11_lifetime_summary.pdf", generated)


def fig12_pareto_3d(eval_df: pd.DataFrame, pareto: dict, generated):
    print("[Fig 12] Pareto 3D")
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    initials = eval_df[eval_df["Method"] == "Initial"].set_index("Workload")
    for workload in WORKLOAD_ORDER:
        color = WORKLOAD_COLORS[workload]
        points = pareto.get(workload, [])
        if points:
            ax.scatter(
                [p["latency_cycles"] for p in points],
                [p["energy_pj"] for p in points],
                [p["peak_aging"] for p in points],
                color=color,
                s=26,
                alpha=0.85,
            )
        init = initials.loc[workload]
        ax.scatter(
            [init["Latency Cycles"]],
            [init["Energy (pJ)"]],
            [init["Peak Aging"]],
            color=color,
            marker="x",
            s=80,
        )
    ax.set_xlabel("Latency")
    ax.set_ylabel("Energy")
    ax.set_zlabel("Peak aging")
    savefig(fig, "fig12_pareto_3d.pdf", generated)


def fig13_training_convergence(training_history: dict, generated):
    print("[Fig 13] Training convergence")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    pred_hist = training_history["predictor"]
    traj_hist = training_history["trajectory"]

    axes[0].plot(pred_hist["epoch"], pred_hist["train_loss"], color=COLOR_NSGA, label="Predictor train")
    axes[0].plot(pred_hist["epoch"], pred_hist["val_loss"], color=COLOR_NSGA, linestyle="--", label="Predictor val")
    axes[0].plot(traj_hist["epoch"], traj_hist["train_loss"], color=COLOR_PPO, label="Trajectory train")
    axes[0].plot(traj_hist["epoch"], traj_hist["val_loss"], color=COLOR_PPO, linestyle="--", label="Trajectory val")
    axes[0].axvline(pred_hist["best_epoch"], color=COLOR_NSGA, linestyle=":", linewidth=1.2)
    axes[0].axvline(traj_hist["best_epoch"], color=COLOR_PPO, linestyle=":", linewidth=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(pred_hist["epoch"], pred_hist["val_r2"], color=COLOR_NSGA, label="Predictor val $R^2$")
    axes[1].plot(traj_hist["epoch"], traj_hist["val_r2"], color=COLOR_PPO, label="Trajectory val $R^2$")
    axes[1].axvline(pred_hist["best_epoch"], color=COLOR_NSGA, linestyle=":", linewidth=1.2)
    axes[1].axvline(traj_hist["best_epoch"], color=COLOR_PPO, linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation $R^2$")
    axes[1].legend(loc="best", fontsize=8)
    savefig(fig, "fig13_training_convergence.pdf", generated)


def fig14_trajectory_pred_vs_actual(predictions, generated):
    print("[Fig 14] Trajectory pred vs actual")
    traj_pred = predictions["trajectory_pred"]
    traj_true = predictions["trajectory_true"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, step in zip(axes, [0, 9]):
        pred = traj_pred[:, step]
        true = traj_true[:, step]
        lo = min(pred.min(), true.min())
        hi = max(pred.max(), true.max())
        ax.scatter(true, pred, s=5, alpha=0.25, color=COLOR_NSGA)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Step {step + 1}")
    savefig(fig, "fig14_trajectory_pred_vs_actual.pdf", generated)


def fig15_aging_graph(predictions, artifacts, generated):
    print("[Fig 15] Aging graph")
    num_nodes = artifacts["num_nodes"]
    graph_idx = workload_sample_indices(artifacts["graph_workloads"])["ResNet-50"]
    true_vals = slice_graph_values(predictions["predictor_true"], graph_idx, num_nodes)
    pred_vals = slice_graph_values(predictions["predictor_pred"], graph_idx, num_nodes)

    acc_graph = AcceleratorGraph(artifacts["cfg"].accelerator)
    acc_graph.build()
    pos = build_positions(acc_graph)
    values = np.concatenate([true_vals, pred_vals])
    norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8))
    for ax, vals, title in zip(axes, [true_vals, pred_vals], ["Observed", "Predicted"]):
        nx.draw_networkx_edges(acc_graph.graph, pos, ax=ax, edge_color="lightgray", width=0.8, arrows=False)
        nx.draw_networkx_nodes(
            acc_graph.graph,
            pos,
            ax=ax,
            node_color=vals,
            cmap="hot",
            vmin=float(values.min()),
            vmax=float(values.max()),
            node_size=280,
        )
        ax.set_title(title)
        ax.axis("off")
    savefig(fig, "fig15_aging_graph.pdf", generated)


def main():
    print("[Setup] Loading artifacts")
    artifacts = load_artifacts()
    predictions = collect_test_predictions(artifacts)
    ablation_df = evaluate_ablation_models(artifacts)

    predictor_r2, predictor_mae, _ = flatten_metrics(
        predictions["predictor_pred"], predictions["predictor_true"]
    )
    step_r2 = per_step_r2(predictions["trajectory_pred"], predictions["trajectory_true"])
    overall_traj_r2 = float(artifacts["metrics"]["trajectory"]["r2"])

    print(f"[Setup] Predictor R2={predictor_r2:.4f}  MAE={predictor_mae:.4f}")
    print(f"[Setup] Trajectory overall R2={overall_traj_r2:.4f}")

    generated: list[Path] = []
    fig1_prediction_scatter(predictions, generated)
    fig2_ablation_bars(ablation_df, generated)
    fig3_trajectory_horizon(step_r2, overall_traj_r2, generated)
    fig4_pareto_front(artifacts["eval_df"], artifacts["pareto"], generated)
    fig5_ppo_reward(artifacts["metrics"]["ppo"]["rewards"], generated)
    fig6_aging_heatmap(predictions, artifacts, generated)
    fig7_method_comparison(artifacts["eval_df"], generated)
    fig8_physics_curves(generated)
    fig9_lifetime_per_workload(artifacts["eval_df"], generated)
    fig10_latency_reduction(artifacts["eval_df"], generated)
    fig11_lifetime_summary(artifacts["eval_df"], generated)
    fig12_pareto_3d(artifacts["eval_df"], artifacts["pareto"], generated)
    fig13_training_convergence(artifacts["training_history"], generated)
    fig14_trajectory_pred_vs_actual(predictions, generated)
    fig15_aging_graph(predictions, artifacts, generated)

    print(f"\nGenerated {len(generated)} plot files")
    for path in generated:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
