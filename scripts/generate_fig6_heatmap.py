"""Generate fig6_aging_heatmap.pdf — Per-Workload Aging Heatmap (GT vs Predicted).

Loads the test dataset and predictor checkpoint, groups the 5000 test samples
by workload (5 workloads × ~1000 samples each), averages the per-node aging
scores, and produces a side-by-side heatmap:
  - Left panel:  Ground-Truth  aging (5 workloads × 28 nodes)
  - Right panel: Predicted     aging (5 workloads × 28 nodes)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from models.hybrid_gnn_transformer import HybridGNNTransformer

# ── constants ──────────────────────────────────────────────────────────────────
WORKLOAD_NAMES = ["ResNet-50", "BERT-Base", "MobileNetV2", "EfficientNet-B4", "ViT-B/16"]
NODES_PER_GRAPH = 28
NUM_WORKLOADS = 5
NODE_FEATURE_DIM = 8
HIDDEN_DIM = 256

PREDICTOR_CKPT = "checkpoints/predictor_best.pt"
TEST_DATA_PATH = "data/processed/aging_test_5000_mac16_feat8.pt"
OUT_PATH = "outputs/plots/fig6_aging_heatmap.pdf"


def load_predictor(device: torch.device) -> HybridGNNTransformer:
    model = HybridGNNTransformer(
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
    )
    sd = torch.load(PREDICTOR_CKPT, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model.to(device)


def load_test_data():
    raw = torch.load(TEST_DATA_PATH, map_location="cpu")
    # raw is a 3-tuple; packed tensors are in raw[0]
    packed = raw[0] if isinstance(raw, (tuple, list)) else raw
    n_samples = 5000
    x = packed["x"].view(n_samples, NODES_PER_GRAPH, NODE_FEATURE_DIM)          # [5000, 28, 8]
    y = packed["y"].view(n_samples, NODES_PER_GRAPH)                              # [5000, 28]
    w_emb = packed["workload_emb"].view(n_samples, NUM_WORKLOADS)                  # [5000, 5]
    workload_ids = w_emb.argmax(dim=1)                                             # [5000]

    # Build a single shared edge_index for one graph (indices 0–27).
    # The packed dataset stores all edges concatenated; extract first graph's edges.
    full_ei = packed["edge_index"]  # [2, n_samples*92]
    edges_per_graph = full_ei.shape[1] // n_samples
    ei = full_ei[:, :edges_per_graph]  # [2, 92]
    return x, y, workload_ids, ei


@torch.no_grad()
def predict_all(model, x: torch.Tensor, ei: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run predictor on all 5000 samples; returns [5000, 28] predictions."""
    n_samples = x.shape[0]
    preds = []
    batch_size = 128
    for start in range(0, n_samples, batch_size):
        xb = x[start : start + batch_size].to(device)          # [B, 28, 8]
        B = xb.shape[0]
        # Flatten nodes across batch and build batched edge_index
        xb_flat = xb.view(B * NODES_PER_GRAPH, NODE_FEATURE_DIM)
        # Repeat ei for B graphs with offset
        offsets = torch.arange(B, device=device).repeat_interleave(ei.shape[1]) * NODES_PER_GRAPH
        ei_batch = ei.to(device).repeat(1, B) + offsets.unsqueeze(0)
        # Batch vector
        batch_vec = torch.arange(B, device=device).repeat_interleave(NODES_PER_GRAPH)

        out = model(xb_flat, ei_batch, batch=batch_vec)  # [B*28, 1]
        preds.append(out.view(B, NODES_PER_GRAPH).cpu())
    return torch.cat(preds, dim=0)  # [5000, 28]


def build_heatmaps(
    y: torch.Tensor,
    pred: torch.Tensor,
    workload_ids: torch.Tensor,
):
    """Average GT and predicted aging per workload × node."""
    gt_heatmap = np.zeros((NUM_WORKLOADS, NODES_PER_GRAPH), dtype=np.float32)
    pr_heatmap = np.zeros((NUM_WORKLOADS, NODES_PER_GRAPH), dtype=np.float32)
    for wl_idx in range(NUM_WORKLOADS):
        mask = workload_ids == wl_idx
        gt_heatmap[wl_idx] = y[mask].mean(dim=0).detach().float().cpu().numpy()
        pr_heatmap[wl_idx] = pred[mask].mean(dim=0).detach().float().cpu().numpy()
    return gt_heatmap, pr_heatmap


def plot_heatmap(gt: np.ndarray, pr: np.ndarray, out_path: str) -> None:
    vmin = min(gt.min(), pr.min())
    vmax = max(gt.max(), pr.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    node_labels = [str(i) for i in range(NODES_PER_GRAPH)]
    x_ticks = np.arange(NODES_PER_GRAPH)

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.08)

    cmap = "YlOrRd"

    for col, (data, title) in enumerate([(gt, "Ground Truth"), (pr, "Predicted")]):
        ax = fig.add_subplot(gs[col])
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(x_ticks[::4])
        ax.set_xticklabels(node_labels[::4], fontsize=8)
        ax.set_yticks(np.arange(NUM_WORKLOADS))
        ax.set_yticklabels(WORKLOAD_NAMES, fontsize=9)
        ax.set_xlabel("Accelerator Node", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        if col == 0:
            ax.set_ylabel("Workload", fontsize=10)
        else:
            ax.set_yticklabels([])

    # Colorbar
    cax = fig.add_subplot(gs[2])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Aging Score", fontsize=10)

    # Error annotations: per-cell absolute diff
    diff = np.abs(gt - pr)
    for wl_idx in range(NUM_WORKLOADS):
        for node_idx in range(NODES_PER_GRAPH):
            if diff[wl_idx, node_idx] > 0.01:
                ax_pred = fig.axes[1]
                ax_pred.add_patch(
                    plt.Rectangle(
                        (node_idx - 0.5, wl_idx - 0.5), 1, 1,
                        fill=False, edgecolor="navy", linewidth=0.6, alpha=0.6
                    )
                )

    fig.suptitle(
        "Per-Workload Node Aging: Ground Truth vs Predicted",
        fontsize=13, fontweight="bold", y=1.01,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    device = torch.device("cpu")
    print("Loading test data …")
    x, y, workload_ids, ei = load_test_data()

    print("Loading predictor …")
    model = load_predictor(device)

    print("Running inference on 5000 test samples …")
    pred = predict_all(model, x, ei, device)

    print("Building per-workload heatmaps …")
    gt_heatmap, pr_heatmap = build_heatmaps(y, pred, workload_ids)

    # Sanity print
    for i, wl in enumerate(WORKLOAD_NAMES):
        print(
            f"  {wl:20s}  GT mean={gt_heatmap[i].mean():.4f}  "
            f"Pred mean={pr_heatmap[i].mean():.4f}  "
            f"MAE={np.abs(gt_heatmap[i] - pr_heatmap[i]).mean():.4f}"
        )

    plot_heatmap(gt_heatmap, pr_heatmap, OUT_PATH)


if __name__ == "__main__":
    main()
