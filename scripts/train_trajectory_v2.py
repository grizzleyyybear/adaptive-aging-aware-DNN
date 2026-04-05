"""
Train the redesigned TrajectoryPredictor (v2) on the packed aging dataset.

Changes from v1:
  - Architecture: encoder_features + current_pred → 512-wide GELU head
  - Loss: discounted MSE + variance-matching term (weight 0.1)
  - Separate LRs: encoder 5e-6 (slow fine-tune), head 1e-3 (aggressive)
  - CosineAnnealingLR, 200 epochs, early-stop patience 30
  - Per-step R² reported in final evaluation

Usage:
    python scripts/train_trajectory_v2.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor
from utils.runtime_eval import cfg_get

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_PT = REPO_ROOT / "data/processed/aging_train_40000_mac16_feat8.pt"
VAL_PT   = REPO_ROOT / "data/processed/aging_val_5000_mac16_feat8.pt"
PREDICTOR_CKPT  = REPO_ROOT / "outputs/best_predictor.pt"
TRAJECTORY_CKPT = REPO_ROOT / "outputs/best_trajectory.pt"

HIDDEN_DIM   = 256
HORIZON      = 10
GAMMA        = 0.95
EPOCHS       = 200
PATIENCE     = 30
BATCH_SIZE   = 64
ENCODER_LR   = 5e-6
HEAD_LR      = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP    = 1.0
VAR_WEIGHT   = 0.1

NODES_PER_GRAPH = 28
EDGES_PER_GRAPH = 92


# ---------------------------------------------------------------------------
# Lazy dataset — reconstructs individual Data objects from the packed file
# ---------------------------------------------------------------------------
class PackedAgingDataset(torch.utils.data.Dataset):
    """
    Wraps the pre-built packed tensor file.  All graphs share the same
    accelerator topology so edge_index is stored once (local indices 0..27).
    """

    def __init__(self, path: Path):
        raw = torch.load(path, map_location="cpu")
        d = raw[0]  # first element holds the data dict
        self.x        = d["x"]            # [N_total, 8]
        self.y        = d["y"]            # [N_total, 1]
        self.y_traj   = d["y_trajectory"] # [N_total, 10]
        self.edge_attr_all = d["edge_attr"]  # [E_total, 2]
        # Edge topology is the same for every graph — cache it once
        self.edge_index = d["edge_index"][:, :EDGES_PER_GRAPH].clone()  # [2, 92]
        self.n_graphs = self.x.shape[0] // NODES_PER_GRAPH

    def __len__(self) -> int:
        return self.n_graphs

    def __getitem__(self, i: int) -> Data:
        sn = i * NODES_PER_GRAPH
        se = i * EDGES_PER_GRAPH
        return Data(
            x          = self.x[sn : sn + NODES_PER_GRAPH],
            edge_index = self.edge_index,
            edge_attr  = self.edge_attr_all[se : se + EDGES_PER_GRAPH],
            y          = self.y[sn : sn + NODES_PER_GRAPH],
            y_trajectory = self.y_traj[sn : sn + NODES_PER_GRAPH],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def evaluate(model: TrajectoryPredictor, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_pred  = []
    all_label = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            label = batch.y_trajectory
            loss = model.trajectory_loss(pred, label, var_weight=VAR_WEIGHT)
            total_loss += loss.item()
            n_batches  += 1
            all_pred.append(pred.cpu().numpy())
            all_label.append(label.cpu().numpy())

    all_pred  = np.concatenate(all_pred,  axis=0)  # [N_total, 10]
    all_label = np.concatenate(all_label, axis=0)

    overall_r2  = r2_score(all_pred.ravel(), all_label.ravel())
    overall_mae = float(np.abs(all_pred - all_label).mean())
    per_step_r2 = [r2_score(all_pred[:, s], all_label[:, s]) for s in range(HORIZON)]
    pred_std    = float(all_pred.std())
    label_std   = float(all_label.std())

    return {
        "loss":         total_loss / max(n_batches, 1),
        "r2":           overall_r2,
        "mae":          overall_mae,
        "per_step_r2":  per_step_r2,
        "pred_std":     pred_std,
        "label_std":    label_std,
        "std_ratio":    pred_std / max(label_std, 1e-9),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # ---- Load pre-trained predictor (frozen topology reference) ----------
    if not PREDICTOR_CKPT.exists():
        raise FileNotFoundError(f"Predictor checkpoint not found: {PREDICTOR_CKPT}")

    predictor = HybridGNNTransformer(
        node_feature_dim=8,
        hidden_dim=HIDDEN_DIM,
        gat_heads=4,
        transformer_layers=2,
        transformer_heads=4,
        seq_len=1,
    ).to(device)
    predictor.load_state_dict(torch.load(PREDICTOR_CKPT, map_location=device))
    predictor.eval()
    print(f"[Setup] Loaded predictor from {PREDICTOR_CKPT}")

    # Verify encoder.head exists and has the expected output shape
    _dummy = torch.zeros(1, HIDDEN_DIM, device=device)
    _out   = predictor.head(_dummy)
    assert _out.shape == (1, 1), f"encoder.head output shape unexpected: {_out.shape}"
    print(f"[Setup] encoder.head verified: output shape {_out.shape} ✓")

    # ---- Build trajectory model ------------------------------------------
    encoder_copy = copy.deepcopy(predictor).to(device)
    model = TrajectoryPredictor(
        gnn_encoder=encoder_copy,
        hidden_dim=HIDDEN_DIM,
        horizon=HORIZON,
        gamma=GAMMA,
    ).to(device)
    print(
        f"[Setup] TrajectoryPredictor parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
        f"  (head only: {sum(p.numel() for p in model.traj_head.parameters()):,})"
    )

    # ---- Data loaders ----------------------------------------------------
    print(f"[Data] Loading train: {TRAIN_PT}")
    train_ds = PackedAgingDataset(TRAIN_PT)
    print(f"[Data] Loading val  : {VAL_PT}")
    val_ds   = PackedAgingDataset(VAL_PT)
    print(f"[Data] Train graphs: {len(train_ds):,}  |  Val graphs: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    # ---- Optimiser: separate LRs for encoder and head --------------------
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENCODER_LR},
        {"params": model.traj_head.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ---- Training loop ---------------------------------------------------
    best_val_loss = float("inf")
    best_val_r2   = -float("inf")
    no_improve    = 0
    best_state    = None

    print(f"\n[Train] Starting: {EPOCHS} epochs, patience={PATIENCE}, batch={BATCH_SIZE}")
    print(f"        Encoder LR={ENCODER_LR:.0e}  Head LR={HEAD_LR:.0e}  var_weight={VAR_WEIGHT}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            label = batch.y_trajectory
            loss  = model.trajectory_loss(pred, label, var_weight=VAR_WEIGHT)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # Validate every 5 epochs (or on epoch 1 to show initial state)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics["loss"]
            val_r2   = val_metrics["r2"]
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f} | "
                f"val_R²={val_r2:.4f} | "
                f"std_ratio={val_metrics['std_ratio']:.3f}"
            )

            if val_r2 > best_val_r2:
                best_val_r2   = val_r2
                best_val_loss = val_loss
                no_improve    = 0
                best_state    = copy.deepcopy(model.state_dict())
            else:
                no_improve += 5 if epoch % 5 == 0 else 1

            if no_improve >= PATIENCE:
                print(f"[Train] Early stop at epoch {epoch} (no improvement for {PATIENCE} val checks)")
                break

    # ---- Restore best and save ------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n[Save] Restored best model (val_R²={best_val_r2:.4f})")

    TRAJECTORY_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TRAJECTORY_CKPT)
    print(f"[Save] Saved to {TRAJECTORY_CKPT}")

    # ---- Final evaluation ------------------------------------------------
    print("\n[Eval] Final evaluation on validation set")
    print("=" * 70)
    final = evaluate(model, val_loader, device)

    print(f"  Overall R²  : {final['r2']:.4f}")
    print(f"  Overall MAE : {final['mae']:.4f}")
    print(f"  Pred std    : {final['pred_std']:.4f}  |  Label std: {final['label_std']:.4f}  |  Ratio: {final['std_ratio']:.3f}")
    print()
    print("  Per-step R²:")
    for step, r2 in enumerate(final["per_step_r2"], start=1):
        bar = "█" * max(0, int(r2 * 30)) if r2 > 0 else ""
        print(f"    Step {step:2d}: {r2:+.4f}  {bar}")

    print("=" * 70)
    print("[Done]")


if __name__ == "__main__":
    main()
