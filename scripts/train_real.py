"""
Train predictor + trajectory models on simulator-generated datasets.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.graph_dataset import AgingDataset
from models.hybrid_gnn_transformer import HybridGNNTransformer
from models.trajectory_predictor import TrajectoryPredictor
from utils.device import (
    configure_torch_runtime,
    dataloader_kwargs,
    describe_device,
    get_device_request,
    resolve_device,
    use_non_blocking,
)


DEVICE = torch.device("cpu")
NON_BLOCKING = False
LOADER_KWARGS = {"num_workers": 0, "pin_memory": False}
PREDICTOR_EPOCHS = 100
TRAJECTORY_EPOCHS = 150
BATCH_SIZE = 64
LR = 1e-4
TRAJ_ENCODER_LR = 5e-6
TRAJ_HEAD_LR = 5e-4
WD = 1e-5
PATIENCE = 15
TRAJECTORY_PATIENCE = 25
GRAD_CLIP = 1.0
OUT_DIR = Path("outputs")


def load_cfg():
    accel = OmegaConf.load("configs/accelerator.yaml")
    workloads = OmegaConf.load("configs/workloads.yaml")
    training = OmegaConf.load("configs/training.yaml")
    experiments = OmegaConf.load("configs/experiments.yaml")
    cfg = OmegaConf.merge(experiments, accel, workloads, training)
    cfg.model.prediction_horizon = 10
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train the real aging predictor and trajectory models")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: cuda, cpu, auto, or cuda:0. Overrides config/runtime.device.",
    )
    return parser.parse_args()


def configure_runtime(device_request: str) -> str:
    global DEVICE, NON_BLOCKING, LOADER_KWARGS

    DEVICE = resolve_device(device_request)
    if DEVICE.type != "cuda":
        raise RuntimeError("scripts/train_real.py requires CUDA for this run.")

    configure_torch_runtime(DEVICE)
    NON_BLOCKING = use_non_blocking(DEVICE)
    LOADER_KWARGS = dataloader_kwargs(DEVICE)
    return describe_device(DEVICE)


def load_datasets(cfg):
    print("[Data] Loading processed datasets")
    train_ds = AgingDataset(root="./data", split="train", size=40000, cfg=cfg, seed=42)
    val_ds = AgingDataset(root="./data", split="val", size=5000, cfg=cfg, seed=43)
    test_ds = AgingDataset(root="./data", split="test", size=5000, cfg=cfg, seed=44)
    print(f"[Data] Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    return train_ds, val_ds, test_ds


def make_loaders(train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
    return train_loader, val_loader, test_loader


def regression_metrics(preds, labels):
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {
        "r2": float(r2_score(labels, preds)),
        "mae": float(mean_absolute_error(labels, preds)),
        "rmse": float(np.sqrt(mean_squared_error(labels, preds))),
    }


def trajectory_step_r2(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return [float(r2_score(labels[:, step], preds[:, step])) for step in range(labels.shape[1])]


def _epoch_summary(model, loader, target_attr: str, optimizer=None, collect_metrics: bool = False):
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    preds = [] if collect_metrics else None
    labels = [] if collect_metrics else None

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=NON_BLOCKING)
        target = getattr(batch, target_attr)

        with torch.set_grad_enabled(is_train):
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            if target_attr == "y":
                pred = pred.view_as(target)
                loss = torch.nn.functional.mse_loss(pred, target)
            else:
                loss = model.trajectory_loss(pred, target)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += float(loss.item()) * batch.num_graphs

        if collect_metrics:
            preds.append(pred.detach().cpu().numpy())
            labels.append(target.detach().cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    if not collect_metrics:
        return avg_loss, None

    metric_dict = regression_metrics(np.concatenate(preds), np.concatenate(labels))
    return avg_loss, metric_dict


def collect_outputs(model, loader, target_attr: str):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=NON_BLOCKING)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = getattr(batch, target_attr)
            if target_attr == "y":
                pred = pred.view_as(target)
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(target.detach().cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train_predictor(train_loader, val_loader, test_loader, node_feature_dim):
    print("\n" + "=" * 72)
    print("[Train] Phase 1/2: Aging predictor")
    print("=" * 72)
    print(
        f"[Train] node_feature_dim={node_feature_dim}  epochs={PREDICTOR_EPOCHS}  "
        f"patience={PATIENCE}  batch={BATCH_SIZE}  lr={LR:.1e}"
    )

    predictor = HybridGNNTransformer(node_feature_dim=node_feature_dim, hidden_dim=256, seq_len=1).to(DEVICE)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PREDICTOR_EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = OUT_DIR / "best_predictor.pt"
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_r2": [],
        "val_mae": [],
        "lr": [],
    }

    for epoch in range(1, PREDICTOR_EPOCHS + 1):
        train_loss, _ = _epoch_summary(predictor, train_loader, target_attr="y", optimizer=optimizer)
        val_loss, val_metrics = _epoch_summary(predictor, val_loader, target_attr="y", collect_metrics=True)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_r2"].append(float(val_metrics["r2"]))
        history["val_mae"].append(float(val_metrics["mae"]))
        history["lr"].append(float(scheduler.get_last_lr()[0]))

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[Predictor] Epoch {epoch:3d}/{PREDICTOR_EPOCHS} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"val_r2={val_metrics['r2']:.4f} | val_mae={val_metrics['mae']:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(predictor.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[Predictor] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    predictor.load_state_dict(torch.load(best_path, map_location=DEVICE))
    preds, labels = collect_outputs(predictor, test_loader, target_attr="y")
    metrics = regression_metrics(preds, labels)
    metrics["best_epoch"] = int(best_epoch)

    print("[Predictor] Final test metrics")
    print(f"  R2   : {metrics['r2']:.4f}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  Best : epoch {best_epoch}")

    history["best_epoch"] = int(best_epoch)
    return predictor, metrics, history


def train_trajectory(predictor, train_loader, val_loader, test_loader, horizon):
    print("\n" + "=" * 72)
    print("[Train] Phase 2/2: Trajectory predictor")
    print("=" * 72)
    print(
        f"[Train] horizon={horizon}  epochs={TRAJECTORY_EPOCHS}  "
        f"patience={TRAJECTORY_PATIENCE}  encoder_lr={TRAJ_ENCODER_LR:.1e}  "
        f"head_lr={TRAJ_HEAD_LR:.1e}"
    )

    traj_model = TrajectoryPredictor(gnn_encoder=copy.deepcopy(predictor), horizon=horizon).to(DEVICE)
    optimizer = torch.optim.AdamW(
        [
            {"params": traj_model.encoder.parameters(), "lr": TRAJ_ENCODER_LR},
            {"params": traj_model.traj_head.parameters(), "lr": TRAJ_HEAD_LR},
        ],
        weight_decay=WD,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAJECTORY_EPOCHS, eta_min=1e-7)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = OUT_DIR / "best_trajectory.pt"
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_r2": [],
        "val_mae": [],
        "encoder_lr": [],
        "head_lr": [],
    }

    for epoch in range(1, TRAJECTORY_EPOCHS + 1):
        train_loss, _ = _epoch_summary(traj_model, train_loader, target_attr="y_trajectory", optimizer=optimizer)
        val_loss, val_metrics = _epoch_summary(traj_model, val_loader, target_attr="y_trajectory", collect_metrics=True)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_r2"].append(float(val_metrics["r2"]))
        history["val_mae"].append(float(val_metrics["mae"]))
        history["encoder_lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["head_lr"].append(float(optimizer.param_groups[1]["lr"]))

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[Trajectory] Epoch {epoch:3d}/{TRAJECTORY_EPOCHS} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"val_r2={val_metrics['r2']:.4f} | val_mae={val_metrics['mae']:.4f} | "
                f"enc_lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"head_lr={optimizer.param_groups[1]['lr']:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(traj_model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAJECTORY_PATIENCE:
                print(f"[Trajectory] Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    traj_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    preds, labels = collect_outputs(traj_model, test_loader, target_attr="y_trajectory")
    metrics = regression_metrics(preds, labels)
    metrics["per_step_r2"] = trajectory_step_r2(preds, labels)
    metrics["best_epoch"] = int(best_epoch)

    print("[Trajectory] Final test metrics")
    print(f"  R2   : {metrics['r2']:.4f}")
    print(f"  MAE  : {metrics['mae']:.4f}")
    print(f"  RMSE : {metrics['rmse']:.4f}")
    print(f"  Best : epoch {best_epoch}")
    for step_idx, step_r2 in enumerate(metrics["per_step_r2"], start=1):
        print(f"  Step {step_idx:2d}: R2={step_r2:.4f}")

    history["best_epoch"] = int(best_epoch)
    return traj_model, metrics, history


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    args = parse_args()
    device_request = args.device or get_device_request(cfg)
    device_desc = configure_runtime(device_request)
    print(f"[Setup] Device request: {device_request}")
    print(f"[Setup] Using device  : {device_desc}")
    print(f"[Setup] DataLoader    : {LOADER_KWARGS}")

    train_ds, val_ds, test_ds = load_datasets(cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    node_feature_dim = int(train_ds[0].x.shape[1])
    horizon = int(train_ds[0].y_trajectory.shape[1])
    if node_feature_dim != 8:
        raise ValueError(f"Expected node_feature_dim=8, got {node_feature_dim}")
    if horizon != 10:
        raise ValueError(f"Expected trajectory horizon=10, got {horizon}")
    print(f"[Setup] node_feature_dim={node_feature_dim}")
    print(f"[Setup] trajectory_horizon={horizon}")

    predictor, pred_metrics, predictor_history = train_predictor(
        train_loader, val_loader, test_loader, node_feature_dim
    )
    torch.cuda.empty_cache()
    _traj_model, traj_metrics, trajectory_history = train_trajectory(
        predictor, train_loader, val_loader, test_loader, horizon
    )

    metrics = {
        "predictor": pred_metrics,
        "trajectory": traj_metrics,
    }
    history = {
        "predictor": predictor_history,
        "trajectory": trajectory_history,
    }

    with open(OUT_DIR / "real_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(OUT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n[Done] Metrics saved to outputs/real_metrics.json")
    print("[Done] Training history saved to outputs/training_history.json")
    print("[Done] Checkpoints saved to outputs/best_predictor.pt and outputs/best_trajectory.pt")


if __name__ == "__main__":
    main()
