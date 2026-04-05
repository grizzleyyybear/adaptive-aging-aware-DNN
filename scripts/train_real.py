"""
Train predictor + trajectory models on simulator-generated datasets.
"""

from __future__ import annotations

import argparse
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
TRAJ_ENCODER_LR = 1e-5
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
        help="Device to use: auto, cpu, cuda, or cuda:0. Overrides config/runtime.device.",
    )
    return parser.parse_args()


def configure_runtime(device_request: str) -> str:
    global DEVICE, NON_BLOCKING, LOADER_KWARGS

    DEVICE = resolve_device(device_request)
    configure_torch_runtime(DEVICE)
    NON_BLOCKING = use_non_blocking(DEVICE)
    LOADER_KWARGS = dataloader_kwargs(DEVICE)
    return describe_device(DEVICE)


def load_datasets(cfg):
    print("Loading datasets...")
    train_ds = AgingDataset(root="./data", split="train", size=40000, cfg=cfg, seed=42)
    val_ds = AgingDataset(root="./data", split="val", size=5000, cfg=cfg, seed=43)
    test_ds = AgingDataset(root="./data", split="test", size=5000, cfg=cfg, seed=44)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
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


def run_predictor_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=NON_BLOCKING)
        with torch.set_grad_enabled(is_train):
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = pred.view_as(batch.y)
            loss = criterion(pred, batch.y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / max(len(loader.dataset), 1)


def collect_predictor_outputs(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=NON_BLOCKING)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = pred.view_as(batch.y)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train_predictor(train_loader, val_loader, test_loader, node_feature_dim):
    print("\n" + "=" * 60)
    print("PHASE 1: Training Aging Predictor")
    print("=" * 60)

    predictor = HybridGNNTransformer(node_feature_dim=node_feature_dim, hidden_dim=256, seq_len=1).to(DEVICE)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PREDICTOR_EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = OUT_DIR / "best_predictor.pt"

    for epoch in range(1, PREDICTOR_EPOCHS + 1):
        train_loss = run_predictor_epoch(predictor, train_loader, optimizer=optimizer)
        val_loss = run_predictor_epoch(predictor, val_loader, optimizer=None)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(predictor.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    predictor.load_state_dict(torch.load(best_path, map_location=DEVICE))
    preds, labels = collect_predictor_outputs(predictor, test_loader)
    metrics = regression_metrics(preds, labels)
    metrics["best_epoch"] = best_epoch

    print("\nPredictor Results:")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Best epoch: {best_epoch}")

    return predictor, metrics


def run_trajectory_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE, non_blocking=NON_BLOCKING)
        with torch.set_grad_enabled(is_train):
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            target = batch.y_trajectory
            loss = model.trajectory_loss(pred, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / max(len(loader.dataset), 1)


def collect_trajectory_outputs(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=NON_BLOCKING)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(batch.y_trajectory.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def trajectory_step_r2(preds, labels):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    return [float(r2_score(labels[:, step], preds[:, step])) for step in range(labels.shape[1])]


def train_trajectory(predictor, train_loader, val_loader, test_loader, horizon):
    print("\n" + "=" * 60)
    print("PHASE 2: Training Trajectory Predictor")
    print("=" * 60)

    traj_model = TrajectoryPredictor(gnn_encoder=predictor, horizon=horizon).to(DEVICE)
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

    for epoch in range(1, TRAJECTORY_EPOCHS + 1):
        train_loss = run_trajectory_epoch(traj_model, train_loader, optimizer=optimizer)
        val_loss = run_trajectory_epoch(traj_model, val_loader, optimizer=None)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Encoder LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Head LR: {optimizer.param_groups[1]['lr']:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(traj_model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= TRAJECTORY_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    traj_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    preds, labels = collect_trajectory_outputs(traj_model, test_loader)
    metrics = regression_metrics(preds, labels)
    metrics["per_step_r2"] = trajectory_step_r2(preds, labels)
    metrics["best_epoch"] = best_epoch

    print("\nTrajectory Results:")
    print(f"  R2:   {metrics['r2']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Best epoch: {best_epoch}")
    for step, step_r2 in enumerate(metrics["per_step_r2"], start=1):
        print(f"  Step {step}: R2={step_r2:.4f}")

    return traj_model, metrics


def main():
    OUT_DIR.mkdir(exist_ok=True)
    cfg = load_cfg()
    args = parse_args()
    device_request = args.device or get_device_request(cfg)
    device_desc = configure_runtime(device_request)
    print(f"Device request: {device_request}")
    print(f"Using device: {device_desc}")
    print(f"DataLoader settings: {LOADER_KWARGS}")

    train_ds, val_ds, test_ds = load_datasets(cfg)
    train_loader, val_loader, test_loader = make_loaders(train_ds, val_ds, test_ds)

    node_feature_dim = int(train_ds[0].x.shape[1])
    horizon = int(train_ds[0].y_trajectory.shape[1])
    print(f"Node feature dim: {node_feature_dim}")
    print(f"Trajectory horizon: {horizon}")

    predictor, pred_metrics = train_predictor(train_loader, val_loader, test_loader, node_feature_dim)
    traj_model, traj_metrics = train_trajectory(predictor, train_loader, val_loader, test_loader, horizon)

    metrics = {
        "predictor": pred_metrics,
        "trajectory": traj_metrics,
    }
    with open(OUT_DIR / "real_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nMetrics saved to outputs/real_metrics.json")
    print("DONE")


if __name__ == "__main__":
    main()
