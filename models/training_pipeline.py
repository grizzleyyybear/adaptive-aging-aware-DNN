import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import wandb
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from omegaconf import DictConfig

from utils.device import (
    configure_torch_runtime,
    dataloader_kwargs,
    describe_device,
    get_device_request,
    resolve_device,
    use_non_blocking,
)

log = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Standardizes training loop for the GNN models.
    """
    def __init__(self, config: DictConfig, model: nn.Module, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = resolve_device(get_device_request(self.config))
        configure_torch_runtime(self.device)
        self.non_blocking = use_non_blocking(self.device)
        self.loader_kwargs = dataloader_kwargs(self.device)
        self.model.to(self.device)
        log.info("TrainingPipeline using %s", describe_device(self.device))
        
        # Configure hyperparameters
        train_cfg = self.config.get('training', {})
        self.epochs = train_cfg.get('epochs', 100)
        self.batch_size = train_cfg.get('batch_size', 32)
        self.lr = train_cfg.get('learning_rate', train_cfg.get('lr', 1e-3))
        self.patience = train_cfg.get('patience', 10)
        self.weight_decay = train_cfg.get('weight_decay', 1e-4)
        
        # We need a proper Train/Val/Test split since the base dataset object doesn't do it automatically
        total_len = len(dataset)
        if total_len == 0:
            raise ValueError("TrainingPipeline requires a non-empty dataset")

        if total_len == 1:
            train_len, val_len, test_len = 1, 0, 0
        elif total_len == 2:
            train_len, val_len, test_len = 1, 0, 1
        else:
            train_len = max(int(0.8 * total_len), 1)
            val_len = max(int(0.1 * total_len), 1)
            test_len = total_len - train_len - val_len
            if test_len <= 0:
                test_len = 1
                if train_len > val_len:
                    train_len -= 1
                else:
                    val_len -= 1
        
        # For simplicity in Phase 2 unless specifically requested, we use random split
        generator = torch.Generator().manual_seed(42)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len], generator=generator)
        
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, **self.loader_kwargs)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, **self.loader_kwargs)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, **self.loader_kwargs)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Handle loss based on model type (single step vs trajectory)
        if hasattr(self.model, 'trajectory_loss'):
            self.criterion = self.model.trajectory_loss
            self.is_trajectory = True
        else:
            self.criterion = nn.MSELoss()
            self.is_trajectory = False
            
    def train(self) -> dict:
        # For trajectory: monitor R² (higher=better). For predictor: monitor loss (lower=better).
        best_val_loss = float('-inf') if self.is_trajectory else float('inf')
        patience_counter = 0
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        prefix = "trajectory_" if self.is_trajectory else "predictor_"
        best_path = checkpoint_dir / f"{prefix}best.pt"
        last_path = checkpoint_dir / f"{prefix}last.pt"
        
        for epoch in range(self.epochs):
            # Training Phase
            self.model.train()
            train_loss = 0.0
            
            for batch in self.train_loader:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
                self.optimizer.zero_grad()
                
                preds = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                target = batch.y_trajectory if self.is_trajectory else batch.y
                loss = self.criterion(preds, target)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item() * batch.num_graphs
                
            train_loss /= max(len(self.train_loader.dataset), 1)
            self.scheduler.step()
            
            # Validation Phase
            val_metrics = self.evaluate(split='val')
            val_loss = val_metrics['loss']
            
            if wandb.run is not None:
                try:
                    wandb.log({
                        f"train/{prefix}loss": train_loss,
                        f"val/{prefix}loss": val_loss,
                        f"val/{prefix}mae": val_metrics['mae'],
                        f"val/{prefix}r2": val_metrics['r2'],
                        "epoch": epoch
                    })
                except Exception as e:
                    log.warning(f"W&B logging failed: {e}")
                
        # Early Stopping and model saving — monitor R² for trajectory, loss for predictor
            monitor_r2 = self.is_trajectory
            if monitor_r2:
                metric_val = val_metrics['r2']
                improved = metric_val > best_val_loss  # best_val_loss reused as best_r2
            else:
                metric_val = val_loss
                improved = metric_val < best_val_loss

            if improved:
                best_val_loss = metric_val
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Save last
        torch.save(self.model.state_dict(), last_path)
        
        # Load best for final test eval
        self.load_checkpoint(best_path)
        test_metrics = self.evaluate(split='test')
        
        return test_metrics
        
    def evaluate(self, split: str = 'test') -> dict:
        self.model.eval()
        loader = self.val_loader if split == 'val' else self.test_loader
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=self.non_blocking)
                preds = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                target = batch.y_trajectory if self.is_trajectory else batch.y
                
                loss = self.criterion(preds, target)
                total_loss += loss.item() * batch.num_graphs
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
        if len(loader.dataset) == 0 or not all_preds:
            return {
                'loss': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'r2': 0.0
            }

        avg_loss = total_loss / max(len(loader.dataset), 1)
        
        preds_np = np.concatenate(all_preds, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        
        mae = mean_absolute_error(targets_np, preds_np)
        rmse = np.sqrt(mean_squared_error(targets_np, preds_np))
        r2 = r2_score(targets_np, preds_np)
        
        return {
            'loss': avg_loss,
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
    def load_checkpoint(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
