import torch
import torch.nn as nn
from models.hybrid_gnn_transformer import HybridGNNTransformer


class TrajectoryPredictor(nn.Module):
    """
    Trajectory predictor with variance-aware design.

    Improvements over v1:
    - Concatenates current aging prediction to encoder features, giving
      the head explicit access to the operating point.
    - Wider 512-unit layers and GELU activations for better non-linear fit.
    - Sigmoid output ensures predictions stay in [0, 1].
    """

    def __init__(
        self,
        gnn_encoder: HybridGNNTransformer,
        hidden_dim: int = 256,
        horizon: int = 10,
        gamma: float = 0.95,
    ):
        super().__init__()
        self.encoder = gnn_encoder
        self.horizon = horizon
        self.gamma = gamma
        enc_dim = getattr(self.encoder, "hidden_dim", hidden_dim)

        # Input: encoder features [enc_dim] + current prediction [1]
        self.traj_head = nn.Sequential(
            nn.Linear(enc_dim + 1, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, horizon),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.encoder.encode_graph(x, edge_index, edge_attr, batch)
        with torch.no_grad():
            current_pred = self.encoder.head(h)  # [N, 1]
        h_aug = torch.cat([h, current_pred], dim=1)  # [N, enc_dim + 1]
        return self.traj_head(h_aug)  # [N, horizon]

    def trajectory_loss(self, pred, target, var_weight: float = 0.1):
        horizon = pred.shape[1]
        weights = torch.tensor(
            [self.gamma ** j for j in range(horizon)],
            dtype=pred.dtype,
            device=pred.device,
        )
        # Discounted MSE
        mse_loss = (weights * (pred - target) ** 2).mean()
        # Variance matching: penalise if predicted spread is too narrow
        pred_var = pred.var(dim=0).mean()
        target_var = target.var(dim=0).mean()
        var_loss = (pred_var - target_var).abs()
        return mse_loss + var_weight * var_loss
