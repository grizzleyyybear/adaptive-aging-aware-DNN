from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical


def _ortho_init(module: nn.Module, gain: float = np.sqrt(2)) -> None:
    """Apply orthogonal initialization — standard PPO best practice."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → ReLU → Linear → add."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.apply(lambda m: _ortho_init(m, gain=np.sqrt(2)))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.norm(x)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return residual + out


class RunningMeanStd:
    """Welford's online algorithm for running observation normalization."""

    def __init__(self, shape: tuple[int, ...] = (), eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[np.newaxis, :]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic with:
    - 3-layer trunk with residual connections
    - Orthogonal weight initialization
    - Separate gain for policy (0.01) and value (1.0) heads
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Projection + 2 residual blocks (effectively 5 linear layers)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # Actor head — small init gain encourages near-uniform early policy
        self.actor_head = nn.Sequential(nn.Linear(hidden_dim, action_dim))

        # Critic head
        self.critic_head = nn.Sequential(nn.Linear(hidden_dim, 1))

        # Orthogonal init for trunk
        for m in self.trunk.modules():
            _ortho_init(m, gain=np.sqrt(2))
        # Small gain for actor → near-uniform initial policy
        _ortho_init(self.actor_head[0], gain=0.01)
        # Unit gain for value head
        _ortho_init(self.critic_head[0], gain=1.0)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns Action Logits and State Value Estimate."""
        x = self.trunk(obs)
        logits = self.actor_head(x)
        val = self.critic_head(x)
        return logits, val

    def get_action(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Samples an action from the current policy distribution.

        Returns:
            action: sampled discrete action
            log_prob: log probability of action
            value: state value estimate
        """
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, val

    def act_deterministic(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.get_action(obs, deterministic=True)

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluates log_probs for specific actions (used in PPO update).

        Returns:
            log_probs: [B]
            values: [B, 1] -> [B]
            entropy: [B]
        """
        logits, val = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, val.squeeze(-1), entropy
