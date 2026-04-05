from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    PPO Policy Network structure using shared representations.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor Head (Policy)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
            # Softmax is applied via Categorical distribution dynamically
        )
        
        # Critic Head (Value)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns Action Logits and State Value Estimate.
        """
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
