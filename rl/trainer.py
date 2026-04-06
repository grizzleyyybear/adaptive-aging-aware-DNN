from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency behavior
    wandb = None

from rl.environment import AgingControlEnv
from rl.policy_network import ActorCritic, RunningMeanStd
from utils.device import configure_torch_runtime, describe_device, resolve_device

log = logging.getLogger(__name__)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, key):
        return getattr(config, key)
    try:
        return config.get(key, default)
    except AttributeError:
        return default


@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_obs: np.ndarray
    episode_rewards: list[float]


class PPOTrainer:
    """
    PPO with:
    - Linear LR decay
    - Value function clipping
    - Running observation normalization
    - KL divergence early stopping per epoch
    - Periodic evaluation with best-checkpoint saving
    - Entropy coefficient annealing
    """

    def __init__(self, env: AgingControlEnv, policy: ActorCritic, config: Any):
        self.env = env
        self.policy = policy
        self.config = config

        self.n_steps = int(_cfg_get(config, "n_steps", 128))
        self.batch_size = int(_cfg_get(config, "batch_size", 64))
        self.n_epochs = int(_cfg_get(config, "n_epochs", _cfg_get(config, "epochs", 4)))
        self.gamma = float(_cfg_get(config, "gamma", 0.99))
        self.gae_lambda = float(_cfg_get(config, "gae_lambda", 0.95))
        self.clip_range = float(_cfg_get(config, "clip_range", _cfg_get(config, "clip_epsilon", 0.2)))
        self.vf_coef = float(_cfg_get(config, "vf_coef", _cfg_get(config, "value_loss_coeff", 0.5)))
        self.max_grad_norm = float(_cfg_get(config, "max_grad_norm", 0.5))
        self.learning_rate = float(_cfg_get(config, "learning_rate", 3e-4))
        self.total_timesteps_default = int(_cfg_get(config, "total_timesteps", 5120))
        self.n_iterations = int(
            _cfg_get(
                config,
                "n_iterations",
                max(int(np.ceil(self.total_timesteps_default / max(self.n_steps, 1))), 1),
            )
        )

        # --- Entropy annealing ---
        self.ent_coef_start = float(_cfg_get(config, "ent_coef", _cfg_get(config, "entropy_coeff", 0.01)))
        self.ent_coef_end = float(_cfg_get(config, "ent_coef_end", self.ent_coef_start * 0.1))
        self.ent_coef = self.ent_coef_start

        # --- KL early-stopping ---
        self.target_kl = float(_cfg_get(config, "target_kl", 0.03))

        # --- Value clipping ---
        self.clip_range_vf = float(_cfg_get(config, "clip_range_vf", self.clip_range))

        # --- Periodic evaluation ---
        self.eval_interval = int(_cfg_get(config, "eval_interval", 5))
        self.eval_episodes = int(_cfg_get(config, "eval_episodes", 5))

        # --- Observation normalization ---
        obs_shape = (env.observation_space.shape[0],)
        self.obs_normalizer = RunningMeanStd(shape=obs_shape)
        self.normalize_obs = bool(_cfg_get(config, "normalize_obs", True))

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, eps=1e-5)
        # Linear LR decay scheduler
        target_iters = max(int(_cfg_get(config, "n_iterations", self.n_iterations)), 1)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=target_iters,
        )

        self.device = resolve_device(_cfg_get(config, "device", None))
        configure_torch_runtime(self.device)
        self.policy.to(self.device)
        log.info("PPOTrainer using %s", describe_device(self.device))

        self._best_eval_reward = -float("inf")

    def _normalize(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if not self.normalize_obs:
            return obs
        if update:
            self.obs_normalizer.update(obs)
        return self.obs_normalizer.normalize(obs).astype(np.float32)

    def train(self, total_timesteps: int | None = None) -> dict:
        total_timesteps = int(total_timesteps if total_timesteps is not None else self.total_timesteps_default)
        target_iterations = max(int(_cfg_get(self.config, "n_iterations", self.n_iterations)), 1)

        obs, _ = self.env.reset()
        obs = self._normalize(obs)
        global_step = 0

        metrics = {
            "reward": [], "policy_loss": [], "value_loss": [],
            "entropy": [], "approx_kl": [], "learning_rate": [],
        }

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = checkpoint_dir / "rl_policy_final.pt"
        best_path = checkpoint_dir / "rl_policy_best.pt"

        for iteration in range(target_iterations):
            # Anneal entropy coefficient linearly
            progress = iteration / max(target_iterations - 1, 1)
            self.ent_coef = self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * progress

            buffer = self._collect_rollouts(obs)
            obs = self._normalize(buffer.next_obs, update=False)
            global_step += len(buffer.rewards)

            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, next_value = self.policy(next_obs_tensor)
                next_value = next_value.squeeze(0).squeeze(-1)

            advantages, returns = self._compute_gae(buffer.rewards, buffer.values, buffer.dones, next_value)
            update_metrics = self._ppo_update(buffer, advantages, returns)

            # Step LR scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            mean_reward = float(np.mean(buffer.episode_rewards)) if buffer.episode_rewards else 0.0
            metrics["reward"].append(mean_reward)
            metrics["policy_loss"].append(update_metrics["policy_loss"])
            metrics["value_loss"].append(update_metrics["value_loss"])
            metrics["entropy"].append(update_metrics["entropy"])
            metrics["approx_kl"].append(update_metrics["approx_kl"])
            metrics["learning_rate"].append(current_lr)

            print(
                f"[PPO] Iteration {iteration + 1}/{target_iterations} | "
                f"reward={mean_reward:.4f} | "
                f"ploss={update_metrics['policy_loss']:.4f} | "
                f"vloss={update_metrics['value_loss']:.4f} | "
                f"kl={update_metrics['approx_kl']:.4f} | "
                f"lr={current_lr:.2e} | "
                f"ent_c={self.ent_coef:.4f}"
            )

            if wandb is not None and getattr(wandb, "run", None) is not None:
                wandb.log(
                    {
                        "rl/reward": mean_reward,
                        "rl/policy_loss": update_metrics["policy_loss"],
                        "rl/value_loss": update_metrics["value_loss"],
                        "rl/entropy": update_metrics["entropy"],
                        "rl/approx_kl": update_metrics["approx_kl"],
                        "rl/learning_rate": current_lr,
                        "rl/ent_coef": self.ent_coef,
                        "global_step": global_step,
                    }
                )

            # --- Periodic evaluation + best checkpoint ---
            if (iteration + 1) % self.eval_interval == 0 or iteration == target_iterations - 1:
                eval_result = self.evaluate(n_episodes=self.eval_episodes)
                eval_reward = eval_result["mean_reward"]
                log.info(
                    "[PPO] Eval @iter %d: reward=%.4f  peak_aging=%.4f",
                    iteration + 1, eval_reward, eval_result["mean_peak_aging"],
                )
                if eval_reward > self._best_eval_reward:
                    self._best_eval_reward = eval_reward
                    torch.save(self.policy.state_dict(), best_path)
                    log.info("[PPO] New best policy saved (reward=%.4f)", eval_reward)

            if global_step >= total_timesteps:
                break

        torch.save(self.policy.state_dict(), final_path)
        return metrics

    def _collect_rollouts(self, initial_obs: np.ndarray) -> RolloutBuffer:
        obs_items = []
        action_items = []
        log_prob_items = []
        reward_items = []
        done_items = []
        value_items = []

        obs = np.asarray(initial_obs, dtype=np.float32)
        episode_rewards: list[float] = []
        running_reward = 0.0

        for _ in range(self.n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_tensor)

            next_obs, reward, terminated, truncated, _ = self.env.step(int(action.item()))
            done = bool(terminated or truncated)

            obs_items.append(obs_tensor.squeeze(0).cpu())
            action_items.append(action.squeeze(0).cpu() if action.ndim > 0 else action.cpu())
            log_prob_items.append(log_prob.squeeze(0).cpu() if log_prob.ndim > 0 else log_prob.cpu())
            reward_items.append(torch.tensor(float(reward), dtype=torch.float32))
            done_items.append(torch.tensor(float(done), dtype=torch.float32))
            value_items.append(value.squeeze(0).squeeze(-1).cpu())

            running_reward += float(reward)

            if done:
                episode_rewards.append(running_reward)
                running_reward = 0.0
                next_obs, _ = self.env.reset()

            next_obs = np.asarray(next_obs, dtype=np.float32)
            obs = self._normalize(next_obs)

        if not episode_rewards:
            episode_rewards.append(running_reward)

        return RolloutBuffer(
            obs=torch.stack(obs_items),
            actions=torch.stack(action_items).long(),
            log_probs=torch.stack(log_prob_items),
            rewards=torch.stack(reward_items),
            dones=torch.stack(done_items),
            values=torch.stack(value_items),
            next_obs=next_obs,
            episode_rewards=episode_rewards,
        )

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.tensor(0.0, dtype=torch.float32)
        next_val = next_value.detach().cpu().float()

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
            advantages[t] = last_advantage
            next_val = values[t]

        returns = advantages + values
        return advantages, returns

    def _ppo_update(self, buffer: RolloutBuffer, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        obs = buffer.obs.to(self.device)
        actions = buffer.actions.to(self.device)
        old_log_probs = buffer.log_probs.to(self.device)
        old_values = buffer.values.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(obs))
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0
        kl_epoch = 0.0
        update_count = 0
        early_stopped = False

        for epoch in range(self.n_epochs):
            if early_stopped:
                break
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_old_values = old_values[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                new_log_probs, values, entropy = self.policy.evaluate_actions(b_obs, b_actions)

                # --- KL divergence early stopping ---
                log_ratio = new_log_probs - b_old_log_probs
                approx_kl = float(((torch.exp(log_ratio) - 1) - log_ratio).mean().item())
                if approx_kl > 1.5 * self.target_kl:
                    early_stopped = True
                    break

                ratio = torch.exp(log_ratio)
                surrogate_1 = ratio * b_advantages
                surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * b_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # --- Clipped value loss ---
                values_clipped = b_old_values + torch.clamp(
                    values - b_old_values, -self.clip_range_vf, self.clip_range_vf
                )
                vf_loss_unclipped = (values - b_returns) ** 2
                vf_loss_clipped = (values_clipped - b_returns) ** 2
                value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

                entropy_bonus = entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_loss_epoch += float(policy_loss.item())
                value_loss_epoch += float(value_loss.item())
                entropy_epoch += float(entropy_bonus.item())
                kl_epoch += approx_kl
                update_count += 1

        update_count = max(update_count, 1)
        return {
            "policy_loss": policy_loss_epoch / update_count,
            "value_loss": value_loss_epoch / update_count,
            "entropy": entropy_epoch / update_count,
            "approx_kl": kl_epoch / update_count,
        }

    def evaluate(self, n_episodes: int = 10) -> dict:
        self.policy.eval()
        rewards = []
        peak_aging = []
        lifetime_improvements = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            obs = self._normalize(obs, update=False)
            done = False
            ep_reward = 0.0
            final_info: dict[str, Any] = {}

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = self.policy.act_deterministic(obs_tensor)
                next_obs, reward, terminated, truncated, info = self.env.step(int(action.item()))
                next_obs = self._normalize(np.asarray(next_obs, dtype=np.float32), update=False)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                final_info = info
                obs = next_obs

            rewards.append(ep_reward)
            peak_aging.append(float(final_info.get("peak_aging", 0.0)))
            lifetime_improvements.append(float(final_info.get("lifetime_extension", 0.0)))

        self.policy.train()
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_peak_aging": float(np.mean(peak_aging)) if peak_aging else 0.0,
            "mean_lifetime_improvement": float(np.mean(lifetime_improvements)) if lifetime_improvements else 0.0,
        }
