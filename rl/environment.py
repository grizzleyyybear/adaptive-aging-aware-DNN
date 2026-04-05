from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aging_models.aging_label_generator import AgingLabelGenerator
from features.feature_builder import FeatureBuilder
from graph.accelerator_graph import AcceleratorGraph
from planning.lifetime_planner import LifetimePlanner
from simulator.workload_runner import WorkloadRunner
from utils.runtime_eval import REFERENCE_STRESS_TIME_S, cfg_get, get_model_device, get_workload_names, normalize_mapping, simulate_mapping


def _cfg_section(container: Any, key: str) -> Any:
    return cfg_get(container, key, {})


class AgingControlEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, *args):
        super().__init__()

        if len(args) == 3:
            cfg, sim, planner = args
            workload_runner = aging_gen = acc_graph = encoder = trajectory_pred = feature_builder = None
        elif len(args) == 9:
            sim, planner, workload_runner, aging_gen, acc_graph, encoder, trajectory_pred, feature_builder, cfg = args
        else:
            raise TypeError(
                "AgingControlEnv expects either (cfg, sim, planner) or "
                "(sim, planner, wr, ag, graph, enc, traj, fb, cfg)"
            )

        self.cfg = cfg
        self.sim = sim
        self.planner: LifetimePlanner = planner

        accel_cfg = cfg_get(cfg, "accelerator", None)
        if accel_cfg in (None, {}):
            accel_cfg = cfg
        if cfg_get(accel_cfg, "mac_clusters", None) is None and cfg_get(accel_cfg, "num_mac_clusters", None) is None:
            accel_cfg = getattr(sim, "cfg", accel_cfg)
        self.acc = acc_graph if acc_graph is not None else getattr(planner, "graph", None)
        if self.acc is None:
            self.acc = AcceleratorGraph(accel_cfg)
            self.acc.build()

        self.wr = workload_runner if workload_runner is not None else WorkloadRunner(cfg_get(cfg, "workloads", None))
        try:
            self.ag = aging_gen if aging_gen is not None else AgingLabelGenerator(cfg=cfg)
        except Exception:
            self.ag = aging_gen
        self.fb = feature_builder if feature_builder is not None else FeatureBuilder(accel_cfg)

        self.enc = encoder
        self.tp = trajectory_pred
        model_for_device = self.enc if self.enc is not None else self.tp
        self.device = get_model_device(model_for_device)
        self.planner.attach_runtime(
            predictor=self.enc,
            simulator=self.sim,
            feature_builder=self.fb,
            aging_generator=self.ag,
            device=self.device,
        )

        env_cfg = _cfg_section(cfg, "environment")
        reward_cfg = _cfg_section(cfg, "reward")
        planning_cfg = _cfg_section(cfg, "planning")
        scheduling_cfg = _cfg_section(cfg, "scheduling")

        self.k_horizon = int(cfg_get(env_cfg, "horizon_length", cfg_get(cfg, "horizon_length", 10)))
        self.W = int(cfg_get(env_cfg, "workload_feature_dim", cfg_get(cfg, "workload_feature_dim", 16)))
        self.L = int(cfg_get(env_cfg, "max_layers", cfg_get(cfg, "max_layers", cfg_get(accel_cfg, "num_layers", 10))))
        self.max_steps = int(cfg_get(env_cfg, "max_episode_steps", self.k_horizon))
        self.time_step_s = float(cfg_get(env_cfg, "step_time_s", 3600.0))
        self.reference_stress_time_s = float(cfg_get(env_cfg, "reference_stress_time_s", REFERENCE_STRESS_TIME_S))
        self.schedule_pattern = str(cfg_get(scheduling_cfg, "default_pattern", "mixed"))

        self.reward_w_peak = float(cfg_get(reward_cfg, "w_peak", 1.0))
        self.reward_w_variance = float(cfg_get(reward_cfg, "w_variance", 0.5))
        self.reward_w_latency = float(cfg_get(reward_cfg, "w_latency", 0.1))
        self.reward_w_lifetime = float(cfg_get(reward_cfg, "w_lifetime", 0.3))
        self.failure_threshold = float(cfg_get(planning_cfg, "failure_threshold", cfg_get(cfg, "failure_threshold", 0.8)))

        self.num_macs = max(int(getattr(self.sim, "num_mac_clusters", cfg_get(accel_cfg, "mac_clusters", 1))), 1)
        self.N = int(self.acc.get_num_nodes())
        self.workload_names = get_workload_names(cfg, self.wr)

        self.action_space = spaces.Discrete(5)
        obs_dim = self.N + (self.N * self.k_horizon) + self.W + self.L + 1 + self.N
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._baseline_metrics_by_workload: dict[str, dict[str, float]] = {}
        self._current_metrics: dict[str, Any] | None = None
        self._current_layers: list[dict[str, Any]] = []
        self.current_workload = self.workload_names[0]
        self.step_count = 0
        self.current_stress_time_s = self.reference_stress_time_s
        self._mapping_int = np.zeros(self.L, dtype=np.int32)
        self.current_mapping = np.zeros(self.L, dtype=np.float32)
        self.aging_vector = np.zeros(self.N, dtype=np.float32)
        self.predicted_trajectory = np.zeros((self.N, self.k_horizon), dtype=np.float32)

    def _mapping_to_obs(self) -> np.ndarray:
        return self._mapping_int.astype(np.float32) / max(self.num_macs - 1, 1)

    def _workload_embedding(self) -> np.ndarray:
        embedding = np.zeros(self.W, dtype=np.float32)
        if self.current_workload in self.workload_names:
            idx = self.workload_names.index(self.current_workload)
            if idx < self.W:
                embedding[idx] = 1.0
        return embedding

    def _get_obs(self) -> np.ndarray:
        progress = np.array([self.step_count / max(self.max_steps, 1)], dtype=np.float32)
        budget_margin = np.clip(self.failure_threshold - self.aging_vector, 0.0, 1.0)

        return np.concatenate(
            [
                self.aging_vector,
                self.predicted_trajectory.reshape(-1),
                self._workload_embedding(),
                self.current_mapping,
                progress,
                budget_margin,
            ]
        ).astype(np.float32)

    def _set_workload(self, workload_name: str) -> None:
        self.current_workload = workload_name
        self._current_layers = list(self.wr.get_workload_layers(workload_name))
        mapping = normalize_mapping(self._mapping_int, len(self._current_layers), self.num_macs)
        self._mapping_int[:] = 0
        self._mapping_int[: len(mapping)] = mapping
        self.current_mapping = self._mapping_to_obs()

    def _ensure_baseline(self) -> None:
        if self.current_workload in self._baseline_metrics_by_workload:
            return
        layers = self._current_layers
        baseline_mapping = np.arange(len(layers), dtype=np.int32) % self.num_macs
        baseline = simulate_mapping(
            simulator=self.sim,
            feature_builder=self.fb,
            graph=self.acc,
            layers=layers,
            mapping=baseline_mapping,
            workload_name=self.current_workload,
            stress_time_s=self.reference_stress_time_s,
            predictor=self.enc,
            aging_generator=self.ag,
            device=self.device,
        )
        self._baseline_metrics_by_workload[self.current_workload] = {
            "latency_cycles": max(float(baseline["latency_cycles"]), 1e-6),
            "energy_pj": max(float(baseline["energy_pj"]), 1e-6),
            "peak_aging": max(float(baseline["peak_aging"]), 1e-6),
        }

    def _evaluate_current_mapping(self, monotonic: bool) -> dict[str, Any]:
        metrics = simulate_mapping(
            simulator=self.sim,
            feature_builder=self.fb,
            graph=self.acc,
            layers=self._current_layers,
            mapping=self._mapping_int[: len(self._current_layers)],
            workload_name=self.current_workload,
            stress_time_s=self.current_stress_time_s,
            predictor=self.enc,
            trajectory_predictor=self.tp,
            aging_generator=self.ag,
            device=self.device,
        )

        aging_scores = np.asarray(metrics["aging_scores"], dtype=np.float32)
        if monotonic:
            aging_scores = np.maximum(self.aging_vector, aging_scores)
        self.aging_vector = aging_scores

        trajectory = metrics["trajectory_scores"]
        if trajectory is None:
            increments = np.linspace(0.01, 0.05, self.k_horizon, dtype=np.float32)
            trajectory = np.clip(self.aging_vector[:, None] + increments[None, :], 0.0, 1.0)
        else:
            trajectory = np.asarray(trajectory, dtype=np.float32)
            if trajectory.ndim != 2:
                trajectory = np.zeros((self.N, self.k_horizon), dtype=np.float32)
            elif trajectory.shape[0] != self.N and trajectory.shape[1] == self.N:
                trajectory = trajectory.T
            if trajectory.shape[0] != self.N:
                trajectory = np.zeros((self.N, self.k_horizon), dtype=np.float32)
            if trajectory.shape[1] != self.k_horizon:
                if trajectory.shape[1] > self.k_horizon:
                    trajectory = trajectory[:, : self.k_horizon]
                else:
                    pad = np.repeat(trajectory[:, -1:], self.k_horizon - trajectory.shape[1], axis=1)
                    trajectory = np.concatenate([trajectory, pad], axis=1)
            trajectory = np.maximum(trajectory, self.aging_vector[:, None])
        self.predicted_trajectory = trajectory.astype(np.float32)

        self._current_metrics = metrics
        self.current_mapping = self._mapping_to_obs()
        return metrics

    def _apply_action(self, action: int) -> None:
        active_layers = len(self._current_layers)
        if active_layers == 0:
            return

        mapping = normalize_mapping(self._mapping_int[:active_layers], active_layers, self.num_macs)
        mac_scores = self.aging_vector[: self.num_macs] if len(self.aging_vector) >= self.num_macs else np.zeros(self.num_macs, dtype=np.float32)

        if action == 1:
            hottest_cluster = int(np.argmax(mac_scores))
            coolest_cluster = int(np.argmin(mac_scores))
            candidates = np.where(mapping == hottest_cluster)[0]
            if len(candidates) > 0:
                mapping[int(candidates[0])] = coolest_cluster
        elif action == 2:
            mapping = (mapping + 1) % self.num_macs
        elif action == 3:
            layer_idx = self.step_count % active_layers
            mapping[layer_idx] = int((mapping[layer_idx] + max(self.num_macs // 2, 1)) % self.num_macs)
        elif action == 4:
            try:
                recommendation = self.planner.recommend_rebalance(self.predicted_trajectory, mapping)
                for layer_idx, cluster_idx in recommendation.get("reassign", []):
                    if 0 <= layer_idx < active_layers:
                        mapping[int(layer_idx)] = int(cluster_idx) % self.num_macs
            except Exception:
                cluster_loads = np.bincount(mapping, minlength=self.num_macs)
                heavy = int(np.argmax(cluster_loads))
                light = int(np.argmin(cluster_loads))
                candidates = np.where(mapping == heavy)[0]
                if len(candidates) > 0:
                    mapping[int(candidates[0])] = light

        self._mapping_int[:] = 0
        self._mapping_int[:active_layers] = mapping
        self.current_mapping = self._mapping_to_obs()

    def step(self, action: int):
        prev_peak = float(np.max(self.aging_vector)) if len(self.aging_vector) else 0.0
        prev_var = float(np.var(self.aging_vector)) if len(self.aging_vector) else 0.0
        prev_lifetime = self.planner.estimate_lifetime_extension(self.aging_vector, self.predicted_trajectory)

        self._apply_action(int(action))
        self.current_stress_time_s += self.time_step_s
        metrics = self._evaluate_current_mapping(monotonic=True)
        self._ensure_baseline()

        curr_peak = float(np.max(self.aging_vector)) if len(self.aging_vector) else 0.0
        curr_var = float(np.var(self.aging_vector)) if len(self.aging_vector) else 0.0
        curr_lifetime = self.planner.estimate_lifetime_extension(self.aging_vector, self.predicted_trajectory)

        baseline = self._baseline_metrics_by_workload[self.current_workload]
        latency_ratio = float(metrics["latency_cycles"] / baseline["latency_cycles"])
        energy_ratio = float(metrics["energy_pj"] / baseline["energy_pj"])

        aging_improvement = (baseline["peak_aging"] - curr_peak) / baseline["peak_aging"]
        latency_improvement = (baseline["latency_cycles"] - float(metrics["latency_cycles"])) / baseline["latency_cycles"]
        aging_mean = float(np.mean(self.aging_vector)) if len(self.aging_vector) else 1e-6
        balance_score = 1.0 - (float(np.std(self.aging_vector)) / max(aging_mean, 1e-6))

        reward = (
            0.5 * aging_improvement
            + 0.3 * balance_score
            + 0.2 * latency_improvement
        )
        reward = float(np.clip(reward, -2.0, 2.0))

        self.step_count += 1
        terminated = bool(curr_peak >= self.failure_threshold)
        truncated = bool(self.step_count >= self.max_steps)

        info = {
            "workload_name": self.current_workload,
            "peak_aging": curr_peak,
            "aging_variance": curr_var,
            "latency_ratio": latency_ratio,
            "energy_ratio": energy_ratio,
            "lifetime_extension": float(curr_lifetime - prev_lifetime),
            "mapping": self._mapping_int[: len(self._current_layers)].astype(int).tolist(),
            "prev_peak_aging": prev_peak,
            "prev_aging_variance": prev_var,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        else:
            self._rng = np.random.default_rng()

        options = options or {}
        self.step_count = 0
        self.current_stress_time_s = self.reference_stress_time_s
        self._mapping_int[:] = 0
        self.current_mapping = self._mapping_to_obs()
        self.aging_vector = np.zeros(self.N, dtype=np.float32)
        self.predicted_trajectory = np.zeros((self.N, self.k_horizon), dtype=np.float32)
        self._current_metrics = None

        if "fixed_workload" in options:
            workload_name = str(options["fixed_workload"])
        elif "workload_stream" in options and options["workload_stream"]:
            workload_name = str(options["workload_stream"][0])
        else:
            available = self.workload_names or ["ResNet-50"]
            workload_name = str(self._rng.choice(available))

        self._set_workload(workload_name)
        initial_mapping = np.arange(len(self._current_layers), dtype=np.int32) % self.num_macs
        self._mapping_int[:] = 0
        self._mapping_int[: len(initial_mapping)] = initial_mapping
        self.current_mapping = self._mapping_to_obs()
        self._ensure_baseline()
        self._evaluate_current_mapping(monotonic=False)
        return self._get_obs(), {"workload_name": self.current_workload}
