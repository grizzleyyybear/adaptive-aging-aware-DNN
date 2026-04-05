from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig

from utils.runtime_eval import MAX_TTF_TIME_S, compute_physics_ttf, compute_predictor_ttf

class LifetimePlanner:
    """
    Tracks and allocates aging budgets across the accelerator components.
    """
    def __init__(self, accelerator_graph: Any, config: DictConfig):
        self.graph = accelerator_graph
        self.config = config
        
        self.num_nodes = self.graph.get_num_nodes()
        self.failure_threshold = self.config.get('failure_threshold', 0.8) # Normalize score representing failure
        self.predictor = None
        self.simulator = None
        self.feature_builder = None
        self.aging_generator = None
        self.device = None

    def attach_runtime(
        self,
        predictor: Any = None,
        simulator: Any = None,
        feature_builder: Any = None,
        aging_generator: Any = None,
        device: Any = None,
    ) -> None:
        self.predictor = predictor
        self.simulator = simulator
        self.feature_builder = feature_builder
        self.aging_generator = aging_generator
        self.device = device
        
    def allocate_budgets(self, target_lifetime_years: float, strategy: str = 'equalized') -> Dict[int, float]:
        """
        Calculates maximum allowable aging per node to meet target lifetime.
        
        Args:
            target_lifetime_years: Global lifetime goal
            strategy: 'equalized', 'capacity_weighted', 'type_weighted'
            
        Returns:
            Dict mapping node_id to max allowable score.
        """
        # For simplicity, treat the max failure threshold as the total budget available over `target_lifetime_years`
        # We assign an instantaneous "budget" conceptually, but mathematically here 
        # budget_i represents the targeted *maximum* score for that node at the end of life.
        
        budgets = {}
        if strategy == 'equalized':
            # Everyone gets same budget up to threshold
            target = self.failure_threshold
            for i in range(self.num_nodes):
                budgets[i] = target
        elif strategy == 'type_weighted':
            # Routers fail faster, allocate them lower thresholds? 
            # OR allocate them more buffer. Let's dictate budget == allowable degradation.
            for i in range(self.num_nodes):
                ntype = self.graph.get_node_info(i)['type']
                if ntype == 'mac':
                    budgets[i] = self.failure_threshold * 1.0
                elif ntype == 'sram':
                    budgets[i] = self.failure_threshold * 0.9
                else:
                    budgets[i] = self.failure_threshold * 0.8
        else: # Capacity weighted
            total_cap = sum(self.graph.get_node_info(i).get('capacity', 1.0) for i in range(self.num_nodes))
            for i in range(self.num_nodes):
                cap_ratio = self.graph.get_node_info(i).get('capacity', 1.0) / total_cap
                # Scaled arbitrarily around threshold
                budgets[i] = min(1.0, self.failure_threshold * (0.5 + cap_ratio * self.num_nodes))
                
        return budgets

    def check_budget_violations(self, current_aging_vector: np.ndarray, budgets: Dict[int, float] = None) -> List[int]:
        """
        Returns list of node indices exceeding their allocated budget threshold.
        """
        if budgets is None:
            budgets = self.allocate_budgets(10.0, 'equalized')
            
        violations = []
        for i, score in enumerate(current_aging_vector):
            if score > budgets.get(i, self.failure_threshold):
                violations.append(i)
        return violations

    def compute_equalization_reward(self, aging_trajectory: np.ndarray) -> float:
        """
        Computes a scalar reward to incentivize uniform aging.
        Higher is better.
        
        Args:
            aging_trajectory: np.ndarray [T, N] or [N] of scores
        """
        # Extract terminal state if sequence
        if len(aging_trajectory.shape) > 1:
            terminal_scores = aging_trajectory[-1, :]
        else:
            terminal_scores = aging_trajectory
            
        variance = np.var(terminal_scores)
        peak = np.max(terminal_scores)
        
        # reward = -Var(A_i(T)) - λ * max_i(A_i(T))
        lambda_val = self.config.get('penalty_lambda', 2.0)
        
        reward = -float(variance) - (lambda_val * float(peak))
        return reward

    def estimate_lifetime_extension(self, current_aging_vector: np.ndarray, predicted_trajectories: np.ndarray | None = None) -> float:
        """
        Estimate remaining lifetime margin in a normalized 0..1 scale.
        """
        current_peak = float(np.max(current_aging_vector)) if len(current_aging_vector) else 0.0
        if predicted_trajectories is None or np.size(predicted_trajectories) == 0:
            return float(np.clip((self.failure_threshold - current_peak) / max(self.failure_threshold, 1e-6), 0.0, 1.0))

        preds = np.asarray(predicted_trajectories)
        if preds.ndim == 1:
            peak_path = np.array([float(np.max(preds))], dtype=np.float32)
        elif preds.shape[0] == self.num_nodes:
            peak_path = preds.max(axis=0)
        else:
            peak_path = preds.max(axis=1)

        crossings = np.where(peak_path >= self.failure_threshold)[0]
        if len(crossings) == 0:
            return 1.0
        first_crossing = float(crossings[0] + 1)
        return float(np.clip(first_crossing / max(len(peak_path), 1), 0.0, 1.0))

    def recommend_rebalance(self, predicted_trajectories: np.ndarray, current_mapping: np.ndarray) -> dict:
        """
        Analyzes predicted future bottlenecks and suggests greedy mapping corrections.
        
        Args:
            predicted_trajectories: [N, k] future score horizon
            current_mapping: array of cluster assignments [L]
        """
        # 1. Find hottest predicted node at horizon k
        terminal_preds = predicted_trajectories[:, -1]
        hottest_node = int(np.argmax(terminal_preds))
        coolest_node = int(np.argmin(terminal_preds))
        
        # We only really care about transferring MAC clustering loads for simple greedy strategies
        hottest_info = self.graph.get_node_info(hottest_node)
        coolest_info = self.graph.get_node_info(coolest_node)
        
        reassign_actions = []
        
        if hottest_info['type'] == 'mac' and coolest_info['type'] == 'mac':
            h_idx = hottest_info['local_idx']
            c_idx = coolest_info['local_idx']
            
            # Find a layer mapped to the hot cluster and move it
            candidate_layers = np.where(current_mapping == h_idx)[0]
            if len(candidate_layers) > 0:
                layer_to_move = candidate_layers[0] # arbitrarily pick first
                reassign_actions.append((int(layer_to_move), int(c_idx)))
                
        return {
            'swap': [], # full cluster swaps
            'reassign': reassign_actions # single layer reassignments
        }

    def compute_ttf(self, aging_trajectory: np.ndarray, failure_threshold: float = 0.8) -> float:
        """
        Estimates expected Time To Failure (years) via linear extrapolation if not yet failed.
        """
        if len(aging_trajectory.shape) > 1:
            current_score = np.max(aging_trajectory[-1, :])
        else:
            current_score = np.max(aging_trajectory)
            
        if current_score >= failure_threshold:
            return 0.0
            
        if current_score <= 1e-6:
            return 10.0 # upper bound cap
            
        # Assuming normalized timestep unit = 1 Year for simplicity in the proxy calculation
        # Trajectory rate = current_score / current_time 
        # TTF = Threshold / Rate
        current_time_years = 1.0 # arbitrary reference scaling
        rate = current_score / current_time_years
        
        ttf = failure_threshold / rate
        return float(np.clip(ttf, 0.0, 10.0))

    def estimate_failure_time(
        self,
        layers: List[dict],
        mapping: np.ndarray,
        workload_name: str,
        failure_threshold: float | None = None,
        max_time_s: float = MAX_TTF_TIME_S,
    ) -> float:
        """
        Estimate TTF in years using the attached runtime components when available.
        """
        threshold = float(failure_threshold if failure_threshold is not None else self.failure_threshold)

        if self.simulator is not None and self.predictor is not None and self.feature_builder is not None:
            return compute_predictor_ttf(
                simulator=self.simulator,
                feature_builder=self.feature_builder,
                graph=self.graph,
                predictor=self.predictor,
                layers=layers,
                mapping=mapping,
                workload_name=workload_name,
                failure_threshold=threshold,
                max_time_s=max_time_s,
                device=self.device,
            )

        if self.simulator is not None and self.aging_generator is not None:
            return compute_physics_ttf(
                simulator=self.simulator,
                aging_generator=self.aging_generator,
                layers=layers,
                mapping=mapping,
                failure_threshold=threshold,
                max_time_s=max_time_s,
            )

        return self.compute_ttf(np.asarray(mapping, dtype=np.float32), threshold)
