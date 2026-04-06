from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from features.feature_builder import FeatureBuilder
from graph.accelerator_graph import AcceleratorGraph
from simulator.timeloop_runner import TimeloopRunner
from simulator.workload_runner import WorkloadRunner
from utils.runtime_eval import REFERENCE_STRESS_TIME_S, cfg_get, get_model_device, normalize_mapping, simulate_mapping

log = logging.getLogger(__name__)


@dataclass
class ParetoSolution:
    mapping: np.ndarray
    peak_aging: float
    latency: float
    energy: float
    workload_name: str = "ResNet-50"

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload_name,
            "mapping": self.mapping.astype(int).tolist(),
            "peak_aging": float(self.peak_aging),
            "latency": float(self.latency),
            "energy": float(self.energy),
        }


# ---------------------------------------------------------------------------
# Evaluation cache — avoids re-simulating identical mappings
# ---------------------------------------------------------------------------

def _mapping_hash(mapping: np.ndarray) -> str:
    return hashlib.sha1(mapping.astype(np.int32).tobytes()).hexdigest()


class _EvalCache:
    """Thread-unsafe LRU-style mapping → objectives cache."""

    def __init__(self, max_size: int = 4096):
        self._store: dict[str, np.ndarray] = {}
        self._max = max_size
        self.hits = 0
        self.misses = 0

    def get(self, mapping: np.ndarray) -> np.ndarray | None:
        key = _mapping_hash(mapping)
        val = self._store.get(key)
        if val is not None:
            self.hits += 1
        else:
            self.misses += 1
        return val

    def put(self, mapping: np.ndarray, objectives: np.ndarray) -> None:
        if len(self._store) >= self._max:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[_mapping_hash(mapping)] = objectives.copy()


# ---------------------------------------------------------------------------
# Convergence callback — tracks hypervolume & detects stagnation
# ---------------------------------------------------------------------------

def _dominated_hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    """Approximate hypervolume for ≤3 objectives via exact decomposition."""
    if F.ndim != 2 or F.shape[0] == 0:
        return 0.0
    # Filter points dominated by reference
    valid = np.all(F < ref, axis=1)
    F = F[valid]
    if len(F) == 0:
        return 0.0
    # Simple product-of-gaps approximation (exact for 1-obj, fast for 2-3)
    n_obj = F.shape[1]
    if n_obj == 1:
        return float(ref[0] - np.min(F[:, 0]))
    # Sort by first objective and sweep
    order = np.argsort(F[:, 0])
    F = F[order]
    hv = 0.0
    prev_bound = ref.copy()
    for i in range(len(F)):
        contribution = 1.0
        for d in range(n_obj):
            contribution *= max(prev_bound[d] - F[i, d], 0.0)
        hv += contribution
        prev_bound = np.minimum(prev_bound, F[i])
    return float(hv)


class ConvergenceCallback(Callback):
    """Stops NSGA-II when hypervolume stagnates for *patience* generations."""

    def __init__(self, patience: int = 15, min_improvement: float = 1e-6, ref_point: np.ndarray | None = None):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.ref_point = ref_point if ref_point is not None else np.array([1.0, 1e9, 1e9])
        self.hv_history: list[float] = []
        self._stagnant = 0

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        if F is None:
            return
        hv = _dominated_hypervolume(F, self.ref_point)
        self.hv_history.append(hv)
        if len(self.hv_history) >= 2:
            improvement = abs(hv - self.hv_history[-2]) / max(abs(self.hv_history[-2]), 1e-12)
            if improvement < self.min_improvement:
                self._stagnant += 1
            else:
                self._stagnant = 0
        if self._stagnant >= self.patience:
            log.info("NSGA-II converged (hypervolume stagnant for %d gens)", self.patience)
            algorithm.termination.force_termination = True

    @property
    def converged(self) -> bool:
        return self._stagnant >= self.patience


class MappingProblem(Problem):
    def __init__(
        self,
        simulator: TimeloopRunner,
        aging_predictor: Any,
        feature_builder: FeatureBuilder,
        acc_graph: AcceleratorGraph,
        layers: list[dict[str, Any]],
        workload_name: str,
        num_clusters: int,
        stress_time_s: float,
        device,
        cache: _EvalCache | None = None,
        balance_weight: float = 0.0,
    ):
        super().__init__(
            n_var=max(len(layers), 1),
            n_obj=3,
            n_ieq_constr=0,
            xl=np.zeros(max(len(layers), 1), dtype=int),
            xu=np.full(max(len(layers), 1), max(int(num_clusters) - 1, 0), dtype=int),
            vtype=int,
        )
        self.simulator = simulator
        self.aging_predictor = aging_predictor
        self.feature_builder = feature_builder
        self.acc_graph = acc_graph
        self.layers = layers
        self.workload_name = workload_name
        self.num_clusters = max(int(num_clusters), 1)
        self.stress_time_s = float(stress_time_s)
        self.device = device
        self._cache = cache or _EvalCache()
        self._balance_weight = float(balance_weight)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(np.asarray(X))
        objectives = np.zeros((X.shape[0], 3), dtype=np.float64)

        for idx, candidate in enumerate(X):
            mapping = normalize_mapping(candidate, len(self.layers), self.num_clusters)

            # --- cache lookup ---
            cached = self._cache.get(mapping)
            if cached is not None:
                objectives[idx] = cached
                continue

            try:
                metrics = simulate_mapping(
                    simulator=self.simulator,
                    feature_builder=self.feature_builder,
                    graph=self.acc_graph,
                    layers=self.layers,
                    mapping=mapping,
                    workload_name=self.workload_name,
                    stress_time_s=self.stress_time_s,
                    predictor=self.aging_predictor,
                    device=self.device,
                )
                peak = metrics["peak_aging"]
                # Penalise imbalanced aging: peak_aging gets a variance surcharge
                if self._balance_weight > 0.0:
                    peak += self._balance_weight * metrics["aging_variance"]
                obj = np.array(
                    [peak, metrics["latency_norm"], metrics["energy_norm"]],
                    dtype=np.float64,
                )
            except Exception:
                obj = np.array([1.0, 1e9, 1e9], dtype=np.float64)

            objectives[idx] = obj
            self._cache.put(mapping, obj)

        out["F"] = objectives


class NSGA2Optimizer:
    """
    Multi-objective workload-to-cluster mapping optimizer backed by the simulator
    and the trained aging predictor.

    Improvements over baseline:
    - Evaluation caching (avoids duplicate simulations)
    - Hypervolume-based convergence detection with patience
    - Richer initial population (load-balanced, perturbed seeds)
    - Aging-variance penalty for balanced degradation
    """

    def __init__(
        self,
        accelerator_config: DictConfig,
        simulator: TimeloopRunner,
        aging_predictor: Any,
        config: DictConfig,
    ):
        self.accel_cfg = accelerator_config
        self.sim = simulator
        self.predictor = aging_predictor
        self.config = config

        self.pop_size = int(cfg_get(config, "pop_size", cfg_get(config, "population_size", 20)))
        self.crossover_prob = float(cfg_get(config, "crossover_prob", 0.9))
        self.mutation_prob = float(cfg_get(config, "mutation_prob", 0.1))
        self.default_generations = int(cfg_get(config, "n_gen", cfg_get(config, "generations", 10)))
        self.seed = int(cfg_get(config, "seed", 42))
        self.reference_stress_time_s = float(cfg_get(config, "stress_time_s", REFERENCE_STRESS_TIME_S))

        # Convergence detection settings
        self.convergence_patience = int(cfg_get(config, "convergence_patience", 15))
        self.convergence_min_improvement = float(cfg_get(config, "convergence_min_improvement", 1e-6))
        # Aging-variance balance weight (0 = disabled)
        self.balance_weight = float(cfg_get(config, "balance_weight", 0.3))

        self.num_layers = int(cfg_get(accelerator_config, "num_layers", 10))
        self.num_clusters = int(cfg_get(accelerator_config, "mac_clusters", cfg_get(accelerator_config, "num_mac_clusters", 64)))
        self.current_workload_name = str(cfg_get(config, "workload_name", "ResNet-50"))

        self.workload_runner = WorkloadRunner(cfg_get(config, "workloads", None))
        self.feature_builder = FeatureBuilder(accelerator_config)
        self.acc_graph = AcceleratorGraph(accelerator_config)
        self.acc_graph.build()
        self.device = get_model_device(aging_predictor)

        self.pareto_solutions: list[ParetoSolution] = []
        self.pareto_history: dict[str, list[ParetoSolution]] = {}
        self.hv_history: list[float] = []
        self._cache = _EvalCache()

    def _build_sampling(self, initial_mapping: Any, num_layers: int) -> np.ndarray | IntegerRandomSampling:
        rng = np.random.default_rng(self.seed)
        nc = max(self.num_clusters, 1)

        # Diverse seed set
        all_to_one = np.zeros(num_layers, dtype=np.int32)
        round_robin = np.arange(num_layers, dtype=np.int32) % nc
        # Load-balanced: evenly distributed then shuffled
        load_balanced = round_robin.copy()
        rng.shuffle(load_balanced)
        # Perturbed round-robin: RR + 1-gene noise
        perturbed_rr = round_robin.copy()
        flip_idx = rng.integers(0, num_layers, size=max(num_layers // 5, 1))
        perturbed_rr[flip_idx] = rng.integers(0, nc, size=len(flip_idx))

        fixed_seeds = [all_to_one, round_robin, load_balanced, perturbed_rr]

        if initial_mapping is not None:
            seed_mapping = normalize_mapping(initial_mapping, num_layers, self.num_clusters)
            if not any(np.array_equal(seed_mapping, s) for s in fixed_seeds):
                fixed_seeds.append(seed_mapping)

        n_fixed = len(fixed_seeds)
        if self.pop_size <= n_fixed:
            return np.vstack(fixed_seeds[: max(self.pop_size, 1)]).astype(int)

        n_random = self.pop_size - n_fixed
        random_part = rng.integers(0, nc, size=(n_random, num_layers), endpoint=False)
        return np.vstack(fixed_seeds + [random_part]).astype(int)

    def run(self, initial_mapping: np.ndarray, n_gen: int = 200, workload_name: str | None = None) -> list[ParetoSolution]:
        workload_name = str(workload_name or self.current_workload_name)
        layers = self.workload_runner.get_workload_layers(workload_name)
        if not layers:
            raise ValueError(f"No layers available for workload '{workload_name}'")

        num_layers = len(layers)
        n_gen = int(n_gen if n_gen is not None else self.default_generations)

        problem = MappingProblem(
            simulator=self.sim,
            aging_predictor=self.predictor,
            feature_builder=self.feature_builder,
            acc_graph=self.acc_graph,
            layers=layers,
            workload_name=workload_name,
            num_clusters=self.num_clusters,
            stress_time_s=self.reference_stress_time_s,
            device=self.device,
            cache=self._cache,
            balance_weight=self.balance_weight,
        )

        convergence_cb = ConvergenceCallback(
            patience=self.convergence_patience,
            min_improvement=self.convergence_min_improvement,
        )

        algorithm = NSGA2(
            pop_size=max(self.pop_size, 2),
            sampling=self._build_sampling(initial_mapping, num_layers),
            crossover=SBX(prob=self.crossover_prob, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=self.mutation_prob, eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        result = minimize(
            problem,
            algorithm,
            get_termination("n_gen", max(n_gen, 1)),
            seed=self.seed,
            save_history=False,
            verbose=bool(cfg_get(self.config, "verbose", False)),
            callback=convergence_cb,
        )

        self.hv_history = list(convergence_cb.hv_history)
        cache_stats = self._cache
        log.info(
            "NSGA-II finished (%d evals, cache hits=%d misses=%d)",
            cache_stats.hits + cache_stats.misses,
            cache_stats.hits,
            cache_stats.misses,
        )

        self.current_workload_name = workload_name
        self.pareto_solutions = []

        if result.X is None or result.F is None:
            self.pareto_history[workload_name] = []
            return []

        X_front = np.atleast_2d(result.X)
        F_front = np.atleast_2d(result.F)

        order = np.lexsort((F_front[:, 2], F_front[:, 1], F_front[:, 0]))
        for idx in order:
            mapping = normalize_mapping(X_front[idx], num_layers, self.num_clusters)
            scores = F_front[idx]
            self.pareto_solutions.append(
                ParetoSolution(
                    mapping=mapping,
                    peak_aging=float(scores[0]),
                    latency=float(scores[1]),
                    energy=float(scores[2]),
                    workload_name=workload_name,
                )
            )

        self.pareto_history[workload_name] = list(self.pareto_solutions)
        return self.pareto_solutions

    def get_pareto_front(self) -> list[ParetoSolution]:
        return list(self.pareto_solutions)

    def save_pareto_solutions(self, path: Path) -> None:
        fronts = self.pareto_history or {self.current_workload_name: self.pareto_solutions}
        path.parent.mkdir(parents=True, exist_ok=True)

        if len(fronts) == 1:
            solutions = next(iter(fronts.values()))
            mappings_payload: Any = [sol.mapping.astype(int).tolist() for sol in solutions]
            objectives_payload: Any = [
                {
                    "peak_aging": float(sol.peak_aging),
                    "latency": float(sol.latency),
                    "energy": float(sol.energy),
                }
                for sol in solutions
            ]
        else:
            mappings_payload = {
                workload: [sol.mapping.astype(int).tolist() for sol in solutions]
                for workload, solutions in fronts.items()
            }
            objectives_payload = {
                workload: [
                    {
                        "peak_aging": float(sol.peak_aging),
                        "latency": float(sol.latency),
                        "energy": float(sol.energy),
                    }
                    for sol in solutions
                ]
                for workload, solutions in fronts.items()
            }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(mappings_payload, handle, indent=2)

        objectives_path = path.with_name("pareto_objectives.json")
        with open(objectives_path, "w", encoding="utf-8") as handle:
            json.dump(objectives_payload, handle, indent=2)
