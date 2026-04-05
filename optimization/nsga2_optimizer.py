from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from pymoo.algorithms.moo.nsga2 import NSGA2
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

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.atleast_2d(np.asarray(X))
        objectives = np.zeros((X.shape[0], 3), dtype=np.float64)

        for idx, candidate in enumerate(X):
            mapping = normalize_mapping(candidate, len(self.layers), self.num_clusters)
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
                objectives[idx] = np.array(
                    [
                        metrics["peak_aging"],
                        metrics["latency_norm"],
                        metrics["energy_norm"],
                    ],
                    dtype=np.float64,
                )
            except Exception:
                objectives[idx] = np.array([1.0, 1e9, 1e9], dtype=np.float64)

        out["F"] = objectives


class NSGA2Optimizer:
    """
    Multi-objective workload-to-cluster mapping optimizer backed by the simulator
    and the trained aging predictor.
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

    def _build_sampling(self, initial_mapping: Any, num_layers: int) -> np.ndarray | IntegerRandomSampling:
        rng = np.random.default_rng(self.seed)

        # Diverse seed set: all-to-one (bottleneck baseline), round-robin (spread), random remainder
        all_to_one = np.zeros(num_layers, dtype=np.int32)
        round_robin = np.arange(num_layers, dtype=np.int32) % max(self.num_clusters, 1)

        fixed_seeds = [all_to_one, round_robin]

        if initial_mapping is not None:
            seed_mapping = normalize_mapping(initial_mapping, num_layers, self.num_clusters)
            # Only add if it differs from the two above
            if not np.array_equal(seed_mapping, round_robin) and not np.array_equal(seed_mapping, all_to_one):
                fixed_seeds.append(seed_mapping)

        n_fixed = len(fixed_seeds)
        if self.pop_size <= n_fixed:
            return np.vstack(fixed_seeds[:max(self.pop_size, 1)]).astype(int)

        n_random = self.pop_size - n_fixed
        random_part = rng.integers(0, max(self.num_clusters, 1), size=(n_random, num_layers), endpoint=False)
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
