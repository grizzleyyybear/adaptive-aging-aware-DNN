from __future__ import annotations

import numpy as np


class MappingChromosome:
    """
    Encodes mapping decisions (Layer -> Cluster).
    """

    def __init__(self, num_layers: int, num_clusters: int):
        self.num_layers = num_layers
        self.num_clusters = num_clusters

    def random_init(self, seed: int = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, self.num_clusters, size=self.num_layers)

    def load_balanced_init(self, seed: int = None) -> np.ndarray:
        """Initialize with even layer-to-cluster distribution."""
        rng = np.random.default_rng(seed)
        base = np.arange(self.num_layers, dtype=np.int32) % self.num_clusters
        rng.shuffle(base)
        return base

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Single point crossover."""
        pt = np.random.randint(1, self.num_layers)
        c1 = np.concatenate([parent1[:pt], parent2[pt:]])
        c2 = np.concatenate([parent2[:pt], parent1[pt:]])
        return c1, c2

    def uniform_crossover(
        self, parent1: np.ndarray, parent2: np.ndarray, swap_prob: float = 0.5, seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform crossover — each gene swaps independently with *swap_prob*."""
        rng = np.random.default_rng(seed)
        mask = rng.random(self.num_layers) < swap_prob
        c1 = np.where(mask, parent2, parent1)
        c2 = np.where(mask, parent1, parent2)
        return c1, c2

    def mutate(self, chromosome: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Random reset mutation."""
        c_new = chromosome.copy()
        mask = np.random.rand(self.num_layers) < mutation_rate
        new_genes = np.random.randint(0, self.num_clusters, size=np.sum(mask))
        c_new[mask] = new_genes
        return c_new

    def is_valid(self, chromosome: np.ndarray, constraints: dict) -> bool:
        """
        Validates if structural dependency limits are met.
        For simple spatial mapping, all bounds [0, C-1] are valid.
        """
        if np.any((chromosome < 0) | (chromosome >= self.num_clusters)):
            return False
        return True

    def repair(self, chromosome: np.ndarray, constraints: dict) -> np.ndarray:
        """Clamps to valid cluster range."""
        return np.clip(chromosome, 0, self.num_clusters - 1)
