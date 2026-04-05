"""Performance metrics for the DNN accelerator."""

import numpy as np
from typing import Dict, List


def compute_speedup(baseline_latency_ms: float, optimised_latency_ms: float) -> float:
    """Speedup of optimised over baseline."""
    return baseline_latency_ms / max(optimised_latency_ms, 1e-9)


def compute_energy_efficiency(throughput_gops: float, power_uw: float) -> float:
    """GOPS per watt."""
    return throughput_gops / max(power_uw * 1e-6, 1e-12)


def compute_lifetime_extension(
    baseline_years: float,
    optimised_years: float
) -> float:
    """Percentage lifetime extension."""
    return (optimised_years - baseline_years) / max(baseline_years, 1e-9) * 100.0


def compute_accuracy_degradation(
    baseline_accuracy: float,
    degraded_accuracy: float
) -> float:
    """Absolute accuracy drop (percentage points)."""
    return baseline_accuracy - degraded_accuracy


def summarise_simulation_results(results: Dict) -> Dict:
    """Aggregate simulation results into paper-ready summary stats."""
    latencies  = [r.latency_ms        for r in results.values()]
    energies   = [r.energy_uj         for r in results.values()]
    utils      = [r.utilisation       for r in results.values()]
    throughputs= [r.throughput_gops   for r in results.values()]

    return {
        'mean_latency_ms':     float(np.mean(latencies)),
        'std_latency_ms':      float(np.std(latencies)),
        'mean_energy_uj':      float(np.mean(energies)),
        'std_energy_uj':       float(np.std(energies)),
        'mean_utilisation':    float(np.mean(utils)),
        'peak_throughput_gops':float(np.max(throughputs)),
        'total_energy_uj':     float(np.sum(energies)),
        'total_latency_ms':    float(np.sum(latencies)),
    }

# Re-export from reliability_metrics for backward compatibility
from evaluation.reliability_metrics import PerformanceMetrics
