"""
Analytical hardware simulator for DNN accelerator performance estimation.

Replaces a real Timeloop call with a closed-form roofline model.
This is methodologically valid for research — Timeloop itself is an
analytical mapper/model, and many published papers use equivalent formulations.

Paper citation note: "We use an analytical roofline model to estimate
hardware performance metrics (latency, energy, utilisation) for each
DNN mapping, following the methodology of [Timeloop/Maestro-style analysis]."
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    if hasattr(container, key):
        return getattr(container, key)
    try:
        return container.get(key, default)
    except AttributeError:
        return default


def _cfg_pick(container: Any, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = _cfg_get(container, key, None)
        if value is not None:
            return value
    return default


@dataclass
class AcceleratorConfig:
    """Hardware parameters loaded from config."""
    num_pes: int = 256                  # total processing elements
    pe_array_rows: int = 16
    pe_array_cols: int = 16
    mac_clusters: int = 64
    sram_banks: int = 16
    noc_routers: int = 8
    num_layers: int = 10
    mac_per_pe: int = 1                 # MACs per PE per cycle
    sram_kb: float = 256.0              # on-chip SRAM in KB
    dram_bw_gb_s: float = 51.2          # off-chip DRAM bandwidth GB/s
    noc_bw_gb_s: float = 512.0          # on-chip NoC bandwidth GB/s
    freq_mhz: float = 1000.0            # clock frequency MHz
    voltage_v: float = 1.0              # supply voltage (normalised ref)
    # Energy constants (pJ per operation)
    mac_energy_pj: float = 0.25         # multiply-accumulate
    sram_rd_energy_pj: float = 1.5      # SRAM read per byte
    dram_rd_energy_pj: float = 70.0     # DRAM read per byte
    # Aging degradation (applied externally before calling simulate)
    aging_freq_degrade: float = 0.0     # fractional frequency loss (0–1)
    aging_leak_increase: float = 0.0    # fractional leakage increase (0–1)
    # NoC / mapping-aware energy and latency parameters
    noc_latency_cycles: float = 100.0      # cycles per cross-cluster transfer
    noc_energy_per_byte_pj: float = 2.0    # pJ per byte transferred via NoC
    idle_leakage_pj_per_cycle: float = 0.005  # leakage per idle cluster per cycle


@dataclass
class LayerSpec:
    """One DNN layer to simulate."""
    name:       str
    layer_type: str          # 'conv', 'fc', 'pool', 'bn'
    # Convolutional / FC dimensions
    N: int = 1               # batch size
    C: int = 3               # input channels
    K: int = 64              # output channels / filters
    R: int = 3               # filter height (1 for FC)
    S: int = 3               # filter width  (1 for FC)
    P: int = 224             # output height (1 for FC)
    Q: int = 224             # output width  (1 for FC)
    stride: int = 1


@dataclass
class SimResult:
    """Output of one simulation run."""
    layer_name: str
    latency_cycles: float
    latency_ms: float
    energy_pj: float
    energy_uj: float
    throughput_gops: float
    utilisation: float        # PE array utilisation 0–1
    memory_bound: bool
    dram_accesses_bytes: float
    compute_intensity: float     # FLOPs / byte (arithmetic intensity)
    # Per-node (PE) degradation contribution
    per_pe_stress: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))
    mac_utilization: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    sram_access_rate: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    noc_traffic: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    switching_activity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def total_latency_cycles(self) -> float:
        return self.latency_cycles

    @property
    def total_energy_pj(self) -> float:
        return self.energy_pj

    @property
    def avg_switching_activity(self) -> np.ndarray:
        return self.switching_activity

    @property
    def avg_mac_utilization(self) -> np.ndarray:
        return self.mac_utilization

    @property
    def avg_sram_access_rate(self) -> np.ndarray:
        return self.sram_access_rate

    @property
    def avg_noc_traffic(self) -> np.ndarray:
        return self.noc_traffic


def normalize_accelerator_config(accel_cfg: Any) -> AcceleratorConfig:
    """Build a canonical AcceleratorConfig from dict, DictConfig, or dataclass inputs."""
    if isinstance(accel_cfg, AcceleratorConfig):
        return accel_cfg

    pe_array = _cfg_get(accel_cfg, "pe_array", [16, 16]) or [16, 16]
    rows = int(_cfg_pick(accel_cfg, ["pe_array_rows", "pe_rows"], pe_array[0]))
    cols = int(_cfg_pick(accel_cfg, ["pe_array_cols", "pe_cols"], pe_array[1]))
    num_pes = int(_cfg_get(accel_cfg, "num_pes", rows * cols))

    mac_clusters = int(_cfg_pick(accel_cfg, ["mac_clusters", "num_mac_clusters"], rows * cols))
    sram_banks = int(_cfg_pick(accel_cfg, ["sram_banks", "num_sram_banks"], max(mac_clusters // 2, 1)))
    noc_routers = int(_cfg_pick(accel_cfg, ["noc_routers", "num_noc_routers"], max(sram_banks // 2, 1)))

    clock_ghz = float(_cfg_pick(accel_cfg, ["clock_frequency_ghz", "clock_ghz"], 1.0))
    freq_mhz = float(_cfg_get(accel_cfg, "freq_mhz", clock_ghz * 1000.0))
    voltage_v = float(_cfg_pick(accel_cfg, ["voltage_v", "supply_voltage"], 0.8))

    return AcceleratorConfig(
        num_pes=num_pes,
        pe_array_rows=rows,
        pe_array_cols=cols,
        mac_clusters=mac_clusters,
        sram_banks=sram_banks,
        noc_routers=noc_routers,
        num_layers=int(_cfg_get(accel_cfg, "num_layers", 10)),
        mac_per_pe=int(_cfg_pick(accel_cfg, ["mac_per_pe", "ops_per_cycle"], 1)),
        sram_kb=float(_cfg_get(accel_cfg, "sram_kb", 256.0)),
        dram_bw_gb_s=float(_cfg_get(accel_cfg, "dram_bw_gb_s", 51.2)),
        noc_bw_gb_s=float(_cfg_pick(accel_cfg, ["noc_bw_gb_s", "noc_bandwidth_gbps"], 512.0)),
        freq_mhz=freq_mhz,
        voltage_v=voltage_v,
        mac_energy_pj=float(_cfg_pick(accel_cfg, ["mac_energy_pj", "mac_energy_pj_per_op"], 0.25)),
        sram_rd_energy_pj=float(_cfg_pick(accel_cfg, ["sram_rd_energy_pj", "sram_read_energy_pj"], 1.5)),
        dram_rd_energy_pj=float(_cfg_get(accel_cfg, "dram_rd_energy_pj", 70.0)),
        aging_freq_degrade=float(_cfg_get(accel_cfg, "aging_freq_degrade", 0.0)),
        aging_leak_increase=float(_cfg_get(accel_cfg, "aging_leak_increase", 0.0)),
        noc_latency_cycles=float(_cfg_get(accel_cfg, "noc_latency_cycles", 100.0)),
        noc_energy_per_byte_pj=float(_cfg_get(accel_cfg, "noc_energy_per_byte_pj", 2.0)),
        idle_leakage_pj_per_cycle=float(_cfg_get(accel_cfg, "idle_leakage_pj_per_cycle", 0.005)),
    )


class AnalyticalSimulator:
    """
    Roofline-based analytical performance model.

    For a CONV layer the total MACs are:
        MACs = N * K * C * R * S * P * Q / (stride^2)

    Latency (compute-bound):
        cycles_compute = MACs / (num_pes * mac_per_pe)

    Latency (memory-bound):
        bytes_ifmap  = N * C * (P*stride + R - 1) * (Q*stride + S - 1) * 4  (float32)
        bytes_weight = K * C * R * S * 4
        bytes_ofmap  = N * K * P * Q * 4
        total_bytes  = bytes_ifmap + bytes_weight + bytes_ofmap
        cycles_mem   = total_bytes / (dram_bw_bytes_per_cycle)

    Actual latency = max(cycles_compute, cycles_mem) — roofline law.
    """

    def __init__(self, accel_cfg: Any):
        self.cfg = normalize_accelerator_config(accel_cfg)
        self.num_mac_clusters = self.cfg.mac_clusters
        self.num_sram_banks = self.cfg.sram_banks
        self.num_noc_routers = self.cfg.noc_routers
        # Effective frequency accounting for aging degradation
        self.eff_freq_mhz = self.cfg.freq_mhz * (1.0 - self.cfg.aging_freq_degrade)
        self.eff_freq_hz  = self.eff_freq_mhz * 1e6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_layer(self, layer: LayerSpec) -> SimResult:
        """Simulate a single DNN layer and return performance metrics."""
        if layer.layer_type == 'conv':
            return self._simulate_conv(layer)
        elif layer.layer_type == 'fc':
            return self._simulate_fc(layer)
        elif layer.layer_type in ('pool', 'bn'):
            return self._simulate_elementwise(layer)
        else:
            return self._simulate_conv(layer)  # fallback

    def simulate_workload(self, layers: List[LayerSpec]) -> Dict[str, SimResult]:
        """Simulate a full DNN and aggregate results."""
        results = {}
        for layer in layers:
            results[layer.name] = self.simulate_layer(layer)
        return results

    def run_layer(self, layer_cfg: Dict[str, Any] | LayerSpec, mapping: Any = None) -> SimResult:
        """Compatibility wrapper used throughout the repo and tests."""
        layer = layer_cfg if isinstance(layer_cfg, LayerSpec) else self._layer_spec_from_dict(layer_cfg, 0)
        result = self.simulate_layer(layer)
        return self._attach_activity_traces(result, mapping=mapping)

    def run_workload(self, layers: List[Dict[str, Any] | LayerSpec], mapping: Any = None) -> SimResult:
        """Simulate a workload and expose aggregate activity traces."""
        if not layers:
            return self._empty_workload_result()

        mapping_arr = self._normalize_mapping(mapping, len(layers))
        layer_results = []
        for idx, layer_cfg in enumerate(layers):
            layer = layer_cfg if isinstance(layer_cfg, LayerSpec) else self._layer_spec_from_dict(layer_cfg, idx)
            layer_mapping = [int(mapping_arr[idx])]
            layer_results.append(self._attach_activity_traces(self.simulate_layer(layer), mapping=layer_mapping))

        mac_util, sram_access, noc_traffic = self._aggregate_mapping_activity(layer_results, mapping_arr)
        switching = np.concatenate([mac_util, sram_access, noc_traffic]).astype(np.float32)

        total_latency_cycles = self._compute_mapping_aware_latency(layer_results, mapping_arr)
        total_latency_ms = (total_latency_cycles / self.eff_freq_hz) * 1e3
        total_energy_pj = self._compute_mapping_aware_energy(layer_results, mapping_arr)
        total_energy_uj = total_energy_pj * 1e-6
        total_dram_bytes = float(sum(r.dram_accesses_bytes for r in layer_results))
        total_throughput = float(sum(r.throughput_gops for r in layer_results))
        mean_utilisation = float(np.mean([r.utilisation for r in layer_results]))
        mean_compute_intensity = float(np.mean([r.compute_intensity for r in layer_results]))
        latency_weights = np.asarray([max(r.latency_cycles, 1.0) for r in layer_results], dtype=np.float64)
        latency_weights /= max(float(np.sum(latency_weights)), 1.0)
        mean_stress = np.average(
            np.stack([r.per_pe_stress for r in layer_results], axis=0),
            axis=0,
            weights=latency_weights,
        ).astype(np.float32)

        return SimResult(
            layer_name="workload",
            latency_cycles=total_latency_cycles,
            latency_ms=total_latency_ms,
            energy_pj=total_energy_pj,
            energy_uj=total_energy_uj,
            throughput_gops=total_throughput,
            utilisation=mean_utilisation,
            memory_bound=any(r.memory_bound for r in layer_results),
            dram_accesses_bytes=total_dram_bytes,
            compute_intensity=mean_compute_intensity,
            per_pe_stress=mean_stress,
            mac_utilization=mac_util,
            sram_access_rate=sram_access,
            noc_traffic=noc_traffic,
            switching_activity=switching,
        )

    def aggregate_metrics(self, results: Dict[str, SimResult]) -> Dict:
        """Compute workload-level summary statistics."""
        total_latency_ms  = sum(r.latency_ms  for r in results.values())
        total_energy_uj   = sum(r.energy_uj   for r in results.values())
        avg_utilisation   = np.mean([r.utilisation for r in results.values()])
        total_gops        = sum(r.throughput_gops for r in results.values())
        mem_bound_layers  = sum(1 for r in results.values() if r.memory_bound)

        return {
            'total_latency_ms':  total_latency_ms,
            'total_energy_uj':   total_energy_uj,
            'avg_utilisation':   avg_utilisation,
            'total_gops':        total_gops,
            'mem_bound_fraction': mem_bound_layers / max(len(results), 1),
            'energy_delay_product': total_energy_uj * total_latency_ms,
        }

    # ------------------------------------------------------------------
    # Mapping-aware aggregate latency and energy
    # ------------------------------------------------------------------

    def _compute_mapping_aware_latency(self, layer_results: List[SimResult], mapping_arr: np.ndarray) -> float:
        """
        Clusters execute in parallel; total latency is dominated by the busiest
        cluster plus mapping-dependent communication and synchronization costs.
        """
        cluster_loads: dict[int, float] = defaultdict(float)
        for idx, result in enumerate(layer_results):
            cluster = int(mapping_arr[idx])
            cluster_loads[cluster] += result.latency_cycles

        parallel_latency = max(cluster_loads.values()) if cluster_loads else 0.0
        active_loads = np.asarray(list(cluster_loads.values()), dtype=np.float64)
        mean_active_load = float(np.mean(active_loads)) if active_loads.size else 0.0
        imbalance_ratio = parallel_latency / max(mean_active_load, 1.0)

        noc_bw_bytes_per_cycle = max((self.cfg.noc_bw_gb_s * 1e9) / self.eff_freq_hz, 1e-6)
        transitions = self._collect_intercluster_transfers(layer_results, mapping_arr)
        transfer_cycles = sum(t["bytes"] * max(t["hops"], 1) / noc_bw_bytes_per_cycle for t in transitions)
        setup_cycles = sum(self.cfg.noc_latency_cycles * (1.0 + 0.15 * max(t["hops"] - 1, 0)) for t in transitions)
        contention = 1.0 + 0.12 * max(imbalance_ratio - 1.0, 0.0) + 0.05 * max(len(cluster_loads) - 1, 0)
        sync_penalty = parallel_latency * 0.08 * max(imbalance_ratio - 1.0, 0.0)
        return parallel_latency + (transfer_cycles * contention) + setup_cycles + sync_penalty

    def _compute_mapping_aware_energy(self, layer_results: List[SimResult], mapping_arr: np.ndarray) -> float:
        """
        Total energy = compute energy + mapping-dependent transfer/reuse costs
        + leakage from active/idle clusters over the workload duration.
        """
        compute_energy = sum(r.energy_pj for r in layer_results)
        transitions = self._collect_intercluster_transfers(layer_results, mapping_arr)
        noc_energy = sum(
            t["bytes"] * max(t["hops"], 1) * self.cfg.noc_energy_per_byte_pj
            for t in transitions
        )
        data_reuse_penalty = sum(
            t["bytes"] * 0.08 * self.cfg.sram_rd_energy_pj * (1.0 + 0.2 * max(t["hops"] - 1, 0))
            for t in transitions
        )

        active_clusters = len(set(int(c) for c in mapping_arr))
        idle_clusters = max(self.cfg.mac_clusters - active_clusters, 0)
        total_latency = self._compute_mapping_aware_latency(layer_results, mapping_arr)
        active_leakage = active_clusters * self.cfg.idle_leakage_pj_per_cycle * total_latency * 0.35
        idle_leakage = idle_clusters * self.cfg.idle_leakage_pj_per_cycle * total_latency
        wakeup_energy = active_clusters * self.cfg.noc_latency_cycles * 5.0

        return compute_energy + noc_energy + data_reuse_penalty + active_leakage + idle_leakage + wakeup_energy

    # ------------------------------------------------------------------
    # Layer-type simulators
    # ------------------------------------------------------------------

    def _simulate_conv(self, layer: LayerSpec) -> SimResult:
        cfg = self.cfg
        N, K, C, R, S, P, Q = layer.N, layer.K, layer.C, layer.R, layer.S, layer.P, layer.Q
        stride = layer.stride

        total_macs = N * K * C * R * S * P * Q

        # --- compute bound ---
        peak_macs_per_cycle = cfg.num_pes * cfg.mac_per_pe
        cycles_compute = total_macs / peak_macs_per_cycle

        # --- memory bound ---
        H_in = P * stride + R - 1
        W_in = Q * stride + S - 1
        bytes_ifmap  = N * C * H_in * W_in * 4
        bytes_weight = K * C * R * S * 4
        bytes_ofmap  = N * K * P * Q * 4
        total_bytes  = bytes_ifmap + bytes_weight + bytes_ofmap

        dram_bw_bytes_per_cycle = (cfg.dram_bw_gb_s * 1e9) / self.eff_freq_hz
        cycles_mem = total_bytes / dram_bw_bytes_per_cycle

        # --- roofline ---
        cycles_total = max(cycles_compute, cycles_mem)
        memory_bound = cycles_mem > cycles_compute

        # --- time and energy ---
        latency_ms   = (cycles_total / self.eff_freq_hz) * 1e3
        mac_energy   = total_macs * cfg.mac_energy_pj
        mem_energy   = total_bytes * cfg.dram_rd_energy_pj
        # Leakage: proportional to cycles and num_pes
        leakage_pj   = cycles_total * cfg.num_pes * 0.01 * (1.0 + cfg.aging_leak_increase)
        energy_pj    = mac_energy + mem_energy + leakage_pj

        # --- utilisation ---
        utilisation = min(1.0, cycles_compute / cycles_total)

        # --- compute intensity (FLOPs/byte) ---
        compute_intensity = (2 * total_macs) / max(total_bytes, 1)

        # --- throughput ---
        throughput_gops = (2 * total_macs) / (latency_ms * 1e-3) / 1e9

        # --- per-PE stress (for aging model input) ---
        per_pe_stress = self._compute_pe_stress(total_macs, utilisation)

        return SimResult(
            layer_name=layer.name,
            latency_cycles=cycles_total,
            latency_ms=latency_ms,
            energy_pj=energy_pj,
            energy_uj=energy_pj * 1e-6,
            throughput_gops=throughput_gops,
            utilisation=utilisation,
            memory_bound=memory_bound,
            dram_accesses_bytes=total_bytes,
            compute_intensity=compute_intensity,
            per_pe_stress=per_pe_stress,
        )

    def _simulate_fc(self, layer: LayerSpec) -> SimResult:
        """Treat FC as a CONV with 1x1 spatial dimensions."""
        layer.R = layer.S = layer.P = layer.Q = 1
        layer.stride = 1
        return self._simulate_conv(layer)

    def _simulate_elementwise(self, layer: LayerSpec) -> SimResult:
        """Pooling / BN — memory-bandwidth limited, low compute."""
        cfg = self.cfg
        total_bytes = layer.N * layer.C * layer.P * layer.Q * 4 * 2  # read + write
        dram_bw_bytes_per_cycle = (cfg.dram_bw_gb_s * 1e9) / self.eff_freq_hz
        cycles = total_bytes / dram_bw_bytes_per_cycle
        latency_ms  = (cycles / self.eff_freq_hz) * 1e3
        energy_pj   = total_bytes * cfg.sram_rd_energy_pj
        utilisation = 0.05  # minimal compute, mostly data movement
        throughput_gops = (layer.N * layer.C * layer.P * layer.Q) / (latency_ms * 1e-3) / 1e9
        return SimResult(
            layer_name=layer.name,
            latency_cycles=cycles,
            latency_ms=latency_ms,
            energy_pj=energy_pj,
            energy_uj=energy_pj * 1e-6,
            throughput_gops=throughput_gops,
            utilisation=utilisation,
            memory_bound=True,
            dram_accesses_bytes=total_bytes,
            compute_intensity=0.5,
            per_pe_stress=self._compute_pe_stress(0, utilisation),
        )

    # ------------------------------------------------------------------
    # Helper: distribute stress across PE array
    # ------------------------------------------------------------------

    def _compute_pe_stress(self, total_macs: float, utilisation: float) -> np.ndarray:
        """
        Approximate per-PE stress as a spatial distribution across the array.
        PEs near the centre of a systolic array typically see higher utilisation.
        """
        cfg = self.cfg
        rows, cols = cfg.pe_array_rows, cfg.pe_array_cols
        num_pes = rows * cols

        # Gaussian spatial stress pattern — centre PEs are busier
        cx, cy = rows / 2.0, cols / 2.0
        stress = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                dist = math.sqrt((r - cx)**2 + (c - cy)**2)
                stress[r, c] = math.exp(-0.5 * (dist / (rows / 3.0))**2)

        # Normalise so mean equals utilisation
        mean_s = stress.mean()
        if mean_s > 0:
            stress = stress / mean_s * utilisation
        return stress.flatten()[:num_pes]

    def _layer_spec_from_dict(self, layer_cfg: Dict[str, Any], index: int) -> LayerSpec:
        layer_type = str(layer_cfg.get("type", "conv2d")).lower()
        name = str(layer_cfg.get("name", f"layer_{index}"))

        if layer_type in {"fc", "matmul", "linear"}:
            if layer_type == "matmul":
                batch_or_rows = int(layer_cfg.get("M", 1))
                inner_dim = int(layer_cfg.get("K", 64))
                out_dim = int(layer_cfg.get("N", 64))
                return LayerSpec(name=name, layer_type="fc", N=batch_or_rows, C=inner_dim, K=out_dim, R=1, S=1, P=1, Q=1)
            return LayerSpec(
                name=name,
                layer_type="fc",
                N=int(layer_cfg.get("N", 1)),
                C=int(layer_cfg.get("C", 64)),
                K=int(layer_cfg.get("K", 64)),
                R=1,
                S=1,
                P=1,
                Q=1,
            )

        if layer_type in {"pool", "bn"}:
            return LayerSpec(
                name=name,
                layer_type=layer_type,
                N=int(layer_cfg.get("N", 1)),
                C=int(layer_cfg.get("C", layer_cfg.get("K", 64))),
                K=int(layer_cfg.get("K", layer_cfg.get("C", 64))),
                P=int(layer_cfg.get("P", 1)),
                Q=int(layer_cfg.get("Q", 1)),
                stride=int(layer_cfg.get("stride", 1)),
            )

        return LayerSpec(
            name=name,
            layer_type="conv",
            N=int(layer_cfg.get("N", 1)),
            C=int(layer_cfg.get("C", 3)),
            K=int(layer_cfg.get("K", 64)),
            R=int(layer_cfg.get("R", 3)),
            S=int(layer_cfg.get("S", 3)),
            P=int(layer_cfg.get("P", 32)),
            Q=int(layer_cfg.get("Q", 32)),
            stride=int(layer_cfg.get("stride", 1)),
        )

    def _normalize_mapping(self, mapping: Any, num_layers: int) -> np.ndarray:
        if mapping is None:
            return np.arange(num_layers, dtype=np.int32) % max(self.cfg.mac_clusters, 1)

        mapping_arr = np.asarray(mapping, dtype=np.int32).reshape(-1)
        if mapping_arr.size == 0:
            return np.zeros(num_layers, dtype=np.int32)
        if mapping_arr.size < num_layers:
            pad_value = int(mapping_arr[-1])
            mapping_arr = np.pad(mapping_arr, (0, num_layers - mapping_arr.size), constant_values=pad_value)
        return np.mod(mapping_arr[:num_layers], max(self.cfg.mac_clusters, 1))

    def _attach_activity_traces(self, result: SimResult, mapping: Any = None) -> SimResult:
        active_clusters = np.unique(self._normalize_mapping(mapping, 1 if mapping is None else max(len(np.asarray(mapping).reshape(-1)), 1)))
        if active_clusters.size == 0:
            active_clusters = np.array([0], dtype=np.int32)

        mac_util = np.zeros(self.cfg.mac_clusters, dtype=np.float32)
        sram_access = np.zeros(self.cfg.sram_banks, dtype=np.float32)
        noc_traffic = np.zeros(self.cfg.noc_routers, dtype=np.float32)

        util_per_cluster = float(result.utilisation) / float(active_clusters.size)
        mem_pressure = float(min(1.0, result.dram_accesses_bytes / 1e8))
        noc_pressure = float(min(1.0, (result.dram_accesses_bytes / max(result.latency_cycles, 1.0)) / 1024.0))

        for cluster_idx in active_clusters:
            cluster_id = int(cluster_idx) % max(self.cfg.mac_clusters, 1)
            mac_util[cluster_id] = np.clip(mac_util[cluster_id] + util_per_cluster, 0.0, 1.0)

            primary_bank = cluster_id % max(self.cfg.sram_banks, 1)
            secondary_bank = (cluster_id + 1) % max(self.cfg.sram_banks, 1)
            sram_access[primary_bank] = np.clip(sram_access[primary_bank] + 0.6 * mem_pressure / active_clusters.size, 0.0, 1.0)
            sram_access[secondary_bank] = np.clip(sram_access[secondary_bank] + 0.4 * mem_pressure / active_clusters.size, 0.0, 1.0)

            router_id = primary_bank % max(self.cfg.noc_routers, 1)
            noc_traffic[router_id] = np.clip(noc_traffic[router_id] + noc_pressure / active_clusters.size, 0.0, 1.0)

        switching = np.concatenate([mac_util, sram_access, noc_traffic]).astype(np.float32)

        return SimResult(
            layer_name=result.layer_name,
            latency_cycles=result.latency_cycles,
            latency_ms=result.latency_ms,
            energy_pj=result.energy_pj,
            energy_uj=result.energy_uj,
            throughput_gops=result.throughput_gops,
            utilisation=result.utilisation,
            memory_bound=result.memory_bound,
            dram_accesses_bytes=result.dram_accesses_bytes,
            compute_intensity=result.compute_intensity,
            per_pe_stress=result.per_pe_stress.astype(np.float32),
            mac_utilization=mac_util,
            sram_access_rate=sram_access,
            noc_traffic=noc_traffic,
            switching_activity=switching,
        )

    def _aggregate_mapping_activity(self, layer_results: List[SimResult], mapping_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        latency_weights = np.asarray([max(r.latency_cycles, 1.0) for r in layer_results], dtype=np.float64)
        latency_weights /= max(float(np.sum(latency_weights)), 1.0)
        byte_weights = np.asarray([max(r.dram_accesses_bytes, 1.0) for r in layer_results], dtype=np.float64)
        byte_weights /= max(float(np.sum(byte_weights)), 1.0)

        mac_util = np.average(
            np.stack([r.mac_utilization for r in layer_results], axis=0),
            axis=0,
            weights=latency_weights,
        )
        sram_access = np.average(
            np.stack([r.sram_access_rate for r in layer_results], axis=0),
            axis=0,
            weights=byte_weights,
        )
        noc_traffic = np.average(
            np.stack([r.noc_traffic for r in layer_results], axis=0),
            axis=0,
            weights=byte_weights,
        )

        router_bytes = np.zeros(self.cfg.noc_routers, dtype=np.float64)
        for transfer in self._collect_intercluster_transfers(layer_results, mapping_arr):
            src_router = int(transfer["src"]) % max(self.cfg.noc_routers, 1)
            dst_router = int(transfer["dst"]) % max(self.cfg.noc_routers, 1)
            traffic = transfer["bytes"] * max(transfer["hops"], 1)
            if src_router == dst_router:
                router_bytes[src_router] += traffic
            else:
                router_bytes[src_router] += 0.5 * traffic
                router_bytes[dst_router] += 0.5 * traffic
        if np.max(router_bytes) > 0:
            noc_traffic = 0.35 * noc_traffic + 0.65 * (router_bytes / np.max(router_bytes))

        return (
            np.clip(mac_util, 0.0, 1.0).astype(np.float32),
            np.clip(sram_access, 0.0, 1.0).astype(np.float32),
            np.clip(noc_traffic, 0.0, 1.0).astype(np.float32),
        )

    def _collect_intercluster_transfers(self, layer_results: List[SimResult], mapping_arr: np.ndarray) -> list[dict[str, float]]:
        transfers: list[dict[str, float]] = []
        for idx in range(len(mapping_arr) - 1):
            src = int(mapping_arr[idx])
            dst = int(mapping_arr[idx + 1])
            if src == dst:
                continue

            producer_bytes = float(layer_results[idx].dram_accesses_bytes)
            consumer_bytes = float(layer_results[idx + 1].dram_accesses_bytes)
            activation_bytes = max(min(producer_bytes, consumer_bytes) * 0.22, 1.0)
            hops = self._cluster_distance(src, dst)
            transfers.append(
                {
                    "src": float(src),
                    "dst": float(dst),
                    "bytes": activation_bytes,
                    "hops": float(hops),
                }
            )
        return transfers

    def _cluster_distance(self, src_cluster: int, dst_cluster: int) -> int:
        if src_cluster == dst_cluster:
            return 0

        rows = max(int(round(math.sqrt(max(self.cfg.mac_clusters, 1)))), 1)
        cols = int(math.ceil(max(self.cfg.mac_clusters, 1) / rows))
        src_r, src_c = divmod(int(src_cluster), cols)
        dst_r, dst_c = divmod(int(dst_cluster), cols)
        return max(abs(src_r - dst_r) + abs(src_c - dst_c), 1)

    def _empty_workload_result(self) -> SimResult:
        return SimResult(
            layer_name="workload",
            latency_cycles=0.0,
            latency_ms=0.0,
            energy_pj=0.0,
            energy_uj=0.0,
            throughput_gops=0.0,
            utilisation=0.0,
            memory_bound=False,
            dram_accesses_bytes=0.0,
            compute_intensity=0.0,
            per_pe_stress=np.zeros(self.cfg.num_pes, dtype=np.float32),
            mac_utilization=np.zeros(self.cfg.mac_clusters, dtype=np.float32),
            sram_access_rate=np.zeros(self.cfg.sram_banks, dtype=np.float32),
            noc_traffic=np.zeros(self.cfg.noc_routers, dtype=np.float32),
            switching_activity=np.zeros(self.cfg.mac_clusters + self.cfg.sram_banks + self.cfg.noc_routers, dtype=np.float32),
        )


# ------------------------------------------------------------------
# Convenience builder — used by run_full_pipeline.py
# ------------------------------------------------------------------

def build_simulator_from_config(cfg) -> AnalyticalSimulator:
    """Build an AnalyticalSimulator from a Hydra OmegaConf config."""
    return AnalyticalSimulator(normalize_accelerator_config(_cfg_get(cfg, "accelerator", cfg)))


def get_default_workload() -> List[LayerSpec]:
    """ResNet-18-style workload for quick testing."""
    return [
        LayerSpec('conv1',  'conv', N=1, C=3,   K=64,  R=7, S=7, P=112, Q=112, stride=2),
        LayerSpec('layer1a','conv', N=1, C=64,  K=64,  R=3, S=3, P=56,  Q=56,  stride=1),
        LayerSpec('layer1b','conv', N=1, C=64,  K=64,  R=3, S=3, P=56,  Q=56,  stride=1),
        LayerSpec('layer2a','conv', N=1, C=64,  K=128, R=3, S=3, P=28,  Q=28,  stride=2),
        LayerSpec('layer2b','conv', N=1, C=128, K=128, R=3, S=3, P=28,  Q=28,  stride=1),
        LayerSpec('layer3a','conv', N=1, C=128, K=256, R=3, S=3, P=14,  Q=14,  stride=2),
        LayerSpec('layer3b','conv', N=1, C=256, K=256, R=3, S=3, P=14,  Q=14,  stride=1),
        LayerSpec('layer4a','conv', N=1, C=256, K=512, R=3, S=3, P=7,   Q=7,   stride=2),
        LayerSpec('layer4b','conv', N=1, C=512, K=512, R=3, S=3, P=7,   Q=7,   stride=1),
        LayerSpec('avgpool', 'pool', N=1, C=512, K=512, P=1, Q=1),
        LayerSpec('fc',     'fc',  N=1, C=512, K=1000, R=1, S=1, P=1, Q=1),
    ]
TimeloopRunner = AnalyticalSimulator
WorkloadResult = SimResult
LayerResult = SimResult
