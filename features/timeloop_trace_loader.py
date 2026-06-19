from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ActivityTrace:
    trace_id: str
    workload_name: str
    latency_cycles: float
    energy_pj: float
    switching_activity: np.ndarray
    mac_utilization: np.ndarray
    sram_access_rate: np.ndarray
    noc_traffic: np.ndarray

    @property
    def total_latency_cycles(self) -> float:
        return self.latency_cycles

    @property
    def total_energy_pj(self) -> float:
        return self.energy_pj

    def activity_dict(self) -> dict[str, np.ndarray]:
        return {
            "switching_activity": self.switching_activity,
            "mac_utilization": self.mac_utilization,
            "sram_access_rate": self.sram_access_rate,
            "noc_traffic": self.noc_traffic,
        }


class TimeloopTraceLoader:
    """Loads compact Timeloop/Accelergy-style activity traces."""

    def __init__(self, cfg: Any = None, accelerator_cfg: Any = None):
        self.cfg = _to_container(cfg)
        self.accelerator_cfg = _to_container(accelerator_cfg) or {}
        self.traces_by_workload: dict[str, list[ActivityTrace]] = {}
        for path in _extract_paths(self.cfg):
            for trace in self._load_file(path):
                self.traces_by_workload.setdefault(trace.workload_name, []).append(trace)

    @property
    def available_workloads(self) -> list[str]:
        return sorted(self.traces_by_workload)

    def has_traces(self) -> bool:
        return any(self.traces_by_workload.values())

    def get_trace(self, workload_name: str, rng: np.random.Generator | None = None) -> ActivityTrace | None:
        traces = self.traces_by_workload.get(workload_name, [])
        if not traces:
            return None
        if rng is None or len(traces) == 1:
            return traces[0]
        return traces[int(rng.integers(0, len(traces)))]

    def _load_file(self, path: str | Path) -> list[ActivityTrace]:
        resolved = _resolve_path(path)
        payload = _load_structured_file(resolved)
        entries = payload.get("traces", payload if isinstance(payload, list) else [])
        if not isinstance(entries, list):
            raise ValueError(f"Activity trace file must contain a 'traces' list: {path}")
        return [self._normalize_trace(entry, idx, resolved) for idx, entry in enumerate(entries)]

    def _normalize_trace(self, entry: dict[str, Any], index: int, path: Path) -> ActivityTrace:
        if not isinstance(entry, dict):
            raise ValueError(f"Activity trace entries must be mappings in {path}")
        workload_name = str(entry.get("workload_name", entry.get("workload", ""))).strip()
        if not workload_name:
            raise ValueError(f"Activity trace missing workload_name in {path}")

        activity = entry.get("activity", entry)
        mac_count = int(_cfg_get(self.accelerator_cfg, "mac_clusters", _cfg_get(self.accelerator_cfg, "num_mac_clusters", 16)))
        sram_count = int(_cfg_get(self.accelerator_cfg, "sram_banks", _cfg_get(self.accelerator_cfg, "num_sram_banks", 8)))
        noc_count = int(_cfg_get(self.accelerator_cfg, "noc_routers", _cfg_get(self.accelerator_cfg, "num_noc_routers", 4)))

        mac = _bounded_array(activity.get("mac_utilization", activity.get("mac_util", [])), mac_count)
        sram = _bounded_array(activity.get("sram_access_rate", activity.get("sram_access", [])), sram_count)
        noc = _bounded_array(activity.get("noc_traffic", []), noc_count)
        switching_raw = activity.get("switching_activity")
        if switching_raw is None:
            switching = np.concatenate([mac, sram, noc]).astype(np.float32)
        else:
            switching = _bounded_array(switching_raw, mac_count + sram_count + noc_count)

        return ActivityTrace(
            trace_id=str(entry.get("trace_id", f"{path.stem}_{index}")),
            workload_name=workload_name,
            latency_cycles=float(entry.get("latency_cycles", entry.get("total_latency_cycles", 1.0))),
            energy_pj=float(entry.get("energy_pj", entry.get("total_energy_pj", 1.0))),
            switching_activity=switching,
            mac_utilization=mac,
            sram_access_rate=sram,
            noc_traffic=noc,
        )


def _bounded_array(values: Any, length: int) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=np.float32).reshape(-1)
    if arr.size == 0:
        arr = np.zeros(length, dtype=np.float32)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size), mode="edge" if arr.size else "constant")
    return np.clip(arr[:length], 0.0, 1.0).astype(np.float32)


def _extract_paths(config: Any) -> list[str]:
    if config is None:
        return []
    if isinstance(config, dict):
        raw_paths = config.get("trace_files") or config.get("activity_trace_files") or config.get("paths") or []
        if isinstance(raw_paths, (str, Path)):
            return [str(raw_paths)]
        if isinstance(raw_paths, list):
            return [str(path) for path in raw_paths]
    if isinstance(config, list):
        return [str(item) for item in config]
    return []


def _load_structured_file(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (Path.cwd(), REPO_ROOT):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Activity trace file not found: {path}")


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _to_container(cfg: Any) -> Any:
    if cfg is None:
        return None
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg
