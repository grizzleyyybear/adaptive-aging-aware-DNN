from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_workloads_from_config(cfg: Any) -> dict[str, list[dict[str, Any]]]:
    """Load normalized workload layer lists from config entries or trace files."""
    config = _to_container(cfg)
    workloads: dict[str, list[dict[str, Any]]] = {}

    for path in _extract_paths(config):
        workloads.update(load_workload_file(path))

    for entry in _extract_inline_entries(config):
        name = str(entry.get("name", "")).strip()
        layers = entry.get("layers")
        if name and isinstance(layers, list):
            workloads[name] = [_normalize_layer(layer, idx) for idx, layer in enumerate(layers)]
            for alias in entry.get("aliases", []) or []:
                workloads[str(alias)] = workloads[name]

    return workloads


def load_workload_file(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    payload = _load_structured_file(_resolve_path(path))
    entries = payload.get("workloads", payload if isinstance(payload, list) else [])
    if not isinstance(entries, list):
        raise ValueError(f"Workload file must contain a 'workloads' list: {path}")

    workloads: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Workload entries must be mappings in {path}")
        name = str(entry.get("name", "")).strip()
        layers = entry.get("layers", [])
        if not name or not isinstance(layers, list) or not layers:
            raise ValueError(f"Workload entry needs a name and non-empty layers list in {path}")
        normalized_layers = [_normalize_layer(layer, idx) for idx, layer in enumerate(layers)]
        workloads[name] = normalized_layers
        for alias in entry.get("aliases", []) or []:
            workloads[str(alias)] = normalized_layers
    return workloads


def _normalize_layer(layer: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(layer, dict):
        raise ValueError("Layer entries must be mappings")

    layer_type = str(layer.get("type", layer.get("op", "conv2d"))).lower()
    normalized = {"name": str(layer.get("name", f"layer_{index}")), "type": layer_type}

    if layer_type in {"matmul", "linear", "fc"}:
        normalized["type"] = "matmul" if layer_type == "matmul" else "fc"
        normalized["M"] = int(layer.get("M", layer.get("rows", layer.get("N", 1))))
        normalized["K"] = int(layer.get("K", layer.get("C", 64)))
        normalized["N"] = int(layer.get("N", layer.get("cols", layer.get("out_features", 64))))
        return normalized

    for key, default in {
        "N": 1,
        "C": 3,
        "K": 64,
        "R": 3,
        "S": 3,
        "P": 32,
        "Q": 32,
        "stride": 1,
    }.items():
        normalized[key] = int(layer.get(key, default))
    normalized["type"] = "conv2d" if layer_type in {"conv", "convolution"} else layer_type
    return normalized


def _extract_paths(config: Any) -> list[str]:
    if config is None:
        return []
    if isinstance(config, dict):
        raw_paths = (
            config.get("trace_files")
            or config.get("workload_trace_files")
            or config.get("paths")
            or config.get("sources")
            or []
        )
        if isinstance(raw_paths, (str, Path)):
            return [str(raw_paths)]
        if isinstance(raw_paths, list):
            return [str(path) for path in raw_paths]
        return []
    if isinstance(config, list):
        return [str(item["path"]) for item in config if isinstance(item, dict) and "path" in item]
    return []


def _extract_inline_entries(config: Any) -> list[dict[str, Any]]:
    if config is None:
        return []
    if isinstance(config, list):
        return [entry for entry in config if isinstance(entry, dict) and "layers" in entry]
    if isinstance(config, dict):
        entries = config.get("workloads") or config.get("items") or []
        return [entry for entry in entries if isinstance(entry, dict) and "layers" in entry]
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
    raise FileNotFoundError(f"Workload trace file not found: {path}")


def _to_container(cfg: Any) -> Any:
    if cfg is None:
        return None
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg
