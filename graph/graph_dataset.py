# graph/graph_dataset.py — COMPLETE IMPLEMENTATION
"""
PyTorch Geometric InMemoryDataset for aging graph samples.
Compatible with torch_geometric==2.7.0
"""

from __future__ import annotations
import logging
import random
import hashlib
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger(__name__)


class AgingDataset(InMemoryDataset):
    """
    Dataset of accelerator hardware graphs labeled with aging scores.

    Each Data object contains:
      x              : node features [N, 8]   float32
      edge_index     : graph edges   [2, E]   long
      edge_attr      : edge features [E, 2]   float32
      y              : aging score   [N, 1]   float32
      y_trajectory   : future aging  [N, k]   float32
      workload_emb   : one-hot       [W]      float32
      latency        : scalar        [1]      float32
      energy         : scalar        [1]      float32
    """

    WORKLOAD_LIST = [
        "ResNet-50", "MobileNetV2", "EfficientNet-B4", "BERT-Base", "ViT-B/16"
    ]
    FEATURE_DIM = 8
    HORIZON = 10
    SECONDS_PER_STEP = 3600.0
    DEFAULT_CFG = {
        "accelerator": {
            "pe_array": [4, 4],
            "pe_array_rows": 4,
            "pe_array_cols": 4,
            "mac_clusters": 16,
            "sram_banks": 8,
            "noc_routers": 4,
            "num_layers": 10,
        },
        "workloads": [],
        "activity_traces": {
            "trace_files": [],
        },
        "dataset": {
            "source": "synthetic",
            "version": "synthetic-v1",
        },
        "aging": {
            "technology_node": "custom",
            "node_nm": 0,
            "label_model_version": "aging-v1",
            "stochastic_variation": False,
            "variation_sigma": 0.0,
            "variation_seed": 42,
            "recovery": {
                "enabled": False,
                "coefficient": 0.0,
            },
            "nbti_A": 0.005,
            "nbti_n": 0.25,
            "hci_B": 0.0001,
            "hci_m": 0.5,
            "tddb_k": 2.5,
            "tddb_beta": 10.0,
        },
        "planning": {
            "failure_threshold": 0.8,
            "nbti": 0.40,
            "hci": 0.35,
            "tddb": 0.25,
        },
        "model": {
            "prediction_horizon": 10,
        },
    }

    def __init__(
        self,
        root: str,
        split: str,          # 'train' | 'val' | 'test'
        size: int,
        cfg: Optional[DictConfig] = None,
        config: Optional[DictConfig] = None,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        user_cfg = cfg if cfg is not None else config
        user_cfg = user_cfg if user_cfg is not None else {}
        self._auto_generate = self._has_required_sections(user_cfg)
        self.split = split
        self.size = size
        base_cfg = OmegaConf.create(self.DEFAULT_CFG)
        if self._has_key(user_cfg, "workloads"):
            user_workloads = self._get_key(user_cfg, "workloads")
            if isinstance(user_workloads, (list, tuple)) or OmegaConf.is_list(user_workloads):
                base_cfg.workloads = []
            else:
                base_cfg.workloads = {}
        self.cfg = OmegaConf.merge(base_cfg, OmegaConf.create(user_cfg))
        self.seed = seed
        self._acc_cfg = self.cfg.accelerator
        self.horizon = int(self.cfg.model.get("prediction_horizon", self.HORIZON))
        self.dataset_source = str(self.cfg.dataset.get("source", "synthetic"))
        self.dataset_version = str(self.cfg.dataset.get("version", "synthetic-v1"))
        super().__init__(root=root, transform=transform)
        # PyG 2.7.0: load() replaces the old from_data_list pattern
        self.load(self.processed_paths[0])
        self._dynamic_samples: List[Data] = [self.get(i) for i in range(len(self))]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        n_mac = int(self._acc_cfg.get("mac_clusters", self._acc_cfg.get("num_mac_clusters", 64)))
        source = self._safe_cache_token(self.dataset_source)
        version = self._safe_cache_token(self.dataset_version)
        tech = self._safe_cache_token(str(self.cfg.aging.get("technology_node", "custom")))
        return [f"aging_{source}_{version}_{tech}_{self.split}_{self.size}_mac{n_mac}_feat{self.FEATURE_DIM}.pt"]

    def download(self) -> None:
        return None

    def process(self) -> None:
        """
        Generate all graph samples and save as PyG dataset.
        Runs once; cached on disk afterward.
        """
        if not self._auto_generate or self.size <= 0:
            self._save_empty_dataset()
            logger.info("Initialized empty dataset cache at %s", self.processed_paths[0])
            return

        # Lazy imports to avoid circular deps
        from simulator.timeloop_runner import TimeloopRunner
        from simulator.workload_runner import WorkloadRunner
        from features.feature_builder import FeatureBuilder
        from features.timeloop_trace_loader import TimeloopTraceLoader
        from aging_models.aging_label_generator import AgingLabelGenerator
        from graph.accelerator_graph import AcceleratorGraph

        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        simulator = TimeloopRunner(self._acc_cfg)
        workload_runner = WorkloadRunner(self.cfg.workloads)
        trace_loader = TimeloopTraceLoader(self.cfg.activity_traces, accelerator_cfg=self._acc_cfg)
        feature_builder = FeatureBuilder(self._acc_cfg)
        aging_gen = AgingLabelGenerator(cfg=self.cfg)
        acc_graph = AcceleratorGraph(self._acc_cfg)
        acc_graph.build()
        num_nodes = acc_graph.get_num_nodes()
        workload_list = self._select_workload_list(workload_runner, trace_loader)
        source_id = self._source_id()
        label_meta = aging_gen.metadata()

        data_list: List[Data] = []
        logger.info(f"Generating {self.size} samples [{self.split}]...")

        for idx in tqdm(range(self.size), desc=f"AgingDataset[{self.split}]"):
            # --- Select workload ---
            wl_name = workload_list[int(rng.integers(0, len(workload_list)))]
            wl_idx = workload_list.index(wl_name)
            layers = workload_runner.get_workload_layers(wl_name)
            n_layers = len(layers)

            # --- Random mapping ---
            mapping = rng.integers(
                0, int(self._acc_cfg.get("mac_clusters", self._acc_cfg.get("num_mac_clusters", 64))), size=n_layers
            ).astype(np.int32)

            # --- Simulate or load imported activity trace ---
            trace = trace_loader.get_trace(wl_name, rng) if self.dataset_source == "imported_trace" else None
            if trace is None:
                result = simulator.run_workload(layers, mapping)
                activity = {
                    "switching_activity": result.switching_activity,
                    "mac_utilization":    result.mac_utilization,
                    "sram_access_rate":   result.sram_access_rate,
                    "noc_traffic":        result.noc_traffic,
                }
                latency_cycles = result.total_latency_cycles
                energy_pj = result.total_energy_pj
                trace_numeric_id = -1
            else:
                activity = trace.activity_dict()
                latency_cycles = trace.total_latency_cycles
                energy_pj = trace.total_energy_pj
                trace_numeric_id = self._stable_id(trace.trace_id)

            # Vary stress time: 1 hour to 500 hours.
            stress_time = float(rng.uniform(3600, 1_800_000))

            # --- Node features [N, 8] ---
            node_features = feature_builder.build_node_features(
                activity_dict=activity,
                workload_name=wl_name,
                latency=latency_cycles,
                energy=energy_pj,
                stress_time_s=stress_time,
            )

            # --- Current aging score [N] ---
            aging_score = aging_gen.compute_aging_score(
                activity, stress_time
            )

            # --- Future trajectory [N, HORIZON] ---
            future_acts = []
            for h in range(self.horizon):
                noise = rng.normal(0, 0.01, size=activity["switching_activity"].shape)
                future_act = float(h + 1) / self.horizon
                future_acts.append({
                    "switching_activity": np.clip(
                        activity["switching_activity"] * (1.0 + future_act * 0.2) + noise,
                        0.0, 1.0
                    ).astype(np.float32),
                    "mac_utilization":    activity["mac_utilization"],
                    "sram_access_rate":   activity["sram_access_rate"],
                    "noc_traffic":        activity["noc_traffic"],
                })
            trajectory = aging_gen.generate_trajectory_labels(
                future_acts, stress_time
            )  # [HORIZON, N]

            # Transpose: [HORIZON, N] → [N, HORIZON]
            y_trajectory = torch.tensor(
                trajectory.T, dtype=torch.float32
            )  # [N, HORIZON]

            # --- Build PyG graph (edge_index must be [2, E] long) ---
            pyg_data = acc_graph.to_pyg(node_features)
            # Defensive assertion
            assert pyg_data.edge_index.shape[0] == 2, \
                f"edge_index shape error: {pyg_data.edge_index.shape}"
            assert pyg_data.edge_index.dtype == torch.long, \
                f"edge_index dtype error: {pyg_data.edge_index.dtype}"

            # --- Workload one-hot [W] ---
            wl_emb = torch.zeros(len(workload_list), dtype=torch.float32)
            wl_emb[wl_idx] = 1.0

            # --- Normalized scalars ---
            lat_norm = float(min(latency_cycles / 1e8, 1.0))
            eng_norm = float(min(energy_pj / 1e9, 1.0))

            # --- Mapping vector (capped at 64 entries) ---
            max_map_len = 64
            map_arr = mapping[:max_map_len].astype(np.float32)
            num_macs = int(self._acc_cfg.get("mac_clusters", self._acc_cfg.get("num_mac_clusters", 64)))
            map_arr = map_arr / max(num_macs - 1, 1)
            if len(map_arr) < max_map_len:
                map_arr = np.pad(map_arr, (0, max_map_len - len(map_arr)))
            mapping_tensor = torch.tensor(map_arr, dtype=torch.float32)

            data = Data(
                x=pyg_data.x,                          # [N, 8]
                edge_index=pyg_data.edge_index,        # [2, E] long
                edge_attr=pyg_data.edge_attr,          # [E, 2]
                y=torch.tensor(
                    aging_score, dtype=torch.float32
                ).unsqueeze(1),                        # [N, 1]
                y_trajectory=y_trajectory,             # [N, HORIZON]
                workload_emb=wl_emb,                   # [5]
                mapping=mapping_tensor,                # [64]
                stress_time=torch.tensor([stress_time], dtype=torch.float32),
                latency=torch.tensor([lat_norm]),      # [1]
                energy=torch.tensor([eng_norm]),       # [1]
                dataset_source_id=torch.tensor([source_id], dtype=torch.long),
                technology_node_nm=torch.tensor([label_meta["technology_node_nm"]], dtype=torch.long),
                label_model_version_id=torch.tensor([self._stable_id(label_meta["label_model_version"])], dtype=torch.long),
                sample_seed=torch.tensor([self.seed], dtype=torch.long),
                trace_id=torch.tensor([trace_numeric_id], dtype=torch.long),
                num_nodes=num_nodes,                   # explicit — required by PyG 2.7
            )
            data_list.append(data)

        # PyG 2.7.0 save API
        self.save(data_list, self.processed_paths[0])
        logger.info(
            f"Saved {len(data_list)} samples → {self.processed_paths[0]}"
        )

    def add_sample(self, data: Data) -> None:
        self._dynamic_samples.append(data)

    def finalize_and_save(self) -> None:
        if self._dynamic_samples:
            self.save(self._dynamic_samples, self.processed_paths[0])
        else:
            self._save_empty_dataset()
        self.load(self.processed_paths[0])
        self._dynamic_samples = [self.get(i) for i in range(len(self))]

    @staticmethod
    def _has_required_sections(cfg: Any) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return "accelerator" in cfg and "workloads" in cfg
        try:
            return "accelerator" in cfg and "workloads" in cfg
        except TypeError:
            return hasattr(cfg, "accelerator") and hasattr(cfg, "workloads")

    @staticmethod
    def _has_key(cfg: Any, key: str) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, dict):
            return key in cfg
        try:
            return key in cfg
        except TypeError:
            return hasattr(cfg, key)

    @staticmethod
    def _get_key(cfg: Any, key: str) -> Any:
        if isinstance(cfg, dict):
            return cfg.get(key)
        try:
            return cfg.get(key)
        except AttributeError:
            return getattr(cfg, key, None)

    def _save_empty_dataset(self) -> None:
        torch.save((Data().to_dict(), {}, Data), self.processed_paths[0])

    def _select_workload_list(self, workload_runner: Any, trace_loader: Any) -> List[str]:
        if self.dataset_source == "imported_trace" and trace_loader.has_traces():
            traced = [name for name in trace_loader.available_workloads if workload_runner.get_workload_layers(name)]
            if traced:
                return traced
        return list(workload_runner.available_workloads or self.WORKLOAD_LIST)

    def _source_id(self) -> int:
        return 1 if self.dataset_source == "imported_trace" else 0

    @staticmethod
    def _stable_id(value: str) -> int:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16) % 2_147_483_647

    @staticmethod
    def _safe_cache_token(value: str) -> str:
        safe = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
        return safe.strip("-") or "default"

# Alias for backward compatibility
AcceleratorGraphDataset = AgingDataset
