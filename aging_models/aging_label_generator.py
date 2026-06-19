import numpy as np
from typing import Any, List

from .nbti_model import NBTIModel
from .hci_model import HCIModel
from .tddb_model import TDDBModel

class AgingLabelGenerator:
    """Combines NBTI + HCI + TDDB into a normalized per-node aging score [0, 1]."""

    TECHNOLOGY_PRESETS = {
        "28nm_bulk": {
            "node_nm": 28,
            "nbti_A": 0.0040,
            "nbti_n": 0.23,
            "hci_B": 0.000080,
            "hci_m": 0.50,
            "tddb_k": 2.10,
            "tddb_beta": 10.50,
            "temperature_K": 358.0,
            "voltage_v": 0.90,
        },
        "14nm_finfet": {
            "node_nm": 14,
            "nbti_A": 0.0055,
            "nbti_n": 0.25,
            "hci_B": 0.000120,
            "hci_m": 0.52,
            "tddb_k": 2.60,
            "tddb_beta": 9.80,
            "temperature_K": 373.0,
            "voltage_v": 0.80,
        },
        "7nm_finfet": {
            "node_nm": 7,
            "nbti_A": 0.0068,
            "nbti_n": 0.27,
            "hci_B": 0.000160,
            "hci_m": 0.55,
            "tddb_k": 3.05,
            "tddb_beta": 9.30,
            "temperature_K": 388.0,
            "voltage_v": 0.72,
        },
    }

    def __init__(self, nbti=None, hci=None, tddb=None, weights=None, cfg=None):
        self.technology_name = "custom"
        self.technology_node_nm = 0
        self.label_model_version = "aging-v1"
        self.recovery_enabled = False
        self.recovery_coefficient = 0.0
        self.stochastic_variation = False
        self.variation_sigma = 0.0
        self.variation_seed = 42
        self.voltage_v = 0.8
        self._variation_cache: dict[int, dict[str, np.ndarray]] = {}

        if cfg is not None:
            acfg = cfg.get('aging', {})
            params = self._resolve_aging_params(acfg)
            self.nbti = NBTIModel(
                A=params.get('nbti_A', 0.005),
                n=params.get('nbti_n', 0.25),
                temperature_K=params.get('temperature_K', 373.0),
            )
            self.hci  = HCIModel(B=params.get('hci_B', 0.0001), m=params.get('hci_m', 0.5))
            self.tddb = TDDBModel(k=params.get('tddb_k', 2.5), beta=params.get('tddb_beta', 10.0))
            self.weights = cfg.get('planning', {})
            self.technology_name = str(params.get('technology_node', acfg.get('technology_node', 'custom')))
            self.technology_node_nm = int(params.get('node_nm', acfg.get('node_nm', 0)))
            self.label_model_version = str(acfg.get('label_model_version', 'aging-v2'))
            self.voltage_v = float(params.get('voltage_v', acfg.get('voltage_v', 0.8)))
            recovery_cfg = acfg.get('recovery', {})
            self.recovery_enabled = bool(recovery_cfg.get('enabled', acfg.get('recovery_enabled', False)))
            self.recovery_coefficient = float(recovery_cfg.get('coefficient', acfg.get('recovery_coefficient', 0.0)))
            self.stochastic_variation = bool(acfg.get('stochastic_variation', False))
            self.variation_sigma = float(acfg.get('variation_sigma', 0.0))
            self.variation_seed = int(acfg.get('variation_seed', cfg.get('seed', 42)))
        else:
            self.nbti = nbti
            self.hci  = hci
            self.tddb = tddb
            self.weights = weights

    def compute_aging_score(self, activity_metrics: dict, stress_time_s: float) -> np.ndarray:
        N = len(activity_metrics['switching_activity'])
        time_arr = np.full(N, stress_time_s)
        sw_act = activity_metrics['switching_activity']

        if all(k in activity_metrics for k in ('mac_utilization', 'sram_access_rate', 'noc_traffic')):
            util = np.concatenate([
                activity_metrics['mac_utilization'],
                activity_metrics['sram_access_rate'],
                activity_metrics['noc_traffic'],
            ])
        else:
            util = activity_metrics.get('mac_utilization', sw_act)
            if len(util) != len(sw_act):
                util = sw_act

        voltage = activity_metrics.get('voltage', np.ones(N) * self.voltage_v)
        current_density = sw_act * util
        e_field = sw_act * voltage

        if self.recovery_enabled and self.recovery_coefficient > 0:
            recovery = self.recovery_coefficient * np.clip(1.0 - sw_act, 0.0, 1.0)
            nbti_time = time_arr * np.clip(1.0 - recovery, 0.05, 1.0)
        else:
            nbti_time = time_arr

        scales = self._variation_scales(N)
        nbti_norm = np.clip((self.nbti.compute_degradation(nbti_time, sw_act) * scales['nbti']) / 0.2, 0, 1)
        hci_norm  = np.clip((self.hci.compute_degradation(current_density, time_arr) * scales['hci']) / 0.1, 0, 1)
        tddb_norm = np.clip(self.tddb.failure_probability(e_field, time_arr) * scales['tddb'], 0, 1)

        score = (
            self.weights.get('nbti', 0.4) * nbti_norm +
            self.weights.get('hci',  0.4) * hci_norm  +
            self.weights.get('tddb', 0.2) * tddb_norm
        )
        return np.clip(score, 0.0, 1.0)

    def generate_trajectory_labels(self, activity_sequence: List[dict], timestep_s: float) -> np.ndarray:
        T = len(activity_sequence)
        N = len(activity_sequence[0]['switching_activity'])
        trajectories = np.zeros((T, N))
        cumulative_time = 0.0
        for t in range(T):
            cumulative_time += timestep_s
            trajectories[t] = self.compute_aging_score(activity_sequence[t], cumulative_time)
        return trajectories

    def metadata(self) -> dict[str, Any]:
        return {
            "technology_node": self.technology_name,
            "technology_node_nm": self.technology_node_nm,
            "label_model_version": self.label_model_version,
            "stochastic_variation": self.stochastic_variation,
            "variation_sigma": self.variation_sigma,
            "recovery_enabled": self.recovery_enabled,
            "recovery_coefficient": self.recovery_coefficient,
        }

    def _resolve_aging_params(self, acfg: Any) -> dict:
        presets = dict(self.TECHNOLOGY_PRESETS)
        try:
            presets.update(acfg.get('technology_presets', {}) or {})
        except AttributeError:
            pass
        technology_name = str(acfg.get('technology_node', acfg.get('preset', 'custom')))
        selected = dict(presets.get(technology_name, {}))
        selected['technology_node'] = technology_name
        override_source = acfg if technology_name == 'custom' else acfg.get('parameter_overrides', {})
        for key in ('node_nm', 'nbti_A', 'nbti_n', 'hci_B', 'hci_m', 'tddb_k', 'tddb_beta', 'temperature_K', 'voltage_v'):
            if key in override_source:
                selected[key] = override_source.get(key)
        return selected

    def _variation_scales(self, n_nodes: int) -> dict[str, np.ndarray]:
        if not self.stochastic_variation or self.variation_sigma <= 0:
            ones = np.ones(n_nodes, dtype=np.float32)
            return {'nbti': ones, 'hci': ones, 'tddb': ones}
        if n_nodes not in self._variation_cache:
            rng = np.random.default_rng(self.variation_seed + n_nodes)
            self._variation_cache[n_nodes] = {
                key: np.clip(rng.lognormal(mean=0.0, sigma=self.variation_sigma, size=n_nodes), 0.6, 1.6).astype(np.float32)
                for key in ('nbti', 'hci', 'tddb')
            }
        return self._variation_cache[n_nodes]

