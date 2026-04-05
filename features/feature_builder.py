import torch
import numpy as np

class FeatureBuilder:
    """
    Builds the [N x 8] node feature matrix for PyTorch Geometric datasets.
    """
    MAX_STRESS_TIME_S = 1_800_000.0

    def __init__(self, acc_cfg):
        """
        Args:
            acc_cfg: Accelerator config dict
        """
        self.acc_cfg = acc_cfg
        self.num_macs = acc_cfg.get('mac_clusters', acc_cfg.get('num_mac_clusters', 64))
        self.num_srams = acc_cfg.get('sram_banks', acc_cfg.get('num_sram_banks', 16))
        self.num_routers = acc_cfg.get('noc_routers', acc_cfg.get('num_noc_routers', 8))
        self.N = self.num_macs + self.num_srams + self.num_routers
        
    def build_node_features(
        self,
        activity_dict: dict,
        workload_name: str,
        latency: float,
        energy: float,
        stress_time_s: float,
    ) -> torch.Tensor:
        """
        Constructs the 8-dimensional feature matrix for all N nodes.
        
        Features:
          0: switching_activity
          1: compute_utilisation
          2: memory_access_rate
          3: duty_cycle
          4: temperature_proxy
          5: node_type_id
          6: workload_type_id
          7: stress_time_normalized
          
        Args:
            activity_dict: keys [switching_activity, mac_utilization, sram_access_rate, noc_traffic]
            workload_name: string (currently unused in node feats since graph_dataset adds workload_emb globally)
            latency: scalar
            energy: scalar
            stress_time_s: scalar cumulative stress time in seconds
            
        Returns:
            torch.Tensor shape [N, 8]
        """
        features = torch.zeros((self.N, 8), dtype=torch.float32)
        idx = 0
        sw_act = activity_dict["switching_activity"]
        mac_util = activity_dict["mac_utilization"]
        sram_util = activity_dict["sram_access_rate"]
        noc_util = activity_dict["noc_traffic"]

        workload_type = 1.0 if "bert" in workload_name.lower() or "vit" in workload_name.lower() else 0.0
        latency_norm = float(min(latency / 1e8, 1.0))
        energy_norm = float(min(energy / 1e9, 1.0))
        stress_time_norm = float(
            np.clip(
                np.log(max(stress_time_s, 1.0)) / np.log(self.MAX_STRESS_TIME_S),
                0.0,
                1.0,
            )
        )
        
        # MACs
        for i in range(self.num_macs):
            if idx < len(sw_act) and i < len(mac_util):
                util = float(mac_util[i])
                activity = float(sw_act[idx])
                features[idx, 0] = activity
                features[idx, 1] = util
                features[idx, 2] = latency_norm
                features[idx, 3] = activity
                features[idx, 4] = min(1.0, 0.25 + 0.6 * util + 0.15 * energy_norm)
                features[idx, 5] = 0.0
                features[idx, 6] = workload_type
                features[idx, 7] = stress_time_norm
            idx += 1
            
        # SRAMs
        for i in range(self.num_srams):
            if idx < len(sw_act) and i < len(sram_util):
                util = float(sram_util[i])
                activity = float(sw_act[idx])
                features[idx, 0] = activity
                features[idx, 1] = 0.0
                features[idx, 2] = util
                features[idx, 3] = activity
                features[idx, 4] = min(1.0, 0.3 + 0.5 * util + 0.2 * energy_norm)
                features[idx, 5] = 1.0
                features[idx, 6] = workload_type
                features[idx, 7] = stress_time_norm
            idx += 1
            
        # Routers
        for i in range(self.num_routers):
            if idx < len(sw_act) and i < len(noc_util):
                util = float(noc_util[i])
                activity = float(sw_act[idx])
                features[idx, 0] = activity
                features[idx, 1] = 0.0
                features[idx, 2] = util
                features[idx, 3] = activity
                features[idx, 4] = min(1.0, 0.25 + 0.5 * util + 0.25 * latency_norm)
                features[idx, 5] = 2.0
                features[idx, 6] = workload_type
                features[idx, 7] = stress_time_norm
            idx += 1
            
        return features
