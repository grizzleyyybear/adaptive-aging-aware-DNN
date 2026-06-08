import networkx as nx
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from torch_geometric.data import Data
import matplotlib.pyplot as plt

class AcceleratorGraph:
    """
    Builds a NetworkX representation of the hardware topology.
    Converts directly to PyTorch Geometric objects.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pe_dims = config.get('pe_array', [64, 64])
        self.total_pes = int(self.pe_dims[0] * self.pe_dims[1])
        
        self.mac_clusters = config.get("mac_clusters", config.get("num_mac_clusters", 64))
        self.sram_banks = config.get("sram_banks", config.get("num_sram_banks", 16))
        self.noc_routers = config.get("noc_routers", config.get("num_noc_routers", 8))
        
        self.graph = nx.DiGraph()
        self.node_info = {}
        
    def build(self) -> nx.DiGraph:
        """
        Constructs the heterogeneous cluster mapping.
        """
        self.graph.clear()
        self.node_info.clear()
        node_idx = 0
        
        # Add MAC clusters
        for i in range(self.mac_clusters):
            self.graph.add_node(node_idx, type="mac", local_idx=i, 
                                aging_score=0.0, capacity=self.total_pes // self.mac_clusters)
            self.node_info[node_idx] = {"type": "mac", "local_idx": i}
            node_idx += 1
            
        # Add SRAM banks
        sram_start = node_idx
        for i in range(self.sram_banks):
            self.graph.add_node(node_idx, type="sram", local_idx=i,
                                aging_score=0.0, capacity=1024.0) # dummy capacity
            self.node_info[node_idx] = {"type": "sram", "local_idx": i}
            node_idx += 1
            
        # Add NoC routers
        router_start = node_idx
        for i in range(self.noc_routers):
            self.graph.add_node(node_idx, type="router", local_idx=i,
                                aging_score=0.0, capacity=256.0)
            self.node_info[node_idx] = {"type": "router", "local_idx": i}
            node_idx += 1
            
        # Edges
        # MAC -> SRAM
        for m in range(self.mac_clusters):
            target_banks = [m % self.sram_banks, (m+1) % self.sram_banks]
            for b in target_banks:
                self.graph.add_edge(m, sram_start + b, weight=1.0, type='compute_mem', latency=1)
                self.graph.add_edge(sram_start + b, m, weight=1.0, type='mem_compute', latency=1)
                
        # SRAM -> Router
        for b in range(self.sram_banks):
            target_router = b % self.noc_routers
            self.graph.add_edge(sram_start + b, router_start + target_router, weight=2.0, type='mem_net', latency=2)
            self.graph.add_edge(router_start + target_router, sram_start + b, weight=2.0, type='net_mem', latency=2)
            
        # Router -> Router Mesh
        for r1 in range(self.noc_routers):
            for r2 in range(r1 + 1, self.noc_routers):
                self.graph.add_edge(router_start + r1, router_start + r2, weight=4.0, type='mesh', latency=3)
                self.graph.add_edge(router_start + r2, router_start + r1, weight=4.0, type='mesh', latency=3)
                
        return self.graph

    def to_pyg(self, node_features: np.ndarray) -> Data:
        """
        Converts internal NetworkX state + dynamic features into a PyG Data object.
        """
        # Build node index mapping
        nodes = list(self.graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # Build edge_index — shape [2, num_edges], dtype long
        edges = list(self.graph.edges())
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.zeros((0, 2), dtype=torch.float32)
        else:
            src = [node_to_idx[u] for u, v in edges]
            dst = [node_to_idx[v] for u, v in edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)  # [2, E]
            
            # Edge attributes
            edge_attr = []
            for u, v in edges:
                w = self.graph[u][v].get('weight', 1.0)
                l = self.graph[u][v].get('latency', 1.0)
                edge_attr.append([w, float(l)])
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
            
        x = torch.tensor(node_features, dtype=torch.float32)
        
        assert edge_index.shape[0] == 2, f"edge_index must be [2, E], got {edge_index.shape}"
        assert edge_index.dtype == torch.long, f"edge_index must be long, got {edge_index.dtype}"
        assert x.shape[0] == num_nodes, f"x rows must equal num_nodes"
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)

    def update_node_features(self, activity_dict: dict) -> None:
        """Updates internal graph structure with raw simulation values. Optional helper."""
        if not self.graph.nodes:
            self.build()

        aliases = {
            "mac": ("mac_utilization", "switching_activity"),
            "sram": ("sram_access_rate", "switching_activity"),
            "router": ("noc_traffic", "switching_activity"),
        }
        for node_id, info in self.node_info.items():
            node_type = info["type"]
            local_idx = int(info["local_idx"])
            for key in aliases.get(node_type, ()):
                values = activity_dict.get(key)
                if values is None or len(values) == 0:
                    continue
                arr = np.asarray(values, dtype=np.float32).reshape(-1)
                self.graph.nodes[node_id]["activity"] = float(arr[min(local_idx, len(arr) - 1)])
                break

    def get_aging_vector(self) -> np.ndarray:
        """
        Returns length N array of current aging_scores.
        """
        scores = [data.get('aging_score', 0.0) for _, data in self.graph.nodes(data=True)]
        return np.array(scores, dtype=np.float32)
        
    def visualize(self, save_path: Path) -> None:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        
        scores = self.get_aging_vector()
        nx.draw(self.graph, pos, node_color=scores, cmap=plt.cm.YlOrRd, 
                with_labels=False, node_size=100, vmin=0., vmax=1.0)
                
        plt.title('Accelerator Topology Aging')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def get_num_nodes(self) -> int:
        return len(self.node_info)
        
    def get_node_info(self, node_id: int) -> Dict[str, Any]:
        return self.node_info[node_id]
