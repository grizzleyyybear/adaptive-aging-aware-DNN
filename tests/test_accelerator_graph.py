import pytest
import numpy as np
import torch
from omegaconf import OmegaConf


from graph.accelerator_graph import AcceleratorGraph
from graph.graph_dataset import AgingDataset
from torch_geometric.data import Data

def test_accelerator_graph():
    cfg_data = {
        'pe_array': [16, 16],
        'mac_clusters': 16,
        'sram_banks': 8,
        'noc_routers': 4
    }
    cfg = OmegaConf.create(cfg_data)
    accel = AcceleratorGraph(cfg)
    
    nx_graph = accel.build()
    
    expected_nodes = 16 + 8 + 4
    assert nx_graph.number_of_nodes() == expected_nodes
    assert accel.get_num_nodes() == expected_nodes
    
    # 16 MACs * 2 banks each * 2 directions = 64 edges
    # 8 SRAMs * 1 router * 2 directions = 16 edges
    # 4 routers fully connected mesh = (4*3/2)*2 = 12 edges
    # total edges expected: 64 + 16 + 12 = 92
    assert nx_graph.number_of_edges() == 92
    
    # test pyg conversion
    dummy_features = np.random.rand(expected_nodes, 21)
    data = accel.to_pyg(dummy_features)
    
    assert isinstance(data, Data)
    assert data.x.shape == (expected_nodes, 21)
    assert data.edge_index.shape[1] == 92
    assert data.edge_attr.shape == (92, 2)
    
def test_dataset_append_and_load(tmp_path):
    root_dir = str(tmp_path)
    
    dataset = AgingDataset(root=root_dir, split="train", size=10, config={})
    
    # Simulate adding data dynamically
    for _ in range(5):
        data = Data(x=torch.rand(28, 5), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
        dataset.add_sample(data)
        
    dataset.finalize_and_save()
    
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample.x.shape == (28, 5)
    
    # Load from disk via new instantiation
    ds2 = AgingDataset(root=root_dir, split="train", size=10, config={})
    assert len(ds2) == 5
