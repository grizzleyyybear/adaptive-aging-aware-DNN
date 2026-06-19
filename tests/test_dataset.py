import pytest
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf

from graph.graph_dataset import AgingDataset
from torch_geometric.data import Data

def test_aging_dataset():
    # Setup dummy directory
    root = Path("./temp_test_data")
    
    cfg = OmegaConf.create({
        'training': {
            'seq_len': 5
        }
    })
    
    # 1. Init empty
    dataset = AgingDataset(root=str(root), split="test", size=10, cfg=cfg)
    assert len(dataset) == 0
    
    # 2. Add dummy samples
    for i in range(3):
        # 10 nodes, 5 features, sequence targets
        d = Data(
            x=torch.rand(10, 5),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            y=torch.rand(10, 1),
            y_trajectory=torch.rand(10, 5)
        )
        dataset.add_sample(d)
        
    # 3. Save
    dataset.finalize_and_save()
    
    # 4. Load
    dataset2 = AgingDataset(root=str(root), split="test", size=10, cfg=cfg)
    assert len(dataset2) == 3
    
    # Access
    sample = dataset2[0]
    assert sample.x.shape == (10, 5)
    
    # Cleanup
    import shutil
    shutil.rmtree(root, ignore_errors=True)


def test_imported_trace_dataset_smoke(tmp_path):
    cfg = OmegaConf.create({
        "seed": 123,
        "accelerator": {
            "pe_array": [4, 4],
            "pe_array_rows": 4,
            "pe_array_cols": 4,
            "mac_clusters": 16,
            "sram_banks": 8,
            "noc_routers": 4,
            "num_layers": 4,
        },
        "model": {"prediction_horizon": 3},
        "workloads": {"trace_files": ["data/workloads/public_tiny.yaml"]},
        "activity_traces": {"trace_files": ["data/activity_traces/public_tiny.json"]},
        "dataset": {"source": "imported_trace", "version": "test-imported"},
        "aging": {
            "technology_node": "14nm_finfet",
            "label_model_version": "aging-v2",
            "stochastic_variation": True,
            "variation_sigma": 0.03,
            "variation_seed": 123,
            "recovery": {"enabled": True, "coefficient": 0.1},
        },
        "planning": {"nbti": 0.4, "hci": 0.35, "tddb": 0.25},
    })

    dataset = AgingDataset(root=str(tmp_path), split="train", size=4, cfg=cfg, seed=123)
    sample = dataset[0]

    assert len(dataset) == 4
    assert sample.x.shape == (28, 8)
    assert sample.y_trajectory.shape == (28, 3)
    assert int(sample.dataset_source_id.item()) == 1
    assert int(sample.technology_node_nm.item()) == 14
    assert int(sample.trace_id.item()) >= 0
