import pytest
import numpy as np
from omegaconf import OmegaConf
from simulator.timeloop_runner import TimeloopRunner, LayerResult, WorkloadResult
from simulator.workload_runner import WorkloadRunner

def test_timeloop_analytical_model():
    # Setup dummy cfg
    cfg_data = {
        'pe_array': [16, 16],
        'mac_clusters': 8,
        'sram_banks': 4,
        'noc_routers': 2,
        'mac_energy_pj_per_op': 0.1,
        'sram_read_energy_pj': 2.0,
        'noc_hop_energy_pj': 0.5,
        'ops_per_cycle': 2
    }
    cfg = OmegaConf.create(cfg_data)
    
    runner = TimeloopRunner(cfg)
    
    # Simple conv workload config
    layer_cfg = {'type': 'conv2d', 'K': 16, 'C': 3, 'R': 3, 'S': 3, 'P': 32, 'Q': 32}
    mapping = np.array([0, 1, 2, 3]) # Mapped across 4 clusters
    
    res = runner.run_layer(layer_cfg, mapping)
    
    # Assert return types
    assert isinstance(res, LayerResult)
    assert res.latency_cycles > 0
    assert res.energy_pj > 0.0
    
    # Assert structural integrity shape
    assert res.mac_utilization.shape[0] == 8
    assert res.sram_access_rate.shape[0] == 4
    assert res.noc_traffic.shape[0] == 2
    assert res.switching_activity.shape[0] == 14 # 8+4+2
    
    # Values should be bounded
    assert np.all((res.switching_activity >= 0.0) & (res.switching_activity <= 1.0))
    
def test_workload_generator():
    runner = WorkloadRunner()
    target_names = ['ResNet-50', 'BERT-Base']
    
    stream = runner.generate_stream(pattern='alternating', workload_names=target_names, total_steps=10, seed=42)
    assert len(stream) == 10
    assert stream[0] == 'ResNet-50'
    assert stream[1] == 'BERT-Base'
    
    # Test layer fetching
    layers = runner.get_workload_layers('BERT-Base')
    assert len(layers) > 0
    assert layers[0]['type'] == 'matmul'


def test_workload_runner_loads_public_trace_file():
    cfg = OmegaConf.create({
        "trace_files": ["data/workloads/public_tiny.yaml"]
    })
    runner = WorkloadRunner(cfg)

    assert "MLPerf-ResNet50-Tiny" in runner.available_workloads
    layers = runner.get_workload_layers("ResNet50-MLPerf-Tiny")
    assert len(layers) == 5
    assert layers[0]["type"] == "conv2d"
    assert layers[0]["K"] == 64
