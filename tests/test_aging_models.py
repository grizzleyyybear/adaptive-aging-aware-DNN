import pytest
import numpy as np
from omegaconf import OmegaConf

from aging_models.nbti_model import NBTIModel
from aging_models.hci_model import HCIModel
from aging_models.tddb_model import TDDBModel
from aging_models.aging_label_generator import AgingLabelGenerator

def test_nbti_monotonicity():
    model = NBTIModel(A=5e-3, n=0.25)
    stress_times = np.array([100.0, 1000.0, 10000.0])
    activity = np.array([0.5, 0.5, 0.5])
    
    degradation = model.compute_degradation(stress_times, activity)
    assert degradation[0] < degradation[1] < degradation[2], "NBTI degradation must be monotonically increasing with time"
    
def test_hci_monotonicity():
    model = HCIModel(B=1e-4, m=0.5)
    stress_times = np.array([100.0, 1000.0, 10000.0])
    current_density = np.array([0.8, 0.8, 0.8])
    
    degradation = model.compute_degradation(current_density, stress_times)
    assert degradation[0] < degradation[1] < degradation[2], "HCI degradation must increase with time"

def test_tddb_probabilities():
    model = TDDBModel(k=2.5, beta=10.0)
    e_field = np.array([1.0, 5.0, 10.0]) 
    stress_time = np.array([3600.0] * 3) # 1 hour
    
    probs = model.failure_probability(e_field, stress_time)
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities must be in [0, 1]"
    assert probs[0] < probs[1] < probs[2], "Failure probability must increase with electric field"

def test_aging_generator_trajectory():
    nbti = NBTIModel(A=5e-3, n=0.25)
    hci = HCIModel(B=1e-4, m=0.5)
    tddb = TDDBModel(k=2.5, beta=10.0)
    
    gen = AgingLabelGenerator(nbti, hci, tddb, weights={'nbti': 0.4, 'hci': 0.4, 'tddb': 0.2})
    
    # Simulate 5 steps for 2 nodes
    seq = []
    for _ in range(5):
        seq.append({
            'switching_activity': np.array([0.2, 0.8]),
            'mac_utilization': np.array([0.1, 0.9])
        })
        
    traj = gen.generate_trajectory_labels(seq, timestep_s=3600.0)
    assert traj.shape == (5, 2), "Shape must be [T, N]"
    
    # Node 1 (index 1) has higher activity, should age faster
    for t in range(5):
        assert traj[t, 0] < traj[t, 1], "Higher activity node must age faster"
        
    # Aging must monotonically increase for both nodes
    for i in range(4):
        assert traj[i, 0] <= traj[i+1, 0]
        assert traj[i, 1] <= traj[i+1, 1]


def test_aging_generator_technology_preset_and_recovery_metadata():
    cfg = OmegaConf.create({
        "seed": 7,
        "aging": {
            "technology_node": "14nm_finfet",
            "label_model_version": "aging-v2",
            "stochastic_variation": True,
            "variation_sigma": 0.05,
            "variation_seed": 7,
            "recovery": {"enabled": True, "coefficient": 0.2},
        },
        "planning": {"nbti": 0.4, "hci": 0.4, "tddb": 0.2},
    })
    gen = AgingLabelGenerator(cfg=cfg)
    activity = {
        "switching_activity": np.array([0.2, 0.8], dtype=np.float32),
        "mac_utilization": np.array([0.2, 0.8], dtype=np.float32),
    }

    score = gen.compute_aging_score(activity, stress_time_s=3600.0)
    metadata = gen.metadata()

    assert score.shape == (2,)
    assert np.all((score >= 0.0) & (score <= 1.0))
    assert metadata["technology_node"] == "14nm_finfet"
    assert metadata["technology_node_nm"] == 14
    assert metadata["recovery_enabled"] is True
