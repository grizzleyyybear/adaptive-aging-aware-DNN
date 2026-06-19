from omegaconf import OmegaConf

from features.timeloop_trace_loader import TimeloopTraceLoader


def test_timeloop_trace_loader_reads_public_tiny_trace():
    accelerator_cfg = OmegaConf.create({
        "mac_clusters": 16,
        "sram_banks": 8,
        "noc_routers": 4,
    })
    cfg = OmegaConf.create({
        "trace_files": ["data/activity_traces/public_tiny.json"]
    })

    loader = TimeloopTraceLoader(cfg, accelerator_cfg=accelerator_cfg)
    trace = loader.get_trace("MLPerf-ResNet50-Tiny")

    assert trace is not None
    assert trace.mac_utilization.shape == (16,)
    assert trace.sram_access_rate.shape == (8,)
    assert trace.noc_traffic.shape == (4,)
    assert trace.switching_activity.shape == (28,)
    assert trace.total_latency_cycles > 0
    assert trace.total_energy_pj > 0
