# Adaptive Aging-Aware DNN Accelerator Management

This repository implements an end-to-end research pipeline for predicting and managing transistor-aging risk in DNN accelerators. It models accelerator components as a heterogeneous graph, predicts per-component aging and future aging trajectories, then uses NSGA-II and PPO to search for workload mappings that reduce stress while tracking latency and energy.

## Problem statement

Long-running DNN inference workloads create uneven switching activity across MAC clusters, SRAM banks, and NoC routers. That uneven activity accelerates NBTI, HCI, and TDDB aging, which can shorten accelerator lifetime or force conservative guardbands. The goal here is to make aging visible at the component level and use that prediction online for mapping and scheduling decisions.

## Architecture

```text
DNN workload layers
      |
      v
Analytical roofline simulator
      |
      v
Activity and feature builder
      |
      v
AcceleratorGraph -> PyTorch Geometric Data
      |
      v
Hybrid GNN-Transformer
      |                         |
      v                         v
Per-node aging score       10-step trajectory predictor
      |                         |
      +-----------+-------------+
                  |
        +---------+---------+
        v                   v
NSGA-II mapper         PPO runtime controller
minimize peak aging,   learn rebalance actions
latency, energy        under reward/budget signals
```

Core implementation:

| Area | Files |
|---|---|
| Dataset and graph | `graph/graph_dataset.py`, `graph/accelerator_graph.py` |
| Simulator/features | `simulator/timeloop_runner.py`, `simulator/workload_runner.py`, `features/feature_builder.py` |
| Prediction models | `models/hybrid_gnn_transformer.py`, `models/trajectory_predictor.py`, `models/training_pipeline.py` |
| Optimization/RL | `optimization/nsga2_optimizer.py`, `rl/environment.py`, `rl/trainer.py` |
| Entry points | `run_eval.py`, `scripts/run_full_pipeline.py`, `generate_figures.py` |

## Current reproducible results

The checked-in `eval_results.json` was regenerated with the RTX-3050-safe quality pipeline using 1,500 synthetic samples. These numbers are stronger than the quick smoke run, but still should be treated as a reproducible benchmark run rather than final publication claims.

| Component | Quality result |
|---|---:|
| Aging predictor R2 | 0.9957 |
| Aging predictor MAE | 0.0055 |
| Trajectory predictor R2 | 0.9474 |
| Trajectory predictor MAE | 0.0342 |
| NSGA-II Pareto solutions | 96 total |
| Best NSGA-II peak-aging reduction | 24.5% |
| PPO reward | 0.6634 -> 0.9299 |
| PPO best reward | 1.1011 |

Generated summary plots are in `figures/` and mirrored to `paper/assets/`. Paper-ready CSV and LaTeX tables are generated in `paper/tables/`.

## Reproduce

```bash
pip install -r requirements.txt

# Fast end-to-end smoke run: dataset, predictor, trajectory model, NSGA-II, PPO.
python run_eval.py --smoke

# Force the RTX 3050 / 4GB-safe CUDA profile.
python run_eval.py --smoke --device cuda --gpu-4gb

# Better RTX 3050 benchmark: 1,000 samples, longer training, larger NSGA-II.
python run_eval.py --quality --device cuda --gpu-4gb

# Promoted run used for the checked-in artifacts.
python run_eval.py --quality --device cuda --gpu-4gb --dataset-size 1500 --pred-epochs 60 --pred-patience 16 --traj-epochs 75 --traj-patience 18 --nsga-pop 20 --nsga-gen 20 --ppo-iters 30 --ppo-steps 32

# Generate figures and paper tables from eval_results.json.
python generate_figures.py

# Run tests.
python -m pytest -q
```

For a larger benchmark run:

```bash
python run_eval.py --full
```

The full mode uses 40,000 samples and is intended for a GPU-capable environment.

## Plain-config full pipeline

`scripts/run_full_pipeline.py` no longer depends on Hydra. It loads the YAML fragments under `configs/` with OmegaConf and accepts dotlist overrides:

```bash
python scripts/run_full_pipeline.py dataset.size=64 training.epochs=5 ppo.total_timesteps=64
```

This keeps the existing YAML config structure while avoiding the Hydra import/runtime issue.

## Repository hygiene

Root-level debug dumps and traces are ignored by `.gitignore`:

```text
eval_trace*.txt
pure_trace*.log
debug.txt
error.log
exception.log
final_error.txt
pipeline_err.txt
graph_err.txt
trace*.txt
```

No tracked root debug/trace files are present in this checkout.
