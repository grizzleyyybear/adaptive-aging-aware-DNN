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
| Simulator/features | `simulator/timeloop_runner.py`, `simulator/workload_runner.py`, `simulator/workload_importer.py`, `features/feature_builder.py`, `features/timeloop_trace_loader.py` |
| Prediction models | `models/hybrid_gnn_transformer.py`, `models/trajectory_predictor.py`, `models/training_pipeline.py` |
| Optimization/RL | `optimization/nsga2_optimizer.py`, `rl/environment.py`, `rl/trainer.py` |
| Entry points | `run_eval.py`, `scripts/run_full_pipeline.py`, `generate_figures.py` |

## Current reproducible results

The checked-in `eval_results.json` was regenerated with the RTX-3050-safe quality pipeline using 1,500 synthetic samples. These numbers are a reproducible benchmark run rather than final publication claims, and now include reviewer-facing metrics (rank correlation, per-step trajectory error, Pareto hypervolume, and runtime-policy baselines).

| Component | Quality result |
|---|---:|
| Aging predictor R2 | 0.9962 |
| Aging predictor MAE | 0.0052 |
| Aging predictor Spearman | 0.9809 |
| Trajectory predictor R2 | 0.9453 |
| Trajectory predictor MAE | 0.0355 |
| NSGA-II Pareto solutions | 94 total |
| Best NSGA-II peak-aging reduction | 26.3% |
| PPO reward | 1.0921 -> 0.9823 |
| PPO best reward | 1.4602 |
| PPO best baseline (round-robin) | 0.1528 |

Generated summary plots are in `figures/` and mirrored to `paper/assets/`. Paper-ready CSV and LaTeX tables are generated in `paper/tables/`, including Spearman, NSGA-II hypervolume, and PPO baseline rows.

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

## Better dataset mode

The default path remains synthetic so tests and smoke runs stay fast. For stronger research runs, the pipeline can now load public workload dimensions and Timeloop/Accelergy-style activity traces:

```bash
python run_eval.py --quality --device cuda \
  --dataset-source imported_trace \
  --workload-traces data/workloads/public_tiny.yaml \
  --activity-traces data/activity_traces/public_tiny.json \
  --aging-technology 14nm_finfet \
  --aging-variation 0.08 \
  --aging-recovery 0.20
```

Schemas live in `data/workloads/schema.json` and `data/activity_traces/schema.json`. The checked-in tiny examples validate the importer only; replace them with full MLPerf/MAESTRO/CoSA workload files and Timeloop/Accelergy traces for A100 runs. Aging presets are documented in `configs/aging_technology.yaml`, and generated PyG samples now store source/technology/provenance IDs.

Suggested A100/PARAM candidate after replacing the tiny traces:

```bash
python run_eval.py --quality --device cuda \
  --dataset-source imported_trace \
  --workload-traces data/workloads/<full-public-workloads>.yaml \
  --activity-traces data/activity_traces/<timeloop-accelergy-traces>.json \
  --aging-technology 14nm_finfet \
  --aging-variation 0.08 \
  --aging-recovery 0.20 \
  --dataset-size 100000 \
  --pred-epochs 140 --pred-patience 25 \
  --traj-epochs 170 --traj-patience 30 \
  --nsga-pop 40 --nsga-gen 40 \
  --ppo-iters 80 --ppo-steps 64

python generate_figures.py
```

## Running on PARAM (A100 / SLURM)

SLURM job scripts and helpers are in `scripts/`:

| Script | Purpose |
|---|---|
| `scripts/param_setup_env.sh` | One-time env setup on a login node (CUDA PyTorch, PyG, requirements, tests) |
| `scripts/param_discover.sh` | Print this site's GPU partitions + your account, then the exact `sbatch` command |
| `scripts/param_a100_full.slurm` | Standard full benchmark (40k synthetic samples) on 1x A100 |
| `scripts/param_a100_enhanced.slurm` | Enhanced run (100k samples, larger NSGA/PPO, 14nm aging preset) |
| `scripts/param_collect_artifacts.sh` | Bundle results/figures/tables/checkpoints into a tarball |

Step by step from the repo root on PARAM:

```bash
# 0. Clone and enter the repo
git clone https://github.com/grizzleyyybear/adaptive-aging-aware-DNN.git
cd adaptive-aging-aware-DNN

# 1. One-time environment setup on a LOGIN node.
#    Adjust module names for your site if needed:
#    PARAM_PYTHON_MODULE=python/3.10.x PARAM_CUDA_MODULE=cuda/12.1 bash scripts/param_setup_env.sh
bash scripts/param_setup_env.sh

# 2a. Discover this site's GPU partition + your account (partition names
#     vary per PARAM site; "gpu" may not exist):
bash scripts/param_discover.sh

# 2b. Submit the standard full benchmark, passing partition+account on the CLI
#     (these override the #SBATCH defaults, so you do NOT need to edit the file):
sbatch --partition=<PART> --account=<ACCT> scripts/param_a100_full.slurm

# 2c. Or submit the enhanced research run
sbatch --partition=<PART> --account=<ACCT> scripts/param_a100_enhanced.slurm

# To use imported public workload + Timeloop/Accelergy traces in the enhanced run,
# replace the tiny example files, then submit with:
#   USE_IMPORTED_TRACES=1 \
#   WORKLOAD_FILE=data/workloads/<full>.yaml \
#   ACTIVITY_FILE=data/activity_traces/<traces>.json \
#   sbatch --partition=<PART> --account=<ACCT> scripts/param_a100_enhanced.slurm

# 3. Monitor
squeue -u "$USER"
tail -f runs/full_*.out      # or runs/enhanced_*.out

# 4. After completion, bundle artifacts to copy back
bash scripts/param_collect_artifacts.sh
```

Notes:
- The pipeline is single-GPU; one A100 is enough. Do **not** pass `--gpu-4gb` on A100.
- Full results go to `eval_results_a100_full.json`; enhanced results to `eval_results_a100_enhanced.json` (the local `eval_results.json` is left untouched unless you point `--results-path` at it).
- If you hit a queue time limit, raise `#SBATCH --time` or lower `--dataset-size`.

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
va