#!/bin/bash
#SBATCH --job-name=aging-full
#SBATCH --partition=dgxnp              # PARAM Siddhi-AI A100 partition (advisory A1)
#SBATCH -N 1
#SBATCH --ntasks-per-node=16          # advisory A6: 16 CPUs per A100
#SBATCH --gres=gpu:A100-SXM4:1        # Siddhi GRES name; generic gpu:1 also works
#SBATCH --time=04:00:00
#SBATCH --output=runs/full_%j.out
#SBATCH --error=runs/full_%j.err
#
# Standard full benchmark on a single A100 (synthetic dataset, 40k samples).
# Verified for CDAC PARAM Siddhi-AI (DGX-A100, 40GB). Submit from repo root:
#   sbatch scripts/param_a100_full.slurm
# Different site/partition? Override on the CLI (no need to edit this file):
#   sbatch --partition=<PART> --gres=gpu:1 scripts/param_a100_full.slurm
set -euo pipefail

ENV_DIR="${PARAM_ENV_DIR:-$HOME/envs/aging-aware-dnn}"
PYTHON_MODULE="${PARAM_PYTHON_MODULE:-python/3.10}"
CUDA_MODULE="${PARAM_CUDA_MODULE:-cuda/12.1}"

export OMP_NUM_THREADS="${SLURM_NTASKS_PER_NODE:-16}"
mkdir -p runs
module purge 2>/dev/null || true
module load "$PYTHON_MODULE" 2>/dev/null || true
module load "$CUDA_MODULE" 2>/dev/null || true
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
nvidia-smi || true

echo "==> Full benchmark (A100, 40k samples)"
# NOTE: do NOT pass --gpu-4gb on A100; use the full-memory profile.
python run_eval.py --full --device cuda \
  --results-path eval_results_a100_full.json \
  --ckpt-dir checkpoints_a100_full

echo "==> Generating figures and tables"
python generate_figures.py

echo "==> Done. Results in eval_results_a100_full.json"
