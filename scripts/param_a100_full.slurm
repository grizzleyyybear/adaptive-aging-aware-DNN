#!/bin/bash
#SBATCH --job-name=aging-full
#SBATCH --partition=gpu                 # CHANGE: your A100 partition (e.g. gpu, gpu, standard-gpu)
#SBATCH --account=YOUR_ACCOUNT          # CHANGE: your PARAM allocation/project account
#SBATCH --gres=gpu:1                    # 1x A100 is sufficient (pipeline is single-GPU)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=runs/full_%j.out
#SBATCH --error=runs/full_%j.err
#
# Standard full benchmark on a single A100 (synthetic dataset, 40k samples).
# Submit from the repo root:
#   sbatch scripts/param_a100_full.slurm
set -euo pipefail

ENV_DIR="${PARAM_ENV_DIR:-$HOME/envs/aging-aware-dnn}"
PYTHON_MODULE="${PARAM_PYTHON_MODULE:-python/3.10}"
CUDA_MODULE="${PARAM_CUDA_MODULE:-cuda/12.1}"

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
