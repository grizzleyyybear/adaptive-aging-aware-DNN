#!/bin/bash
#SBATCH --job-name=aging-enh
#SBATCH --partition=gpu                 # CHANGE: your A100 partition
#SBATCH --account=YOUR_ACCOUNT          # CHANGE: your PARAM allocation/project account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/enhanced_%j.out
#SBATCH --error=runs/enhanced_%j.err
#
# Enhanced research-grade run on a single A100.
# Large dataset, longer training, larger NSGA-II/PPO budgets, 14nm aging
# preset with stochastic variation and NBTI recovery.
#
# Submit from the repo root:
#   sbatch scripts/param_a100_enhanced.slurm
#
# To use imported public workload + Timeloop/Accelergy traces instead of the
# synthetic generator, set USE_IMPORTED_TRACES=1 and point the *_FILE vars at
# your full trace files (replace the tiny validation examples first).
set -euo pipefail

ENV_DIR="${PARAM_ENV_DIR:-$HOME/envs/aging-aware-dnn}"
PYTHON_MODULE="${PARAM_PYTHON_MODULE:-python/3.10}"
CUDA_MODULE="${PARAM_CUDA_MODULE:-cuda/12.1}"

USE_IMPORTED_TRACES="${USE_IMPORTED_TRACES:-0}"
WORKLOAD_FILE="${WORKLOAD_FILE:-data/workloads/public_tiny.yaml}"
ACTIVITY_FILE="${ACTIVITY_FILE:-data/activity_traces/public_tiny.json}"

mkdir -p runs
module purge 2>/dev/null || true
module load "$PYTHON_MODULE" 2>/dev/null || true
module load "$CUDA_MODULE" 2>/dev/null || true
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
nvidia-smi || true

COMMON_ARGS=(
  --quality --device cuda
  --dataset-size 100000
  --pred-epochs 140 --pred-patience 25
  --traj-epochs 170 --traj-patience 30
  --nsga-pop 40 --nsga-gen 40
  --ppo-iters 80 --ppo-steps 64
  --aging-technology 14nm_finfet
  --aging-variation 0.08
  --aging-recovery 0.20
  --results-path eval_results_a100_enhanced.json
  --ckpt-dir checkpoints_a100_enhanced
)

if [[ "$USE_IMPORTED_TRACES" == "1" ]]; then
  echo "==> Enhanced run with IMPORTED traces"
  python run_eval.py "${COMMON_ARGS[@]}" \
    --dataset-source imported_trace \
    --workload-traces "$WORKLOAD_FILE" \
    --activity-traces "$ACTIVITY_FILE"
else
  echo "==> Enhanced run with SYNTHETIC dataset (14nm preset + variation + recovery)"
  python run_eval.py "${COMMON_ARGS[@]}"
fi

echo "==> Generating figures and tables"
python generate_figures.py

echo "==> Done. Results in eval_results_a100_enhanced.json"
