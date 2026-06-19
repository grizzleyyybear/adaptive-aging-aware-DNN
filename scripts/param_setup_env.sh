#!/bin/bash
# One-time environment setup for PARAM A100 nodes.
# Run this on a login node before submitting jobs.
#
#   bash scripts/param_setup_env.sh
#
# Override the module name / python if your PARAM site differs:
#   PARAM_PYTHON_MODULE=python/3.10.x PARAM_CUDA_MODULE=cuda/12.1 bash scripts/param_setup_env.sh
set -euo pipefail

ENV_DIR="${PARAM_ENV_DIR:-$HOME/envs/aging-aware-dnn}"
PYTHON_MODULE="${PARAM_PYTHON_MODULE:-python/3.10}"
CUDA_MODULE="${PARAM_CUDA_MODULE:-cuda/12.1}"
TORCH_CUDA_WHEEL="${PARAM_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

echo "==> Loading modules ($PYTHON_MODULE, $CUDA_MODULE)"
module purge 2>/dev/null || true
module load "$PYTHON_MODULE" 2>/dev/null || echo "WARN: could not load $PYTHON_MODULE (adjust PARAM_PYTHON_MODULE)"
module load "$CUDA_MODULE" 2>/dev/null || echo "WARN: could not load $CUDA_MODULE (adjust PARAM_CUDA_MODULE)"

echo "==> Creating virtualenv at $ENV_DIR"
python -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel

echo "==> Installing CUDA PyTorch"
pip install torch --index-url "$TORCH_CUDA_WHEEL"

echo "==> Installing torch-geometric"
pip install torch-geometric

echo "==> Installing remaining requirements"
pip install -r requirements.txt

echo "==> Ensuring pytest is available"
pip install pytest

echo "==> Verifying CUDA visibility"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

echo "==> Running test suite"
python -m pytest -q

echo "==> Done. Activate later with: source $ENV_DIR/bin/activate"
