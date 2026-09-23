#!/usr/bin/env bash
# Source this file: source swift_env/activate.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/.venv-swift/bin/activate"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/.cache/modelscope}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${MODELSCOPE_CACHE}" "${HF_HOME}"
echo "[env] VIRTUAL_ENV=${VIRTUAL_ENV}"
echo "[env] python=$(command -v python3)"
