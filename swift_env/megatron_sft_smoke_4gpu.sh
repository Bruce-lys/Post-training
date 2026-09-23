#!/usr/bin/env bash
# 4-GPU smoke wrapper for Qwen3.8-Flash-Next.
# Default layout: TP=4 PP=1 EP=4 ETP=1 SP=true (verified on 4×H200, ~85GiB).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MIN_GPUS="${MIN_GPUS:-4}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export EP="${EP:-4}"
export SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
export MAX_LENGTH="${MAX_LENGTH:-512}"
# Clear optional PP layer override unless user set it.
export DECODER_FIRST_PP_LAYERS="${DECODER_FIRST_PP_LAYERS:-}"

# Prefer venv python over system cu130 torch.
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PATH="${ROOT}/.venv-swift/bin:${PATH}"

exec bash "${SCRIPT_DIR}/megatron_sft_smoke.sh"
