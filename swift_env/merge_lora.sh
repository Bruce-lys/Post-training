#!/usr/bin/env bash
# Merge a Megatron-SWIFT LoRA checkpoint into the HF base model (recipe from hy/qwen38_flash_next_lora, verified).
# Usage: CKPT=<run>/checkpoint-112 OUT=<dir> [GPUS=0,1,2,3] bash swift_env/merge_lora.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${CKPT:?set CKPT=<checkpoint dir with adapter_model.safetensors>}"
OUT="${OUT:?set OUT=<merged output dir>}"
BASE="${BASE:-/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next}"
GPUS="${GPUS:-0,1,2,3}"
NPROC=$(awk -F',' '{print NF}' <<<"$GPUS")
LOG="${LOG:-${ROOT}/swift_env/logs/merge_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
[[ -e "$OUT" ]] && { echo "Refusing to overwrite existing output: $OUT" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPUS" NPROC_PER_NODE="$NPROC" MASTER_PORT="${MASTER_PORT:-29700}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODELSCOPE_CACHE="${ROOT}/.cache/modelscope" HF_HOME="${ROOT}/.cache/huggingface"
export PATH="${ROOT}/.venv-swift/bin:${PATH}"
echo "merge $CKPT -> $OUT (gpus $GPUS, tp=$NPROC ep=$NPROC) log $LOG"
megatron export \
  --model "$BASE" \
  --adapters "$CKPT" \
  --output_dir "$OUT" \
  --to_hf true \
  --merge_lora true \
  --save_safetensors true \
  --save_missing_weights true \
  --torch_dtype bfloat16 \
  --tensor_model_parallel_size "$NPROC" \
  --expert_model_parallel_size "$NPROC" \
  --expert_tensor_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --context_parallel_size 1 \
  --test_convert_precision false \
  2>&1 | tee "$LOG"
echo "rc=${PIPESTATUS[0]}" | tee -a "$LOG"
