#!/usr/bin/env bash
# Megatron LoRA SFT for Qwen3.8-Flash-Next (qwen4_exp).
# Adapted from ms-swift example:
#   https://github.com/modelscope/ms-swift/blob/main/examples/models/qwen4_exp/megatron_sft.sh
#
# 8×H200 default: TP=2 PP=2 EP=4; 4×H200: TP=4 PP=1 EP=4 (see QWEN38_FLASH_NEXT_PARALLEL.md).
# Env: source swift_env/activate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-swift"

if [[ -x "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/.cache/modelscope}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
mkdir -p "${MODELSCOPE_CACHE}" "${HF_HOME}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${SCRIPT_DIR}/check_gpu.sh"

MODEL="${MODEL:-/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/megatron_sft_lora}"
TP="${TP:-2}"
PP="${PP:-2}"
EP="${EP:-4}"
GBS="${GBS:-8}"
MBS="${MBS:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

LOG_DIR="${ROOT}/swift_env/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG="${LOG_DIR}/megatron_sft_$(date +%Y%m%d_%H%M%S).log"

echo "[run] MODEL=${MODEL}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] TP=${TP} PP=${PP} EP=${EP} GBS=${GBS} seq=${MAX_LENGTH}"
echo "[run] log=${LOG}"

megatron sft \
  --model "${MODEL}" \
  --dataset 'swift/self-cognition#500' \
  --model_name swift-robot \
  --model_author swift \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules in_proj out_proj linear_proj linear_qkv \
  --tensor_model_parallel_size "${TP}" \
  --expert_model_parallel_size "${EP}" \
  --expert_tensor_parallel_size 1 \
  --pipeline_model_parallel_size "${PP}" \
  --decoder_first_pipeline_num_layers 12 \
  --sequence_parallel true \
  --context_parallel_size 1 \
  --moe_grouped_gemm true \
  --moe_permute_fusion true \
  --moe_aux_loss_coeff 1e-3 \
  --micro_batch_size "${MBS}" \
  --global_batch_size "${GBS}" \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --num_train_epochs 3 \
  --finetune true \
  --cross_entropy_loss_fusion true \
  --lr 1e-4 \
  --lr_warmup_fraction 0.05 \
  --min_lr 1e-5 \
  --max_length "${MAX_LENGTH}" \
  --split_dataset_ratio 0.01 \
  --eval_steps 1000 \
  --save_steps 50 \
  --save_safetensors true \
  --merge_lora true \
  --use_precision_aware_optimizer true \
  --no_save_optim true \
  --no_save_rng true \
  --attention_backend auto \
  --padding_free false \
  --vit_attn_impl sdpa \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --load_from_cache_file true \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG}"

echo "[done] log=${LOG}"
