#!/usr/bin/env bash
# Minimal smoke: 2 steps, local JSONL, LoRA — verify train loop starts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-swift"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -x "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Single-GPU: CUDA_VISIBLE_DEVICES=5 bash swift_env/megatron_sft_smoke.sh
# (or GPU_ID=5 — maps to local cuda:0 inside the job)
GPU_ID="${GPU_ID:-}"
if [[ -n "${GPU_ID}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${ROOT}/.cache/modelscope}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
mkdir -p "${MODELSCOPE_CACHE}" "${HF_HOME}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29551}"

bash "${SCRIPT_DIR}/check_gpu.sh"

# Auto parallel layout from visible GPU count (override with TP/PP/EP env).
NUM_GPUS="$(python3 - <<'PY'
import os
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not cvd:
    import torch
    print(torch.cuda.device_count())
else:
    print(len([x for x in cvd.split(",") if x.strip() != ""]))
PY
)"

# Qwen3.8-Flash-Next (~336GB disk): experts ~230G + PLE ngram ~95G + rest.
# LoRA still materializes full base weights at init.
# Working 4×H200: TP=4 PP=1 EP=4 ETP=1 SP=true (~85GiB peak; TP shards ngram, EP shards experts).
# Working 8×H200: TP=2 PP=2 EP=4 (official-style).
# Bad 4× layouts (OOM): TP=1 PP=2 EP=2 — ngram not TP-sharded on early PP stage.
# Hopper backward needs Triton>=3.7.1 (fla gated_delta_rule); torch pins 3.5.1 — override if needed.
MIN_GPUS="${MIN_GPUS:-4}"
if [[ "${NUM_GPUS}" -lt "${MIN_GPUS}" && "${ALLOW_UNDERPROVISIONED:-0}" != "1" ]]; then
  echo "[FAIL] Need at least ${MIN_GPUS} free GPUs for Qwen3.8-Flash-Next smoke. Got NUM_GPUS=${NUM_GPUS}."
  echo "       4×H200: CUDA_VISIBLE_DEVICES=0,1,2,3 bash swift_env/megatron_sft_smoke.sh"
  echo "       8×H200: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash swift_env/megatron_sft_smoke.sh"
  exit 1
fi

export NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS}}"

if [[ "${NUM_GPUS}" -ge 8 ]]; then
  TP="${TP:-2}"
  PP="${PP:-2}"
  EP="${EP:-4}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
  SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
  DECODER_FIRST_PP_LAYERS="${DECODER_FIRST_PP_LAYERS:-12}"
  MAX_LENGTH="${MAX_LENGTH:-1024}"
  echo "[smoke] 8-GPU layout (NUM_GPUS=${NUM_GPUS}, TP/PP/EP=${TP}/${PP}/${EP})"
elif [[ "${NUM_GPUS}" -eq 4 ]]; then
  TP="${TP:-4}"
  PP="${PP:-1}"
  EP="${EP:-4}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
  SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
  DECODER_FIRST_PP_LAYERS="${DECODER_FIRST_PP_LAYERS:-}"
  MAX_LENGTH="${MAX_LENGTH:-512}"
  echo "[smoke] 4-GPU layout (NUM_GPUS=${NUM_GPUS}, TP/PP/EP=${TP}/${PP}/${EP}, SP=${SEQUENCE_PARALLEL})"
else
  TP="${TP:-}"
  PP="${PP:-}"
  EP="${EP:-}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${NUM_GPUS}}"
  SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-true}"
  DECODER_FIRST_PP_LAYERS="${DECODER_FIRST_PP_LAYERS:-}"
  MAX_LENGTH="${MAX_LENGTH:-512}"
  if [[ -z "${TP}" || -z "${PP}" || -z "${EP}" ]]; then
    echo "[FAIL] NUM_GPUS=${NUM_GPUS}: set TP PP EP (4-GPU default: TP=4 PP=1 EP=4; 8-GPU: TP=2 PP=2 EP=4)"
    exit 1
  fi
  echo "[smoke] custom layout (NUM_GPUS=${NUM_GPUS}, TP/PP/EP=${TP}/${PP}/${EP})"
fi

MODEL="${MODEL:-/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/megatron_smoke}"
DATA_JSON="${DATA_JSON:-${SCRIPT_DIR}/data/smoke_alpaca.jsonl}"
MAX_STEPS="${MAX_STEPS:-2}"

LOG_DIR="${ROOT}/swift_env/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "$(dirname "${DATA_JSON}")"
LOG="${LOG_DIR}/megatron_smoke_$(date +%Y%m%d_%H%M%S).log"

if [[ ! -f "${DATA_JSON}" ]]; then
  cat > "${DATA_JSON}" <<'EOF'
{"instruction": "Say hello in one word.", "input": "", "output": "Hello"}
{"instruction": "What is 2+2?", "input": "", "output": "4"}
{"instruction": "Capital of France?", "input": "", "output": "Paris"}
{"instruction": "Reverse 'cat'.", "input": "", "output": "tac"}
EOF
  echo "[data] wrote ${DATA_JSON}"
fi

echo "[smoke] MODEL=${MODEL}"
echo "[smoke] DATA=${DATA_JSON}"
echo "[smoke] NUM_GPUS=${NUM_GPUS} NPROC=${NPROC_PER_NODE} TP=${TP} PP=${PP} EP=${EP}"
echo "[smoke] gbs=${GLOBAL_BATCH_SIZE} max_steps=${MAX_STEPS} seq=${MAX_LENGTH}"
echo "[smoke] log=${LOG}"

EXTRA_ARGS=()
if [[ -n "${DECODER_FIRST_PP_LAYERS}" ]]; then
  EXTRA_ARGS+=(--decoder_first_pipeline_num_layers "${DECODER_FIRST_PP_LAYERS}")
fi

megatron sft \
  --model "${MODEL}" \
  --dataset "${DATA_JSON}" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules in_proj out_proj linear_proj linear_qkv \
  --tensor_model_parallel_size "${TP}" \
  --expert_model_parallel_size "${EP}" \
  --expert_tensor_parallel_size 1 \
  --pipeline_model_parallel_size "${PP}" \
  "${EXTRA_ARGS[@]}" \
  --sequence_parallel "${SEQUENCE_PARALLEL}" \
  --context_parallel_size 1 \
  --moe_grouped_gemm true \
  --moe_permute_fusion true \
  --moe_aux_loss_coeff 1e-3 \
  --micro_batch_size 1 \
  --global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --train_iters "${MAX_STEPS}" \
  --finetune true \
  --cross_entropy_loss_fusion true \
  --lr 1e-4 \
  --lr_warmup_fraction 0.0 \
  --min_lr 1e-5 \
  --max_length "${MAX_LENGTH}" \
  --save_steps 9999 \
  --eval_steps 9999 \
  --save_safetensors true \
  --merge_lora false \
  --no_save_optim true \
  --no_save_rng true \
  --attention_backend auto \
  --padding_free false \
  --vit_attn_impl sdpa \
  --dataloader_num_workers 2 \
  --dataset_num_proc 2 \
  --logging_steps 1 \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG}"

echo "[done] smoke log=${LOG}"
