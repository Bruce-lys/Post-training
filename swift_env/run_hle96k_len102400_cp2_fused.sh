#!/usr/bin/env bash
# Qwen3.8-Flash-Next LoRA SFT | hle96k | len 102400 | TP2/PP2/CP2/EP4 | cuDNN fused attention
# 用法(在 8 卡 GPU 节点上,例如 0019):
#   nohup bash /kwkj-k8s/llm_team/lys/megatron-swift/swift_env/run_hle96k_len102400_cp2_fused.sh > /dev/null 2>&1 &
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/cainn/megatron-swift
# shellcheck disable=SC1091
source "$ROOT/swift_env/activate.sh"
unset PYTORCH_CUDA_ALLOC_CONF                       # activate.sh 用的是 torch 2.9 已弃用的旧名
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8   # GC: 让缓存块及时还给驱动,给 Triton/cuDNN 留空间
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

TS=$(date +%Y%m%d_%H%M%S)
RUN=sft_qwen38_hle96k_lora_r32_a64_len102400_tp2pp2cp2ep4_fused_${TS}
MODEL=/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next
DATA=/kwkj-k8s/llm_team/sjy/hle-5.3/sft_qwen38_27b_96k/data/hle_correct_sft.jsonl
OUT_DIR=/kwkj-k8s/llm_team/chwang/models/Qwen3.8-Flash-Next-hle96k-lora-r32-a64-len102400-tp2pp2cp2ep4-fused-${TS}
LOG_DIR=/kwkj-k8s/llm_team/chwang/logs/${RUN}
TB_DIR=${LOG_DIR}/tensorboard
LOG_FILE=${LOG_DIR}/train.log
mkdir -p "$OUT_DIR" "$TB_DIR"

# 启动前确认 8 卡空闲(任一卡已用 >2GB 就退出,不抢别人的卡)
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
if [ "$busy" -gt 0 ]; then
  echo "ABORT: ${busy} GPU(s) already in use" | tee -a "$LOG_FILE"; nvidia-smi | tee -a "$LOG_FILE"; exit 1
fi

{
  echo "[launch] host=$(hostname) time=${TS}"
  echo "[launch] RUN=${RUN}"
  echo "[launch] OUT_DIR=${OUT_DIR}"
  echo "[launch] LOG_FILE=${LOG_FILE}"
} | tee -a "$LOG_FILE"

python -m torch.distributed.run --nproc_per_node 8 \
  "$ROOT/.venv-swift/lib/python3.12/site-packages/swift/cli/_megatron/sft.py" \
  --model "$MODEL" \
  --dataset "$DATA" \
  --save_safetensors true \
  --load_from_cache_file true \
  --enable_thinking true \
  --no_add_non_thinking_prefix \
  --loss_scale ignore_empty_think \
  --split_dataset_ratio 0.01 \
  --tuner_type lora \
  --target_modules in_proj out_proj linear_proj linear_qkv \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --lora_dtype bfloat16 \
  --language_model_only true \
  --tensor_model_parallel_size 2 \
  --pipeline_model_parallel_size 2 \
  --context_parallel_size 2 \
  --expert_model_parallel_size 4 \
  --decoder_first_pipeline_num_layers 20 \
  --moe_permute_fusion true \
  --moe_grouped_gemm true \
  --moe_shared_expert_overlap true \
  --moe_aux_loss_coeff 1e-6 \
  --micro_batch_size 1 \
  --global_batch_size 4 \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --num_train_epochs 3 \
  --packing true \
  --packing_length 102400 \
  --padding_free true \
  --finetune true \
  --freeze_vit true \
  --freeze_aligner true \
  --cross_entropy_loss_fusion true \
  --lr 1e-4 \
  --lr_warmup_fraction 0.05 \
  --min_lr 1e-6 \
  --output_dir "$OUT_DIR" \
  --report_to tensorboard \
  --tensorboard_dir "$TB_DIR" \
  --logging_steps 1 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 6 \
  --max_length 102400 \
  --dataloader_num_workers 8 \
  --dataset_num_proc 4 \
  --no_save_optim true \
  --no_save_rng true \
  --sequence_parallel true \
  --optimizer_cpu_offload true \
  --optimizer_offload_fraction 0.64 \
  --overlap_grad_reduce true \
  --attention_backend fused \
  2>&1 | tee -a "$LOG_FILE"
