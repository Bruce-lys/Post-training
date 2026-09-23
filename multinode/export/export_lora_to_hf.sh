#!/usr/bin/env bash
# Merge a LoRA checkpoint of the tb226 formal run into the Qwen3.8-Flash-Next base and export HF safetensors.
#   CKPT=/.../v1-20260918-182955/checkpoint-1750 OUT=/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next-tb226-ckpt1750 \
#   DRY_RUN=1 bash export_lora_to_hf.sh        # validate only
#   ... DRY_RUN=0 START_EXPORT=YES_I_CONFIRM bash export_lora_to_hf.sh
# Single node, 8 GPUs (TP8 PP1 EP8): loads the HF base into mcore, loads the adapter dist-ckpt (saved with TP8 PP2 EP8;
# PP is resharded by mcore dist-checkpointing), merges LoRA in bf16, saves HF, then runs swift's own
# test_convert_precision (fp32 CPU forward of the exported HF model vs the merged mcore model on a short example).
set -euo pipefail
ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
OVERLAY=$ROOT/multinode/overlays/mcore-bridge-9d610ffb
EXPECTED_OVERLAY_COMMIT=8417fcf510ada1b88b1052aaa2581638f628fe3a
BASE=/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next
CKPT="${CKPT:?set CKPT=<checkpoint-N dir>}"
OUT="${OUT:?set OUT=<output HF dir>}"
DRY_RUN="${DRY_RUN:-1}"
START_EXPORT="${START_EXPORT:-NO}"
MASTER_PORT="${MASTER_PORT:-29690}"
LOG_DIR=$ROOT/multinode/logs/export/$(basename "$CKPT")_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"; LOG=$LOG_DIR/export.log

actual_commit=$(git -C "$OVERLAY" rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_OVERLAY_COMMIT" ]] || { echo "ABORT: overlay commit $actual_commit != $EXPECTED_OVERLAY_COMMIT"; exit 1; }
[[ -z "$(git -C "$OVERLAY" status --short)" ]] || { echo "ABORT: overlay worktree is dirty"; exit 1; }
[[ -f "$CKPT/latest_checkpointed_iteration.txt" && -f "$CKPT/args.json" && -f "$CKPT/adapter_config.json" ]] || { echo "ABORT: $CKPT is not a swift megatron LoRA checkpoint"; exit 1; }
ITER=$(cat "$CKPT/latest_checkpointed_iteration.txt"); [[ -d "$CKPT/iter_$(printf %07d "$ITER")" ]] || { echo "ABORT: iter dir missing for $ITER"; exit 1; }
[[ -f "$BASE/config.json" ]] || { echo "ABORT: base model missing"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "ABORT: OUT exists: $OUT"; exit 1; }
python3 - "$CKPT/args.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
assert a["tuner_type"] == "lora" and a["lora_rank"] == 32 and a["lora_alpha"] == 64, (a["tuner_type"], a["lora_rank"], a["lora_alpha"])
assert a["model"] == "/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"
assert a["language_model_only"] is True and a["packing"] is False
print("[validate] ckpt args ok: lora r=%d alpha=%d targets=%s" % (a["lora_rank"], a["lora_alpha"], a["target_modules"]))
PY
echo "[validate] ckpt=$CKPT iteration=$ITER overlay=$actual_commit out=$OUT" | tee -a "$LOG"

source "$VENV/bin/activate"
export PYTHONPATH="$OVERLAY/src:$ROOT/pyextra${PYTHONPATH:+:$PYTHONPATH}"
export USE_MCORE_GDN=True QSA_SPARSE_KERNEL=True PLE_CHUNK_SIZE=16384
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NPROC_PER_NODE=8 MASTER_PORT CUDA_DEVICE_MAX_CONNECTIONS=1
export MODELSCOPE_CACHE=$ROOT/.cache/modelscope HF_HOME=$ROOT/.cache/huggingface

CMD=(megatron export
  --to_hf true --model "$BASE" --mcore_adapter "$CKPT" --merge_lora true --save_missing_weights true
  --test_convert_precision true --test_convert_dtype float32
  --tensor_model_parallel_size 8 --pipeline_model_parallel_size 1
  --expert_model_parallel_size 8 --expert_tensor_parallel_size 1 --sequence_parallel true --moe_grouped_gemm true
  --template qwen3_8 --max_length 212992 --padding_free true --language_model_only true --torch_dtype bfloat16
  --output_dir "$OUT")
printf '[launch] %q ' "${CMD[@]}" | tee -a "$LOG"; echo | tee -a "$LOG"
if [[ "$DRY_RUN" == "1" ]]; then echo "[launch] DRY_RUN=1: validation complete; export not started." | tee -a "$LOG"; exit 0; fi
[[ "$START_EXPORT" == "YES_I_CONFIRM" ]] || { echo "ABORT: set START_EXPORT=YES_I_CONFIRM"; exit 1; }
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
[[ "$busy" -eq 0 ]] || { echo "ABORT: $busy GPU(s) busy" | tee -a "$LOG"; exit 1; }
cd "$ROOT"
"${CMD[@]}" 2>&1 | tee -a "$LOG"
echo "[done] rc=${PIPESTATUS[0]} out=$OUT" | tee -a "$LOG"
ls "$OUT" | head -20 | tee -a "$LOG"; du -sh "$OUT" | tee -a "$LOG"
