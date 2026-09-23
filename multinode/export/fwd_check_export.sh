#!/usr/bin/env bash
# Train/inference consistency check for an exported (LoRA-merged) checkpoint. Single node, 8 GPUs, steps run one at a time:
#   STEP=mg      Megatron forward dump on the 5 fwd_check texts with the UNMERGED adapter (the model training computed)
#   STEP=hf      transformers forward on the exported merged HF checkpoint (bf16, sdpa)     -> train_*.pt format
#   STEP=sglang  sglang serving the exported checkpoint, teacher-forced prefill logprobs   -> sglang_*.pt
#   STEP=compare CPU only: train(adapter) vs sglang(export), train(adapter) vs hf(export)
#   STEP=teacher_hf / teacher_mg / teacher_compare   layer-2 merge-equivalence: teacher-forced per-layer comparison
#       (HF fp32 hidden states of the exported model vs Megatron fp32 base + unmerged adapter fed the same layer inputs)
# usage: CKPT=<checkpoint-N> OUT=<exported HF dir> STEP=mg DRY_RUN=1 bash fwd_check_export.sh
set -euo pipefail
ROOT=/kwkj-k8s/llm_team/lys/megatron-swift; M=$ROOT/multinode; E=$M/export
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
CKPT="${CKPT:?}"; OUT="${OUT:?}"; STEP="${STEP:?mg|hf|sglang|compare}"; DRY_RUN="${DRY_RUN:-1}"
ITER=$(cat "$CKPT/latest_checkpointed_iteration.txt")
FC=$ROOT/outputs/fwd_check
TRAIN=$FC/train_sparse_adapter_$ITER; HF=$FC/hf_export_$ITER; SG=$FC/sglang_export_$ITER
case "$STEP" in
  mg)
    CFG=$M/tb226_flash_next_lora_fwd_check_sparse_adapter_$ITER.yaml
    sed -e "s#__CKPT__#$CKPT#g" -e "s#__ITER__#$ITER#g" "$E/tb226_flash_next_lora_fwd_check_sparse_adapter.template.yaml" > "$CFG"
    echo "[mg] config=$CFG dump=$TRAIN"
    VARIANT=sparse CONFIG="$CFG" DRY_RUN="$DRY_RUN" START_FWD_CHECK="${START_FWD_CHECK:-NO}" TS="adapter_${ITER}_$(date +%H%M%S)" bash "$E/launch_node_fwd_check_adapter.sh"
    ;;
  hf)
    [[ -d "$OUT" && -d "$TRAIN" ]] || { echo "ABORT: need exported dir and $TRAIN (run STEP=mg first)"; exit 1; }
    source "$VENV/bin/activate"; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    echo "[hf] model=$OUT -> $HF"
    [[ "$DRY_RUN" == "1" ]] && { echo "[hf] DRY_RUN=1: not started"; exit 0; }
    python "$E/hf_forward_reference_model.py" --model "$OUT" --train-dir "$TRAIN" --out "$HF" --indices 0 1 2 3 4 --attn-impl sdpa --dtype bfloat16 --max-memory-gib 118
    ;;
  sglang)
    [[ -d "$OUT" && -d "$TRAIN" ]] || { echo "ABORT: need exported dir and $TRAIN"; exit 1; }
    source "$VENV/bin/activate"; cd "$M"
    echo "[sglang] model=$OUT -> $SG"
    [[ "$DRY_RUN" == "1" ]] && { echo "[sglang] DRY_RUN=1: not started"; exit 0; }
    MODEL_DIR="$OUT" CONTAINER=qwen38-export-check bash serve_fwd_check_sglang.sh start
    python sglang_prefill_logprobs.py --train-dir "$TRAIN" --out "$SG" --url http://127.0.0.1:30100 --topk 100 --full-vocab-max-t 4096 --full-vocab-topk 2048 || true
    CONTAINER=qwen38-export-check bash serve_fwd_check_sglang.sh stop
    ;;
  compare)
    source "$VENV/bin/activate"; cd "$M"
    [[ -d "$TRAIN" ]] || { echo "ABORT: $TRAIN missing"; exit 1; }
    if [[ -d "$SG" ]]; then python compare_forward.py --train "$TRAIN" --sglang "$SG" --out "$FC/report_export_${ITER}_train_vs_sglang.json"; fi
    if [[ -d "$HF" ]]; then python compare_forward.py --train "$TRAIN" --sglang "${SG:-$SG}" --train-dense "$HF" --out "$FC/report_export_${ITER}_train_vs_hf.json" || true; fi
    python - "$FC" "$ITER" <<'PY'
import json, os, sys
fc, it = sys.argv[1], sys.argv[2]
for name in ("train_vs_sglang", "train_vs_hf"):
    p = os.path.join(fc, "report_export_%s_%s.json" % (it, name))
    if not os.path.exists(p): print(name, ": no report"); continue
    r = json.load(open(p))
    key = "train_vs_sglang" if name == "train_vs_sglang" else "sparse_vs_dense"
    print("==", name, "(adapter-in-Megatron vs exported model)")
    for e in r.get(key, []):
        if key == "train_vs_sglang":
            print("  idx %d T=%6d argmax=%.3f mean_nll train=%.4f other=%.4f gap=%+.2f%% |dnll|=%.3f" % (e["index"], e["T"], e["argmax_agree"], e["train_mean_nll"], e["sglang_mean_nll"], 100*(e["train_mean_nll"]-e["sglang_mean_nll"])/e["sglang_mean_nll"], e["nll_abs_diff"]["mean"]))
        else:
            print("  idx %d T=%6d argmax=%.3f |dnll|=%.3f rel_l2=%.3f" % (e["index"], e["T"], e["argmax_agree"], e["nll_abs_diff_mean"], e["rel_l2_mean"]))
print("reference floor (base model, no LoRA, 2026-09-18): argmax 0.80-0.85 per token, mean NLL gap <0.4% on the 6k-200k texts, |dnll| ~0.5")
PY
    ;;
  teacher_hf)
    # HF fp32 per-layer hidden states of the EXPORTED model on text 0 (needs $TRAIN/train_0.pt for the input ids)
    [[ -d "$OUT" && -f "$TRAIN/train_0.pt" ]] || { echo "ABORT: need exported dir and $TRAIN/train_0.pt (STEP=mg first)"; exit 1; }
    source "$VENV/bin/activate"; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    echo "[teacher_hf] model=$OUT -> $FC/hidden_hf_export_$ITER"
    [[ "$DRY_RUN" == "1" ]] && { echo "[teacher_hf] DRY_RUN=1: not started"; exit 0; }
    python "$E/hf_hidden_dump_model.py" --model "$OUT" --train-dir "$TRAIN" --index 0 --out "$FC/hidden_hf_export_$ITER" --max-memory-gib 118
    ;;
  teacher_mg)
    [[ "$DRY_RUN" == "1" || -f "$FC/hidden_hf_export_$ITER/layers.pt" ]] || { echo "ABORT: run STEP=teacher_hf first"; exit 1; }
    CFG=$M/tb226_flash_next_lora_fwd_check_teacher_adapter_$ITER.yaml
    sed -e "s#__CKPT__#$CKPT#g" -e "s#__ITER__#$ITER#g" "$E/tb226_flash_next_lora_fwd_check_teacher_adapter.template.yaml" > "$CFG"
    echo "[teacher_mg] config=$CFG dump=$FC/hidden_mg_teacher_adapter_$ITER"
    VARIANT=sparse CONFIG="$CFG" DRY_RUN="$DRY_RUN" START_FWD_CHECK="${START_FWD_CHECK:-NO}" TS="teacher_adapter_${ITER}_$(date +%H%M%S)" bash "$E/launch_node_fwd_check_adapter.sh"
    ;;
  teacher_compare)
    source "$VENV/bin/activate"
    python "$E/compare_hidden_teacher.py" "$FC/hidden_hf_export_$ITER/layers.pt" "$FC/hidden_mg_teacher_adapter_$ITER/layers.pt" | tail -12
    ;;
  *) echo "unknown STEP=$STEP"; exit 1;;
esac
