#!/usr/bin/env bash
# Four-node PP split smoke launcher. Defaults to validation only.
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
MULTINODE=$ROOT/multinode
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
OVERLAY=$MULTINODE/overlays/mcore-bridge-9d610ffb
EXPECTED_OVERLAY_COMMIT=1f2bda0a30927c5270e20046ba86498af2499541
PP_FIRST_LAYERS="${PP_FIRST_LAYERS:?set PP_FIRST_LAYERS to 22, 23, or 24}"
NODES="${NODES:?set NODES=\"host0 host1 host2 host3\" (host0 = master)}"
DRY_RUN="${DRY_RUN:-1}"
START_PP_SMOKE="${START_PP_SMOKE:-NO}"

case "$PP_FIRST_LAYERS" in
  22) DEFAULT_PORT=29629 ;;
  23) DEFAULT_PORT=29630 ;;
  24) DEFAULT_PORT=29631 ;;
  *) echo "ABORT: PP_FIRST_LAYERS must be 22, 23, or 24"; exit 1 ;;
esac

CONFIG="${CONFIG:-$MULTINODE/tb226_flash_next_lora_4node_qsa_ppsplit${PP_FIRST_LAYERS}_smoke.yaml}"
MASTER_PORT="${MASTER_PORT:-$DEFAULT_PORT}"
TAG="${TAG:-tb226_flash_next_lora_4node_qsa_pp_split_ab}"
TS="${TS:-ppsplit${PP_FIRST_LAYERS}_$(date +%Y%m%d_%H%M%S)}"

read -r -a NODE_ARR <<<"$NODES"
export NNODES=${#NODE_ARR[@]}
[[ "$NNODES" -eq 4 ]] || { echo "ABORT: PP smoke requires exactly four nodes"; exit 1; }

ME=$(hostname -s)
export NODE_RANK=-1
for i in "${!NODE_ARR[@]}"; do
  [[ "${NODE_ARR[$i]}" == "$ME" ]] && NODE_RANK=$i
done
[[ "$NODE_RANK" -ge 0 ]] || { echo "ABORT: $ME not in NODES=[$NODES]"; exit 1; }

export MASTER_ADDR
MASTER_ADDR=$(getent hosts "${NODE_ARR[0]}" | awk '{print $1; exit}')
export MASTER_PORT
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export QSA_SPARSE_KERNEL=True
export PYTHONPATH="$OVERLAY/src:$ROOT/pyextra${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR=$MULTINODE/logs/$TAG/$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/train.node${NODE_RANK}.log
GPU_LOG=$LOG_DIR/gpu.node${NODE_RANK}.csv
cp -f "$CONFIG" "$LOG_DIR/config.node${NODE_RANK}.yaml"

source "$VENV/bin/activate"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  export "${line%%=*}=${line#*=}"
done < <(python - "$CONFIG" <<'PY'
import sys
import yaml

for key, value in (yaml.safe_load(open(sys.argv[1])).get("ENV") or {}).items():
    print(f"{key}={value}")
PY
)

actual_commit=$(git -C "$OVERLAY" rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_OVERLAY_COMMIT" ]] || {
  echo "ABORT: overlay commit $actual_commit != $EXPECTED_OVERLAY_COMMIT"
  exit 1
}
[[ -z "$(git -C "$OVERLAY" status --short)" ]] || {
  echo "ABORT: overlay worktree is dirty"
  exit 1
}

python - "$CONFIG" "$OVERLAY" "$PP_FIRST_LAYERS" <<'PY' | tee -a "$LOG"
import inspect
import json
import sys
from pathlib import Path

import yaml

config_path, overlay, split_text = sys.argv[1:]
split = int(split_text)
config = yaml.safe_load(open(config_path))
dataset = Path("/kwkj-k8s/llm_team/lys/megatron-swift/multinode/smoke_data/pp_split_ab/real_step5_repeated_40rows.jsonl")
assert config["dataset"] == str(dataset)
assert sum(1 for _ in dataset.open(encoding="utf-8")) == 40
summary = json.load(open(str(dataset) + ".summary.json"))
assert summary["steps"] == 5 and summary["warmup_steps"] == 2
assert summary["measured_steps"] == [3, 4, 5]
assert summary["samples_per_step"] == 8 and summary["tokens_per_step"] == 827160
assert config["ENV"]["QSA_SPARSE_KERNEL"] == "True"
assert str(config["ENV"]["PP_FIRST_LAYERS"]) == str(split)
assert config["ENV"]["PERF_STEP_LABELS"] == "W1,W2,M1,M2,M3"
assert config["packing"] is False and config["padding_free"] is True
assert config["tensor_model_parallel_size"] == 8
assert config["pipeline_model_parallel_size"] == 2
assert config["context_parallel_size"] == 1
assert config["expert_model_parallel_size"] == 16
assert config["expert_tensor_parallel_size"] == 1
assert config["sequence_parallel"] is True
assert config["decoder_first_pipeline_num_layers"] == split
assert config["micro_batch_size"] == 1 and config["global_batch_size"] == 8
assert config["dataset_shuffle"] is False and config["train_dataloader_shuffle"] is False
assert config["train_iters"] == 5
assert config["save_strategy"] == "steps" and config["save_steps"] == 1000000
assert config["ENV"]["PROFILE_DISABLE_SAVE"] == 1
assert "smoke" in config["output_dir"].lower()
assert any("pp_split_timing.py" in path for path in config["external_plugins"])
assert not any("nsys" in path.lower() for path in config["external_plugins"])

import mcore_bridge
from mcore_bridge.model.modules import QSASparseCoreAttention, qsa_sparse_supported

source = Path(inspect.getfile(mcore_bridge)).resolve()
expected_root = (Path(overlay) / "src").resolve()
assert source.is_relative_to(expected_root), f"mcore_bridge loaded from {source}, not {expected_root}"
assert qsa_sparse_supported(256)
print(f"[validate] split={split}/{48 - split} dataset={dataset} rows=40 tokens_per_step={summary['tokens_per_step']}")
print(f"[validate] mcore_bridge={source}")
print(f"[validate] qsa_core={inspect.getfile(QSASparseCoreAttention)}")
print("[validate] QSA=true packing=false padding_free=true TP=8 PP=2 CP=1 EP=16 train_iters=5 nsys=off save=off")
PY

echo "[launch] host=$ME rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT split=$PP_FIRST_LAYERS/$((48 - PP_FIRST_LAYERS)) dry_run=$DRY_RUN" | tee -a "$LOG"
echo "[launch] config=$CONFIG overlay=$actual_commit" | tee -a "$LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[launch] DRY_RUN=1: validation complete; PP smoke was not started." | tee -a "$LOG"
  exit 0
fi
[[ "$DRY_RUN" == "0" ]] || { echo "ABORT: DRY_RUN must be 0 or 1"; exit 1; }
[[ "$START_PP_SMOKE" == "YES_I_CONFIRM" ]] || {
  echo "ABORT: set START_PP_SMOKE=YES_I_CONFIRM on all four nodes"
  exit 1
}

if [[ "$NODE_RANK" -eq 0 ]] && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$MASTER_PORT$"; then
  echo "ABORT: master port $MASTER_PORT is already in use on $ME" | tee -a "$LOG"
  exit 1
fi

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{count++} END{print count+0}')
[[ "$busy" -eq 0 ]] || {
  echo "ABORT: $busy GPU(s) busy on $ME" | tee -a "$LOG"
  nvidia-smi | tee -a "$LOG"
  exit 1
}

if [[ -f $MULTINODE/secrets.env ]]; then
  set -a
  source "$MULTINODE/secrets.env"
  set +a
fi

echo "epoch_s,host,gpu_index,memory_used_mib,utilization_gpu_pct" >"$GPU_LOG"
(
  while :; do
    epoch=$(date +%s)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F', *' -v epoch="$epoch" -v host="$ME" '{print epoch "," host "," $1 "," $2 "," $3}'
    sleep 1
  done
) >>"$GPU_LOG" 2>&1 &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

cd "$ROOT"
set +e
megatron sft "$CONFIG" 2>&1 | tee -a "$LOG"
train_rc=${PIPESTATUS[0]}
set -e
exit "$train_rc"
