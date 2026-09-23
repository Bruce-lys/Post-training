#!/usr/bin/env bash
# Four-node no-packing smoke launcher (EP8 + PP 24/24 + dp_balance_order). Validation-only unless confirmed.
# On EACH of the four nodes (host0 = master), same TS on all nodes:
#   NODES="h0 h1 h2 h3" TS=<tag> DRY_RUN=1 bash launch_node_qsa_nopack_smoke.sh
#   NODES="h0 h1 h2 h3" TS=<tag> DRY_RUN=0 START_NOPACK_SMOKE=YES_I_CONFIRM bash launch_node_qsa_nopack_smoke.sh
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
MULTINODE=$ROOT/multinode
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
OVERLAY=$MULTINODE/overlays/mcore-bridge-9d610ffb
EXPECTED_OVERLAY_COMMIT=1f2bda0a30927c5270e20046ba86498af2499541
CONFIG="${CONFIG:-$MULTINODE/tb226_flash_next_lora_4node_qsa_len_curve_smoke.yaml}"
NODES="${NODES:?set NODES=\"host0 host1 host2 host3\" (host0 = master)}"
MASTER_PORT="${MASTER_PORT:-29650}"
DRY_RUN="${DRY_RUN:-1}"
START_NOPACK_SMOKE="${START_NOPACK_SMOKE:-NO}"
TAG="${TAG:-tb226_flash_next_lora_4node_qsa_nopack_smoke}"
TS="${TS:-$(basename "$CONFIG" .yaml)_$(date +%Y%m%d_%H%M%S)}"

read -r -a NODE_ARR <<<"$NODES"
export NNODES=${#NODE_ARR[@]}
[[ "$NNODES" -eq 4 ]] || { echo "ABORT: this smoke requires exactly four nodes"; exit 1; }

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
[[ "$DRY_RUN" == "1" ]] && export CUDA_VISIBLE_DEVICES=""

actual_commit=$(git -C "$OVERLAY" rev-parse HEAD)
[[ "$actual_commit" == "$EXPECTED_OVERLAY_COMMIT" ]] || {
  echo "ABORT: overlay commit $actual_commit != $EXPECTED_OVERLAY_COMMIT"
  exit 1
}
[[ -z "$(git -C "$OVERLAY" status --short)" ]] || {
  echo "ABORT: overlay worktree is dirty"
  exit 1
}

python - "$CONFIG" "$OVERLAY" <<'PY' | tee -a "$LOG"
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import yaml

config_path, overlay = sys.argv[1:]
config = yaml.safe_load(open(config_path))
name = Path(config_path).stem
smoke_dir = Path("/kwkj-k8s/llm_team/lys/megatron-swift/multinode/smoke_data/len_curve")
summary = json.loads((smoke_dir / "summary.json").read_text())
expected = summary["subset"]
dataset = Path(config["dataset"])
assert str(dataset) == expected["path"], (config["dataset"], expected["path"])
assert sum(1 for _ in dataset.open(encoding="utf-8")) == expected["rows"]
h = hashlib.sha256()
with dataset.open("rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
assert h.hexdigest() == expected["sha256"], "smoke dataset changed since summary.json"

env = config["ENV"]
assert env["QSA_SPARSE_KERNEL"] == "True"
assert "DP_TOKEN_BALANCE" not in env
assert str(env["DP_BALANCE_ORDER"]) == "True"
lengths_file = Path(env["DP_BALANCE_LENGTHS_FILE"])
lengths = json.loads(lengths_file.read_text())["lengths"]
assert len(lengths) == expected["rows"], (len(lengths), expected["rows"])
assert env["PROFILE_DISABLE_SAVE"] == 1
assert config["packing"] is False and config["padding_free"] is True
assert config["max_length"] == 212992
assert config["tensor_model_parallel_size"] == 8
assert config["pipeline_model_parallel_size"] == 2
assert config["context_parallel_size"] == 1
assert config["expert_model_parallel_size"] in (8, 16)
assert config["expert_tensor_parallel_size"] == 1
assert config["sequence_parallel"] is True
assert config["decoder_first_pipeline_num_layers"] in (22, 24)
assert config["apply_rope_fusion"] is False
assert config["micro_batch_size"] == 1 and config["global_batch_size"] == 8
assert config["group_by_length"] is False
assert config.get("data_sharding") is False
assert config["dataset_shuffle"] is False and config["train_dataloader_shuffle"] is False
assert config["train_iters"] == 6
assert "num_train_epochs" not in config
assert config["save_steps"] == 1000000 and config["no_save_optim"] is True
assert "smoke" in config["output_dir"].lower()
names = [Path(p).name for p in config["external_plugins"]]
assert not any("dp_token_balance" in n for n in names)
for need in ("ple_chunked.py", "dp_balance_order.py", "mem_probe.py", "iteration_timing.py"):
    assert need in names, need
assert not any("packed_ple_varlen" in n or "nsys" in n.lower() or "pp_split_timing" in n for n in names)

import mcore_bridge
from mcore_bridge.model.modules import QSASparseCoreAttention, qsa_sparse_supported

source = Path(inspect.getfile(mcore_bridge)).resolve()
expected_root = (Path(overlay) / "src").resolve()
assert source.is_relative_to(expected_root), f"mcore_bridge loaded from {source}, not {expected_root}"
assert qsa_sparse_supported(256)
for plugin in config["external_plugins"]:
    if plugin.endswith("dp_balance_order.py"):
        spec = importlib.util.spec_from_file_location("dp_balance_order", plugin)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.self_check()
print(f"[validate] config={name} dataset={dataset} rows={expected['rows']} sha256=ok lengths_file_rows={len(lengths)}")
print(f"[validate] mcore_bridge={source}")
print(f"[validate] qsa_core={inspect.getfile(QSASparseCoreAttention)}")
print(f"[validate] packing=false padding_free=true gbs=8 train_iters=6 (len-curve) EP={config['expert_model_parallel_size']} split={config['decoder_first_pipeline_num_layers']}/{48 - config['decoder_first_pipeline_num_layers']}")
print("[validate] QSA=true TP=8 PP=2 CP=1 ETP=1 plugins=ple_chunked>dp_balance_order>mem_probe>iteration_timing save=off")
PY

echo "[launch] host=$ME rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT dry_run=$DRY_RUN" | tee -a "$LOG"
echo "[launch] config=$CONFIG overlay=$actual_commit" | tee -a "$LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[launch] DRY_RUN=1: validation complete; smoke was not started." | tee -a "$LOG"
  exit 0
fi
[[ "$DRY_RUN" == "0" ]] || { echo "ABORT: DRY_RUN must be 0 or 1"; exit 1; }
[[ "$START_NOPACK_SMOKE" == "YES_I_CONFIRM" ]] || {
  echo "ABORT: set START_NOPACK_SMOKE=YES_I_CONFIRM on all four nodes"
  exit 1
}

if [[ "$NODE_RANK" -eq 0 ]] && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$MASTER_PORT$"; then
  echo "ABORT: master port $MASTER_PORT is already in use on $ME" | tee -a "$LOG"
  exit 1
fi

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
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

echo "epoch_s,host,gpu_index,memory_used_mib,utilization_gpu_pct" > "$GPU_LOG"
(
  while :; do
    now=$(date +%s)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -v t="$now" -v h="$ME" -F', *' '{print t","h","$1","$2","$3}' >> "$GPU_LOG"
    sleep 1
  done
) &
monitor_pid=$!
cleanup_monitor() { kill "$monitor_pid" 2>/dev/null || true; }
trap cleanup_monitor EXIT

cd "$ROOT"
set +e
megatron sft "$CONFIG" 2>&1 | tee -a "$LOG"
train_rc=${PIPESTATUS[0]}
set -e
exit "$train_rc"
