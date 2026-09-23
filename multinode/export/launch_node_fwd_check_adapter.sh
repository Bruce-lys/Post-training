#!/usr/bin/env bash
# Single-node forward-only logits dump (TP8 PP1 EP8). Validation-only unless confirmed.
#   VARIANT=sparse|dense DRY_RUN=1 bash launch_node_fwd_check.sh
#   VARIANT=sparse DRY_RUN=0 START_FWD_CHECK=YES_I_CONFIRM bash launch_node_fwd_check.sh
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
MULTINODE=$ROOT/multinode
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
OVERLAY=$MULTINODE/overlays/mcore-bridge-9d610ffb
EXPECTED_OVERLAY_COMMIT=8417fcf510ada1b88b1052aaa2581638f628fe3a   # qsa-bitmap-chunked
VARIANT="${VARIANT:-sparse}"
CONFIG="${CONFIG:-$MULTINODE/tb226_flash_next_lora_fwd_check_${VARIANT}.yaml}"
MASTER_PORT="${MASTER_PORT:-29670}"
DRY_RUN="${DRY_RUN:-1}"
START_FWD_CHECK="${START_FWD_CHECK:-NO}"
TAG="${TAG:-tb226_flash_next_lora_fwd_check}"
TS="${TS:-${VARIANT}_$(date +%Y%m%d_%H%M%S)}"

ME=$(hostname -s)
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export PYTHONPATH="$OVERLAY/src:$ROOT/pyextra${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR=$MULTINODE/logs/$TAG/$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/train.node0.log
cp -f "$CONFIG" "$LOG_DIR/config.node0.yaml"

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
[[ "$actual_commit" == "$EXPECTED_OVERLAY_COMMIT" ]] || { echo "ABORT: overlay commit $actual_commit != $EXPECTED_OVERLAY_COMMIT"; exit 1; }
[[ -z "$(git -C "$OVERLAY" status --short)" ]] || { echo "ABORT: overlay worktree is dirty"; exit 1; }

python - "$CONFIG" "$OVERLAY" <<'PY' | tee -a "$LOG"
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import yaml

config_path, overlay = sys.argv[1:]
config = yaml.safe_load(open(config_path))
name = Path(config_path).stem
summary = json.loads(Path("/kwkj-k8s/llm_team/lys/megatron-swift/multinode/smoke_data/fwd_check/summary.json").read_text())
dataset = Path(config["dataset"])
assert str(dataset) == summary["path"]
assert sum(1 for _ in dataset.open(encoding="utf-8")) == summary["rows"]
h = hashlib.sha256()
with dataset.open("rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
assert h.hexdigest() == summary["sha256"], "fwd_check texts changed since summary.json"
env = config["ENV"]
assert env["QSA_SPARSE_KERNEL"] in ("True", "0", 0)
assert env.get("FWD_DUMP_DIR", env.get("FWD_HIDDEN_DIR", "")).startswith("/kwkj-k8s/llm_team/lys/megatron-swift/outputs/fwd_check/")
assert config.get("mcore_adapter") and os.path.isfile(config["mcore_adapter"] + "/latest_checkpointed_iteration.txt"), "adapter checkpoint"
assert config["packing"] is False and config["padding_free"] is True
assert config["tensor_model_parallel_size"] == 8 and config["pipeline_model_parallel_size"] == 1
assert config["expert_model_parallel_size"] == 8 and config["context_parallel_size"] == 1
assert "decoder_first_pipeline_num_layers" not in config
assert config["micro_batch_size"] == 1 and 1 <= config["global_batch_size"] <= summary["rows"]
assert config["lora_dropout"] == 0.0
assert config["dataset_shuffle"] is False and config["train_dataloader_shuffle"] is False
assert config["train_iters"] == 1 and "num_train_epochs" not in config
assert config["save_steps"] == 1000000 and config["no_save_optim"] is True
assert "smoke" in config["output_dir"].lower()
names = [Path(p).name for p in config["external_plugins"]]
assert names in (["ple_chunked.py", "forward_logits_dump.py"], ["ple_chunked.py", "forward_hidden_dump.py"], ["ple_chunked.py", "forward_hidden_teacher.py"]), names

import mcore_bridge
from mcore_bridge.model.modules import QSASparseCoreAttention, qsa_sparse_supported

source = Path(inspect.getfile(mcore_bridge)).resolve()
assert source.is_relative_to((Path(overlay) / "src").resolve()), f"mcore_bridge loaded from {source}"
if str(env["QSA_SPARSE_KERNEL"]) == "True":
    assert qsa_sparse_supported(256)
else:
    assert not qsa_sparse_supported(256), "QSA_SPARSE_KERNEL=0 should disable the sparse kernel"
print(f"[validate] config={name} dataset={dataset} rows={summary['rows']} sha256=ok")
print(f"[validate] mcore_bridge={source}")
print(f"[validate] TP=8 PP=1 EP=8 CP=1 padding_free=true QSA_SPARSE_KERNEL={env['QSA_SPARSE_KERNEL']} dump={env.get('FWD_DUMP_DIR', env.get('FWD_HIDDEN_DIR'))}")
PY

echo "[launch] host=$ME single-node dry_run=$DRY_RUN config=$CONFIG overlay=$actual_commit" | tee -a "$LOG"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[launch] DRY_RUN=1: validation complete; forward check was not started." | tee -a "$LOG"
  exit 0
fi
[[ "$DRY_RUN" == "0" ]] || { echo "ABORT: DRY_RUN must be 0 or 1"; exit 1; }
[[ "$START_FWD_CHECK" == "YES_I_CONFIRM" ]] || { echo "ABORT: set START_FWD_CHECK=YES_I_CONFIRM"; exit 1; }

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
[[ "$busy" -eq 0 ]] || { echo "ABORT: $busy GPU(s) busy on $ME" | tee -a "$LOG"; nvidia-smi | tee -a "$LOG"; exit 1; }
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$MASTER_PORT$"; then
  echo "ABORT: master port $MASTER_PORT is already in use on $ME" | tee -a "$LOG"; exit 1
fi

cd "$ROOT"
set +e
megatron sft "$CONFIG" 2>&1 | tee -a "$LOG"
train_rc=${PIPESTATUS[0]}
set -e
# the plugin exits the ranks with code 0 after the dump; DONE marker is the real success signal
DUMP_DIR="${FWD_DUMP_DIR:-${FWD_HIDDEN_DIR:-}}"
if [[ -n "$DUMP_DIR" && -f "$DUMP_DIR/DONE" ]]; then
  echo "[launch] forward dump complete: $(cat "$DUMP_DIR/DONE")" | tee -a "$LOG"
  ls -la "$DUMP_DIR" | tee -a "$LOG"
  exit 0
fi
echo "[launch] forward dump did NOT complete (rc=$train_rc)" | tee -a "$LOG"
exit 1
