#!/usr/bin/env bash
# Four-node QSA THD formal launcher. Validation-only unless explicitly confirmed.
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
OVERLAY=$ROOT/multinode/overlays/mcore-bridge-9d610ffb
EXPECTED_OVERLAY_COMMIT=1f2bda0a30927c5270e20046ba86498af2499541
CONFIG="${CONFIG:-$ROOT/multinode/tb226_flash_next_lora_4node_212k_qsa_thd_overlay_formal.yaml}"
NODES="${NODES:?set NODES=\"host0 host1 host2 host3\" (host0 = master)}"
MASTER_PORT="${MASTER_PORT:-29600}"
DRY_RUN="${DRY_RUN:-1}"
START_FORMAL_TRAINING="${START_FORMAL_TRAINING:-NO}"
TAG="${TAG:-$(basename "$CONFIG" .yaml)}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

read -r -a NODE_ARR <<<"$NODES"
export NNODES=${#NODE_ARR[@]}
[[ "$NNODES" -eq 4 ]] || { echo "ABORT: formal training requires exactly four nodes"; exit 1; }

ME=$(hostname -s)
export NODE_RANK=-1
for i in "${!NODE_ARR[@]}"; do
  [[ "${NODE_ARR[$i]}" == "$ME" ]] && NODE_RANK=$i
done
[[ "$NODE_RANK" -ge 0 ]] || { echo "ABORT: $ME not in NODES=[$NODES]"; exit 1; }

export MASTER_ADDR
MASTER_ADDR=$(getent hosts "${NODE_ARR[0]}" | awk '{print $1; exit}')
export MASTER_PORT

LOG_DIR=$ROOT/multinode/logs/$TAG/$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/train.node${NODE_RANK}.log
cp -f "$CONFIG" "$LOG_DIR/config.node${NODE_RANK}.yaml"

source "$VENV/bin/activate"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  export "${line%%=*}=${line#*=}"
done < <(python - "$CONFIG" <<'PY'
import sys
import yaml

for key, value in (yaml.safe_load(open(sys.argv[1])).get('ENV') or {}).items():
    print(f'{key}={value}')
PY
)

export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export QSA_SPARSE_KERNEL=True
export PYTHONPATH="$OVERLAY/src:$ROOT/pyextra${PYTHONPATH:+:$PYTHONPATH}"

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
import inspect
import os
import sys
from pathlib import Path

import yaml

config_path, overlay = sys.argv[1:]
config = yaml.safe_load(open(config_path))
expected_dataset = '/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl'
expected_output = '/kwkj-k8s/llm_team/lys/megatron-swift/outputs/tb226_flash_next_lora_4node_212k_qsa_thd_overlay_formal'
assert config['dataset'] == expected_dataset
assert os.path.getsize(expected_dataset) == 8932384131, 'formal dataset size changed'
assert config['output_dir'] == expected_output
assert config['ENV']['QSA_SPARSE_KERNEL'] == 'True'
assert config['packing'] is False
assert config['padding_free'] is True
assert config['context_parallel_size'] == 1
assert config['sequence_parallel'] is True
assert config['apply_rope_fusion'] is False
assert config['micro_batch_size'] == 1
assert config['global_batch_size'] == 8
assert config['group_by_length'] is False
assert type(config['max_length']) is int and config['max_length'] == 212992
assert config['num_train_epochs'] == 1
assert 'train_iters' not in config
assert config['save_steps'] == 250 and config['save_total_limit'] == 3
assert config['no_save_optim'] is False and config['no_save_rng'] is False
assert not any('mem_probe.py' in p for p in config['external_plugins'])

import mcore_bridge
from mcore_bridge.model.modules import QSASparseCoreAttention, qsa_sparse_supported

source = Path(inspect.getfile(mcore_bridge)).resolve()
expected_root = (Path(overlay) / 'src').resolve()
assert source.is_relative_to(expected_root), f'mcore_bridge loaded from {source}, not {expected_root}'
assert qsa_sparse_supported(256), 'QSA sparse kernel does not support head_dim=256'
print(f'[validate] dataset={expected_dataset} bytes={os.path.getsize(expected_dataset)}')
print(f'[validate] mcore_bridge={source}')
print(f'[validate] qsa_core={inspect.getfile(QSASparseCoreAttention)}')
print('[validate] packing=false padding_free=true group_by_length=false CP=1 SP=true '
      'max_length=212992 epochs=1 recoverable_checkpoints=true')
PY

echo "[launch] host=$ME rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT dry_run=$DRY_RUN" | tee -a "$LOG"
echo "[launch] config=$CONFIG overlay=$actual_commit" | tee -a "$LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[launch] DRY_RUN=1: validation complete; formal training was not started." | tee -a "$LOG"
  exit 0
fi
[[ "$DRY_RUN" == "0" ]] || { echo "ABORT: DRY_RUN must be 0 or 1"; exit 1; }
[[ "$START_FORMAL_TRAINING" == "YES_I_CONFIRM" ]] || {
  echo "ABORT: set START_FORMAL_TRAINING=YES_I_CONFIRM on all four nodes"
  exit 1
}

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
[[ "$busy" -eq 0 ]] || {
  echo "ABORT: $busy GPU(s) busy on $ME" | tee -a "$LOG"
  nvidia-smi | tee -a "$LOG"
  exit 1
}

if [[ -f $ROOT/multinode/secrets.env ]]; then
  set -a
  source "$ROOT/multinode/secrets.env"
  set +a
fi

{
  echo "[launch] NCCL_IB_HCA=$NCCL_IB_HCA NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
} | tee -a "$LOG"

cd "$ROOT"
megatron sft "$CONFIG" 2>&1 | tee -a "$LOG"
