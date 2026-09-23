#!/usr/bin/env bash
# Four-node QSA THD formal launcher v7 (v6 + overlay qsa-bitmap-chunked, bit-exact chunked selection_to_block_bitmap). Validation-only unless explicitly confirmed.
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
OVERLAY=$ROOT/multinode/overlays/mcore-bridge-9d610ffb
EXPECTED_OVERLAY_COMMIT=8417fcf510ada1b88b1052aaa2581638f628fe3a   # qsa-bitmap-chunked
CONFIG="${CONFIG:-$ROOT/multinode/tb226_flash_next_lora_4node_212k_qsa_thd_formal_ep8_pp24_v7.yaml}"
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
expected_output = '/kwkj-k8s/llm_team/lys/megatron-swift/outputs/tb226_flash_next_lora_4node_212k_qsa_thd_formal_ep8_pp24_v3'
plugin_dir = Path('/kwkj-k8s/llm_team/lys/megatron-swift/multinode/plugins')
assert config['dataset'] == expected_dataset
assert os.path.getsize(expected_dataset) == 8932384131, 'formal dataset size changed'
assert config['output_dir'] == expected_output
assert config['ENV']['QSA_SPARSE_KERNEL'] == 'True'
assert config['ENV']['PYTORCH_CUDA_ALLOC_CONF'].startswith('expandable_segments:True'), 'torch 2.9.1 reads PYTORCH_CUDA_ALLOC_CONF, not PYTORCH_ALLOC_CONF'
assert str(config['ENV'].get('DP_BALANCE_ORDER')) == 'True'
assert 'DP_TOKEN_BALANCE' not in config['ENV']
lengths_file = config['ENV']['DP_BALANCE_LENGTHS_FILE']
assert lengths_file == '/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_v3_lengths_by_row.json', lengths_file
import json
_lengths_doc = json.load(open(lengths_file, encoding='utf-8'))
assert _lengths_doc['source'] == expected_dataset, _lengths_doc['source']
_rows = sum(1 for _ in open(expected_dataset, encoding='utf-8'))
assert len(_lengths_doc['lengths']) == _rows, (len(_lengths_doc['lengths']), _rows)
assert config['packing'] is False
assert config['padding_free'] is True
assert config['tensor_model_parallel_size'] == 8
assert config['pipeline_model_parallel_size'] == 2
assert config['context_parallel_size'] == 1
assert config['expert_model_parallel_size'] == 8
assert config['expert_tensor_parallel_size'] == 1
assert config['sequence_parallel'] is True
assert config['decoder_first_pipeline_num_layers'] == 24
assert config['optimizer_cpu_offload'] is False
assert 'optimizer_offload_fraction' not in config
assert config['apply_rope_fusion'] is False
assert config['micro_batch_size'] == 1
assert config['global_batch_size'] == 8
assert config['group_by_length'] is False
assert config.get('data_sharding') is False
assert config.get('dataset_shuffle') is True
assert type(config['max_length']) is int and config['max_length'] == 212992
assert config['num_train_epochs'] == 1
assert config.get('cross_entropy_fusion_impl', 'native') == 'native' and 'megatron_extra_kwargs' not in config
assert config['finetune'] is False
assert config['mcore_adapter'] == '/kwkj-k8s/llm_team/lys/megatron-swift/outputs/tb226_flash_next_lora_4node_212k_qsa_thd_formal_ep8_pp24_v3/v0-20260918-013006/checkpoint-250'
assert open(config['mcore_adapter'] + '/latest_checkpointed_iteration.txt').read().strip() == '250'
assert os.path.isdir(config['mcore_adapter'] + '/iter_0000250')
assert any(path.endswith('cuda_cache_release.py') for path in config['external_plugins'])
import importlib.util as _ilu
_ccr = _ilu.spec_from_file_location('cuda_cache_release', plugin_dir / 'cuda_cache_release.py')
_ccr_mod = _ilu.module_from_spec(_ccr); _ccr.loader.exec_module(_ccr_mod); _ccr_mod.self_check()
assert 'train_iters' not in config
assert config['save_steps'] == 250 and config['save_total_limit'] == 3
assert config['no_save_optim'] is False and config['no_save_rng'] is False
assert any(path.endswith('ple_chunked.py') for path in config['external_plugins'])
assert any(path.endswith('dp_balance_order.py') for path in config['external_plugins'])
assert not any('dp_token_balance' in path for path in config['external_plugins'])
assert not any('iteration_timing' in path for path in config['external_plugins'])
assert not any('mem_probe.py' in path for path in config['external_plugins'])
assert not any('nsys' in path.lower() for path in config['external_plugins'])
assert not any('pp_split_timing.py' in path for path in config['external_plugins'])

import mcore_bridge
from mcore_bridge.model.modules import QSASparseCoreAttention, qsa_sparse_supported

source = Path(inspect.getfile(mcore_bridge)).resolve()
expected_root = (Path(overlay) / 'src').resolve()
assert source.is_relative_to(expected_root), f'mcore_bridge loaded from {source}, not {expected_root}'
assert qsa_sparse_supported(256), 'QSA sparse kernel does not support head_dim=256'

import importlib.util

plugin_path = plugin_dir / 'dp_balance_order.py'
spec = importlib.util.spec_from_file_location('dp_balance_order', plugin_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.self_check()
print(f'[validate] dataset={expected_dataset} bytes={os.path.getsize(expected_dataset)}')
print(f'[validate] mcore_bridge={source}')
print(f'[validate] qsa_core={inspect.getfile(QSASparseCoreAttention)}')
print('[validate] packing=false padding_free=true group_by_length=false data_sharding=false')
print('[validate] QSA=true TP=8 PP=2 CP=1 EP=8 ETP=1 split=24/24 cpu_offload=off ce=native cache_release=on expandable_segments=on(PYTORCH_CUDA_ALLOC_CONF) resume=ckpt250 dp_balance_order=on lengths_cache_rows=' + str(len(_lengths_doc['lengths'])))
print('[validate] max_length=212992 epochs=1 recoverable_checkpoints=true')
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
