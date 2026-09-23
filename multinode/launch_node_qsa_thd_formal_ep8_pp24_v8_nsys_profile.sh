#!/usr/bin/env bash
# Four-node Nsight Systems PROFILING launcher on the v8 formal recipe (2026-09-20).
# = launch_node_qsa_thd_formal_ep8_pp24_v8.sh (same overlay commit, same validation) + the nsys wrapper from
#   launch_node_qsa_nopack_nsys_smoke.sh. The config must be the v8 formal YAML except for the profiling-only keys
#   (train_iters=10, finetune=true, separate output_dir, save off, + nsys_iteration_capture.py); this script diffs the
#   two YAMLs and aborts on any other difference, so the profiled step is the formal step.
# nsys runs only on NSYS_NODES (default: node rank 0 = PP stage 0 / DP0, node rank 2 = PP stage 1 / DP0); the capture
# window is gated by cudaProfilerApi from plugins/nsys_iteration_capture.py: steps NSYS_CAPTURE_STEP..NSYS_CAPTURE_STEP_END
# (default 6..6, i.e. one steady-state step after three warm-up steps). Validation-only unless explicitly confirmed.
# On EACH of the four nodes (host0 = master), same TS on all nodes:
#   NODES="h0 h1 h2 h3" TS=<tag> DRY_RUN=1 bash launch_node_qsa_thd_formal_ep8_pp24_v8_nsys_profile.sh
#   NODES="h0 h1 h2 h3" TS=<tag> DRY_RUN=0 START_NSYS_PROFILE=YES_I_CONFIRM bash launch_node_qsa_thd_formal_ep8_pp24_v8_nsys_profile.sh
# Results: multinode/profiles/<TAG>/<TS>/nsys.node{0,2}.nsys-rep + .stats.txt (cuda_gpu_kern_sum, cuda_api_sum, nvtx_sum, osrt_sum).
set -euo pipefail

ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
MULTINODE=$ROOT/multinode
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
OVERLAY=$MULTINODE/overlays/mcore-bridge-9d610ffb
EXPECTED_OVERLAY_COMMIT=8417fcf510ada1b88b1052aaa2581638f628fe3a   # qsa-bitmap-chunked (same as formal v8)
FORMAL_CONFIG=$MULTINODE/tb226_flash_next_lora_4node_212k_qsa_thd_formal_ep8_pp24_v8.yaml
CONFIG="${CONFIG:-$MULTINODE/tb226_flash_next_lora_4node_212k_qsa_thd_formal_ep8_pp24_v8_nsys_profile.yaml}"
NODES="${NODES:?set NODES=\"host0 host1 host2 host3\" (host0 = master)}"
MASTER_PORT="${MASTER_PORT:-29660}"
NSYS_NODES="${NSYS_NODES:-}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt}"
export NSYS_CAPTURE_STEP="${NSYS_CAPTURE_STEP:-6}"
export NSYS_CAPTURE_STEP_END="${NSYS_CAPTURE_STEP_END:-$NSYS_CAPTURE_STEP}"
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"   # stop = keep training after cudaProfilerStop (stop-shutdown would kill the job)
DRY_RUN="${DRY_RUN:-1}"
START_NSYS_PROFILE="${START_NSYS_PROFILE:-NO}"
TAG="${TAG:-$(basename "$CONFIG" .yaml)}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

read -r -a NODE_ARR <<<"$NODES"
export NNODES=${#NODE_ARR[@]}
[[ "$NNODES" -eq 4 ]] || { echo "ABORT: the formal recipe requires exactly four nodes"; exit 1; }

ME=$(hostname -s)
export NODE_RANK=-1
for i in "${!NODE_ARR[@]}"; do
  [[ "${NODE_ARR[$i]}" == "$ME" ]] && NODE_RANK=$i
done
[[ "$NODE_RANK" -ge 0 ]] || { echo "ABORT: $ME not in NODES=[$NODES]"; exit 1; }

export MASTER_ADDR
MASTER_ADDR=$(getent hosts "${NODE_ARR[0]}" | awk '{print $1; exit}')
export MASTER_PORT

LOG_DIR=$MULTINODE/logs/$TAG/$TS
PROFILE_DIR=$MULTINODE/profiles/$TAG/$TS
mkdir -p "$LOG_DIR" "$PROFILE_DIR"
[[ -n "$NSYS_NODES" ]] || NSYS_NODES="${NODE_ARR[0]} ${NODE_ARR[2]}"
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

for key, value in (yaml.safe_load(open(sys.argv[1])).get('ENV') or {}).items():
    print(f'{key}={value}')
PY
)
[[ "$DRY_RUN" == "1" ]] && export CUDA_VISIBLE_DEVICES=""   # validation must not touch GPUs that may belong to a running job

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

python - "$CONFIG" "$FORMAL_CONFIG" "$OVERLAY" <<'PY' | tee -a "$LOG"
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path

import yaml

config_path, formal_path, overlay = sys.argv[1:]
config = yaml.safe_load(open(config_path))
formal = yaml.safe_load(open(formal_path))
plugin_dir = Path('/kwkj-k8s/llm_team/lys/megatron-swift/multinode/plugins')

# 1. The profiling config is the formal config except for these keys (all profiling-only, none touch model math).
ALLOWED_DIFF = {'ENV', 'output_dir', 'tensorboard_dir', 'finetune', 'train_iters', 'num_train_epochs',
                'save_steps', 'save_total_limit', 'no_save_optim', 'no_save_rng', 'swanlab_exp_name', 'external_plugins'}
ALLOWED_ENV_DIFF = {'PROFILE_DISABLE_SAVE', 'CUDA_CACHE_RELEASE_LOG_EVERY'}
diff = sorted(k for k in set(config) | set(formal) if config.get(k, '<absent>') != formal.get(k, '<absent>'))
unexpected = [k for k in diff if k not in ALLOWED_DIFF]
assert not unexpected, f'profiling config differs from formal v8 in non-profiling keys: {unexpected}'
env, fenv = config['ENV'], formal['ENV']
env_diff = sorted(k for k in set(env) | set(fenv) if env.get(k, '<absent>') != fenv.get(k, '<absent>'))
unexpected_env = [k for k in env_diff if k not in ALLOWED_ENV_DIFF]
assert not unexpected_env, f'profiling ENV differs from formal v8 in non-profiling keys: {unexpected_env}'
assert set(env) >= set(fenv), 'formal ENV keys are missing from the profiling ENV'

# 2. The profiling-only keys have exactly the intended values.
assert config['finetune'] is True, 'profiling loads adapter weights only (iteration 0); resuming would fight train_iters'
assert config['mcore_adapter'] == formal['mcore_adapter']
assert open(config['mcore_adapter'] + '/latest_checkpointed_iteration.txt').read().strip() == '250'
assert os.path.isdir(config['mcore_adapter'] + '/iter_0000250')
assert type(config['train_iters']) is int and 6 <= config['train_iters'] <= 20, config['train_iters']
assert 'num_train_epochs' not in config
assert 'nsys_profile' in config['output_dir'] and config['output_dir'] != formal['output_dir']
assert config['tensorboard_dir'].startswith(config['output_dir'])
assert config['save_steps'] == 1000000 and config['no_save_optim'] is True and config['no_save_rng'] is True
assert env['PROFILE_DISABLE_SAVE'] == 1
formal_plugins = list(formal['external_plugins'])
assert config['external_plugins'][:len(formal_plugins)] == formal_plugins, 'formal plugin list/order changed'
extra = config['external_plugins'][len(formal_plugins):]
assert [Path(p).name for p in extra] == ['nsys_iteration_capture.py'], extra
for p in config['external_plugins']:
    assert os.path.isfile(p), p
names = [Path(p).name for p in config['external_plugins']]
for banned in ('iteration_timing', 'pp_split_timing', 'pipeline_timing', 'mem_probe', 'mem_snapshot', 'dp_token_balance', 'packed_ple_varlen'):
    assert not any(banned in n for n in names), banned
start, end = int(os.environ['NSYS_CAPTURE_STEP']), int(os.environ['NSYS_CAPTURE_STEP_END'])
assert 4 <= start <= end < config['train_iters'], f'capture window {start}..{end} must be after warm-up (>=4) and before the last step'
assert end - start + 1 <= 2, 'nsys buffers ~30 GB per captured step; capture at most two steps'

# 3. The formal invariants (copied from launch_node_qsa_thd_formal_ep8_pp24_v8.sh).
expected_dataset = '/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl'
assert config['dataset'] == expected_dataset
assert os.path.getsize(expected_dataset) == 8932384131, 'formal dataset size changed'
assert env['QSA_SPARSE_KERNEL'] == 'True'
assert env['PYTORCH_CUDA_ALLOC_CONF'].startswith('expandable_segments:True')
assert str(env.get('DP_BALANCE_ORDER')) == 'True'
assert 'DP_TOKEN_BALANCE' not in env
lengths_file = env['DP_BALANCE_LENGTHS_FILE']
assert lengths_file == '/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_v3_lengths_by_row.shuffled_seed42.json', lengths_file
_lengths_doc = json.load(open(lengths_file, encoding='utf-8'))
assert _lengths_doc['source'] == expected_dataset, _lengths_doc['source']
_rows = sum(1 for _ in open(expected_dataset, encoding='utf-8'))
assert len(_lengths_doc['lengths']) == _rows, (len(_lengths_doc['lengths']), _rows)
import numpy as _np
from swift.utils.np_utils import get_seed as _get_seed
assert _lengths_doc['shuffle']['hf_seed'] == int(_get_seed(_np.random.RandomState(int(config.get('data_seed', 42))))), 'lengths file must be in swift shuffled order for this data_seed'
assert _lengths_doc['shuffle']['n'] == _rows
assert config['packing'] is False and config['padding_free'] is True
assert config['tensor_model_parallel_size'] == 8 and config['pipeline_model_parallel_size'] == 2
assert config['context_parallel_size'] == 1 and config['expert_model_parallel_size'] == 8
assert config['expert_tensor_parallel_size'] == 1 and config['sequence_parallel'] is True
assert config['decoder_first_pipeline_num_layers'] == 24
assert config['optimizer_cpu_offload'] is False and 'optimizer_offload_fraction' not in config
assert config['apply_rope_fusion'] is False
assert config['micro_batch_size'] == 1 and config['global_batch_size'] == 8
assert config['group_by_length'] is False and config.get('data_sharding') is False and config.get('dataset_shuffle') is True
assert type(config['max_length']) is int and config['max_length'] == 212992
assert config.get('cross_entropy_fusion_impl', 'native') == 'native' and 'megatron_extra_kwargs' not in config

_ccr = importlib.util.spec_from_file_location('cuda_cache_release', plugin_dir / 'cuda_cache_release.py')
_ccr_mod = importlib.util.module_from_spec(_ccr); _ccr.loader.exec_module(_ccr_mod); _ccr_mod.self_check()
_nic = importlib.util.spec_from_file_location('nsys_iteration_capture', plugin_dir / 'nsys_iteration_capture.py')
_nic_mod = importlib.util.module_from_spec(_nic); _nic.loader.exec_module(_nic_mod); _nic_mod.self_check()

import mcore_bridge
from mcore_bridge.model.modules import QSASparseCoreAttention, qsa_sparse_supported

source = Path(inspect.getfile(mcore_bridge)).resolve()
expected_root = (Path(overlay) / 'src').resolve()
assert source.is_relative_to(expected_root), f'mcore_bridge loaded from {source}, not {expected_root}'
assert qsa_sparse_supported(256), 'QSA sparse kernel does not support head_dim=256'

spec = importlib.util.spec_from_file_location('dp_balance_order', plugin_dir / 'dp_balance_order.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.self_check()
print(f'[validate] profiling config == formal v8 except {diff} / ENV {env_diff}')
print(f'[validate] dataset={expected_dataset} bytes={os.path.getsize(expected_dataset)} lengths_cache_rows={len(_lengths_doc["lengths"])}')
print(f'[validate] mcore_bridge={source}')
print(f'[validate] qsa_core={inspect.getfile(QSASparseCoreAttention)}')
print('[validate] QSA=true TP=8 PP=2 CP=1 EP=8 ETP=1 split=24/24 cpu_offload=off ce=native cache_release=on expandable_segments=on packing=false padding_free=true')
print(f'[validate] adapter=ckpt250(weights only, finetune=true) train_iters={config["train_iters"]} save=off nsys_capture_steps={start}..{end}')
PY

echo "[launch] host=$ME rank=$NODE_RANK/$NNODES master=$MASTER_ADDR:$MASTER_PORT dry_run=$DRY_RUN" | tee -a "$LOG"
echo "[launch] config=$CONFIG overlay=$actual_commit" | tee -a "$LOG"

PROFILE_THIS_NODE=0
for host in $NSYS_NODES; do
  [[ "$host" == "$ME" ]] && PROFILE_THIS_NODE=1
done
if [[ "$PROFILE_THIS_NODE" == "1" ]]; then
  command -v nsys >/dev/null || { echo "ABORT: nsys not found on $ME"; exit 1; }
  echo "[launch] nsys=$(command -v nsys) $(nsys --version 2>/dev/null | head -1)" | tee -a "$LOG"
fi
echo "[launch] nsys_this_node=$PROFILE_THIS_NODE nsys_nodes=[$NSYS_NODES] trace=$NSYS_TRACE capture_steps=$NSYS_CAPTURE_STEP..$NSYS_CAPTURE_STEP_END" | tee -a "$LOG"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[launch] DRY_RUN=1: validation complete; profiling run was not started." | tee -a "$LOG"
  exit 0
fi
[[ "$DRY_RUN" == "0" ]] || { echo "ABORT: DRY_RUN must be 0 or 1"; exit 1; }
[[ "$START_NSYS_PROFILE" == "YES_I_CONFIRM" ]] || {
  echo "ABORT: set START_NSYS_PROFILE=YES_I_CONFIRM on all four nodes"
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

{
  echo "[launch] NCCL_IB_HCA=$NCCL_IB_HCA NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
} | tee -a "$LOG"

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
if [[ "$PROFILE_THIS_NODE" == "1" ]]; then
  export NSYS_CONTROL_CUDA_PROFILER=1
  # nsys session buffers (~30 GB per captured step) go to GPFS scratch, not the node-local /tmp
  NSYS_SCRATCH="$MULTINODE/profiles/.nsys_tmp/${ME}_${TS}"
  mkdir -p "$NSYS_SCRATCH"
  export TMPDIR="$NSYS_SCRATCH"
  LOCAL_NSYS="$NSYS_SCRATCH/${TAG}_${TS}_node${NODE_RANK}"
  echo "[nsys] local_report=${LOCAL_NSYS}.nsys-rep capture_steps=$NSYS_CAPTURE_STEP..$NSYS_CAPTURE_STEP_END capture_range_end=$NSYS_CAPTURE_RANGE_END" | tee -a "$LOG"
  set +e
  nsys profile \
    --trace="$NSYS_TRACE" \
    --sample=none \
    --cpuctxsw=none \
    --backtrace=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end="$NSYS_CAPTURE_RANGE_END" \
    --force-overwrite=true \
    --output="$LOCAL_NSYS" \
    env TMPDIR=/tmp megatron sft "$CONFIG" 2>&1 | tee -a "$LOG"
  train_rc=${PIPESTATUS[0]}
  set -e
  report="${LOCAL_NSYS}.nsys-rep"
  if [[ -f "$report" ]]; then
    cp -f "$report" "$PROFILE_DIR/nsys.node${NODE_RANK}.nsys-rep"
    nsys stats --force-export=true \
      --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,osrt_sum \
      "$report" > "$PROFILE_DIR/nsys.node${NODE_RANK}.stats.txt" 2>&1 || true
    nsys export --type sqlite --force-overwrite true -o "$PROFILE_DIR/nsys.node${NODE_RANK}.sqlite" "$report" >/dev/null 2>&1 || echo "[nsys] sqlite export failed" | tee -a "$LOG"
    ls -lh "$report" "$PROFILE_DIR/nsys.node${NODE_RANK}.nsys-rep" "$PROFILE_DIR/nsys.node${NODE_RANK}.sqlite" 2>/dev/null | tee -a "$LOG"
    rm -rf "$NSYS_SCRATCH"
  else
    echo "[nsys] ERROR: report not found: $report" | tee -a "$LOG"
  fi
  exit "$train_rc"
else
  export NSYS_CONTROL_CUDA_PROFILER=0
  set +e
  megatron sft "$CONFIG" 2>&1 | tee -a "$LOG"
  train_rc=${PIPESTATUS[0]}
  set -e
  exit "$train_rc"
fi
