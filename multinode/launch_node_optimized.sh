#!/usr/bin/env bash
# 4 机启动器：在每台机上各执行一次（同一条命令），NODE_RANK 由主机名在 NODES 里的位置推导。
#   NODES="hd03-gpu2-0019 hd03-gpu2-0002 hd03-gpu2-0005 hd03-gpu2-0022" bash launch_node.sh
# 可选：
#   CONFIG=<yaml>  TAG=<名字>
#   SMOKE=1                 短序列冒烟（默认 SMOKE_LEN=16384 SMOKE_ITERS=5）
#   SMOKE_LEN=32768         测 QSA [s,s] mask：TP8+SP 下 16K 时 indexer 仍是 no-op，需 >16384
#   CP=2                    覆盖 context_parallel_size；>1 时 QSA 稀疏选择会被框架禁用
set -euo pipefail
ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
VENV=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift
CONFIG="${CONFIG:-$ROOT/multinode/tb226_flash_next_lora_4node_207k_optimized.yaml}"
NODES="${NODES:?set NODES=\"host0 host1 host2 host3\" (host0 = master)}"
MASTER_PORT="${MASTER_PORT:-29600}"
TAG="${TAG:-$(basename "$CONFIG" .yaml)}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

read -r -a NODE_ARR <<<"$NODES"
export NNODES=${#NODE_ARR[@]}
ME=$(hostname -s)
export NODE_RANK=-1
for i in "${!NODE_ARR[@]}"; do [[ "${NODE_ARR[$i]}" == "$ME" ]] && NODE_RANK=$i; done
[[ $NODE_RANK -ge 0 ]] || { echo "ABORT: $ME not in NODES=[$NODES]"; exit 1; }
export MASTER_ADDR=$(getent hosts "${NODE_ARR[0]}" | awk '{print $1; exit}')
export MASTER_PORT

LOG_DIR=$ROOT/multinode/logs/$TAG/$TS
mkdir -p "$LOG_DIR"
LOG=$LOG_DIR/train.node${NODE_RANK}.log
cp -f "$CONFIG" "$LOG_DIR/config.node${NODE_RANK}.yaml"

# GPU 必须全空
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
[[ "$busy" -eq 0 ]] || { echo "ABORT: $busy GPU(s) busy on $ME" | tee -a "$LOG"; nvidia-smi | tee -a "$LOG"; exit 1; }

# 导出 yaml 里的 ENV 块
source "$VENV/bin/activate"
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  export "${line%%=*}=${line#*=}"
done < <(python - "$CONFIG" <<'PY'
import sys, yaml
for k, v in (yaml.safe_load(open(sys.argv[1])).get("ENV") or {}).items():
    print(f"{k}={v}")
PY
)
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# swanlab 装在 venv 之外的 pyextra（不动 cainn 的共享 venv）；密钥放 multinode/secrets.env（chmod 600，不进 yaml）
export PYTHONPATH=$ROOT/pyextra${PYTHONPATH:+:$PYTHONPATH}
[[ -f $ROOT/multinode/secrets.env ]] && { set -a; source $ROOT/multinode/secrets.env; set +a; }

RUN_CFG="$CONFIG"
if [[ "${SMOKE:-0}" == "1" || -n "${CP:-}" ]]; then
  RUN_CFG=$LOG_DIR/run.yaml
  SMOKE="${SMOKE:-0}" SMOKE_LEN="${SMOKE_LEN:-16384}" SMOKE_ITERS="${SMOKE_ITERS:-5}" \
  MEMPROBE="${MEMPROBE:-0}" CP="${CP:-}" python - "$CONFIG" "$RUN_CFG" <<'PY' | tee -a "$LOG"
import os, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
if os.environ.get("SMOKE") == "1":
    sl = int(os.environ.get("SMOKE_LEN", "16384"))
    c.update(max_length=sl, train_iters=int(os.environ.get("SMOKE_ITERS", "5")), save_steps=1000000,
             output_dir=c["output_dir"] + "_smoke",
             tensorboard_dir=c["output_dir"] + "_smoke/tensorboard")
    if c.get("packing"):
        c["packing_length"] = sl
    c.pop("num_train_epochs", None)
    if os.environ.get("MEMPROBE") == "1":
        c["external_plugins"] = list(c.get("external_plugins", [])) + [
            "/kwkj-k8s/llm_team/lys/megatron-swift/multinode/plugins/mem_probe.py"
        ]
if os.environ.get("CP"):
    c["context_parallel_size"] = int(os.environ["CP"])
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
print(f"[launch] packing={c.get('packing')} padding_free={c.get('padding_free')} "
      f"cp={c.get('context_parallel_size')} max_length={c.get('max_length')}", flush=True)
if int(c.get("context_parallel_size") or 1) > 1:
    print("[launch] WARN: CP>1 disables QSA sparse selection in mcore-bridge "
          "(indexer needs keys from other CP ranks; TE cannot take a custom mask under CP).",
          flush=True)
PY
fi
python -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); assert (c["packing"], c["padding_free"], c["context_parallel_size"], c["group_by_length"]) == (False, False, 1, True) and type(c["max_length"]) is int, "QSA invariant failed"' "$RUN_CFG"

echo "[launch] QSA invariants validated: packing=false padding_free=false CP=1 group_by_length=true"
{
  echo "[launch] host=$ME NODE_RANK=$NODE_RANK/$NNODES MASTER=$MASTER_ADDR:$MASTER_PORT ts=$TS"
  echo "[launch] config=$RUN_CFG"
  echo "[launch] NCCL_IB_HCA=$NCCL_IB_HCA NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
} | tee -a "$LOG"

cd "$ROOT"
megatron sft "$RUN_CFG" 2>&1 | tee -a "$LOG"
