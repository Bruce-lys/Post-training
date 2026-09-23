#!/usr/bin/env bash
# SGLang server for the forward check: same image/recipe as swift_env/serve_qwen38.sh but deterministic
# prefill settings: TP8, context 220k, NO speculative decoding, mamba/GDN state in float32 (matches the
# model config and the training-side GDN), one request at a time.
#   bash serve_fwd_check_sglang.sh start|stop|status|logs
set -euo pipefail
ROOT=/kwkj-k8s/llm_team/lys/megatron-swift
MODEL_DIR="${MODEL_DIR:-/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
CONTAINER="${CONTAINER:-qwen38-fwdcheck}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-30100}"
CONTEXT_LEN="${CONTEXT_LEN:-220000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.cache/sglang}"
TP="${TP:-$(awk -F',' '{print NF}' <<<"$GPUS")}"

case "${1:-start}" in
  stop)   docker rm -f "$CONTAINER" 2>/dev/null || true; echo "Stopped $CONTAINER"; exit 0 ;;
  status) docker ps -a --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}'
          curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" || echo "(not ready)"; exit 0 ;;
  logs)   docker logs -f "$CONTAINER"; exit 0 ;;
  start)  ;;
  *) echo "Usage: $0 {start|stop|status|logs}" >&2; exit 1 ;;
esac

[[ -f "$MODEL_DIR/config.json" ]] || { echo "Model not found: $MODEL_DIR/config.json" >&2; exit 1; }
mkdir -p "$CACHE_DIR"/{huggingface,torch,triton,tmp}
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000{c++} END{print c+0}')
[[ "$busy" -eq 0 ]] || { echo "ABORT: $busy GPU(s) busy on $(hostname -s)"; nvidia-smi; exit 1; }

bind_src() {
  local p="$1" alt="/home/${USER:-hera}/kwkj-k8s${1#/kwkj-k8s}"
  if [[ "$p" == /kwkj-k8s/* && -e "$alt" ]]; then printf '%s' "$alt"; else printf '%s' "$p"; fi
}
MODEL_BIND="$(bind_src "$MODEL_DIR")"
CACHE_BIND="$(bind_src "$CACHE_DIR")"
# docker needs the inner quotes for a multi-GPU device list: --gpus '"device=0,1,..."'
GPU_ARGS=(--gpus "\"device=${GPUS}\"")
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  GPU_ARGS=(--runtime=nvidia -e "NVIDIA_VISIBLE_DEVICES=${GPUS}" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
fi

docker rm -f "$CONTAINER" 2>/dev/null || true
echo "Starting SGLang (fwd-check): model=$MODEL_DIR gpus=$GPUS tp=$TP port=$PORT ctx=$CONTEXT_LEN image=$IMAGE"
docker run -d --name "$CONTAINER" "${GPU_ARGS[@]}" \
  --network host --ipc=host --shm-size=64g \
  -v "${MODEL_BIND}:/model:ro" -v "${CACHE_BIND}:/workspace/cache" \
  -e HF_HOME=/workspace/cache/huggingface -e TORCH_HOME=/workspace/cache/torch \
  -e TRITON_CACHE_DIR=/workspace/cache/triton -e TMPDIR=/workspace/cache/tmp -e PYTHONUNBUFFERED=1 \
  "$IMAGE" python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name Qwen3.8-Flash-Next \
    --trust-remote-code \
    --tp "$TP" \
    --context-length "$CONTEXT_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --chunked-prefill-size 8192 \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --linear-attn-verify-backend triton \
    --mamba-ssm-dtype float32 \
    --max-running-requests 1 \
    --disable-radix-cache \
    --host 0.0.0.0 \
    --port "$PORT" >/dev/null

echo "Waiting for http://127.0.0.1:${PORT}/v1/models ..."
for _ in $(seq 1 480); do
  curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && {
    echo "Ready: http://$(hostname -I | awk '{print $1}'):${PORT}"; exit 0; }
  docker ps --filter "name=^/${CONTAINER}$" --format '{{.Status}}' | grep -q Up || {
    echo "Container exited. Last logs:" >&2; docker logs --tail 80 "$CONTAINER"; exit 1; }
  sleep 5
done
echo "timeout waiting for server" >&2; exit 1
