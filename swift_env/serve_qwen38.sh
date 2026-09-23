#!/usr/bin/env bash
# Serve any Qwen3.8-Flash-Next model (base or merged LoRA) with SGLang, using the official H200
# low-latency recipe from https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
#
# Usage:
#   MODEL_DIR=<hf dir> [GPUS=0,1,2,3] [PORT=30000] [CONTAINER=name] [SERVED_NAME=name] \
#     bash swift_env/serve_qwen38.sh {start|stop|status|logs|test}
#
# Node quirks handled automatically:
#   - docker only sees the GPFS through /home/$USER/kwkj-k8s, not /kwkj-k8s
#   - some nodes reject `--gpus` (nvidia-cuda-mps-control mount error) and need `--runtime=nvidia`
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_DIR="${MODEL_DIR:-/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
CONTAINER="${CONTAINER:-qwen38-sglang}"
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-30000}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-96}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.cache/sglang}"
TP="${TP:-$(awk -F',' '{print NF}' <<<"$GPUS")}"

case "${1:-start}" in
  stop)   docker rm -f "$CONTAINER" 2>/dev/null || true; echo "Stopped $CONTAINER"; exit 0 ;;
  status) docker ps -a --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}'
          curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" || echo "(not ready)"; exit 0 ;;
  logs)   docker logs -f "$CONTAINER"; exit 0 ;;
  test)   curl -s --max-time 180 "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"${2:-你好，用一句话介绍你自己}\"}],\"max_tokens\":512}"
          echo; exit 0 ;;
  start)  ;;
  *) echo "Usage: $0 {start|stop|status|logs|test}" >&2; exit 1 ;;
esac

[[ -f "$MODEL_DIR/config.json" ]] || { echo "Model not found: $MODEL_DIR/config.json" >&2; exit 1; }
mkdir -p "$CACHE_DIR"/{huggingface,torch,triton,tmp}

# GPFS is only visible to docker under /home/$USER/kwkj-k8s on these nodes
bind_src() {
  local p="$1" alt="/home/${USER:-hera}/kwkj-k8s${1#/kwkj-k8s}"
  if [[ "$p" == /kwkj-k8s/* && -e "$alt" ]]; then printf '%s' "$alt"; else printf '%s' "$p"; fi
}
MODEL_BIND="$(bind_src "$MODEL_DIR")"
CACHE_BIND="$(bind_src "$CACHE_DIR")"

# GPU passthrough: prefer the nvidia runtime (works on every node here); fall back to --gpus
GPU_ARGS=(--gpus "device=${GPUS}")
if docker info 2>/dev/null | grep -qw nvidia; then
  GPU_ARGS=(--runtime=nvidia -e "NVIDIA_VISIBLE_DEVICES=${GPUS}" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
fi

docker rm -f "$CONTAINER" 2>/dev/null || true
echo "Starting SGLang: model=$MODEL_DIR gpus=$GPUS tp=$TP port=$PORT served_as=$SERVED_NAME image=$IMAGE"
docker run -d --name "$CONTAINER" "${GPU_ARGS[@]}" \
  --network host --ipc=host --shm-size=64g \
  -v "${MODEL_BIND}:/model:ro" -v "${CACHE_BIND}:/workspace/cache" \
  -e HF_HOME=/workspace/cache/huggingface -e TORCH_HOME=/workspace/cache/torch \
  -e TRITON_CACHE_DIR=/workspace/cache/triton -e TMPDIR=/workspace/cache/tmp -e PYTHONUNBUFFERED=1 \
  "$IMAGE" python3 -m sglang.launch_server \
    --model-path /model \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    --tp "$TP" \
    --mem-fraction-static "$MEM_FRACTION" \
    --chunked-prefill-size 8192 \
    --linear-attn-prefill-backend flashinfer \
    --linear-attn-decode-backend flashinfer \
    --linear-attn-verify-backend triton \
    --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --reasoning-parser auto \
    --host 0.0.0.0 \
    --port "$PORT" >/dev/null

echo "Waiting for http://127.0.0.1:${PORT}/v1/models ..."
for _ in $(seq 1 480); do
  curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && {
    echo "Ready: http://$(hostname -I | awk '{print $1}'):${PORT}/v1   model=${SERVED_NAME}"; exit 0; }
  docker ps --filter "name=^/${CONTAINER}$" --format '{{.Status}}' | grep -q Up || {
    echo "Container exited. Last logs:" >&2; docker logs --tail 60 "$CONTAINER"; exit 1; }
  sleep 5
done
echo "timeout waiting for server" >&2; exit 1
