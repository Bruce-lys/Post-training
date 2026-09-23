#!/usr/bin/env bash
# Serve a Qwen3.8-Flash-Next (or merged fine-tune) with the cluster's SGLang image (recipe from cainn start_sglang.sh).
# Usage: MODEL_DIR=<hf dir> [GPUS=4,5 TP_SIZE=2 EP_SIZE=2 PORT=8014 CONTAINER=name] bash swift_env/serve_sglang.sh {start|stop|status|logs}
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
CONTAINER="${CONTAINER:-mswift-sglang}"
PORT="${PORT:-8014}"
GPUS="${GPUS:-4,5}"
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
MODEL_DIR="${MODEL_DIR:-/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next}"
SERVED_NAME="${SERVED_NAME:-agens-3.0-flash}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/.cache/sglang}"
cmd="${1:-start}"
case "$cmd" in
  stop)   docker rm -f "$CONTAINER" 2>/dev/null || true; echo "Stopped $CONTAINER"; exit 0 ;;
  status) docker ps -a --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'; curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" || echo "(not ready)"; exit 0 ;;
  logs)   docker logs -f "$CONTAINER"; exit 0 ;;
  start)  ;;
  *) echo "Usage: $0 {start|stop|status|logs}" >&2; exit 1 ;;
esac
[[ -f "$MODEL_DIR/config.json" ]] || { echo "Model not found: $MODEL_DIR" >&2; exit 1; }
mkdir -p "$CACHE_DIR"/{huggingface,torch,triton,tmp}
docker rm -f "$CONTAINER" 2>/dev/null || true
echo "Starting SGLang: image=$IMAGE model=$MODEL_DIR gpus=$GPUS tp=$TP_SIZE ep=$EP_SIZE port=$PORT served_as=$SERVED_NAME"
docker run -d --name "$CONTAINER" --gpus "\"device=${GPUS}\"" --network host --ipc=host --shm-size=64g \
  -v "${MODEL_DIR}:/model:ro" -v "${CACHE_DIR}:/workspace/cache" \
  -e HF_HOME=/workspace/cache/huggingface -e TORCH_HOME=/workspace/cache/torch \
  -e TRITON_CACHE_DIR=/workspace/cache/triton -e TMPDIR=/workspace/cache/tmp -e PYTHONUNBUFFERED=1 \
  "$IMAGE" python3 -m sglang.launch_server --model-path /model --served-model-name "$SERVED_NAME"   --host 0.0.0.0 --port "$PORT" --tp-size "$TP_SIZE" --ep-size "$EP_SIZE" --dtype bfloat16   --context-length "$CONTEXT_LENGTH" --mem-fraction-static "$MEM_FRACTION" --max-running-requests "$MAX_RUNNING_REQUESTS"   --chunked-prefill-size 8192 --schedule-policy fcfs --reasoning-parser qwen3 --tool-call-parser qwen3_coder   --enable-multimodal --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer   --trust-remote-code --mamba-ssm-dtype bfloat16
echo "Waiting for http://127.0.0.1:${PORT}/v1/models ..."
for _ in $(seq 1 360); do
  curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { echo "Ready: http://127.0.0.1:${PORT}/v1  model=$SERVED_NAME"; exit 0; }
  docker ps --filter "name=^/${CONTAINER}$" --format '{{.Status}}' | grep -q Up || { echo "Container exited. Last logs:" >&2; docker logs --tail 80 "$CONTAINER"; exit 1; }
  sleep 5
done
echo "timeout waiting for server" >&2; exit 1
echo "container $CONTAINER started; watch: bash $0 logs ; check: bash $0 status"
