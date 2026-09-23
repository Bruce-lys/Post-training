#!/usr/bin/env bash
# Serve the merged agens-3.0-flash identity LoRA model with SGLang (official Qwen3.8-Flash-Next H200 low-latency recipe).
# Usage: bash swift_env/serve_identity_v1.sh {start|stop|status|logs}   (env: GPUS=4,5,6,7 PORT=30000 CONTAINER=agens-identity-sglang)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT}/merged/agens-3.0-flash-identity-v1}"
IMAGE="${IMAGE:-lmsysorg/sglang:qwen38flashnext}"
CONTAINER="${CONTAINER:-agens-identity-sglang}"
GPUS="${GPUS:-4,5,6,7}"; PORT="${PORT:-30000}"; SERVED_NAME="${SERVED_NAME:-agens-3.0-flash}"
CACHE_DIR="${ROOT}/.cache/sglang"; mkdir -p "$CACHE_DIR"/{huggingface,torch,triton,tmp}
case "${1:-start}" in
  stop)   docker rm -f "$CONTAINER" 2>/dev/null || true; echo "Stopped $CONTAINER"; exit 0 ;;
  status) docker ps -a --filter "name=^/${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}'; curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" || echo "(not ready)"; exit 0 ;;
  logs)   docker logs -f "$CONTAINER"; exit 0 ;;
  start)  ;;
esac
# docker on these nodes only sees the GPFS through /home/$USER/kwkj-k8s, not /kwkj-k8s (see cainn start_sglang.sh)
bind_src() { local p="$1"; local alt="/home/${USER:-hera}/kwkj-k8s${p#/kwkj-k8s}"; if [[ "$p" == /kwkj-k8s/* && -e "$alt" ]]; then printf '%s' "$alt"; else printf '%s' "$p"; fi; }
MODEL_BIND="$(bind_src "$MODEL_DIR")"; CACHE_BIND="$(bind_src "$CACHE_DIR")"
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="${GPUS}" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --network host --ipc=host --shm-size=64g \
  -v "${MODEL_BIND}:/model:ro" -v "${CACHE_BIND}:/workspace/cache" \
  -e HF_HOME=/workspace/cache/huggingface -e TORCH_HOME=/workspace/cache/torch -e TRITON_CACHE_DIR=/workspace/cache/triton \
  -e TMPDIR=/workspace/cache/tmp -e PYTHONUNBUFFERED=1 \
  "$IMAGE" python3 -m sglang.launch_server \
  --model-path /model --served-model-name "$SERVED_NAME" --trust-remote-code \
  --tp 4 --mem-fraction-static 0.85 --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer --linear-attn-verify-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --max-running-requests 96 --reasoning-parser auto --host 0.0.0.0 --port "$PORT"
echo "started $CONTAINER on GPUs $GPUS port $PORT; logs: bash $0 logs"
