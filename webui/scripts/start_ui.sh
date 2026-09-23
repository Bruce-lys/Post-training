#!/usr/bin/env bash
# Start the Gradio UI on this node in the background. usage: start_ui.sh [port]
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST=$(hostname)
mkdir -p "$ROOT/logs" "$ROOT/run"
PIDF="$ROOT/run/ui.$HOST.pid"
if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "UI already running on $HOST pid $(cat "$PIDF") port $(cat "$ROOT/run/ui.$HOST.port" 2>/dev/null)"; exit 0
fi
cd "$ROOT"
PORT=${1:-$("$ROOT/.venv/bin/python" -c 'from core.settings import load_ui_settings; print(load_ui_settings().port)' 2>/dev/null || echo 17870)}
for _ in $(seq 0 9); do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$PORT\$"; then PORT=$((PORT+1)); else break; fi
done
export GRADIO_TEMP_DIR="/tmp/mswift_webui_${USER:-hera}"
export GRADIO_ANALYTICS_ENABLED=0 HF_HUB_OFFLINE=1
mkdir -p "$GRADIO_TEMP_DIR"
nohup "$ROOT/.venv/bin/python" app.py --port "$PORT" ${MSWIFT_WEBUI_AUTH:---no-auth} >"$ROOT/logs/ui.$HOST.log" 2>&1 &
echo $! > "$PIDF"; echo "$PORT" > "$ROOT/run/ui.$HOST.port"
sleep 4
if kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "Megatron-SWIFT WebUI started on $HOST: http://$(hostname -I | awk '{print $1}'):$PORT  (pid $(cat "$PIDF"), log logs/ui.$HOST.log)"
else
  echo "UI failed to start; see logs/ui.$HOST.log" >&2; tail -20 "$ROOT/logs/ui.$HOST.log"; exit 1
fi
