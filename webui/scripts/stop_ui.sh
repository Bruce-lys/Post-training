#!/usr/bin/env bash
# Stop the Gradio UI started by start_ui.sh on this node (training processes are untouched).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST=$(hostname)
PIDF="$ROOT/run/ui.$HOST.pid"
[[ -f "$PIDF" ]] || { echo "no pid file for $HOST"; exit 0; }
PID=$(cat "$PIDF")
if kill -0 "$PID" 2>/dev/null && tr '\0' ' ' < "/proc/$PID/cmdline" | grep -q "app.py"; then
  kill -TERM "$PID"; sleep 2; kill -0 "$PID" 2>/dev/null && kill -KILL "$PID"
  echo "stopped UI pid $PID"
else
  echo "pid $PID is not the UI (already gone?)"
fi
rm -f "$PIDF" "$ROOT/run/ui.$HOST.port"
