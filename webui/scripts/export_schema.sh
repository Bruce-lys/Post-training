#!/usr/bin/env bash
# Export the megatron sft argument schema using the training venv (heavy import, ~1 min).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJ=$(cd "$ROOT/.." && pwd)
PY="${TRAIN_PYTHON:-$PROJ/.venv-swift/bin/python}"
mkdir -p "$ROOT/schema"
"$PY" "$ROOT/scripts/export_schema.py" "$ROOT/schema/megatron_sft_args.json"
