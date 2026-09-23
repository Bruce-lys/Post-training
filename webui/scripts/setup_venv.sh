#!/usr/bin/env bash
# Create the UI-only venv (host python3.10 + virtualenv; host python lacks ensurepip).
# The training env (.venv-swift, py3.12) is NOT touched.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
cd "$ROOT"
python3 -m virtualenv --version >/dev/null 2>&1 || python3 -m pip install --user virtualenv -i "$PYPI_MIRROR"
[[ -x .venv/bin/python ]] || python3 -m virtualenv --always-copy .venv
.venv/bin/pip install -q -r requirements.txt -i "$PYPI_MIRROR"
.venv/bin/python -c "import sys, gradio, pandas, plotly; print('UI venv OK', sys.version.split()[0], 'gradio', gradio.__version__)"
