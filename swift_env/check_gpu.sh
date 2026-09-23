#!/usr/bin/env bash
# Preflight before megatron smoke/train — fail fast if GPU/torch mismatch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-swift"

if [[ -x "${VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

fail=0

echo "=== GPU preflight ==="
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L 2>/dev/null | head -3; then
    echo "[OK] nvidia-smi sees GPUs"
  else
    echo "[FAIL] nvidia-smi found but no GPU listed"
    fail=1
  fi
else
  echo "[FAIL] nvidia-smi not found — not on a GPU node?"
  fail=1
fi

python3 - <<'PY' || fail=1
import sys
import torch

print("torch", torch.__version__)
if "+cu130" in torch.__version__ or "+cu131" in torch.__version__:
    print("[FAIL] torch is cu130+; H200 nodes (driver 570 / CUDA 12.8) need cu128")
    print("       fix: bash swift_env/pin_torch_cu128.sh")
    sys.exit(1)
if not torch.cuda.is_available():
    print("[FAIL] torch.cuda.is_available()=False")
    print("       common causes: wrong torch build, or login node without GPU")
    print("       fix: bash swift_env/pin_torch_cu128.sh  (on GPU compute node)")
    sys.exit(1)
print("[OK] torch sees", torch.cuda.device_count(), "GPU(s)")
for i in range(min(8, torch.cuda.device_count())):
    print(" ", i, torch.cuda.get_device_name(i))
PY

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
echo "[done] GPU preflight PASS"
