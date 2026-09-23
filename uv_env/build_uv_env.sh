#!/usr/bin/env bash
# Build a self-contained uv environment for lys/megatron-swift.
# Reuses prebuilt binaries (TE torch binding, flash_attn_3) from the cainn env instead of compiling.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERE="${ROOT}/uv_env"
VENV="${ROOT}/.venv-swift"
CAINN_SP=/kwkj-k8s/llm_team/cainn/megatron-swift/.venv-swift/lib/python3.12/site-packages
PROXY="${PROXY:-http://119.91.116.199:13128}"
TUNA=https://pypi.tuna.tsinghua.edu.cn/simple
TORCH_IDX=https://download.pytorch.org/whl/cu128

# everything uv produces stays inside the project
export UV_CACHE_DIR="${ROOT}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT}/.uv-python"
export UV_LINK_MODE=hardlink
export PATH="${ROOT}/bin:${HOME}/.local/bin:${PATH}"
export http_proxy="${PROXY}" https_proxy="${PROXY}" HTTP_PROXY="${PROXY}" HTTPS_PROXY="${PROXY}"
export no_proxy="localhost,127.0.0.1"

mkdir -p "${HERE}/logs" "${ROOT}/bin"
LOG="${HERE}/logs/build_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1
step(){ echo; echo "===== [$(date +%H:%M:%S)] $*"; }
fail(){ echo "STATE: FAILED at $*"; exit 1; }

step "0. uv binary -> ${ROOT}/bin/uv (copy; ~/.local/bin/uv is node-local)"
[[ -x "${ROOT}/bin/uv" ]] || cp "${HOME}/.local/bin/uv" "${ROOT}/bin/uv" || fail uv-copy
uv --version || fail uv

step "1. python 3.12.13 into ${UV_PYTHON_INSTALL_DIR}"
uv python install 3.12.13 || fail python-install
PY="${UV_PYTHON_INSTALL_DIR}/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
[[ -x "${PY}" ]] || fail python-path
echo "python=${PY}"

step "2. constraints from cainn freeze (drop git / flash_attn_3 / cu13 leftovers)"
grep -v -E '^(ms_swift|mcore_bridge|flash_attn_3)' "${ROOT}/.freeze_cainn.txt" \
  | grep -v -E '^nvidia-[a-z-]+(-cu13)?==(1[0-9]\.|9\.20|2\.29|3\.4\.5|0\.8\.1)' \
  > "${HERE}/constraints.txt"
wc -l "${HERE}/constraints.txt"

step "3. create venv ${VENV}"
if [[ -L "${VENV}" ]]; then rm "${VENV}"; fi
[[ -e "${VENV}" ]] && fail "venv-exists (${VENV} is a real dir; remove it manually first)"
uv venv "${VENV}" --python "${PY}" --seed || fail venv
grep -E '^home' "${VENV}/pyvenv.cfg"

step "4. uv pip install (PyPI mirror + pytorch cu128; git via proxy; 3 attempts)"
ok=0
for attempt in 1 2 3; do
  echo "[attempt ${attempt}]"
  uv pip install --python "${VENV}/bin/python" \
    --index-url "${TUNA}" --extra-index-url "${TORCH_IDX}" --index-strategy unsafe-best-match \
    -r "${HERE}/requirements.in" -c "${HERE}/constraints.txt" --override "${HERE}/overrides.txt" \
    && { ok=1; break; }
  echo "[attempt ${attempt} failed; retry in 20s]"; sleep 20
done
[[ ${ok} -eq 1 ]] || fail uv-pip-install

SP="${VENV}/lib/python3.12/site-packages"

step "5. reuse prebuilt transformer_engine_torch 2.14.1 from cainn (1 .so + dist-info)"
cp -a "${CAINN_SP}/transformer_engine/wheel_lib/transformer_engine_torch.cpython-312-x86_64-linux-gnu.so" "${SP}/transformer_engine/wheel_lib/" || fail te-so
cp -a "${CAINN_SP}/transformer_engine_torch-2.14.1.dist-info" "${SP}/" || fail te-distinfo

step "6. reuse prebuilt flash_attn_3 3.0.0 from cainn (1.6G)"
cp -a "${CAINN_SP}/flash_attn_3" "${CAINN_SP}/flash_attn_config.py" "${CAINN_SP}/flash_attn_interface.py" \
      "${CAINN_SP}/flash_attn_3-3.0.0-py3.12.egg-info" "${SP}/" || fail fa3-copy
rm -rf "${SP}/flash_attn_3/__pycache__"

step "7. verify"
"${VENV}/bin/python" - <<'PY'
import sys
print("python", sys.executable)
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
import transformers; print("transformers", transformers.__version__)
import swift; print("swift", swift.__version__)
import mcore_bridge; print("mcore_bridge ok")
import megatron.core; print("megatron.core", megatron.core.__version__)
import transformer_engine.pytorch as te; from importlib.metadata import version
print("TE", version("transformer-engine"), "LayerNormLinear", hasattr(te, "LayerNormLinear"))
import flash_attn_3; import flash_attn_interface; print("flash_attn_3 ok", flash_attn_interface.__file__)
import fla; print("fla ok")
import qwen_vl_utils, decord; print("qwen_vl_utils/decord ok")
PY
[[ $? -eq 0 ]] || fail verify
uv pip check --python "${VENV}/bin/python" || echo "[warn] uv pip check reported issues (see above)"
uv pip freeze --python "${VENV}/bin/python" > "${HERE}/freeze_lys.txt"
echo "--- freeze diff (cainn vs lys):"
diff <(sort "${ROOT}/.freeze_cainn.txt") <(sort "${HERE}/freeze_lys.txt") || true
echo "STATE: DONE"
