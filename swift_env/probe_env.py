#!/usr/bin/env python3
"""Verify Megatron-SWIFT + mcore-bridge env for qwen4_exp."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_MODEL = Path("/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next")


def _header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def step_runtime() -> None:
    _header("Runtime")
    print("python:", sys.executable)
    print("version:", sys.version.split()[0])


def step_cli() -> bool:
    _header("CLI")
    venv_bin = ROOT / ".venv-swift" / "bin"
    megatron = shutil.which("megatron") or (
        str(venv_bin / "megatron") if (venv_bin / "megatron").is_file() else None
    )
    swift = shutil.which("swift") or (
        str(venv_bin / "swift") if (venv_bin / "swift").is_file() else None
    )
    print("megatron:", megatron or "NOT FOUND")
    print("swift:", swift or "NOT FOUND")
    if not megatron:
        print("[FAIL] `megatron` CLI missing — run: bash swift_env/install.sh")
        return False
    print("[PASS] megatron CLI present")
    return True


def step_transformers() -> bool:
    _header("Transformers + qwen4_exp config")
    import transformers

    print("transformers:", transformers.__version__)
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(str(HF_MODEL), trust_remote_code=True)
        print("[PASS] AutoConfig loaded")
        print("  model_type:", getattr(cfg, "model_type", None))
        print("  architectures:", getattr(cfg, "architectures", None))
        return True
    except Exception as exc:
        print("[FAIL]", type(exc).__name__, exc)
        if "qwen4_exp" in str(exc):
            print("\nHint: need transformers >= 5.16.0 (qwen4_exp added in 5.16). Run:")
            print("  bash swift_env/upgrade_transformers.sh")
        return False


def step_mcore_bridge() -> bool:
    _header("mcore-bridge")
    try:
        import mcore_bridge  # noqa: F401

        print("[PASS] mcore_bridge import OK")
        try:
            from mcore_bridge import hf_to_mcore_config

            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(str(HF_MODEL), trust_remote_code=True)
            kwargs = hf_to_mcore_config(cfg)
            print("[PASS] hf_to_mcore_config keys:", sorted(kwargs.keys())[:12], "...")
            return True
        except Exception as exc:
            print("[WARN] hf_to_mcore_config:", type(exc).__name__, exc)
            return False
    except Exception as exc:
        print("[FAIL]", type(exc).__name__, exc)
        return False


def step_ms_swift() -> bool:
    _header("ms-swift")
    try:
        import swift  # noqa: F401

        from swift.version import __version__

        print("[PASS] ms-swift version:", __version__)
        return True
    except Exception as exc:
        print("[FAIL]", type(exc).__name__, exc)
        return False


def step_raw_config() -> None:
    _header("Model on disk")
    raw = json.loads((HF_MODEL / "config.json").read_text(encoding="utf-8"))
    tc = raw.get("text_config") or raw
    print("path:", HF_MODEL)
    print("model_type:", raw.get("model_type"))
    print("layers:", tc.get("num_hidden_layers"), "experts:", tc.get("num_experts"))


def main() -> int:
    if not HF_MODEL.is_dir():
        print(f"HF model missing: {HF_MODEL}", file=sys.stderr)
        return 2

    step_runtime()
    step_raw_config()
    ok_cli = step_cli()
    ok_tf = step_transformers()
    ok_mb = step_mcore_bridge() if ok_tf else False
    ok_swift = step_ms_swift()

    _header("Summary")
    print("megatron CLI:", "PASS" if ok_cli else "FAIL")
    print("transformers: ", "PASS" if ok_tf else "FAIL")
    print("mcore-bridge: ", "PASS" if ok_mb else "FAIL")
    print("ms-swift:     ", "PASS" if ok_swift else "FAIL")

    if ok_cli and ok_tf and ok_mb and ok_swift:
        print("\nNext: bash swift_env/megatron_sft_smoke.sh")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
