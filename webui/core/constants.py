"""Static constants shared by the Megatron-SWIFT Web UI."""
from __future__ import annotations

import pathlib
import socket

WEBUI_ROOT = pathlib.Path(__file__).resolve().parents[1]        # .../megatron-swift/webui
PROJECT_ROOT = WEBUI_ROOT.parent                                 # .../megatron-swift
SHARED_MOUNT = "/kwkj-k8s"

DEFAULT_UI_PORT = 17870
UI_PORT_RANGE = 10
MASTER_PORT_BASE = 29600
MASTER_PORT_RANGE = 200

SETTINGS_DIR = WEBUI_ROOT / "settings"
PRESETS_DIR = WEBUI_ROOT / "presets"
REGISTRY_DIR = WEBUI_ROOT / "registry"
RUNS_DIR = REGISTRY_DIR / "runs"
EVENTS_FILE = REGISTRY_DIR / "events.jsonl"
LOGS_DIR = WEBUI_ROOT / "logs"
SCHEMA_FILE = WEBUI_ROOT / "schema" / "megatron_sft_args.json"

DEFAULT_MODEL = "/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"
DEFAULT_TRAIN_VENV = str(PROJECT_ROOT / ".venv-swift")
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "outputs")
DEFAULT_DATA_ROOT = str(PROJECT_ROOT / "swift_env" / "data")

RUN_STATUSES = ("launching", "running", "finished", "failed", "aborted", "unknown")
STATUS_ICON = {"running": "🟢", "launching": "🟡", "finished": "✅", "failed": "❌", "aborted": "⛔", "unknown": "❔"}
STATUS_LABEL = {"running": "运行中", "launching": "启动中", "finished": "已完成", "failed": "失败", "aborted": "已中止",
                "unknown": "未知"}


def hostname() -> str:
    return socket.gethostname()
