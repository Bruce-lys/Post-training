"""Named parameter presets stored as JSON files, plus the built-in ones derived from swift_env/*.sh."""
from __future__ import annotations

import json
import os
import re
import time

from . import param_schema as ps, registry
from .constants import PRESETS_DIR

# Built-in presets mirror the reference scripts (values not listed fall back to param_schema defaults):
#   smoke_4gpu / smoke_8gpu  <- swift_env/megatron_sft_smoke*.sh   (2 steps, no save)
#   sft_lora_8gpu            <- swift_env/megatron_sft.sh            (TP2 PP2 EP4, 2K, 3 epochs)
#   sft_lora_4gpu_8k         <- hy/qwen38_flash_next_lora train_8k_seed3407 (the only run verified end to end)
BUILTIN: dict[str, dict] = {
    "smoke_4gpu": {"cuda_visible_devices": "0,1,2,3", "tp": 4, "pp": 1, "ep": 4, "etp": 1, "cp": 1,
                   "global_batch_size": 4, "max_length": 512, "train_iters": 2, "num_train_epochs": 0,
                   "lr_warmup_fraction": 0.0, "save_steps": 9999, "eval_steps": 9999,
                   "decoder_first_pipeline_num_layers": 0, "dataset_num_proc": 2, "dataloader_num_workers": 2,
                   "use_precision_aware_optimizer": False},
    "smoke_8gpu": {"cuda_visible_devices": "0,1,2,3,4,5,6,7", "tp": 2, "pp": 2, "ep": 4, "etp": 1, "cp": 1,
                   "global_batch_size": 8, "max_length": 1024, "train_iters": 2, "num_train_epochs": 0,
                   "lr_warmup_fraction": 0.0, "save_steps": 9999, "eval_steps": 9999,
                   "decoder_first_pipeline_num_layers": 12, "dataset_num_proc": 2, "dataloader_num_workers": 2,
                   "use_precision_aware_optimizer": False},
    "sft_lora_8gpu": {},
    "sft_lora_4gpu_8k": {"cuda_visible_devices": "0,1,2,3", "tp": 4, "pp": 1, "ep": 4, "etp": 1, "cp": 1,
                         "global_batch_size": 8, "max_length": 8192, "num_train_epochs": 3, "train_iters": 0,
                         "seed": 3407, "decoder_first_pipeline_num_layers": 0},
}
BUILTIN_NOTES = {
    "smoke_4gpu": "4 卡 2 步冒烟（TP4/PP1/EP4，512 长度，不保存）",
    "smoke_8gpu": "8 卡 2 步冒烟（TP2/PP2/EP4，1024 长度，不保存）",
    "sft_lora_8gpu": "8 卡正式 LoRA SFT（= swift_env/megatron_sft.sh：TP2/PP2/EP4，2K，3 epochs）",
    "sft_lora_4gpu_8k": "4 卡 LoRA SFT 8K（= hy 已跑通的 train_8k_seed3407 配方：TP4/PP1/EP4，3 epochs）",
}


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-一-鿿]+", "-", name.strip()).strip("-")
    return s or "preset"


def builtin_state(name: str) -> dict:
    return ps.compute_derived({**ps.defaults(), **BUILTIN[name]})


def ensure_builtin() -> None:
    """Write the built-in presets once (users may edit/delete them afterwards)."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    for name in BUILTIN:
        path = PRESETS_DIR / f"{name}.json"
        if not path.is_file():
            save_preset(name, builtin_state(name), BUILTIN_NOTES.get(name, ""))


def list_presets() -> list[str]:
    if not PRESETS_DIR.is_dir():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def save_preset(name: str, state: dict, note: str = "") -> str:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESETS_DIR / f"{_slug(name)}.json"
    doc = {"name": name, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note, "state": state}
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path.stem


def load_preset(name: str) -> dict | None:
    path = PRESETS_DIR / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("state")


def preset_note(name: str) -> str:
    path = PRESETS_DIR / f"{name}.json"
    if not path.is_file():
        return ""
    return json.loads(path.read_text(encoding="utf-8")).get("note", "")


def delete_preset(name: str) -> bool:
    path = PRESETS_DIR / f"{name}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


def state_from_run(run_id: str) -> dict | None:
    doc = registry.get(run_id)
    return (doc or {}).get("state")
