#!/usr/bin/env python3
"""Build the whole Gradio app without a browser and call the Train-tab handlers directly.

Catches component-construction errors and handler exceptions that pytest (core only) cannot see."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gradio as gr  # noqa: E402

import app  # noqa: E402
from core import param_schema as ps, presets  # noqa: E402
from core.settings import load_paths_env  # noqa: E402
from ui import tab_runs, tab_train, top  # noqa: E402

failures = 0


def check(name, fn):
    global failures
    try:
        out = fn()
        print(f"[OK] {name}: {str(out)[:160]!r}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {name}: {exc}")


demo = app.build()
print(f"[OK] app.build(): {len(demo.blocks)} blocks, {len(demo.fns)} event handlers")

with gr.Blocks():
    t = top.build()
    train = tab_train.build(t)
paths = load_paths_env()
model = paths["MODEL_ROOT"]
ds = f"{paths['DATA_ROOT']}/smoke_alpaca.jsonl"
defaults = [ps.defaults()[k] for k in train["edit_keys"]]

check("do_preview(defaults)", lambda: train["do_preview"]("smoke", model, "", ds, *defaults)[0].splitlines()[:3])
check("do_check(defaults)", lambda: train["do_check"]("smoke", model, "", ds, *defaults)[0].splitlines()[-1])
full = [("full" if k == "tuner_type" else v) for k, v in zip(train["edit_keys"], defaults)]
check("do_preview(full)", lambda: train["do_preview"]("smoke", model, "", ds, *full)[0].splitlines()[:4])
check("on_tuner(full)", lambda: train["on_tuner"]("full", 1e-4))
check("do_load(smoke_4gpu)", lambda: train["do_load"]("smoke_4gpu")[-1])
check("do_save(tmp)", lambda: train["do_save"]("ui-smoke-tmp", *defaults)[1])
presets.delete_preset("ui-smoke-tmp")
check("set_state_values(sft_lora_4gpu_8k)", lambda: len(train["set_state_values"](presets.builtin_state("sft_lora_4gpu_8k"))))
check("poll(empty)", lambda: train["poll"]("")[0])
runs = tab_runs.runs_frame()
check("runs_frame", lambda: runs.shape)
latest = runs["run_id"].iloc[0] if not runs.empty else ""
check("poll(latest run)", lambda: train["poll"](latest)[0])
check("runs details", lambda: tab_runs._details(latest)[0].splitlines()[0])
check("preview_dataset", lambda: tab_train.preview_dataset(ds)[0])
print("UI SMOKE", "FAILED" if failures else "PASSED")
sys.exit(1 if failures else 0)
