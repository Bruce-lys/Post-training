#!/usr/bin/env python3
"""Headless checks that exercise the launcher without the browser.

  headless_smoke.py validate          # build plans for every built-in preset (lora + full): no errors, argv sane
  headless_smoke.py fake [seconds]    # fake-train end to end: launch -> metrics appear -> abort -> registry -> no strays
                                      # then a second short run that finishes on its own -> status finished
"""
from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import command as cmd, launcher, monitor, param_schema as ps, presets, registry  # noqa: E402
from core.settings import UiSettings, load_paths_env  # noqa: E402


def validate() -> int:
    presets.ensure_builtin()
    paths = load_paths_env()
    ui = UiSettings(fake_train=False)
    failures = 0
    for name in presets.BUILTIN:
        for tuner in ("lora", "full"):
            state = {**presets.builtin_state(name), "tuner_type": tuner}
            if tuner == "full":
                state["lr"] = 1e-5
            p = launcher.plan(state, f"validate-{name}-{tuner}", paths["MODEL_ROOT"], f"{paths['DATA_ROOT']}/smoke_alpaca.jsonl", paths, ui)
            flags = cmd.parse_flags(p.argv)
            lora_flags = {"lora_rank", "lora_alpha", "lora_dropout", "target_modules", "merge_lora", "use_rslora"}
            ok = not p.errors and (bool(lora_flags & set(flags)) == (tuner == "lora")) and flags["tuner_type"] == [tuner]
            ok = ok and flags["add_version"] == ["false"] and flags["output_dir"] == [p.run_root]
            print(f"[{'OK' if ok else 'FAIL'}] {name}/{tuner}: {len(p.argv)} tokens, errors={p.errors}, warnings={len(p.warnings)}")
            if not ok:
                failures += 1
                print("   argv:", " ".join(p.argv))
    return 1 if failures else 0


def _wait(run_id: str, pred, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = registry.get(run_id) or {}
        if pred(doc):
            return True
        time.sleep(1)
    return False


def fake(seconds: float = 20.0) -> int:
    paths = load_paths_env()
    ui = UiSettings(fake_train=True, fake_interval=1.0)
    state = {**presets.builtin_state("smoke_4gpu"), "train_iters": int(seconds)}
    p = launcher.plan(state, "headless-fake", paths["MODEL_ROOT"], f"{paths['DATA_ROOT']}/smoke_alpaca.jsonl", paths, ui)
    ok, msg = launcher.launch(p)
    print(msg)
    if not ok:
        return 1
    if not _wait(p.run_id, lambda d: monitor.progress(d.get("run_root", "")).get("step", 0) >= 3, 30):
        print("FAIL: no metrics after 30s")
        return 1
    df = monitor.load_metrics(p.run_root)
    print(f"metrics rows={len(df)} cols={list(df.columns)}")
    print("abort:", launcher.abort(p.run_id))
    doc = registry.get(p.run_id)
    alive = monitor.pid_alive(monitor.read_pid(p.run_root), p.run_root)
    stray = monitor.find_procs(f"--out {p.run_root}")
    print(f"status={doc.get('status')} exit_code={monitor.exit_code(p.run_root)} alive={alive} stray={stray}")
    if doc.get("status") != "aborted" or alive or stray:
        return 1

    state["train_iters"] = 4
    p2 = launcher.plan(state, "headless-fake-finish", paths["MODEL_ROOT"], f"{paths['DATA_ROOT']}/smoke_alpaca.jsonl", paths, ui)
    ok, msg = launcher.launch(p2)
    print(msg)
    if not ok:
        return 1
    if not _wait(p2.run_id, lambda d: launcher.finalize_if_exited(d) is not None or d.get("status") == "finished", 40):
        print("FAIL: second run did not finish")
        return 1
    doc2 = registry.get(p2.run_id)
    print(f"second run status={doc2.get('status')} exit_code={doc2.get('exit_code')} ckpt={doc2.get('last_checkpoint')}")
    return 0 if doc2.get("status") == "finished" and doc2.get("exit_code") == 0 else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        sys.exit(validate())
    if mode == "fake":
        sys.exit(fake(float(sys.argv[2]) if len(sys.argv) > 2 else 20.0))
    print(__doc__)
    sys.exit(2)
