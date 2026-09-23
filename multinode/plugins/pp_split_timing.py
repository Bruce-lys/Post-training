"""Measure PP split smoke iterations and train-step work without Nsight."""

from __future__ import annotations

import os
import time

import torch
from swift.megatron.callbacks.default_flow import DefaultFlowCallback
from swift.megatron.trainers.base import BaseMegatronTrainer


_iteration = 0
_labels = [item.strip() for item in os.environ.get("PERF_STEP_LABELS", "W1,W2,M1,M2,M3").split(",")]
_split = os.environ.get("PP_FIRST_LAYERS", "?")
_emit_timing = int(os.environ.get("LOCAL_RANK", "-1")) == 0
_original_run_train_step = BaseMegatronTrainer.run_train_step
_original_train_step = BaseMegatronTrainer.train_step
_original_default_flow_on_step_end = DefaultFlowCallback.on_step_end


def _identity() -> tuple[str, str]:
    return os.environ.get("RANK", "?"), os.uname().nodename.split(".")[0]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_run_train_step(self, *args, **kwargs):
    global _iteration
    _iteration += 1
    _sync()
    started = time.perf_counter()
    try:
        return _original_run_train_step(self, *args, **kwargs)
    finally:
        _sync()
        rank, host = _identity()
        label = _labels[_iteration - 1] if _iteration <= len(_labels) else "?"
        if _emit_timing:
            print(
                f"[ppperf] step={_iteration} label={label} split={_split} "
                f"phase=FULL_ITERATION seconds={time.perf_counter() - started:.6f} "
                f"rank={rank} host={host}",
                flush=True,
            )


def _timed_train_step(self, *args, **kwargs):
    _sync()
    started = time.perf_counter()
    try:
        return _original_train_step(self, *args, **kwargs)
    finally:
        _sync()
        rank, host = _identity()
        label = _labels[_iteration - 1] if 0 < _iteration <= len(_labels) else "?"
        if _emit_timing:
            print(
                f"[ppperf] step={_iteration} label={label} split={_split} "
                f"phase=TRAIN_STEP seconds={time.perf_counter() - started:.6f} "
                f"rank={rank} host={host}",
                flush=True,
            )


def _disable_checkpoint(self, *args, **kwargs):
    result = _original_default_flow_on_step_end(self, *args, **kwargs)
    if os.environ.get("PROFILE_DISABLE_SAVE", "0") == "1":
        self.state.should_save = False
    return result


if not getattr(BaseMegatronTrainer.run_train_step, "_pp_split_timing", False):
    _timed_run_train_step._pp_split_timing = True
    BaseMegatronTrainer.run_train_step = _timed_run_train_step
    BaseMegatronTrainer.train_step = _timed_train_step
    DefaultFlowCallback.on_step_end = _disable_checkpoint
