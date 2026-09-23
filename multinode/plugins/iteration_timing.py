"""Measure every complete training iteration without enabling a profiler."""

import os
import time

import torch
from swift.megatron.callbacks.default_flow import DefaultFlowCallback
from swift.megatron.trainers.base import BaseMegatronTrainer


_iteration = 0
_labels = [item.strip() for item in os.environ.get("PERF_STEP_LABELS", "W,A,B,B,A").split(",")]
_original_run_train_step = BaseMegatronTrainer.run_train_step
_original_default_flow_on_step_end = DefaultFlowCallback.on_step_end


def _timed_run_train_step(self, *args, **kwargs):
    global _iteration
    _iteration += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        return _original_run_train_step(self, *args, **kwargs)
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        label = _labels[_iteration - 1] if _iteration <= len(_labels) else "?"
        host = os.uname().nodename.split(".")[0]
        print(
            f"[perf] step={_iteration} variant={label} phase=FULL_ITERATION "
            f"seconds={elapsed:.6f} rank={os.environ.get('RANK', '?')} host={host}",
            flush=True,
        )


def _disable_checkpoint(self, *args, **kwargs):
    result = _original_default_flow_on_step_end(self, *args, **kwargs)
    if os.environ.get("PROFILE_DISABLE_SAVE", "0") == "1":
        self.state.should_save = False
    return result


if not getattr(BaseMegatronTrainer.run_train_step, "_iteration_timing", False):
    _timed_run_train_step._iteration_timing = True
    BaseMegatronTrainer.run_train_step = _timed_run_train_step
    DefaultFlowCallback.on_step_end = _disable_checkpoint
