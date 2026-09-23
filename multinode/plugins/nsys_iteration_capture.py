"""Profile complete Swift/Megatron iterations and their major phases (measurement only, no change to model math).

cudaProfilerStart is issued at the start of step NSYS_CAPTURE_STEP and cudaProfilerStop at the end of step
NSYS_CAPTURE_STEP_END (default: the same step, i.e. a single-step window), but only when NSYS_CONTROL_CUDA_PROFILER=1
(set by the launcher on the nodes that run under `nsys profile --capture-range=cudaProfilerApi`). Steps are counted
per process from 1, independent of the checkpoint iteration. While the window is open every step gets an NVTX
FULL_ITERATION_<n> range and TRAIN_STEP / ON_LOG_TOTAL / PRINT_CALLBACK / TENSORBOARD_CALLBACK / PRINT_MEMORY_ALLREDUCE
sub-ranges. PROFILE_DISABLE_SAVE=1 suppresses checkpoint saving (should_save=False after every step).
2026-09-20: added NSYS_CAPTURE_STEP_END (backward compatible: unset = one step) and self_check().
"""

import os
import time

import torch
from swift.megatron.callbacks.print import PrintCallback
from swift.megatron.callbacks.tensorboard import TensorboardCallback
from swift.megatron.callbacks.default_flow import DefaultFlowCallback
from swift.megatron.trainers.base import BaseMegatronTrainer

import swift.megatron.callbacks.print as print_callback_module


_target = int(os.environ.get("NSYS_CAPTURE_STEP", "3"))
_target_end = int(os.environ.get("NSYS_CAPTURE_STEP_END", str(_target)))
assert _target_end >= _target, (_target, _target_end)
_control_profiler = os.environ.get("NSYS_CONTROL_CUDA_PROFILER", "0") == "1"
_iteration = 0
_capture_active = False


def _rank():
    return os.environ.get("RANK", "?")


def _host():
    return os.uname().nodename.split(".")[0]


def _timed_range(name, function, *args, **kwargs):
    started = time.perf_counter()
    if _capture_active:
        torch.cuda.nvtx.range_push(name)
    try:
        return function(*args, **kwargs)
    finally:
        if _capture_active:
            torch.cuda.nvtx.range_pop()
            print(
                f"[timing] step={_iteration} phase={name} seconds={time.perf_counter() - started:.6f} "
                f"rank={_rank()} host={_host()}",
                flush=True,
            )


_original_run_train_step = BaseMegatronTrainer.run_train_step
_original_train_step = BaseMegatronTrainer.train_step
_original_on_log = BaseMegatronTrainer.on_log
_original_print_on_log = PrintCallback.on_log
_original_tensorboard_on_log = TensorboardCallback.on_log
_original_default_flow_on_step_end = DefaultFlowCallback.on_step_end
_original_reduce_max = print_callback_module.reduce_max_stat_across_model_parallel_group


def _profiled_run_train_step(self, *args, **kwargs):
    global _capture_active, _iteration
    _iteration += 1
    cuda = torch.cuda.is_available()
    start_capture = _control_profiler and cuda and _iteration == _target
    stop_capture = _control_profiler and cuda and _iteration == _target_end
    started = time.perf_counter()

    if start_capture:
        torch.cuda.synchronize()
        status = torch.cuda.cudart().cudaProfilerStart()
        _capture_active = True
        print(
            f"[nsys] capture_start step={_iteration} rank={_rank()} host={_host()} status={status}",
            flush=True,
        )
    if _capture_active:
        torch.cuda.nvtx.range_push(f"FULL_ITERATION_{_iteration}")

    try:
        return _original_run_train_step(self, *args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"[timing] step={_iteration} phase=FULL_ITERATION seconds={elapsed:.6f} "
            f"rank={_rank()} host={_host()}",
            flush=True,
        )
        if _capture_active:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        if stop_capture and _capture_active:
            _capture_active = False
            status = torch.cuda.cudart().cudaProfilerStop()
            print(
                f"[nsys] capture_stop step={_iteration} rank={_rank()} host={_host()} status={status}",
                flush=True,
            )


def _profiled_train_step(self, *args, **kwargs):
    return _timed_range("TRAIN_STEP", _original_train_step, self, *args, **kwargs)


def _profiled_on_log(self, *args, **kwargs):
    return _timed_range("ON_LOG_TOTAL", _original_on_log, self, *args, **kwargs)


def _profiled_print_on_log(self, *args, **kwargs):
    return _timed_range("PRINT_CALLBACK", _original_print_on_log, self, *args, **kwargs)


def _profiled_tensorboard_on_log(self, *args, **kwargs):
    return _timed_range("TENSORBOARD_CALLBACK", _original_tensorboard_on_log, self, *args, **kwargs)


def _profiled_default_flow_on_step_end(self, *args, **kwargs):
    result = _original_default_flow_on_step_end(self, *args, **kwargs)
    if os.environ.get("PROFILE_DISABLE_SAVE", "0") == "1":
        self.state.should_save = False
    return result


def _profiled_reduce_max(*args, **kwargs):
    return _timed_range("PRINT_MEMORY_ALLREDUCE", _original_reduce_max, *args, **kwargs)


if not getattr(BaseMegatronTrainer.run_train_step, "_nsys_iteration_capture", False):
    _profiled_run_train_step._nsys_iteration_capture = True
    BaseMegatronTrainer.run_train_step = _profiled_run_train_step
    BaseMegatronTrainer.train_step = _profiled_train_step
    BaseMegatronTrainer.on_log = _profiled_on_log
    PrintCallback.on_log = _profiled_print_on_log
    TensorboardCallback.on_log = _profiled_tensorboard_on_log
    DefaultFlowCallback.on_step_end = _profiled_default_flow_on_step_end
    print_callback_module.reduce_max_stat_across_model_parallel_group = _profiled_reduce_max
    if _rank() == "0":
        print(
            f"[nsys] iteration capture plugin loaded: control_profiler={_control_profiler} "
            f"steps={_target}..{_target_end} disable_save={os.environ.get('PROFILE_DISABLE_SAVE', '0')}",
            flush=True,
        )


def self_check() -> None:
    assert getattr(BaseMegatronTrainer.run_train_step, "_nsys_iteration_capture", False)
    assert DefaultFlowCallback.on_step_end is _profiled_default_flow_on_step_end
