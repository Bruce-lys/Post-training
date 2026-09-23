"""Profile one complete Swift/Megatron iteration and its major phases."""

import os
import time

import torch
from swift.megatron.callbacks.print import PrintCallback
from swift.megatron.callbacks.tensorboard import TensorboardCallback
from swift.megatron.callbacks.default_flow import DefaultFlowCallback
from swift.megatron.trainers.base import BaseMegatronTrainer

import swift.megatron.callbacks.print as print_callback_module


_target = int(os.environ.get("NSYS_CAPTURE_STEP", "3"))
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
    capture = _control_profiler and _iteration == _target and torch.cuda.is_available()
    started = time.perf_counter()

    if capture:
        torch.cuda.synchronize()
        status = torch.cuda.cudart().cudaProfilerStart()
        _capture_active = True
        torch.cuda.nvtx.range_push(f"FULL_ITERATION_{_iteration}")
        print(
            f"[nsys] capture_start step={_iteration} rank={_rank()} host={_host()} status={status}",
            flush=True,
        )

    try:
        return _original_run_train_step(self, *args, **kwargs)
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"[timing] step={_iteration} phase=FULL_ITERATION seconds={elapsed:.6f} "
            f"rank={_rank()} host={_host()}",
            flush=True,
        )
        if capture:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
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
