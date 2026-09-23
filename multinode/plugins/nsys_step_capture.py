"""Start and stop Nsight Systems collection around one configured train step."""

import os

import torch
from swift.megatron.trainers.base import BaseMegatronTrainer


_original_train_step = BaseMegatronTrainer.train_step
_step = 0


def _profiled_train_step(self, *args, **kwargs):
    global _step
    _step += 1
    target = int(os.environ.get("NSYS_CAPTURE_STEP", "6"))
    enabled = os.environ.get("NSYS_CONTROL_CUDA_PROFILER", "0") == "1"
    capturing = enabled and _step == target and torch.cuda.is_available()

    if capturing:
        torch.cuda.synchronize()
        start_status = torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"TRAIN_STEP_{_step}")
        print(
            f"[nsys] capture_start step={_step} rank={os.environ.get('RANK', '?')} "
            f"host={os.uname().nodename.split('.')[0]} status={start_status}",
            flush=True,
        )

    try:
        return _original_train_step(self, *args, **kwargs)
    finally:
        if capturing:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            stop_status = torch.cuda.cudart().cudaProfilerStop()
            print(
                f"[nsys] capture_stop step={_step} rank={os.environ.get('RANK', '?')} "
                f"host={os.uname().nodename.split('.')[0]} status={stop_status}",
                flush=True,
            )


if not getattr(BaseMegatronTrainer.train_step, "_nsys_step_capture", False):
    _profiled_train_step._nsys_step_capture = True
    BaseMegatronTrainer.train_step = _profiled_train_step
