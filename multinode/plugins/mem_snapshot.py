"""Record the CUDA allocator history on selected ranks and dump a snapshot at the end of one step.

Measurement only (no effect on math). Env:
  MEM_SNAPSHOT_RANKS  comma list of global ranks (default "8,0")
  MEM_SNAPSHOT_STEP   dump after this 1-based step (default 1)
  MEM_SNAPSHOT_DIR    output dir (default /kwkj-k8s/llm_team/lys/megatron-swift/outputs/mem_snapshots)
  MEM_SNAPSHOT_MAX    max recorded events (default 2000000)
The snapshot (torch.cuda.memory._dump_snapshot) is replayed offline to attribute the peak by call stack.
"""

from __future__ import annotations

import os
import time

import torch
from swift.megatron.trainers.base import BaseMegatronTrainer

_RANK = os.environ.get("RANK", "0")
_RANKS = set(x.strip() for x in os.environ.get("MEM_SNAPSHOT_RANKS", "8,0").split(",") if x.strip())
_STEP = int(os.environ.get("MEM_SNAPSHOT_STEP", "1"))
_DIR = os.environ.get("MEM_SNAPSHOT_DIR", "/kwkj-k8s/llm_team/lys/megatron-swift/outputs/mem_snapshots")
_MAX = int(os.environ.get("MEM_SNAPSHOT_MAX", "2000000"))
_ACTIVE = _RANK in _RANKS
_step = 0
_original_run_train_step = BaseMegatronTrainer.run_train_step

if _ACTIVE and torch.cuda.is_available():
    torch.cuda.memory._record_memory_history(max_entries=_MAX)
    print(f"[mem-snapshot] rank={_RANK} recording allocator history (max_entries={_MAX}), dump after step {_STEP}", flush=True)


def _run_train_step(self, *args, **kwargs):
    global _step
    result = _original_run_train_step(self, *args, **kwargs)
    _step += 1
    if _ACTIVE and _step == _STEP:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"snapshot_rank{_RANK}_step{_step}_{time.strftime('%Y%m%d_%H%M%S')}.pickle")
        t0 = time.time()
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)
        gib = 2**30
        print(f"[mem-snapshot] rank={_RANK} step={_step} dumped {path} ({os.path.getsize(path) / 2**20:.0f} MiB, {time.time() - t0:.1f}s) "
              f"max_allocated={torch.cuda.max_memory_allocated() / gib:.1f}GiB max_reserved={torch.cuda.max_memory_reserved() / gib:.1f}GiB", flush=True)
        print(torch.cuda.memory_summary(abbreviated=True), flush=True)
    return result


if not getattr(BaseMegatronTrainer.run_train_step, "_mem_snapshot", False):
    _run_train_step._mem_snapshot = True
    BaseMegatronTrainer.run_train_step = _run_train_step


def self_check() -> None:
    assert getattr(BaseMegatronTrainer.run_train_step, "_mem_snapshot", False)
