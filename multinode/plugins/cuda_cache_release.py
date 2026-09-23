"""Release the CUDA caching-allocator pool at step boundaries (math-neutral). v2.

Why: the 2026-09-18 formal run (EP8/PP24, native fused cross-entropy) died at step 299 with
CUDA OOM on the last PP stage while only 49.4 GiB were live: 62.5 GiB were "reserved but
unallocated" cache fragments carried across steps, so a 23.55 GiB fp32 logits allocation for a
204k-token microbatch found just 19.4 GiB physically free. Variable-length padding_free
microbatches fragment the pool; torch.cuda.empty_cache() after the optimizer step (all streams
quiescent) hands the unused cached blocks back to the driver so the next step starts from
reserved ~= allocated.

v1 also released before every microbatch forward; that forced cudaFree/device syncs while
pipeline P2P was in flight, the pool re-grew with a different segment geometry, and at the end of
step 1 NCCL's lazy communicator creation hit cudaMalloc OOM on one rank (S1c smoke). v2 therefore
releases ONLY at step end and only from CUDA_CACHE_RELEASE_START_STEP (default 2) on, after NCCL
has created its communicators.

Hooks: BaseMegatronTrainer.run_train_step (after each optimizer step), same patch point as
iteration_timing.py. Env: CUDA_CACHE_RELEASE=0 disables; CUDA_CACHE_RELEASE_START_STEP (default 2);
CUDA_CACHE_RELEASE_MIN_GIB (default 4) skips the cudaFree when the pool has fewer cached bytes;
CUDA_CACHE_RELEASE_LOG_EVERY (default 50) prints allocator stats on rank 0/8/16/24 every N steps.
"""

from __future__ import annotations

import os

import torch
from swift.megatron.trainers.base import BaseMegatronTrainer

_ENABLED = os.environ.get("CUDA_CACHE_RELEASE", "1") != "0"
_START_STEP = int(os.environ.get("CUDA_CACHE_RELEASE_START_STEP", "2"))
_MIN_GIB = float(os.environ.get("CUDA_CACHE_RELEASE_MIN_GIB", "4"))
_LOG_EVERY = int(os.environ.get("CUDA_CACHE_RELEASE_LOG_EVERY", "50"))
_RANK = os.environ.get("RANK", "0")
_step = 0
_releases = 0
_original_run_train_step = BaseMegatronTrainer.run_train_step


def _run_train_step(self, *args, **kwargs):
    global _step, _releases
    result = _original_run_train_step(self, *args, **kwargs)
    _step += 1
    gib = 2**30
    if _ENABLED and _step >= _START_STEP and torch.cuda.is_available():
        if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() >= _MIN_GIB * gib:
            torch.cuda.empty_cache()
            _releases += 1
    if _LOG_EVERY and _step % _LOG_EVERY == 0 and _RANK in ("0", "8", "16", "24"):
        print(
            "[cuda-cache-release] rank=%s step=%d releases=%d allocated=%.1fGiB reserved=%.1fGiB "
            "max_allocated=%.1fGiB max_reserved=%.1fGiB"
            % (_RANK, _step, _releases, torch.cuda.memory_allocated() / gib, torch.cuda.memory_reserved() / gib,
               torch.cuda.max_memory_allocated() / gib, torch.cuda.max_memory_reserved() / gib),
            flush=True,
        )
    return result


if _ENABLED and not getattr(BaseMegatronTrainer.run_train_step, "_cuda_cache_release", False):
    _run_train_step._cuda_cache_release = True
    BaseMegatronTrainer.run_train_step = _run_train_step
    if _RANK == "0":
        print("[cuda-cache-release] v2 enabled: step-end only, start_step=%d min_gib=%s log_every=%d"
              % (_START_STEP, _MIN_GIB, _LOG_EVERY), flush=True)


def self_check() -> None:
    assert getattr(BaseMegatronTrainer.run_train_step, "_cuda_cache_release", False)
