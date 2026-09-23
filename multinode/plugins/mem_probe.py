"""Smoke-only probe: after every train step print, from EVERY rank, the true
peak allocated vs reserved CUDA memory (swift's 'memory(GiB)' is max_reserved
on the last rank only, which says nothing about per-stage headroom)."""
import os
import torch
from swift.megatron.trainers.base import BaseMegatronTrainer

_orig = BaseMegatronTrainer.train_step


def _probed(self, *args, **kwargs):
    out = _orig(self, *args, **kwargs)
    if torch.cuda.is_available():
        print(f"[memprobe] rank={os.environ.get('RANK', '?')} host={os.uname().nodename.split('.')[0]} "
              f"max_alloc={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB "
              f"max_reserved={torch.cuda.max_memory_reserved() / 2**30:.1f}GiB", flush=True)
    return out


if not getattr(BaseMegatronTrainer.train_step, "_memprobe", False):
    _probed._memprobe = True
    BaseMegatronTrainer.train_step = _probed
