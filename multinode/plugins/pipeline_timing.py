"""Per-rank pipeline timing: microbatch forward/backward compute vs P2P wait, per step (measurement only).

Wraps megatron.core.pipeline_parallel.schedules.forward_step / backward_step (host time with a
torch.cuda.synchronize() at the end of each call, so the number is GPU compute incl. any collectives
issued inside the microbatch) and the P2PCommunicator send/recv methods (host time; these calls
block until the transfer completes, so they measure pipeline waiting). Prints one line per rank per
step:  [pp-timing] rank=R step=S fwd_n=.. fwd_s=.. bwd_n=.. bwd_s=.. p2p_n=.. p2p_s=.. p2p_max_s=..
Overhead: one device sync per microbatch fwd/bwd; steps are ~100 s so this is <1%.
Env: PP_TIMING=0 disables.
"""

from __future__ import annotations

import os
import time

import torch
from megatron.core.pipeline_parallel import p2p_communication as _p2p
from megatron.core.pipeline_parallel import schedules as _sched
from swift.megatron.trainers.base import BaseMegatronTrainer

_ENABLED = os.environ.get("PP_TIMING", "1") != "0"
_RANK = os.environ.get("RANK", "0")
_acc = {"fwd_n": 0, "fwd_s": 0.0, "bwd_n": 0, "bwd_s": 0.0, "p2p_n": 0, "p2p_s": 0.0, "p2p_max_s": 0.0}
_step = 0


_detail = {"fwd": [], "bwd": [], "p2p": [], "tok": []}


def _count_tokens(batch):
    """Token count of one microbatch from whatever the data iterator yields (dict / tuple / tensor)."""
    try:
        if isinstance(batch, dict):
            for key in ("input_ids", "tokens", "labels"):
                if key in batch and hasattr(batch[key], "numel"):
                    return int(batch[key].numel())
            for v in batch.values():
                if hasattr(v, "numel") and v.dim() >= 1 and v.dtype in (torch.int64, torch.int32):
                    return int(v.numel())
        elif isinstance(batch, (list, tuple)) and batch:
            return _count_tokens(batch[0])
        elif hasattr(batch, "numel"):
            return int(batch.numel())
    except Exception:
        pass
    return -1


class _CountingIterator:
    """Wraps the microbatch data iterator so each fetched batch's token count is recorded."""

    def __init__(self, it):
        self._it = it

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._it)
        _detail["tok"].append(_count_tokens(batch))
        return batch

    def __getattr__(self, name):
        return getattr(self._it, name)


def _timed(name, fn, sync):
    def wrapper(*args, **kwargs):
        if name == "fwd" and len(args) >= 2 and args[1] is not None and not isinstance(args[1], _CountingIterator):
            args = (args[0], _CountingIterator(args[1])) + tuple(args[2:])
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        _detail[name].append(dt)
        _acc[name + "_n"] += 1
        _acc[name + "_s"] += dt
        if name == "p2p" and dt > _acc["p2p_max_s"]:
            _acc["p2p_max_s"] = dt
        return out
    wrapper._pp_timing = True
    return wrapper


_original_run_train_step = BaseMegatronTrainer.run_train_step


def _run_train_step(self, *args, **kwargs):
    global _step
    for k in _acc:
        _acc[k] = 0 if k.endswith("_n") else 0.0
    for k in _detail:
        _detail[k] = []
    t0 = time.perf_counter()
    result = _original_run_train_step(self, *args, **kwargs)
    torch.cuda.synchronize()
    _step += 1
    total = time.perf_counter() - t0
    print("[pp-timing] rank=%s step=%d total_s=%.1f fwd_n=%d fwd_s=%.1f bwd_n=%d bwd_s=%.1f p2p_n=%d p2p_s=%.1f p2p_max_s=%.1f other_s=%.1f"
          % (_RANK, _step, total, _acc["fwd_n"], _acc["fwd_s"], _acc["bwd_n"], _acc["bwd_s"], _acc["p2p_n"], _acc["p2p_s"],
             _acc["p2p_max_s"], total - _acc["fwd_s"] - _acc["bwd_s"] - _acc["p2p_s"]), flush=True)
    if os.environ.get("PP_TIMING_DETAIL", "0") != "0":
        fmt = lambda xs: "[" + ",".join("%.1f" % x for x in xs) + "]"
        print("[pp-timing-detail] rank=%s step=%d tok=%s fwd=%s bwd=%s p2p=%s" % (_RANK, _step, _detail["tok"], fmt(_detail["fwd"]), fmt(_detail["bwd"]), fmt(_detail["p2p"])), flush=True)
    return result


if _ENABLED and not getattr(_sched.forward_step, "_pp_timing", False):
    _sched.forward_step = _timed("fwd", _sched.forward_step, sync=True)
    _sched.backward_step = _timed("bwd", _sched.backward_step, sync=True)
    # mcore 0.19: P2P goes through P2PCommunicator methods (schedules.py calls p2p_communicator.<method>)
    for fname in ("recv_forward", "recv_backward", "send_forward", "send_backward",
                  "send_forward_recv_backward", "send_backward_recv_forward",
                  "send_forward_recv_forward", "send_backward_recv_backward",
                  "send_forward_backward_recv_forward_backward"):
        if hasattr(_p2p.P2PCommunicator, fname):
            setattr(_p2p.P2PCommunicator, fname, _timed("p2p", getattr(_p2p.P2PCommunicator, fname), sync=True))
    _run_train_step._pp_timing = True
    BaseMegatronTrainer.run_train_step = _run_train_step
    if _RANK == "0":
        print("[pp-timing] enabled", flush=True)


def self_check() -> None:
    assert getattr(_sched.forward_step, "_pp_timing", False) and getattr(_sched.backward_step, "_pp_timing", False)
    assert getattr(_p2p.P2PCommunicator.recv_forward, "_pp_timing", False)
    assert getattr(_p2p.P2PCommunicator.send_forward_recv_backward, "_pp_timing", False)
