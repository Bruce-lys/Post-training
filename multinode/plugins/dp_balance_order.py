"""DP token balance + pipeline-aware microbatch ordering, with lengths read from a cache file.

Extends the idea of dp_token_balance.py (same sampler patch point) with two changes:

1. Lengths come from a JSON cache (env ``DP_BALANCE_LENGTHS_FILE``: ``{"lengths": [...]}`` in
   dataset row order) instead of tokenizing every row at the first iteration, which took
   ~5 hours per launch with lazy_tokenize on the formal dataset.
2. After splitting each global batch into two equal-token DP halves, each half is ordered for
   1F1B pipeline parallelism: the pipeline's only bubbles are the FIRST microbatch's forward
   (stage 1 waits for stage 0) and the LAST microbatch's backward (stage 0 waits for stage 1),
   so the shortest samples go first and last and the longest in the middle:
   [shortest, longest, 2nd longest, ..., 2nd shortest].

Sampler contract (MegatronPretrainingRandomSampler, data_sharding=False, DP=2): the shuffled
global order is consumed as idx_range_total[dp_rank::dp_size], so even positions are DP0's
microbatches in order and odd positions are DP1's. This plugin rewrites every
global_batch_size window of that order before the stride is applied. It does not change
TP/PP/CP/EP, QSA, packing, or padding_free.

Disable with DP_BALANCE_ORDER=0. Requires data_sharding=false, group_by_length=false, DP=2,
global_batch_size divisible by 2*micro_batch_size, and a lengths file covering the dataset.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Iterable, Sequence

import torch
from swift.megatron.trainers.batch_sampler import MegatronPretrainingRandomSampler

_PATCH_ATTR = "_dp_balance_order_patch"
ENV_ENABLE = "DP_BALANCE_ORDER"
ENV_FILE = "DP_BALANCE_LENGTHS_FILE"
_cache: list[int] | None = None


def _enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "1") != "0"


def _should_log() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _global_batch_size() -> int:
    env_value = os.environ.get("DP_BALANCE_GBS")
    if env_value:
        return int(env_value)
    try:
        from megatron.training import get_args

        return int(get_args().global_batch_size)
    except Exception:
        return int(os.environ.get("GLOBAL_BATCH_SIZE", "8"))


def _lengths(total_samples: int) -> list[int]:
    global _cache
    if _cache is not None and len(_cache) >= total_samples:
        return _cache
    path = os.environ.get(ENV_FILE)
    if not path:
        raise RuntimeError(f"{ENV_FILE} is not set; dp_balance_order needs a lengths cache file")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lengths = [max(int(x), 0) for x in data["lengths"]]
    if len(lengths) < total_samples:
        raise RuntimeError(f"lengths file {path} has {len(lengths)} rows, dataset has {total_samples}")
    _cache = lengths
    return lengths


def balance_halves(indices: Sequence[int], lengths: Sequence[int]) -> tuple[list[int], list[int]]:
    """Split one window into two equal-size groups with the closest token totals (DP0, DP1)."""
    if len(indices) % 2 != 0:
        raise ValueError(f"window size must be even, got {len(indices)}")
    tokens = [int(lengths[i]) for i in indices]
    total = sum(tokens)
    half = len(indices) // 2
    best = min(
        itertools.combinations(range(len(indices)), half),
        key=lambda picked: abs(total - 2 * sum(tokens[p] for p in picked)),
    )
    dp0 = set(best)
    g0 = [indices[p] for p in range(len(indices)) if p in dp0]
    g1 = [indices[p] for p in range(len(indices)) if p not in dp0]
    return g0, g1


def order_for_pipeline(group: Sequence[int], lengths: Sequence[int]) -> list[int]:
    """[shortest, longest, 2nd longest, ..., 2nd shortest]: keeps both 1F1B bubbles short."""
    if len(group) <= 2:
        return sorted(group, key=lambda i: lengths[i])
    s = sorted(group, key=lambda i: lengths[i])
    return [s[0]] + list(reversed(s[2:])) + [s[1]]


def rebalance_windows(indices: Sequence[int], lengths: Sequence[int], window: int) -> list[int]:
    if window <= 0 or window % 2 != 0:
        raise ValueError(f"window must be a positive even integer, got {window}")
    out: list[int] = []
    usable = (len(indices) // window) * window
    for start in range(0, usable, window):
        g0, g1 = balance_halves(indices[start:start + window], lengths)
        g0, g1 = order_for_pipeline(g0, lengths), order_for_pipeline(g1, lengths)
        for a, b in zip(g0, g1):
            out.extend((a, b))
    out.extend(indices[usable:])
    return out


def _dp_gap(window: Sequence[int], lengths: Sequence[int]) -> int:
    totals = [0, 0]
    for p, i in enumerate(window):
        totals[p % 2] += int(lengths[i])
    return abs(totals[0] - totals[1])


def _iter_with_balance_order(self):
    skip = self.data_sharding or getattr(self, "group_by_length", False) or self.data_parallel_size != 2
    if not _enabled() or skip:
        if _enabled() and _should_log() and skip:
            print(
                "[dpbalance-order] skipped: need DP=2, data_sharding=false, group_by_length=false; "
                f"got dp={self.data_parallel_size} sharding={self.data_sharding} "
                f"group_by_length={getattr(self, 'group_by_length', False)}",
                flush=True,
            )
        yield from _original_iter(self)
        return

    global_batch_size = _global_batch_size()
    if global_batch_size % self.micro_batch_times_data_parallel_size != 0:
        raise RuntimeError(
            "dp_balance_order requires global_batch_size divisible by "
            f"micro_batch_size*dp_size={self.micro_batch_times_data_parallel_size}, got {global_batch_size}"
        )

    active_total_samples = self.total_samples - self.last_batch_size
    self.epoch = self.consumed_samples // active_total_samples
    current_epoch_samples = self.consumed_samples % active_total_samples
    assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

    full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
    full_bucket_offset = current_epoch_samples
    if self.shuffle:
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        idx_range_total = torch.randperm(full_bucket_size, generator=generator).tolist()
    else:
        idx_range_total = list(range(full_bucket_size))

    lengths = _lengths(self.total_samples)
    first = idx_range_total[:global_batch_size]
    before_gap = _dp_gap(first, lengths) if first else 0
    idx_range_total = rebalance_windows(idx_range_total, lengths, global_batch_size)
    first = idx_range_total[:global_batch_size]
    if _should_log() and first:
        print(
            f"[dpbalance-order] epoch={self.epoch} gbs={global_batch_size} lengths_file={os.environ.get(ENV_FILE)} "
            f"first_window_gap {before_gap}->{_dp_gap(first, lengths)} "
            f"dp0_mb_lengths={[lengths[i] for i in first[0::2]]} dp1_mb_lengths={[lengths[i] for i in first[1::2]]}",
            flush=True,
        )

    idx_range_active = idx_range_total[full_bucket_offset:]
    idx_range: Iterable[int] = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

    batch = []
    for idx in idx_range:
        batch.append(idx)
        if len(batch) == self.micro_batch_size:
            self.consumed_samples += self.micro_batch_times_data_parallel_size
            yield batch
            batch = []


def self_check() -> None:
    # Same 8 real samples as dpbalance_ab_run_20260916a; balance result matches dp_token_balance.py.
    tokens = [42761, 17524, 54257, 193722, 103736, 147886, 117720, 149554]
    g0, g1 = balance_halves(list(range(8)), tokens)
    assert (g0, g1) == ([0, 4, 6, 7], [1, 2, 3, 5]), (g0, g1)
    assert order_for_pipeline(g0, tokens) == [0, 7, 6, 4]
    assert order_for_pipeline(g1, tokens) == [1, 3, 5, 2]
    out = rebalance_windows(list(range(8)), tokens, 8)
    assert out == [0, 1, 7, 3, 6, 5, 4, 2], out
    assert _dp_gap(out, tokens) == 382
    # first/last microbatch of each replica are its two shortest samples
    assert [tokens[i] for i in out[0::2]] == [42761, 149554, 117720, 103736]
    assert [tokens[i] for i in out[1::2]] == [17524, 193722, 147886, 54257]


_original_iter = MegatronPretrainingRandomSampler.__iter__
if not getattr(MegatronPretrainingRandomSampler.__iter__, _PATCH_ATTR, False):
    self_check()
    setattr(_iter_with_balance_order, _PATCH_ATTR, True)
    MegatronPretrainingRandomSampler.__iter__ = _iter_with_balance_order


if __name__ == "__main__":
    self_check()
    print("[dpbalance-order] self-check passed")
