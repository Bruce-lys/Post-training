"""Rebalance each global batch so DP replicas see similar token counts.

Smoke A/B (dpbalance_ab_run_20260916a) used even/odd DP assignment on one
real 8-sample batch: 107.176 s/step -> 96.000 s/step (-10.43% step time).

MegatronPretrainingRandomSampler with data_sharding=False and DP=2 assigns
idx_range_total[dp_rank::dp_size], i.e. even/odd positions of the shuffled
global order. This plugin reorders every global_batch_size window before
that stride. It does not change TP/PP/CP/EP, QSA, packing, or padding_free.

Disable with DP_TOKEN_BALANCE=0. Requires data_sharding=false, group_by_length=false,
DP=2, and global_batch_size divisible by 2 * micro_batch_size.
"""

from __future__ import annotations

import itertools
import os
from typing import Iterable, Sequence

import torch
from swift.megatron.trainers.batch_sampler import MegatronPretrainingRandomSampler


_PATCH_ATTR = "_dp_token_balance_patch"
_LENGTH_CACHE_ATTR = "_dp_token_balance_lengths"


def _enabled() -> bool:
    return os.environ.get("DP_TOKEN_BALANCE", "1") != "0"


def _should_log() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _global_batch_size(sampler: MegatronPretrainingRandomSampler) -> int:
    env_value = os.environ.get("DP_TOKEN_BALANCE_GBS")
    if env_value:
        return int(env_value)
    try:
        from megatron.training import get_args

        return int(get_args().global_batch_size)
    except Exception:
        return int(os.environ.get("GLOBAL_BATCH_SIZE", "8"))


def _row_length(row) -> int:
    if not isinstance(row, dict):
        raise TypeError(f"dataset row is {type(row)}, expected dict")
    for key in ("lengths", "length"):
        if key in row and row[key] is not None:
            value = row[key]
            return max(value) if isinstance(value, list) else int(value)
    if "input_ids" in row:
        return len(row["input_ids"])
    raise KeyError("dataset row has no lengths/length/input_ids")


def _dataset_lengths(dataset, total_samples: int) -> list[int]:
    cached = getattr(dataset, _LENGTH_CACHE_ATTR, None)
    if cached is not None and len(cached) >= total_samples:
        return cached
    column_names = getattr(dataset, "column_names", None)
    lengths: list[int]
    if column_names and "lengths" in column_names:
        raw = dataset["lengths"]
        lengths = [max(item) if isinstance(item, list) else int(item) for item in raw]
    elif column_names and "length" in column_names:
        raw = dataset["length"]
        lengths = [max(item) if isinstance(item, list) else int(item) for item in raw]
    else:
        lengths = [_row_length(dataset[index]) for index in range(total_samples)]
    try:
        setattr(dataset, _LENGTH_CACHE_ATTR, lengths)
    except Exception:
        pass
    return lengths


def balance_even_odd(indices: Sequence[int], lengths: Sequence[int]) -> list[int]:
    """Split one global batch into two equal groups with closest token totals, then interleave."""
    if len(indices) % 2 != 0:
        raise ValueError(f"window size must be even, got {len(indices)}")
    tokens = [int(lengths[index]) for index in indices]
    total = sum(tokens)
    group_size = len(indices) // 2
    best = min(
        itertools.combinations(range(len(indices)), group_size),
        key=lambda picked: abs(total - 2 * sum(tokens[item] for item in picked)),
    )
    dp0_positions = set(best)
    groups = [
        [indices[position] for position, _ in enumerate(indices) if (position in dp0_positions) == (rank == 0)]
        for rank in range(2)
    ]
    return [item for pair in zip(*groups) for item in pair]


def rebalance_windows(indices: Sequence[int], lengths: Sequence[int], window: int) -> list[int]:
    if window <= 0 or window % 2 != 0:
        raise ValueError(f"window must be a positive even integer, got {window}")
    output: list[int] = []
    usable = (len(indices) // window) * window
    for start in range(0, usable, window):
        output.extend(balance_even_odd(indices[start : start + window], lengths))
    output.extend(indices[usable:])
    return output


def _dp_token_gap(indices: Sequence[int], lengths: Sequence[int]) -> int:
    totals = [0, 0]
    for position, index in enumerate(indices):
        totals[position % 2] += int(lengths[index])
    return abs(totals[0] - totals[1])


def _iter_with_dp_token_balance(self):
    if (
        not _enabled()
        or self.data_sharding
        or getattr(self, "group_by_length", False)
        or self.data_parallel_size != 2
    ):
        if _enabled() and _should_log() and (
            self.data_sharding or getattr(self, "group_by_length", False) or self.data_parallel_size != 2
        ):
            print(
                "[dpbalance] skipped: need DP=2, data_sharding=false, group_by_length=false; "
                f"got dp={self.data_parallel_size} sharding={self.data_sharding} "
                f"group_by_length={getattr(self, 'group_by_length', False)}",
                flush=True,
            )
        yield from _original_iter(self)
        return

    global_batch_size = _global_batch_size(self)
    if global_batch_size % self.micro_batch_times_data_parallel_size != 0:
        raise RuntimeError(
            f"DP token balance requires global_batch_size divisible by "
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

    lengths = _dataset_lengths(self.dataset, self.total_samples)
    before_gap = _dp_token_gap(idx_range_total[:global_batch_size], lengths) if idx_range_total else 0
    idx_range_total = rebalance_windows(idx_range_total, lengths, global_batch_size)
    after_gap = _dp_token_gap(idx_range_total[:global_batch_size], lengths) if idx_range_total else 0
    if _should_log():
        print(
            f"[dpbalance] epoch={self.epoch} gbs={global_batch_size} "
            f"first_window_gap {before_gap}->{after_gap}",
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
    # Same 8 real samples as dpbalance_ab_run_20260916a variant A/B.
    tokens = [42761, 17524, 54257, 193722, 103736, 147886, 117720, 149554]
    original = list(range(8))
    balanced = balance_even_odd(original, tokens)
    assert balanced == [0, 1, 4, 2, 6, 3, 7, 5], balanced
    assert _dp_token_gap(original, tokens) == 190212
    assert _dp_token_gap(balanced, tokens) == 382


_original_iter = MegatronPretrainingRandomSampler.__iter__
if not getattr(MegatronPretrainingRandomSampler.__iter__, _PATCH_ATTR, False):
    self_check()
    setattr(_iter_with_dp_token_balance, _PATCH_ATTR, True)
    MegatronPretrainingRandomSampler.__iter__ = _iter_with_dp_token_balance


if __name__ == "__main__":
    self_check()
    print("[dpbalance] self-check passed")
