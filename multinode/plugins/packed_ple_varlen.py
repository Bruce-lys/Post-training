"""Memory-safe packed-input PLE compatibility patch for Qwen3.8-Flash-Next.

The mcore-bridge THD path expands all packed samples to
``num_samples * max_seqlen`` before running PLE.  Highly imbalanced packs can
therefore create a much larger FP32 GroupedRMSNorm temporary than the original
packed sequence.  This patch forms contiguous groups whose rectangular size is capped at 1.5 times the
number of real tokens in the pack, preserving batching for similarly sized
samples while splitting pathological groups.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import inspect
import torch

from mcore_bridge.model.modules.ple import Qwen4ExpTextPLELayer
from swift.utils import get_logger

logger = get_logger()
_PATCH_ATTR = "_swift_packed_ple_varlen_patch"


def _build_groups(lengths: Sequence[int], token_budget: int) -> List[Tuple[int, int]]:
    """Return contiguous [start, end) groups with rows * max_len <= budget."""
    if not lengths:
        return []
    groups: List[Tuple[int, int]] = []
    group_start = 0
    group_max = 0
    for index, length in enumerate(lengths):
        candidate_max = max(group_max, length)
        candidate_rows = index - group_start + 1
        if index > group_start and candidate_rows * candidate_max > token_budget:
            groups.append((group_start, index))
            group_start = index
            group_max = length
        else:
            group_max = candidate_max
    groups.append((group_start, len(lengths)))
    return groups


_original_forward_impl = Qwen4ExpTextPLELayer._forward_impl

# Fail loudly if a future mcore-bridge release changes or fixes this code path.
if not getattr(_original_forward_impl, _PATCH_ATTR, False):
    _source = inspect.getsource(_original_forward_impl)
    _expected = (
        "hid = hidden_states.new_zeros((num_samples, max_len, hidden_states.shape[-1]))",
        "res = self.compute(hid, toks)",
    )
    if not all(fragment in _source for fragment in _expected):
        raise RuntimeError(
            "packed_ple_varlen.py does not match the installed mcore-bridge PLE implementation; "
            "review or remove the compatibility patch before training."
        )


def _memory_safe_forward_impl(self, hidden_states, input_ids, packed_seq_params=None):
    thd = packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", "bshd") == "thd"
    if not thd:
        return _original_forward_impl(self, hidden_states, input_ids, packed_seq_params)

    num_samples = int(packed_seq_params.num_samples)
    cu = packed_seq_params.cu_seqlens_q.reshape(-1)
    total = int(hidden_states.shape[0])
    if num_samples <= 0 or cu.numel() < num_samples + 1:
        raise ValueError(
            f"Invalid packed PLE metadata: num_samples={num_samples}, cu_seqlens entries={cu.numel()}."
        )

    boundaries = [int(cu[index]) for index in range(num_samples + 1)]
    if boundaries[0] != 0 or any(end <= start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError(f"Invalid packed PLE boundaries: {boundaries}.")
    valid_end = boundaries[-1]
    if valid_end > total or input_ids.shape[-1] < valid_end:
        raise ValueError(
            f"Packed PLE boundaries end at {valid_end}, but hidden/input lengths are "
            f"{total}/{input_ids.shape[-1]}."
        )

    lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
    # Bound PLE-only padding overhead while keeping similarly sized rows batched.
    token_budget = (3 * sum(lengths) + 1) // 2
    groups = _build_groups(lengths, token_budget)
    pieces = []

    for first, stop in groups:
        group_lengths = lengths[first:stop]
        rows = stop - first
        max_len = max(group_lengths)

        if rows == 1:
            start, end = boundaries[first], boundaries[first + 1]
            result = self.compute(
                hidden_states[start:end, 0].unsqueeze(0),
                input_ids[0, start:end].unsqueeze(0),
            )
            pieces.append(result[0])
            continue

        hid = hidden_states.new_zeros((rows, max_len, hidden_states.shape[-1]))
        toks = input_ids.new_full((rows, max_len), self.ple_embedding.eos_token_id)
        for row, sample_index in enumerate(range(first, stop)):
            start, end = boundaries[sample_index], boundaries[sample_index + 1]
            length = end - start
            hid[row, :length] = hidden_states[start:end, 0]
            toks[row, :length] = input_ids[0, start:end]

        result = self.compute(hid, toks)
        pieces.extend(result[row, :length] for row, length in enumerate(group_lengths))

    output = torch.cat(pieces, dim=0).unsqueeze(1)
    if valid_end < total:
        padding = output.new_zeros((total - valid_end, 1, output.shape[-1]))
        output = torch.cat((output, padding), dim=0)

    if len(groups) > 1 and not getattr(self, "_packed_ple_split_logged", False):
        original_rectangle = num_samples * max(lengths)
        largest_group_rectangle = max(
            (stop - first) * max(lengths[first:stop]) for first, stop in groups
        )
        logger.warning(
            "Memory-safe packed PLE split enabled: %d samples, %d groups, "
            "rectangular tokens %d -> max %d.",
            num_samples,
            len(groups),
            original_rectangle,
            largest_group_rectangle,
        )
        self._packed_ple_split_logged = True
    return output


if not getattr(Qwen4ExpTextPLELayer._forward_impl, _PATCH_ATTR, False):
    setattr(_memory_safe_forward_impl, _PATCH_ATTR, True)
    Qwen4ExpTextPLELayer._forward_impl = _memory_safe_forward_impl
    logger.info("Applied memory-safe packed PLE patch for Qwen3.8-Flash-Next.")
