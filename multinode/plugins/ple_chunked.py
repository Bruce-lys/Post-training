"""Chunked PLE compute for Qwen3.8-Flash-Next (long-sequence memory fix).

``Qwen4ExpTextPLELayer.compute`` runs on the *full* gathered sequence (SP is
undone before PLE) and materialises several fp32 temporaries of shape
[L, hc_count * hidden]; at L ~ 200K each one is ~8.5 GiB and the PP stage that
holds layer 2 OOMs during the recompute of this layer.

Every op inside ``compute`` is causal with a bounded look-back:
  * n-gram hash embedding: previous ``ngram_size - 1`` tokens (eos-segment aware)
  * key/value/norm/gate: per token
  * depthwise causal conv: previous ``short_conv_state_len`` tokens
so the sequence can be processed in chunks with an overlap of
``short_conv_state_len + (ngram_size - 1)`` tokens; the overlap prefix is
recomputed and dropped.  Each chunk is wrapped in a non-reentrant checkpoint so
backward recomputes one chunk at a time.  Output is numerically identical to
the unchunked path (verified by ple_chunked_test.py).

Load via YAML ``external_plugins`` *after* packed_ple_varlen.py (that patch
calls ``self.compute``, so it picks the chunked version up automatically).
Env: ``PLE_CHUNK_SIZE`` (tokens, default 16384; 0 disables).
"""

from __future__ import annotations

import os

import torch
from torch.utils.checkpoint import checkpoint

from mcore_bridge.model.modules.ple import Qwen4ExpTextPLELayer
from swift.utils import get_logger

logger = get_logger()
_PATCH_ATTR = "_swift_ple_chunked_patch"
CHUNK = int(os.environ.get("PLE_CHUNK_SIZE", "16384"))

_original_compute = Qwen4ExpTextPLELayer.compute


def _chunked_compute(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    seq_len = hidden_states.shape[1]
    if CHUNK <= 0 or seq_len <= CHUNK:
        return _original_compute(self, hidden_states, input_ids)

    overlap = int(self.short_conv_state_len) + int(self.ple_embedding.ngram_size) - 1
    pieces = []
    for start in range(0, seq_len, CHUNK):
        end = min(start + CHUNK, seq_len)
        ctx_start = max(0, start - overlap)
        drop = start - ctx_start

        def _run(h, ids, drop=drop):
            return _original_compute(self, h, ids)[:, drop:]

        h = hidden_states[:, ctx_start:end]
        ids = input_ids[:, ctx_start:end]
        if torch.is_grad_enabled() and h.requires_grad:
            pieces.append(checkpoint(_run, h, ids, use_reentrant=False))
        else:
            pieces.append(_run(h, ids))

    if not getattr(self, "_ple_chunked_logged", False):
        logger.warning("Chunked PLE compute enabled: seq_len=%d chunk=%d overlap=%d chunks=%d",
                       seq_len, CHUNK, overlap, len(pieces))
        self._ple_chunked_logged = True
    return torch.cat(pieces, dim=1)


if not getattr(Qwen4ExpTextPLELayer.compute, _PATCH_ATTR, False):
    setattr(_chunked_compute, _PATCH_ATTR, True)
    Qwen4ExpTextPLELayer.compute = _chunked_compute
    logger.info("Applied chunked PLE compute patch (PLE_CHUNK_SIZE=%d).", CHUNK)
