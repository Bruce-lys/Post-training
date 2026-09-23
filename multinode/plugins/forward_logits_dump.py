"""Forward-only logits dump on the training path (train-side reference for the SGLang comparison).

Hooks BaseMegatronTrainer.train_step: on the first call it runs ONE global batch through mcore's
forward_backward_func(forward_only=True, collect_non_loss_data=True) under torch.no_grad() with the
model in eval mode, so every microbatch goes through exactly the training forward (padding_free/thd,
QSA sparse kernel, GDN, chunked PLE, MoE) but no backward, no optimizer step, no checkpoint.
Labels are NOT passed to the model, so the last PP stage returns vocab-parallel logits [b, s, V/TP].
Logits are gathered across TP in sequence chunks (never the full [T, V] at once) and, per sample,
TP-rank-0 of the last stage saves FWD_DUMP_DIR/train_{i}.pt with:
  input_ids [T], labels [T], nll_next [T-1] (-logp of input_ids[t+1] at t), argmax [T], entropy [T],
  topk_ids/topk_logprobs [T, K], sample_pos + sample_logits (fp32 full-vocab rows at strided positions),
  full_logits (bf16 [T, V], only when T <= FWD_FULL_VOCAB_MAX_T).
Then all ranks barrier and the process exits with code 0.

Env: FWD_DUMP_DIR (required), FWD_TOPK=100, FWD_SAMPLE_STRIDE=64, FWD_SAMPLE_TAIL=512,
     FWD_FULL_VOCAB_MAX_T=4096, FWD_VOCAB_SIZE=248320, FWD_GATHER_CHUNK=2048
"""

from __future__ import annotations

import json
import os
import sys
import time
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from swift.megatron.trainers.base import BaseMegatronTrainer

OUT = os.environ.get("FWD_DUMP_DIR")
TOPK = int(os.environ.get("FWD_TOPK", "100"))
STRIDE = int(os.environ.get("FWD_SAMPLE_STRIDE", "64"))
TAIL = int(os.environ.get("FWD_SAMPLE_TAIL", "512"))
FULL_MAX_T = int(os.environ.get("FWD_FULL_VOCAB_MAX_T", "4096"))
VOCAB = int(os.environ.get("FWD_VOCAB_SIZE", "248320"))
CHUNK = int(os.environ.get("FWD_GATHER_CHUNK", "2048"))
_PATCH_ATTR = "_forward_logits_dump"
_state = {"mb": 0}


def _rank() -> str:
    return os.environ.get("RANK", "?")


def _log(msg: str) -> None:
    print(f"[fwddump] rank={_rank()} {msg}", flush=True)


def _process_and_save(output_tensor: torch.Tensor, data: dict, labels: torch.Tensor | None) -> None:
    """output_tensor: [b, s, V_local] on the last PP stage (b == 1 under padding_free)."""
    idx = _state["mb"]
    _state["mb"] += 1
    if output_tensor.dim() != 3 or output_tensor.shape[0] != 1:
        raise RuntimeError(f"expected [1, T, V_local] logits, got {tuple(output_tensor.shape)}")
    logits_local = output_tensor[0]  # [T, V_local]
    T = logits_local.shape[0]
    input_ids = data["input_ids"].reshape(-1)[:T].to(torch.long)
    device = logits_local.device
    tp_rank = mpu.get_tensor_model_parallel_rank()
    save_here = tp_rank == 0 and mpu.get_data_parallel_rank() == 0

    nll_next = torch.empty(max(T - 1, 0), dtype=torch.float32, device=device)
    argmax = torch.empty(T, dtype=torch.long, device=device)
    entropy = torch.empty(T, dtype=torch.float32, device=device)
    topk_ids = torch.empty(T, TOPK, dtype=torch.int32, device=device)
    topk_lp = torch.empty(T, TOPK, dtype=torch.float32, device=device)
    sample_pos = sorted(set(range(0, T, STRIDE)) | set(range(max(0, T - TAIL), T)))
    sample_set = set(sample_pos)
    sample_rows = []
    full_rows = [] if T <= FULL_MAX_T else None

    started = time.perf_counter()
    for s in range(0, T, CHUNK):
        e = min(s + CHUNK, T)
        # every TP rank participates in the gather; only tp0 keeps the result
        gathered = gather_from_tensor_model_parallel_region(logits_local[s:e].contiguous())  # [c, V_pad]
        lg = gathered[:, :VOCAB].float()
        lp = torch.log_softmax(lg, dim=-1)
        argmax[s:e] = lp.argmax(dim=-1)
        tk = lp.topk(TOPK, dim=-1)
        topk_ids[s:e] = tk.indices.to(torch.int32)
        topk_lp[s:e] = tk.values
        entropy[s:e] = -(lp.exp() * lp).sum(dim=-1)
        n = min(e, T - 1) - s
        if n > 0:
            nxt = input_ids[s + 1:s + 1 + n]
            nll_next[s:s + n] = -lp[:n].gather(1, nxt[:, None])[:, 0]
        pos_here = [p for p in range(s, e) if p in sample_set]
        if pos_here:
            sample_rows.append(lg[[p - s for p in pos_here]].cpu())
        if full_rows is not None:
            full_rows.append(gathered[:, :VOCAB].to(torch.bfloat16).cpu())
        del gathered, lg, lp, tk
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if save_here:
        os.makedirs(OUT, exist_ok=True)
        payload = {
            "index": idx,
            "T": T,
            "vocab": VOCAB,
            "input_ids": input_ids.cpu(),
            "labels": labels.reshape(-1)[:T].cpu() if labels is not None else None,
            "nll_next": nll_next.cpu(),
            "argmax": argmax.cpu(),
            "entropy": entropy.cpu(),
            "topk_ids": topk_ids.cpu(),
            "topk_logprobs": topk_lp.cpu(),
            "sample_pos": torch.tensor(sample_pos, dtype=torch.long),
            "sample_logits": torch.cat(sample_rows, 0) if sample_rows else torch.empty(0, VOCAB),
            "full_logits": torch.cat(full_rows, 0) if full_rows else None,
            "qsa_sparse_kernel": os.environ.get("QSA_SPARSE_KERNEL"),
            "tp": mpu.get_tensor_model_parallel_world_size(),
            "pp": mpu.get_pipeline_model_parallel_world_size(),
            "ep": os.environ.get("FWD_EP", ""),
        }
        path = os.path.join(OUT, f"train_{idx}.pt")
        torch.save(payload, path)
        masked = int((payload["labels"] != -100).sum()) if payload["labels"] is not None else -1
        _log(f"saved {path} T={T} mean_nll_next={float(nll_next.mean()):.4f} "
             f"label_tokens={masked} gather+softmax={elapsed:.1f}s")


def _loss_func(output_tensor, non_loss_data: bool = False, *, data, labels):
    if non_loss_data:
        _process_and_save(output_tensor, data, labels)
        return {"index": _state["mb"] - 1}
    # Not expected (forward_only + collect_non_loss_data); keep mcore's contract anyway.
    return torch.zeros(1, device=output_tensor.device), 1, {}


def _forward_step(data_iterator, model, *, trainer):
    inner = model
    while hasattr(inner, "module"):  # DDP -> (Float16Module in bf16) -> model
        inner = inner.module
    vp_stage = getattr(inner, "vp_stage", None)
    data = trainer.get_batch(data_iterator, vp_stage)
    data.pop("loss_scale", None)
    data.pop("channel", None)
    labels = data.pop("labels", None)  # no labels -> model returns logits
    output_tensor = model(**data)
    return output_tensor, partial(_loss_func, data=data, labels=labels)


def _dump_then_exit(self, train_data_iterator):
    if OUT is None:
        raise RuntimeError("FWD_DUMP_DIR is not set")
    args = self.args
    forward_backward_func = get_forward_backward_func()
    data_iterator = self._replace_data_iterator(train_data_iterator)
    for m in self.wrapped_models:
        m.eval()
    _log(f"forward-only dump: num_microbatches={args.num_microbatches} seq_length={args.seq_length} "
         f"QSA_SPARSE_KERNEL={os.environ.get('QSA_SPARSE_KERNEL')} out={OUT}")
    started = time.perf_counter()
    with torch.no_grad():
        forward_backward_func(
            forward_step_func=partial(_forward_step, trainer=self),
            data_iterator=data_iterator,
            model=self.wrapped_models,
            num_microbatches=args.num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    if _rank() == "0":
        with open(os.path.join(OUT, "DONE"), "w") as f:
            json.dump({"num_microbatches": args.num_microbatches, "seconds": time.perf_counter() - started}, f)
    _log(f"done in {time.perf_counter() - started:.1f}s; exiting before any training step")
    sys.stdout.flush()
    sys.exit(0)


if not getattr(BaseMegatronTrainer.train_step, _PATCH_ATTR, False):
    setattr(_dump_then_exit, _PATCH_ATTR, True)
    BaseMegatronTrainer.train_step = _dump_then_exit
