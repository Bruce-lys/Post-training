#!/usr/bin/env python3
"""Third-party referee: transformers (native qwen4_exp) forward on the fwd_check texts, same dump format.

  python hf_forward_reference.py --train-dir OUT/fwd_check/train_sparse --out OUT/fwd_check/hf --indices 0 1
Loads the base model in bf16 across all visible GPUs (device_map=auto), feeds the exact input_ids saved by
forward_logits_dump.py, and writes hf/train_{i}.pt with the same fields (nll_next, argmax, topk, sample rows,
full_logits for short texts), so compare_forward.py can pit HF against SGLang (--train hf --sglang sglang)
and against the Megatron dump (--train hf --train-dense train_sparse).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

MODEL = "/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"


def stats(logits: torch.Tensor, input_ids: torch.Tensor, topk: int, stride: int, tail: int, full_max_t: int, vocab: int):
    """logits [T, V] on any device -> dict like forward_logits_dump.py (computed in fp32, chunked)."""
    T = logits.shape[0]
    dev = logits.device
    nll_next = torch.empty(max(T - 1, 0), dtype=torch.float32)
    argmax = torch.empty(T, dtype=torch.long)
    entropy = torch.empty(T, dtype=torch.float32)
    topk_ids = torch.empty(T, topk, dtype=torch.int32)
    topk_lp = torch.empty(T, topk, dtype=torch.float32)
    sample_pos = sorted(set(range(0, T, stride)) | set(range(max(0, T - tail), T)))
    sset = set(sample_pos)
    sample_rows, full_rows = [], ([] if T <= full_max_t else None)
    chunk = 2048
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lg = logits[s:e, :vocab].float()
        lp = torch.log_softmax(lg, dim=-1)
        argmax[s:e] = lp.argmax(dim=-1).cpu()
        tk = lp.topk(topk, dim=-1)
        topk_ids[s:e] = tk.indices.to(torch.int32).cpu()
        topk_lp[s:e] = tk.values.cpu()
        entropy[s:e] = (-(lp.exp() * lp).sum(dim=-1)).cpu()
        n = min(e, T - 1) - s
        if n > 0:
            nxt = input_ids[s + 1:s + 1 + n].to(dev)
            nll_next[s:s + n] = (-lp[:n].gather(1, nxt[:, None])[:, 0]).cpu()
        pos_here = [p for p in range(s, e) if p in sset]
        if pos_here:
            sample_rows.append(lg[[p - s for p in pos_here]].cpu())
        if full_rows is not None:
            full_rows.append(logits[s:e, :vocab].to(torch.bfloat16).cpu())
    return {
        "T": T, "vocab": vocab, "input_ids": input_ids.cpu(), "nll_next": nll_next, "argmax": argmax, "entropy": entropy,
        "topk_ids": topk_ids, "topk_logprobs": topk_lp, "sample_pos": torch.tensor(sample_pos, dtype=torch.long),
        "sample_logits": torch.cat(sample_rows, 0) if sample_rows else torch.empty(0, vocab),
        "full_logits": torch.cat(full_rows, 0) if full_rows else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--indices", type=int, nargs="+", default=[0])
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--attn-impl", default=None, help="e.g. eager / sdpa / flash_attention_2 (default: model default)")
    ap.add_argument("--max-memory-gib", type=int, default=0, help="cap per visible GPU for device_map=auto (0 = no cap)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    kwargs = dict(dtype=getattr(torch, args.dtype), device_map="auto", trust_remote_code=True)
    if args.attn_impl:
        kwargs["attn_implementation"] = args.attn_impl
    if args.max_memory_gib:
        kwargs["max_memory"] = {i: f"{args.max_memory_gib}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kwargs)
    model.eval()
    print(f"[hf] loaded {type(model).__name__} in {time.time() - t0:.0f}s; attn_impl={getattr(model.config, '_attn_implementation', None)}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    report = []
    for i in args.indices:
        d = torch.load(os.path.join(args.train_dir, f"train_{i}.pt"), map_location="cpu")
        ids = d["input_ids"].long()
        first = next(model.parameters()).device
        t1 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids[None].to(first), use_cache=False)
        logits = out.logits[0]
        r = stats(logits, ids, args.topk, 64, 512, 4096, d.get("vocab", 248320))
        r.update({"index": i, "labels": d.get("labels"), "source": "transformers", "dtype": args.dtype, "attn_impl": getattr(model.config, "_attn_implementation", None)})
        torch.save(r, os.path.join(args.out, f"train_{i}.pt"))
        rep = {"index": i, "T": r["T"], "mean_nll_next": float(r["nll_next"].mean()), "seconds": round(time.time() - t1, 1)}
        report.append(rep)
        print("[hf]", json.dumps(rep), flush=True)
        del out, logits
        torch.cuda.empty_cache()
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print("[hf] DONE", flush=True)


if __name__ == "__main__":
    main()
