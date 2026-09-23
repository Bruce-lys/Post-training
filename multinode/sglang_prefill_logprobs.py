#!/usr/bin/env python3
"""Teacher-forcing prefill through a running SGLang server: per-position logprobs for the fwd_check texts.

  python sglang_prefill_logprobs.py --train-dir OUT/fwd_check/train_sparse --out OUT/fwd_check/sglang \
      --url http://127.0.0.1:30000 [--topk 100] [--full-vocab-max-t 4096] [--full-vocab-topk 2048]

Feeds the EXACT input_ids saved by forward_logits_dump.py (no re-tokenization) to /generate with
return_logprob + logprob_start_len=0 + top_logprobs_num, max_new_tokens=1, temperature=0.
Saves per text: sglang_{i}.pt with token_logprobs [T] (position 0 = nan), top_ids/top_logprobs [T, K].
SGLang convention: input_token_logprobs[i] is log p(token_i | tokens_<i); input_top_logprobs[i] is the
top-K distribution at that same prediction point, i.e. it aligns with the train-side row t = i - 1.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import requests
import torch


def query(url: str, ids: list[int], topk: int, timeout: int) -> dict:
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": topk,
        "return_text_in_logprobs": False,
    }
    r = requests.post(f"{url}/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse(meta: dict, T: int, topk: int):
    tok = meta["input_token_logprobs"]
    top = meta.get("input_top_logprobs") or []
    token_lp = torch.full((T,), float("nan"))
    top_ids = torch.full((T, topk), -1, dtype=torch.int32)
    top_lp = torch.full((T, topk), float("nan"))
    for i, entry in enumerate(tok[:T]):
        if entry is None:
            continue
        lp = entry[0] if isinstance(entry, (list, tuple)) else entry
        if lp is not None:
            token_lp[i] = float(lp)
    for i, entries in enumerate(top[:T]):
        if not entries:
            continue
        for k, entry in enumerate(entries[:topk]):
            if entry is None:
                continue
            top_lp[i, k] = float(entry[0])
            top_ids[i, k] = int(entry[1])
    return token_lp, top_ids, top_lp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--full-vocab-max-t", type=int, default=4096)
    ap.add_argument("--full-vocab-topk", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.train_dir, "train_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    report = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        ids = d["input_ids"].tolist()
        T = len(ids)
        k = args.full_vocab_topk if T <= args.full_vocab_max_t else args.topk
        t0 = time.time()
        resp = query(args.url, ids, k, args.timeout)
        dt = time.time() - t0
        meta = resp["meta_info"] if "meta_info" in resp else resp[0]["meta_info"]
        token_lp, top_ids, top_lp = parse(meta, T, k)
        out = os.path.join(args.out, f"sglang_{d['index']}.pt")
        torch.save({"index": d["index"], "T": T, "topk": k, "token_logprobs": token_lp, "top_ids": top_ids,
                    "top_logprobs": top_lp, "input_ids": d["input_ids"], "seconds": dt,
                    "server": args.url, "prompt_tokens": meta.get("prompt_tokens")}, out)
        valid = token_lp[1:]
        report.append({"index": d["index"], "T": T, "topk": k, "seconds": round(dt, 1),
                       "mean_nll": float((-valid[~torch.isnan(valid)]).mean()) if T > 1 else None})
        print(json.dumps(report[-1]), flush=True)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
