#!/usr/bin/env python3
"""Stage table + kernel-category table for ONE profiled run (node0 = PP stage 0, node2 = PP stage 1).

  python analyze_nsys_single.py --profile-dir PROFILES/<tag>/<ts> --iteration 6 [--node0-host H --node2-host H]
Reuses analyze_db/aggregate/print_summary from analyze_nsys_iteration_pair.py (same columns as the 09-16 tables).
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_nsys_iteration_pair import aggregate, analyze_db, print_summary  # noqa: E402

MASK = -16777216


def category(name: str) -> str:
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if "_qsa_bs" in n or "qsa" in n:
        return "qsa_sparse_attn"
    if "flash" in n or "fmha" in n or "fused_attn" in n:
        return "dense_attn"
    if any(k in n for k in ("chunk_", "gated_delta", "causal_conv", "solve_tril", "wy_", "recompute_w")):
        return "gdn_linear_attn"
    if any(k in n for k in ("grouped", "group_gemm", "moe", "permute", "radixfind", "gathertopk", "sort", "topk", "router")):
        return "moe_router_grouped_gemm"
    if any(k in n for k in ("gemm", "cutlass", "cublas", "xmma", "nvjet", "ampere_bf16", "hopper")):
        return "dense_gemm"
    if any(k in n for k in ("cross_entropy", "softmax", "vocab")):
        return "logits_ce_softmax"
    if "norm" in n or "rms" in n:
        return "norm"
    if any(k in n for k in ("ngram", "ple", "embedding", "index_select", "gather", "scatter", "index_")):
        return "embedding_index_ple"
    if any(k in n for k in ("elementwise", "copy", "fill", "cat", "reduce_kernel", "where", "cumsum", "arange")):
        return "elementwise_copy_reduce"
    if "multi_tensor" in n or "adam" in n:
        return "optimizer"
    return "other"


def kernel_categories(db: Path, iteration: int) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    win = c.execute(
        "SELECT start, end, globalTid FROM NVTX_EVENTS WHERE text = ? AND end IS NOT NULL ORDER BY start LIMIT 1",
        (f"FULL_ITERATION_{iteration}",),
    ).fetchone()
    if not win:
        return {}
    s, e, tid = win
    pid = tid & MASK
    rows = c.execute(
        "SELECT k.start, k.end, st.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds st ON k.demangledName = st.id "
        "WHERE k.globalPid = ? AND k.start >= ? AND k.end <= ?",
        (pid, s, e),
    ).fetchall()
    tot, cnt = collections.Counter(), collections.Counter()
    for st, en, nm in rows:
        k = category(nm)
        tot[k] += (en - st) / 1e9
        cnt[k] += 1
    return {
        "iteration_s": (e - s) / 1e9,
        "categories": {k: {"seconds": round(v, 3), "count": cnt[k]} for k, v in tot.most_common()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", type=Path, required=True)
    ap.add_argument("--iteration", type=int, default=6)
    ap.add_argument("--node0-host", default="hd03-gpu2-0019")
    ap.add_argument("--node2-host", default="hd03-gpu2-0005")
    ap.add_argument("--run-name", default="run")
    args = ap.parse_args()
    db0, db2 = args.profile_dir / "nsys.node0.sqlite", args.profile_dir / "nsys.node2.sqlite"
    rows = []
    rows.extend(analyze_db(db0, args.run_name, args.node0_host, 0, 0, args.iteration))
    rows.extend(analyze_db(db2, args.run_name, args.node2_host, 1, 16, args.iteration))
    summary = aggregate(rows)
    cats = {
        "stage0_rank0": kernel_categories(db0, args.iteration),
        "stage1_rank16": kernel_categories(db2, args.iteration),
    }
    out = args.profile_dir / f"analysis_iteration{args.iteration}.json"
    out.write_text(json.dumps({"per_rank": rows, "stage_summary": summary, "kernel_categories": cats}, indent=2) + "\n",
                   encoding="utf-8")
    print_summary(summary)
    for stage, d in cats.items():
        print(f"\n[kernel categories] {stage} iteration_s={d.get('iteration_s', 0):.1f}")
        for k, v in d.get("categories", {}).items():
            print(f"  {k:26s} {v['seconds']:8.2f}s  n={v['count']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
