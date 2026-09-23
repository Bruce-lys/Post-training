#!/usr/bin/env python3
"""Compare train-side forward dumps with SGLang prefill logprobs (and optionally train-sparse vs train-dense).

  python compare_forward.py --train OUT/fwd_check/train_sparse --sglang OUT/fwd_check/sglang [--train-dense OUT/fwd_check/train_dense] --out report.json

Alignment: train row t predicts token t+1  <->  sglang position i = t+1 (token_logprobs[i], top_*[i]).
Metrics per text (all positions unless noted):
  argmax_agree      fraction of t where train argmax == sglang top-1
  nll_abs_diff      |train nll_next[t] - (-sglang token_logprob[t+1])| : mean / p99 / max
  top10_jaccard     mean Jaccard of the top-10 id sets
  union_cosine / union_rel_l2 / union_max_abs   on the union of both top-K sets, comparing log-probs
                    (renormalised over the union) at the strided sample positions
  full_vocab_*      (short text only, when both sides have >= 2048 entries) cosine / rel-L2 / max-abs of logits
  by_position       the above binned by 2048 tokens: a step at the QSA budget boundary = sparse-path issue
Thresholds (bf16 vs bf16): argmax_agree > 0.995, nll mean |diff| < 0.01, top10_jaccard > 0.95,
                          union_cosine > 0.999, union_rel_l2 < 1e-2.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import torch


def load_pairs(train_dir: str, sglang_dir: str):
    for f in sorted(glob.glob(os.path.join(train_dir, "train_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1][:-3])):
        d = torch.load(f, map_location="cpu")
        s = os.path.join(sglang_dir, f"sglang_{d['index']}.pt")
        if os.path.exists(s):
            yield d, torch.load(s, map_location="cpu")


def union_metrics(tr_ids, tr_lp, sg_ids, sg_lp):
    """Compare the two top-K distributions on the union of their ids (log-probs, renormalised)."""
    ids = sorted(set(tr_ids.tolist()) | set(int(x) for x in sg_ids.tolist() if x >= 0))
    idx = {i: k for k, i in enumerate(ids)}
    a = torch.full((len(ids),), -30.0)
    b = torch.full((len(ids),), -30.0)
    for i, lp in zip(tr_ids.tolist(), tr_lp.tolist()):
        a[idx[i]] = lp
    for i, lp in zip(sg_ids.tolist(), sg_lp.tolist()):
        if i >= 0 and not math.isnan(lp):
            b[idx[int(i)]] = lp
    pa, pb = torch.softmax(a, 0), torch.softmax(b, 0)
    cos = float(torch.dot(pa, pb) / (pa.norm() * pb.norm() + 1e-12))
    rel = float((pa - pb).norm() / (pa.norm() + 1e-12))
    mx = float((a - b).abs().max())
    return cos, rel, mx


def compare_text(d: dict, s: dict, bin_size: int = 2048) -> dict:
    T = d["T"]
    assert s["T"] == T
    n = T - 1
    tr_argmax = d["argmax"][:n]
    sg_top1 = s["top_ids"][1:T, 0].long()
    agree = (tr_argmax == sg_top1).float()
    sg_nll = -s["token_logprobs"][1:T]
    diff = (d["nll_next"][:n] - sg_nll).abs()
    valid = ~torch.isnan(diff)
    tr10 = d["topk_ids"][:n, :10].tolist()
    sg10 = s["top_ids"][1:T, :10].tolist()
    jac = torch.tensor([
        len(set(a) & set(b)) / max(1, len(set(a) | set(b))) for a, b in zip(tr10, sg10)
    ])
    sample_pos = [p for p in d["sample_pos"].tolist() if p < n]
    ucos, urel, umax = [], [], []
    K = min(d["topk_ids"].shape[1], s["top_ids"].shape[1])
    for t in sample_pos:
        c, r, m = union_metrics(d["topk_ids"][t, :K], d["topk_logprobs"][t, :K], s["top_ids"][t + 1, :K], s["top_logprobs"][t + 1, :K])
        ucos.append(c); urel.append(r); umax.append(m)
    ucos, urel, umax = torch.tensor(ucos), torch.tensor(urel), torch.tensor(umax)

    by_pos = []
    for b0 in range(0, n, bin_size):
        b1 = min(b0 + bin_size, n)
        sel = [k for k, t in enumerate(sample_pos) if b0 <= t < b1]
        v = valid[b0:b1]
        by_pos.append({
            "start": b0, "end": b1,
            "argmax_agree": round(float(agree[b0:b1].mean()), 5),
            "nll_abs_diff_mean": round(float(diff[b0:b1][v].mean()), 5) if v.any() else None,
            "top10_jaccard": round(float(jac[b0:b1].mean()), 4),
            "union_cosine_mean": round(float(ucos[sel].mean()), 6) if sel else None,
        })

    result = {
        "index": d["index"], "T": T, "topk_used": K,
        "argmax_agree": round(float(agree.mean()), 5),
        "nll_abs_diff": {"mean": round(float(diff[valid].mean()), 5), "p99": round(float(diff[valid].quantile(0.99)), 5), "max": round(float(diff[valid].max()), 5)},
        "train_mean_nll": round(float(d["nll_next"][:n].mean()), 5), "sglang_mean_nll": round(float(sg_nll[valid].mean()), 5),
        "top10_jaccard": round(float(jac.mean()), 4),
        "union_cosine": {"mean": round(float(ucos.mean()), 6), "min": round(float(ucos.min()), 6)},
        "union_rel_l2": {"mean": round(float(urel.mean()), 5), "max": round(float(urel.max()), 5)},
        "union_max_abs_logprob": round(float(umax.max()), 4),
        "by_position": by_pos,
    }
    if d.get("full_logits") is not None and s["top_ids"].shape[1] >= 2048:
        # full-vocab view on the short text: build a dense sglang logprob vector from its top-2048 (mass ~1)
        fl = torch.log_softmax(d["full_logits"][:n].float(), dim=-1)
        cos, rel, mx = [], [], []
        for t in range(n):
            ids = s["top_ids"][t + 1].long(); lp = s["top_logprobs"][t + 1]
            ok = ids >= 0
            a = fl[t, ids[ok]]
            b = lp[ok]
            pa, pb = torch.softmax(a, 0), torch.softmax(b, 0)
            cos.append(float(torch.dot(pa, pb) / (pa.norm() * pb.norm() + 1e-12)))
            rel.append(float((pa - pb).norm() / (pa.norm() + 1e-12)))
            mx.append(float((a - b).abs().max()))
        result["full_vocab_top2048"] = {"cosine_mean": round(sum(cos) / len(cos), 6), "cosine_min": round(min(cos), 6),
                                        "rel_l2_mean": round(sum(rel) / len(rel), 5), "max_abs_logprob": round(max(mx), 4)}
    return result


def compare_train_train(d: dict, e: dict) -> dict:
    """train-sparse vs train-dense on sample rows (full-vocab logits)."""
    assert d["T"] == e["T"] and d["sample_pos"].tolist() == e["sample_pos"].tolist()
    a, b = d["sample_logits"].float(), e["sample_logits"].float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    rel = (a - b).norm(dim=-1) / (a.norm(dim=-1) + 1e-12)
    return {"index": d["index"], "T": d["T"], "cosine_min": round(float(cos.min()), 6), "rel_l2_mean": round(float(rel.mean()), 5),
            "max_abs": round(float((a - b).abs().max()), 4), "argmax_agree": round(float((d["argmax"] == e["argmax"]).float().mean()), 5),
            "nll_abs_diff_mean": round(float((d["nll_next"] - e["nll_next"]).abs().mean()), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--sglang", required=True)
    ap.add_argument("--train-dense")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = {"train_vs_sglang": [], "sparse_vs_dense": []}
    for d, s in load_pairs(args.train, args.sglang):
        r = compare_text(d, s)
        report["train_vs_sglang"].append(r)
        print(f"text {r['index']} T={r['T']}: argmax_agree={r['argmax_agree']} nll|d| mean={r['nll_abs_diff']['mean']} "
              f"p99={r['nll_abs_diff']['p99']} top10_jac={r['top10_jaccard']} union_cos={r['union_cosine']['mean']} "
              f"rel_l2={r['union_rel_l2']['mean']}" + (f" full2048_cos={r['full_vocab_top2048']['cosine_mean']}" if "full_vocab_top2048" in r else ""))
        for b in r["by_position"]:
            print(f"    [{b['start']:>7}-{b['end']:>7}) agree={b['argmax_agree']} nll|d|={b['nll_abs_diff_mean']} jac10={b['top10_jaccard']} cos={b['union_cosine_mean']}")
    if args.train_dense:
        dense = {torch.load(f, map_location='cpu')['index']: torch.load(f, map_location='cpu') for f in glob.glob(os.path.join(args.train_dense, 'train_*.pt'))}
        for f in sorted(glob.glob(os.path.join(args.train, "train_*.pt"))):
            d = torch.load(f, map_location="cpu")
            if d["index"] in dense:
                r = compare_train_train(d, dense[d["index"]])
                report["sparse_vs_dense"].append(r)
                print(f"sparse vs dense text {r['index']} T={r['T']}: cos_min={r['cosine_min']} rel_l2={r['rel_l2_mean']} max_abs={r['max_abs']} argmax_agree={r['argmax_agree']} nll|d|={r['nll_abs_diff_mean']}")
    ok = all(r["argmax_agree"] > 0.995 and r["nll_abs_diff"]["mean"] < 0.01 and r["top10_jaccard"] > 0.95
             and r["union_cosine"]["mean"] > 0.999 and r["union_rel_l2"]["mean"] < 1e-2 for r in report["train_vs_sglang"])
    report["forward_pass"] = bool(ok) and bool(report["train_vs_sglang"])
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("FORWARD:", "PASS" if report["forward_pass"] else "FAIL", "->", args.out)


if __name__ == "__main__":
    main()
