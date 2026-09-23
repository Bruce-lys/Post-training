"""Teacher-forced per-layer comparison: a = HF hidden dump (exported merged model, fp32), b = Megatron teacher dump
(base fp32 + unmerged LoRA adapter, each layer fed the HF layer input). usage: python compare_hidden_teacher.py A/layers.pt B/layers.pt [n_real]
PASS: per-token update error, median over layers <= 3e-4 and worst-layer median <= 1e-3 (per-token medians are immune to the 1-3
routing-flip tokens per layer that dominate whole-layer L2), teacher-forced argmax agree == 1.0."""
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu"); b = torch.load(sys.argv[2], map_location="cpu")
n_real = int(sys.argv[3]) if len(sys.argv) > 3 else None
T = a["layer_0"].shape[0]
if n_real is None:
    ids = b["input_ids"]; n_real = T
    while n_real > 0 and int(ids[n_real - 1]) == 248044: n_real -= 1
print(f"T={T} real tokens={n_real} (pads excluded)")
print(f"{'layer':>8} {'kind':>4} {'cum_rel':>9} {'cum_cos_min':>11} {'cum_cos_p1':>10} {'upd_rel':>9} {'upd_cos':>8}")
for i in range(48):
    x, y = a[f"layer_{i}"].float()[:n_real], b[f"layer_{i}"].float()[:n_real]
    pa = a["embed"].float().repeat(1, x.shape[-1] // a["embed"].shape[-1]) if i == 0 else a[f"layer_{i-1}"].float()
    pb = b["embed"].float().repeat(1, x.shape[-1] // b["embed"].shape[-1]) if i == 0 else b[f"layer_{i-1}"].float()
    ua, ub = x - pa[:n_real], y - pb[:n_real]
    cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    ucos = torch.nn.functional.cosine_similarity(ua.reshape(-1), ub.reshape(-1), dim=0)
    kind = "Q" if (i + 1) % 4 == 0 else "G"
    print(f"layer_{i:<3} {kind:>4} {float((x-y).norm()/x.norm()):9.2e} {float(cos.min()):11.5f} {float(cos.quantile(0.01)):10.5f} {float((ua-ub).norm()/ua.norm()):9.2e} {float(ucos):8.5f}")
x, y = a["mixer"].float()[:n_real], b["mixer"].float()[:n_real]
cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
print(f"{'mixer':>8} {'':>4} {float((x-y).norm()/x.norm()):9.2e} {float(cos.min()):11.5f} {float(cos.quantile(0.01)):10.5f}")
am, bm = a["logits_argmax"][:n_real - 1], b["logits_argmax"][:n_real - 1]
print("argmax agree (real positions):", float((am == bm).float().mean()), " n=", n_real - 1)

import statistics as _st
_tok_med, _tok_p99 = [], []
for i in range(48):
    x, y = a[f"layer_{i}"].float()[:n_real], b[f"layer_{i}"].float()[:n_real]
    pa = a["embed"].float().repeat(1, x.shape[-1] // a["embed"].shape[-1]) if i == 0 else a[f"layer_{i-1}"].float()
    pb = b["embed"].float().repeat(1, x.shape[-1] // b["embed"].shape[-1]) if i == 0 else b[f"layer_{i-1}"].float()
    ua, ub = x - pa[:n_real], y - pb[:n_real]
    e = (ua - ub).norm(dim=-1) / (ua.norm(dim=-1) + 1e-12)   # per-token relative error of this layer's update
    _tok_med.append(float(e.median())); _tok_p99.append(float(e.quantile(0.99)))
_med = _st.median(_tok_med); _worst = max(_tok_med); _p99 = _st.median(_tok_p99)
_agree = float((am == bm).float().mean())
print("SUMMARY per-token upd_rel: median-over-layers %.2e | worst-layer median %.2e | median-over-layers p99 %.2e | teacher-forced argmax agree %.4f" % (_med, _worst, _p99, _agree))
print("criteria: median-over-layers <= 3e-4, worst-layer median <= 1e-3, argmax agree == 1.0 (base-vs-base 2026-09-18: ~1e-4 / ~3e-4 / 1.000) ->", "PASS" if (_med <= 3e-4 and _worst <= 1e-3 and _agree == 1.0) else "CHECK")
