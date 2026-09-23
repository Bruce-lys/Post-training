#!/usr/bin/env python3
"""Layer-by-layer comparison of the two fp32 hidden-state dumps (HF vs Megatron)."""
import sys
import torch

a = torch.load(sys.argv[1], map_location="cpu")  # hf
b = torch.load(sys.argv[2], map_location="cpu")  # megatron
print("HF modules:", {k: v for k, v in a["modules"].items() if not k.startswith("layer_")})
print("MG modules:", {k: v for k, v in b["modules"].items() if not k.startswith("layer_")})
if "input_ids" in b:
    print("input_ids equal to HF text:", int(b["input_ids"].numel()), "tokens")
keys = ["embed"] + [f"layer_{i}" for i in range(48)] + ["mixer", "final_norm"]
print(f"{'key':>12} {'hf shape':>16} {'mg shape':>16} {'rel_l2':>10} {'max_abs':>10} {'cos_min':>9}")
for k in keys:
    if k not in a or k not in b:
        print(f"{k:>12} {'-' if k not in a else tuple(a[k].shape)!s:>16} {'-' if k not in b else tuple(b[k].shape)!s:>16}")
        continue
    x, y = a[k].float(), b[k].float()
    if x.shape != y.shape:
        print(f"{k:>12} {tuple(x.shape)!s:>16} {tuple(y.shape)!s:>16}  SHAPE MISMATCH")
        continue
    rel = float((x - y).norm() / (x.norm() + 1e-12))
    mx = float((x - y).abs().max())
    cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    print(f"{k:>12} {tuple(x.shape)!s:>16} {tuple(y.shape)!s:>16} {rel:10.2e} {mx:10.3e} {float(cos.min()):9.5f}")
if "logits_argmax" in a and "logits_argmax" in b:
    print("logits argmax agree:", float((a["logits_argmax"] == b["logits_argmax"]).float().mean()))
    print("logits rows 0..3 max|d|:", float((a["logits_row0_16"][:4] - b["logits_row0_16"][:4]).abs().max()))
