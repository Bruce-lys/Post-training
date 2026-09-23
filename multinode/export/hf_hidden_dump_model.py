#!/usr/bin/env python3
"""transformers fp32: dump per-layer hidden states (decoder layer outputs = hyper-connection stream) for one text.

  python hf_hidden_dump.py --train-dir OUT/fwd_check/train_sparse --index 0 --out OUT/fwd_check/hidden_hf_fp32 --max-memory-gib 118
Writes out/layers.pt: {"embed": [T,H], "layer_{i}": [T,nH] for i in 0..L-1 (0-based, output of layer i),
"mixer": [T,H] (after hyper_connection_mixer), "final_norm": [T,H], "logits_argmax": [T]} all fp32 on CPU.
"""
from __future__ import annotations

import argparse
import os
import re
import time

import torch

MODEL = "/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--model", default=MODEL, help="HF model dir (default: base); pass the exported merged checkpoint")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-memory-gib", type=int, default=118)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="auto", trust_remote_code=True,
        max_memory={i: f"{args.max_memory_gib}GiB" for i in range(torch.cuda.device_count())})
    model.eval()
    print(f"[hfhid] loaded {type(model).__name__} in {time.time() - t0:.0f}s", flush=True)

    store: dict[str, torch.Tensor] = {}
    names = [n for n, _ in model.named_modules()]
    layer_pat = re.compile(r"(?:^|\.)layers\.(\d+)$")
    hooks = []

    def keep(key):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            store[key] = t.detach().float().reshape(-1, t.shape[-1]).cpu()
        return hook

    picked = {}
    for n, m in model.named_modules():
        mm = layer_pat.search(n)
        if mm and "visual" not in n and "mtp" not in n:
            picked[f"layer_{int(mm.group(1))}"] = n
            hooks.append(m.register_forward_hook(keep(f"layer_{int(mm.group(1))}")))
        elif n.endswith("embed_tokens") and "visual" not in n:
            picked["embed"] = n; hooks.append(m.register_forward_hook(keep("embed")))
        elif n.endswith("hyper_connection_mixer") and "layers." not in n:
            picked["mixer"] = n; hooks.append(m.register_forward_hook(keep("mixer")))
        elif re.search(r"(^|\.)language_model\.norm$|(^|\.)model\.norm$", n):
            picked["final_norm"] = n; hooks.append(m.register_forward_hook(keep("final_norm")))
    print("[hfhid] hooked:", {k: v for k, v in picked.items() if not k.startswith("layer_")}, "layers:", sum(k.startswith("layer_") for k in picked), flush=True)

    d = torch.load(os.path.join(args.train_dir, f"train_{args.index}.pt"), map_location="cpu")
    ids = d["input_ids"].long()
    first = next(model.parameters()).device
    with torch.no_grad():
        out = model(input_ids=ids[None].to(first), use_cache=False)
    store["logits_argmax"] = out.logits[0].float().argmax(-1).cpu()
    store["logits_row0_16"] = out.logits[0, :16].float().cpu()
    for h in hooks:
        h.remove()
    os.makedirs(args.out, exist_ok=True)
    torch.save({"index": args.index, "T": int(ids.numel()), "dtype": args.dtype, "modules": picked, **store}, os.path.join(args.out, "layers.pt"))
    print("[hfhid] saved", os.path.join(args.out, "layers.pt"), "keys:", sorted(k for k in store if not k.startswith("layer_"))[:6], "n_layers:", sum(k.startswith("layer_") for k in store), flush=True)
    print("[hfhid] DONE", flush=True)


if __name__ == "__main__":
    main()
