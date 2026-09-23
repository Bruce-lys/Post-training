#!/usr/bin/env python3
"""Compare an exported (LoRA-merged) HF checkpoint against the base, tensor by tensor, without loading whole models.

  python check_export.py --base /kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next --export <OUT> [--report report.json]

Checks:
  1. config.json: identical except keys listed in ALLOWED_CONFIG_DIFFS (reported either way).
  2. tensor name sets identical (MTP, visual, ngram, router, norms must all be present in the export).
  3. dtype and shape identical per tensor.
  4. FROZEN tensors (everything not touched by LoRA: ngram tables, embeddings, norms, routers, hyper-connections, MTP,
     visual, lm_head, conv1d ...) must be byte-identical.
  5. LoRA-target tensors (linear_attn in_proj*/out_proj, self_attn q/k/v/o, expert gate_up/down, shared expert): report
     relative delta ||W'-W||/||W||, fraction of changed elements, and how many target tensors have ZERO change
     (that is the bf16 merge-rounding loss to watch).
Exit code 0 only if 2-4 pass and every LoRA-target tensor changed.
"""
import argparse, json, os, re, sys, collections
import torch
from safetensors import safe_open

# HF-side name fragments of the LoRA targets (megatron: in_proj, out_proj, linear_proj, linear_qkv, linear_fc1, linear_fc2)
TARGET_RE = re.compile(r"\.(linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj|qkv_proj|gate_proj)|mlp\.experts\.(gate_up_proj|down_proj)|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))\.")
ALLOWED_CONFIG_DIFFS = {"transformers_version", "torch_dtype", "dtype", "_name_or_path"}

def load_index(d):
    p = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(p):
        return json.load(open(p))["weight_map"]
    files = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    wm = {}
    for f in files:
        with safe_open(os.path.join(d, f), "pt") as s:
            for k in s.keys(): wm[k] = f
    return wm

def flat_cfg(o, prefix=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(flat_cfg(v, prefix + k + "."))
    else:
        out[prefix[:-1]] = o
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--export", required=True); ap.add_argument("--report")
    a = ap.parse_args()
    ok = True; rep = {"config_diffs": {}, "missing_in_export": [], "extra_in_export": [], "dtype_shape_mismatch": [], "frozen_changed": [], "targets": [], "targets_unchanged": []}
    cb, ce = flat_cfg(json.load(open(os.path.join(a.base, "config.json")))), flat_cfg(json.load(open(os.path.join(a.export, "config.json"))))
    for k in sorted(set(cb) | set(ce)):
        if cb.get(k) != ce.get(k):
            rep["config_diffs"][k] = [cb.get(k), ce.get(k)]
            if k.split(".")[-1] not in ALLOWED_CONFIG_DIFFS: ok = False
    print("config diffs:", json.dumps(rep["config_diffs"], indent=1) if rep["config_diffs"] else "none")
    wb, we = load_index(a.base), load_index(a.export)
    rep["missing_in_export"] = sorted(set(wb) - set(we)); rep["extra_in_export"] = sorted(set(we) - set(wb))
    if rep["missing_in_export"] or rep["extra_in_export"]: ok = False
    print("tensors: base %d export %d missing %d extra %d" % (len(wb), len(we), len(rep["missing_in_export"]), len(rep["extra_in_export"])))
    for n in rep["missing_in_export"][:10]: print("  MISSING", n)
    for n in rep["extra_in_export"][:10]: print("  EXTRA  ", n)
    handles = {}
    def get(d, wm, name):
        f = os.path.join(d, wm[name])
        if f not in handles: handles[f] = safe_open(f, "pt")
        return handles[f].get_tensor(name)
    common = sorted(set(wb) & set(we)); n_frozen = n_target = 0; bytes_checked = 0
    for i, name in enumerate(common):
        tb, te = get(a.base, wb, name), get(a.export, we, name)
        if tb.dtype != te.dtype or tb.shape != te.shape:
            rep["dtype_shape_mismatch"].append([name, str(tb.dtype), str(te.dtype), list(tb.shape), list(te.shape)]); ok = False; continue
        bytes_checked += tb.numel() * tb.element_size()
        same = torch.equal(tb, te)
        if TARGET_RE.search("." + name + "."):
            n_target += 1
            if same:
                rep["targets_unchanged"].append(name); ok = False
            else:
                d = (te.float() - tb.float()); rel = float(d.norm() / (tb.float().norm() + 1e-12)); frac = float((d != 0).float().mean())
                rep["targets"].append({"name": name, "rel_delta": rel, "frac_changed": frac})
        else:
            n_frozen += 1
            if not same:
                rep["frozen_changed"].append(name); ok = False
        if i % 200 == 0: print("  ... %d/%d tensors, %.1f GB compared" % (i, len(common), bytes_checked / 1e9), flush=True)
    print("frozen tensors: %d (changed: %d) | target tensors: %d (unchanged after merge: %d)" % (n_frozen, len(rep["frozen_changed"]), n_target, len(rep["targets_unchanged"])))
    for n in rep["frozen_changed"][:10]: print("  FROZEN CHANGED", n)
    for n in rep["targets_unchanged"][:10]: print("  TARGET UNCHANGED", n)
    if rep["targets"]:
        rels = sorted(t["rel_delta"] for t in rep["targets"]); fr = sorted(t["frac_changed"] for t in rep["targets"])
        q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
        print("LoRA delta rel-norm: min %.2e p10 %.2e median %.2e p90 %.2e max %.2e" % (rels[0], q(rels, .1), q(rels, .5), q(rels, .9), rels[-1]))
        print("fraction of elements changed per target tensor: min %.3f median %.3f max %.3f  (low values = bf16 rounding swallowed the LoRA delta)" % (fr[0], q(fr, .5), fr[-1]))
        fam = collections.defaultdict(list)
        for t in rep["targets"]: fam[re.sub(r"\.\d+\.", ".N.", t["name"])].append(t["rel_delta"])
        for k, v in sorted(fam.items()): print("  %-90s n=%3d median rel_delta %.2e" % (k, len(v), sorted(v)[len(v) // 2]))
    rep["ok"] = ok
    if a.report: json.dump(rep, open(a.report, "w"), indent=1)
    print("RESULT:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
