#!/usr/bin/env python3
"""Judge a packing smoke (and the A/B equivalence pair) against the pass criteria.

  python analyze_packing_smoke.py --log-dir LOG --output-dir OUT [--ab-unpacked OUT_A --ab-packed OUT_B]
Writes LOG/result.json and prints a verdict table. Read-only apart from result.json.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from pathlib import Path

# Anchored to line start: the training log also dumps dataset samples (terminal-bench
# trajectories) that legitimately contain words like "Traceback" inside JSON strings.
FORBIDDEN = [
    r"^.*falling back to full attention",
    r"^.*QSA sparse selection disabled",
    r"^(\[rank\d+\]: )?torch\.OutOfMemoryError",
    r"^(\[rank\d+\]: )?.*CUDA out of memory\. Tried to allocate",
    r"^\[[0-9: -]+\] \S+ \[\d+\] NCCL WARN .*(timed out|Timeout|abort|unhandled|failed)",
    r"^(\[rank\d+\]: )?Traceback \(most recent call last\):",
    r"^.*packed_ple_varlen\.py does not match",
    r"^.*ChildFailedError",
]
REQUIRED_BY_MODE = {
    "packing": [r"packed PLE (patch|split enabled)", r"Chunked PLE compute enabled", r"Setting QSA_SPARSE_KERNEL: True"],
    "nopack": [r"\[dpbalance-order\] epoch=", r"Chunked PLE compute enabled", r"Setting QSA_SPARSE_KERNEL: True"],
}
REQUIRED = REQUIRED_BY_MODE["packing"]
PERF = re.compile(r"\[perf\] step=(\d+) variant=(\S+) phase=FULL_ITERATION seconds=([0-9.]+) rank=(\d+)")
MEM = re.compile(r"\[memprobe\] rank=(\d+) host=(\S+) max_alloc=([0-9.]+)GiB max_reserved=([0-9.]+)GiB")
SUMMARY = Path("/kwkj-k8s/llm_team/lys/megatron-swift/multinode/smoke_data/packing_smoke/summary.json")


def read_logging(output_dir: Path):
    files = sorted(glob.glob(str(output_dir / "v*/logging.jsonl")))
    if not files:
        return None, []
    rows, meta = [], {}
    for line in open(files[-1], encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "iteration" in r:
            rows.append(r)
        elif "train_dataset" in r:
            meta = r
    return meta, rows


def scan_logs(log_dir: Path, mode: str = "packing"):
    text = {}
    for p in sorted(log_dir.glob("train.node*.log")):
        text[p.name] = p.read_text(encoding="utf-8", errors="replace")
    forbidden = {pat: [n for n, t in text.items() if re.search(pat, t, re.MULTILINE)] for pat in FORBIDDEN}
    required = {pat: [n for n, t in text.items() if re.search(pat, t, re.MULTILINE)] for pat in REQUIRED_BY_MODE[mode]}
    perf = {}
    for t in text.values():
        for step, label, sec, rank in PERF.findall(t):
            d = perf.setdefault(int(step), {"label": label, "max_s": 0.0, "rank0_s": None})
            d["max_s"] = max(d["max_s"], float(sec))
            if rank == "0":
                d["rank0_s"] = float(sec)
    mem = {}
    for t in text.values():
        for rank, host, alloc, reserved in MEM.findall(t):
            rank = int(rank)
            cur = mem.get(rank)
            if cur is None or float(alloc) > cur["max_alloc"]:
                mem[rank] = {
                    "host": host,
                    "max_alloc": float(alloc),
                    "max_reserved": float(reserved),
                    "stage": 0 if rank < 16 else 1,
                }
    return forbidden, required, mem, sorted(text), perf


def gpu_stats(log_dir: Path):
    out = {}
    for p in sorted(log_dir.glob("gpu.node*.csv")):
        rows = list(csv.DictReader(p.open()))
        if not rows:
            continue
        ts = sorted({int(r["epoch_s"]) for r in rows})
        cut = ts[int(len(ts) * 0.4)]
        steady = [r for r in rows if int(r["epoch_s"]) >= cut]
        out[p.name] = {
            "util_mean_steady": round(statistics.mean(float(r["utilization_gpu_pct"]) for r in steady), 1),
            "mem_used_peak_mib": max(int(float(r["memory_used_mib"])) for r in rows),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ab-unpacked")
    ap.add_argument("--ab-packed")
    ap.add_argument("--max-alloc-gib", type=float, default=125.0)
    ap.add_argument("--max-steady-step-s", type=float, default=100.0)
    ap.add_argument("--mode", choices=["packing", "nopack"], default="packing")
    args = ap.parse_args()
    log_dir, out_dir = Path(args.log_dir), Path(args.output_dir)
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}

    forbidden, required, mem, logs, perf = scan_logs(log_dir, args.mode)
    meta, rows = read_logging(out_dir)
    checks = {}
    checks["logs_present"] = len(logs) == 4
    checks["no_forbidden_strings"] = not any(forbidden.values())
    checks["required_strings"] = all(required.values())
    checks["memprobe_all_32_ranks"] = len(mem) == 32
    checks["max_alloc_within_limit"] = bool(mem) and max(m["max_alloc"] for m in mem.values()) <= args.max_alloc_gib
    steps = [(r["iteration"], r.get("train_speed(s/it)"), r.get("loss"), r.get("grad_norm")) for r in rows]
    losses = [r["loss"] for r in rows if isinstance(r.get("loss"), (int, float))]
    checks["loss_sane"] = bool(losses) and all(0.3 <= x <= 1.0 for x in losses)
    steady = [r["train_speed(s/it)"] for r in rows[3:] if "train_speed(s/it)" in r]
    checks["steady_step_time_ok"] = (not steady) or statistics.median(steady) <= args.max_steady_step_s
    packs = None
    if meta:
        m = re.search(r"size=(\d+)", meta.get("train_dataset", ""))
        packs = int(m.group(1)) if m else None
    est = summary.get("subset", {}).get("ffd_packs_estimate")
    is_ab = bool(args.ab_unpacked and args.ab_packed) or "ab_" in log_dir.name
    if args.mode == "nopack":
        checks["pack_count_vs_estimate"] = True
    elif is_ab:
        ab_packs = summary.get("ab", {}).get("ffd_packs")
        checks["pack_count_vs_estimate"] = packs is None or ab_packs is None or packs == ab_packs
    else:
        checks["pack_count_vs_estimate"] = packs is None or est is None or abs(packs - est) <= 0.1 * est

    result = {
        "log_dir": str(log_dir),
        "output_dir": str(out_dir),
        "node_logs": logs,
        "forbidden_hits": {k: v for k, v in forbidden.items() if v},
        "required_hits": required,
        "memprobe": {str(k): v for k, v in sorted(mem.items())},
        "stage_max_alloc_gib": {
            s: max((m["max_alloc"] for m in mem.values() if m["stage"] == s), default=None) for s in (0, 1)
        },
        "steps": steps,
        "steady_median_s": statistics.median(steady) if steady else None,
        "packed_rows": packs,
        "ffd_estimate": est,
        "gpu": gpu_stats(log_dir),
        "perf_full_iteration": {str(k): v for k, v in sorted(perf.items())},
        "checks": checks,
    }

    if args.ab_unpacked and args.ab_packed:
        _, a = read_logging(Path(args.ab_unpacked))
        _, b = read_logging(Path(args.ab_packed))
        if a and b:
            la, lb = a[0]["loss"], b[0]["loss"]
            ga, gb = a[0].get("grad_norm"), b[0].get("grad_norm")
            dl = abs(la - lb)
            dg = abs(ga - gb) / max(abs(ga), 1e-9) if ga is not None and gb is not None else None
            if dl <= 5e-3 and (dg is None or dg <= 0.02):
                verdict = "pass"
            elif dl <= 2e-2:
                verdict = "investigate"
            else:
                verdict = "fail"
            result["ab"] = {
                "loss_unpacked": la,
                "loss_packed": lb,
                "loss_abs_diff": dl,
                "grad_norm_unpacked": ga,
                "grad_norm_packed": gb,
                "grad_norm_rel_diff": dg,
                "verdict": verdict,
            }
            checks["ab_equivalence"] = verdict == "pass"
        else:
            checks["ab_equivalence"] = False
            result["ab"] = {"error": "missing logging.jsonl for one of the A/B runs"}

    result["overall"] = all(checks.values())
    (log_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'check':32s} result")
    for k, v in checks.items():
        print(f"{k:32s} {'PASS' if v else 'FAIL'}")
    print("overall:", "PASS" if result["overall"] else "FAIL")
    print(
        "stage max_alloc GiB:", result["stage_max_alloc_gib"],
        "| steady median s:", result["steady_median_s"],
        "| packed rows:", packs, "(est", est, ")",
    )
    if perf:
        print("[perf] FULL_ITERATION max over ranks:", {k: round(v["max_s"], 1) for k, v in sorted(perf.items())})
    if "ab" in result:
        print("A/B:", json.dumps(result["ab"]))
    print("wrote", log_dir / "result.json")


if __name__ == "__main__":
    main()
