#!/usr/bin/env python3
"""GPU-free stand-in for ``megatron sft``: writes <out>/logging.jsonl rows in ms-swift's format at a steady pace.

Used with ``fake_train=true`` in settings/ui.json to exercise launcher, registry, monitor and abort end to end
without touching the cluster's GPUs. Exit code 0 on completion, 143 when aborted with SIGTERM."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--interval", type=float, default=2.0)
ap.add_argument("--run-id", default="fake")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
metrics = os.path.join(args.out, "logging.jsonl")
stop = False


def _term(signum, _frame):
    global stop
    stop = True
    print(f"[fake_train] signal {signum} received, stopping", flush=True)


signal.signal(signal.SIGTERM, _term)
signal.signal(signal.SIGINT, _term)


def append(row: dict) -> None:
    with open(metrics, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s" if m else f"{s}s"


print(f"[INFO:swift] fake train run_id={args.run_id} iters={args.iters} interval={args.interval}s", flush=True)
print(f"[INFO:swift] logging_path: {metrics}", flush=True)
random.seed(2026)
t0 = time.time()
done = 0
for it in range(1, args.iters + 1):
    if stop:
        break
    time.sleep(args.interval)
    loss = 0.9 * math.exp(-it / max(4, args.iters / 3)) + 0.35 + random.uniform(-0.02, 0.02)
    row = {"loss": round(loss, 8), "grad_norm": round(0.8 + random.uniform(-0.3, 0.5), 8),
           "learning_rate": round(1e-4 * (0.5 * (1 + math.cos(math.pi * it / args.iters))), 10),
           "load_balancing_loss": round(1.8 - 0.4 * it / args.iters + random.uniform(-0.05, 0.05), 8),
           "iteration": f"{it}/{args.iters}", "elapsed_time": fmt(time.time() - t0),
           "remaining_time": fmt((args.iters - it) * args.interval), "memory(GiB)": round(90 + random.uniform(0, 6), 2),
           "train_speed(s/it)": round(args.interval, 6)}
    append(row)
    print(str(row), flush=True)
    print(f"Train: {int(100 * it / args.iters):3d}%|{'█' * (it * 20 // args.iters):<20}| {it}/{args.iters}", flush=True)
    done = it
if not stop:
    ckpt = os.path.join(args.out, f"checkpoint-{done}")
    os.makedirs(ckpt, exist_ok=True)
    append({"last_model_checkpoint": ckpt, "best_model_checkpoint": None, "best_metric": None})
    print(f"[INFO:swift] last_model_checkpoint: {ckpt}", flush=True)
print("[fake_train] done" if not stop else "[fake_train] aborted", flush=True)
sys.exit(0 if not stop else 143)
