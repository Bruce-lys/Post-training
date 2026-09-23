#!/usr/bin/env python3
"""Analyze the nine-step length-stratified DP scheduling experiment."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


PATTERN = re.compile(
    r"\[perf\] step=(\d+) variant=([A-Z0-9]+) phase=FULL_ITERATION "
    r"seconds=([0-9.]+) rank=(\d+)"
)
EXPECTED = {1: "W", 2: "AS", 3: "BS", 4: "BM", 5: "AM", 6: "AX1", 7: "BX1", 8: "BX2", 9: "AX2"}
PAIRS = {
    "short": ("AS", "BS"),
    "medium": ("AM", "BM"),
    "mix1": ("AX1", "BX1"),
    "mix2": ("AX2", "BX2"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--steps-per-epoch", type=int, default=3342)
    parser.add_argument("--output")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    records = {}
    for path in sorted(log_dir.glob("train.node*.log")):
        for step, label, seconds, rank in PATTERN.findall(path.read_text(encoding="utf-8", errors="replace")):
            records[(int(step), int(rank))] = {
                "step": int(step),
                "label": label,
                "seconds": float(seconds),
                "rank": int(rank),
                "node_log": path.name,
            }

    by_step = defaultdict(list)
    for record in records.values():
        by_step[record["step"]].append(record)

    steps = []
    label_times = {}
    for step, label in EXPECTED.items():
        rows = by_step[step]
        if len(rows) != 32:
            raise RuntimeError(f"step {step}: expected 32 unique ranks, got {len(rows)}")
        if {row["label"] for row in rows} != {label}:
            raise RuntimeError(f"step {step}: unexpected labels")
        values = [row["seconds"] for row in rows]
        global_seconds = max(values)
        label_times[label] = global_seconds
        steps.append(
            {
                "step": step,
                "label": label,
                "global_step_seconds": global_seconds,
                "rank_min_seconds": min(values),
                "rank_max_seconds": max(values),
                "rank_spread_seconds": max(values) - min(values),
            }
        )

    groups = {}
    for name, (a_label, b_label) in PAIRS.items():
        a = label_times[a_label]
        b = label_times[b_label]
        groups[name] = {
            "A_label": a_label,
            "B_label": b_label,
            "A_seconds": a,
            "B_seconds": b,
            "step_time_reduction_pct": (a - b) / a * 100.0,
            "throughput_speedup_pct": (a / b - 1.0) * 100.0,
        }

    mix_a = statistics.mean([groups["mix1"]["A_seconds"], groups["mix2"]["A_seconds"]])
    mix_b = statistics.mean([groups["mix1"]["B_seconds"], groups["mix2"]["B_seconds"]])
    result = {
        "log_dir": str(log_dir),
        "unique_perf_records": len(records),
        "steps": steps,
        "groups": groups,
        "representative_mix": {
            "sample_fractions": {"short": 0.375, "medium": 0.5, "long": 0.125},
            "A_mean_seconds": mix_a,
            "B_mean_seconds": mix_b,
            "step_time_reduction_pct": (mix_a - mix_b) / mix_a * 100.0,
            "throughput_speedup_pct": (mix_a / mix_b - 1.0) * 100.0,
            "projected_A_epoch_hours": mix_a * args.steps_per_epoch / 3600.0,
            "projected_B_epoch_hours": mix_b * args.steps_per_epoch / 3600.0,
        },
        "decision_note": "The representative-mix projection uses two real batches and remains a smoke-test estimate.",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
