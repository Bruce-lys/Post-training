#!/usr/bin/env python3
"""Summarize the four-node DP balancing A/B experiment from rank logs."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


PATTERN = re.compile(
    r"\[perf\] step=(\d+) variant=([WAB]) phase=FULL_ITERATION "
    r"seconds=([0-9.]+) rank=(\d+)"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--output")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    records = {}
    for path in sorted(log_dir.glob("train.node*.log")):
        for match in PATTERN.finditer(path.read_text(encoding="utf-8", errors="replace")):
            step, variant, seconds, rank = match.groups()
            key = (int(step), int(rank))
            records[key] = {
                "step": int(step),
                "variant": variant,
                "seconds": float(seconds),
                "rank": int(rank),
                "node_log": path.name,
            }

    by_step = defaultdict(list)
    for record in records.values():
        by_step[record["step"]].append(record)

    expected = {1: "W", 2: "A", 3: "B", 4: "B", 5: "A"}
    steps = []
    for step, variant in expected.items():
        rows = by_step[step]
        if len(rows) != 32:
            raise RuntimeError(f"step {step}: expected 32 unique ranks, got {len(rows)}")
        if {row["variant"] for row in rows} != {variant}:
            raise RuntimeError(f"step {step}: unexpected variant labels")
        values = [row["seconds"] for row in rows]
        steps.append(
            {
                "step": step,
                "variant": variant,
                "global_step_seconds": max(values),
                "rank_min_seconds": min(values),
                "rank_mean_seconds": statistics.mean(values),
                "rank_max_seconds": max(values),
                "rank_spread_seconds": max(values) - min(values),
            }
        )

    a_values = [row["global_step_seconds"] for row in steps if row["variant"] == "A"]
    b_values = [row["global_step_seconds"] for row in steps if row["variant"] == "B"]
    a_mean = statistics.mean(a_values)
    b_mean = statistics.mean(b_values)
    speedup_pct = (a_mean / b_mean - 1.0) * 100.0
    time_reduction_pct = (a_mean - b_mean) / a_mean * 100.0
    result = {
        "log_dir": str(log_dir),
        "unique_perf_records": len(records),
        "steps": steps,
        "A_global_step_mean_seconds": a_mean,
        "B_global_step_mean_seconds": b_mean,
        "throughput_speedup_pct_B_vs_A": speedup_pct,
        "step_time_reduction_pct_B_vs_A": time_reduction_pct,
        "decision_note": (
            "Treat this as a positive first signal only if both B steps are faster than both A steps. "
            "A production change still requires a confirmation run on broader real-data batches."
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
