#!/usr/bin/env python3
"""Analyze three PP split smoke runs from four-node logs and GPU samples."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


PERF_RE = re.compile(
    r"\[ppperf\] step=(\d+) label=(\S+) split=(\d+) phase=(\S+) "
    r"seconds=([0-9.]+) rank=(\d+) host=(\S+)"
)
ERROR_RE = re.compile(
    r"CUDA error:|CUDA out of memory|torch\.OutOfMemoryError|NCCL WARN|"
    r"NCCL.*(?:unhandled|error|failed|timed out)|Traceback \(most recent call last\)|"
    r"ChildFailedError|DistBackendError",
    re.IGNORECASE,
)
MEASURED_STEPS = (3, 4, 5)


def median(values: list[float]) -> float:
    return round(statistics.median(values), 6)


def parse_run(split: int, run_dir: Path) -> dict:
    records = []
    errors = []
    for node in range(4):
        log_path = run_dir / f"train.node{node}.log"
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        for line_number, line in enumerate(log_path.open(encoding="utf-8", errors="replace"), 1):
            for match in PERF_RE.finditer(line):
                step, label, found_split, phase, elapsed, rank, host = match.groups()
                if int(found_split) != split:
                    raise RuntimeError(f"{log_path}:{line_number}: split={found_split}, expected {split}")
                records.append(
                    {
                        "step": int(step),
                        "label": label,
                        "split": split,
                        "phase": phase,
                        "seconds": float(elapsed),
                        "rank": int(rank),
                        "host": host,
                    }
                )
            if ERROR_RE.search(line):
                errors.append({"file": str(log_path), "line": line_number, "text": line.strip()[:500]})

    phase_keys = {(item["rank"], item["step"], item["phase"]) for item in records}
    if len(phase_keys) != len(records):
        raise RuntimeError(f"{run_dir}: duplicate rank/step/phase timing records")

    selected_ranks = []
    for rank_base in (0, 8, 16, 24):
        candidates = []
        for rank in range(rank_base, rank_base + 8):
            keys = {
                (item["step"], item["phase"])
                for item in records
                if item["rank"] == rank
            }
            if all(
                (step, phase) in keys
                for step in MEASURED_STEPS
                for phase in ("TRAIN_STEP", "FULL_ITERATION")
            ):
                candidates.append(rank)
        if not candidates:
            raise RuntimeError(
                f"{run_dir}: no complete representative rank in [{rank_base}, {rank_base + 7}]"
            )
        selected_ranks.append(min(candidates))

    steps = []
    for step in MEASURED_STEPS:
        full_by_rank = {
            item["rank"]: item["seconds"]
            for item in records
            if item["step"] == step
            and item["phase"] == "FULL_ITERATION"
            and item["rank"] in selected_ranks
        }
        train_by_rank = {
            item["rank"]: item["seconds"]
            for item in records
            if item["step"] == step
            and item["phase"] == "TRAIN_STEP"
            and item["rank"] in selected_ranks
        }
        if set(full_by_rank) != set(selected_ranks) or set(train_by_rank) != set(selected_ranks):
            raise RuntimeError(f"{run_dir}: incomplete representative timing for measured step {step}")
        stage0_value = train_by_rank[selected_ranks[0]]
        stage1_value = train_by_rank[selected_ranks[2]]
        steps.append(
            {
                "step": step,
                "global_step_s": round(max(full_by_rank.values()), 6),
                "stage0_0019_train_s": round(stage0_value, 6),
                "stage1_0005_train_s": round(stage1_value, 6),
                "stage1_minus_stage0_s": round(stage1_value - stage0_value, 6),
                "representative_rank_spread_s": round(
                    max(full_by_rank.values()) - min(full_by_rank.values()), 6
                ),
            }
        )

    peak_memory: dict[str, dict[str, int]] = {}
    for node in range(4):
        path = run_dir / f"gpu.node{node}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8", errors="replace") as stream:
            for row in csv.DictReader(stream):
                try:
                    host = row["host"]
                    gpu = row["gpu_index"]
                    memory = int(row["memory_used_mib"])
                except (KeyError, TypeError, ValueError):
                    continue
                host_peaks = peak_memory.setdefault(host, {})
                host_peaks[gpu] = max(host_peaks.get(gpu, 0), memory)

    global_values = [item["global_step_s"] for item in steps]
    stage0_values = [item["stage0_0019_train_s"] for item in steps]
    stage1_values = [item["stage1_0005_train_s"] for item in steps]
    all_peaks = [value for host in peak_memory.values() for value in host.values()]
    return {
        "split": f"{split}/{48 - split}",
        "first_stage_layers": split,
        "run_dir": str(run_dir),
        "selected_ranks": selected_ranks,
        "measured_steps": steps,
        "global_step_median_s": median(global_values),
        "global_step_min_s": round(min(global_values), 6),
        "global_step_max_s": round(max(global_values), 6),
        "stage0_0019_median_s": median(stage0_values),
        "stage1_0005_median_s": median(stage1_values),
        "stage_gap_median_s": median([b - a for a, b in zip(stage0_values, stage1_values)]),
        "peak_memory_mib": peak_memory,
        "peak_memory_max_mib": max(all_peaks, default=0),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run22", type=Path, required=True)
    parser.add_argument("--run23", type=Path, required=True)
    parser.add_argument("--run24", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [parse_run(22, args.run22), parse_run(23, args.run23), parse_run(24, args.run24)]
    baseline = runs[0]["global_step_median_s"]
    for run in runs:
        current = run["global_step_median_s"]
        run["step_time_change_vs_22_pct"] = round((current / baseline - 1.0) * 100.0, 3)
        run["throughput_change_vs_22_pct"] = round((baseline / current - 1.0) * 100.0, 3)
    eligible = [run for run in runs if not run["errors"]]
    winner = min(eligible, key=lambda item: item["global_step_median_s"]) if eligible else None
    payload = {"runs": runs, "winner": winner["split"] if winner else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("split\tglobal_median_s\tstage0_0019_s\tstage1_0005_s\tstage_gap_s\tpeak_mem_mib\tstep_delta_pct\tthroughput_delta_pct\terrors")
    for run in runs:
        print(
            f"{run['split']}\t{run['global_step_median_s']}\t{run['stage0_0019_median_s']}\t"
            f"{run['stage1_0005_median_s']}\t{run['stage_gap_median_s']}\t{run['peak_memory_max_mib']}\t"
            f"{run['step_time_change_vs_22_pct']}\t{run['throughput_change_vs_22_pct']}\t{len(run['errors'])}"
        )
    print(f"winner={payload['winner']}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
