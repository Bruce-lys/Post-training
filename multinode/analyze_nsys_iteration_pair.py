#!/usr/bin/env python3
"""Compare first and repeated Nsight iterations using per-rank NVTX windows."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


MASK_PROCESS = -16777216


def union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    left, right = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        elif end > right:
            right = end
    return total + right - left


def largest_gap_ns(intervals: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    if not intervals:
        return end - start, start
    merged: list[list[int]] = []
    for left, right in sorted(intervals):
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    best_size = merged[0][0] - start
    best_start = start
    for previous, current in zip(merged, merged[1:]):
        size = current[0] - previous[1]
        if size > best_size:
            best_size = size
            best_start = previous[1]
    tail = end - merged[-1][1]
    if tail > best_size:
        best_size = tail
        best_start = merged[-1][1]
    return best_size, best_start


def seconds(value: int | float) -> float:
    return round(value / 1e9, 6)


def category(name: str) -> str:
    lower = name.lower()
    if "nccl" not in lower:
        return "compute"
    if "sendrecv" in lower:
        return "sendrecv"
    if "allgather" in lower:
        return "allgather"
    if "reducescatter" in lower:
        return "reducescatter"
    if "allreduce" in lower:
        return "allreduce"
    if "broadcast" in lower:
        return "broadcast"
    return "nccl_other"


def analyze_db(
    db_path: Path,
    run: str,
    node: str,
    stage: int,
    global_rank_base: int,
    iteration: int,
) -> list[dict]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    window_name = f"FULL_ITERATION_{iteration}"
    windows = list(
        connection.execute(
            """
            SELECT start, end, globalTid
            FROM NVTX_EVENTS
            WHERE text = ? AND end IS NOT NULL
            ORDER BY (globalTid & ?)
            """,
            (window_name, MASK_PROCESS),
        )
    )
    if len(windows) != 8:
        raise RuntimeError(f"{db_path}: expected 8 {window_name} ranges, found {len(windows)}")

    results = []
    for local_rank, (window_start, window_end, global_tid) in enumerate(windows):
        global_pid = global_tid & MASK_PROCESS
        rows = connection.execute(
            """
            SELECT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            WHERE k.globalPid = ? AND k.end > ? AND k.start < ?
            ORDER BY k.start
            """,
            (global_pid, window_start, window_end),
        )

        intervals: dict[str, list[tuple[int, int]]] = {
            "all": [],
            "nccl": [],
            "compute": [],
            "sendrecv": [],
            "allgather": [],
            "allreduce": [],
            "reducescatter": [],
            "broadcast": [],
            "nccl_other": [],
            "qsa": [],
            "ple": [],
        }
        counts = {"qsa": 0, "ple": 0}
        longest_comm = (0, "", 0)
        for kernel_start, kernel_end, name in rows:
            left = max(kernel_start, window_start)
            right = min(kernel_end, window_end)
            if right <= left:
                continue
            interval = (left, right)
            intervals["all"].append(interval)
            kind = category(name)
            if kind == "compute":
                intervals["compute"].append(interval)
            else:
                intervals["nccl"].append(interval)
                intervals[kind].append(interval)
                duration = right - left
                if duration > longest_comm[0]:
                    longest_comm = (duration, name, left)
            lower = name.lower()
            if "_qsa_" in lower:
                intervals["qsa"].append(interval)
                counts["qsa"] += 1
            if "_ple_" in lower:
                intervals["ple"].append(interval)
                counts["ple"] += 1

        gap_size, gap_start = largest_gap_ns(intervals["all"], window_start, window_end)
        cpu_ranges: dict[str, list[int]] = {"ncclCommInitRankConfig": [], "ncclGroupEnd": []}
        for event_start, event_end, name in connection.execute(
            """
            SELECT n.start, n.end, COALESCE(n.text, s.value)
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON s.id = n.textId
            WHERE (n.globalTid & ?) = ?
              AND n.end IS NOT NULL AND n.end > ? AND n.start < ?
              AND COALESCE(n.text, s.value) IN ('ncclCommInitRankConfig', 'ncclGroupEnd')
            """,
            (MASK_PROCESS, global_pid, window_start, window_end),
        ):
            cpu_ranges[name].append(min(event_end, window_end) - max(event_start, window_start))

        active = union_ns(intervals["all"])
        item = {
            "run": run,
            "node": node,
            "stage": stage,
            "local_rank": local_rank,
            "global_rank": global_rank_base + local_rank,
            "iteration": iteration,
            "iteration_s": seconds(window_end - window_start),
            "gpu_active_s": seconds(active),
            "gpu_idle_s": seconds(window_end - window_start - active),
            "nccl_s": seconds(union_ns(intervals["nccl"])),
            "compute_s": seconds(union_ns(intervals["compute"])),
            "sendrecv_s": seconds(union_ns(intervals["sendrecv"])),
            "allgather_s": seconds(union_ns(intervals["allgather"])),
            "allreduce_s": seconds(union_ns(intervals["allreduce"])),
            "reducescatter_s": seconds(union_ns(intervals["reducescatter"])),
            "qsa_s": seconds(union_ns(intervals["qsa"])),
            "qsa_count": counts["qsa"],
            "ple_s": seconds(union_ns(intervals["ple"])),
            "ple_count": counts["ple"],
            "largest_gpu_gap_s": seconds(gap_size),
            "largest_gpu_gap_offset_s": seconds(gap_start - window_start),
            "longest_comm_s": seconds(longest_comm[0]),
            "longest_comm_name": longest_comm[1],
            "longest_comm_offset_s": seconds(longest_comm[2] - window_start),
            "comm_init_s": seconds(sum(cpu_ranges["ncclCommInitRankConfig"])),
            "comm_init_count": len(cpu_ranges["ncclCommInitRankConfig"]),
            "group_end_s": seconds(sum(cpu_ranges["ncclGroupEnd"])),
            "group_end_max_s": seconds(max(cpu_ranges["ncclGroupEnd"], default=0)),
        }
        results.append(item)
    connection.close()
    return results


def aggregate(rows: list[dict]) -> list[dict]:
    numeric_keys = [
        "iteration_s",
        "gpu_active_s",
        "gpu_idle_s",
        "nccl_s",
        "compute_s",
        "sendrecv_s",
        "allgather_s",
        "allreduce_s",
        "reducescatter_s",
        "qsa_s",
        "qsa_count",
        "ple_s",
        "ple_count",
        "largest_gpu_gap_s",
        "longest_comm_s",
        "comm_init_s",
        "comm_init_count",
        "group_end_s",
        "group_end_max_s",
    ]
    output = []
    groups = sorted({(row["run"], row["stage"], row["node"]) for row in rows})
    for run, stage, node in groups:
        selected = [row for row in rows if (row["run"], row["stage"], row["node"]) == (run, stage, node)]
        item = {"run": run, "stage": stage, "node": node, "ranks": len(selected)}
        for key in numeric_keys:
            values = [float(row[key]) for row in selected]
            item[f"{key}_mean"] = round(statistics.mean(values), 6)
            item[f"{key}_min"] = round(min(values), 6)
            item[f"{key}_max"] = round(max(values), 6)
        output.append(item)
    return output


def print_summary(summary: list[dict]) -> None:
    columns = [
        "run",
        "stage",
        "iteration_s_mean",
        "gpu_active_s_mean",
        "gpu_idle_s_mean",
        "nccl_s_mean",
        "compute_s_mean",
        "sendrecv_s_mean",
        "allgather_s_mean",
        "allreduce_s_mean",
        "reducescatter_s_mean",
        "qsa_s_mean",
        "qsa_count_mean",
        "ple_s_mean",
        "largest_gpu_gap_s_max",
        "longest_comm_s_max",
        "comm_init_s_mean",
        "group_end_max_s_max",
    ]
    print("\t".join(columns))
    for row in summary:
        print("\t".join(str(row[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-node0", type=Path, required=True)
    parser.add_argument("--first-node2", type=Path, required=True)
    parser.add_argument("--repeat-node0", type=Path, required=True)
    parser.add_argument("--repeat-node2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = [
        (args.first_node0, "first", "hd03-gpu2-0019", 0, 0, 1),
        (args.first_node2, "first", "hd03-gpu2-0005", 1, 16, 1),
        (args.repeat_node0, "repeat", "hd03-gpu2-0019", 0, 0, 3),
        (args.repeat_node2, "repeat", "hd03-gpu2-0005", 1, 16, 3),
    ]
    rows: list[dict] = []
    for db_path, run, node, stage, rank_base, iteration in inputs:
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        rows.extend(analyze_db(db_path, run, node, stage, rank_base, iteration))
    summary = aggregate(rows)
    payload = {"per_rank": rows, "stage_summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
