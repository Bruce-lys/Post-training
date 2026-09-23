#!/usr/bin/env python3
"""Pool Qwen/GLM pass trajectories and keep a short subset per task.

Per task, order by assistant-turn count and take the shortest 10.
Tasks that have 10 then drop trajectories that are long relative to
the shortest one:
  min <= 100: drop turns > 1.75 * min
  100 < min <= 150: drop turns > 1.5 * min
  min > 150: drop turns > min + 55
"""
from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path("/kwkj-k8s/llm_team/lys/megatron-swift/zhh-tb4")
OUT_JSONL = OUT_DIR / "traj_qwen_glm_top10_by_turns.jsonl"
OUT_SUMMARY = OUT_DIR / "traj_qwen_glm_top10_by_turns.summary.json"
KEEP = 10
RESAMPLE_TARGET = 15

SOURCES = [
    {
        "name": "qwen38_pass_226",
        "model": "qwen",
        "path": "/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_swift_agent_traj_v2.jsonl",
        "schema": "swift_agent",
    },
    {
        "name": "qwen38_stable7_pass_89",
        "model": "qwen",
        "path": "/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/runs/QWEN38_PASS_TRAJ/qwen38_stable7_pass_89.jsonl",
        "schema": "openai",
    },
    {
        "name": "glm53_pass_128",
        "model": "glm",
        "path": "/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/runs/GLM53_PASS_TRAJ/glm53_pass_128.jsonl",
        "schema": "openai",
    },
]


def _parse_tool_calls(raw):
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)
    raise TypeError(type(raw).__name__)


def _parse_arguments(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def fold_think(content, reasoning) -> str:
    body = "" if content is None else content
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False)
    if reasoning and "<think>" not in body:
        rc = reasoning if isinstance(reasoning, str) else json.dumps(reasoning, ensure_ascii=False)
        return f"<think>\n{rc}\n</think>\n\n{body}".rstrip()
    return body


def trial_key(trial) -> str:
    if not trial:
        return ""
    return str(trial).rstrip("/").split("/")[-1]


def task_of(meta: dict) -> str:
    return str(meta.get("task_id") or meta.get("task") or "unknown")


def convert_openai(row: dict, source: dict, src_index: int) -> dict:
    out = []
    for m in row.get("messages") or []:
        role = m.get("role")
        if role in ("system", "user"):
            out.append({"role": role, "content": m.get("content") or ""})
        elif role == "assistant":
            if not any(x["role"] == "user" for x in out):
                continue
            rc = m.get("reasoning_content")
            if rc is None:
                rc = m.get("reasoning", m.get("thinking"))
            out.append({"role": "assistant", "content": fold_think(m.get("content"), rc)})
            for tc in _parse_tool_calls(m.get("tool_calls")):
                fn = tc.get("function") or {}
                out.append({
                    "role": "tool_call",
                    "content": json.dumps(
                        {"name": fn.get("name"), "arguments": _parse_arguments(fn.get("arguments"))},
                        ensure_ascii=False,
                    ),
                })
        elif role == "tool":
            content = m.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({"role": "tool_response", "content": content or ""})
        else:
            raise ValueError(f"{source['name']} #{src_index}: unknown role {role!r}")
    meta_in = dict(row.get("meta") or {})
    return _pack(out, row.get("tools") or [], meta_in, source, src_index)


def convert_swift(row: dict, source: dict, src_index: int) -> dict:
    meta_in = dict(row.get("meta") or {})
    return _pack(row["messages"], row.get("tools") or [], meta_in, source, src_index)


def _pack(messages, tools, meta_in, source, src_index) -> dict:
    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    n_think = sum(1 for m in messages if m["role"] == "assistant" and "<think>" in (m.get("content") or ""))
    n_extra_user = sum(1 for i, m in enumerate(messages) if i > 1 and m["role"] == "user")
    tools_s = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
    trial = meta_in.get("trial_dir")
    return {
        "messages": messages,
        "tools": tools_s,
        "meta": {
            "source": source["name"],
            "model": source["model"],
            "task_id": task_of(meta_in),
            "reward": meta_in.get("reward", 1.0),
            "trial_dir": trial,
            "trial_key": trial_key(trial),
            "steps": meta_in.get("steps"),
            "src_index": src_index,
            "n_messages": len(messages),
            "n_assistant": n_asst,
            "n_assistant_with_think": n_think,
            "n_assistant_without_think": n_asst - n_think,
            "n_extra_user": n_extra_user,
            "has_thinking": n_think > 0,
        },
    }


def spread_limit(min_turns: int) -> float:
    """Upper bound on assistant turns, given the shortest trajectory of the task."""
    if min_turns > 150:
        return min_turns + 55
    if min_turns > 100:
        return 1.5 * min_turns
    return 1.75 * min_turns


def main() -> None:
    seen = {}
    dup = Counter()
    loaded = Counter()
    for source in SOURCES:
        with open(source["path"]) as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                rec = convert_swift(row, source, idx) if source["schema"] == "swift_agent" else convert_openai(row, source, idx)
                loaded[source["name"]] += 1
                key = rec["meta"]["trial_key"] or f"{source['name']}:{idx}"
                if key in seen:
                    dup[f"{seen[key]['meta']['source']} == {source['name']}"] += 1
                    continue
                seen[key] = rec

    by_task = defaultdict(list)
    for rec in seen.values():
        by_task[rec["meta"]["task_id"]].append(rec)

    kept = []
    per_task = {}
    for task, items in sorted(by_task.items()):
        items.sort(key=lambda r: (r["meta"]["n_assistant"], r["meta"]["model"], r["meta"]["trial_key"]))
        shortest = items[:KEEP]
        min_turns = shortest[0]["meta"]["n_assistant"]
        dropped = []
        if len(items) >= KEEP:
            limit = spread_limit(min_turns)
            chosen = [r for r in shortest if r["meta"]["n_assistant"] <= limit]
            dropped = [r for r in shortest if r["meta"]["n_assistant"] > limit]
            rule = "min+55" if min_turns > 150 else ("1.5xmin" if min_turns > 100 else "1.75xmin")
        else:
            limit = None
            chosen = shortest
            rule = "under_10_no_spread_filter"
        turns = [r["meta"]["n_assistant"] for r in chosen]
        avail_turns = [r["meta"]["n_assistant"] for r in items]
        for rank, rec in enumerate(chosen, start=1):
            rec["meta"]["rank_by_turns"] = rank
            rec["meta"]["n_available"] = len(items)
            rec["meta"]["spread_limit"] = limit
            kept.append(rec)
        med = sorted(turns)[len(turns) // 2]
        per_task[task] = {
            "available": len(items),
            "kept": len(chosen),
            "dropped_by_spread": len(dropped),
            "dropped_turns": [r["meta"]["n_assistant"] for r in dropped],
            "spread_rule": rule,
            "spread_limit": limit,
            "resample": len(chosen) < KEEP,
            "resample_target": RESAMPLE_TARGET if len(chosen) < KEEP else None,
            "need_more_to_15": max(0, RESAMPLE_TARGET - len(chosen)) if len(chosen) < KEEP else 0,
            "turns_kept_min": min(turns),
            "turns_kept_median": med,
            "turns_kept_max": max(turns),
            "turns_kept": turns,
            "turns_available_min": min(avail_turns),
            "turns_available_max": max(avail_turns),
            "sources_kept": dict(Counter(r["meta"]["source"] for r in chosen)),
            "models_kept": dict(Counter(r["meta"]["model"] for r in chosen)),
            "asst_without_think": sum(r["meta"]["n_assistant_without_think"] for r in chosen),
            "extra_user_msgs": sum(r["meta"]["n_extra_user"] for r in chosen),
        }

    kept.sort(key=lambda r: (r["meta"]["task_id"], r["meta"]["rank_by_turns"]))
    with OUT_JSONL.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    full = [t for t, s in per_task.items() if s["kept"] >= KEEP]
    short = [t for t, s in per_task.items() if s["kept"] < KEEP]
    summary = {
        "out": str(OUT_JSONL),
        "keep_per_task": KEEP,
        "resample_target": RESAMPLE_TARGET,
        "spread_rule": "among tasks with >=10: min>150 -> drop > min+55; min>100 -> drop > 1.5*min; else drop > 1.75*min",
        "order": "n_assistant ascending, then model, then trial_key; first 10, then spread filter",
        "loaded": dict(loaded),
        "unique_trials": len(seen),
        "duplicate_trials_skipped": dict(dup),
        "n_tasks": len(per_task),
        "n_kept": len(kept),
        "tasks_with_10": full,
        "tasks_under_10": {t: per_task[t]["kept"] for t in short},
        "per_task": per_task,
        "notes": [
            "Historical <think> is kept. This file is not turn-split.",
            "OpenAI trajectories are converted to swift-agent messages so they match qwen38_pass_226_swift_agent_traj_v2.",
            "GLM assistant turns with empty reasoning_content have no <think> block.",
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "unique_trials": len(seen),
        "n_tasks": len(per_task),
        "n_kept": len(kept),
        "n_with_10": len(full),
        "n_under_10": len(short),
        "under_10": {t: per_task[t]["kept"] for t in short},
        "dup": dict(dup),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
