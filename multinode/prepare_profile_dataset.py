#!/usr/bin/env python3
"""Build a deterministic six-step profiling dataset from the formal JSONL."""

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from swift import get_processor, get_template


MODEL = "/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"
SOURCE = (
    "/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/"
    "qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl"
)
MAX_LENGTH = 212_992


def reservoir_sample(path: str, size: int, seed: int):
    rng = random.Random(seed)
    sample = []
    total = 0
    with open(path, "rb") as f:
        for line_number, line in enumerate(f, 1):
            total += 1
            record = (line_number, len(line.rstrip(b"\n")), line)
            if len(sample) < size:
                sample.append(record)
            else:
                slot = rng.randrange(total)
                if slot < size:
                    sample[slot] = record
    return total, sample


def bucket(length: int) -> str:
    if length < 65_536:
        return "short"
    if length < 163_840:
        return "medium"
    if length <= MAX_LENGTH:
        return "long"
    return "deleted"


def largest_remainder_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {key: value * total for key, value in weights.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    for key, _ in sorted(raw.items(), key=lambda item: item[1] - counts[item[0]], reverse=True)[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def choose_near_median(pool, count: int):
    ordered = sorted(pool, key=lambda row: row["tokens"])
    center = len(ordered) // 2
    start = max(0, min(len(ordered) - count, center - count // 2))
    return ordered[start : start + count]


def choose_mixed(pools, counts, rng):
    rows = []
    for name, count in counts.items():
        rows.extend(rng.sample(pools[name], count))
    rng.shuffle(rows)
    return rows


def balanced_two_way(rows):
    groups = [[], []]
    totals = [0, 0]
    for row in sorted(rows, key=lambda item: item["tokens"], reverse=True):
        eligible = [i for i in range(2) if len(groups[i]) < 4]
        target = min(eligible, key=lambda i: totals[i])
        groups[target].append(row)
        totals[target] += row["tokens"]
    # With shuffle disabled, the Megatron sampler assigns even rows to DP0 and
    # odd rows to DP1. Interleave the groups so each replica gets its intended
    # four-sample partition.
    interleaved = [row for pair in zip(groups[0], groups[1]) for row in pair]
    return interleaved, totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    started = time.time()
    total, raw_sample = reservoir_sample(SOURCE, args.sample_size, args.seed)
    print(f"[prepare] scanned={total} sampled={len(raw_sample)} seconds={time.time() - started:.1f}", flush=True)

    processor = get_processor(MODEL)
    template = get_template(
        processor,
        template_type="qwen3_8",
        agent_template="qwen3_5",
        loss_scale="last_round",
        truncation_strategy="raise",
        max_length=10_000_000,
    )
    template.set_mode("train")

    encoded = []
    for index, (line_number, byte_length, line) in enumerate(raw_sample, 1):
        sample = json.loads(line)
        token_length = len(template.encode(sample)["input_ids"])
        encoded.append(
            {
                "line": line_number,
                "bytes": byte_length,
                "tokens": token_length,
                "bucket": bucket(token_length),
                "sample": sample,
            }
        )
        if index % 16 == 0 or index == len(raw_sample):
            print(f"[prepare] encoded={index}/{len(raw_sample)}", flush=True)

    pools = {
        name: [row for row in encoded if row["bucket"] == name]
        for name in ("short", "medium", "long")
    }
    for name, pool in pools.items():
        if len(pool) < 16:
            raise RuntimeError(f"not enough {name} samples in the reservoir: {len(pool)}")

    retained = [row for row in encoded if row["bucket"] != "deleted"]
    sample_counts = Counter(row["bucket"] for row in encoded)
    token_sums = Counter()
    for row in retained:
        token_sums[row["bucket"]] += row["tokens"]
    weights = {name: len(pools[name]) / len(retained) for name in pools}
    mixed_counts = largest_remainder_counts(weights, 8)

    warm = choose_near_median(pools["medium"], 8)
    short = choose_near_median(pools["short"], 8)
    medium_candidates = [row for row in pools["medium"] if row not in warm]
    medium = choose_near_median(medium_candidates, 8)

    balanced_raw = choose_mixed(pools, mixed_counts, rng)
    balanced, balanced_totals = balanced_two_way(balanced_raw)
    random_mix = choose_mixed(pools, mixed_counts, rng)

    steps = [
        (1, "warmup_medium", warm, None),
        (2, "short", short, None),
        (3, "medium", medium, None),
        (4, "balanced_real_mix", balanced, None),
        (5, "random_real_mix", random_mix, None),
        (6, "profile_repeat_of_step5", random_mix, 5),
    ]

    manifest = []
    with output.open("w", encoding="utf-8") as out:
        output_row = 0
        for step, purpose, rows, repeat_of in steps:
            for within_step, row in enumerate(rows):
                output_row += 1
                sample = json.loads(json.dumps(row["sample"], ensure_ascii=False))
                sample.setdefault("meta", {}).update(
                    {
                        "profile_step": step,
                        "profile_purpose": purpose,
                        "profile_source_line": row["line"],
                        "profile_template_tokens": row["tokens"],
                        "profile_repeat_of_step": repeat_of,
                    }
                )
                out.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
                manifest.append(
                    {
                        "output_row": output_row,
                        "step": step,
                        "within_step": within_step,
                        "purpose": purpose,
                        "repeat_of_step": repeat_of,
                        "source_line": row["line"],
                        "tokens": row["tokens"],
                        "bucket": row["bucket"],
                    }
                )

    retained_tokens = sum(row["tokens"] for row in retained)
    summary = {
        "source": SOURCE,
        "model": MODEL,
        "seed": args.seed,
        "source_samples": total,
        "reservoir_size": len(encoded),
        "max_length": MAX_LENGTH,
        "sample_counts": dict(sample_counts),
        "sample_fractions_all": {name: sample_counts[name] / len(encoded) for name in sample_counts},
        "sample_fractions_retained": weights,
        "token_fractions_retained": {name: token_sums[name] / retained_tokens for name in pools},
        "mean_tokens_retained": sum(row["tokens"] for row in retained) / len(retained),
        "estimated_retained_samples": round(total * len(retained) / len(encoded)),
        "estimated_steps_per_epoch_gbs8": math.ceil(total * len(retained) / len(encoded) / 8),
        "mixed_counts_per_step": mixed_counts,
        "balanced_dp_token_totals": balanced_totals,
        "steps": [
            {
                "step": step,
                "purpose": purpose,
                "tokens": [row["tokens"] for row in rows],
                "total_tokens": sum(row["tokens"] for row in rows),
            }
            for step, purpose, rows, _ in steps
        ],
    }

    summary_path = output.with_suffix(output.suffix + ".summary.json")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[prepare] output={output} rows={len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
