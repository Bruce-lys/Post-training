#!/usr/bin/env python3
"""Build a nine-step length-stratified DP scheduling A/B dataset."""

import argparse
import itertools
import json
import random
from pathlib import Path


def tokens(row):
    return int(row["meta"]["profile_template_tokens"])


def source_line(row):
    return int(row["meta"]["profile_source_line"])


def dp_totals(rows):
    return [sum(tokens(row) for row in rows[rank::2]) for rank in range(2)]


def balanced_order(rows):
    total = sum(map(tokens, rows))
    dp0 = set(
        min(
            itertools.combinations(range(8), 4),
            key=lambda indices: abs(total - 2 * sum(tokens(rows[index]) for index in indices)),
        )
    )
    groups = [
        [row for index, row in enumerate(rows) if (index in dp0) == (rank == 0)]
        for rank in range(2)
    ]
    return [row for pair in zip(*groups) for row in pair]


def describe(rows):
    totals = dp_totals(rows)
    return {
        "total_tokens": sum(totals),
        "dp_token_totals": totals,
        "dp_token_gap": abs(totals[0] - totals[1]),
        "order_source_lines": [source_line(row) for row in rows],
        "tokens": [tokens(row) for row in rows],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    source_rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    if len(source_rows) != 48:
        raise RuntimeError(f"expected 48 profile rows, got {len(source_rows)}")

    profile_groups = {
        "warmup": source_rows[0:8],
        "short": source_rows[8:16],
        "medium": source_rows[16:24],
        "mix1": source_rows[24:32],
        "mix2": source_rows[32:40],
    }
    expected_profile_steps = {"warmup": 1, "short": 2, "medium": 3, "mix1": 4, "mix2": 5}
    for name, rows in profile_groups.items():
        actual = {row["meta"]["profile_step"] for row in rows}
        if actual != {expected_profile_steps[name]}:
            raise RuntimeError(f"{name}: unexpected source profile steps {actual}")

    variants = {}
    for offset, name in enumerate(("short", "medium", "mix1", "mix2"), 1):
        original = list(profile_groups[name])
        random.Random(args.seed + offset).shuffle(original)
        variants[name] = {"A": original, "B": balanced_order(original)}

    sequence = [
        (1, "W", "warmup", "W", profile_groups["warmup"]),
        (2, "AS", "short", "A", variants["short"]["A"]),
        (3, "BS", "short", "B", variants["short"]["B"]),
        (4, "BM", "medium", "B", variants["medium"]["B"]),
        (5, "AM", "medium", "A", variants["medium"]["A"]),
        (6, "AX1", "mix1", "A", variants["mix1"]["A"]),
        (7, "BX1", "mix1", "B", variants["mix1"]["B"]),
        (8, "BX2", "mix2", "B", variants["mix2"]["B"]),
        (9, "AX2", "mix2", "A", variants["mix2"]["A"]),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with output.open("w", encoding="utf-8") as stream:
        for step, label, group, variant, rows in sequence:
            for position, original_row in enumerate(rows):
                row = json.loads(json.dumps(original_row, ensure_ascii=False))
                row.setdefault("meta", {}).update(
                    {
                        "stratified_ab_step": step,
                        "stratified_ab_label": label,
                        "stratified_ab_group": group,
                        "stratified_ab_variant": variant,
                        "stratified_ab_position": position,
                        "stratified_ab_expected_dp_rank": position % 2,
                    }
                )
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                manifest.append(
                    {
                        "step": step,
                        "label": label,
                        "group": group,
                        "variant": variant,
                        "position": position,
                        "expected_dp_rank": position % 2,
                        "source_line": source_line(row),
                        "tokens": tokens(row),
                    }
                )

    groups = {
        name: {variant: describe(rows) for variant, rows in pair.items()}
        for name, pair in variants.items()
    }
    summary = {
        "source": str(source),
        "seed": args.seed,
        "rows": len(manifest),
        "steps": len(sequence),
        "step_labels": [label for _, label, _, _, _ in sequence],
        "samples_per_step": 8,
        "sampler_assignment": "even positions -> DP0; odd positions -> DP1",
        "mixed_sample_counts": {"short": 6, "medium": 8, "long": 2},
        "mixed_sample_fractions": {"short": 0.375, "medium": 0.5, "long": 0.125},
        "groups": groups,
        "manifest": manifest,
    }
    Path(str(output) + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
