#!/usr/bin/env python3
"""Build a W-A-B-B-A dataset to isolate DP token-load balancing."""

import argparse
import itertools
import json
from pathlib import Path


def token_count(row):
    return int(row["meta"]["profile_template_tokens"])


def source_line(row):
    return int(row["meta"]["profile_source_line"])


def dp_totals(rows):
    return [sum(token_count(row) for row in rows[rank::2]) for rank in range(2)]


def balanced_order(rows):
    total = sum(map(token_count, rows))
    best = min(
        itertools.combinations(range(8), 4),
        key=lambda indices: abs(total - 2 * sum(token_count(rows[index]) for index in indices)),
    )
    dp0_indices = set(best)
    groups = [
        [row for index, row in enumerate(rows) if (index in dp0_indices) == (rank == 0)]
        for rank in range(2)
    ]
    return [row for pair in zip(*groups) for row in pair]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    source_rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    if len(source_rows) != 24:
        raise RuntimeError(f"expected 24 timing3 rows, got {len(source_rows)}")

    original = source_rows[:8]
    expected = [27671, 1788, 18761, 6146, 13761, 4001, 9734, 22730]
    if [source_line(row) for row in original] != expected:
        raise RuntimeError("timing3 representative sample identity/order changed")
    balanced = balanced_order(original)

    steps = [
        (1, "W", "common_warmup", balanced),
        (2, "A", "original_unbalanced", original),
        (3, "B", "dp_token_balanced", balanced),
        (4, "B", "dp_token_balanced", balanced),
        (5, "A", "original_unbalanced", original),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with output.open("w", encoding="utf-8") as stream:
        for step, variant, purpose, rows in steps:
            for position, original_row in enumerate(rows):
                row = json.loads(json.dumps(original_row, ensure_ascii=False))
                row.setdefault("meta", {}).update(
                    {
                        "dp_ab_step": step,
                        "dp_ab_variant": variant,
                        "dp_ab_purpose": purpose,
                        "dp_ab_position": position,
                        "dp_ab_expected_dp_rank": position % 2,
                    }
                )
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                manifest.append(
                    {
                        "step": step,
                        "variant": variant,
                        "position": position,
                        "expected_dp_rank": position % 2,
                        "source_line": source_line(row),
                        "tokens": token_count(row),
                    }
                )

    def describe(label, rows):
        totals = dp_totals(rows)
        return {
            "variant": label,
            "tokens_per_step": sum(totals),
            "dp_token_totals": totals,
            "dp_token_gap": abs(totals[0] - totals[1]),
            "order_source_lines": [source_line(row) for row in rows],
            "dp0_source_lines": [source_line(row) for row in rows[0::2]],
            "dp1_source_lines": [source_line(row) for row in rows[1::2]],
        }

    summary = {
        "source": str(source),
        "rows": len(manifest),
        "steps": 5,
        "step_sequence": [variant for _, variant, _, _ in steps],
        "samples_per_step": 8,
        "sampler_assignment": "even positions -> DP0; odd positions -> DP1",
        "A": describe("A_original_unbalanced", original),
        "B": describe("B_dp_token_balanced", balanced),
        "manifest": manifest,
    }
    Path(str(output) + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
