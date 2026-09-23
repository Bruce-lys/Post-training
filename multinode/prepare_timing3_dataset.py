#!/usr/bin/env python3
"""Repeat the representative real-data batch from profile step 5 three times."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    rows = [json.loads(line) for line in source.open(encoding="utf-8")]
    representative = rows[32:40]
    if len(rows) != 48 or len(representative) != 8:
        raise RuntimeError(f"expected 48 source rows and 8 representative rows, got {len(rows)} and {len(representative)}")
    if not all(row.get("meta", {}).get("profile_step") == 5 for row in representative):
        raise RuntimeError("source rows 33-40 are not profile step 5")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with output.open("w", encoding="utf-8") as stream:
        for step in range(1, 4):
            for within_step, original in enumerate(representative):
                row = json.loads(json.dumps(original, ensure_ascii=False))
                row.setdefault("meta", {}).update(
                    {
                        "timing_step": step,
                        "timing_purpose": "repeat_of_real_profile_step5",
                    }
                )
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                manifest.append(
                    {
                        "step": step,
                        "within_step": within_step,
                        "source_line": row["meta"]["profile_source_line"],
                        "tokens": row["meta"]["profile_template_tokens"],
                        "bucket": next(
                            name
                            for name, upper in (("short", 65_536), ("medium", 163_840), ("long", 212_993))
                            if row["meta"]["profile_template_tokens"] < upper
                        ),
                    }
                )

    summary = {
        "source": str(source),
        "source_profile_step": 5,
        "rows": len(manifest),
        "steps": 3,
        "samples_per_step": 8,
        "tokens_per_step": sum(item["tokens"] for item in manifest[:8]),
        "source_lines_per_step": [item["source_line"] for item in manifest[:8]],
        "manifest": manifest,
    }
    Path(str(output) + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
