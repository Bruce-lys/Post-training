#!/usr/bin/env python3
"""Build a five-step repeated-batch dataset and explicit PP split configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


SPLITS = (22, 23, 24)
STEPS = 5
SAMPLES_PER_STEP = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.open(encoding="utf-8")]
    representative = rows[32:40]
    if len(rows) != 48 or len(representative) != SAMPLES_PER_STEP:
        raise RuntimeError(
            f"expected 48 source rows and 8 representative rows, got {len(rows)} and {len(representative)}"
        )
    if not all(row.get("meta", {}).get("profile_step") == 5 for row in representative):
        raise RuntimeError("source rows 33-40 are not profile step 5")

    data_dir = args.root / "smoke_data" / "pp_split_ab"
    data_path = data_dir / "real_step5_repeated_40rows.jsonl"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    with data_path.open("w", encoding="utf-8") as stream:
        for step in range(1, STEPS + 1):
            for within_step, original in enumerate(representative):
                row = copy.deepcopy(original)
                row.setdefault("meta", {}).update(
                    {
                        "pp_split_step": step,
                        "pp_split_purpose": "same_real_batch_warmup2_measure3",
                    }
                )
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                manifest.append(
                    {
                        "step": step,
                        "within_step": within_step,
                        "source_line": row["meta"]["profile_source_line"],
                        "tokens": row["meta"]["profile_template_tokens"],
                    }
                )

    tokens_per_step = sum(item["tokens"] for item in manifest[:SAMPLES_PER_STEP])
    summary = {
        "source": str(args.source),
        "source_profile_step": 5,
        "rows": len(manifest),
        "steps": STEPS,
        "warmup_steps": 2,
        "measured_steps": [3, 4, 5],
        "samples_per_step": SAMPLES_PER_STEP,
        "tokens_per_step": tokens_per_step,
        "source_lines_per_step": [item["source_line"] for item in manifest[:SAMPLES_PER_STEP]],
        "manifest": manifest,
    }
    summary_path = Path(str(data_path) + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    generated = []
    for split in SPLITS:
        config = copy.deepcopy(base)
        config["dataset"] = str(data_path)
        config["output_dir"] = f"/kwkj-k8s/llm_team/lys/megatron-swift/outputs/qsa4node_ppsplit{split}_smoke"
        config["tensorboard_dir"] = config["output_dir"] + "/tensorboard"
        config["decoder_first_pipeline_num_layers"] = split
        config["train_iters"] = STEPS
        config["external_plugins"] = [
            str(args.root / "plugins" / "ple_chunked.py"),
            str(args.root / "plugins" / "pp_split_timing.py"),
        ]
        config["ENV"].pop("NSYS_CAPTURE_STEP", None)
        config["ENV"]["PP_FIRST_LAYERS"] = split
        config["ENV"]["PERF_STEP_LABELS"] = "W1,W2,M1,M2,M3"
        config["ENV"]["PROFILE_DISABLE_SAVE"] = 1
        config_path = args.root / f"tb226_flash_next_lora_4node_qsa_ppsplit{split}_smoke.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        generated.append({"split": split, "config": str(config_path), "sha256": sha256(config_path)})

    print(
        json.dumps(
            {
                "dataset": str(data_path),
                "dataset_sha256": sha256(data_path),
                "summary": str(summary_path),
                "rows": len(manifest),
                "tokens_per_step": tokens_per_step,
                "configs": generated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
