"""Unit tests for the parameter table, command assembly and validation (no Gradio, no GPU)."""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import command as cmd, param_schema as ps, presets  # noqa: E402

REF_SCRIPT = ROOT.parent / "swift_env" / "megatron_sft.sh"
VENV = "/tmp/venv"


def _argv(state, tuner="lora"):
    s = {**ps.defaults(), **state, "tuner_type": tuner}
    return cmd.build_argv(s, "/models/m", "/data/d.jsonl", "/out/run", VENV)


def test_defaults_roundtrip():
    s = ps.compute_derived(ps.defaults())
    flags = cmd.parse_flags(_argv({}))
    back = ps.state_from_flags(flags, ps.defaults())
    for spec in ps.PARAMS:
        if spec.flag is None or spec.widget == "readonly" or spec.only_for == "full":
            continue
        if spec.optional and spec.default in (None, "", 0):
            continue
        if spec.key == "recompute_method":
            continue
        assert back[spec.key] == s[spec.key], spec.key


def test_lora_full_switch():
    lora = cmd.parse_flags(_argv({}, "lora"))
    full = cmd.parse_flags(_argv({}, "full"))
    assert lora["tuner_type"] == ["lora"] and full["tuner_type"] == ["full"]
    for f in ("lora_rank", "lora_alpha", "lora_dropout", "target_modules", "merge_lora", "use_rslora"):
        assert f in lora and f not in full
    assert lora["target_modules"] == ["in_proj", "out_proj", "linear_proj", "linear_qkv"]
    assert full.get("freeze_parameters_regex") is None       # optional, empty -> omitted


def test_optional_and_conditional_flags():
    f = cmd.parse_flags(_argv({"train_iters": 2, "pp": 1, "decoder_first_pipeline_num_layers": 12,
                               "recompute_granularity": "selective", "save_total_limit": 0}))
    assert f["train_iters"] == ["2"] and "num_train_epochs" not in f
    assert "decoder_first_pipeline_num_layers" not in f            # PP == 1
    assert "recompute_method" not in f and "recompute_num_layers" not in f
    assert "save_total_limit" not in f
    g = cmd.parse_flags(_argv({"pp": 2, "decoder_first_pipeline_num_layers": 12, "save_total_limit": 3}))
    assert g["decoder_first_pipeline_num_layers"] == ["12"] and g["save_total_limit"] == ["3"]
    assert g["num_train_epochs"] == ["3"] and "train_iters" not in g


def test_fixed_flags_and_output_dir():
    f = cmd.parse_flags(_argv({}))
    assert f["add_version"] == ["false"] and f["finetune"] == ["true"] and f["output_dir"] == ["/out/run"]
    assert f["model"] == ["/models/m"] and f["dataset"] == ["/data/d.jsonl"]
    assert _argv({})[0] == f"{VENV}/bin/megatron" and _argv({})[1] == "sft"


def test_extra_args_override():
    f = cmd.parse_flags(_argv({"extra_args": "--lr 3e-5\n--moe_router_dtype fp32\n# comment"}))
    assert f["lr"] == ["3e-5"] and f["moe_router_dtype"] == ["fp32"]
    assert sum(1 for t in _argv({"extra_args": "--lr 3e-5"}) if t == "--lr") == 1


def test_validation_rules():
    ok_err, _ = ps.validate_state(ps.defaults())
    assert ok_err == []
    err, _ = ps.validate_state({**ps.defaults(), "cuda_visible_devices": "0,1,2,3", "tp": 1, "pp": 1, "ep": 8})
    assert any("MoE" in e for e in err)                     # TP x DP = 4 != EP x ETP = 8
    err, _ = ps.validate_state({**ps.defaults(), "cuda_visible_devices": "0,1,2,3,4,5", "tp": 4, "pp": 1, "ep": 4})
    assert any("整除" in e for e in err)                    # 6 GPUs not divisible by TP x PP x CP = 4
    err, warn = ps.validate_state({**ps.defaults(), "pp": 1, "cuda_visible_devices": "0,1,2,3", "tp": 4, "ep": 4})
    assert err == [] and any("PP = 1" in w for w in warn)
    err, _ = ps.validate_state({**ps.defaults(), "save_total_limit": 1})
    assert any("≥ 2" in e for e in err)
    err, _ = ps.validate_state({**ps.defaults(), "train_iters": 0, "num_train_epochs": 0})
    assert any("至少一个" in e for e in err)
    _, warn = ps.validate_state({**ps.defaults(), "split_dataset_ratio": 0.01, "eval_iters": -1})
    assert any("eval" in w for w in warn)
    _, warn = ps.validate_state({**ps.defaults(), "tuner_type": "full"})
    assert any("全量" in w for w in warn)


def test_builtin_presets_valid():
    for name in presets.BUILTIN:
        s = presets.builtin_state(name)
        err, _ = ps.validate_state(s)
        assert err == [], (name, err)


@pytest.mark.skipif(not REF_SCRIPT.is_file(), reason="reference script not mounted")
def test_matches_reference_script():
    """sft_lora_8gpu (= defaults) must reproduce every flag of swift_env/megatron_sft.sh with the same value,
    except the ones the UI owns (dataset / output_dir / self-cognition placeholders / eval split)."""
    text = REF_SCRIPT.read_text(encoding="utf-8")
    body = text[text.index("megatron sft"):]
    ref: dict[str, list[str]] = {}
    for m in re.finditer(r"--(\w+)((?:\s+(?!--)[^\s\\]+)*)", body):
        vals = [v.strip("'\"") for v in m.group(2).split()]
        ref[m.group(1)] = vals
    env = {"TP": "2", "EP": "4", "PP": "2", "MBS": "1", "GBS": "8", "MAX_LENGTH": "2048", "MODEL": "/models/m",
           "OUTPUT_DIR": "/out/run"}
    ref = {k: [env.get(v[2:-1], v) if v.startswith("${") else v for v in vals] for k, vals in ref.items()}
    ours = cmd.parse_flags(_argv({}))
    # merge_lora: the script merges on every save; the UI defaults to false (336GB export per checkpoint) like hy's run
    skip = {"dataset", "output_dir", "model_name", "model_author", "split_dataset_ratio", "model", "merge_lora"}
    for flag, vals in ref.items():
        if flag in skip:
            continue
        assert flag in ours, f"missing --{flag}"
        a, b = [x.lower() for x in ours[flag]], [x.lower() for x in vals]
        if flag in ("lr", "min_lr", "moe_aux_loss_coeff", "lr_warmup_fraction"):
            assert float(a[0]) == float(b[0]), flag
        else:
            assert a == b, (flag, a, b)
