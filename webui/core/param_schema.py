"""Declarative description of the user-facing training parameters and how they map onto ``megatron sft`` argv.

Every ParamSpec turns into one Gradio component (see ui/tab_train.py::make_component) and, unless it is a
pure UI helper (``flag=None``), into ``--<flag> <value>`` on the command line. The defaults reproduce
swift_env/megatron_sft.sh (the reference LoRA recipe) unless noted."""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from . import schema

LORA_TARGET_DEFAULT = "in_proj out_proj linear_proj linear_qkv"
TUNER_CHOICES = ("lora", "full")
GROUPS = ("model", "data", "train", "lora", "full", "parallel", "moe", "save", "advanced")
GROUP_LABELS = {
    "model": "模型", "data": "数据", "train": "训练超参", "lora": "LoRA", "full": "全量微调",
    "parallel": "并行 / GPU", "moe": "MoE / 显存", "save": "保存 / 日志 / 评估", "advanced": "高级参数",
}


@dataclass
class ParamSpec:
    key: str
    label: str
    group: str
    widget: str                      # int | number | dropdown | checkbox | text | multiline | readonly
    default: Any = None
    flag: str | None = None          # None -> UI-only helper (not emitted)
    choices: tuple = ()
    info: str = ""
    minimum: float | None = None
    step: float | None = None
    slider: tuple | None = None      # (min, max, step)
    int_choices: tuple | None = None
    nargs: bool = False              # split the text on whitespace into several argv tokens
    optional: bool = False           # empty / 0 -> flag omitted
    only_for: str | None = None      # "lora" | "full": emitted (and shown) only for that tuner
    extra: dict = field(default_factory=dict)


def _i(key, label, group, default, flag=None, info="", minimum=0, slider=None, int_choices=None, optional=False, only_for=None):
    return ParamSpec(key, label, group, "int", default, flag, info=info, minimum=minimum, step=1, slider=slider,
                     int_choices=int_choices, optional=optional, only_for=only_for)


def _f(key, label, group, default, flag=None, info="", minimum=None, only_for=None):
    return ParamSpec(key, label, group, "number", default, flag, info=info, minimum=minimum, only_for=only_for)


def _b(key, label, group, default, flag=None, info="", only_for=None):
    return ParamSpec(key, label, group, "checkbox", default, flag, info=info, only_for=only_for)


def _d(key, label, group, default, choices, flag=None, info="", only_for=None):
    return ParamSpec(key, label, group, "dropdown", default, flag, choices=tuple(choices), info=info, only_for=only_for)


def _t(key, label, group, default, flag=None, info="", nargs=False, optional=False, only_for=None):
    return ParamSpec(key, label, group, "text", default, flag, info=info, nargs=nargs, optional=optional, only_for=only_for)


def _sc(name: str, fallback: tuple) -> tuple:
    c = schema.choices(name)
    return tuple(c) if c else fallback


PARAMS: list[ParamSpec] = [
    # ---- model
    _d("tuner_type", "训练方式", "model", "lora", TUNER_CHOICES, "tuner_type",
       info="lora = 只训 LoRA 适配器（推荐）；full = 全量微调（336GB MoE 显存需求极大）"),
    _d("torch_dtype", "权重精度", "model", "bfloat16", ("bfloat16", "float16", "float32"), "torch_dtype",
       info="加载权重的 dtype"),
    _t("template", "对话模板", "model", "", "template", info="留空自动推断（本模型为 qwen3_8）", optional=True),
    _t("system", "系统提示词", "model", "", "system", info="留空用数据/模板自带的 system", optional=True),

    # ---- data
    _t("val_dataset", "验证集", "data", "", "val_dataset", info="可选；jsonl 路径，留空则按右边比例从训练集切分", optional=True),
    _f("split_dataset_ratio", "验证集切分比例", "data", 0.0, "split_dataset_ratio",
       info="0 = 不切分；还需 评估 iters > 0 才会真正评估", minimum=0),
    _i("max_length", "最大序列长度", "data", 2048, "max_length", info="超过按截断策略处理", minimum=64, slider=(256, 65536, 256)),
    _d("truncation_strategy", "截断策略", "data", "delete", _sc("truncation_strategy", ("delete", "left", "right", "split")),
       "truncation_strategy", info="delete = 丢弃超长样本"),
    _b("padding_free", "padding_free", "data", False, "padding_free", info="去 padding 拼接；本模型脚本默认关"),
    _b("packing", "packing", "data", False, "packing", info="多样本打包到一条序列（会强制 padding_free）"),
    _i("dataset_num_proc", "数据预处理进程数", "data", 4, "dataset_num_proc", minimum=1),
    _i("dataloader_num_workers", "dataloader workers", "data", 4, "dataloader_num_workers", minimum=0),

    # ---- train
    _i("num_train_epochs", "训练轮数 (epochs)", "train", 3, "num_train_epochs", info="当 训练步数 为 0 时生效", minimum=0),
    _i("train_iters", "训练步数 (iters)", "train", 0, "train_iters", info="> 0 时按步数训练并忽略 epochs；smoke 用 2",
       minimum=0, optional=True),
    _f("lr", "学习率", "train", 1e-4, "lr", info="LoRA 推荐 1e-4；全量推荐 1e-5", minimum=0),
    _f("min_lr", "最小学习率", "train", 1e-5, "min_lr", minimum=0),
    _d("lr_decay_style", "学习率衰减", "train", "cosine",
       _sc("lr_decay_style", ("constant", "linear", "cosine", "inverse-square-root", "WSD")), "lr_decay_style"),
    _f("lr_warmup_fraction", "warmup 比例", "train", 0.05, "lr_warmup_fraction", minimum=0),
    _f("weight_decay", "weight decay", "train", 0.1, "weight_decay", minimum=0),
    _f("clip_grad", "梯度裁剪", "train", 1.0, "clip_grad", minimum=0),
    _i("global_batch_size", "全局 batch", "train", 8, "global_batch_size", info="必须能被 DP 整除", minimum=1),
    _i("micro_batch_size", "micro batch", "train", 1, "micro_batch_size", minimum=1),
    _i("seed", "随机种子", "train", 42, "seed", minimum=0),

    # ---- lora
    _i("lora_rank", "LoRA rank", "lora", 8, "lora_rank", minimum=1, slider=(1, 256, 1), only_for="lora"),
    _i("lora_alpha", "LoRA alpha", "lora", 32, "lora_alpha", minimum=1, slider=(1, 512, 1), only_for="lora"),
    _f("lora_dropout", "LoRA dropout", "lora", 0.05, "lora_dropout", minimum=0, only_for="lora"),
    _t("target_modules", "目标模块", "lora", LORA_TARGET_DEFAULT, "target_modules",
       info="空格分隔；all-linear = 全部线性层", nargs=True, only_for="lora"),
    _b("merge_lora", "保存时合并 LoRA", "lora", False, "merge_lora",
       info="true 会在每次保存时导出合并后的完整权重（本模型 336GB，很慢）", only_for="lora"),
    _b("use_rslora", "rsLoRA", "lora", False, "use_rslora", only_for="lora"),

    # ---- full
    _t("freeze_parameters_regex", "冻结参数正则", "full", "", "freeze_parameters_regex",
       info="可选；匹配到的参数不训练", optional=True, only_for="full"),
    _t("trainable_parameters_regex", "可训练参数正则", "full", "", "trainable_parameters_regex",
       info="可选；只训练匹配到的参数", optional=True, only_for="full"),

    # ---- parallel
    _t("cuda_visible_devices", "使用的 GPU", "parallel", "0,1,2,3,4,5,6,7", None,
       info="CUDA_VISIBLE_DEVICES；逗号分隔，卡数 = 进程数"),
    ParamSpec("nproc", "进程数 (NPROC_PER_NODE)", "parallel", "readonly", info="= GPU 数"),
    _i("tp", "TP", "parallel", 2, "tensor_model_parallel_size", int_choices=(1, 2, 4, 8)),
    _i("pp", "PP", "parallel", 2, "pipeline_model_parallel_size", int_choices=(1, 2, 4, 8)),
    _i("ep", "EP", "parallel", 4, "expert_model_parallel_size", int_choices=(1, 2, 4, 8)),
    _i("etp", "ETP", "parallel", 1, "expert_tensor_parallel_size", int_choices=(1, 2, 4, 8)),
    _i("cp", "CP", "parallel", 1, "context_parallel_size", int_choices=(1, 2, 4, 8)),
    ParamSpec("dp", "DP", "parallel", "readonly", info="= GPU 数 / (TP x PP x CP)"),
    _b("sequence_parallel", "sequence parallel", "parallel", True, "sequence_parallel", info="TP > 1 时建议开"),
    _i("decoder_first_pipeline_num_layers", "PP 首段层数", "parallel", 12, "decoder_first_pipeline_num_layers",
       info="PP > 1 时生效；0 = 不设置（均分）", minimum=0, optional=True),

    # ---- moe / memory
    _b("moe_grouped_gemm", "grouped GEMM", "moe", True, "moe_grouped_gemm"),
    _b("moe_permute_fusion", "permute fusion", "moe", True, "moe_permute_fusion"),
    _f("moe_aux_loss_coeff", "MoE aux loss 系数", "moe", 1e-3, "moe_aux_loss_coeff", minimum=0),
    _d("recompute_granularity", "重计算粒度", "moe", "full",
       _sc("recompute_granularity", ("selective", "full", "none")), "recompute_granularity", info="full 最省显存"),
    _d("recompute_method", "重计算方式", "moe", "uniform", ("uniform", "block", "none"), "recompute_method",
       info="仅 粒度=full 时有效；none = 不传"),
    _i("recompute_num_layers", "重计算层数", "moe", 1, "recompute_num_layers", minimum=1),
    _b("use_precision_aware_optimizer", "精度感知优化器", "moe", True, "use_precision_aware_optimizer", info="省优化器显存"),
    _d("attention_backend", "attention 后端", "moe", "auto", ("auto", "flash", "fused", "unfused", "local"), "attention_backend"),

    # ---- save / log / eval
    _i("save_steps", "保存间隔 (steps)", "save", 50, "save_steps", minimum=1),
    _i("save_total_limit", "最多保留 checkpoint 数", "save", 0, "save_total_limit",
       info="0 = 不限制；设置时必须 ≥ 2", minimum=0, optional=True),
    _i("eval_steps", "评估间隔 (steps)", "save", 1000, "eval_steps", minimum=1),
    _i("eval_iters", "评估 iters", "save", -1, "eval_iters",
       info="-1 = 不评估（ms-swift 默认）；要画 eval_loss 曲线需 > 0 且有验证集", minimum=-1),
    _i("logging_steps", "日志间隔 (steps)", "save", 1, "logging_steps", minimum=1),
    _b("no_save_optim", "不保存优化器状态", "save", True, "no_save_optim", info="只留权重，checkpoint 小很多"),
    _b("no_save_rng", "不保存 RNG 状态", "save", True, "no_save_rng"),
    _b("save_safetensors", "safetensors 格式", "save", True, "save_safetensors"),

    # ---- advanced
    ParamSpec("extra_args", "额外参数", "advanced", "multiline", "",
              info="每行一个 --flag value，原样追加到命令末尾；与上面重复的以这里为准"),
]
BY_KEY: dict[str, ParamSpec] = {p.key: p for p in PARAMS}
EDIT_KEYS = [p.key for p in PARAMS if p.widget != "readonly"]

# Flags always emitted (from swift_env/megatron_sft.sh); values are not user-facing.
FIXED_FLAGS: list[tuple[str, str]] = [
    ("finetune", "true"),
    ("cross_entropy_loss_fusion", "true"),
    ("vit_attn_impl", "sdpa"),
    ("load_from_cache_file", "true"),
    ("report_to", "tensorboard"),
    ("add_version", "false"),          # the UI owns the output directory -> no v0-<ts> suffix
]


def defaults() -> dict[str, Any]:
    return {p.key: p.default for p in PARAMS if p.widget != "readonly"}


def _cast(spec: ParamSpec, raw: Any) -> Any:
    if raw is None or raw == "":
        return None if spec.widget in ("int", "number") else raw
    if spec.widget == "int":
        return int(float(raw))
    if spec.widget == "number":
        return float(raw)
    if spec.widget == "checkbox":
        return bool(raw) if not isinstance(raw, str) else raw.lower() in ("1", "true", "yes", "on")
    return raw


def _fmt(spec: ParamSpec, value: Any) -> str:
    if spec.widget == "int":
        return str(int(value))
    if spec.widget == "number":
        v = float(value)
        return str(int(v)) if v.is_integer() and abs(v) >= 1 else repr(v)
    if spec.widget == "checkbox":
        return "true" if value else "false"
    return str(value)


def normalize(state: dict) -> dict:
    """Fill missing keys with defaults and cast values to their widget type."""
    s = dict(defaults())
    for k, v in (state or {}).items():
        if k in BY_KEY and BY_KEY[k].widget != "readonly":
            s[k] = _cast(BY_KEY[k], v)
    return s


def gpu_list(state: dict) -> list[str]:
    raw = str(state.get("cuda_visible_devices") or "").replace(" ", "")
    return [x for x in raw.split(",") if x != ""]


def compute_derived(state: dict) -> dict:
    s = normalize(state)
    n = len(gpu_list(s))
    s["nproc"] = n
    try:
        s["dp"] = n // max(1, int(s["tp"]) * int(s["pp"]) * int(s["cp"]))
    except (TypeError, ValueError):
        s["dp"] = None
    return s


def visible_for_tuner(spec: ParamSpec, tuner: str) -> bool:
    return spec.only_for is None or spec.only_for == tuner


def parse_extra_args(text: str) -> list[str]:
    """Free-form advanced arguments: one ``--flag value`` per line (shell-like splitting)."""
    out: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.extend(shlex.split(line))
    return out


def _is_zero(spec: ParamSpec, val: Any) -> bool:
    if val is None or val == "":
        return True
    if spec.widget in ("int", "number"):
        try:
            return float(val) == 0
        except (TypeError, ValueError):
            return True
    return False


def state_flags(state: dict) -> list[tuple[str, list[str]]]:
    """Ordered (flag, values) pairs derived from the state (fixed flags and extra args excluded)."""
    s = compute_derived(state)
    tuner = s.get("tuner_type") or "lora"
    out: list[tuple[str, list[str]]] = []
    for spec in PARAMS:
        if spec.flag is None or spec.widget == "readonly":
            continue
        if not visible_for_tuner(spec, tuner):
            continue
        val = s.get(spec.key)
        if spec.optional and _is_zero(spec, val):
            continue
        if spec.key == "recompute_method" and (val == "none" or s.get("recompute_granularity") != "full"):
            continue
        if spec.key == "recompute_num_layers" and s.get("recompute_granularity") != "full":
            continue
        if spec.key == "decoder_first_pipeline_num_layers" and int(s.get("pp") or 1) <= 1:
            continue
        if spec.key == "num_train_epochs" and int(s.get("train_iters") or 0) > 0:
            continue
        if val is None:
            continue
        if spec.nargs:
            vals = str(val).split()
            if not vals:
                continue
            out.append((spec.flag, vals))
        else:
            out.append((spec.flag, [_fmt(spec, val)]))
    return out


def state_from_flags(flags: dict[str, list[str]], base: dict | None = None) -> dict:
    """Inverse of state_flags for the flags that belong to a ParamSpec (used by tests and 'clone from args.json')."""
    s = dict(base or defaults())
    for spec in PARAMS:
        if spec.flag is None or spec.widget == "readonly" or spec.flag not in flags:
            continue
        vals = flags[spec.flag]
        if spec.nargs:
            s[spec.key] = " ".join(vals)
        elif vals:
            s[spec.key] = _cast(spec, vals[0])
    return s


def validate_state(state: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the launch."""
    s = compute_derived(state)
    errors: list[str] = []
    warns: list[str] = []
    n = s["nproc"]
    try:
        tp, pp, ep, etp, cp = (int(s[k]) for k in ("tp", "pp", "ep", "etp", "cp"))
        if n <= 0:
            errors.append("未指定 GPU（使用的 GPU 为空）")
        else:
            if n % (tp * pp * cp) != 0:
                errors.append(f"GPU 数 {n} 不能被 TP x PP x CP = {tp * pp * cp} 整除")
            dp = n // (tp * pp * cp) if n % (tp * pp * cp) == 0 else 0
            if dp and tp * dp != ep * etp:
                errors.append(f"MoE 约束不满足：TP x DP = {tp * dp} 必须等于 EP x ETP = {ep * etp}")
            if dp and int(s["global_batch_size"]) % dp != 0:
                errors.append(f"全局 batch {s['global_batch_size']} 不能被 DP = {dp} 整除")
        if pp <= 1 and int(s.get("decoder_first_pipeline_num_layers") or 0) > 0:
            warns.append("PP = 1 时 PP 首段层数 会被忽略（不传）")
        if s.get("sequence_parallel") and tp <= 1:
            warns.append("TP = 1 时 sequence parallel 会被 ms-swift 自动关闭")
    except (KeyError, TypeError, ValueError):
        errors.append("并行参数存在空值或非法数值")
    if s.get("recompute_granularity") != "full" and s.get("recompute_method") not in (None, "none"):
        warns.append("重计算粒度不是 full 时，重计算方式/层数不会传（selective + method 会报错）")
    if s.get("packing") and s.get("padding_free") is False:
        warns.append("packing 会强制 padding_free = true")
    if int(s.get("train_iters") or 0) <= 0 and int(s.get("num_train_epochs") or 0) <= 0:
        errors.append("训练轮数 与 训练步数 至少一个 > 0")
    if int(s.get("eval_iters") or -1) <= 0 and (float(s.get("split_dataset_ratio") or 0) > 0 or s.get("val_dataset")):
        warns.append("配置了验证集但 评估 iters ≤ 0：ms-swift 不会做评估，也不会有 eval_loss 曲线")
    if int(s.get("save_total_limit") or 0) == 1:
        errors.append("最多保留 checkpoint 数 必须 ≥ 2（或 0 = 不限制）")
    if s.get("tuner_type") == "full":
        warns.append("全量微调：Qwen3.8-Flash-Next 336GB MoE，8 x H200 显存远不够，除非只训极少参数")
        if float(s.get("lr") or 0) > 5e-5:
            warns.append(f"全量微调学习率 {s.get('lr')} 偏大，ms-swift 默认 1e-5")
    for t in parse_extra_args(s.get("extra_args") or ""):
        if t.startswith("--") and not schema.is_known_flag(t):
            warns.append(f"额外参数 {t} 不在 megatron sft 参数表里（可能拼错）")
    return errors, warns
