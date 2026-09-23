"""Train tab (LlamaFactory layout): dataset row, parameter groups, action buttons, output / curves / log."""
from __future__ import annotations

import json
import pathlib

import gradio as gr
import pandas as pd

from core import command as cmd, launcher, monitor, param_schema as ps, plots, presets, registry
from core.constants import STATUS_ICON, STATUS_LABEL, hostname
from core.settings import load_paths_env, load_ui_settings

EDIT_KEYS = ps.EDIT_KEYS
RO_KEYS = [p.key for p in ps.PARAMS if p.widget == "readonly"]
OPEN_GROUPS = ("model", "data", "train", "lora", "full", "parallel")
LEVEL = {"ok": "通过", "warn": "提示", "error": "错误"}
TRAIN_PLOTS = ("Loss", "学习率", "梯度范数", "显存 (GiB)")


# --------------------------------------------------------------------------- helpers
def state_from_values(values) -> dict:
    return ps.normalize(dict(zip(EDIT_KEYS, values)))


def values_from_state(state: dict) -> list:
    s = ps.compute_derived(state)
    return [s.get(k) for k in EDIT_KEYS]


def readonly_values(state: dict) -> list:
    s = ps.compute_derived(state)
    return [str(s.get(k)) if s.get(k) is not None else "" for k in RO_KEYS]


def make_component(spec: ps.ParamSpec, value):
    info = spec.info or (f"--{spec.flag}" if spec.flag else "")
    if spec.widget == "dropdown":
        return gr.Dropdown(choices=list(spec.choices), value=value, label=spec.label, info=info)
    if spec.widget == "checkbox":
        return gr.Checkbox(value=bool(value), label=spec.label, info=info)
    if spec.widget == "int" and spec.int_choices:
        return gr.Dropdown(choices=[int(c) for c in spec.int_choices], value=int(value) if value is not None else None,
                           label=spec.label, info=info, allow_custom_value=True)
    if spec.widget == "int" and spec.slider:
        lo, hi, st = spec.slider
        v = lo if value is None else max(lo, min(hi, value))
        return gr.Slider(minimum=lo, maximum=hi, step=st, value=v, label=spec.label, info=info)
    if spec.widget == "int":
        return gr.Number(value=value, label=spec.label, info=info, precision=0, minimum=spec.minimum, step=1)
    if spec.widget == "number":
        return gr.Number(value=value, label=spec.label, info=info, minimum=spec.minimum)
    if spec.widget == "multiline":
        return gr.Textbox(value=value or "", label=spec.label, info=info, lines=4, placeholder="--moe_router_dtype fp32\n--packing_length 4096")
    if spec.widget == "text":
        return gr.Textbox(value="" if value is None else str(value), label=spec.label, info=info)
    return gr.Textbox(value="" if value is None else str(value), label=spec.label, info=info, interactive=False)


def list_datasets(data_dir: str) -> list[str]:
    d = pathlib.Path(data_dir or "")
    if not d.is_dir():
        return []
    files = sorted(p for p in d.rglob("*.jsonl") if p.is_file() and len(p.relative_to(d).parts) <= 3)
    return [str(p) for p in files]


def preview_dataset(path: str, n: int = 5):
    p = pathlib.Path(path or "")
    if not p.is_file():
        return f"`{path}` 不是本地文件（ModelScope/HF 数据集 ID 无法预览）", pd.DataFrame()
    rows, total = [], 0
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            if len(rows) < n:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"_error": "非法 JSON 行"})
    table = []
    for r in rows:
        if not isinstance(r, dict):
            table.append(["?", str(r)[:200], "", ""])
            continue
        if "messages" in r:
            msgs = r["messages"] if isinstance(r["messages"], list) else []
            user = next((m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"), "")
            asst = next((m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"), "")
            table.append(["messages", str(user)[:200], str(asst)[:200], f"{len(msgs)} 轮"])
        elif "instruction" in r or "query" in r:
            q = r.get("instruction", r.get("query", ""))
            if r.get("input"):
                q = f"{q}\n{r['input']}"
            table.append(["alpaca", str(q)[:200], str(r.get("output", r.get("response", "")))[:200], ""])
        else:
            table.append(["?", str(r)[:200], "", ", ".join(sorted(r.keys()))[:120]])
    df = pd.DataFrame(table, columns=["格式", "输入 (前 200 字)", "输出 (前 200 字)", "备注"])
    return f"`{p}`  —  共 {total} 行，显示前 {len(rows)} 行", df


def status_text(run_id: str) -> str:
    doc = registry.get(run_id) if run_id else None
    if not doc:
        return "尚未启动训练"
    launcher.finalize_if_exited(doc)
    doc = registry.get(run_id) or doc
    info = monitor.run_status(doc)
    st = info.get("status", "unknown")
    if doc.get("status") != st and st != "unknown" and doc.get("status") != "aborted":
        registry.update(run_id, status=st)
    prog = f"{info['step']}/{info['total']}" if info.get("total") else str(info["step"])
    loss = f"  ·  loss {info['loss']:.4f}" if info.get("loss") is not None else ""
    rem = f"  ·  剩余 {info['remaining']}" if info.get("remaining") and st == "running" else ""
    return (f"{STATUS_ICON.get(st, '')} **{STATUS_LABEL.get(st, st)}**  `{run_id}`  ·  步数 **{prog}**{loss}{rem}"
            f"  ·  pid {info.get('pid') or '-'}  ·  输出 `{doc.get('run_root')}`")


def log_text(run_id: str, max_bytes: int = 48_000) -> str:
    doc = registry.get(run_id) if run_id else None
    if not doc:
        return ""
    return monitor.read_tail(monitor.train_log_path(doc["run_root"]), max_bytes)[-max_bytes:]


def format_checks(p: launcher.LaunchPlan, checks: list[launcher.Check]) -> str:
    lines = [f"run_id: {p.run_id}", f"输出目录: {p.run_root}", f"模型: {p.model_path}", f"数据集: {p.dataset}",
             f"GPU: {p.env.get('CUDA_VISIBLE_DEVICES')}  (NPROC_PER_NODE={p.env.get('NPROC_PER_NODE')}, MASTER_PORT={p.env.get('MASTER_PORT')})",
             "", "启动前检查:"]
    lines += [f"  [{LEVEL[c.level]}] {c.name}: {c.message}" for c in checks]
    n_err = sum(1 for c in checks if c.level == "error")
    lines.append("")
    lines.append("❌ 有错误，无法启动" if n_err else "✅ 可以启动")
    return "\n".join(lines)


# --------------------------------------------------------------------------- build
def build(top: dict) -> dict:
    presets.ensure_builtin()
    paths0 = load_paths_env()
    base = ps.compute_derived(ps.defaults())
    comps: dict[str, gr.components.Component] = {}
    groups: dict[str, gr.Group] = {}
    current_run = gr.State("")

    # ---- dataset row
    with gr.Row():
        data_dir = gr.Textbox(value=paths0.get("DATA_ROOT", ""), label="数据目录", info="扫描其下的 *.jsonl", scale=2)
        ds_choices = list_datasets(paths0.get("DATA_ROOT", ""))
        dataset = gr.Dropdown(choices=ds_choices, value=ds_choices[0] if ds_choices else None, allow_custom_value=True,
                              label="数据集", info="jsonl 路径（alpaca 或 messages 格式），也可填 ModelScope 数据集 ID", scale=4)
        preview_ds_btn = gr.Button("预览数据集", scale=1)
    ds_info = gr.Markdown("")
    ds_table = gr.Dataframe(interactive=False, wrap=True, visible=False)

    def on_data_dir(d, current):
        files = list_datasets(d)
        keep = current if current in files else (files[0] if files else current)
        return gr.update(choices=files, value=keep)
    data_dir.change(on_data_dir, [data_dir, dataset], dataset)

    def on_preview(path):
        msg, df = preview_dataset(path)
        return msg, gr.update(value=df, visible=not df.empty)
    preview_ds_btn.click(on_preview, dataset, [ds_info, ds_table])

    # ---- parameters
    for group in ps.GROUPS:
        specs = [p for p in ps.PARAMS if p.group == group]
        visible = group not in ("lora", "full") or group == base["tuner_type"]
        with gr.Group(visible=visible) as grp:
            with gr.Accordion(ps.GROUP_LABELS[group], open=group in OPEN_GROUPS):
                if group == "advanced":
                    for spec in specs:
                        comps[spec.key] = make_component(spec, base.get(spec.key))
                else:
                    for i in range(0, len(specs), 5):
                        with gr.Row():
                            for spec in specs[i:i + 5]:
                                comps[spec.key] = make_component(spec, base.get(spec.key))
        groups[group] = grp

    edit_comps = [comps[k] for k in EDIT_KEYS]
    ro_comps = [comps[k] for k in RO_KEYS]
    top_inputs = [top["name"], top["model_path"], top["output_root"], dataset]

    # ---- buttons
    with gr.Row():
        preview_btn = gr.Button("预览命令")
        check_btn = gr.Button("启动前检查")
        preset_name = gr.Textbox(label="配置名", placeholder="保存当前参数为配置", scale=1)
        save_btn = gr.Button("保存配置")
        preset_dd = gr.Dropdown(choices=presets.list_presets(), label="已保存配置", scale=1)
        load_btn = gr.Button("加载配置")
        reset_btn = gr.Button("恢复默认")
    with gr.Row():
        start_btn = gr.Button("开始训练", variant="primary", scale=3)
        abort_btn = gr.Button("中止训练", variant="stop", scale=1)

    output = gr.Textbox(label="输出", lines=8, max_lines=16, show_copy_button=True)
    with gr.Accordion("命令预览", open=False):
        preview_code = gr.Code(language="shell", lines=24)
    status_md = gr.Markdown("尚未启动训练")
    plot_comps = []
    with gr.Row():
        for name in TRAIN_PLOTS[:2]:
            plot_comps.append(gr.Plot(label=name))
    with gr.Row():
        for name in TRAIN_PLOTS[2:]:
            plot_comps.append(gr.Plot(label=name))
    log_box = gr.Textbox(label="训练日志", lines=18, max_lines=18, autoscroll=True, show_copy_button=True)

    # ---- handlers
    def refresh(*values):
        return readonly_values(state_from_values(values))
    for c in edit_comps:
        c.input(refresh, edit_comps, ro_comps, show_progress="hidden")

    def on_tuner(tuner, lr):
        rec = {"lora": 1e-4, "full": 1e-5}
        new_lr = rec.get(tuner, lr) if lr in rec.values() or lr is None else lr
        return gr.update(visible=tuner == "lora"), gr.update(visible=tuner == "full"), new_lr
    comps["tuner_type"].change(on_tuner, [comps["tuner_type"], comps["lr"]], [groups["lora"], groups["full"], comps["lr"]])

    def set_state_values(state):
        s = ps.compute_derived(state)
        return values_from_state(s) + readonly_values(s) + [gr.update(visible=s["tuner_type"] == "lora"),
                                                           gr.update(visible=s["tuner_type"] == "full")]
    apply_state_outputs = edit_comps + ro_comps + [groups["lora"], groups["full"]]

    reset_btn.click(lambda: set_state_values(ps.defaults()), None, apply_state_outputs)

    def make_plan(nm, mp, orp, ds, *values):
        paths = dict(load_paths_env())
        if orp and orp.strip():
            paths["OUTPUT_ROOT"] = orp.strip()
        return launcher.plan(state_from_values(values), nm, mp or "", ds or "", paths, load_ui_settings())

    def do_preview(nm, mp, orp, ds, *values):
        try:
            p = make_plan(nm, mp, orp, ds, *values)
        except Exception as exc:  # noqa: BLE001
            return f"预览失败: {exc}", ""
        lines = [f"run_id: {p.run_id}", f"输出目录: {p.run_root}"]
        lines += [f"[错误] {e}" for e in p.errors] + [f"[提示] {w}" for w in p.warnings]
        return "\n".join(lines), launcher.preview_text(p)
    preview_btn.click(do_preview, [*top_inputs, *edit_comps], [output, preview_code])

    def do_check(nm, mp, orp, ds, *values):
        try:
            p = make_plan(nm, mp, orp, ds, *values)
            return format_checks(p, launcher.preflight_host(p)), launcher.preview_text(p)
        except Exception as exc:  # noqa: BLE001
            return f"检查失败: {exc}", ""
    check_btn.click(do_check, [*top_inputs, *edit_comps], [output, preview_code])

    def do_save(pname, *values):
        if not (pname or "").strip():
            return gr.update(), "请先填写配置名"
        stem = presets.save_preset(pname, ps.compute_derived(state_from_values(values)))
        return gr.update(choices=presets.list_presets(), value=stem), f"已保存配置 {stem}"
    save_btn.click(do_save, [preset_name, *edit_comps], [preset_dd, output])

    def do_load(pname):
        st = presets.load_preset(pname) if pname else None
        if not st:
            return [gr.update()] * len(apply_state_outputs) + ["配置不存在"]
        note = presets.preset_note(pname)
        return set_state_values(st) + [f"已加载配置 {pname}" + (f"：{note}" if note else "")]
    load_btn.click(do_load, preset_dd, apply_state_outputs + [output])

    def do_launch(nm, mp, orp, ds, *values):
        try:
            p = make_plan(nm, mp, orp, ds, *values)
            ok, msg = launcher.launch(p)
        except Exception as exc:  # noqa: BLE001
            return f"启动异常: {exc}", gr.update(), gr.update()
        return ("✅ " if ok else "❌ ") + msg, (p.run_id if ok else gr.update()), (launcher.preview_text(p) if ok else gr.update())

    def do_abort(run_id):
        if not run_id:
            return "当前没有由本页启动的训练"
        return launcher.abort(run_id)
    abort_btn.click(do_abort, current_run, output)

    def poll(run_id):
        doc = registry.get(run_id) if run_id else None
        df = monitor.load_metrics(doc["run_root"]) if doc else pd.DataFrame()
        return [status_text(run_id), *[plots.panel(df, n) for n in TRAIN_PLOTS], log_text(run_id)]
    poll_outputs = [status_md, *plot_comps, log_box]
    timer = gr.Timer(4, active=True)
    timer.tick(poll, current_run, poll_outputs, show_progress="hidden")

    mine = [d for d in registry.list_runs() if d.get("hostname") == hostname()]
    current_run.value = mine[0]["run_id"] if mine else ""

    return {"edit_comps": edit_comps, "ro_comps": ro_comps, "top_inputs": top_inputs, "start_btn": start_btn,
            "output": output, "preview_code": preview_code, "current_run": current_run, "do_launch": do_launch,
            "poll": poll, "poll_outputs": poll_outputs, "set_state_values": set_state_values,
            "apply_state_outputs": apply_state_outputs, "dataset": dataset,
            "do_preview": do_preview, "do_check": do_check, "do_load": do_load, "do_save": do_save,
            "on_tuner": on_tuner, "edit_keys": EDIT_KEYS}
