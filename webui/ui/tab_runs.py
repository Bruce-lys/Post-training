"""Runs tab: history table, details (command / config / log / curves), clone-to-train, abort, delete."""
from __future__ import annotations

import json
import pathlib

import gradio as gr
import pandas as pd

from core import launcher, monitor, plots, registry
from core.constants import STATUS_ICON, STATUS_LABEL, hostname

COLUMNS = ["run_id", "实验名", "主机", "状态", "方式", "GPU", "开始时间", "进度", "最新 loss"]
RUN_PLOTS = ("Loss", "学习率", "梯度范数", "显存 (GiB)", "MoE 均衡损失", "速度 (s/it)")


def runs_frame(local_only: bool = False, status: str = "全部") -> pd.DataFrame:
    docs = launcher.refresh_all()
    rows = []
    for d in docs:
        if local_only and d.get("hostname") != hostname():
            continue
        if status != "全部" and d.get("status") != status:
            continue
        root = d.get("run_root", "")
        prog = monitor.progress(root) if root else {"step": 0, "total": None, "loss": None}
        st = d.get("status", "")
        rows.append([d["run_id"], d.get("name", ""), d.get("hostname", ""), f"{STATUS_ICON.get(st, '')} {STATUS_LABEL.get(st, st)}",
                     d.get("tuner_type", ""), d.get("gpus", ""), d.get("created_at", ""),
                     f"{prog['step']}/{prog['total']}" if prog.get("total") else str(prog["step"]),
                     f"{prog['loss']:.4f}" if prog.get("loss") is not None else ""])
    return pd.DataFrame(rows, columns=COLUMNS)


def run_choices() -> list[str]:
    return [d["run_id"] for d in registry.list_runs()]


def _details(run_id: str):
    doc = registry.get(run_id) if run_id else None
    if not doc:
        return "未选择 run", "", "", False
    root = pathlib.Path(doc.get("run_root", ""))
    script = root / "ui" / "run.sh"
    cfg = root / "ui" / "train_config.json"
    info = monitor.run_status(doc)
    st = info.get("status", doc.get("status"))
    md = (f"### {STATUS_ICON.get(st, '')} {run_id}  —  **{STATUS_LABEL.get(st, st)}**\n\n"
          f"主机 `{doc.get('hostname')}`{'' if info['local'] else '（异机，只读）'} · 方式 **{doc.get('tuner_type')}** · "
          f"GPU `{doc.get('gpus')}` · {'假训练' if doc.get('fake_train') else 'GPU 训练'} · pid {info.get('pid') or '-'} · "
          f"退出码 {info.get('exit_code') if info.get('exit_code') is not None else '-'}\n\n"
          f"模型 `{doc.get('model')}`\n\n数据 `{doc.get('dataset')}`\n\n输出 `{doc.get('run_root')}`\n\n"
          f"开始 {doc.get('created_at')} · 结束 {doc.get('finished_at', '-')} · 最近 checkpoint `{doc.get('last_checkpoint') or monitor.last_checkpoint(str(root)) or '-'}`")
    if doc.get("note"):
        md += f"\n\n备注: {doc['note']}"
    stxt = script.read_text(encoding="utf-8") if script.is_file() else "(无 run.sh)"
    ctxt = cfg.read_text(encoding="utf-8") if cfg.is_file() else "(无 train_config.json)"
    abortable = info["local"] and st in ("running", "launching")
    return md, stxt, ctxt, abortable


def build() -> dict:
    with gr.Row():
        refresh_btn = gr.Button("刷新", scale=1)
        local_only = gr.Checkbox(label="只看本机", value=False, scale=1)
        status_f = gr.Dropdown(choices=["全部", "running", "launching", "finished", "failed", "aborted"], value="全部",
                               label="状态过滤", scale=2)
        auto = gr.Checkbox(label="自动刷新选中 run", value=True, scale=1)
    runs_df = gr.Dataframe(value=runs_frame(), headers=COLUMNS, interactive=False, wrap=True, label="训练记录（点击一行查看）")
    selected = gr.Textbox(label="选中的 run_id", interactive=False)
    with gr.Row():
        clone_btn = gr.Button("复刻参数到 Train")
        abort_confirm = gr.Checkbox(label="确认中止", value=False)
        abort_btn = gr.Button("中止训练", variant="stop", interactive=False)
        del_confirm = gr.Checkbox(label="确认删除记录（不删输出目录）", value=False)
        del_btn = gr.Button("删除记录")
        note = gr.Textbox(label="备注", scale=2)
        note_btn = gr.Button("保存备注")
    action_out = gr.Textbox(label="操作结果", lines=2)
    detail_md = gr.Markdown("未选择 run")
    plot_comps = []
    for i in range(0, len(RUN_PLOTS), 3):
        with gr.Row():
            for name in RUN_PLOTS[i:i + 3]:
                plot_comps.append(gr.Plot(label=name))
    with gr.Row():
        custom_cols = gr.Dropdown(choices=[], multiselect=True, label="自定义曲线列", scale=3)
        p_custom = gr.Plot(label="自定义", scale=5)
    with gr.Row():
        log_filter = gr.Textbox(label="日志过滤（包含子串）", scale=2)
        gpu_df = gr.Dataframe(label=f"GPU ({hostname()})", interactive=False, scale=4)
    log_box = gr.Textbox(label="训练日志", lines=22, max_lines=22, autoscroll=True, show_copy_button=True)
    with gr.Accordion("启动脚本 / 参数快照", open=False):
        script_code = gr.Code(label="ui/run.sh", language="shell", lines=16)
        cfg_code = gr.Code(label="ui/train_config.json", language="json", lines=16)

    def do_refresh(local, st):
        return runs_frame(local, st)
    refresh_btn.click(do_refresh, [local_only, status_f], runs_df)
    local_only.change(do_refresh, [local_only, status_f], runs_df)
    status_f.change(do_refresh, [local_only, status_f], runs_df)

    def _logs(run_id, filt):
        doc = registry.get(run_id) if run_id else None
        if not doc:
            return ""
        text = monitor.read_tail(monitor.train_log_path(doc["run_root"]), 60_000)
        if (filt or "").strip():
            text = "\n".join(line for line in text.splitlines() if filt in line)
        return text[-60_000:]

    def poll(run_id, filt):
        md, stxt, ctxt, abortable = _details(run_id)
        return md, gr.update(interactive=abortable), monitor.gpu_status(), _logs(run_id, filt), stxt, ctxt
    poll_outputs = [detail_md, abort_btn, gpu_df, log_box, script_code, cfg_code]

    def poll_plots(run_id, cols):
        doc = registry.get(run_id) if run_id else None
        df = monitor.load_metrics(doc["run_root"]) if doc and doc.get("run_root") else pd.DataFrame()
        numeric = plots.available_columns(df)
        keep = [c for c in (cols or []) if c in numeric]
        return [*[plots.panel(df, n) for n in RUN_PLOTS], gr.update(choices=numeric, value=keep), plots.custom(df, keep)]
    plot_outputs = [*plot_comps, custom_cols, p_custom]

    def on_select(df: pd.DataFrame, evt: gr.SelectData):
        try:
            run_id = str(df.iloc[evt.index[0]]["run_id"])
        except (IndexError, KeyError, TypeError):
            return "", ""
        return run_id, (registry.get(run_id) or {}).get("note", "")
    runs_df.select(on_select, runs_df, [selected, note]).then(poll, [selected, log_filter], poll_outputs).then(
        poll_plots, [selected, custom_cols], plot_outputs)
    log_filter.submit(poll, [selected, log_filter], poll_outputs)
    custom_cols.change(poll_plots, [selected, custom_cols], plot_outputs)

    t_fast = gr.Timer(4, active=True)
    t_slow = gr.Timer(10, active=True)
    t_fast.tick(poll, [selected, log_filter], poll_outputs, show_progress="hidden")
    t_slow.tick(poll_plots, [selected, custom_cols], plot_outputs, show_progress="hidden")
    auto.change(lambda on: (gr.Timer(active=on), gr.Timer(active=on)), auto, [t_fast, t_slow])

    def do_abort(run_id, ok):
        if not run_id:
            return "未选择 run"
        if not ok:
            return "请先勾选 确认中止"
        return launcher.abort(run_id)
    abort_btn.click(do_abort, [selected, abort_confirm], action_out).then(
        poll, [selected, log_filter], poll_outputs).then(do_refresh, [local_only, status_f], runs_df)

    def do_delete(run_id, ok, local, st):
        if not run_id:
            return "未选择 run", gr.update()
        if not ok:
            return "请先勾选 确认删除记录", gr.update()
        doc = registry.get(run_id)
        if doc and doc.get("status") in ("running", "launching") and doc.get("hostname") == hostname():
            return "运行中的 run 不能删除，请先中止", gr.update()
        registry.delete(run_id)
        return f"已删除记录 {run_id}（输出目录保留：{(doc or {}).get('run_root')}）", runs_frame(local, st)
    del_btn.click(do_delete, [selected, del_confirm, local_only, status_f], [action_out, runs_df])

    def do_note(run_id, text):
        if not run_id:
            return "未选择 run"
        registry.update(run_id, note=text)
        return "备注已保存"
    note_btn.click(do_note, [selected, note], action_out)

    return {"selected": selected, "clone_btn": clone_btn, "runs_df": runs_df, "refresh": do_refresh,
            "local_only": local_only, "status_f": status_f}
