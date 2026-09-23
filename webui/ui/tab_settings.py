"""Settings tab: paths.env fields, UI settings (port / auth / fake train), model table."""
from __future__ import annotations

import json

import gradio as gr
import pandas as pd

from core import settings as st
from core.constants import hostname

ICON = {"ok": "✅", "warn": "⚠", "error": "❌"}


def build() -> dict:
    values = st.load_paths_env()
    ui = st.load_ui_settings()
    fields: dict[str, gr.Textbox] = {}

    gr.Markdown(f"路径设置存于 `webui/settings/paths.env`（所有节点共用），本机覆盖存于 `paths.{hostname()}.env`（只保存不同的项）。")
    for f in st.ENV_FIELDS:
        fields[f.name] = gr.Textbox(value=values.get(f.name, ""), label=f"{f.name} — {f.label}", info=f.help)
    with gr.Row():
        scope = gr.Radio(choices=["shared", "host"], value="shared", label="保存作用域", info="shared=所有节点共用；host=仅本机覆盖")
        validate_btn = gr.Button("校验路径")
        save_btn = gr.Button("保存路径", variant="primary")
    env_msg = gr.Textbox(label="", interactive=False)
    check_df = gr.Dataframe(headers=["字段", "状态", "说明"], interactive=False, wrap=True, label="校验结果")

    gr.Markdown("### UI 设置")
    with gr.Row():
        port = gr.Number(value=ui.port, label="UI 端口（重启 UI 生效）", precision=0)
        auth_user = gr.Textbox(value=ui.auth_user, label="登录用户名（重启生效）")
        auth_pw = gr.Textbox(value=ui.auth_password, label="登录密码（重启生效）", type="password")
    with gr.Row():
        fake = gr.Checkbox(value=ui.fake_train, label="假训练模式（不占 GPU，只写模拟的 loss 曲线，用来试 UI）")
        fake_interval = gr.Number(value=ui.fake_interval, label="假训练每步秒数", minimum=0.2)
    model_table = gr.Code(value=json.dumps(ui.model_table, indent=2, ensure_ascii=False), label="模型列表（名称 → 路径，JSON）",
                          language="json", lines=6)
    save_ui_btn = gr.Button("保存 UI 设置", variant="primary")
    ui_msg = gr.Textbox(label="", interactive=False)

    names = [f.name for f in st.ENV_FIELDS]
    field_comps = [fields[n] for n in names]

    def do_validate(*vals):
        rows = st.validate_paths_env(dict(zip(names, vals)))
        return pd.DataFrame([(n, ICON[s], m) for n, s, m in rows], columns=["字段", "状态", "说明"])
    validate_btn.click(do_validate, field_comps, check_df)

    def do_save(sc, *vals):
        return f"已保存到 {st.save_paths_env(dict(zip(names, vals)), sc)}"
    save_btn.click(do_save, [scope, *field_comps], env_msg)

    def do_save_ui(p, u, pw, fk, fi, mt):
        cur = st.load_ui_settings()
        try:
            table = json.loads(mt or "{}")
            if not isinstance(table, dict):
                raise ValueError("必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as exc:
            return f"模型列表不是合法 JSON: {exc}"
        cur.port, cur.auth_user, cur.auth_password = int(p), u, pw
        cur.fake_train, cur.fake_interval, cur.model_table = bool(fk), float(fi), table
        return f"已保存到 {st.save_ui_settings(cur)}"
    save_ui_btn.click(do_save_ui, [port, auth_user, auth_pw, fake, fake_interval, model_table], ui_msg)

    return {"fields": fields}
