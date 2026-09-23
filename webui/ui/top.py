"""Top block shared by every tab (LlamaFactory style): experiment name, model name/path, output root."""
from __future__ import annotations

import gradio as gr

from core.settings import load_paths_env, load_ui_settings


def build() -> dict:
    env = load_paths_env()
    ui = load_ui_settings()
    names = list(ui.model_table.keys())
    with gr.Row():
        name = gr.Textbox(value="run", label="实验名", info="run_id 前缀", scale=1)
        model_name = gr.Dropdown(choices=names, value=names[0] if names else None, allow_custom_value=True,
                                 label="模型名称", info="选择后自动填模型路径（设置页可维护列表）", scale=2)
        model_path = gr.Textbox(value=env.get("MODEL_ROOT", ""), label="模型路径", info="HF 格式模型目录（--model）", scale=4)
        output_root = gr.Textbox(value=env.get("OUTPUT_ROOT", ""), label="输出根目录",
                                 info="每次训练在其下建 <run_id>/", scale=3)

    def on_model(nm):
        path = load_ui_settings().model_table.get(nm or "")
        return path if path else gr.update()
    model_name.change(on_model, model_name, model_path)

    return {"name": name, "model_name": model_name, "model_path": model_path, "output_root": output_root}
