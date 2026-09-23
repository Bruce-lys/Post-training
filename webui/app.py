#!/usr/bin/env python3
"""Megatron-SWIFT Web UI entry point: LlamaFactory-style Gradio front end for ``megatron sft`` runs."""
from __future__ import annotations

import argparse
import sys

import gradio as gr

from core.constants import hostname
from core.settings import load_ui_settings
from ui import tab_runs, tab_settings, tab_train, top

CSS = """
.gradio-container { max-width: 1600px !important; }
footer { display: none !important; }
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Megatron-SWIFT WebUI", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"## Megatron-SWIFT 训练 WebUI  <small>节点 `{hostname()}`</small>")
        top_comps = top.build()
        with gr.Tabs() as tabs:
            with gr.Tab("Train", id="train"):
                train = tab_train.build(top_comps)
            with gr.Tab("训练记录", id="runs"):
                runs = tab_runs.build()
            with gr.Tab("设置", id="settings"):
                tab_settings.build()

        # ---- cross-tab wiring -------------------------------------------------
        def launch(nm, mp, orp, ds, *values):
            msg, run_id, code = train["do_launch"](nm, mp, orp, ds, *values)
            table = tab_runs.runs_frame() if isinstance(run_id, str) else gr.update()
            return msg, run_id, code, table
        train["start_btn"].click(launch, [*train["top_inputs"], *train["edit_comps"]],
                                 [train["output"], train["current_run"], train["preview_code"], runs["runs_df"]]).then(
            train["poll"], train["current_run"], train["poll_outputs"])

        def clone_to_train(run_id):
            from core import presets
            state = presets.state_from_run(run_id) if run_id else None
            if not state:
                return [gr.update()] * len(train["apply_state_outputs"]) + [gr.update(), gr.update(), gr.update()]
            from core import registry
            doc = registry.get(run_id) or {}
            return train["set_state_values"](state) + [doc.get("model") or gr.update(), doc.get("dataset") or gr.update(),
                                                       gr.Tabs(selected="train")]
        runs["clone_btn"].click(clone_to_train, runs["selected"],
                                train["apply_state_outputs"] + [top_comps["model_path"], train["dataset"], tabs])

        tabs.select(lambda: tab_runs.runs_frame(), None, runs["runs_df"], show_progress="hidden")
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-auth", action="store_true")
    args = ap.parse_args()
    ui = load_ui_settings()
    demo = build()
    auth = None if args.no_auth else (ui.auth_user, ui.auth_password)
    demo.queue(default_concurrency_limit=8).launch(server_name=args.host, server_port=args.port or ui.port,
                                                   auth=auth, share=False, show_api=False, inbrowser=False)


if __name__ == "__main__":
    sys.exit(main())
