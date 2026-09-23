"""Metrics DataFrame -> plotly figures."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .monitor import numeric_columns

PANELS = {
    "Loss": ["loss", "eval_loss"],
    "学习率": ["learning_rate"],
    "梯度范数": ["grad_norm"],
    "显存 (GiB)": ["memory(GiB)"],
    "MoE 均衡损失": ["load_balancing_loss", "seq_load_balancing_loss", "global_load_balancing_loss", "z_loss"],
    "速度 (s/it)": ["train_speed(s/it)"],
}


def figure(df: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    if df is None or df.empty or not columns:
        fig.update_layout(title=f"{title}（暂无数据）", height=280, margin=dict(l=40, r=20, t=40, b=30))
        return fig
    x = df["step"].tolist() if "step" in df.columns else list(range(1, len(df) + 1))
    for c in columns:
        if c not in df.columns:
            continue
        fig.add_trace(go.Scatter(x=x, y=df[c].tolist(), mode="lines+markers", name=c))
    fig.update_layout(title=title, xaxis_title="step", height=280, margin=dict(l=40, r=20, t=40, b=30),
                      legend=dict(orientation="h", y=-0.25), hovermode="x unified")
    return fig


def panel(df: pd.DataFrame, name: str) -> go.Figure:
    cols = [c for c in PANELS[name] if not df.empty and c in df.columns]
    return figure(df, cols, name)


def custom(df: pd.DataFrame, columns: list[str]) -> go.Figure:
    return figure(df, [c for c in (columns or []) if c in df.columns], "自定义")


def available_columns(df: pd.DataFrame) -> list[str]:
    return numeric_columns(df)
