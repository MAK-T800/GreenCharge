"""
GreenCharge Dashboard Utilities
================================
Clean, editorial chart factories and helpers. Black theme, no decoration.
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from config import DASHBOARD_CONFIG, MODEL_NAMES, PLOTS_DIR

# ── Palette ──────────────────────────────────────────────────────────────
BLACK = "#000000"
NEAR_BLACK = "#0a0a0a"
DARK = "#111111"
BORDER = "#222222"
MUTED = "#555555"
GRAY = "#888888"
LIGHT = "#bbbbbb"
WHITE = "#eeeeee"
ACCENT = "#e63946"
ACCENT_DIM = "rgba(230,57,70,0.12)"

# Re-export for app.py
PRIMARY = ACCENT
SECONDARY = "#888888"
TEXT_COLOR = WHITE
CARD_COLOR = DARK
BG_COLOR = BLACK
DANGER = ACCENT
WARNING = "#f4a261"
INFO = "#888888"


def get_plotly_layout(title="", height=400, xaxis_title="", yaxis_title=""):
    """Minimal black layout."""
    return go.Layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color=WHITE, family="Inter, Segoe UI, sans-serif"),
            x=0, xanchor="left", y=0.98,
        ),
        plot_bgcolor=BLACK,
        paper_bgcolor=BLACK,
        height=height,
        font=dict(color=GRAY, family="Inter, Segoe UI, sans-serif", size=11),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=11, color=MUTED)),
            gridcolor="#1a1a1a", zerolinecolor="#1a1a1a",
            tickfont=dict(size=10, color=MUTED),
            showline=False,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=11, color=MUTED)),
            gridcolor="#1a1a1a", zerolinecolor="#1a1a1a",
            tickfont=dict(size=10, color=MUTED),
            showline=False,
        ),
        margin=dict(l=50, r=20, t=45, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=GRAY),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        hoverlabel=dict(
            bgcolor="#111111", font_size=12, font_color=WHITE,
            font_family="Inter, sans-serif", bordercolor="#333333",
        ),
    )


def plot_consumption_timeline(df, building_id=None):
    if building_id:
        data = df[df["building_id"] == building_id].copy()
        title = f"Energy Consumption / Building {building_id}"
    else:
        data = df.groupby("timestamp")["meter_reading"].mean().reset_index()
        title = "Average Energy Consumption"

    fig = go.Figure(layout=get_plotly_layout(title, 360, "", "kWh"))
    fig.add_trace(go.Scatter(
        x=data["timestamp"], y=data["meter_reading"],
        mode='lines', fill='tozeroy',
        line=dict(color=ACCENT, width=1.2),
        fillcolor=ACCENT_DIM,
        name="Consumption",
    ))
    return fig


def plot_hourly_heatmap(df):
    pivot = df.pivot_table(values="meter_reading", index="day_of_week", columns="hour", aggfunc="mean")
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}" for h in pivot.columns],
        y=day_labels[:len(pivot.index)],
        colorscale=[[0, BLACK], [0.4, "#1a0a0b"], [0.7, "#6b1520"], [1, ACCENT]],
        colorbar=dict(
            title=dict(text="kWh", font=dict(color=GRAY, size=10)),
            tickfont=dict(color=MUTED, size=9), thickness=10, len=0.8,
            outlinewidth=0,
        ),
        xgap=2, ygap=2,
    ))
    fig.update_layout(get_plotly_layout("Hourly Heatmap", 320, "Hour", ""))
    return fig


def plot_building_comparison(df):
    bldg_avg = df.groupby("building_id")["meter_reading"].mean().sort_values(ascending=False).head(12)

    fig = go.Figure(data=go.Bar(
        x=[f"B{b}" for b in bldg_avg.index],
        y=bldg_avg.values,
        marker=dict(color=ACCENT, opacity=0.75, line=dict(width=0)),
    ))
    fig.update_layout(get_plotly_layout("Avg Load by Building", 320, "", "kWh"))
    return fig


def plot_forecast(actuals, predictions, model_name="Model", timestamps=None):
    fig = go.Figure(layout=get_plotly_layout(f"{model_name} / Forecast", 400, "", "kWh"))

    act_arr = np.array(actuals) if actuals is not None and len(actuals) > 0 else None
    pred_arr = np.array(predictions)
    n = len(pred_arr)
    x = timestamps if timestamps is not None else list(range(n))

    if act_arr is not None and len(act_arr) == n:
        fig.add_trace(go.Scatter(
            x=x, y=act_arr, mode='lines',
            line=dict(color=GRAY, width=1), name="Actual", opacity=0.7,
        ))

    fig.add_trace(go.Scatter(
        x=x, y=pred_arr, mode='lines',
        line=dict(color=ACCENT, width=1.8), name="Predicted",
    ))

    if act_arr is not None and len(act_arr) == n:
        residual = np.abs(act_arr - pred_arr)
        fig.add_trace(go.Scatter(
            x=x, y=residual, mode='lines', fill='tozeroy',
            line=dict(color=ACCENT, width=0.4), fillcolor=ACCENT_DIM,
            name="Error", yaxis="y2",
        ))
        fig.update_layout(yaxis2=dict(
            title="", overlaying="y", side="right",
            gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9, color=MUTED),
        ))
    return fig


def plot_model_comparison_bar(metrics_df):
    if metrics_df.empty:
        return go.Figure()

    fig = make_subplots(rows=1, cols=4,
        subplot_titles=["R2", "RMSE", "MAE", "MAPE (%)"],
        horizontal_spacing=0.08)

    models = metrics_df["Model"].tolist()
    colors = [ACCENT if "BiLSTM" in m or "Ensemble" in m else GRAY for m in models]

    for i, (metric, col) in enumerate([("R2", 1), ("RMSE", 2), ("MAE", 3), ("MAPE", 4)]):
        if metric in metrics_df.columns:
            fig.add_trace(go.Bar(
                x=models, y=metrics_df[metric],
                marker=dict(color=colors, line=dict(width=0)),
                showlegend=False,
            ), row=1, col=col)

    fig.update_layout(
        height=350, plot_bgcolor=BLACK, paper_bgcolor=BLACK,
        font=dict(color=GRAY, family="Inter, sans-serif", size=9),
        margin=dict(l=40, r=20, t=50, b=50),
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor="#1a1a1a", tickfont=dict(size=8, color=MUTED))
    return fig


def plot_anomaly_timeline(df):
    fig = go.Figure(layout=get_plotly_layout("Anomaly Timeline", 380, "", "kWh"))
    normal = df[df.get("is_anomaly", 0) == 0]
    if len(normal) > 0:
        fig.add_trace(go.Scatter(
            x=normal["timestamp"], y=normal["meter_reading"],
            mode='lines', line=dict(color=MUTED, width=0.8),
            name="Normal", opacity=0.5,
        ))

    severity_cfg = {"Low": (WARNING, "circle", 6), "Medium": ("#FF9100", "diamond", 8), "High": (ACCENT, "x", 10)}
    for severity, (color, symbol, size) in severity_cfg.items():
        mask = (df.get("is_anomaly", 0) == 1) & (df.get("severity", "") == severity)
        anom = df[mask]
        if len(anom) > 0:
            fig.add_trace(go.Scatter(
                x=anom["timestamp"], y=anom["meter_reading"],
                mode='markers', marker=dict(color=color, size=size, symbol=symbol),
                name=f"{severity}",
            ))
    return fig


def plot_load_balance_comparison(original, optimized, title="Load Balancing"):
    hours = list(range(24))
    orig = [original.get(h, original.get(str(h), 0)) for h in hours]
    opt = [optimized.get(h, optimized.get(str(h), 0)) for h in hours]
    x_labels = [f"{h:02d}" for h in hours]

    fig = go.Figure(layout=get_plotly_layout(title, 360, "Hour", "kW"))
    fig.add_trace(go.Bar(x=x_labels, y=orig, name="Before", marker=dict(color=MUTED, opacity=0.5)))
    fig.add_trace(go.Bar(x=x_labels, y=opt, name="After", marker=dict(color=ACCENT, opacity=0.8)))
    fig.update_layout(barmode='group', bargap=0.15)
    return fig


def plot_solar_integration(solar_results):
    hours = list(range(24))
    fig = go.Figure(layout=get_plotly_layout("Solar Integration", 380, "Hour", "kW"))

    for key, name, color, fill in [
        ("original_load", "Original", MUTED, "rgba(136,136,136,0.06)"),
        ("solar_output", "Solar", WARNING, "rgba(244,162,97,0.08)"),
        ("net_load", "Net Grid", ACCENT, ACCENT_DIM),
    ]:
        vals = [solar_results[key].get(h, solar_results[key].get(str(h), 0)) for h in hours]
        fig.add_trace(go.Scatter(
            x=[f"{h:02d}" for h in hours], y=vals,
            mode='lines', fill='tozeroy',
            line=dict(color=color, width=1.5), fillcolor=fill, name=name,
        ))
    return fig


def plot_staggered_scheduling(stagger_results):
    hours = list(range(24))
    fig = go.Figure(layout=get_plotly_layout("Staggered Scheduling", 360, "Hour", "kW"))
    fig.add_trace(go.Scatter(
        x=[f"{h:02d}" for h in hours], y=stagger_results["original_profile"],
        mode='lines+markers', line=dict(color=MUTED, width=1.5, dash='dot'),
        marker=dict(size=5), name="Original",
    ))
    fig.add_trace(go.Scatter(
        x=[f"{h:02d}" for h in hours], y=stagger_results["staggered_profile"],
        mode='lines+markers', line=dict(color=ACCENT, width=2),
        marker=dict(size=5), name="Staggered",
    ))
    return fig


def plot_model_comparison_radar(metrics_df):
    """Not used in new design — returns empty."""
    return go.Figure()


def get_plot_path(filename):
    """Return path to a saved matplotlib plot."""
    return os.path.join(PLOTS_DIR, filename)


def render_kpi_cards(st, kpis):
    """Minimal metric cards."""
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        col.metric(kpi["label"], kpi["value"], delta=kpi.get("delta", None))
