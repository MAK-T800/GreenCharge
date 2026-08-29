"""
GreenCharge Dashboard
=====================
Black editorial design. XGBoost + BiLSTM focused.
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_TRAIN, PROCESSED_VAL, PROCESSED_FEATURES,
    METRICS_DIR, MODELS_DIR, PLOTS_DIR, MODEL_NAMES,
    TARGET_COL, LOAD_BALANCE_CONFIG
)
from dashboard_utils import *
from models.evaluation import load_metrics, compare_all_models


# ─── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenCharge",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* Root */
    .stApp {
        background: #000000;
        font-family: 'Inter', -apple-system, sans-serif;
        color: #eeeeee;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 1px solid #1a1a1a !important;
    }

    /* Kill all default padding bloat */
    .block-container { padding-top: 2rem; }

    /* Headings */
    h1 { color: #ffffff !important; font-weight: 900 !important; letter-spacing: -1px !important; }
    h2 { color: #eeeeee !important; font-weight: 700 !important; letter-spacing: -0.5px !important; }
    h3 { color: #cccccc !important; font-weight: 600 !important; }

    /* Metric cards */
    .stMetric {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 4px;
        padding: 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-weight: 700;
        font-size: 24px !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #666666;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetricDelta"] {
        color: #555555;
        font-size: 11px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #222222;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        color: #666666;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #aaaaaa; }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #ffffff !important;
        border-bottom: 2px solid #e63946 !important;
    }

    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: #1a1a1a;
        margin: 28px 0;
    }

    /* Radio nav */
    .stRadio > div { gap: 0; }
    .stRadio > div > label {
        background: transparent;
        border: none;
        border-left: 2px solid transparent;
        border-radius: 0;
        padding: 10px 16px !important;
        color: #777777;
        margin: 0;
    }
    .stRadio > div > label:hover {
        color: #cccccc;
        background: #0a0a0a;
    }
    .stRadio > div > label[data-checked="true"] {
        color: #ffffff;
        border-left: 2px solid #e63946;
        background: #0a0a0a;
    }

    /* Select */
    .stSelectbox [data-baseweb="select"] > div {
        background: #0a0a0a;
        border: 1px solid #222222;
        border-radius: 4px;
    }

    /* DataFrames */
    .stDataFrame { border-radius: 4px; overflow: hidden; }

    /* Slider */
    .stSlider > div > div > div { color: #888888; }

    /* Images */
    .stImage { border-radius: 4px; overflow: hidden; }

    /* Captions */
    .plot-caption {
        color: #555555;
        font-size: 12px;
        margin-top: 6px;
        line-height: 1.4;
    }

    /* Section rule */
    .section-rule {
        border-top: 1px solid #222222;
        padding-top: 28px;
        margin-top: 28px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }

    /* Kill emoji mess in sidebar */
    [data-testid="stSidebar"] .stMarkdown p { font-size: 12px; color: #555555; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = {}
    if os.path.exists(PROCESSED_FEATURES):
        data["full"] = pd.read_parquet(PROCESSED_FEATURES)
    if os.path.exists(PROCESSED_TRAIN):
        data["train"] = pd.read_parquet(PROCESSED_TRAIN)
    if os.path.exists(PROCESSED_VAL):
        data["val"] = pd.read_parquet(PROCESSED_VAL)
    return data


@st.cache_data
def load_all_metrics():
    return load_metrics(METRICS_DIR)


@st.cache_data
def load_comparison():
    return compare_all_models(METRICS_DIR)


@st.cache_data
def load_model_predictions():
    pred_path = os.path.join(MODELS_DIR, "all_predictions.json")
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            return json.load(f)
    return {}


def load_single_metric(name):
    p = os.path.join(METRICS_DIR, f"{name}.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return d.get("metrics", d)
    return {}


# ─── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 16px 0 8px;">
        <div style="font-size: 28px; font-weight: 900; color: #ffffff; letter-spacing: -1px;">
            GreenCharge
        </div>
        <div style="font-size: 11px; color: #555555; margin-top: 2px; letter-spacing: 0.5px;">
            Load Forecasting System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Nav",
        ["Overview", "Model Results", "Anomalies", "Load Balancing", "Recommendations"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("College Hostel Energy System")
    st.caption("SDG 12: Responsible Consumption")


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <h1 style="font-size: 44px; margin-bottom: 0; line-height: 1.1;">
        Energy Intelligence<br>
        <span style="color: #e63946;">for College Hostels</span>
    </h1>
    <p style="color: #555555; font-size: 15px; margin-top: 12px; max-width: 600px;">
        AI-powered load forecasting and optimization using XGBoost and BiLSTM
        neural networks, trained on 168-hour lookback windows across 12 hostel buildings.
    </p>
    """, unsafe_allow_html=True)

    data = load_data()

    if "full" in data:
        df = data["full"]

        total_energy = df[TARGET_COL].sum()
        avg_daily = df.groupby(df["timestamp"].dt.date)[TARGET_COL].sum().mean()
        peak_load = df[TARGET_COL].max()
        n_buildings = df["building_id"].nunique()

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Energy", f"{total_energy:,.0f} kWh")
        c2.metric("Avg Daily", f"{avg_daily:,.0f} kWh")
        c3.metric("Peak Load", f"{peak_load:,.1f} kW")
        c4.metric("Buildings", f"{n_buildings}")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_consumption_timeline(df), width='stretch')
        with col2:
            if "hour" in df.columns and "day_of_week" in df.columns:
                st.plotly_chart(plot_hourly_heatmap(df), width='stretch')

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plot_building_comparison(df), width='stretch')
        with col4:
            monthly = df.groupby(df["timestamp"].dt.month)[TARGET_COL].mean()
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            fig = go.Figure(data=go.Scatter(
                x=months[:len(monthly)], y=monthly.values,
                mode='lines+markers',
                line=dict(color=ACCENT, width=2),
                marker=dict(size=7, color=ACCENT),
                fill='tozeroy', fillcolor=ACCENT_DIM,
            ))
            fig.update_layout(get_plotly_layout("Monthly Trend", 320, "", "Avg kWh"))
            st.plotly_chart(fig, width='stretch')
    else:
        st.warning("No processed data. Run `python data_pipeline.py` first.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Model Results (XGBoost + BiLSTM only)
# ═════════════════════════════════════════════════════════════════════════
elif page == "Model Results":
    st.markdown("""
    <h1 style="font-size: 40px; margin-bottom: 0; line-height: 1.1;">
        Model Results
    </h1>
    <p style="color: #555555; font-size: 14px; margin-top: 8px;">
        XGBoost and XGBoost + BiLSTM / 168-hour lookback / validation set
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Load metrics
    m_xgb = load_single_metric("xgboost")
    m_bil = load_single_metric("xgb_bilstm")

    # ── Side-by-side metrics ────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<h2 style="font-size: 22px; margin-bottom: 16px;">XGBoost</h2>', unsafe_allow_html=True)
        if m_xgb:
            ca, cb, cc, cd = st.columns(4)
            ca.metric("R2", f"{m_xgb.get('R2', 0):.4f}")
            cb.metric("RMSE", f"{m_xgb.get('RMSE', 0):.2f}")
            cc.metric("MAE", f"{m_xgb.get('MAE', 0):.2f}")
            cd.metric("MAPE", f"{m_xgb.get('MAPE', 0):.2f}%")

    with col_b:
        st.markdown('<h2 style="font-size: 22px; margin-bottom: 16px;">XGBoost + BiLSTM</h2>', unsafe_allow_html=True)
        if m_bil:
            ca, cb, cc, cd = st.columns(4)
            ca.metric("R2", f"{m_bil.get('R2', 0):.4f}")
            cb.metric("RMSE", f"{m_bil.get('RMSE', 0):.2f}")
            cc.metric("MAE", f"{m_bil.get('MAE', 0):.2f}")
            cd.metric("MAPE", f"{m_bil.get('MAPE', 0):.2f}%")

    st.markdown("---")

    # ── Comparison bar chart (matplotlib) ───────────────────────────
    comp_path = os.path.join(PLOTS_DIR, "comparison_xgb_vs_bilstm.png")
    if os.path.exists(comp_path):
        st.image(comp_path)
        st.markdown('<p class="plot-caption">Side-by-side metric comparison. Lower RMSE/MAE/MAPE is better, higher R2 is better.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs: XGBoost / XGBoost+BiLSTM ──────────────────────────────
    tab1, tab2 = st.tabs(["XGBoost", "XGBoost + BiLSTM"])

    with tab1:
        st.markdown('<h2 style="font-size: 24px; margin: 16px 0 8px;">XGBoost</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #555; font-size: 13px; margin-bottom: 20px;">Gradient-boosted decision trees. Fast, interpretable, strong baseline.</p>', unsafe_allow_html=True)

        # Forecast
        forecast_path = os.path.join(PLOTS_DIR, "xgboost_forecast.png")
        if os.path.exists(forecast_path):
            st.image(forecast_path)
            st.markdown('<p class="plot-caption">Actual vs predicted energy consumption on the validation set (500 samples). Red fill shows prediction error magnitude.</p>', unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            scatter_path = os.path.join(PLOTS_DIR, "xgboost_scatter.png")
            if os.path.exists(scatter_path):
                st.image(scatter_path)
                st.markdown('<p class="plot-caption">Scatter plot of actual vs predicted values. Points near the diagonal indicate accurate predictions.</p>', unsafe_allow_html=True)
        with col2:
            error_path = os.path.join(PLOTS_DIR, "xgboost_errors.png")
            if os.path.exists(error_path):
                st.image(error_path)
                st.markdown('<p class="plot-caption">Distribution of prediction errors. A narrow, centered distribution indicates consistent model performance.</p>', unsafe_allow_html=True)

        # Interactive plotly forecast
        st.markdown("---")
        predictions = load_model_predictions()
        if "xgboost" in predictions:
            pd_data = predictions["xgboost"]
            st.plotly_chart(plot_forecast(
                pd_data.get("actuals", []), pd_data.get("predictions", []), "XGBoost"
            ), width='stretch')
            st.markdown('<p class="plot-caption">Interactive forecast. Hover for exact values.</p>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<h2 style="font-size: 24px; margin: 16px 0 8px;">XGBoost + BiLSTM</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #555; font-size: 13px; margin-bottom: 20px;">Hybrid architecture: XGBoost base predictions + BiLSTM residual correction on 168h sequences.</p>', unsafe_allow_html=True)

        forecast_path = os.path.join(PLOTS_DIR, "xgb_bilstm_forecast.png")
        if os.path.exists(forecast_path):
            st.image(forecast_path)
            st.markdown('<p class="plot-caption">Actual vs predicted with residual correction. The BiLSTM refines XGBoost predictions by learning temporal residual patterns.</p>', unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            scatter_path = os.path.join(PLOTS_DIR, "xgb_bilstm_scatter.png")
            if os.path.exists(scatter_path):
                st.image(scatter_path)
                st.markdown('<p class="plot-caption">Scatter plot. Tighter clustering around the diagonal compared to standalone XGBoost.</p>', unsafe_allow_html=True)
        with col2:
            error_path = os.path.join(PLOTS_DIR, "xgb_bilstm_errors.png")
            if os.path.exists(error_path):
                st.image(error_path)
                st.markdown('<p class="plot-caption">Error distribution for the hybrid model.</p>', unsafe_allow_html=True)

        st.markdown("---")
        if "xgb_bilstm" in predictions:
            pd_data = predictions["xgb_bilstm"]
            st.plotly_chart(plot_forecast(
                pd_data.get("actuals", []), pd_data.get("predictions", []), "XGBoost + BiLSTM"
            ), width='stretch')
            st.markdown('<p class="plot-caption">Interactive forecast. Hover for exact values.</p>', unsafe_allow_html=True)

    # ── Full leaderboard (collapsed) ────────────────────────────────
    st.markdown("---")
    with st.expander("Full Model Leaderboard"):
        comparison_df = load_comparison()
        if not comparison_df.empty:
            st.dataframe(
                comparison_df.style.format({
                    "R2": "{:.6f}", "RMSE": "{:.4f}", "MAE": "{:.4f}", "MAPE": "{:.4f}%"
                }),
                width='stretch', height=min(400, 35 * len(comparison_df) + 38),
            )


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Anomalies
# ═════════════════════════════════════════════════════════════════════════
elif page == "Anomalies":
    st.markdown("""
    <h1 style="font-size: 40px; margin-bottom: 0;">Anomaly Detection</h1>
    <p style="color: #555555; font-size: 14px; margin-top: 8px;">
        Statistical outlier detection across hostel buildings
    </p>
    """, unsafe_allow_html=True)

    data = load_data()

    if "full" in data:
        df = data["full"]
        has_anomalies = "is_anomaly" in df.columns

        if has_anomalies:
            n_total = len(df)
            n_anomalies = int(df["is_anomaly"].sum())
            high = len(df[df.get("severity", "") == "High"]) if "severity" in df.columns else 0

            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Readings", f"{n_total:,}")
            c2.metric("Anomalies", f"{n_anomalies:,}", delta=f"{100*n_anomalies/n_total:.2f}%")
            c3.metric("High Severity", f"{high:,}")

            st.markdown("---")
            st.plotly_chart(plot_anomaly_timeline(df.head(5000)), width='stretch')

            if n_anomalies > 0:
                st.markdown("---")
                st.markdown('<h2 style="font-size: 20px;">Recent Alerts</h2>', unsafe_allow_html=True)

                anomalies = df[df["is_anomaly"] == 1].sort_values("timestamp", ascending=False).head(10)
                for _, row in anomalies.iterrows():
                    severity = row.get("severity", "?")
                    color = {"High": "#e63946", "Medium": "#f4a261", "Low": "#888888"}.get(severity, "#555")
                    st.markdown(f"""
                    <div style="
                        border-left: 3px solid {color};
                        padding: 8px 14px;
                        margin: 4px 0;
                        background: #0a0a0a;
                        font-size: 13px;
                        color: #aaa;
                    ">
                        <span style="color: {color}; font-weight: 600;">{severity}</span>
                        &nbsp;/&nbsp; Building {row.get('building_id', '?')}
                        &nbsp;/&nbsp; {row.get('timestamp', '')}
                        &nbsp;/&nbsp; {row.get(TARGET_COL, 0):.1f} kWh
                        &nbsp;/&nbsp; {row.get('deviation_pct', 0):+.1f}%
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Anomaly detection not run yet. Run `python train.py`.")
    else:
        st.warning("No data. Run `python data_pipeline.py` first.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Load Balancing
# ═════════════════════════════════════════════════════════════════════════
elif page == "Load Balancing":
    st.markdown("""
    <h1 style="font-size: 40px; margin-bottom: 0;">Load Balancing</h1>
    <p style="color: #555555; font-size: 14px; margin-top: 8px;">
        Peak shaving, solar integration, staggered scheduling
    </p>
    """, unsafe_allow_html=True)

    data = load_data()
    lb_path = os.path.join(MODELS_DIR, "load_balance_results.json")

    if os.path.exists(lb_path):
        with open(lb_path) as f:
            lb = json.load(f)

        peak = lb.get("peak_shaving", {})
        solar = lb.get("solar_integration", {})
        stagger = lb.get("staggered_scheduling", {})
        combined = lb.get("strategy_c", {})

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak Reduction", f"{peak.get('peak_reduction_pct', 0):.1f}%")
        c2.metric("Solar Savings", f"{solar.get('solar_savings_pct', 0):.1f}%")
        c3.metric("Daily Savings", f"INR {combined.get('total_daily_savings', 0):.0f}")
        c4.metric("CO2 Reduced", f"{combined.get('carbon_reduced_kg', 0):.1f} kg/day")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Peak Shaving", "Solar", "Staggering"])

        with tab1:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(plot_load_balance_comparison(
                    peak.get("original_load", {}), peak.get("shaved_load", {}),
                    "Before vs After"
                ), width='stretch')
            with col2:
                st.metric("Cost Savings", f"INR {peak.get('cost_savings_daily', 0):.2f}/day")
                st.markdown(f"Peak hours: `{peak.get('peak_hours', [])}`")

        with tab2:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(plot_solar_integration(solar), width='stretch')
            with col2:
                st.metric("Generation", f"{solar.get('total_solar_kwh', 0):.1f} kWh/day")
                st.metric("Savings", f"{solar.get('solar_savings_pct', 0):.1f}%")
                st.metric("CO2", f"{solar.get('carbon_reduced_kg', 0):.1f} kg/day")

        with tab3:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(plot_staggered_scheduling(stagger), width='stretch')
            with col2:
                st.metric("Peak Reduction", f"{stagger.get('peak_reduction_pct', 0):.1f}%")
                st.metric("Load Factor (before)", f"{stagger.get('load_factor_before', 0):.3f}")
                st.metric("Load Factor (after)", f"{stagger.get('load_factor_after', 0):.3f}")

        st.markdown("---")
        st.markdown('<h2 style="font-size: 20px;">Combined Strategy</h2>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Daily", f"INR {combined.get('total_daily_savings', 0):.0f}")
        c2.metric("Monthly", f"INR {combined.get('total_daily_savings', 0) * 30:,.0f}")
        c3.metric("Annual", f"INR {combined.get('total_daily_savings', 0) * 365:,.0f}")

    elif "full" in data:
        st.info("Running load balancing simulation...")
        from load_balancing import run_load_balancing
        with st.spinner("Simulating..."):
            lb_results = run_load_balancing(data["full"])
            os.makedirs(os.path.dirname(lb_path), exist_ok=True)
            def conv(obj):
                if isinstance(obj, (np.int64, np.int32)): return int(obj)
                if isinstance(obj, (np.float64, np.float32)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, dict): return {k: conv(v) for k, v in obj.items()}
                if isinstance(obj, list): return [conv(v) for v in obj]
                return obj
            with open(lb_path, 'w') as f:
                json.dump(conv(lb_results), f, indent=2)
        st.rerun()
    else:
        st.warning("No data. Run `python data_pipeline.py` first.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Recommendations
# ═════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":
    st.markdown("""
    <h1 style="font-size: 40px; margin-bottom: 0;">Recommendations</h1>
    <p style="color: #555555; font-size: 14px; margin-top: 8px;">
        Energy optimization strategies aligned with SDG 12
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # SDG Card
    st.markdown(f"""
    <div style="
        background: #0a0a0a; border: 1px solid #1a1a1a;
        border-radius: 4px; padding: 28px; margin-bottom: 28px;
    ">
        <div style="font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 8px;">
            SDG 12: Responsible Consumption & Production
        </div>
        <div style="color: #777; font-size: 14px; line-height: 1.6; max-width: 700px;">
            GreenCharge promotes responsible energy consumption through AI-driven load forecasting,
            anomaly detection, and smart optimization strategies for college hostels.
        </div>
    </div>
    """, unsafe_allow_html=True)

    recs = [
        ("HIGH", "#e63946", "Install Smart Sub-Meters per Floor",
         "Deploy IoT-based smart meters on each hostel floor. Enables floor-level anomaly detection.",
         "15-20% waste reduction"),
        ("HIGH", "#e63946", "Automated Peak Shaving",
         "Use AI predictions to reduce non-critical loads during forecasted peaks. Pre-cool ACs, shift geysers.",
         "20-30% peak reduction"),
        ("MED", "#f4a261", "Rooftop Solar Panels",
         "100kW rooftop installation to offset grid consumption during 10am-3pm solar peak.",
         "10-15% renewable energy"),
        ("MED", "#f4a261", "Stagger High-Power Appliances",
         "Schedule geysers and washing machines across buildings to prevent simultaneous spikes.",
         "15-25% peak reduction"),
        ("LOW", "#888888", "Student Energy Dashboard",
         "Public display showing real-time consumption, floor comparisons, gamified targets.",
         "5-10% behavioral savings"),
        ("LOW", "#888888", "Upgrade Appliances",
         "Replace old fans, lights, geysers with BEE 5-star rated equipment.",
         "20-40% appliance savings"),
    ]

    for priority, color, title, desc, impact in recs:
        st.markdown(f"""
        <div style="
            border-left: 3px solid {color};
            padding: 14px 18px;
            margin: 6px 0;
            background: #0a0a0a;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #eee; font-weight: 600; font-size: 15px;">{title}</span>
                <span style="color: {color}; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;">{priority}</span>
            </div>
            <div style="color: #777; font-size: 13px; margin-top: 6px; line-height: 1.5;">{desc}</div>
            <div style="color: #555; font-size: 11px; margin-top: 6px;">Impact: {impact}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Calculator
    st.markdown('<h2 style="font-size: 20px;">Carbon Footprint Calculator</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        daily_kwh = st.slider("Daily Consumption (kWh)", 100, 5000, 1500)
        savings_pct = st.slider("Expected Savings (%)", 5, 50, 20)
        carbon_factor = st.number_input("Carbon Factor (kg CO2/kWh)", value=0.82)
    with c2:
        saved = daily_kwh * savings_pct / 100
        st.metric("Daily Energy Saved", f"{saved:.0f} kWh")
        st.metric("Daily CO2 Saved", f"{saved * carbon_factor:.1f} kg")
        st.metric("Monthly Cost Saved", f"INR {saved * LOAD_BALANCE_CONFIG['cost_per_kwh_peak'] * 30:,.0f}")
        st.metric("Annual CO2 Saved", f"{saved * carbon_factor * 365 / 1000:.1f} tonnes")
