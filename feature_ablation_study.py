"""
GreenCharge Feature Ablation Study
====================================
Compares R² (and other metrics) when specific feature groups are removed.

Ablation scenarios:
  1. Baseline         — All features
  2. No Lag Features  — Remove lag_1h … lag_168h
  3. No Weather       — Remove air_temperature, cloud_coverage, dew_temperature,
                         precip_depth_1_hr, sea_level_pressure, wind_direction, wind_speed
  4. No Cyclical      — Replace sin/cos pairs with raw hour / day_of_week integers
  5. No Rolling Stats — Remove rolling_mean_* and rolling_std_*
  6. No Building Meta — Remove square_feet, year_built, floor_count
  7. No Calendar Flags— Remove is_weekend, is_holiday, semester_period

Results are saved to:
  - models/saved/metrics/feature_ablation_results.json
  - plots/feature_ablation_r2.png
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_TRAIN, PROCESSED_VAL,
    MODELS_DIR, METRICS_DIR, PLOTS_DIR, TARGET_COL,
    XGBOOST_CONFIG,
)
from models.evaluation import compute_metrics

warnings.filterwarnings("ignore")

# ── Feature-group definitions ───────────────────────────────────────────
LAG_FEATURES = [
    "lag_1h", "lag_2h", "lag_3h", "lag_6h",
    "lag_12h", "lag_24h", "lag_48h", "lag_168h",
]

WEATHER_FEATURES = [
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
]

CYCLICAL_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]

ROLLING_FEATURES = [
    "rolling_mean_6h", "rolling_std_6h",
    "rolling_mean_12h", "rolling_std_12h",
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_48h", "rolling_std_48h",
    "rolling_mean_168h", "rolling_std_168h",
]

BUILDING_META_FEATURES = [
    "square_feet", "year_built", "floor_count",
]

CALENDAR_FEATURES = [
    "is_weekend", "is_holiday", "semester_period",
]

# Columns to always exclude (non-feature columns)
EXCLUDE_COLS = {
    TARGET_COL, "meter_reading_log", "timestamp",
    "building_id", "meter", "site_id", "row_id", "primary_use",
}


# ── Helper: get usable feature columns from a dataframe ─────────────────
def _get_feature_cols(df):
    """Return numeric feature columns, excluding identifiers & target."""
    return [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in ("float64", "float32", "int64", "int32")
    ]


# ── Train a lightweight XGBoost and return metrics ──────────────────────
def _train_and_evaluate(X_train, y_train, X_val, y_val):
    """Train XGBoost on given arrays and return metrics dict."""
    import xgboost as xgb

    cfg = XGBOOST_CONFIG.copy()
    cfg.pop("early_stopping_rounds", None)
    # Use fewer estimators for speed during ablation
    cfg["n_estimators"] = min(cfg.get("n_estimators", 500), 500)

    model = xgb.XGBRegressor(
        **cfg,
        objective="reg:squarederror",
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = np.maximum(model.predict(X_val), 0)
    return compute_metrics(y_val, preds)


# ── Main ablation runner ────────────────────────────────────────────────
def run_ablation():
    print("=" * 70)
    print("  GreenCharge Feature Ablation Study")
    print("=" * 70)

    # Load processed data
    print("\n[Ablation] Loading processed train / val data ...")
    train_df = pd.read_parquet(PROCESSED_TRAIN)
    val_df   = pd.read_parquet(PROCESSED_VAL)
    print(f"  Train: {train_df.shape}  |  Val: {val_df.shape}")

    all_feature_cols = _get_feature_cols(train_df)
    print(f"  Total feature columns available: {len(all_feature_cols)}")

    y_train = train_df[TARGET_COL].values
    y_val   = val_df[TARGET_COL].values

    # Define ablation scenarios:  label → columns to DROP
    scenarios = {
        "Baseline (All Features)": [],
        "Without Lag Features": LAG_FEATURES,
        "Without Weather Features": WEATHER_FEATURES,
        "Without Cyclical Encodings": CYCLICAL_FEATURES,
        "Without Rolling Statistics": ROLLING_FEATURES,
        "Without Building Metadata": BUILDING_META_FEATURES,
        "Without Calendar Flags": CALENDAR_FEATURES,
    }

    results = {}
    start = time.time()

    for label, drop_cols in scenarios.items():
        print(f"\n{'-' * 60}")
        print(f"  Scenario: {label}")

        # Determine which columns to keep
        keep_cols = [c for c in all_feature_cols if c not in drop_cols]

        # Safety: also ensure the dropped columns actually existed
        actually_dropped = [c for c in drop_cols if c in all_feature_cols]
        print(f"  Features dropped : {len(actually_dropped)}  ->  {actually_dropped}")
        print(f"  Features kept    : {len(keep_cols)}")

        X_tr = train_df[keep_cols].values
        X_vl = val_df[keep_cols].values

        metrics = _train_and_evaluate(X_tr, y_train, X_vl, y_val)

        results[label] = {
            "metrics": metrics,
            "features_dropped": actually_dropped,
            "num_features_used": len(keep_cols),
        }

        print(f"  R²   = {metrics['R2']:.6f}")
        print(f"  RMSE = {metrics['RMSE']:.4f}")
        print(f"  MAE  = {metrics['MAE']:.4f}")
        print(f"  MAPE = {metrics['MAPE']:.4f}%")

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  All scenarios completed in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # ── Summary table ───────────────────────────────────────────────────
    baseline_r2 = results["Baseline (All Features)"]["metrics"]["R2"]

    print(f"\n{'-' * 90}")
    print(f"  {'Scenario':<35} {'R2':>10} {'Delta_R2':>10} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    print(f"{'-' * 90}")
    for label, info in results.items():
        m = info["metrics"]
        delta = m["R2"] - baseline_r2
        marker = " *" if label.startswith("Baseline") else ""
        print(
            f"  {label:<35} {m['R2']:>10.6f} {delta:>+10.6f} "
            f"{m['RMSE']:>10.4f} {m['MAE']:>10.4f} {m['MAPE']:>9.4f}%{marker}"
        )
    print(f"{'-' * 90}")

    # ── Save JSON results ───────────────────────────────────────────────
    out_path = os.path.join(METRICS_DIR, "feature_ablation_results.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # ── Plot ────────────────────────────────────────────────────────────
    _plot_results(results, baseline_r2)

    return results


# ── Plotting ────────────────────────────────────────────────────────────
def _plot_results(results, baseline_r2):
    """Create a white-and-blue bar chart of R2 and Delta-R2 for each ablation scenario."""
    labels = list(results.keys())
    r2_vals = [results[l]["metrics"]["R2"] for l in labels]
    delta_vals = [r - baseline_r2 for r in r2_vals]

    # Short labels for the plot
    short = [l.replace("Without ", "No ").replace("Baseline (All Features)", "Baseline") for l in labels]

    # Blue palette
    BLUE_DARK  = "#0D47A1"
    BLUE_MID   = "#1565C0"
    BLUE_LIGHT = "#64B5F6"
    BLUE_PALE  = "#BBDEFB"
    TEXT_COLOR = "#222222"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 2]})

    # --- Left: R2 bars ---
    ax1 = axes[0]
    colors = []
    for d in delta_vals:
        if d == 0:
            colors.append(BLUE_DARK)    # baseline
        elif d < -0.005:
            colors.append(BLUE_PALE)    # big drop
        elif d < 0:
            colors.append(BLUE_LIGHT)   # small drop
        else:
            colors.append(BLUE_MID)     # improvement
    bars = ax1.barh(short, r2_vals, color=colors, edgecolor="white", linewidth=0.9)
    ax1.set_xlabel("R2 Score", fontsize=12, color=TEXT_COLOR)
    ax1.set_title("R2 by Feature Group Ablation", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax1.axvline(baseline_r2, color=BLUE_DARK, linestyle="--", linewidth=1, alpha=0.5, label="Baseline R2")
    ax1.legend(loc="lower right", frameon=True, edgecolor="#cccccc", facecolor="white")
    for bar, val in zip(bars, r2_vals):
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=10, color=TEXT_COLOR)
    ax1.invert_yaxis()
    ax1.set_xlim(left=min(r2_vals) - 0.05, right=max(r2_vals) + 0.06)

    # --- Right: Delta-R2 bars ---
    ax2 = axes[1]
    d_colors = [BLUE_PALE if d < -0.005 else (BLUE_LIGHT if d < 0 else BLUE_MID) for d in delta_vals]
    bars2 = ax2.barh(short, delta_vals, color=d_colors, edgecolor="white", linewidth=0.9)
    ax2.set_xlabel("Delta R2 (vs Baseline)", fontsize=12, color=TEXT_COLOR)
    ax2.set_title("Impact of Removing Feature Group", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax2.axvline(0, color="#999999", linewidth=1, alpha=0.6)
    for bar, val in zip(bars2, delta_vals):
        offset = 0.0005 if val >= 0 else -0.0005
        ha = "left" if val >= 0 else "right"
        ax2.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                 f"{val:+.4f}", va="center", ha=ha, fontsize=10, color=TEXT_COLOR)
    ax2.invert_yaxis()

    # White + blue styling
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="#555555")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(0.6)
        ax.tick_params(axis="y", labelsize=10, colors="#333333")
        ax.tick_params(axis="x", labelsize=10, colors="#555555")
        ax.grid(axis="x", color="#e0e0e0", linestyle="-", linewidth=0.4)

    fig.patch.set_facecolor("white")
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "feature_ablation_r2.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Plot saved to : {plot_path}")


# ── Entry point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ablation()
