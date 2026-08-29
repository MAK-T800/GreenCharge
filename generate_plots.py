"""
Generate publication-quality matplotlib plots for XGBoost and XGBoost+BiLSTM.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, PLOTS_DIR, METRICS_DIR

# ── Style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#000000',
    'axes.facecolor': '#000000',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#999999',
    'axes.grid': True,
    'grid.color': '#1a1a1a',
    'grid.linewidth': 0.5,
    'text.color': '#cccccc',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica', 'Arial'],
    'font.size': 11,
    'legend.facecolor': '#111111',
    'legend.edgecolor': '#333333',
    'legend.fontsize': 10,
    'savefig.facecolor': '#000000',
    'savefig.bbox': 'tight',
    'savefig.dpi': 180,
})

ACCENT = '#e63946'       # Verge-style red
WHITE = '#ffffff'
GRAY = '#888888'
LIGHT = '#cccccc'

# ── Load predictions ────────────────────────────────────────────────────
pred_path = os.path.join(MODELS_DIR, "all_predictions.json")
with open(pred_path, 'r') as f:
    preds = json.load(f)

os.makedirs(PLOTS_DIR, exist_ok=True)


def load_metrics(name):
    p = os.path.join(METRICS_DIR, f"{name}.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        return d.get("metrics", d)
    return {}


def plot_forecast(key, display_name, filename):
    """Actual vs Predicted + residual subplot."""
    data = preds[key]
    act = np.array(data["actuals"])
    pred = np.array(data["predictions"])
    n = min(len(act), len(pred))
    act, pred = act[:n], pred[:n]
    residual = act - pred
    x = np.arange(n)
    met = load_metrics(key)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6.5),
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
                                    sharex=True)

    # Top: Actual vs Predicted
    ax1.plot(x, act, color=GRAY, linewidth=0.9, alpha=0.8, label='Actual')
    ax1.plot(x, pred, color=ACCENT, linewidth=1.3, label='Predicted')
    ax1.fill_between(x, act, pred, alpha=0.06, color=ACCENT)
    ax1.set_ylabel('Energy (kWh)', fontsize=11, color='#999999')
    ax1.legend(loc='upper right', framealpha=0.8)
    ax1.set_title(display_name, fontsize=16, fontweight='bold',
                  color=WHITE, loc='left', pad=12)

    # Metrics annotation
    if met:
        txt = f"R\u00b2 = {met.get('R2', 0):.4f}    RMSE = {met.get('RMSE', 0):.2f}    MAE = {met.get('MAE', 0):.2f}    MAPE = {met.get('MAPE', 0):.2f}%"
        ax1.text(0.99, 0.95, txt, transform=ax1.transAxes, fontsize=9,
                 color='#666666', ha='right', va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111',
                           edgecolor='#222222', alpha=0.9))

    ax1.tick_params(axis='x', labelbottom=False)

    # Bottom: Residual
    ax2.bar(x, residual, color=ACCENT, alpha=0.35, width=1.0)
    ax2.axhline(0, color='#333333', linewidth=0.8)
    ax2.set_ylabel('Residual', fontsize=10, color='#999999')
    ax2.set_xlabel('Validation Sample Index', fontsize=10, color='#999999')

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    out = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_scatter(key, display_name, filename):
    """Actual vs Predicted scatter plot with perfect-prediction line."""
    data = preds[key]
    act = np.array(data["actuals"])
    pred = np.array(data["predictions"])
    n = min(len(act), len(pred))
    act, pred = act[:n], pred[:n]
    met = load_metrics(key)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(act, pred, s=8, alpha=0.4, color=ACCENT, edgecolors='none')
    lims = [min(act.min(), pred.min()), max(act.max(), pred.max())]
    ax.plot(lims, lims, '--', color='#444444', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('Actual (kWh)', fontsize=12)
    ax.set_ylabel('Predicted (kWh)', fontsize=12)
    ax.set_title(f'{display_name} \u2014 Scatter', fontsize=15, fontweight='bold',
                 color=WHITE, loc='left', pad=12)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', framealpha=0.8)

    if met:
        txt = f"R\u00b2 = {met.get('R2', 0):.4f}"
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=12,
                color=ACCENT, ha='right', va='bottom', fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    out = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_error_distribution(key, display_name, filename):
    """Error histogram."""
    data = preds[key]
    act = np.array(data["actuals"])
    pred = np.array(data["predictions"])
    n = min(len(act), len(pred))
    errors = act[:n] - pred[:n]

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.hist(errors, bins=60, color=ACCENT, alpha=0.6, edgecolor='#000000', linewidth=0.3)
    ax.axvline(0, color='#444444', linewidth=1, linestyle='--')
    ax.axvline(errors.mean(), color=WHITE, linewidth=1, linestyle='-', alpha=0.7,
               label=f'Mean = {errors.mean():.2f}')
    ax.set_xlabel('Prediction Error (kWh)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'{display_name} \u2014 Error Distribution', fontsize=15, fontweight='bold',
                 color=WHITE, loc='left', pad=12)
    ax.legend(loc='upper right', framealpha=0.8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    out = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_comparison_bar():
    """Side-by-side metrics comparison of XGBoost vs XGBoost+BiLSTM."""
    m1 = load_metrics("xgboost")
    m2 = load_metrics("xgb_bilstm")
    if not m1 or not m2:
        print("  Skipping comparison (metrics missing)")
        return

    labels = ['R\u00b2', 'RMSE', 'MAE', 'MAPE (%)']
    v1 = [m1['R2'], m1['RMSE'], m1['MAE'], m1['MAPE']]
    v2 = [m2['R2'], m2['RMSE'], m2['MAE'], m2['MAPE']]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, label, a, b in zip(axes, labels, v1, v2):
        bars = ax.bar(['XGBoost', 'XGB+BiLSTM'], [a, b],
                       color=[GRAY, ACCENT], width=0.55, edgecolor='none')
        ax.set_title(label, fontsize=13, fontweight='bold', color=WHITE, pad=10)
        for bar, val in zip(bars, [a, b]):
            fmt = f'{val:.4f}' if label == 'R\u00b2' else f'{val:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(a, b),
                    fmt, ha='center', va='bottom', fontsize=10, color=LIGHT)
        ax.set_ylim(0, max(a, b) * 1.18)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='y', left=False, labelleft=False)

    fig.suptitle('XGBoost vs XGBoost + BiLSTM', fontsize=17, fontweight='bold',
                 color=WHITE, y=1.02)

    out = os.path.join(PLOTS_DIR, "comparison_xgb_vs_bilstm.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Generate all plots ──────────────────────────────────────────────────
print("Generating matplotlib plots...")
print()

for key, name in [("xgboost", "XGBoost"), ("xgb_bilstm", "XGBoost + BiLSTM")]:
    if key in preds:
        plot_forecast(key, name, f"{key}_forecast.png")
        plot_scatter(key, name, f"{key}_scatter.png")
        plot_error_distribution(key, name, f"{key}_errors.png")
    else:
        print(f"  {key} not found in predictions, skipping")

plot_comparison_bar()

print()
print("All plots generated.")
