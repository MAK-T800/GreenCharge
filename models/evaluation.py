"""
GreenCharge Evaluation Module
=============================
Shared evaluation utility computing R², RMSE, MAE, and MAPE for every model.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    """
    Compute all 4 evaluation metrics.
    
    Returns:
        dict with keys: R2, RMSE, MAE, MAPE
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Handle edge cases
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE: avoid division by zero
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
    else:
        mape = float('inf')
    
    return {
        "R2": round(float(r2), 6),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE": round(float(mape), 4),
    }


def print_metrics(model_name, metrics):
    """Print formatted metrics to console."""
    print(f"\n{'='*50}")
    print(f"  Model: {model_name}")
    print(f"{'='*50}")
    print(f"  R²   : {metrics['R2']:.6f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  MAE  : {metrics['MAE']:.4f}")
    print(f"  MAPE : {metrics['MAPE']:.4f}%")
    print(f"{'='*50}\n")


def save_metrics(model_name, metrics, metrics_dir):
    """Save metrics as JSON file."""
    os.makedirs(metrics_dir, exist_ok=True)
    filepath = os.path.join(metrics_dir, f"{model_name}.json")
    
    data = {
        "model_name": model_name,
        "metrics": metrics,
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Metrics saved to: {filepath}")
    return filepath


def load_metrics(metrics_dir):
    """Load all saved metrics from directory."""
    all_metrics = {}
    if not os.path.exists(metrics_dir):
        return all_metrics
        
    for fname in os.listdir(metrics_dir):
        if fname.endswith('.json') and fname != 'comparison_report.json':
            filepath = os.path.join(metrics_dir, fname)
            with open(filepath, 'r') as f:
                data = json.load(f)
            model_name = data.get("model_name", fname.replace('.json', ''))
            all_metrics[model_name] = data.get("metrics", data)
    
    return all_metrics


def compare_all_models(metrics_dir):
    """
    Load all model metrics and create a comparison DataFrame.
    Sorted by R² descending.
    """
    all_metrics = load_metrics(metrics_dir)
    
    if not all_metrics:
        print("No metrics found!")
        return pd.DataFrame()
    
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed rank
    df.index.name = "Rank"
    
    return df


def generate_comparison_report(metrics_dir):
    """Generate and save a comprehensive comparison report."""
    df = compare_all_models(metrics_dir)
    
    if df.empty:
        print("No models to compare!")
        return df
    
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON LEADERBOARD")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)
    
    # Best model per metric
    print(f"\n  Best R²   : {df.loc[df['R2'].idxmax(), 'Model']} ({df['R2'].max():.6f})")
    print(f"  Best RMSE : {df.loc[df['RMSE'].idxmin(), 'Model']} ({df['RMSE'].min():.4f})")
    print(f"  Best MAE  : {df.loc[df['MAE'].idxmin(), 'Model']} ({df['MAE'].min():.4f})")
    print(f"  Best MAPE : {df.loc[df['MAPE'].idxmin(), 'Model']} ({df['MAPE'].min():.4f}%)")
    
    # Save comparison report
    report = {
        "leaderboard": df.to_dict(orient="records"),
        "best_by_metric": {
            "R2": {"model": df.loc[df['R2'].idxmax(), 'Model'], "value": float(df['R2'].max())},
            "RMSE": {"model": df.loc[df['RMSE'].idxmin(), 'Model'], "value": float(df['RMSE'].min())},
            "MAE": {"model": df.loc[df['MAE'].idxmin(), 'Model'], "value": float(df['MAE'].min())},
            "MAPE": {"model": df.loc[df['MAPE'].idxmin(), 'Model'], "value": float(df['MAPE'].min())},
        }
    }
    
    report_path = os.path.join(metrics_dir, "comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {report_path}")
    
    return df
