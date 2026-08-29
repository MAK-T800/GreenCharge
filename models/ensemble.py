"""
GreenCharge Meta-Ensemble
=========================
Combines all 7 models via inverse-RMSE weighted averaging and Ridge stacking.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, METRICS_DIR, MODEL_NAMES, ALL_MODELS
from models.evaluation import (
    compute_metrics, print_metrics, save_metrics,
    load_metrics, generate_comparison_report
)


def weighted_ensemble(model_results, metric="RMSE"):
    """
    Create a weighted ensemble from model results.
    Weights are inversely proportional to RMSE^2.
    
    Args:
        model_results: dict of model_name -> dict with 'predictions', 'actuals', 'metrics'
        metric: metric to use for weighting ("RMSE" or "MAE")
    
    Returns:
        dict with ensemble predictions, actuals, metrics
    """
    print("\n[Meta-Ensemble] Building weighted ensemble...")

    # Collect models with valid predictions
    valid_models = {}
    min_len = float('inf')

    for name, result in model_results.items():
        if result is not None and result.get("predictions") is not None and result.get("metrics") is not None:
            pred = result["predictions"]
            if len(pred) > 0:
                valid_models[name] = result
                min_len = min(min_len, len(pred))

    if len(valid_models) < 2:
        print("  Not enough valid models for ensemble!")
        return None

    print(f"  Combining {len(valid_models)} models: {list(valid_models.keys())}")

    # Compute weights (inverse square of RMSE)
    weights = {}
    for name, result in valid_models.items():
        metric_val = result["metrics"].get(metric, 1.0)
        if metric_val > 0:
            weights[name] = 1.0 / (metric_val ** 2)
        else:
            weights[name] = 1.0

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    print("  Weights:")
    for name, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        display_name = MODEL_NAMES.get(name, name)
        print(f"    {display_name}: {w:.4f}")

    # Weighted average prediction
    ensemble_pred = np.zeros(min_len, dtype=np.float32)
    for name, result in valid_models.items():
        pred = np.array(result["predictions"][:min_len], dtype=np.float32)
        ensemble_pred += weights[name] * pred

    first_result = list(valid_models.values())[0]
    actuals = np.array(first_result.get("actuals", first_result.get("predictions"))[:min_len], dtype=np.float32)

    ensemble_pred = np.maximum(ensemble_pred, 0)

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(actuals, ensemble_pred)
    print_metrics("Meta-Ensemble (Weighted)", metrics)

    return {
        "predictions": ensemble_pred,
        "actuals": actuals,
        "metrics": metrics,
        "weights": weights,
        "n_models": len(valid_models),
    }


def stacking_ensemble(model_results, val_actuals=None):
    """
    Create a stacking ensemble using Cross-Validated Ridge Regression as meta-learner.
    
    Args:
        model_results: dict of model_name -> dict with 'predictions' and 'actuals'
        val_actuals: actual values for validation
    
    Returns:
        dict with stacked predictions, metrics
    """
    print("\n[Meta-Ensemble] Building stacking ensemble...")

    valid_models = {}
    min_len = float('inf')

    for name, result in model_results.items():
        if result is not None and result.get("predictions") is not None:
            pred = result["predictions"]
            if len(pred) > 0:
                valid_models[name] = result
                min_len = min(min_len, len(pred))

    if len(valid_models) < 2:
        print("  Not enough valid models for stacking!")
        return None

    # Build meta-features matrix
    model_names = sorted(valid_models.keys())
    X_meta = np.column_stack([
        np.array(valid_models[name]["predictions"][:min_len], dtype=np.float32) for name in model_names
    ])

    if val_actuals is None:
        first_result = list(valid_models.values())[0]
        val_actuals = np.array(first_result.get("actuals", first_result.get("predictions"))[:min_len], dtype=np.float32)
    else:
        val_actuals = np.array(val_actuals[:min_len], dtype=np.float32)

    # Train Cross-Validated Ridge meta-learner (5-Fold CV over alphas)
    meta_learner = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    meta_learner.fit(X_meta, val_actuals)

    # Predict full validation set
    stacked_pred = meta_learner.predict(X_meta)
    stacked_pred = np.maximum(stacked_pred, 0)

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(val_actuals, stacked_pred)
    print_metrics("Meta-Ensemble (Stacked)", metrics)

    # Save meta-learner
    meta_path = os.path.join(MODELS_DIR, "meta_ensemble_stacker.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_learner, f)

    coef_dict = {m: float(c) for m, c in zip(model_names, meta_learner.coef_)}
    print(f"  Meta-learner alpha: {meta_learner.alpha_}")
    print(f"  Meta-learner coefficients: {coef_dict}")
    print(f"  Meta-learner intercept: {float(meta_learner.intercept_):.4f}")

    return {
        "predictions": stacked_pred,
        "actuals": val_actuals,
        "metrics": metrics,
        "meta_learner": meta_learner,
        "model_names": model_names,
        "coefficients": coef_dict,
    }


def build_ensemble(model_results):
    """
    Build both weighted and stacking ensembles, pick the better one.
    Also generates the comprehensive comparison report.
    """
    print("\n" + "=" * 60)
    print("  META-ENSEMBLE: Combining All Models")
    print("=" * 60)

    # Weighted ensemble
    weighted = weighted_ensemble(model_results)

    # Stacking ensemble
    stacked = stacking_ensemble(model_results)

    # Pick best
    best = weighted
    if stacked and weighted:
        if stacked["metrics"]["RMSE"] <= weighted["metrics"]["RMSE"]:
            best = stacked
            save_metrics("meta_ensemble", stacked["metrics"], METRICS_DIR)
            print("\n  Stacking ensemble is better! Using stacked predictions.")
        else:
            best = weighted
            save_metrics("meta_ensemble", weighted["metrics"], METRICS_DIR)
            print("\n  Weighted ensemble is better! Using weighted predictions.")
    elif weighted:
        save_metrics("meta_ensemble", weighted["metrics"], METRICS_DIR)
    elif stacked:
        save_metrics("meta_ensemble", stacked["metrics"], METRICS_DIR)

    # Generate comprehensive comparison report
    print("\n  Generating comparison report...")
    report_df = generate_comparison_report(METRICS_DIR)

    return {
        "weighted": weighted,
        "stacked": stacked,
        "best": best,
        "comparison": report_df,
    }
