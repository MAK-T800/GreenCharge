"""
GreenCharge Tree-Based Models (Models 2-4)
==========================================
XGBoost, LightGBM, and Random Forest for tabular load forecasting.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    XGBOOST_CONFIG, RANDOM_FOREST_CONFIG,
    MODELS_DIR, METRICS_DIR, TARGET_COL
)
from models.evaluation import compute_metrics, print_metrics, save_metrics

warnings.filterwarnings("ignore")


def get_tabular_features(df):
    """Get feature columns suitable for tree models."""
    exclude = [
        TARGET_COL, "meter_reading_log", "timestamp",
        "building_id", "meter", "site_id", "row_id", "primary_use"
    ]
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]


# =========================================================================
# Model 2: XGBoost
# =========================================================================
def train_xgboost(train_df, val_df, config=None):
    """
    Train XGBoost regressor.
    
    Returns:
        dict with model, predictions, metrics, feature_importance
    """
    if config is None:
        config = XGBOOST_CONFIG.copy()

    print("\n[XGBoost] Training Model 2: XGBoost...")

    feature_cols = get_tabular_features(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    early_stopping = config.pop("early_stopping_rounds", 50)
    
    model = xgb.XGBRegressor(
        **config,
        objective="reg:squarederror",
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Predict
    predictions = model.predict(X_val)
    predictions = np.maximum(predictions, 0)

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(y_val, predictions)
    print_metrics("XGBoost", metrics)
    save_metrics("xgboost", metrics, METRICS_DIR)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgboost.json")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    print(f"  Model saved to: {model_path}")
    
    # Save feature importance
    imp_path = os.path.join(MODELS_DIR, "xgboost_feature_importance.json")
    with open(imp_path, 'w') as f:
        json.dump({k: float(v) for k, v in list(importance.items())[:20]}, f, indent=2)

    config["early_stopping_rounds"] = early_stopping  # restore
    
    return {
        "model": model,
        "predictions": predictions,
        "actuals": y_val,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": feature_cols,
    }


# =========================================================================
# Model 3: LightGBM
# =========================================================================
def train_lightgbm(train_df, val_df, config=None):
    """
    Train LightGBM regressor.
    
    Returns:
        dict with model, predictions, actuals, metrics, feature_importance
    """
    if config is None:
        config = LIGHTGBM_CONFIG.copy()

    print("\n[LightGBM] Training Model 3: LightGBM...")

    feature_cols = get_tabular_features(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    early_stopping = config.pop("early_stopping_rounds", 50)
    
    model = lgb.LGBMRegressor(**config)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)]
    )

    # Predict
    predictions = model.predict(X_val)
    predictions = np.maximum(predictions, 0)

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(y_val, predictions)
    print_metrics("LightGBM", metrics)
    save_metrics("lightgbm", metrics, METRICS_DIR)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Save model
    model_path = os.path.join(MODELS_DIR, "lightgbm.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {model_path}")

    config["early_stopping_rounds"] = early_stopping  # restore
    
    return {
        "model": model,
        "predictions": predictions,
        "actuals": y_val,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": feature_cols,
    }


# =========================================================================
# Model 4: Random Forest
# =========================================================================
def train_random_forest(train_df, val_df, config=None):
    """
    Train Random Forest regressor.
    
    Returns:
        dict with model, predictions, actuals, metrics, feature_importance
    """
    if config is None:
        config = RANDOM_FOREST_CONFIG.copy()

    print("\n[Random Forest] Training Model 4: Random Forest...")

    feature_cols = get_tabular_features(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    model = RandomForestRegressor(**config)
    model.fit(X_train, y_train)

    if config.get("oob_score"):
        print(f"  OOB Score: {model.oob_score_:.6f}")

    # Predict
    predictions = model.predict(X_val)
    predictions = np.maximum(predictions, 0)

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(y_val, predictions)
    print_metrics("Random Forest", metrics)
    save_metrics("random_forest", metrics, METRICS_DIR)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # Save model
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to: {model_path}")

    return {
        "model": model,
        "predictions": predictions,
        "actuals": y_val,
        "metrics": metrics,
        "feature_importance": importance,
        "feature_cols": feature_cols,
    }


def train_all_tree_models(train_df, val_df):
    """Train XGBoost and Random Forest models."""
    results = {}
    results["xgboost"] = train_xgboost(train_df, val_df)
    results["random_forest"] = train_random_forest(train_df, val_df)
    return results

