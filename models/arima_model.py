"""
GreenCharge ARIMA Model (Model 1)
==================================
SARIMA model for individual hostel building load forecasting.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARIMA_CONFIG, MODELS_DIR, METRICS_DIR, FORECAST_HOURS, TARGET_COL
from models.evaluation import compute_metrics, print_metrics, save_metrics

warnings.filterwarnings("ignore")


def find_best_params(series, config=None):
    """
    Find best SARIMA parameters via grid search on AIC.
    Uses a reduced search space for efficiency.
    """
    if config is None:
        config = ARIMA_CONFIG

    # Reduced search for speed
    p_range = range(0, min(config["max_p"] + 1, 3))
    d_range = range(0, min(config["max_d"] + 1, 2))
    q_range = range(0, min(config["max_q"] + 1, 3))
    
    seasonal_period = config["seasonal_period"]
    
    best_aic = float("inf")
    best_params = (1, 1, 1)
    best_seasonal = (1, 0, 1, seasonal_period)
    
    # Non-seasonal parameters
    for p, d, q in product(p_range, d_range, q_range):
        if p == 0 and q == 0:
            continue
        try:
            model = SARIMAX(
                series,
                order=(p, d, q),
                seasonal_order=(1, 0, 1, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False, maxiter=100)
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = (p, d, q)
        except Exception:
            continue

    print(f"  Best ARIMA params: {best_params}, AIC: {best_aic:.2f}")
    return best_params, best_seasonal


def train_sarima(train_series, val_series=None, config=None, building_id=None):
    """
    Train SARIMA model on a single building's time series.
    
    Args:
        train_series: pd.Series with DatetimeIndex
        val_series: pd.Series for evaluation
        config: ARIMA hyperparameters
        building_id: building identifier for logging
    
    Returns:
        dict with model, predictions, metrics
    """
    if config is None:
        config = ARIMA_CONFIG

    label = f"Building {building_id}" if building_id else "SARIMA"
    print(f"\n[SARIMA] Training for {label}...")
    print(f"  Train size: {len(train_series)}, Val size: {len(val_series) if val_series is not None else 0}")

    # Resample to hourly if needed and fill gaps
    train_series = train_series.resample("h").mean().ffill().bfill()
    
    # Limit series length for speed (last 30 days for training)
    max_train = 24 * 30
    if len(train_series) > max_train:
        train_series = train_series[-max_train:]

    # Find best params (or use defaults for speed)
    order = (2, 1, 1)
    seasonal_order = (1, 0, 1, config["seasonal_period"])
    
    # Fit model
    print(f"  Fitting SARIMA{order}x{seasonal_order}...")
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False, maxiter=200)
    print(f"  AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")

    # Forecast
    metrics = None
    predictions = None
    
    if val_series is not None:
        val_series = val_series.resample("h").mean().ffill().bfill()
        n_forecast = min(len(val_series), FORECAST_HOURS * 7)  # Up to 7 days
        
        forecast = results.forecast(steps=n_forecast)
        predictions = forecast.values
        actuals = val_series.iloc[:n_forecast].values
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        # Compute R², RMSE, MAE, MAPE
        metrics = compute_metrics(actuals, predictions)
        print_metrics("SARIMA", metrics)
        save_metrics("sarima", metrics, METRICS_DIR)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "sarima.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  Model saved to: {model_path}")

    return {
        "model": results,
        "predictions": predictions,
        "metrics": metrics,
        "order": order,
        "seasonal_order": seasonal_order,
    }


def predict_sarima(model_results, steps=24):
    """Generate forecast from a fitted SARIMA model."""
    forecast = model_results.forecast(steps=steps)
    return np.maximum(forecast.values, 0)


def train_from_dataframe(train_df, val_df, building_id=None):
    """
    Convenience function to train SARIMA from DataFrames.
    Picks a single building if building_id specified.
    """
    if building_id is None:
        building_id = train_df["building_id"].unique()[0]
    
    # Filter for specific building
    train_bldg = train_df[train_df["building_id"] == building_id].copy()
    val_bldg = val_df[val_df["building_id"] == building_id].copy()
    
    if len(train_bldg) == 0:
        print(f"  No data for building {building_id}")
        return None
    
    # Create time series
    train_series = train_bldg.set_index("timestamp")[TARGET_COL]
    val_series = val_bldg.set_index("timestamp")[TARGET_COL] if len(val_bldg) > 0 else None
    
    return train_sarima(train_series, val_series, building_id=building_id)
