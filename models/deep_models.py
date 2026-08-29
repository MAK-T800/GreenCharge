"""
GreenCharge Deep Learning Models
================================
ANN (Tabular) and CNN (1D Sequence, 168-hour lookback) for load forecasting.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ANN_CONFIG, CNN_CONFIG,
    MODELS_DIR, METRICS_DIR, TARGET_COL, LOOKBACK_HOURS
)
from models.evaluation import compute_metrics, print_metrics, save_metrics

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks, Model


def get_tabular_features(df):
    """Get numeric feature columns suitable for deep models."""
    exclude = [
        TARGET_COL, "meter_reading_log", "timestamp",
        "building_id", "meter", "site_id", "row_id", "primary_use"
    ]
    return [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]


def prepare_sequence_data(train_df, val_df, lookback=LOOKBACK_HOURS):
    """
    Prepare per-building sequence data with 168-hour lookback window.
    Aligns validation sequences exactly with val_df rows by prepending
    the trailing lookback window from train_df for each building.
    Returns val_indices so predictions can be reordered back to val_df.index.
    """
    feature_cols = get_tabular_features(train_df)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(train_df[feature_cols].values)
    scaler_y.fit(train_df[[TARGET_COL]].values)

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    val_indices_list = []

    for bid in train_df["building_id"].unique():
        b_train = train_df[train_df["building_id"] == bid].sort_values("timestamp")
        b_val = val_df[val_df["building_id"] == bid].sort_values("timestamp")

        if len(b_train) <= lookback:
            continue

        X_tr_scaled = scaler_X.transform(b_train[feature_cols].values)
        y_tr_scaled = scaler_y.transform(b_train[[TARGET_COL]].values).flatten()

        for i in range(lookback, len(X_tr_scaled)):
            X_train_list.append(X_tr_scaled[i - lookback : i])
            y_train_list.append(y_tr_scaled[i])

        if len(b_val) > 0:
            b_combined = pd.concat([b_train.iloc[-lookback:], b_val])
            X_comb_scaled = scaler_X.transform(b_combined[feature_cols].values)
            y_comb_scaled = scaler_y.transform(b_combined[[TARGET_COL]].values).flatten()

            for i in range(lookback, len(X_comb_scaled)):
                X_val_list.append(X_comb_scaled[i - lookback : i])
                y_val_list.append(y_comb_scaled[i])
                val_indices_list.append(b_combined.index[i])

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_val = np.array(X_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.float32)
    val_indices = np.array(val_indices_list)

    return X_train, y_train, X_val, y_val, scaler_X, scaler_y, feature_cols, val_indices


def get_callbacks(model_name, patience=7):
    """Standard callbacks for deep learning models."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0
        ),
    ]


# =========================================================================
# Model: ANN (Artificial Neural Network on Tabular Features)
# =========================================================================
def build_ann(input_dim, config=None):
    """Build ANN model for tabular features."""
    if config is None:
        config = ANN_CONFIG

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in config["hidden_layers"]:
        model.add(layers.Dense(units, activation='relu'))
        if config.get("batch_norm"):
            model.add(layers.BatchNormalization())
        model.add(layers.Dropout(config["dropout"]))

    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]),
                  loss='mse', metrics=['mae'])
    return model


def train_ann(train_df, val_df, config=None):
    """Train ANN model on tabular features."""
    if config is None:
        config = ANN_CONFIG

    print("\n[ANN] Training Model: ANN...")

    feature_cols = get_tabular_features(train_df)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(train_df[feature_cols].values)
    y_train = scaler_y.fit_transform(train_df[[TARGET_COL]].values).flatten()

    X_val = scaler_X.transform(val_df[feature_cols].values)
    y_val = scaler_y.transform(val_df[[TARGET_COL]].values).flatten()
    actuals = val_df[TARGET_COL].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    model = build_ann(X_train.shape[1], config)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=get_callbacks("ann", config["patience"]),
        verbose=1,
    )

    # Predict and inverse transform
    predictions_scaled = model.predict(X_val, verbose=0).flatten()
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    predictions = np.maximum(predictions, 0)

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(actuals, predictions)
    print_metrics("ANN", metrics)
    save_metrics("ann", metrics, METRICS_DIR)

    # Save model
    model_path = os.path.join(MODELS_DIR, "ann.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    return {
        "model": model,
        "predictions": predictions,
        "actuals": actuals,
        "metrics": metrics,
        "history": history.history,
        "scalers": (scaler_X, scaler_y),
        "feature_cols": feature_cols,
    }


# =========================================================================
# Model: CNN (1D Convolutional Neural Network, 168h window)
# =========================================================================
def build_cnn(input_shape, config=None):
    """Build 1D CNN model for time series forecasting."""
    if config is None:
        config = CNN_CONFIG

    model = keras.Sequential([
        layers.Conv1D(config["filters_1"], config["kernel_size"],
                      activation='relu', padding='same',
                      input_shape=input_shape),
        layers.Conv1D(config["filters_2"], config["kernel_size"],
                      activation='relu', padding='same'),
        layers.MaxPooling1D(config["pool_size"]),
        layers.Flatten(),
        layers.Dense(config["dense_units"], activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]),
                  loss='mse', metrics=['mae'])
    return model


def train_cnn(train_df, val_df, config=None):
    """Train 1D CNN model with 168-hour lookback window."""
    if config is None:
        config = CNN_CONFIG

    print(f"\n[CNN] Training Model: CNN ({LOOKBACK_HOURS}h window)...")

    X_train, y_train, X_val, y_val, scaler_X, scaler_y, feature_cols, val_indices = \
        prepare_sequence_data(train_df, val_df, lookback=LOOKBACK_HOURS)

    print(f"  Sequence shapes - X_train: {X_train.shape}, X_val: {X_val.shape}")

    model = build_cnn(X_train.shape[1:], config)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=get_callbacks("cnn", config["patience"]),
        verbose=1,
    )

    # Predict and inverse transform
    predictions_scaled = model.predict(X_val, verbose=0).flatten()
    pred_raw = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    pred_raw = np.maximum(pred_raw, 0)

    # Reorder predictions back to val_df row order
    predictions = np.zeros(len(val_df), dtype=np.float32)
    predictions[val_indices] = pred_raw
    actuals = val_df[TARGET_COL].values

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(actuals, predictions)
    print_metrics("CNN", metrics)
    save_metrics("cnn", metrics, METRICS_DIR)

    # Save model
    model_path = os.path.join(MODELS_DIR, "cnn.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    return {
        "model": model,
        "predictions": predictions,
        "actuals": actuals,
        "metrics": metrics,
        "history": history.history,
        "scalers": (scaler_X, scaler_y),
        "feature_cols": feature_cols,
    }


def train_all_deep_models(train_df, val_df):
    """Train deep learning models (ANN, CNN)."""
    results = {}
    results["ann"] = train_ann(train_df, val_df)
    results["cnn"] = train_cnn(train_df, val_df)
    return results
