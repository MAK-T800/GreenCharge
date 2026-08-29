"""
GreenCharge Hybrid Models
=========================
CNN+BiLSTM (168h window), XGBoost+BiLSTM (168h residual correction), and ANN+ANN (Dual Branch).
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CNN_BILSTM_CONFIG, XGB_BILSTM_CONFIG, DUAL_ANN_CONFIG,
    XGBOOST_CONFIG, MODELS_DIR, METRICS_DIR, TARGET_COL,
    LOOKBACK_HOURS
)
from models.evaluation import compute_metrics, print_metrics, save_metrics
from models.deep_models import (
    prepare_sequence_data, get_callbacks, get_tabular_features
)

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


# =========================================================================
# Model: CNN + BiLSTM (168-hour Temporal Window)
# =========================================================================
def build_cnn_bilstm(input_shape, config=None):
    """Build CNN + BiLSTM hybrid model."""
    if config is None:
        config = CNN_BILSTM_CONFIG

    model = keras.Sequential([
        layers.Conv1D(config["cnn_filters_1"], config["kernel_size"],
                      activation='relu', padding='same',
                      input_shape=input_shape),
        layers.Conv1D(config["cnn_filters_2"], config["kernel_size"],
                      activation='relu', padding='same'),
        layers.Dropout(config["dropout"]),
        layers.Bidirectional(layers.LSTM(config["bilstm_units"], return_sequences=False)),
        layers.Dense(config["dense_units"], activation='relu'),
        layers.Dropout(config["dropout"]),
        layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]),
                  loss='mse', metrics=['mae'])
    return model


def train_cnn_bilstm(train_df, val_df, config=None):
    """Train CNN + BiLSTM model with 168-hour lookback window."""
    if config is None:
        config = CNN_BILSTM_CONFIG

    print(f"\n[CNN+BiLSTM] Training Hybrid Model: CNN + BiLSTM ({LOOKBACK_HOURS}h window)...")

    X_train, y_train, X_val, y_val, scaler_X, scaler_y, feature_cols, val_indices = \
        prepare_sequence_data(train_df, val_df, lookback=LOOKBACK_HOURS)
    print(f"  Sequence shapes - X_train: {X_train.shape}, X_val: {X_val.shape}")

    model = build_cnn_bilstm(X_train.shape[1:], config)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=get_callbacks("cnn_bilstm", config["patience"]),
        verbose=1,
    )

    predictions_scaled = model.predict(X_val, verbose=0).flatten()
    pred_raw = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    pred_raw = np.maximum(pred_raw, 0)

    # Reorder predictions back to val_df row order
    predictions = np.zeros(len(val_df), dtype=np.float32)
    predictions[val_indices] = pred_raw
    actuals = val_df[TARGET_COL].values

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(actuals, predictions)
    print_metrics("CNN + BiLSTM (168h)", metrics)
    save_metrics("cnn_bilstm", metrics, METRICS_DIR)

    model_path = os.path.join(MODELS_DIR, "cnn_bilstm.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    return {
        "model": model, "predictions": predictions, "actuals": actuals,
        "metrics": metrics, "history": history.history,
        "scalers": (scaler_X, scaler_y), "feature_cols": feature_cols,
    }


# =========================================================================
# Model: XGBoost + BiLSTM (Residual Sequence Modeling, 168h window)
# =========================================================================
def train_xgb_bilstm(train_df, val_df, config=None):
    """
    Train XGBoost + BiLSTM hybrid model.
    Stage 1: XGBoost on tabular features (with 168h lags & rolling stats) -> base load
    Stage 2: BiLSTM on 168h lookback residual sequences -> dynamic error correction
    """
    if config is None:
        config = XGB_BILSTM_CONFIG

    print(f"\n[XGBoost+BiLSTM] Training Hybrid Model: XGBoost + BiLSTM ({LOOKBACK_HOURS}h residual window)...")

    feature_cols = get_tabular_features(train_df)

    # Stage 1: XGBoost base model
    print("  Stage 1: Training XGBoost base model on tabular features...")
    xgb_config = XGBOOST_CONFIG.copy()
    xgb_config.pop("early_stopping_rounds", None)

    xgb_model = xgb.XGBRegressor(
        **xgb_config, objective="reg:squarederror", tree_method="hist"
    )
    X_train_tab = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_val_tab = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    xgb_model.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)

    train_pred_xgb = xgb_model.predict(X_train_tab)
    val_pred_xgb = xgb_model.predict(X_val_tab)

    # Compute residuals
    train_df_copy = train_df.copy()
    val_df_copy = val_df.copy()
    train_df_copy["residual"] = y_train - train_pred_xgb
    val_df_copy["residual"] = y_val - val_pred_xgb

    # Stage 2: BiLSTM on per-building residual sequences
    print(f"  Stage 2: Training BiLSTM on residual sequences ({LOOKBACK_HOURS}h window)...")
    scaler_r = StandardScaler()
    scaler_r.fit(train_df_copy[["residual"]].values)

    lookback = LOOKBACK_HOURS
    X_train_res_list, y_train_res_list = [], []
    X_val_res_list, y_val_res_list = [], []
    val_res_indices_list = []

    for bid in train_df["building_id"].unique():
        b_train = train_df_copy[train_df_copy["building_id"] == bid].sort_values("timestamp")
        b_val = val_df_copy[val_df_copy["building_id"] == bid].sort_values("timestamp")

        if len(b_train) <= lookback:
            continue

        r_tr_scaled = scaler_r.transform(b_train[["residual"]].values).flatten()

        for i in range(lookback, len(r_tr_scaled)):
            X_train_res_list.append(r_tr_scaled[i - lookback : i])
            y_train_res_list.append(r_tr_scaled[i])

        if len(b_val) > 0:
            b_comb = pd.concat([b_train.iloc[-lookback:], b_val])
            r_comb_scaled = scaler_r.transform(b_comb[["residual"]].values).flatten()

            for i in range(lookback, len(r_comb_scaled)):
                X_val_res_list.append(r_comb_scaled[i - lookback : i])
                y_val_res_list.append(r_comb_scaled[i])
                val_res_indices_list.append(b_comb.index[i])

    X_train_seq = np.array(X_train_res_list, dtype=np.float32)[..., np.newaxis]
    y_train_seq = np.array(y_train_res_list, dtype=np.float32)
    X_val_seq = np.array(X_val_res_list, dtype=np.float32)[..., np.newaxis]
    y_val_seq = np.array(y_val_res_list, dtype=np.float32)
    val_res_indices = np.array(val_res_indices_list)

    print(f"  Residual sequence shapes - X_train: {X_train_seq.shape}, X_val: {X_val_seq.shape}")

    bilstm = keras.Sequential([
        layers.Bidirectional(layers.LSTM(config["bilstm_units"], return_sequences=False),
                             input_shape=(lookback, 1)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(config.get("bilstm_dropout", 0.2)),
        layers.Dense(1)
    ])
    bilstm.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]),
                   loss='mse', metrics=['mae'])

    bilstm.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=get_callbacks("xgb_bilstm", config["patience"]),
        verbose=1,
    )

    # Predict residual correction and align back to val_df
    residual_pred_scaled = bilstm.predict(X_val_seq, verbose=0).flatten()
    pred_res_raw = scaler_r.inverse_transform(residual_pred_scaled.reshape(-1, 1)).flatten()
    pred_res_aligned = np.zeros(len(val_df), dtype=np.float32)
    pred_res_aligned[val_res_indices] = pred_res_raw

    # Final combined prediction
    final_predictions = np.maximum(val_pred_xgb + pred_res_aligned, 0)
    final_actuals = val_df[TARGET_COL].values

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(final_actuals, final_predictions)
    print_metrics("XGBoost + BiLSTM", metrics)
    save_metrics("xgb_bilstm", metrics, METRICS_DIR)

    # Save models
    xgb_path = os.path.join(MODELS_DIR, "xgb_bilstm_xgb.json")
    xgb_model.save_model(xgb_path)
    bilstm_path = os.path.join(MODELS_DIR, "xgb_bilstm_bilstm.keras")
    bilstm.save(bilstm_path)
    print(f"  Models saved to: {MODELS_DIR}")

    return {
        "xgb_model": xgb_model, "bilstm_model": bilstm,
        "predictions": final_predictions, "actuals": final_actuals,
        "metrics": metrics, "feature_cols": feature_cols,
    }


# =========================================================================
# Model: ANN + ANN (Dual Branch ANN)
# =========================================================================
def train_dual_ann(train_df, val_df, config=None):
    """
    Train Dual-ANN model:
    ANN-A: Temporal branch (time, sin/cos cyclical, 168h lags & rolling stats)
    ANN-B: Contextual branch (weather, site, building specs)
    Fusion Layer: Deep multi-modal dense network
    """
    if config is None:
        config = DUAL_ANN_CONFIG

    print("\n[Dual-ANN] Training Hybrid Model: ANN + ANN (Dual Branch)...")

    feature_cols = get_tabular_features(train_df)

    # Split features into temporal and contextual
    temporal_keywords = [
        "hour", "day", "week", "month", "quarter", "lag", "rolling",
        "sin", "cos", "weekend", "holiday", "semester"
    ]
    contextual_keywords = [
        "temperature", "cloud", "dew", "precip", "pressure",
        "wind", "square_feet", "year_built", "floor_count"
    ]

    temporal_cols = [c for c in feature_cols if any(k in c for k in temporal_keywords)]
    contextual_cols = [c for c in feature_cols if any(k in c for k in contextual_keywords)]

    remaining = [c for c in feature_cols if c not in temporal_cols and c not in contextual_cols]
    temporal_cols.extend(remaining)

    print(f"  Temporal features: {len(temporal_cols)}, Contextual features: {len(contextual_cols)}")

    scaler_t = StandardScaler()
    scaler_c = StandardScaler()
    scaler_y = StandardScaler()

    X_train_t = scaler_t.fit_transform(train_df[temporal_cols].values)
    X_train_c = scaler_c.fit_transform(train_df[contextual_cols].values) if contextual_cols else np.zeros((len(train_df), 1))
    y_train = scaler_y.fit_transform(train_df[[TARGET_COL]].values).flatten()

    X_val_t = scaler_t.transform(val_df[temporal_cols].values)
    X_val_c = scaler_c.transform(val_df[contextual_cols].values) if contextual_cols else np.zeros((len(val_df), 1))
    y_val = scaler_y.transform(val_df[[TARGET_COL]].values).flatten()
    actuals = val_df[TARGET_COL].values

    # Build Dual-ANN
    input_t = layers.Input(shape=(X_train_t.shape[1],), name="temporal_input")
    input_c = layers.Input(shape=(X_train_c.shape[1],), name="contextual_input")

    # ANN-A: Temporal branch
    x_a = input_t
    for units in config["ann_a_layers"]:
        x_a = layers.Dense(units, activation='relu')(x_a)
        x_a = layers.Dropout(config["dropout"])(x_a)

    # ANN-B: Contextual branch
    x_b = input_c
    for units in config["ann_b_layers"]:
        x_b = layers.Dense(units, activation='relu')(x_b)
        x_b = layers.Dropout(config["dropout"])(x_b)

    # Fusion
    fused = layers.Concatenate()([x_a, x_b])
    for units in config["fusion_layers"]:
        fused = layers.Dense(units, activation='relu')(fused)
        fused = layers.Dropout(config["dropout"])(fused)

    output = layers.Dense(1)(fused)

    model = Model(inputs=[input_t, input_c], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]),
                  loss='mse', metrics=['mae'])

    history = model.fit(
        [X_train_t, X_train_c], y_train,
        validation_data=([X_val_t, X_val_c], y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=get_callbacks("dual_ann", config["patience"]),
        verbose=1,
    )

    predictions_scaled = model.predict([X_val_t, X_val_c], verbose=0).flatten()
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    predictions = np.maximum(predictions, 0)

    # Compute R², RMSE, MAE, MAPE
    metrics = compute_metrics(actuals, predictions)
    print_metrics("ANN + ANN (Dual)", metrics)
    save_metrics("dual_ann", metrics, METRICS_DIR)

    model_path = os.path.join(MODELS_DIR, "dual_ann.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    return {
        "model": model, "predictions": predictions, "actuals": actuals,
        "metrics": metrics, "history": history.history,
        "temporal_cols": temporal_cols, "contextual_cols": contextual_cols,
    }


def train_all_hybrid_models(train_df, val_df):
    """Train all 3 focused hybrid models."""
    results = {}
    results["xgb_bilstm"] = train_xgb_bilstm(train_df, val_df)
    results["cnn_bilstm"] = train_cnn_bilstm(train_df, val_df)
    results["dual_ann"] = train_dual_ann(train_df, val_df)
    return results
