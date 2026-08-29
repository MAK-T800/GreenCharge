"""
Dedicated Meta-Ensemble Evaluator and Updater
=============================================
Loads the 7 trained models, computes aligned predictions across the entire
validation dataset (N = 12,997), evaluates Meta-Ensemble blending (both Weighted
and Stacking), and updates all metrics and predictions artifacts.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from keras import models as km

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_TRAIN, PROCESSED_VAL, TARGET_COL,
    MODELS_DIR, METRICS_DIR, LOOKBACK_HOURS
)
from models.deep_models import get_tabular_features
from models.evaluation import compute_metrics, save_metrics, print_metrics
from models.ensemble import build_ensemble
from train import save_predictions, convert_to_serializable


def main():
    print("=" * 70)
    print("  GREENCHARGE: META-ENSEMBLE RE-EVALUATION & ALIGNMENT PIPELINE")
    print("=" * 70)

    train_df = pd.read_parquet(PROCESSED_TRAIN)
    val_df = pd.read_parquet(PROCESSED_VAL)
    feature_cols = get_tabular_features(train_df)
    y_val = val_df[TARGET_COL].values
    N_val = len(val_df)

    print(f"Loaded data: Train N = {len(train_df)}, Val N = {N_val}")
    print(f"Target Column: {TARGET_COL}")

    all_results = {}

    # -------------------------------------------------------------
    # 1. XGBoost
    # -------------------------------------------------------------
    print("\n[1/7] Evaluating XGBoost...")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost.json"))
    pred_xgb = np.maximum(xgb_model.predict(val_df[feature_cols].values), 0)
    met_xgb = compute_metrics(y_val, pred_xgb)
    print_metrics("XGBoost", met_xgb)
    save_metrics("xgboost", met_xgb, METRICS_DIR)
    all_results["xgboost"] = {
        "predictions": pred_xgb, "actuals": y_val, "metrics": met_xgb
    }

    # -------------------------------------------------------------
    # 2. Random Forest
    # -------------------------------------------------------------
    print("\n[2/7] Evaluating Random Forest...")
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    pred_rf = np.maximum(rf_model.predict(val_df[feature_cols].values), 0)
    met_rf = compute_metrics(y_val, pred_rf)
    print_metrics("Random Forest", met_rf)
    save_metrics("random_forest", met_rf, METRICS_DIR)
    all_results["random_forest"] = {
        "predictions": pred_rf, "actuals": y_val, "metrics": met_rf
    }

    # -------------------------------------------------------------
    # 3. ANN (Tabular Feedforward)
    # -------------------------------------------------------------
    print("\n[3/7] Evaluating ANN...")
    scaler_X = StandardScaler().fit(train_df[feature_cols].values)
    scaler_y = StandardScaler().fit(train_df[[TARGET_COL]].values)
    ann_model = km.load_model(os.path.join(MODELS_DIR, "ann.keras"))
    pred_ann_s = ann_model.predict(scaler_X.transform(val_df[feature_cols].values), verbose=0).flatten()
    pred_ann = np.maximum(scaler_y.inverse_transform(pred_ann_s.reshape(-1, 1)).flatten(), 0)
    met_ann = compute_metrics(y_val, pred_ann)
    print_metrics("ANN", met_ann)
    save_metrics("ann", met_ann, METRICS_DIR)
    all_results["ann"] = {
        "predictions": pred_ann, "actuals": y_val, "metrics": met_ann
    }

    # -------------------------------------------------------------
    # Sequence Data Preparation (Per-Building 168h Lookback)
    # -------------------------------------------------------------
    print(f"\nPreparing 168h Sequence Inputs for CNN & BiLSTM...")
    X_val_list = []
    val_indices = []

    for bid in train_df["building_id"].unique():
        b_train = train_df[train_df["building_id"] == bid].sort_values("timestamp")
        b_val = val_df[val_df["building_id"] == bid].sort_values("timestamp")
        if len(b_val) == 0:
            continue
        b_comb = pd.concat([b_train.iloc[-LOOKBACK_HOURS:], b_val])
        X_comb_s = scaler_X.transform(b_comb[feature_cols].values)
        for i in range(LOOKBACK_HOURS, len(b_comb)):
            X_val_list.append(X_comb_s[i - LOOKBACK_HOURS : i])
            val_indices.append(b_comb.index[i])

    X_val_seq = np.array(X_val_list, dtype=np.float32)
    val_indices = np.array(val_indices)

    # -------------------------------------------------------------
    # 4. CNN (168h Window)
    # -------------------------------------------------------------
    print("\n[4/7] Evaluating CNN (168h)...")
    cnn_model = km.load_model(os.path.join(MODELS_DIR, "cnn.keras"))
    pred_cnn_raw = np.maximum(scaler_y.inverse_transform(cnn_model.predict(X_val_seq, verbose=0)).flatten(), 0)
    pred_cnn = np.zeros(N_val, dtype=np.float32)
    pred_cnn[val_indices] = pred_cnn_raw
    met_cnn = compute_metrics(y_val, pred_cnn)
    print_metrics("CNN (168h)", met_cnn)
    save_metrics("cnn", met_cnn, METRICS_DIR)
    all_results["cnn"] = {
        "predictions": pred_cnn, "actuals": y_val, "metrics": met_cnn
    }

    # -------------------------------------------------------------
    # 5. CNN + BiLSTM (168h Window)
    # -------------------------------------------------------------
    print("\n[5/7] Evaluating CNN + BiLSTM (168h)...")
    cnn_bilstm_model = km.load_model(os.path.join(MODELS_DIR, "cnn_bilstm.keras"))
    pred_cnn_bilstm_raw = np.maximum(scaler_y.inverse_transform(cnn_bilstm_model.predict(X_val_seq, verbose=0)).flatten(), 0)
    pred_cnn_bilstm = np.zeros(N_val, dtype=np.float32)
    pred_cnn_bilstm[val_indices] = pred_cnn_bilstm_raw
    met_cnn_bilstm = compute_metrics(y_val, pred_cnn_bilstm)
    print_metrics("CNN + BiLSTM (168h)", met_cnn_bilstm)
    save_metrics("cnn_bilstm", met_cnn_bilstm, METRICS_DIR)
    all_results["cnn_bilstm"] = {
        "predictions": pred_cnn_bilstm, "actuals": y_val, "metrics": met_cnn_bilstm
    }

    # -------------------------------------------------------------
    # 6. XGBoost + BiLSTM (168h Residual Sequence)
    # -------------------------------------------------------------
    print("\n[6/7] Evaluating XGBoost + BiLSTM (168h Residual)...")
    xgb_base = xgb.XGBRegressor()
    xgb_base.load_model(os.path.join(MODELS_DIR, "xgb_bilstm_xgb.json"))
    bilstm_res = km.load_model(os.path.join(MODELS_DIR, "xgb_bilstm_bilstm.keras"))
    pred_xgb_base = xgb_base.predict(val_df[feature_cols].values)

    train_res = train_df[TARGET_COL].values - xgb_base.predict(train_df[feature_cols].values)
    val_res = y_val - pred_xgb_base
    scaler_r = StandardScaler().fit(train_res.reshape(-1, 1))

    train_df_c = train_df.copy()
    val_df_c = val_df.copy()
    train_df_c["residual"] = train_res
    val_df_c["residual"] = val_res

    X_val_res_list, val_res_indices = [], []
    for bid in train_df["building_id"].unique():
        b_tr = train_df_c[train_df_c["building_id"] == bid].sort_values("timestamp")
        b_v = val_df_c[val_df_c["building_id"] == bid].sort_values("timestamp")
        if len(b_v) == 0:
            continue
        b_comb = pd.concat([b_tr.iloc[-LOOKBACK_HOURS:], b_v])
        r_comb_s = scaler_r.transform(b_comb[["residual"]].values).flatten()
        for i in range(LOOKBACK_HOURS, len(r_comb_s)):
            X_val_res_list.append(r_comb_s[i - LOOKBACK_HOURS : i])
            val_res_indices.append(b_comb.index[i])

    X_val_res_seq = np.array(X_val_res_list, dtype=np.float32)[..., np.newaxis]
    pred_res_raw = scaler_r.inverse_transform(bilstm_res.predict(X_val_res_seq, verbose=0)).flatten()
    pred_res_aligned = np.zeros(N_val, dtype=np.float32)
    pred_res_aligned[np.array(val_res_indices)] = pred_res_raw

    pred_xgb_bilstm = np.maximum(pred_xgb_base + pred_res_aligned, 0)
    met_xgb_bilstm = compute_metrics(y_val, pred_xgb_bilstm)
    print_metrics("XGBoost + BiLSTM", met_xgb_bilstm)
    save_metrics("xgb_bilstm", met_xgb_bilstm, METRICS_DIR)
    all_results["xgb_bilstm"] = {
        "predictions": pred_xgb_bilstm, "actuals": y_val, "metrics": met_xgb_bilstm
    }

    # -------------------------------------------------------------
    # 7. Dual ANN (Temporal + Contextual Branches)
    # -------------------------------------------------------------
    print("\n[7/7] Evaluating Dual ANN...")
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

    scaler_t = StandardScaler().fit(train_df[temporal_cols].values)
    scaler_c = StandardScaler().fit(train_df[contextual_cols].values) if contextual_cols else None
    X_val_t = scaler_t.transform(val_df[temporal_cols].values)
    X_val_c = scaler_c.transform(val_df[contextual_cols].values) if scaler_c else np.zeros((N_val, 1))

    dual_ann_model = km.load_model(os.path.join(MODELS_DIR, "dual_ann.keras"))
    pred_dual_s = dual_ann_model.predict([X_val_t, X_val_c], verbose=0).flatten()
    pred_dual = np.maximum(scaler_y.inverse_transform(pred_dual_s.reshape(-1, 1)).flatten(), 0)
    met_dual = compute_metrics(y_val, pred_dual)
    print_metrics("ANN + ANN (Dual)", met_dual)
    save_metrics("dual_ann", met_dual, METRICS_DIR)
    all_results["dual_ann"] = {
        "predictions": pred_dual, "actuals": y_val, "metrics": met_dual
    }

    # -------------------------------------------------------------
    # Build Meta-Ensemble
    # -------------------------------------------------------------
    ens_res = build_ensemble(all_results)
    if ens_res and ens_res.get("best"):
        all_results["meta_ensemble"] = ens_res["best"]

    # -------------------------------------------------------------
    # Save Predictions
    # -------------------------------------------------------------
    save_predictions(all_results)

    print("\n" + "=" * 70)
    print("  META-ENSEMBLE EVALUATION & LEADERBOARD REFRESH COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
