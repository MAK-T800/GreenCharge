"""
Fast resume training: train remaining deep/hybrid models with reduced
lookback (48 instead of 168) and subsampled sequences for CPU speed.
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import config
from config import (
    PROCESSED_TRAIN, PROCESSED_VAL, PROCESSED_FEATURES,
    MODELS_DIR, METRICS_DIR, TARGET_COL,
)
from models.evaluation import compute_metrics, print_metrics, save_metrics

# ── Override configs for speed ──────────────────────────────────────────
FAST_LOOKBACK = 48   # 48h instead of 168h
MAX_TRAIN_SEQ = 8000
MAX_VAL_SEQ   = 2000
FAST_EPOCHS   = 15
FAST_PATIENCE = 4
FAST_BATCH    = 256

# Patch all deep/hybrid configs
for cfg_name in [
    "CNN_CONFIG", "TRANSFORMER_CONFIG",
    "CNN_BILSTM_CONFIG", "CNN_LSTM_CONFIG", "XGB_BILSTM_CONFIG",
    "XGB_TRANSFORMER_CONFIG", "DUAL_ANN_CONFIG", "XGB_BILSTM_TRANSFORMER_CONFIG",
]:
    cfg = getattr(config, cfg_name)
    cfg["epochs"]   = FAST_EPOCHS
    cfg["patience"]  = FAST_PATIENCE
    cfg["batch_size"] = FAST_BATCH

# Monkey-patch LOOKBACK_HOURS in config and deep_models
config.LOOKBACK_HOURS = FAST_LOOKBACK

import models.deep_models as dm
dm.LOOKBACK_HOURS = FAST_LOOKBACK

# Patch prepare_sequence_data to subsample
_orig_prepare = dm.prepare_sequence_data

def fast_prepare_sequence_data(train_df, val_df, lookback=FAST_LOOKBACK):
    """Prepare sequence data with subsampling for CPU speed."""
    X_train, y_train, X_val, y_val, sX, sY, fcols = _orig_prepare(
        train_df, val_df, lookback
    )
    # Subsample
    if len(X_train) > MAX_TRAIN_SEQ:
        idx = np.random.choice(len(X_train), MAX_TRAIN_SEQ, replace=False)
        idx.sort()
        X_train, y_train = X_train[idx], y_train[idx]
    if len(X_val) > MAX_VAL_SEQ:
        idx = np.random.choice(len(X_val), MAX_VAL_SEQ, replace=False)
        idx.sort()
        X_val, y_val = X_val[idx], y_val[idx]
    return X_train, y_train, X_val, y_val, sX, sY, fcols

dm.prepare_sequence_data = fast_prepare_sequence_data

# Also patch in hybrid_models
import models.hybrid_models as hm
hm.prepare_sequence_data = fast_prepare_sequence_data
hm.LOOKBACK_HOURS = FAST_LOOKBACK


def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def main():
    start = time.time()
    np.random.seed(42)
    
    print("=" * 70)
    print("  FAST RESUME: Training remaining models (reduced lookback=48)")
    print("=" * 70)

    train_df = pd.read_parquet(PROCESSED_TRAIN)
    val_df   = pd.read_parquet(PROCESSED_VAL)
    all_results = {}

    # ── Reload already-trained models' predictions ──────────────────────
    # SARIMA
    from models.arima_model import train_from_dataframe
    try:
        bid = train_df["building_id"].unique()[0]
        all_results["sarima"] = train_from_dataframe(train_df, val_df, building_id=bid)
        print("  [OK] SARIMA")
    except Exception as e:
        print(f"  [WARN] SARIMA: {e}")

    # Tree models (fast)
    from models.tree_models import train_all_tree_models
    try:
        all_results.update(train_all_tree_models(train_df, val_df))
        print("  [OK] Tree models")
    except Exception as e:
        print(f"  [WARN] Tree: {e}")

    # LSTM – reload from saved
    try:
        from keras import models as km
        lstm_path = os.path.join(MODELS_DIR, "lstm.keras")
        if os.path.exists(lstm_path):
            _, _, X_val_seq, y_val_seq, _, sY, _ = fast_prepare_sequence_data(train_df, val_df)
            lstm_model = km.load_model(lstm_path)
            pred_s = lstm_model.predict(X_val_seq, verbose=0, batch_size=512).flatten()
            preds = np.maximum(sY.inverse_transform(pred_s.reshape(-1,1)).flatten(), 0)
            acts  = sY.inverse_transform(y_val_seq.reshape(-1,1)).flatten()
            met = compute_metrics(acts, preds)
            print_metrics("LSTM", met)
            save_metrics("lstm", met, METRICS_DIR)
            all_results["lstm"] = {"predictions": preds, "actuals": acts, "metrics": met}
            print("  [OK] LSTM (reloaded)")
    except Exception as e:
        print(f"  [WARN] LSTM reload: {e}")

    # ANN – reload from saved
    try:
        ann_path = os.path.join(MODELS_DIR, "ann.keras")
        if os.path.exists(ann_path):
            from models.deep_models import get_tabular_features
            from sklearn.preprocessing import StandardScaler
            fcols = get_tabular_features(train_df)
            sX, sY2 = StandardScaler(), StandardScaler()
            sX.fit(train_df[fcols].values)
            sY2.fit(train_df[[TARGET_COL]].values)
            Xv = sX.transform(val_df[fcols].values)
            yv = sY2.transform(val_df[[TARGET_COL]].values).flatten()
            ann_model = km.load_model(ann_path)
            pred_s = ann_model.predict(Xv, verbose=0).flatten()
            preds = np.maximum(sY2.inverse_transform(pred_s.reshape(-1,1)).flatten(), 0)
            acts = sY2.inverse_transform(yv.reshape(-1,1)).flatten()
            met = compute_metrics(acts, preds)
            print_metrics("ANN", met)
            save_metrics("ann", met, METRICS_DIR)
            all_results["ann"] = {"predictions": preds, "actuals": acts, "metrics": met}
            print("  [OK] ANN (reloaded)")
    except Exception as e:
        print(f"  [WARN] ANN reload: {e}")

    # ── Train missing models ────────────────────────────────────────────
    from models.deep_models import train_cnn, train_transformer

    for name, trainer in [("cnn", train_cnn), ("transformer", train_transformer)]:
        print(f"\n{'='*60}")
        print(f"  Training: {name.upper()}")
        print(f"{'='*60}")
        try:
            result = trainer(train_df, val_df)
            all_results[name] = result
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    from models.hybrid_models import (
        train_cnn_bilstm, train_cnn_lstm, train_xgb_bilstm,
        train_xgb_transformer, train_dual_ann, train_xgb_bilstm_transformer,
    )

    hybrid_trainers = {
        "cnn_bilstm": train_cnn_bilstm,
        "cnn_lstm": train_cnn_lstm,
        "xgb_bilstm": train_xgb_bilstm,
        "xgb_transformer": train_xgb_transformer,
        "dual_ann": train_dual_ann,
        "xgb_bilstm_transformer": train_xgb_bilstm_transformer,
    }

    for name, trainer in hybrid_trainers.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name.upper()}")
        print(f"{'='*60}")
        try:
            result = trainer(train_df, val_df)
            all_results[name] = result
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    # ── Meta-Ensemble ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  META-ENSEMBLE")
    print(f"{'='*60}")
    from models.ensemble import build_ensemble
    ens_input = {k: v for k, v in all_results.items() if v and v.get("predictions") is not None}
    try:
        build_ensemble(ens_input)
    except Exception as e:
        print(f"  [WARN] Ensemble: {e}")

    # Save predictions for dashboard
    predictions = {}
    for name, result in all_results.items():
        if result is not None:
            pred = result.get("predictions")
            act  = result.get("actuals")
            if pred is not None:
                entry = {"predictions": pred[:500].tolist() if hasattr(pred, 'tolist') else list(pred[:500])}
                if act is not None:
                    entry["actuals"] = act[:500].tolist() if hasattr(act, 'tolist') else list(act[:500])
                predictions[name] = entry
    with open(os.path.join(MODELS_DIR, "all_predictions.json"), 'w') as f:
        json.dump(predictions, f)

    # ── Anomaly Detection ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ANOMALY DETECTION")
    print(f"{'='*60}")
    try:
        full_df = pd.read_parquet(PROCESSED_FEATURES)
        from models.anomaly_detection import run_anomaly_detection
        anomaly_df = run_anomaly_detection(full_df)
        anomaly_df.to_parquet(PROCESSED_FEATURES, index=False)
        print("  Anomaly results saved.")
    except Exception as e:
        print(f"  [WARN] Anomaly: {e}")

    # ── Load Balancing ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  LOAD BALANCING")
    print(f"{'='*60}")
    try:
        full_df = pd.read_parquet(PROCESSED_FEATURES)
        from load_balancing import run_load_balancing
        lb_results = run_load_balancing(full_df)
        lb_path = os.path.join(MODELS_DIR, "load_balance_results.json")
        with open(lb_path, "w") as f:
            json.dump(convert_to_serializable(lb_results), f, indent=2)
        print(f"  Load balance saved to: {lb_path}")
    except Exception as e:
        print(f"  [WARN] Load balancing: {e}")

    elapsed = (time.time() - start) / 60
    print(f"\n{'='*70}")
    print(f"  FAST RESUME COMPLETE in {elapsed:.1f} minutes")
    print(f"  Metrics: {METRICS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
