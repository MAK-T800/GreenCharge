"""Resume training from saved artifacts and finish pipeline steps."""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import config
from config import (
    PROCESSED_TRAIN,
    PROCESSED_VAL,
    PROCESSED_FEATURES,
    MODELS_DIR,
    METRICS_DIR,
)
from models.evaluation import compute_metrics, load_metrics
from train import save_predictions, convert_to_serializable

# Faster CPU training for remaining deep/hybrid models
FAST_EPOCHS = 20
FAST_PATIENCE = 5
for cfg_name in [
    "ANN_CONFIG",
    "CNN_CONFIG",
    "TRANSFORMER_CONFIG",
    "CNN_BILSTM_CONFIG",
    "CNN_LSTM_CONFIG",
    "XGB_BILSTM_CONFIG",
    "XGB_TRANSFORMER_CONFIG",
    "DUAL_ANN_CONFIG",
    "XGB_BILSTM_TRANSFORMER_CONFIG",
]:
    cfg = getattr(config, cfg_name)
    cfg["epochs"] = FAST_EPOCHS
    cfg["patience"] = FAST_PATIENCE


def reload_lstm_results(train_df, val_df):
    """Reload LSTM predictions from saved model without retraining."""
    from models.deep_models import prepare_sequence_data
    import keras

    model_path = os.path.join(MODELS_DIR, "lstm.keras")
    if not os.path.exists(model_path):
        return None

    _, _, X_val, y_val, _, scaler_y, _ = prepare_sequence_data(train_df, val_df)
    model = keras.models.load_model(model_path)
    pred_scaled = model.predict(X_val, verbose=0).flatten()
    predictions = np.maximum(
        scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten(), 0
    )
    actuals = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
    metrics = load_metrics(METRICS_DIR).get("lstm", {}).get("metrics")
    if metrics is None:
        metrics = compute_metrics(actuals, predictions)

    print("[Resume] LSTM loaded from saved model.")
    return {"predictions": predictions, "actuals": actuals, "metrics": metrics}


def main():
    start = time.time()
    print("=" * 70)
    print("  RESUMING GreenCharge Training Pipeline")
    print("=" * 70)

    train_df = pd.read_parquet(PROCESSED_TRAIN)
    val_df = pd.read_parquet(PROCESSED_VAL)
    all_results = {}

    # Fast models: retrain to get fresh prediction arrays
    from models.arima_model import train_from_dataframe
    from models.tree_models import train_all_tree_models

    try:
        building_id = train_df["building_id"].unique()[0]
        all_results["sarima"] = train_from_dataframe(
            train_df, val_df, building_id=building_id
        )
    except Exception as exc:
        print(f"  [WARN] SARIMA failed: {exc}")

    try:
        all_results.update(train_all_tree_models(train_df, val_df))
    except Exception as exc:
        print(f"  [WARN] Tree models failed: {exc}")

    try:
        lstm_result = reload_lstm_results(train_df, val_df)
        if lstm_result:
            all_results["lstm"] = lstm_result
    except Exception as exc:
        print(f"  [WARN] LSTM reload failed: {exc}")

    from models.deep_models import train_ann, train_cnn, train_transformer

    for name, trainer in [
        ("ann", train_ann),
        ("cnn", train_cnn),
        ("transformer", train_transformer),
    ]:
        metrics_path = os.path.join(METRICS_DIR, f"{name}.json")
        if os.path.exists(metrics_path):
            print(f"  [Skip] {name} metrics already exist.")
            continue
        try:
            all_results[name] = trainer(train_df, val_df)
        except Exception as exc:
            print(f"  [WARN] {name} failed: {exc}")

    from models.hybrid_models import (
        train_cnn_bilstm,
        train_cnn_lstm,
        train_xgb_bilstm,
        train_xgb_transformer,
        train_dual_ann,
        train_xgb_bilstm_transformer,
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
        metrics_path = os.path.join(METRICS_DIR, f"{name}.json")
        if os.path.exists(metrics_path):
            print(f"  [Skip] {name} metrics already exist.")
            continue
        try:
            all_results[name] = trainer(train_df, val_df)
        except Exception as exc:
            print(f"  [WARN] {name} failed: {exc}")

    from models.ensemble import build_ensemble

    ensemble_results = {
        name: result
        for name, result in all_results.items()
        if result and result.get("predictions") is not None
    }
    try:
        build_ensemble(ensemble_results)
    except Exception as exc:
        print(f"  [WARN] Ensemble failed: {exc}")

    save_predictions(ensemble_results)

    try:
        full_df = pd.read_parquet(PROCESSED_FEATURES)
        from models.anomaly_detection import run_anomaly_detection

        anomaly_df = run_anomaly_detection(full_df)
        anomaly_df.to_parquet(PROCESSED_FEATURES, index=False)
        print("  Anomaly results saved to feature matrix.")
    except Exception as exc:
        print(f"  [WARN] Anomaly detection failed: {exc}")

    try:
        full_df = pd.read_parquet(PROCESSED_FEATURES)
        from load_balancing import run_load_balancing

        lb_results = run_load_balancing(full_df)
        lb_path = os.path.join(MODELS_DIR, "load_balance_results.json")
        with open(lb_path, "w") as handle:
            json.dump(convert_to_serializable(lb_results), handle, indent=2)
        print(f"  Load balancing results saved to: {lb_path}")
    except Exception as exc:
        print(f"  [WARN] Load balancing failed: {exc}")

    elapsed = (time.time() - start) / 60
    print("\n" + "=" * 70)
    print(f"  RESUME COMPLETE in {elapsed:.1f} minutes")
    print(f"  Metrics in: {METRICS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
