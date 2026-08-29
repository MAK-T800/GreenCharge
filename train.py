"""
GreenCharge Training Pipeline
==============================
End-to-end orchestrator for the 7 selected models + Meta-Ensemble:
- Standalone: XGBoost, Random Forest, ANN, CNN (168h window)
- Hybrids: XGBoost+BiLSTM (168h residual), CNN+BiLSTM (168h window), ANN+ANN (Dual Branch)
- Meta-Ensemble + Anomaly Detection + Load Balancing
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_TRAIN, PROCESSED_VAL, PROCESSED_FEATURES,
    MODELS_DIR, METRICS_DIR, TARGET_COL, MODEL_NAMES, LOOKBACK_HOURS
)

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def convert_to_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
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


def run_pipeline():
    """Step 1: Load or generate processed data."""
    print("\n" + "=" * 70)
    print("  STEP 1: DATA PIPELINE")
    print("=" * 70)

    if os.path.exists(PROCESSED_TRAIN) and os.path.exists(PROCESSED_VAL):
        print("  Processed data already exists. Loading...")
        train_df = pd.read_parquet(PROCESSED_TRAIN)
        val_df = pd.read_parquet(PROCESSED_VAL)
        print(f"  Train: {train_df.shape}, Val: {val_df.shape}")
    else:
        from data_pipeline import run_pipeline as dp_run
        train_df, val_df, _ = dp_run()

    return train_df, val_df


def clean_old_metrics():
    """Remove metrics for models outside the requested 7-model suite."""
    valid_keys = set(MODEL_NAMES.keys())
    if os.path.exists(METRICS_DIR):
        for fname in os.listdir(METRICS_DIR):
            if fname.endswith(".json"):
                base_name = fname.replace(".json", "")
                if base_name not in valid_keys and base_name != "comparison_report":
                    try:
                        os.remove(os.path.join(METRICS_DIR, fname))
                        print(f"  Removed obsolete metric file: {fname}")
                    except Exception:
                        pass


def save_predictions(all_results):
    """Save model predictions for dashboard visualization."""
    predictions = {}
    for name, result in all_results.items():
        if result is not None:
            pred = result.get("predictions")
            act = result.get("actuals")
            if pred is not None:
                entry = {"predictions": pred[:500].tolist() if hasattr(pred, 'tolist') else list(pred[:500])}
                if act is not None:
                    entry["actuals"] = act[:500].tolist() if hasattr(act, 'tolist') else list(act[:500])
                predictions[name] = entry

    pred_path = os.path.join(MODELS_DIR, "all_predictions.json")
    with open(pred_path, 'w') as f:
        json.dump(predictions, f)
    print(f"  Predictions saved to: {pred_path}")


def main():
    """Main training pipeline."""
    start_time = time.time()

    print("+" + "=" * 68 + "+")
    print("|" + " GreenCharge: AI-Based Load Forecasting Training Pipeline".center(68) + "|")
    print("|" + f" 7 Models + Ensemble ({LOOKBACK_HOURS}h Window) + Anomaly + Load Balancing".center(68) + "|")
    print("+" + "=" * 68 + "+")

    # Step 1: Data Pipeline
    train_df, val_df = run_pipeline()

    # Clean old metrics to keep comparison clean
    clean_old_metrics()

    all_results = {}

    # Step 2: Tree-Based Models (XGBoost, Random Forest)
    print("\n" + "=" * 70)
    print("  STEP 2: TREE-BASED MODELS (XGBoost, Random Forest)")
    print("=" * 70)
    from models.tree_models import train_xgboost, train_random_forest

    try:
        all_results["xgboost"] = train_xgboost(train_df, val_df)
    except Exception as e:
        print(f"  [WARN] XGBoost failed: {e}")

    try:
        all_results["random_forest"] = train_random_forest(train_df, val_df)
    except Exception as e:
        print(f"  [WARN] Random Forest failed: {e}")

    # Step 3: Deep Learning Models (ANN, CNN with 168h window)
    print("\n" + "=" * 70)
    print(f"  STEP 3: DEEP LEARNING MODELS (ANN, CNN with {LOOKBACK_HOURS}h window)")
    print("=" * 70)
    from models.deep_models import train_ann, train_cnn

    try:
        all_results["ann"] = train_ann(train_df, val_df)
    except Exception as e:
        print(f"  [WARN] ANN failed: {e}")

    try:
        all_results["cnn"] = train_cnn(train_df, val_df)
    except Exception as e:
        print(f"  [WARN] CNN failed: {e}")

    # Step 4: Hybrid Models (XGBoost+BiLSTM, CNN+BiLSTM, Dual ANN)
    print("\n" + "=" * 70)
    print(f"  STEP 4: HYBRID MODELS (XGB+BiLSTM, CNN+BiLSTM, Dual ANN with {LOOKBACK_HOURS}h window)")
    print("=" * 70)
    from models.hybrid_models import (
        train_xgb_bilstm, train_cnn_bilstm, train_dual_ann
    )

    try:
        all_results["xgb_bilstm"] = train_xgb_bilstm(train_df, val_df)
    except Exception as e:
        print(f"  [WARN] XGBoost + BiLSTM failed: {e}")

    try:
        all_results["cnn_bilstm"] = train_cnn_bilstm(train_df, val_df)
    except Exception as e:
        print(f"  [WARN] CNN + BiLSTM failed: {e}")

    try:
        all_results["dual_ann"] = train_dual_ann(train_df, val_df)
    except Exception as e:
        print(f"  [WARN] Dual ANN failed: {e}")

    # Step 5: Meta-Ensemble
    print("\n" + "=" * 70)
    print("  STEP 5: META-ENSEMBLE COMBINATION")
    print("=" * 70)
    from models.ensemble import build_ensemble

    ensemble_inputs = {k: v for k, v in all_results.items() if v and v.get("predictions") is not None}
    try:
        ens_res = build_ensemble(ensemble_inputs)
        if ens_res and ens_res.get("best"):
            all_results["meta_ensemble"] = ens_res["best"]
    except Exception as e:
        print(f"  [WARN] Ensemble failed: {e}")

    # Step 6: Save predictions
    save_predictions(all_results)

    # Step 7: Anomaly Detection
    print("\n" + "=" * 70)
    print("  STEP 6: ANOMALY DETECTION")
    print("=" * 70)
    try:
        full_df = pd.read_parquet(PROCESSED_FEATURES)
        from models.anomaly_detection import run_anomaly_detection
        anomaly_df = run_anomaly_detection(full_df)
        anomaly_df.to_parquet(PROCESSED_FEATURES, index=False)
        print("  Anomaly results saved to feature matrix.")
    except Exception as e:
        print(f"  [WARN] Anomaly detection failed: {e}")

    # Step 8: Load Balancing
    print("\n" + "=" * 70)
    print("  STEP 7: LOAD BALANCING SIMULATION")
    print("=" * 70)
    try:
        full_df = pd.read_parquet(PROCESSED_FEATURES)
        from load_balancing import run_load_balancing
        lb_results = run_load_balancing(full_df)
        lb_path = os.path.join(MODELS_DIR, "load_balance_results.json")
        with open(lb_path, 'w') as f:
            json.dump(convert_to_serializable(lb_results), f, indent=2)
        print(f"  Load balancing results saved to: {lb_path}")
    except Exception as e:
        print(f"  [WARN] Load balancing failed: {e}")

    # Final Summary
    elapsed = time.time() - start_time
    print("\n" + "+" + "=" * 68 + "+")
    print("|" + " TRAINING COMPLETE".center(68) + "|")
    print("+" + "=" * 68 + "+")
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print(f"  Models trained: {len(all_results)}")
    print(f"  Metrics saved to: {METRICS_DIR}")
    print(f"\n  To launch dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
