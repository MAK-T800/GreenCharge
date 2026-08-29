"""
GreenCharge Anomaly Detection
==============================
Isolation Forest, Z-Score, and Autoencoder-based anomaly detection.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANOMALY_CONFIG, MODELS_DIR, TARGET_COL

warnings.filterwarnings("ignore")


def detect_anomalies_isolation_forest(df, config=None):
    """
    Detect anomalies using Isolation Forest.
    
    Returns:
        DataFrame with anomaly flags and scores
    """
    if config is None:
        config = ANOMALY_CONFIG["isolation_forest"]

    print("\n[Anomaly] Running Isolation Forest...")

    features = [TARGET_COL]
    # Add time features if available
    for col in ["hour", "day_of_week", "month"]:
        if col in df.columns:
            features.append(col)

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=config["contamination"],
        n_estimators=config["n_estimators"],
        random_state=config["random_state"],
    )
    
    labels = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    result = df.copy()
    result["anomaly_if"] = (labels == -1).astype(int)
    result["anomaly_score_if"] = -scores  # Higher = more anomalous

    n_anomalies = result["anomaly_if"].sum()
    print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(df):.2f}%)")

    # Save model
    model_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({"model": model, "scaler": scaler, "features": features}, f)

    return result, model


def detect_anomalies_zscore(df, threshold=None):
    """
    Detect anomalies using Z-Score method.
    
    Returns:
        DataFrame with z-score anomaly flags
    """
    if threshold is None:
        threshold = ANOMALY_CONFIG["z_score_threshold"]

    print(f"\n[Anomaly] Running Z-Score detection (threshold={threshold})...")

    result = df.copy()
    
    # Per-building z-score
    result["zscore"] = result.groupby("building_id")[TARGET_COL].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    result["anomaly_zscore"] = (result["zscore"].abs() > threshold).astype(int)

    n_anomalies = result["anomaly_zscore"].sum()
    print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(df):.2f}%)")

    return result


def detect_anomalies_autoencoder(df, config=None):
    """
    Detect anomalies using Autoencoder reconstruction error.
    
    Returns:
        DataFrame with autoencoder anomaly flags
    """
    if config is None:
        config = ANOMALY_CONFIG["autoencoder"]

    print("\n[Anomaly] Running Autoencoder anomaly detection...")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow import keras
    from keras import layers

    features = [TARGET_COL]
    for col in ["hour", "day_of_week", "month", "is_weekend"]:
        if col in df.columns:
            features.append(col)

    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]
    encoding_dim = config["encoding_dim"]

    # Build autoencoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim * 2, activation='relu')(encoder_input)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=0.1,
        verbose=0,
    )

    # Compute reconstruction error
    reconstructed = autoencoder.predict(X_scaled, verbose=0)
    mse_per_sample = np.mean((X_scaled - reconstructed) ** 2, axis=1)

    # Threshold: mean + 2*std of reconstruction error
    threshold = mse_per_sample.mean() + 2 * mse_per_sample.std()

    result = df.copy()
    result["recon_error"] = mse_per_sample
    result["anomaly_ae"] = (mse_per_sample > threshold).astype(int)

    n_anomalies = result["anomaly_ae"].sum()
    print(f"  Detected {n_anomalies} anomalies ({100*n_anomalies/len(df):.2f}%)")

    # Save autoencoder
    ae_path = os.path.join(MODELS_DIR, "anomaly_autoencoder.keras")
    autoencoder.save(ae_path)

    return result, autoencoder


def classify_severity(df):
    """
    Classify anomaly severity based on deviation magnitude.
    
    Returns:
        DataFrame with severity column
    """
    result = df.copy()

    # Compute combined anomaly flag
    anomaly_cols = [c for c in result.columns if c.startswith("anomaly_") and c not in ["anomaly_score_if"]]
    if not anomaly_cols:
        result["is_anomaly"] = 0
        result["severity"] = "Normal"
        return result

    result["anomaly_votes"] = result[anomaly_cols].sum(axis=1)
    result["is_anomaly"] = (result["anomaly_votes"] >= 1).astype(int)

    # Severity based on z-score magnitude and vote count
    conditions = [
        (result["anomaly_votes"] >= 3),  # All methods agree
        (result["anomaly_votes"] == 2),  # Two methods agree
        (result["anomaly_votes"] == 1),  # One method flags
    ]
    choices = ["High", "Medium", "Low"]
    result["severity"] = np.select(conditions, choices, default="Normal")

    # Compute deviation percentage
    result["deviation_pct"] = result.groupby("building_id")[TARGET_COL].transform(
        lambda x: ((x - x.mean()) / (x.mean() + 1e-8)) * 100
    )

    severity_counts = result[result["is_anomaly"] == 1]["severity"].value_counts()
    print(f"\n[Anomaly] Severity distribution:")
    for sev, count in severity_counts.items():
        print(f"  {sev}: {count}")

    return result


def run_anomaly_detection(df):
    """
    Run all anomaly detection methods and combine results.
    
    Returns:
        DataFrame with all anomaly flags, scores, and severity
    """
    print("\n" + "=" * 60)
    print("  ANOMALY DETECTION")
    print("=" * 60)

    # Method 1: Isolation Forest
    df, if_model = detect_anomalies_isolation_forest(df)

    # Method 2: Z-Score
    df = detect_anomalies_zscore(df)

    # Method 3: Autoencoder
    df, ae_model = detect_anomalies_autoencoder(df)

    # Classify severity
    df = classify_severity(df)

    total_anomalies = df["is_anomaly"].sum()
    print(f"\n  Total anomalies detected: {total_anomalies} ({100*total_anomalies/len(df):.2f}%)")

    return df
