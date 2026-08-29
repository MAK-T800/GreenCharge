"""
GreenCharge Configuration
=========================
Centralized configuration for paths, hyperparameters, feature lists, and constants.
"""

import os

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
METRICS_DIR = os.path.join(MODELS_DIR, "metrics")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Raw data files
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
BUILDING_META_CSV = os.path.join(DATA_DIR, "building_metadata.csv")
WEATHER_TRAIN_CSV = os.path.join(DATA_DIR, "weather_train.csv")
WEATHER_TEST_CSV = os.path.join(DATA_DIR, "weather_test.csv")

# Processed data files
PROCESSED_TRAIN = os.path.join(PROCESSED_DIR, "train_processed.parquet")
PROCESSED_VAL = os.path.join(PROCESSED_DIR, "val_processed.parquet")
PROCESSED_FEATURES = os.path.join(PROCESSED_DIR, "feature_matrix.parquet")

# Create directories
for d in [MODELS_DIR, METRICS_DIR, PROCESSED_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# Data Configuration
# =============================================================================
# Meter types
METER_TYPES = {
    0: "Electricity",
    1: "Chilled Water",
    2: "Steam",
    3: "Hot Water"
}

# Focus on electricity meter only
TARGET_METER = 0

# Building filter — Lodging/residential as hostel proxy
TARGET_PRIMARY_USE = "Lodging/residential"

# Sample buildings for dashboard demo (selected from Lodging/residential)
# These are picked for data quality and diverse patterns
SAMPLE_BUILDING_IDS = [6, 12, 27, 33, 34, 35, 36, 37, 49, 56, 57, 58]

# Train/Validation split ratio (chronological)
TRAIN_RATIO = 0.8

# Chunk size for reading large CSVs
CHUNK_SIZE = 500_000

# =============================================================================
# Feature Engineering
# =============================================================================
# Time features
TIME_FEATURES = [
    "hour", "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_holiday", "quarter", "semester_period"
]

# Lag features (in hours)
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]  # 168 = 1 week

# Rolling window sizes (in hours)
ROLLING_WINDOWS = [6, 12, 24, 48, 168]

# Weather features to use
WEATHER_FEATURES = [
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed"
]

# Building metadata features
BUILDING_FEATURES = ["square_feet", "year_built", "floor_count"]

# Target column
TARGET_COL = "meter_reading"

# =============================================================================
# Model Hyperparameters
# =============================================================================

# Sequence models lookback & forecast horizon
LOOKBACK_HOURS = 168    # 1 week
FORECAST_HOURS = 24     # 1 day ahead

# --- ARIMA ---
ARIMA_CONFIG = {
    "max_p": 3,
    "max_d": 2,
    "max_q": 3,
    "seasonal_period": 24,  # hourly data, daily seasonality
    "max_P": 2,
    "max_D": 1,
    "max_Q": 2,
}

# --- XGBoost ---
XGBOOST_CONFIG = {
    "n_estimators": 1000,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "random_state": 42,
}

# --- LightGBM ---
LIGHTGBM_CONFIG = {
    "n_estimators": 1000,
    "max_depth": 10,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "verbose": -1,
}

# --- Random Forest ---
RANDOM_FOREST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "oob_score": True,
    "random_state": 42,
    "n_jobs": -1,
}

# --- Deep Learning Common ---
DL_COMMON = {
    "batch_size": 128,
    "epochs": 35,
    "patience": 7,  # early stopping
    "learning_rate": 0.001,
    "validation_split": 0.2,
}

# --- ANN ---
ANN_CONFIG = {
    **DL_COMMON,
    "hidden_layers": [128, 64, 32],
    "dropout": 0.3,
    "batch_norm": True,
}

# --- CNN ---
CNN_CONFIG = {
    **DL_COMMON,
    "filters_1": 64,
    "filters_2": 32,
    "kernel_size": 3,
    "pool_size": 2,
    "dense_units": 64,
}

# --- CNN + BiLSTM ---
CNN_BILSTM_CONFIG = {
    **DL_COMMON,
    "cnn_filters_1": 64,
    "cnn_filters_2": 32,
    "kernel_size": 3,
    "bilstm_units": 64,
    "dense_units": 32,
    "dropout": 0.2,
}

# --- XGBoost + BiLSTM ---
XGB_BILSTM_CONFIG = {
    "xgb": XGBOOST_CONFIG,
    "bilstm_units": 64,
    "bilstm_dropout": 0.2,
    **DL_COMMON,
}

# --- Dual ANN ---
DUAL_ANN_CONFIG = {
    **DL_COMMON,
    "ann_a_layers": [64, 32],  # temporal features (including 168h lags/rolling)
    "ann_b_layers": [64, 32],  # contextual features
    "fusion_layers": [64, 32],
    "dropout": 0.3,
}

# =============================================================================
# Anomaly Detection
# =============================================================================
ANOMALY_CONFIG = {
    "isolation_forest": {
        "contamination": 0.05,
        "n_estimators": 200,
        "random_state": 42,
    },
    "z_score_threshold": 3.0,
    "autoencoder": {
        "encoding_dim": 16,
        "epochs": 50,
        "batch_size": 64,
    }
}

# =============================================================================
# Load Balancing
# =============================================================================
LOAD_BALANCE_CONFIG = {
    # Peak shaving
    "peak_percentile": 90,     # Define peak as top 10%
    "shift_percentage": 0.25,  # Shift 25% of peak load to off-peak

    # Solar simulation
    "panel_capacity_kw": 100,        # 100 kW rooftop solar capacity
    "panel_efficiency": 0.18,        # 18% efficiency
    "solar_peak_hours": (10, 15),    # 10am - 3pm
    "max_solar_irradiance": 1000,    # W/m² at peak

    # Appliance categories with typical power (kW) and probability
    "appliances": {
        "AC": {"power_kw": 1.5, "usage_hours": (13, 23), "probability": 0.7},
        "Heater": {"power_kw": 2.0, "usage_hours": (5, 10), "probability": 0.4},
        "Geyser": {"power_kw": 3.0, "usage_hours": (6, 9), "probability": 0.8},
        "Washing Machine": {"power_kw": 0.5, "usage_hours": (8, 12), "probability": 0.3},
        "Lighting": {"power_kw": 0.1, "usage_hours": (18, 24), "probability": 0.95},
        "Laptop/Phone Charging": {"power_kw": 0.08, "usage_hours": (20, 7), "probability": 0.9},
    },

    # Electricity cost (INR per kWh)
    "cost_per_kwh_peak": 8.0,
    "cost_per_kwh_offpeak": 4.5,
    "peak_hours": (9, 21),

    # Carbon emission factor (kg CO2 per kWh)
    "carbon_factor": 0.82,
}

# =============================================================================
# Dashboard
# =============================================================================
DASHBOARD_CONFIG = {
    "theme": "dark",
    "primary_color": "#00E676",     # Green accent
    "secondary_color": "#00BFA5",   # Teal
    "background_color": "#0A1628",  # Deep navy
    "card_color": "#122240",        # Slightly lighter navy
    "text_color": "#E0E0E0",        # Light gray
    "accent_gradient": ["#00E676", "#00BFA5", "#1DE9B6"],
    "danger_color": "#FF5252",
    "warning_color": "#FFD740",
    "info_color": "#448AFF",
}

# =============================================================================
# Model Names Registry (Focused 7 Models + Ensemble)
# =============================================================================
MODEL_NAMES = {
    # Standalone
    "xgboost": "XGBoost",
    "cnn": "CNN (168h)",
    "random_forest": "Random Forest",
    "ann": "ANN",
    # Hybrid
    "xgb_bilstm": "XGBoost + BiLSTM",
    "cnn_bilstm": "CNN + BiLSTM (168h)",
    "dual_ann": "ANN + ANN (Dual)",
    # Ensemble
    "meta_ensemble": "Meta-Ensemble",
}

STANDALONE_MODELS = ["xgboost", "cnn", "random_forest", "ann"]
HYBRID_MODELS = ["xgb_bilstm", "cnn_bilstm", "dual_ann"]
ALL_MODELS = STANDALONE_MODELS + HYBRID_MODELS
