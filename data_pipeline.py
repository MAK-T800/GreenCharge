"""
GreenCharge Data Pipeline
=========================
Ingests raw CSVs, filters for hostel buildings, engineers features,
and outputs processed parquet files for model training.
"""

import os
import warnings
import numpy as np
import pandas as pd
import holidays

from config import (
    TRAIN_CSV, BUILDING_META_CSV, WEATHER_TRAIN_CSV,
    PROCESSED_TRAIN, PROCESSED_VAL, PROCESSED_FEATURES, PROCESSED_DIR,
    TARGET_METER, TARGET_PRIMARY_USE, SAMPLE_BUILDING_IDS,
    TRAIN_RATIO, CHUNK_SIZE,
    TIME_FEATURES, LAG_HOURS, ROLLING_WINDOWS,
    WEATHER_FEATURES, BUILDING_FEATURES, TARGET_COL
)

warnings.filterwarnings("ignore")


def load_building_metadata():
    """Load building metadata and filter for Lodging/residential (hostel proxy)."""
    print("[Pipeline] Loading building metadata...")
    meta = pd.read_csv(BUILDING_META_CSV)
    hostel_meta = meta[meta["primary_use"] == TARGET_PRIMARY_USE].copy()
    print(f"  Found {len(hostel_meta)} Lodging/residential buildings (hostel proxy)")

    # Fill missing values
    hostel_meta["year_built"] = hostel_meta["year_built"].fillna(
        hostel_meta["year_built"].median()
    )
    hostel_meta["floor_count"] = hostel_meta["floor_count"].fillna(
        hostel_meta["floor_count"].median()
    )

    return hostel_meta


def load_weather_data():
    """Load weather training data."""
    print("[Pipeline] Loading weather data...")
    weather = pd.read_csv(WEATHER_TRAIN_CSV, parse_dates=["timestamp"])

    # Forward-fill missing weather values per site
    weather = weather.sort_values(["site_id", "timestamp"])
    for col in WEATHER_FEATURES:
        if col in weather.columns:
            weather[col] = weather.groupby("site_id")[col].transform(
                lambda x: x.ffill().bfill()
            )

    # Fill any remaining NaN with 0
    weather[WEATHER_FEATURES] = weather[WEATHER_FEATURES].fillna(0)
    print(f"  Weather data: {len(weather)} rows, {weather['site_id'].nunique()} sites")
    return weather


def load_train_data(hostel_building_ids, sample_ids=None):
    """
    Load training data in chunks, filtering for hostel building IDs
    and electricity meter only.
    """
    print("[Pipeline] Loading training data (chunked)...")
    if sample_ids is not None:
        target_ids = set(sample_ids)
    else:
        target_ids = set(hostel_building_ids)

    chunks = []
    total_rows = 0
    kept_rows = 0

    for chunk in pd.read_csv(TRAIN_CSV, chunksize=CHUNK_SIZE, parse_dates=["timestamp"]):
        total_rows += len(chunk)
        # Filter for target buildings and electricity meter
        filtered = chunk[
            (chunk["building_id"].isin(target_ids)) &
            (chunk["meter"] == TARGET_METER)
        ].copy()
        if len(filtered) > 0:
            chunks.append(filtered)
            kept_rows += len(filtered)

        if total_rows % 5_000_000 == 0:
            print(f"  Processed {total_rows:,} rows, kept {kept_rows:,}...")

    if not chunks:
        raise ValueError("No data found for target buildings!")

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total: processed {total_rows:,} rows, kept {kept_rows:,} ({100*kept_rows/total_rows:.1f}%)")

    # Remove zero/negative readings
    before = len(df)
    df = df[df[TARGET_COL] > 0].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} zero/negative readings")

    return df


def add_time_features(df):
    """Add time-based features to the dataframe."""
    print("[Pipeline] Engineering time features...")
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["quarter"] = ts.dt.quarter
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Holiday detection (US holidays as proxy)
    us_holidays = holidays.US(years=[2016, 2017])
    df["is_holiday"] = ts.dt.date.map(lambda d: 1 if d in us_holidays else 0)

    # Semester period: approximate academic calendar
    # Fall: Aug-Dec (8-12), Spring: Jan-May (1-5), Summer: Jun-Jul (6-7)
    month = df["month"]
    df["semester_period"] = np.where(
        month.between(1, 5), 1,  # Spring
        np.where(month.between(8, 12), 2, 0)  # Fall / Summer
    )

    # Cyclical encoding for hour and day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(df):
    """Add lag features per building."""
    print("[Pipeline] Engineering lag features...")
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    for lag in LAG_HOURS:
        col_name = f"lag_{lag}h"
        df[col_name] = df.groupby("building_id")[TARGET_COL].shift(lag)

    return df


def add_rolling_features(df):
    """Add rolling statistics per building."""
    print("[Pipeline] Engineering rolling features...")
    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    for window in ROLLING_WINDOWS:
        grp = df.groupby("building_id")[TARGET_COL]
        df[f"rolling_mean_{window}h"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    # Fill NaN from rolling/lag features
    lag_roll_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("rolling_")]
    df[lag_roll_cols] = df[lag_roll_cols].fillna(0)

    return df


def merge_all(train_df, weather_df, meta_df):
    """Merge training data with weather and building metadata."""
    print("[Pipeline] Merging datasets...")

    # Merge building metadata
    df = train_df.merge(
        meta_df[["building_id", "site_id", "square_feet", "year_built", "floor_count"]],
        on="building_id",
        how="left"
    )

    # Merge weather data
    df = df.merge(
        weather_df,
        on=["site_id", "timestamp"],
        how="left"
    )

    # Fill any remaining NaNs in weather
    for col in WEATHER_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill building features
    for col in BUILDING_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"  Merged dataset: {len(df)} rows, {df.shape[1]} columns")
    return df


def normalize_target(df):
    """Log-transform the target for better distribution."""
    print("[Pipeline] Log-transforming target variable...")
    df["meter_reading_log"] = np.log1p(df[TARGET_COL])
    return df


def split_train_val(df):
    """Chronological train/validation split."""
    print("[Pipeline] Splitting train/validation...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"  Val:   {len(val_df)} rows ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")

    return train_df, val_df


def get_feature_columns(df):
    """Return list of feature columns (excluding target, timestamp, IDs)."""
    exclude = [TARGET_COL, "meter_reading_log", "timestamp", "building_id", "meter", "site_id", "row_id", "primary_use"]
    return [c for c in df.columns if c not in exclude]


def run_pipeline(sample_ids=None):
    """
    Execute the full data pipeline.

    Args:
        sample_ids: Optional list of building IDs to process. 
                    If None, uses SAMPLE_BUILDING_IDS from config.
    """
    print("=" * 60)
    print("GreenCharge Data Pipeline")
    print("=" * 60)

    if sample_ids is None:
        sample_ids = SAMPLE_BUILDING_IDS

    # Step 1: Load metadata
    meta_df = load_building_metadata()

    # Step 2: Load weather
    weather_df = load_weather_data()

    # Step 3: Load training data (filtered)
    train_df = load_train_data(
        hostel_building_ids=meta_df["building_id"].tolist(),
        sample_ids=sample_ids
    )

    # Step 4: Merge datasets
    df = merge_all(train_df, weather_df, meta_df)

    # Step 5: Feature engineering
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = normalize_target(df)

    # Step 6: Split
    train_data, val_data = split_train_val(df)

    # Step 7: Save
    print("[Pipeline] Saving processed data...")
    train_data.to_parquet(PROCESSED_TRAIN, index=False)
    val_data.to_parquet(PROCESSED_VAL, index=False)

    # Save full feature matrix
    df.to_parquet(PROCESSED_FEATURES, index=False)

    feature_cols = get_feature_columns(df)
    print(f"\n[Pipeline] Complete!")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols[:10]}...")
    print(f"  Saved to: {PROCESSED_DIR}")

    return train_data, val_data, feature_cols


if __name__ == "__main__":
    train_data, val_data, feature_cols = run_pipeline()
    print(f"\nTrain shape: {train_data.shape}")
    print(f"Val shape:   {val_data.shape}")
    print(f"Features:    {len(feature_cols)}")
