"""
GreenCharge Load Balancing Module
==================================
Peak shaving, staggered appliance scheduling, solar integration simulation,
and multi-strategy load balancing.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LOAD_BALANCE_CONFIG, TARGET_COL


def compute_peak_stats(hourly_load):
    """Compute peak load statistics."""
    stats = {
        "peak_load_kw": float(hourly_load.max()),
        "avg_load_kw": float(hourly_load.mean()),
        "min_load_kw": float(hourly_load.min()),
        "peak_to_avg_ratio": float(hourly_load.max() / (hourly_load.mean() + 1e-8)),
        "load_factor": float(hourly_load.mean() / (hourly_load.max() + 1e-8)),
        "total_energy_kwh": float(hourly_load.sum()),
    }
    return stats


# =========================================================================
# Peak Shaving Simulation
# =========================================================================
def simulate_peak_shaving(df, config=None):
    """
    Simulate peak shaving by shifting a percentage of peak-hour load to off-peak.
    
    Returns:
        dict with before/after loads, savings
    """
    if config is None:
        config = LOAD_BALANCE_CONFIG

    print("\n[Load Balancing] Simulating Peak Shaving...")

    hourly_load = df.groupby(df["timestamp"].dt.hour)[TARGET_COL].mean()
    
    percentile_threshold = np.percentile(hourly_load.values, config["peak_percentile"])
    shift_pct = config["shift_percentage"]

    original_load = hourly_load.copy()
    shaved_load = hourly_load.copy()

    # Identify peak and off-peak hours
    peak_hours = hourly_load[hourly_load >= percentile_threshold].index.tolist()
    off_peak_hours = hourly_load[hourly_load < percentile_threshold].index.tolist()

    # Calculate total shifted energy
    total_shifted = 0
    for h in peak_hours:
        shift_amount = shaved_load[h] * shift_pct
        shaved_load[h] -= shift_amount
        total_shifted += shift_amount

    # Distribute shifted energy to off-peak hours
    if off_peak_hours:
        per_hour_addition = total_shifted / len(off_peak_hours)
        for h in off_peak_hours:
            shaved_load[h] += per_hour_addition

    # Compute savings
    before_stats = compute_peak_stats(original_load)
    after_stats = compute_peak_stats(shaved_load)

    peak_reduction = (before_stats["peak_load_kw"] - after_stats["peak_load_kw"]) / before_stats["peak_load_kw"] * 100

    # Cost savings
    peak_h = config["peak_hours"]
    cost_before = sum(original_load[h] * config["cost_per_kwh_peak"] if peak_h[0] <= h < peak_h[1]
                      else original_load[h] * config["cost_per_kwh_offpeak"] for h in range(24))
    cost_after = sum(shaved_load[h] * config["cost_per_kwh_peak"] if peak_h[0] <= h < peak_h[1]
                     else shaved_load[h] * config["cost_per_kwh_offpeak"] for h in range(24))
    cost_savings = cost_before - cost_after

    print(f"  Peak reduction: {peak_reduction:.1f}%")
    print(f"  Daily cost savings: INR {cost_savings:.2f}")

    return {
        "original_load": original_load.to_dict(),
        "shaved_load": shaved_load.to_dict(),
        "peak_hours": peak_hours,
        "off_peak_hours": off_peak_hours,
        "peak_reduction_pct": peak_reduction,
        "cost_savings_daily": cost_savings,
        "before_stats": before_stats,
        "after_stats": after_stats,
    }


# =========================================================================
# Solar Integration Simulation
# =========================================================================
def simulate_solar_generation(df, config=None):
    """
    Simulate rooftop solar panel output based on weather data.
    
    Returns:
        dict with solar output, net load, savings
    """
    if config is None:
        config = LOAD_BALANCE_CONFIG

    print("\n[Load Balancing] Simulating Solar Integration...")

    hourly_load = df.groupby(df["timestamp"].dt.hour)[TARGET_COL].mean()

    # Simulate solar generation using bell curve peaking at noon
    solar_output = {}
    for hour in range(24):
        if config["solar_peak_hours"][0] <= hour <= config["solar_peak_hours"][1]:
            # Bell curve centered at solar noon (12:30)
            solar_noon = 12.5
            sigma = 2.5
            factor = np.exp(-((hour - solar_noon) ** 2) / (2 * sigma ** 2))

            # Adjust by cloud coverage if available
            if "cloud_coverage" in df.columns:
                avg_cloud = df[df["timestamp"].dt.hour == hour]["cloud_coverage"].mean()
                cloud_factor = max(0.2, 1 - (avg_cloud / 10.0))
            else:
                cloud_factor = 0.85  # Default 85% clear

            solar_kw = config["panel_capacity_kw"] * config["panel_efficiency"] * factor * cloud_factor
        else:
            solar_kw = 0.0
        solar_output[hour] = solar_kw

    # Net load = grid demand - solar
    net_load = {}
    for hour in range(24):
        net_load[hour] = max(0, hourly_load.get(hour, 0) - solar_output[hour])

    total_solar = sum(solar_output.values())
    total_original = sum(hourly_load.values)
    total_net = sum(net_load.values())
    solar_savings_pct = (total_original - total_net) / (total_original + 1e-8) * 100

    # Cost savings from solar
    cost_savings = total_solar * config["cost_per_kwh_offpeak"]

    # Carbon reduction
    carbon_reduced = total_solar * config["carbon_factor"]

    print(f"  Total solar generation: {total_solar:.1f} kWh/day")
    print(f"  Energy savings: {solar_savings_pct:.1f}%")
    print(f"  Daily cost savings: INR {cost_savings:.2f}")
    print(f"  Carbon reduced: {carbon_reduced:.1f} kg CO2/day")

    return {
        "solar_output": solar_output,
        "net_load": net_load,
        "original_load": hourly_load.to_dict(),
        "total_solar_kwh": total_solar,
        "solar_savings_pct": solar_savings_pct,
        "cost_savings_daily": cost_savings,
        "carbon_reduced_kg": carbon_reduced,
    }


# =========================================================================
# Staggered Appliance Scheduling
# =========================================================================
def simulate_staggered_scheduling(n_rooms=200, config=None):
    """
    Simulate staggered appliance usage across hostel rooms.
    
    Returns:
        dict with original and staggered load profiles
    """
    if config is None:
        config = LOAD_BALANCE_CONFIG

    print(f"\n[Load Balancing] Simulating Staggered Scheduling ({n_rooms} rooms)...")

    appliances = config["appliances"]
    np.random.seed(42)

    # Original: all appliances used at their typical hours
    original_profile = np.zeros(24)
    staggered_profile = np.zeros(24)

    for app_name, app in appliances.items():
        start, end = app["usage_hours"]
        n_users = int(n_rooms * app["probability"])

        # Original: all use at same time
        if start < end:
            hours = list(range(start, end))
        else:
            hours = list(range(start, 24)) + list(range(0, end))

        for h in hours:
            original_profile[h] += n_users * app["power_kw"] / len(hours)

        # Staggered: distribute usage across wider window with offsets
        stagger_offset = max(1, len(hours) // 3)
        groups = 3
        group_size = n_users // groups

        for g in range(groups):
            offset = g * stagger_offset
            staggered_hours = [(h + offset) % 24 for h in hours]
            for h in staggered_hours:
                staggered_profile[h] += group_size * app["power_kw"] / len(hours)

    peak_reduction = (original_profile.max() - staggered_profile.max()) / original_profile.max() * 100
    load_factor_before = original_profile.mean() / (original_profile.max() + 1e-8)
    load_factor_after = staggered_profile.mean() / (staggered_profile.max() + 1e-8)

    print(f"  Original peak: {original_profile.max():.1f} kW")
    print(f"  Staggered peak: {staggered_profile.max():.1f} kW")
    print(f"  Peak reduction: {peak_reduction:.1f}%")
    print(f"  Load factor improvement: {load_factor_before:.3f} -> {load_factor_after:.3f}")

    return {
        "original_profile": original_profile.tolist(),
        "staggered_profile": staggered_profile.tolist(),
        "peak_reduction_pct": peak_reduction,
        "load_factor_before": load_factor_before,
        "load_factor_after": load_factor_after,
        "appliances": {k: v for k, v in appliances.items()},
    }


# =========================================================================
# Combined Load Balancing Strategies
# =========================================================================
def strategy_a_solar_timeshift(df, config=None):
    """Strategy A: Time-shift heavy loads to solar-peak hours (10am–3pm)."""
    print("\n[Strategy A] Time-shifting loads to solar-peak hours...")
    if config is None:
        config = LOAD_BALANCE_CONFIG

    hourly_load = df.groupby(df["timestamp"].dt.hour)[TARGET_COL].mean()
    solar_hours = list(range(config["solar_peak_hours"][0], config["solar_peak_hours"][1] + 1))
    non_solar_hours = [h for h in range(24) if h not in solar_hours]

    shifted_load = hourly_load.copy()
    total_shifted = 0

    for h in non_solar_hours:
        if hourly_load[h] > hourly_load.mean():
            shift_amount = (hourly_load[h] - hourly_load.mean()) * 0.3
            shifted_load[h] -= shift_amount
            total_shifted += shift_amount

    if solar_hours:
        per_hour_add = total_shifted / len(solar_hours)
        for h in solar_hours:
            shifted_load[h] += per_hour_add

    return {
        "original": hourly_load.to_dict(),
        "optimized": shifted_load.to_dict(),
        "total_shifted_kwh": total_shifted,
    }


def strategy_b_building_distribution(df, config=None):
    """Strategy B: Distribute loads evenly across buildings in a site."""
    print("\n[Strategy B] Distributing loads across buildings...")

    building_loads = df.groupby("building_id")[TARGET_COL].mean()
    avg_load = building_loads.mean()

    original = building_loads.to_dict()
    balanced = {bid: avg_load for bid in building_loads.index}

    imbalance_before = building_loads.std() / (building_loads.mean() + 1e-8)
    imbalance_after = 0.0  # Perfect balance

    return {
        "original": original,
        "balanced": balanced,
        "imbalance_before": float(imbalance_before),
        "imbalance_after": imbalance_after,
    }


def strategy_c_combined(df, config=None):
    """Strategy C: Combined solar + staggering + peak shaving."""
    print("\n[Strategy C] Combined optimization...")

    peak_results = simulate_peak_shaving(df, config)
    solar_results = simulate_solar_generation(df, config)
    stagger_results = simulate_staggered_scheduling(config=config)

    total_savings = (
        peak_results["cost_savings_daily"] +
        solar_results["cost_savings_daily"]
    )

    total_peak_reduction = (
        peak_results["peak_reduction_pct"] +
        stagger_results["peak_reduction_pct"]
    ) / 2

    return {
        "peak_shaving": peak_results,
        "solar": solar_results,
        "staggering": stagger_results,
        "total_daily_savings": total_savings,
        "total_peak_reduction_pct": total_peak_reduction,
        "carbon_reduced_kg": solar_results["carbon_reduced_kg"],
    }


def run_load_balancing(df, config=None):
    """
    Run all load balancing simulations.
    
    Returns:
        dict with results from all strategies
    """
    print("\n" + "=" * 60)
    print("  LOAD BALANCING & OPTIMIZATION")
    print("=" * 60)

    results = {
        "peak_shaving": simulate_peak_shaving(df, config),
        "solar_integration": simulate_solar_generation(df, config),
        "staggered_scheduling": simulate_staggered_scheduling(config=config),
        "strategy_a": strategy_a_solar_timeshift(df, config),
        "strategy_b": strategy_b_building_distribution(df, config),
        "strategy_c": strategy_c_combined(df, config),
    }

    print("\n" + "=" * 60)
    print("  LOAD BALANCING SUMMARY")
    print("=" * 60)
    print(f"  Peak Shaving: {results['peak_shaving']['peak_reduction_pct']:.1f}% peak reduction")
    print(f"  Solar Integration: {results['solar_integration']['solar_savings_pct']:.1f}% energy savings")
    print(f"  Staggered Scheduling: {results['staggered_scheduling']['peak_reduction_pct']:.1f}% peak reduction")
    print(f"  Combined Strategy: INR {results['strategy_c']['total_daily_savings']:.2f}/day savings")
    print(f"  Carbon Reduction: {results['strategy_c']['carbon_reduced_kg']:.1f} kg CO2/day")

    return results
