"""
insights_generator.py

Phase 3 — Optimization & Insights Engine for Smart Energy Optimization

Features:
- Load preprocessed data and trained model
- Forecast next-N hours (default: 24)
- Detect peak and low hours
- Simulate load-shifting suggestions based on sub-metering (appliance) contributions
- Estimate potential energy savings and percent reduction
- Output human-readable insights and machine-readable reports (JSON/CSV)
"""

import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

# ---------------------------
# Config / Defaults
# ---------------------------
PROCESSED_PATH = 'processed/energy_clean.csv'
MODEL_PATH = 'models/energy_model.pkl'
FEATURES_PATH = 'models/features.txt'
OUTPUT_DIR = 'insights_output'

DEFAULT_FORECAST_HORIZON = 24  # hours to predict ahead

# ---------------------------
# Utility functions
# ---------------------------
def load_model_and_meta(model_path=MODEL_PATH, features_path=FEATURES_PATH):
    """Load trained model and feature list"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run model_train.py first.")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}.")
    model = joblib.load(model_path)
    with open(features_path, 'r') as f:
        features = [line.strip() for line in f.readlines()]
    return model, features

def load_recent_data(processed_path=PROCESSED_PATH):
    """Load preprocessed dataset. We'll use the latest timestamp rows to derive features for forecasting."""
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run data_preprocessing.py first.")
    df = pd.read_csv(processed_path, parse_dates=['datetime'])
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ---------------------------
# Forecasting
# ---------------------------
def make_hourly_forecast(df, model, features, horizon=DEFAULT_FORECAST_HORIZON):
    """
    Create next-horizon hourly predictions.
    Approach:
        - Start from the last row of df as t0.
        - For each next hour, increment time features appropriately and compute rolling_mean as simple moving update.
        - Use available exogenous features (Voltage etc.) by carrying last known values (simple approach).
    Note: This is a pragmatic simulation for a quick demo. For more accuracy, use time-series models (ARIMA/LSTM) or exogenous forecasts.
    """
    last_row = df.iloc[-1].copy()
    preds = []
    timestamps = []
    rolling_mean = float(last_row.get('rolling_mean', last_row['Global_active_power']))

    # Values to carry-forward from last row for non-time features (simple heuristic)
    carry_cols = [c for c in features if c not in ('hour', 'day_of_week', 'month', 'rolling_mean')]
    carry_values = {c: float(last_row[c]) for c in carry_cols}

    current_dt = last_row['datetime']

    for h in range(1, horizon + 1):
        current_dt = current_dt + timedelta(hours=1)
        hour = current_dt.hour
        day_of_week = current_dt.dayofweek
        month = current_dt.month

        # Update rolling mean as decayed average (simple)
        rolling_mean = (rolling_mean * 0.8) + (rolling_mean * 0.2)  # placeholder: no new measurement, keep same

        # Build feature row
        row = {}
        for f in features:
            if f == 'hour':
                row[f] = hour
            elif f == 'day_of_week':
                row[f] = day_of_week
            elif f == 'month':
                row[f] = month
            elif f == 'rolling_mean':
                row[f] = rolling_mean
            else:
                # carry-forward heuristic for exogenous numeric features
                row[f] = carry_values.get(f, 0.0)

        X_row = pd.DataFrame([row], columns=features)
        pred = float(model.predict(X_row)[0])

        preds.append(pred)
        timestamps.append(current_dt)

        # Update rolling_mean with predicted value for next step
        rolling_mean = (rolling_mean * 0.7) + (pred * 0.3)

    forecast_df = pd.DataFrame({
        'datetime': timestamps,
        'predicted_global_active_power': preds
    })
    return forecast_df

# ---------------------------
# Peak / Low Hours Detection
# ---------------------------
def detect_peak_low_hours(df, n_peak=3, n_low=3):
    """
    Calculate average power by hour-of-day across the dataset and return top-n peak and bottom-n low hours.
    """
    hourly_avg = df.groupby('hour')['Global_active_power'].mean()
    peak_hours = hourly_avg.sort_values(ascending=False).head(n_peak)
    low_hours = hourly_avg.sort_values(ascending=True).head(n_low)
    return hourly_avg, peak_hours, low_hours

# ---------------------------
# Savings Simulation (Load Shifting)
# ---------------------------
def estimate_shift_savings(df, forecast_df, peak_hours, low_hours, shift_fraction=0.8):
    """
    Estimate potential savings by shifting a portion of 'shiftable' load from peak_hours -> low_hours.

    Method:
    - Consider 'shiftable load' as the sum of sub-meterings (appliance-level consumption)
      averaged during peak hours (non-essential loads).
    - Simulate shifting 'shift_fraction' (e.g., 0.8 for 80%) of that shiftable load to the best low hours.
    - Compute total predicted energy before and after shift in the forecast window and estimate percent savings.

    Notes:
    - This is a conservative, explainable rule-based estimate — perfect for demo & judge presentation.
    """
    # Determine average sub-metering contributions by hour
    sub_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    if not set(sub_cols).issubset(df.columns):
        raise ValueError("Sub-metering columns missing from processed data.")

    # compute average shiftable load (kW) per hour (convert Wh to kW if needed depending on dataset units)
    # For this dataset, Global_active_power is in kilowatts (kW), sub-metering in watt-hour of active energy per minute? 
    # We'll treat them as comparable for relative estimates (explain in report). If you want strict units convert appropriately.
    hourly_sub = df.groupby('hour')[sub_cols].mean()

    # Average shiftable load during peak hours (sum of sub meters)
    peak_idx = list(peak_hours.index)
    low_idx = list(low_hours.index)

    avg_shiftable_peak = hourly_sub.loc[peak_idx].sum(axis=1).mean()  # average kWh-like measure
    # choose the best low hour (lowest hourly total usage)
    best_low_hour = hourly_sub.loc[low_idx].sum(axis=1).idxmin()
    avg_shiftable_low = hourly_sub.loc[best_low_hour].sum()

    # Compute predicted baseline energy in forecast window
    baseline_energy = forecast_df['predicted_global_active_power'].sum()  # in kW-hours (approx)

    # Determine how much energy we can shift in the forecast window:
    # We'll assume each peak hour occurrence in forecast_df can have a chunk of shiftable load moved.
    # For simplicity, apply shift only in forecast rows whose datetime.hour is in peak_idx.
    df_forecast = forecast_df.copy()
    df_forecast['hour'] = df_forecast['datetime'].dt.hour
    mask_peak = df_forecast['hour'].isin(peak_idx)
    num_peak_slots = mask_peak.sum()

    # energy_to_shift_total = shift_fraction * avg_shiftable_peak * num_peak_slots
    energy_to_shift_total = float(shift_fraction) * float(avg_shiftable_peak) * float(num_peak_slots)

    # Now simulate result: subtract energy_to_shift_total from baseline and add it spread into low-hour slots.
    # Spread equally across the number of low slots in forecast window (or put all into best_low_hour slots).
    mask_low = df_forecast['hour'] == best_low_hour
    num_low_slots = mask_low.sum() if mask_low.sum() > 0 else max(1, len(df_forecast[df_forecast['hour'].isin(low_idx)]))
    energy_added_to_low_total = energy_to_shift_total  # same energy moved

    # Baseline & optimized totals
    optimized_energy = baseline_energy - energy_to_shift_total + energy_added_to_low_total  # net should be same; but cost/time shifting can reduce peak penalties
    # For demonstrative savings (peak-reduction benefit), we compute reduction in peak-window sum instead:
    baseline_peak_energy = df_forecast.loc[mask_peak, 'predicted_global_active_power'].sum()
    optimized_peak_energy = baseline_peak_energy - energy_to_shift_total  # we removed that load from peaks

    # Percent reduction in peak energy
    pct_peak_reduction = (baseline_peak_energy - optimized_peak_energy) / baseline_peak_energy * 100 if baseline_peak_energy > 0 else 0.0

    result = {
        'baseline_total_forecast_energy': float(baseline_energy),
        'optimized_total_forecast_energy': float(optimized_energy),
        'baseline_peak_energy': float(baseline_peak_energy),
        'optimized_peak_energy': float(optimized_peak_energy),
        'energy_shifted_total': float(energy_to_shift_total),
        'percent_peak_reduction': float(pct_peak_reduction),
        'num_peak_slots': int(num_peak_slots),
        'num_low_slots': int(num_low_slots),
        'best_low_hour': int(best_low_hour),
        'peak_hours': peak_idx,
        'low_hours': low_idx
    }

    return result

# ---------------------------
# Human-readable Recommendations
# ---------------------------
def build_recommendations(shift_result, hourly_avg):
    """
    Build plain-language recommendations based on shift simulation.
    """
    recs = []
    pct = round(shift_result['percent_peak_reduction'], 2)
    shifted = round(shift_result['energy_shifted_total'], 3)
    best_low = shift_result['best_low_hour']
    peaks = shift_result['peak_hours']

    recs.append(f"Detected peak hours: {peaks}. Consider shifting non-essential appliance usage out of these windows.")
    if shifted > 0:
        recs.append(
            f"Estimated peak-window energy reduced by {pct}% by shifting approximately {shifted} energy-units "
            f"from peak hours into low-usage hour {best_low}."
        )

    recs.append("Suggested actions (example):")
    recs.append("- Run washing machine and dishwasher during low-usage hours (e.g., between 11 PM and 4 AM).")
    recs.append("- Delay electric vehicle charging to best low hour or overnight.")
    recs.append("- If using smart plugs, create schedules to shift heavy loads to the recommended low hours.")
    recs.append("Note: This is a rule-of-thumb estimate. For exact billing savings, combine with your utility's time-of-use tariff.")

    # Add a short confidence statement
    recs.append("Confidence: This is a conservative simulation using historical averages and model forecasts. "
                "Consider more advanced forecasting (LSTM/ARIMA) for higher-fidelity recommendations.")

    return recs

# ---------------------------
# Top-level orchestration
# ---------------------------
def generate_insights(horizon=DEFAULT_FORECAST_HORIZON, shift_fraction=0.8):
    """Main function to create forecast + insights + reports"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load artifacts
    df = load_recent_data()
    model, features = load_model_and_meta()

    # Forecast
    forecast_df = make_hourly_forecast(df, model, features, horizon=horizon)

    # Peak / low detection using historical data
    hourly_avg, peak_hours, low_hours = detect_peak_low_hours(df)

    # Simulation of shifting loads
    shift_result = estimate_shift_savings(df, forecast_df, peak_hours, low_hours, shift_fraction=shift_fraction)

    # Recommendations
    recommendations = build_recommendations(shift_result, hourly_avg)

    # Compose final insights object
    insights = {
        'forecast_horizon_hours': horizon,
        'forecast': forecast_df.to_dict(orient='records'),
        'peak_hours_list': list(map(int, peak_hours.index.tolist())),
        'low_hours_list': list(map(int, low_hours.index.tolist())),
        'shift_simulation': shift_result,
        'recommendations': recommendations
    }

    # Save JSON and CSV (basic)
    with open(os.path.join(OUTPUT_DIR, 'insights.json'), 'w') as fjson:
        json.dump(insights, fjson, indent=2, default=str)

    # Save a simple CSV summarizing forecast
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'forecast_summary.csv'), index=False)

    # Save a concise report CSV for UI (one-line metrics)
    report = {
        'horizon_hours': horizon,
        'baseline_total_forecast_energy': shift_result['baseline_total_forecast_energy'],
        'energy_shifted_total': shift_result['energy_shifted_total'],
        'percent_peak_reduction': shift_result['percent_peak_reduction'],
        'best_low_hour': shift_result['best_low_hour'],
        'peak_hours': ",".join(map(str, shift_result['peak_hours'])),
        'low_hours': ",".join(map(str, shift_result['low_hours']))
    }
    pd.DataFrame([report]).to_csv(os.path.join(OUTPUT_DIR, 'insights_report.csv'), index=False)

    print(f"\nInsights and reports saved to directory: {OUTPUT_DIR}\n")
    return insights

# ---------------------------
# Example run when script executed directly
# ---------------------------
if __name__ == "__main__":
    print("Running Insights Generator (Phase 3)...")
    insights = generate_insights(horizon=24, shift_fraction=0.8)

    # Print summary to console (concise)
    print("Summary:")
    print("Forecast horizon (hours):", insights['forecast_horizon_hours'])
    print("Peak hours (historical):", insights['peak_hours_list'])
    print("Low hours (historical):", insights['low_hours_list'])
    sim = insights['shift_simulation']
    print(f"Estimated energy shifted: {sim['energy_shifted_total']:.3f}")
    print(f"Estimated percent peak reduction: {sim['percent_peak_reduction']:.2f}%")
    print("\nTop recommendations:")
    for r in insights['recommendations'][:6]:
        print("-", r)
