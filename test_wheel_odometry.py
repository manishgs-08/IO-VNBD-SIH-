"""
Wheel Odometry GPS-Outage Simulation — V-Vw4.csv
=================================================
Compares the Corrected DR (using GPS-derived dataset velocity) 
with Wheel Odometry DR (using wheel-speed-derived velocity).

Both use the CORRECTED yaw-rate integration and coordinate math.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CSV_PATH = os.path.join(
    "IO-VNBD-SIH-",
    "Synchronised V abd S datasets",
    "Uncategorised IOVNB Dataset",
    "V-Dataset",
    "V-Vw4.csv",
)
PLOT_DIR = os.path.join("outputs", "plots", "wheel_odometry")
METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

R_EARTH = 6_371_000
SAMPLE_RATE = 10
OUTAGE_DURATIONS_S = [10, 30, 60, 120, 300]
MIN_VELOCITY_KMH = 10.0
MIN_WINDOW_SPACING_S = 60
NUM_WINDOWS = 50
EFFECTIVE_RADIUS = 0.27881  # derived from audit

def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    x = R_EARTH * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = R_EARTH * (lat_rad - lat0_rad)
    return x, y

def run_dr_window(timestamps, velocity_ms, yaw_rate_rps, ref_x, ref_y, heading0_rad, start_idx, n_samples):
    end_idx = start_idx + n_samples
    est_x = np.zeros(n_samples)
    est_y = np.zeros(n_samples)
    est_h = np.zeros(n_samples)

    est_x[0] = ref_x[start_idx]
    est_y[0] = ref_y[start_idx]
    est_h[0] = heading0_rad

    for k in range(1, n_samples):
        i = start_idx + k
        dt = timestamps[i] - timestamps[i - 1]
        
        # CORRECTED YAW RATE
        corrected_yaw_rate = -yaw_rate_rps[i - 1]
        est_h[k] = est_h[k - 1] + corrected_yaw_rate * dt
        
        # CORRECTED NAV MATH
        est_x[k] = est_x[k - 1] + velocity_ms[i - 1] * np.sin(est_h[k - 1]) * dt
        est_y[k] = est_y[k - 1] + velocity_ms[i - 1] * np.cos(est_h[k - 1]) * dt

    ref_x_win = ref_x[start_idx:end_idx]
    ref_y_win = ref_y[start_idx:end_idx]
    errors = np.sqrt((est_x - ref_x_win) ** 2 + (est_y - ref_y_win) ** 2)

    return {
        "est_x": est_x,
        "est_y": est_y,
        "est_heading": est_h,
        "ref_x": ref_x_win,
        "ref_y": ref_y_win,
        "errors": errors,
        "final_error": errors[-1],
    }

def select_window_starts(velocity_kmh, n_total, max_outage_samples, min_vel, min_spacing_samples, num_windows):
    max_start = n_total - max_outage_samples
    candidates = np.where((velocity_kmh[:max_start] >= min_vel))[0]
    step = max(1, len(candidates) // num_windows)
    selected = candidates[::step]
    
    filtered = [selected[0]]
    for idx in selected[1:]:
        if idx - filtered[-1] >= min_spacing_samples:
            filtered.append(idx)
    return np.array(filtered)

def main():
    print("Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    n_total = len(df)
    
    timestamps    = df["Time Since Start of Day (seconds)"].values
    velocity_kmh  = df["Velocity (km/hr)"].values
    velocity_dataset_ms = velocity_kmh / 3.6
    
    ws_fl = df["Wheel Speed Front Left (rad/sec)"].values
    ws_fr = df["Wheel Speed Front Right (rad/sec)"].values
    ws_rl = df["Wheel Speed Rear Left (rad/sec)"].values
    ws_rr = df["Wheel Speed Rear Right (rad/sec)"].values
    mean_ws = np.mean([ws_fl, ws_fr, ws_rl, ws_rr], axis=0)
    velocity_wheel_ms = mean_ws * EFFECTIVE_RADIUS
    
    yaw_rate_rps  = np.radians(df["Yaw Rate (deg/sec)"].values)
    lat_deg       = df["Latitude (degrees)"].values
    lon_deg       = df["Longitude (degrees)"].values
    heading_deg   = df["Heading (degrees)"].values

    lat0, lon0 = lat_deg[0], lon_deg[0]
    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat0, lon0)

    max_outage_samples = max(OUTAGE_DURATIONS_S) * SAMPLE_RATE
    min_spacing_samples = MIN_WINDOW_SPACING_S * SAMPLE_RATE

    starts = select_window_starts(velocity_kmh, n_total, max_outage_samples, MIN_VELOCITY_KMH, min_spacing_samples, NUM_WINDOWS)
    
    all_results = []
    
    for dur_s in OUTAGE_DURATIONS_S:
        n_samples = dur_s * SAMPLE_RATE
        errs_base = []
        errs_wheel = []
        
        for si in starts:
            heading0 = np.radians(heading_deg[si])
            
            # Baseline: Dataset Velocity
            res_base = run_dr_window(timestamps, velocity_dataset_ms, yaw_rate_rps, ref_x, ref_y, heading0, si, n_samples)
            errs_base.append(res_base["final_error"])
            
            # New: Wheel Odometry Velocity
            res_wheel = run_dr_window(timestamps, velocity_wheel_ms, yaw_rate_rps, ref_x, ref_y, heading0, si, n_samples)
            errs_wheel.append(res_wheel["final_error"])
            
        errs_base = np.array(errs_base)
        errs_wheel = np.array(errs_wheel)
        
        mean_base = np.mean(errs_base)
        mean_wheel = np.mean(errs_wheel)
        imp = mean_base / mean_wheel if mean_wheel > 0 else 1.0
        
        row = {
            "outage_duration_s": dur_s,
            "base_mean": mean_base,
            "base_median": np.median(errs_base),
            "base_rmse": np.sqrt(np.mean(errs_base**2)),
            "base_max": np.max(errs_base),
            "wheel_mean": mean_wheel,
            "wheel_median": np.median(errs_wheel),
            "wheel_rmse": np.sqrt(np.mean(errs_wheel**2)),
            "wheel_max": np.max(errs_wheel),
            "improvement_factor": imp
        }
        all_results.append(row)
        print(f"Outage {dur_s:>3d}s | Base Mean: {mean_base:>6.1f}m | Wheel Mean: {mean_wheel:>6.1f}m | Imp: {imp:.2f}x")

    metrics_df = pd.DataFrame(all_results)
    metrics_df.to_csv(os.path.join(METRICS_DIR, "wheel_odometry_comparison.csv"), index=False)

if __name__ == "__main__":
    main()
