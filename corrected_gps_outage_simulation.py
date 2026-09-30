"""
Corrected GPS-Outage Simulation — V-Vw4.csv
============================================
Evaluates 2D dead-reckoning accuracy under simulated GPS outages
of varying duration (10 s – 300 s) at multiple locations.

MATH & SIGN FIXES APPLIED (Based on Yaw-Rate Audit):
1. Navigation heading math: x_east = v * sin(H), y_north = v * cos(H)
2. Yaw-Rate sign: H_next = H_current + (-yaw_rate_dps) * dt
3. NO BIAS SUBTRACTION (as the -0.23 deg/s mean was proven false).
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
PLOT_DIR = os.path.join("outputs", "plots", "corrected_gps_outage")
METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

R_EARTH = 6_371_000  # metres
SAMPLE_RATE = 10     # Hz
OUTAGE_DURATIONS_S = [10, 30, 60, 120, 300]
MIN_VELOCITY_KMH = 10.0
MIN_WINDOW_SPACING_S = 60
NUM_WINDOWS = 50

# ──────────────────────────────────────────────
# Helper: GPS → local Cartesian (equirectangular)
# ──────────────────────────────────────────────
def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    # x = East, y = North
    x = R_EARTH * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = R_EARTH * (lat_rad - lat0_rad)
    return x, y

# ──────────────────────────────────────────────
# Corrected DR logic over a single outage window
# ──────────────────────────────────────────────
def run_corrected_dr_window(timestamps, velocity_ms, yaw_rate_rps,
                            ref_x, ref_y, heading0_rad, start_idx, n_samples):
    end_idx = start_idx + n_samples
    est_x = np.zeros(n_samples)
    est_y = np.zeros(n_samples)
    est_h = np.zeros(n_samples)

    # Initialise from GPS at start of outage
    est_x[0] = ref_x[start_idx]
    est_y[0] = ref_y[start_idx]
    est_h[0] = heading0_rad

    for k in range(1, n_samples):
        i = start_idx + k
        dt = timestamps[i] - timestamps[i - 1]
        
        # 1. Yaw Rate Sign Fix: Negate raw CCW yaw rate to match CW heading
        corrected_yaw_rate = -yaw_rate_rps[i - 1]
        est_h[k] = est_h[k - 1] + corrected_yaw_rate * dt
        
        # 2. Navigation Coordinate Fix: East=sin, North=cos
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

# ──────────────────────────────────────────────
# Select outage window start indices
# ──────────────────────────────────────────────
def select_window_starts(velocity_kmh, n_total, max_outage_samples,
                         min_vel, min_spacing_samples, num_windows):
    max_start = n_total - max_outage_samples
    candidates = np.where((velocity_kmh[:max_start] >= min_vel))[0]
    
    if len(candidates) == 0:
        raise ValueError("No valid window start positions found.")
        
    step = max(1, len(candidates) // num_windows)
    selected = candidates[::step]
    
    filtered = [selected[0]]
    for idx in selected[1:]:
        if idx - filtered[-1] >= min_spacing_samples:
            filtered.append(idx)
            
    return np.array(filtered)

# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    n_total = len(df)
    
    timestamps    = df["Time Since Start of Day (seconds)"].values
    velocity_kmh  = df["Velocity (km/hr)"].values
    velocity_ms   = velocity_kmh / 3.6
    yaw_rate_dps  = df["Yaw Rate (deg/sec)"].values
    yaw_rate_rps  = np.radians(yaw_rate_dps)
    lat_deg       = df["Latitude (degrees)"].values
    lon_deg       = df["Longitude (degrees)"].values
    heading_deg   = df["Heading (degrees)"].values

    lat0, lon0 = lat_deg[0], lon_deg[0]
    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat0, lon0)

    max_outage_samples = max(OUTAGE_DURATIONS_S) * SAMPLE_RATE
    min_spacing_samples = MIN_WINDOW_SPACING_S * SAMPLE_RATE

    starts = select_window_starts(
        velocity_kmh, n_total, max_outage_samples,
        MIN_VELOCITY_KMH, min_spacing_samples, NUM_WINDOWS,
    )
    print(f"Selected {len(starts)} outage start locations")

    all_results = []
    representative_windows = {}

    for dur_s in OUTAGE_DURATIONS_S:
        n_samples = dur_s * SAMPLE_RATE
        errors_final = []
        window_results_for_dur = []

        for si in starts:
            heading0 = np.radians(heading_deg[si])
            res = run_corrected_dr_window(
                timestamps, velocity_ms, yaw_rate_rps,
                ref_x, ref_y, heading0, si, n_samples,
            )
            errors_final.append(res["final_error"])
            window_results_for_dur.append((si, res))

        errors_final = np.array(errors_final)
        
        row = {
            "outage_duration_s": dur_s,
            "mean_error_m":      np.mean(errors_final),
            "median_error_m":    np.median(errors_final),
            "rmse_m":            np.sqrt(np.mean(errors_final ** 2)),
            "max_error_m":       np.max(errors_final),
            "final_error_mean_m": np.mean(errors_final),
        }
        all_results.append(row)

        sorted_idx = np.argsort(errors_final)
        picks = {
            "best":   sorted_idx[0],
            "median": sorted_idx[len(sorted_idx) // 2],
            "worst":  sorted_idx[-1],
        }
        representative_windows[dur_s] = {
            label: window_results_for_dur[i] for label, i in picks.items()
        }
        print(f"  Outage {dur_s:>3d}s  |  mean={row['mean_error_m']:>8.1f} m  "
              f"median={row['median_error_m']:>8.1f} m  "
              f"rmse={row['rmse_m']:>8.1f} m")

    metrics_df = pd.DataFrame(all_results)
    
    csv_path = os.path.join(METRICS_DIR, "corrected_gps_outage_errors.csv")
    metrics_df.to_csv(csv_path, index=False)
    
    # ── Plots ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(metrics_df["outage_duration_s"], metrics_df["mean_error_m"], "o-", color="crimson", linewidth=2, label="Mean final error")
    ax.plot(metrics_df["outage_duration_s"], metrics_df["median_error_m"], "s--", color="steelblue", linewidth=2, label="Median final error")
    ax.plot(metrics_df["outage_duration_s"], metrics_df["max_error_m"], "^:", color="darkorange", linewidth=2, label="Max final error")
    ax.set_xlabel("GPS Outage Duration (s)", fontsize=12)
    ax.set_ylabel("Position Error at End of Outage (m)", fontsize=12)
    ax.set_title("CORRECTED Dead-Reckoning Error vs GPS Outage Duration", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "error_vs_outage_duration.png"), dpi=150)
    plt.close(fig)

    show_durations = [30, 120, 300]
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Representative CORRECTED GPS-Outage Windows — DR vs GPS Reference", fontsize=14, y=0.98)
    for row_i, dur_s in enumerate(show_durations):
        for col_i, label in enumerate(["best", "median", "worst"]):
            ax = axes[row_i, col_i]
            si, res = representative_windows[dur_s][label]
            ax.plot(res["ref_x"], res["ref_y"], linewidth=1.5, color="steelblue", label="GPS Reference")
            ax.plot(res["est_x"], res["est_y"], linewidth=1.5, color="crimson", label="Dead Reckoning")
            ax.plot(res["ref_x"][0], res["ref_y"][0], "go", markersize=7)
            ax.plot(res["ref_x"][-1], res["ref_y"][-1], "bs", markersize=7)
            ax.plot(res["est_x"][-1], res["est_y"][-1], "r^", markersize=7)
            ax.set_title(f"{dur_s}s outage — {label} (err={res['final_error']:.1f} m)", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if row_i == 0 and col_i == 0: ax.legend(fontsize=8, loc="best")
            if col_i == 0: ax.set_ylabel("Y — North (m)", fontsize=9)
            if row_i == 2: ax.set_xlabel("X — East (m)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOT_DIR, "representative_outage_trajectories.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = []
    box_labels = []
    for dur_s in OUTAGE_DURATIONS_S:
        finals = []
        n_samples = dur_s * SAMPLE_RATE
        for si in starts:
            heading0 = np.radians(heading_deg[si])
            res = run_corrected_dr_window(timestamps, velocity_ms, yaw_rate_rps, ref_x, ref_y, heading0, si, n_samples)
            finals.append(res["final_error"])
        box_data.append(finals)
        box_labels.append(f"{dur_s}s")
    ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, boxprops=dict(facecolor="lightcoral", alpha=0.7), medianprops=dict(color="black", linewidth=2))
    ax.set_xlabel("GPS Outage Duration", fontsize=12)
    ax.set_ylabel("Final Position Error (m)", fontsize=12)
    ax.set_title("Distribution of CORRECTED DR Final Error Across Outage Windows", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "error_distribution_boxplot.png"), dpi=150)
    plt.close(fig)

    print("\nCorrected GPS outage simulation complete.")

if __name__ == "__main__":
    main()
