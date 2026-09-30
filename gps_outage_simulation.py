"""
GPS-Outage Simulation — V-Vw4.csv
==================================
Evaluates 2D dead-reckoning accuracy under simulated GPS outages
of varying duration (10 s – 300 s) at multiple locations in the trip.

GPS is used ONLY for:
  - Initialising DR state at the start of each outage window
  - Reference trajectory for error evaluation after the outage

During each outage window, only velocity + yaw rate are used.
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
PLOT_DIR = os.path.join("outputs", "plots", "gps_outage")
METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

R_EARTH = 6_371_000  # metres
SAMPLE_RATE = 10     # Hz

OUTAGE_DURATIONS_S = [10, 30, 60, 120, 300]

# Window selection parameters
MIN_VELOCITY_KMH = 10.0   # vehicle must be moving at window start
MIN_WINDOW_SPACING_S = 60  # minimum gap between adjacent window starts
NUM_WINDOWS = 50           # number of outage start locations to test


# ──────────────────────────────────────────────
# Helper: GPS → local Cartesian (equirectangular)
# ──────────────────────────────────────────────
def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """Convert GPS lat/lon to local x/y (metres) centred at (lat0, lon0)."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)
    x = R_EARTH * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = R_EARTH * (lat_rad - lat0_rad)
    return x, y


# ──────────────────────────────────────────────
# Dead-reckoning over a single outage window
# ──────────────────────────────────────────────
def run_dr_window(timestamps, velocity_ms, yaw_rate_rps,
                  ref_x, ref_y, heading0_rad, start_idx, n_samples):
    """
    Propagate DR from start_idx for n_samples steps.

    Parameters
    ----------
    timestamps   : full timestamp array (seconds)
    velocity_ms  : full velocity array (m/s)
    yaw_rate_rps : full yaw-rate array (rad/s)
    ref_x, ref_y : full GPS reference in local Cartesian (m)
    heading0_rad : initial heading at start_idx (radians)
    start_idx    : index where outage begins
    n_samples    : number of samples in the outage window

    Returns
    -------
    dict with arrays: est_x, est_y, est_heading, ref_x_win, ref_y_win,
                      errors, final_error
    """
    end_idx = start_idx + n_samples  # exclusive

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
        est_h[k] = est_h[k - 1] + yaw_rate_rps[i - 1] * dt
        est_x[k] = est_x[k - 1] + velocity_ms[i - 1] * np.cos(est_h[k - 1]) * dt
        est_y[k] = est_y[k - 1] + velocity_ms[i - 1] * np.sin(est_h[k - 1]) * dt

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
    """
    Pick evenly-spaced indices where the vehicle is moving and
    there is enough data remaining for the longest outage.

    Strategy:
      1. Identify all candidate indices where velocity >= min_vel
         and index + max_outage_samples <= n_total.
      2. From those candidates, select ~num_windows evenly spaced.
      3. Enforce minimum spacing between selected indices.
    """
    max_start = n_total - max_outage_samples
    candidates = np.where(
        (velocity_kmh[:max_start] >= min_vel)
    )[0]

    if len(candidates) == 0:
        raise ValueError("No valid window start positions found.")

    # Evenly space through candidates
    step = max(1, len(candidates) // num_windows)
    selected = candidates[::step]

    # Enforce minimum spacing
    filtered = [selected[0]]
    for idx in selected[1:]:
        if idx - filtered[-1] >= min_spacing_samples:
            filtered.append(idx)

    return np.array(filtered)


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main():
    # ── 1. Load dataset ──────────────────────
    print("=" * 70)
    print("Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    n_total = len(df)
    print(f"Rows: {n_total}")
    print("=" * 70)

    timestamps    = df["Time Since Start of Day (seconds)"].values
    velocity_kmh  = df["Velocity (km/hr)"].values
    velocity_ms   = velocity_kmh / 3.6
    yaw_rate_dps  = df["Yaw Rate (deg/sec)"].values
    yaw_rate_rps  = np.radians(yaw_rate_dps)
    lat_deg       = df["Latitude (degrees)"].values
    lon_deg       = df["Longitude (degrees)"].values
    heading_deg   = df["Heading (degrees)"].values

    # Convert full GPS to local Cartesian (origin = first row)
    lat0, lon0 = lat_deg[0], lon_deg[0]
    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat0, lon0)

    # ── 2. Select window start locations ─────
    max_outage_samples = max(OUTAGE_DURATIONS_S) * SAMPLE_RATE
    min_spacing_samples = MIN_WINDOW_SPACING_S * SAMPLE_RATE

    starts = select_window_starts(
        velocity_kmh, n_total, max_outage_samples,
        MIN_VELOCITY_KMH, min_spacing_samples, NUM_WINDOWS,
    )
    print(f"\nSelected {len(starts)} outage start locations")
    print(f"  First start index : {starts[0]}  (t = {timestamps[starts[0]]:.1f} s)")
    print(f"  Last  start index : {starts[-1]}  (t = {timestamps[starts[-1]]:.1f} s)")

    # ── 3. Run experiments ───────────────────
    all_results = []  # list of dicts for the summary table

    # Store per-window results for representative plotting
    representative_windows = {}

    for dur_s in OUTAGE_DURATIONS_S:
        n_samples = dur_s * SAMPLE_RATE
        errors_final = []
        errors_all_max = []
        errors_all_mean = []
        window_results_for_dur = []

        for si in starts:
            heading0 = np.radians(heading_deg[si])
            res = run_dr_window(
                timestamps, velocity_ms, yaw_rate_rps,
                ref_x, ref_y, heading0, si, n_samples,
            )
            errors_final.append(res["final_error"])
            errors_all_max.append(np.max(res["errors"]))
            errors_all_mean.append(np.mean(res["errors"]))
            window_results_for_dur.append((si, res))

        errors_final = np.array(errors_final)
        errors_all_max = np.array(errors_all_max)
        errors_all_mean = np.array(errors_all_mean)

        row = {
            "outage_duration_s": dur_s,
            "mean_error_m":      np.mean(errors_final),
            "median_error_m":    np.median(errors_final),
            "rmse_m":            np.sqrt(np.mean(errors_final ** 2)),
            "max_error_m":       np.max(errors_final),
            "final_error_mean_m": np.mean(errors_final),
        }
        all_results.append(row)

        # Keep a few representative windows for trajectory plotting
        # Pick worst, median, and best
        sorted_idx = np.argsort(errors_final)
        picks = {
            "best":   sorted_idx[0],
            "median": sorted_idx[len(sorted_idx) // 2],
            "worst":  sorted_idx[-1],
        }
        representative_windows[dur_s] = {
            label: window_results_for_dur[i] for label, i in picks.items()
        }

        print(f"\n  Outage {dur_s:>3d}s  |  mean={row['mean_error_m']:>8.1f} m  "
              f"median={row['median_error_m']:>8.1f} m  "
              f"rmse={row['rmse_m']:>8.1f} m  "
              f"max={row['max_error_m']:>8.1f} m")

    # ── 4. Build metrics table ───────────────
    metrics_df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("GPS Outage — Position Error Summary (final error at each window)")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    csv_path = os.path.join(METRICS_DIR, "gps_outage_errors.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── 5. Plot: Final error vs outage duration ─
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        metrics_df["outage_duration_s"],
        metrics_df["mean_error_m"],
        "o-", color="crimson", linewidth=2, markersize=8, label="Mean final error",
    )
    ax.plot(
        metrics_df["outage_duration_s"],
        metrics_df["median_error_m"],
        "s--", color="steelblue", linewidth=2, markersize=8, label="Median final error",
    )
    ax.plot(
        metrics_df["outage_duration_s"],
        metrics_df["max_error_m"],
        "^:", color="darkorange", linewidth=2, markersize=8, label="Max final error",
    )
    ax.set_xlabel("GPS Outage Duration (s)", fontsize=12)
    ax.set_ylabel("Position Error at End of Outage (m)", fontsize=12)
    ax.set_title("Dead-Reckoning Error vs GPS Outage Duration", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_dur = os.path.join(PLOT_DIR, "error_vs_outage_duration.png")
    fig.savefig(path_dur, dpi=150)
    plt.close(fig)
    print(f"Saved: {path_dur}")

    # ── 6. Plot: Representative outage trajectories ─
    # Show 3 durations × 3 quality levels = 9 panels
    show_durations = [30, 120, 300]
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(
        "Representative GPS-Outage Windows — DR vs GPS Reference",
        fontsize=14, y=0.98,
    )

    for row_i, dur_s in enumerate(show_durations):
        for col_i, label in enumerate(["best", "median", "worst"]):
            ax = axes[row_i, col_i]
            si, res = representative_windows[dur_s][label]

            ax.plot(res["ref_x"], res["ref_y"],
                    linewidth=1.5, color="steelblue", label="GPS Reference")
            ax.plot(res["est_x"], res["est_y"],
                    linewidth=1.5, color="crimson", label="Dead Reckoning")

            # Mark start and end
            ax.plot(res["ref_x"][0], res["ref_y"][0], "go", markersize=7)
            ax.plot(res["ref_x"][-1], res["ref_y"][-1], "bs", markersize=7)
            ax.plot(res["est_x"][-1], res["est_y"][-1], "r^", markersize=7)

            final_err = res["final_error"]
            ax.set_title(f"{dur_s}s outage — {label} (err={final_err:.1f} m)",
                         fontsize=10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=8, loc="best")
            if col_i == 0:
                ax.set_ylabel("Y — North (m)", fontsize=9)
            if row_i == 2:
                ax.set_xlabel("X — East (m)", fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path_rep = os.path.join(PLOT_DIR, "representative_outage_trajectories.png")
    fig.savefig(path_rep, dpi=150)
    plt.close(fig)
    print(f"Saved: {path_rep}")

    # ── 7. Plot: Error distribution box-plots ─
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = []
    box_labels = []
    for dur_s in OUTAGE_DURATIONS_S:
        n_samples = dur_s * SAMPLE_RATE
        finals = []
        for si in starts:
            heading0 = np.radians(heading_deg[si])
            res = run_dr_window(
                timestamps, velocity_ms, yaw_rate_rps,
                ref_x, ref_y, heading0, si, n_samples,
            )
            finals.append(res["final_error"])
        box_data.append(finals)
        box_labels.append(f"{dur_s}s")

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor="lightcoral", alpha=0.7),
                    medianprops=dict(color="black", linewidth=2))
    ax.set_xlabel("GPS Outage Duration", fontsize=12)
    ax.set_ylabel("Final Position Error (m)", fontsize=12)
    ax.set_title("Distribution of DR Final Error Across Outage Windows", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path_box = os.path.join(PLOT_DIR, "error_distribution_boxplot.png")
    fig.savefig(path_box, dpi=150)
    plt.close(fig)
    print(f"Saved: {path_box}")

    print("\n" + "=" * 70)
    print("GPS outage simulation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
