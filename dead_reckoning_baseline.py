"""
Dead-Reckoning Baseline — V-Vw4.csv
====================================
Simple 2D vehicle dead-reckoning using velocity and yaw rate only.
GPS is used ONLY for:
  - initialising position (first row)
  - initialising heading  (first row)
  - reference trajectory  (evaluation)

No filtering, Kalman, ML, or sensor fusion.
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
PLOT_DIR = os.path.join("outputs", "plots", "dead_reckoning")
os.makedirs(PLOT_DIR, exist_ok=True)

R_EARTH = 6_371_000  # metres


# ──────────────────────────────────────────────
# Helper: GPS → local Cartesian (equirectangular)
# ──────────────────────────────────────────────
def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    """Convert GPS lat/lon arrays to local x/y (metres) centred at (lat0, lon0)."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    lat0_rad = np.radians(lat0_deg)
    lon0_rad = np.radians(lon0_deg)

    x = R_EARTH * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = R_EARTH * (lat_rad - lat0_rad)
    return x, y


# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────
print("=" * 70)
print("Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, skipinitialspace=True)
print(f"Rows: {len(df)},  Columns: {len(df.columns)}")
print("=" * 70)

# ──────────────────────────────────────────────
# 2. Extract raw signals
# ──────────────────────────────────────────────
timestamps  = df["Time Since Start of Day (seconds)"].values
velocity_kmh = df["Velocity (km/hr)"].values
yaw_rate_dps = df["Yaw Rate (deg/sec)"].values

lat_deg = df["Latitude (degrees)"].values
lon_deg = df["Longitude (degrees)"].values
heading_gps_deg = df["Heading (degrees)"].values

# ──────────────────────────────────────────────
# 3. Unit conversions
# ──────────────────────────────────────────────
velocity_ms  = velocity_kmh / 3.6            # km/h → m/s
yaw_rate_rps = np.radians(yaw_rate_dps)      # deg/s → rad/s

# ──────────────────────────────────────────────
# 4. Reference trajectory in local Cartesian
# ──────────────────────────────────────────────
lat0 = lat_deg[0]
lon0 = lon_deg[0]

ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat0, lon0)

# ──────────────────────────────────────────────
# 5. Dead-reckoning propagation
# ──────────────────────────────────────────────
n = len(timestamps)

est_x = np.zeros(n)
est_y = np.zeros(n)
est_heading = np.zeros(n)

# Initialise from GPS (first sample only)
est_x[0] = 0.0  # origin
est_y[0] = 0.0
est_heading[0] = np.radians(heading_gps_deg[0])

print("\nRunning dead-reckoning propagation ...")

for i in range(1, n):
    dt = timestamps[i] - timestamps[i - 1]

    # Update heading
    est_heading[i] = est_heading[i - 1] + yaw_rate_rps[i - 1] * dt

    # Update position
    est_x[i] = est_x[i - 1] + velocity_ms[i - 1] * np.cos(est_heading[i - 1]) * dt
    est_y[i] = est_y[i - 1] + velocity_ms[i - 1] * np.sin(est_heading[i - 1]) * dt

print("Dead-reckoning complete.")

# ──────────────────────────────────────────────
# 6. Build results dataframe
# ──────────────────────────────────────────────
results = pd.DataFrame({
    "timestamp":         timestamps,
    "estimated_x":       est_x,
    "estimated_y":       est_y,
    "reference_x":       ref_x,
    "reference_y":       ref_y,
    "estimated_heading":  est_heading,
})

# ──────────────────────────────────────────────
# 7. Position error
# ──────────────────────────────────────────────
error = np.sqrt((est_x - ref_x) ** 2 + (est_y - ref_y) ** 2)
results["position_error"] = error

mean_err  = np.mean(error)
rmse      = np.sqrt(np.mean(error ** 2))
max_err   = np.max(error)
final_err = error[-1]

print("\n--- Position Error Summary ---")
print(f"  Mean error  : {mean_err:,.2f} m")
print(f"  RMSE        : {rmse:,.2f} m")
print(f"  Max error   : {max_err:,.2f} m")
print(f"  Final error : {final_err:,.2f} m")

# ──────────────────────────────────────────────
# 8. Plot: Reference vs DR trajectory
# ──────────────────────────────────────────────
elapsed = timestamps - timestamps[0]

fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(ref_x, ref_y, linewidth=0.6, color="steelblue", label="GPS Reference")
ax.plot(est_x, est_y, linewidth=0.6, color="crimson",   label="Dead Reckoning")
ax.plot(ref_x[0], ref_y[0], "go", markersize=8, label="Start")
ax.plot(ref_x[-1], ref_y[-1], "bs", markersize=8, label="GPS End")
ax.plot(est_x[-1], est_y[-1], "r^", markersize=8, label="DR End")
ax.set_xlabel("X — East (m)")
ax.set_ylabel("Y — North (m)")
ax.set_title("GPS Reference vs Dead-Reckoning Trajectory — V-Vw4")
ax.legend(loc="best")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()

path_traj = os.path.join(PLOT_DIR, "trajectory_comparison.png")
fig.savefig(path_traj, dpi=150)
plt.close(fig)
print(f"\nSaved: {path_traj}")

# ──────────────────────────────────────────────
# 9. Plot: Position error vs time
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(elapsed, error, linewidth=0.5, color="darkorange")
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("Position Error (m)")
ax.set_title("Dead-Reckoning Position Error vs Time — V-Vw4")
ax.grid(True, alpha=0.3)
fig.tight_layout()

path_err = os.path.join(PLOT_DIR, "position_error_vs_time.png")
fig.savefig(path_err, dpi=150)
plt.close(fig)
print(f"Saved: {path_err}")

print("\n" + "=" * 70)
print("Dead-reckoning baseline complete.")
print("=" * 70)
