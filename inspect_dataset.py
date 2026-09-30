"""
Dataset Inspection Script — V-Vw4.csv
IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning)

Prints structural information, descriptive statistics, and timing analysis.
Generates plots: GPS trajectory, velocity, yaw rate, accelerations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
CSV_PATH = os.path.join(
    "IO-VNBD-SIH-",
    "Synchronised V abd S datasets",
    "Uncategorised IOVNB Dataset",
    "V-Dataset",
    "V-Vw4.csv",
)
PLOT_DIR = os.path.join("outputs", "plots", "dataset_inspection")
os.makedirs(PLOT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# 1. Load the dataset
# ──────────────────────────────────────────────
print("=" * 70)
print("Loading:", CSV_PATH)
print("=" * 70)

df = pd.read_csv(CSV_PATH, skipinitialspace=True)

# ──────────────────────────────────────────────
# 2. Structural information
# ──────────────────────────────────────────────
print(f"\nNumber of rows   : {len(df)}")
print(f"Number of columns: {len(df.columns)}")

print("\n--- Column Names ---")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>2}. {col}")

print("\n--- Data Types ---")
print(df.dtypes.to_string())

print("\n--- Missing Values Per Column ---")
missing = df.isnull().sum()
print(missing.to_string())

# ──────────────────────────────────────────────
# 3. First 5 rows
# ──────────────────────────────────────────────
print("\n--- First 5 Rows ---")
print(df.head(5).to_string())

# ──────────────────────────────────────────────
# 4. Last 5 rows
# ──────────────────────────────────────────────
print("\n--- Last 5 Rows ---")
print(df.tail(5).to_string())

# ──────────────────────────────────────────────
# 5. Descriptive statistics for key columns
# ──────────────────────────────────────────────
stat_columns = [
    "Velocity (km/hr)",
    "Heading (degrees)",
    "Yaw Rate (deg/sec)",
    "Wheel Speed Front Left (rad/sec)",
    "Wheel Speed Front Right (rad/sec)",
    "Wheel Speed Rear Left (rad/sec)",
    "Wheel Speed Rear Right (rad/sec)",
    "Indicated Longitudinal Acceleration (g)",
    "Indicated Lateral Acceleration (g)",
    "Steering Angle (degrees)",
    "Latitude (degrees)",
    "Longitude (degrees)",
]

print("\n--- Descriptive Statistics (Key Columns) ---")
print(df[stat_columns].describe().to_string())

# ──────────────────────────────────────────────
# 6. Timing analysis
# ──────────────────────────────────────────────
time_col = "Time Since Start of Day (seconds)"
sample_col = "Sample period (seconds)"

t_min = df[time_col].min()
t_max = df[time_col].max()
duration = t_max - t_min

sp_mean = df[sample_col].mean()
sp_min = df[sample_col].min()
sp_max = df[sample_col].max()

print("\n--- Timing Analysis ---")
print(f"  Minimum timestamp        : {t_min:.2f} s")
print(f"  Maximum timestamp        : {t_max:.2f} s")
print(f"  Total recording duration : {duration:.2f} s  ({duration / 3600:.2f} hr)")
print(f"  Mean sample period       : {sp_mean:.6f} s")
print(f"  Min  sample period       : {sp_min:.6f} s")
print(f"  Max  sample period       : {sp_max:.6f} s")

# ──────────────────────────────────────────────
# Helper: elapsed time array for x-axes
# ──────────────────────────────────────────────
elapsed = df[time_col].values - t_min  # seconds from trip start

# ──────────────────────────────────────────────
# 7. Plot — GPS trajectory
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(
    df["Longitude (degrees)"],
    df["Latitude (degrees)"],
    c=elapsed,
    cmap="viridis",
    s=1,
    alpha=0.7,
)
cbar = fig.colorbar(sc, ax=ax, label="Elapsed Time (s)")
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.set_title("GPS Trajectory — V-Vw4")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_traj = os.path.join(PLOT_DIR, "gps_trajectory.png")
fig.savefig(path_traj, dpi=150)
plt.close(fig)
print(f"\nSaved: {path_traj}")

# ──────────────────────────────────────────────
# 8. Plot — Velocity vs time
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(elapsed, df["Velocity (km/hr)"].values, linewidth=0.4, color="steelblue")
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("Velocity (km/hr)")
ax.set_title("Vehicle Velocity vs Time — V-Vw4")
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_vel = os.path.join(PLOT_DIR, "velocity_vs_time.png")
fig.savefig(path_vel, dpi=150)
plt.close(fig)
print(f"Saved: {path_vel}")

# ──────────────────────────────────────────────
# 9. Plot — Yaw rate vs time
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(elapsed, df["Yaw Rate (deg/sec)"].values, linewidth=0.4, color="darkorange")
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("Yaw Rate (deg/sec)")
ax.set_title("Yaw Rate vs Time — V-Vw4")
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_yaw = os.path.join(PLOT_DIR, "yaw_rate_vs_time.png")
fig.savefig(path_yaw, dpi=150)
plt.close(fig)
print(f"Saved: {path_yaw}")

# ──────────────────────────────────────────────
# 10. Plot — Longitudinal & lateral acceleration vs time
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(
    elapsed,
    df["Indicated Longitudinal Acceleration (g)"].values,
    linewidth=0.4,
    color="crimson",
    label="Longitudinal Acceleration (g)",
)
ax.plot(
    elapsed,
    df["Indicated Lateral Acceleration (g)"].values,
    linewidth=0.4,
    color="teal",
    label="Lateral Acceleration (g)",
)
ax.set_xlabel("Elapsed Time (s)")
ax.set_ylabel("Acceleration (g)")
ax.set_title("Longitudinal & Lateral Acceleration vs Time — V-Vw4")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_acc = os.path.join(PLOT_DIR, "acceleration_vs_time.png")
fig.savefig(path_acc, dpi=150)
plt.close(fig)
print(f"Saved: {path_acc}")

print("\n" + "=" * 70)
print("Dataset inspection complete.")
print("=" * 70)
