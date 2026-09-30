"""
EKF GPS-Outage Simulation — V-Vw4.csv
======================================
Implements a 5-state Extended Kalman Filter (EKF).
State: [East, North, Velocity, Heading, YawRateBias]

Because accelerometers cannot be used without Pitch/Roll sensors
(to remove gravity), the EKF uses:
- Prediction (Control): Yaw Rate (gyro)
- Update (Observation): Wheel Speed (odometry) & GPS (only during pre-outage)

During the 60 seconds prior to an outage, the EKF uses GPS position 
and heading to converge the YawRateBias.
During the outage, GPS is denied. The EKF relies purely on the 
Wheel Speed observation and the converged YawRateBias.
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
PLOT_DIR = os.path.join("outputs", "plots", "ekf_outage")
METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

R_EARTH = 6_371_000
SAMPLE_RATE = 10
OUTAGE_DURATIONS_S = [10, 30, 60, 120, 300]
PRE_OUTAGE_S = 60  # Time to converge EKF bias before outage
MIN_VELOCITY_KMH = 10.0
MIN_WINDOW_SPACING_S = 60
NUM_WINDOWS = 50
EFFECTIVE_RADIUS = 0.27881

def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    x = R_EARTH * np.cos(np.radians(lat0_deg)) * (np.radians(lon_deg) - np.radians(lon0_deg))
    y = R_EARTH * (np.radians(lat_deg) - np.radians(lat0_deg))
    return x, y

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# ──────────────────────────────────────────────
# Extended Kalman Filter
# ──────────────────────────────────────────────
class EKF:
    def __init__(self):
        # x = [East, North, v, psi, b_w]
        self.x = np.zeros(5)
        self.P = np.diag([10.0, 10.0, 2.0, 0.1, 0.01])
        
        # Process Noise
        self.Q = np.diag([
            0.01,    # East
            0.01,    # North
            0.5,     # v (accel noise)
            0.001,   # psi (gyro noise)
            1e-7     # b_w (bias random walk)
        ])
        
        # Measurement Noise (GPS)
        self.R_gps = np.diag([
            2.0,     # East (m)
            2.0,     # North (m)
            0.01     # Heading (rad)
        ])
        
        # Measurement Noise (Wheel Speed)
        self.R_wheel = np.array([[0.05]])
        
    def predict(self, yaw_rate_meas_rps, dt):
        # Unpack state
        E, N, v, psi, b_w = self.x
        
        # 1. State Prediction
        # Correct yaw rate sign: -meas
        true_yaw_rate = -yaw_rate_meas_rps - b_w
        
        self.x[0] = E + v * np.sin(psi) * dt
        self.x[1] = N + v * np.cos(psi) * dt
        self.x[2] = v  # Constant velocity model
        self.x[3] = psi + true_yaw_rate * dt
        self.x[3] = wrap_angle(self.x[3])
        self.x[4] = b_w
        
        # 2. Jacobian of f(x)
        F = np.eye(5)
        F[0, 2] = np.sin(psi) * dt
        F[0, 3] = v * np.cos(psi) * dt
        F[1, 2] = np.cos(psi) * dt
        F[1, 3] = -v * np.sin(psi) * dt
        F[3, 4] = -dt
        
        # 3. Covariance Prediction
        self.P = F @ self.P @ F.T + self.Q
        
    def update_gps(self, E_meas, N_meas, psi_meas):
        # Observation z = [E, N, psi]
        z = np.array([E_meas, N_meas, psi_meas])
        
        # Expected observation
        z_hat = np.array([self.x[0], self.x[1], self.x[3]])
        
        # Innovation
        y = z - z_hat
        y[2] = wrap_angle(y[2])
        
        # Jacobian of h(x)
        H = np.zeros((3, 5))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 3] = 1.0
        
        # Kalman Gain
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update
        self.x = self.x + K @ y
        self.x[3] = wrap_angle(self.x[3])
        self.P = (np.eye(5) - K @ H) @ self.P
        
    def update_wheel(self, v_wheel):
        # Observation z = [v_wheel]
        z = np.array([v_wheel])
        z_hat = np.array([self.x[2]])
        y = z - z_hat
        
        H = np.zeros((1, 5))
        H[0, 2] = 1.0
        
        S = H @ self.P @ H.T + self.R_wheel
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

# ──────────────────────────────────────────────
# Outage Simulation Window
# ──────────────────────────────────────────────
def run_ekf_window(timestamps, v_wheel, yaw_rate_rps, ref_x, ref_y, heading_rad, start_idx, dur_s):
    pre_samples = PRE_OUTAGE_S * SAMPLE_RATE
    outage_samples = dur_s * SAMPLE_RATE
    
    ekf = EKF()
    # Initialize state at start of pre-outage window
    pre_idx = start_idx - pre_samples
    ekf.x = np.array([ref_x[pre_idx], ref_y[pre_idx], v_wheel[pre_idx], heading_rad[pre_idx], 0.0])
    
    # 1. Pre-Outage Convergence Phase (GPS Available)
    for k in range(1, pre_samples):
        i = pre_idx + k
        dt = timestamps[i] - timestamps[i - 1]
        
        ekf.predict(yaw_rate_rps[i - 1], dt)
        ekf.update_wheel(v_wheel[i])
        ekf.update_gps(ref_x[i], ref_y[i], heading_rad[i])
        
    # Bias is now converged
    converged_bias = ekf.x[4]
    
    # Force the EKF position and heading to perfectly match ground truth
    # right at the start of the outage, exactly like the DR baseline, 
    # to ensure an apples-to-apples comparison of propagation error!
    ekf.x[0] = ref_x[start_idx]
    ekf.x[1] = ref_y[start_idx]
    ekf.x[3] = heading_rad[start_idx]
    # We leave the velocity and bias as estimated by the EKF
    
    # 2. Outage Phase (GPS Denied)
    est_x = np.zeros(outage_samples)
    est_y = np.zeros(outage_samples)
    
    est_x[0] = ekf.x[0]
    est_y[0] = ekf.x[1]
    
    for k in range(1, outage_samples):
        i = start_idx + k
        dt = timestamps[i] - timestamps[i - 1]
        
        ekf.predict(yaw_rate_rps[i - 1], dt)
        ekf.update_wheel(v_wheel[i])
        # NO GPS UPDATE HERE
        
        est_x[k] = ekf.x[0]
        est_y[k] = ekf.x[1]
        
    ref_x_win = ref_x[start_idx:start_idx+outage_samples]
    ref_y_win = ref_y[start_idx:start_idx+outage_samples]
    errors = np.sqrt((est_x - ref_x_win)**2 + (est_y - ref_y_win)**2)
    
    return {
        "final_error": errors[-1],
        "converged_bias": np.degrees(converged_bias),
        "est_x": est_x,
        "est_y": est_y,
        "ref_x": ref_x_win,
        "ref_y": ref_y_win,
    }

def select_window_starts(velocity_kmh, n_total, max_outage_samples, min_vel, min_spacing_samples, num_windows):
    pre_samples = PRE_OUTAGE_S * SAMPLE_RATE
    max_start = n_total - max_outage_samples
    
    # We must ensure there is enough room BEFORE the start_idx for the pre_outage window
    candidates = np.where((velocity_kmh[pre_samples:max_start] >= min_vel))[0] + pre_samples
    
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
    
    timestamps    = df["Time Since Start of Day (seconds)"].values
    velocity_kmh  = df["Velocity (km/hr)"].values
    
    ws_fl = df["Wheel Speed Front Left (rad/sec)"].values
    ws_fr = df["Wheel Speed Front Right (rad/sec)"].values
    ws_rl = df["Wheel Speed Rear Left (rad/sec)"].values
    ws_rr = df["Wheel Speed Rear Right (rad/sec)"].values
    v_wheel = np.mean([ws_fl, ws_fr, ws_rl, ws_rr], axis=0) * EFFECTIVE_RADIUS
    
    yaw_rate_rps  = np.radians(df["Yaw Rate (deg/sec)"].values)
    lat_deg       = df["Latitude (degrees)"].values
    lon_deg       = df["Longitude (degrees)"].values
    heading_deg   = df["Heading (degrees)"].values
    heading_rad   = np.radians(heading_deg)

    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat_deg[0], lon_deg[0])

    max_outage_samples = max(OUTAGE_DURATIONS_S) * SAMPLE_RATE
    min_spacing_samples = MIN_WINDOW_SPACING_S * SAMPLE_RATE

    starts = select_window_starts(velocity_kmh, len(df), max_outage_samples, MIN_VELOCITY_KMH, min_spacing_samples, NUM_WINDOWS)
    
    # Load Baseline DR results for comparison
    base_metrics = pd.read_csv(os.path.join(METRICS_DIR, "wheel_odometry_comparison.csv"))
    
    all_results = []
    representative = {}
    
    for dur_s in OUTAGE_DURATIONS_S:
        finals = []
        biases = []
        window_res = []
        
        for si in starts:
            res = run_ekf_window(timestamps, v_wheel, yaw_rate_rps, ref_x, ref_y, heading_rad, si, dur_s)
            finals.append(res["final_error"])
            biases.append(res["converged_bias"])
            window_res.append((si, res))
            
        finals = np.array(finals)
        mean_ekf = np.mean(finals)
        
        # Get corresponding base
        row_base = base_metrics[base_metrics["outage_duration_s"] == dur_s].iloc[0]
        mean_base = row_base["wheel_mean"]
        imp = mean_base / mean_ekf if mean_ekf > 0 else 1.0
        
        row = {
            "outage_duration_s": dur_s,
            "ekf_mean": mean_ekf,
            "ekf_median": np.median(finals),
            "ekf_rmse": np.sqrt(np.mean(finals**2)),
            "ekf_max": np.max(finals),
            "improvement_factor": imp,
            "mean_converged_bias_deg_s": np.mean(biases)
        }
        all_results.append(row)
        print(f"Outage {dur_s:>3d}s | Base: {mean_base:>6.1f}m | EKF: {mean_ekf:>6.1f}m | Imp: {imp:.2f}x | Bias: {row['mean_converged_bias_deg_s']:.4f} deg/s")

        sorted_idx = np.argsort(finals)
        representative[dur_s] = {
            "best": window_res[sorted_idx[0]][1],
            "median": window_res[sorted_idx[len(sorted_idx) // 2]][1],
            "worst": window_res[sorted_idx[-1]][1],
        }

    ekf_df = pd.DataFrame(all_results)
    ekf_df.to_csv(os.path.join(METRICS_DIR, "ekf_outage_errors.csv"), index=False)
    
    # ── Plots ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(base_metrics["outage_duration_s"], base_metrics["wheel_mean"], "s--", color="gray", label="Baseline (Wheel Odometry DR)")
    ax.plot(ekf_df["outage_duration_s"], ekf_df["ekf_mean"], "o-", color="crimson", linewidth=2, label="EKF Mean Error")
    ax.set_xlabel("GPS Outage Duration (s)")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("EKF Sensor Fusion vs Baseline Dead-Reckoning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOT_DIR, "error_vs_outage_duration.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([finals for dur_s in OUTAGE_DURATIONS_S], tick_labels=[f"{d}s" for d in OUTAGE_DURATIONS_S], patch_artist=True, boxprops=dict(facecolor="lightcoral", alpha=0.7))
    ax.set_xlabel("GPS Outage Duration")
    ax.set_ylabel("Final Position Error (m)")
    ax.set_title("Distribution of EKF Final Error")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(PLOT_DIR, "error_distribution_boxplot.png"), dpi=150)
    plt.close(fig)

    show_durations = [30, 120, 300]
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle("Representative EKF Outage Trajectories", fontsize=14, y=0.98)
    for row_i, dur_s in enumerate(show_durations):
        for col_i, label in enumerate(["best", "median", "worst"]):
            ax = axes[row_i, col_i]
            res = representative[dur_s][label]
            ax.plot(res["ref_x"], res["ref_y"], linewidth=1.5, color="steelblue", label="GPS Reference")
            ax.plot(res["est_x"], res["est_y"], linewidth=1.5, color="crimson", label="EKF")
            ax.set_title(f"{dur_s}s outage — {label} (err={res['final_error']:.1f} m)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if row_i == 0 and col_i == 0: ax.legend(fontsize=8, loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOT_DIR, "representative_outage_trajectories.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
