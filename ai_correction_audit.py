import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(
    "IO-VNBD-SIH-",
    "Synchronised V abd S datasets",
    "Uncategorised IOVNB Dataset",
    "V-Dataset",
    "V-Vw4.csv",
)
PLOT_DIR = os.path.join("outputs", "plots", "ai_audit")
os.makedirs(PLOT_DIR, exist_ok=True)

R_EARTH = 6_371_000
EFFECTIVE_RADIUS = 0.27881

def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    x = R_EARTH * np.cos(np.radians(lat0_deg)) * (np.radians(lon_deg) - np.radians(lon0_deg))
    y = R_EARTH * (np.radians(lat_deg) - np.radians(lat0_deg))
    return x, y

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def main():
    print("=" * 70)
    print("AI Correction Dataset Audit:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    
    # 1. Inputs (Available during outage)
    # ----------------------------------
    timestamps = df["Time Since Start of Day (seconds)"].values
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt[0] = 0.1
    
    ws_fl = df["Wheel Speed Front Left (rad/sec)"].values
    ws_fr = df["Wheel Speed Front Right (rad/sec)"].values
    ws_rl = df["Wheel Speed Rear Left (rad/sec)"].values
    ws_rr = df["Wheel Speed Rear Right (rad/sec)"].values
    v_wheel = np.mean([ws_fl, ws_fr, ws_rl, ws_rr], axis=0) * EFFECTIVE_RADIUS
    
    yaw_rate_dps = df["Yaw Rate (deg/sec)"].values
    yaw_rate_rps = np.radians(yaw_rate_dps)
    
    acc_long_g = df["Indicated Longitudinal Acceleration (g)"].values
    acc_lat_g = df["Indicated Lateral Acceleration (g)"].values
    steering_angle = df["Steering Angle (degrees)"].values
    
    # 2. Ground Truth (GPS, ONLY for training labels)
    # ----------------------------------------------
    lat_deg = df["Latitude (degrees)"].values
    lon_deg = df["Longitude (degrees)"].values
    heading_deg = df["Heading (degrees)"].values
    heading_rad = np.radians(heading_deg)
    
    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat_deg[0], lon_deg[0])
    
    # GPS derived velocities (for target analysis)
    # v_east = dx / dt, v_north = dy / dt
    dx = np.diff(ref_x, prepend=ref_x[0])
    dy = np.diff(ref_y, prepend=ref_y[0])
    gps_v_east = dx / dt
    gps_v_north = dy / dt
    gps_v_mag = np.sqrt(gps_v_east**2 + gps_v_north**2)
    
    gps_yaw_rate = wrap_angle(np.diff(heading_rad, prepend=heading_rad[0])) / dt
    
    # 3. Simulate continuous open-loop DR to calculate residuals
    # --------------------------------------------------------
    n = len(timestamps)
    dr_x = np.zeros(n)
    dr_y = np.zeros(n)
    dr_h = np.zeros(n)
    dr_h[0] = heading_rad[0]
    
    # EKF style bias tracking (simple high pass filter approximation for pre-outage)
    b_w = 0.0
    
    dr_v_east = np.zeros(n)
    dr_v_north = np.zeros(n)
    
    for i in range(1, n):
        d_t = dt[i]
        
        # Simple DR prediction
        dr_h[i] = dr_h[i-1] + (-yaw_rate_rps[i-1] - b_w) * d_t
        dr_h[i] = wrap_angle(dr_h[i])
        
        dr_v_east[i] = v_wheel[i-1] * np.sin(dr_h[i-1])
        dr_v_north[i] = v_wheel[i-1] * np.cos(dr_h[i-1])
        
        dr_x[i] = dr_x[i-1] + dr_v_east[i] * d_t
        dr_y[i] = dr_y[i-1] + dr_v_north[i] * d_t
        
    # 4. Target Feasibility Analysis
    # ------------------------------
    
    # Option 1: Position Residuals (Cumulative)
    res_x = ref_x - dr_x
    res_y = ref_y - dr_y
    
    # Option 2: Velocity Residuals (Instantaneous)
    res_v_east = gps_v_east - dr_v_east
    res_v_north = gps_v_north - dr_v_north
    
    # Option 3: Heading / Yaw Rate Residuals
    res_h = wrap_angle(heading_rad - dr_h)
    res_yaw_rate = gps_yaw_rate - (-yaw_rate_rps) # True yaw rate vs corrected sensor yaw rate
    
    print("\n--- Target Stability Analysis ---")
    print(f"Position Residuals Std: X={np.std(res_x):.2f} m, Y={np.std(res_y):.2f} m (Non-stationary, integrates over time)")
    print(f"Velocity Residuals Std: E={np.std(res_v_east):.2f} m/s, N={np.std(res_v_north):.2f} m/s (Stationary)")
    print(f"Heading Residual Std: {np.degrees(np.std(res_h)):.2f} deg (Non-stationary)")
    print(f"Yaw Rate Residual Std: {np.degrees(np.std(res_yaw_rate)):.2f} deg/s (Stationary)")
    
    # 5. Visualizations
    # -----------------
    time_s = timestamps - timestamps[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Position Error
    pos_err_mag = np.sqrt(res_x**2 + res_y**2)
    axes[0].plot(time_s, pos_err_mag, label="Position Error (m)")
    axes[0].set_title("Target Option 1: Cumulative Position Error (Unstable for direct ML)")
    axes[0].set_ylabel("Error (m)")
    
    # Plot 2: Velocity Residuals
    axes[1].plot(time_s, res_v_east, alpha=0.7, label="East Velocity Error (m/s)")
    axes[1].plot(time_s, res_v_north, alpha=0.7, label="North Velocity Error (m/s)")
    axes[1].set_title("Target Option 2: Instantaneous Velocity Residuals (Stable for ML)")
    axes[1].set_ylabel("Error (m/s)")
    axes[1].legend()
    
    # Plot 3: Yaw Rate / Heading
    axes[2].plot(time_s, np.degrees(res_yaw_rate), label="Yaw Rate Error (deg/s)", alpha=0.7)
    axes[2].set_title("Target Option 3: Instantaneous Yaw Rate Residuals (Stable for ML)")
    axes[2].set_ylabel("Error (deg/s)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ai_target_analysis.png"))
    plt.close(fig)
    
    # Correlation of inputs with Velocity Residuals
    # (To see if neural network inputs contain signal)
    print("\n--- Input Feature Correlation with Velocity Errors ---")
    for feat_name, feat_data in [
        ("Wheel Speed", v_wheel),
        ("Yaw Rate", yaw_rate_rps),
        ("Longitudinal Accel", acc_long_g),
        ("Lateral Accel", acc_lat_g),
        ("Steering Angle", steering_angle)
    ]:
        corr = np.corrcoef(feat_data, res_v_east)[0, 1]
        print(f"{feat_name:20s}: {corr:+.4f}")
        
    print("\nAudit Complete.")

if __name__ == "__main__":
    main()
