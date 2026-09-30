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
PLOT_DIR = os.path.join("outputs", "plots", "ekf_audit")
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("EKF Sensor Audit:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    
    # Extract relevant fields
    vel_kmh = df["Velocity (km/hr)"].values
    vel_ms = vel_kmh / 3.6
    dt = np.diff(df["Time Since Start of Day (seconds)"].values)
    acc_long_g = df["Indicated Longitudinal Acceleration (g)"].values
    acc_lat_g = df["Indicated Lateral Acceleration (g)"].values
    
    # 1. Accelerometer Analysis
    acc_long_ms2 = acc_long_g * 9.81
    acc_lat_ms2 = acc_lat_g * 9.81
    
    # Compare derivative of velocity with longitudinal acceleration
    d_vel_dt = np.diff(vel_ms) / dt
    
    # Because lengths differ by 1, truncate acc
    acc_long_ms2 = acc_long_ms2[:-1]
    acc_lat_ms2 = acc_lat_ms2[:-1]
    
    # Correlation between derivative of velocity and longitudinal accel
    corr_long = np.corrcoef(d_vel_dt, acc_long_ms2)[0, 1]
    corr_neg_long = np.corrcoef(d_vel_dt, -acc_long_ms2)[0, 1]
    
    # Bias check during stationary periods
    stationary = vel_ms[:-1] < 0.1
    mean_bias_long = np.mean(acc_long_ms2[stationary])
    mean_bias_lat = np.mean(acc_lat_ms2[stationary])
    
    print("\n--- Accelerometer Audit ---")
    print(f"Longitudinal Accel Correlation with dV/dt (positive): {corr_long:.4f}")
    print(f"Longitudinal Accel Correlation with dV/dt (negative): {corr_neg_long:.4f}")
    print(f"Mean Long Accel (Stationary): {mean_bias_long:.4f} m/s^2")
    print(f"Mean Lat Accel (Stationary): {mean_bias_lat:.4f} m/s^2")
    
    # 2. Plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = df["Time Since Start of Day (seconds)"].values[:-1]
    
    axes[0].plot(t, d_vel_dt, label="dV/dt (GPS derived)", alpha=0.7)
    
    # Apply correct sign based on correlation
    if corr_long > corr_neg_long:
        actual_acc_long = acc_long_ms2
        print("Sign convention: Positive is FORWARD.")
    else:
        actual_acc_long = -acc_long_ms2
        print("Sign convention: Negative is FORWARD.")
        
    axes[0].plot(t, actual_acc_long, label="Longitudinal Accel (IMU)", alpha=0.7)
    axes[0].set_ylabel("Accel (m/s^2)")
    axes[0].legend()
    axes[0].set_title("Longitudinal Acceleration: IMU vs dV/dt")
    
    axes[1].plot(t, acc_lat_ms2, label="Lateral Accel (g * 9.81)")
    axes[1].set_ylabel("Accel (m/s^2)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    
    fig.savefig(os.path.join(PLOT_DIR, "accelerometer_audit.png"))
    plt.close(fig)

    print("\nAudit Complete.")

if __name__ == "__main__":
    main()
