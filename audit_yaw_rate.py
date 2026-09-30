import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("IO-VNBD-SIH-", "Synchronised V abd S datasets", "Uncategorised IOVNB Dataset", "V-Dataset", "V-Vw4.csv")
PLOT_DIR = os.path.join("outputs", "plots", "yaw_rate_audit")
METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

R_EARTH = 6_371_000

def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    x = R_EARTH * np.cos(np.radians(lat0_deg)) * (np.radians(lon_deg) - np.radians(lon0_deg))
    y = R_EARTH * (np.radians(lat_deg) - np.radians(lat0_deg))
    return x, y

def wrap_angle_deg(angle):
    return (angle + 180) % 360 - 180

def main():
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    timestamps = df["Time Since Start of Day (seconds)"].values
    velocity_kmh = df["Velocity (km/hr)"].values
    yaw_rate_dps = df["Yaw Rate (deg/sec)"].values
    lat_deg = df["Latitude (degrees)"].values
    lon_deg = df["Longitude (degrees)"].values
    heading_deg = df["Heading (degrees)"].values
    
    dt = np.diff(timestamps)
    vel_ms = velocity_kmh / 3.6
    n = len(timestamps)
    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat_deg[0], lon_deg[0])
    
    # 1. Heading Convention Audit
    dx = np.diff(ref_x)
    dy = np.diff(ref_y)
    moving = velocity_kmh[:-1] > 10
    
    math_angle = np.degrees(np.arctan2(dy[moving], dx[moving])) % 360
    nav_heading = (90 - math_angle) % 360
    reported_heading = heading_deg[:-1][moving]
    
    math_diff = np.abs(wrap_angle_deg(math_angle - reported_heading))
    nav_diff = np.abs(wrap_angle_deg(nav_heading - reported_heading))
    
    is_nav = np.median(nav_diff) < np.median(math_diff)
    
    # 2. Yaw Rate Sign Audit
    gps_yaw_rate = wrap_angle_deg(np.diff(heading_deg)) / dt
    corr_positive = np.corrcoef(gps_yaw_rate[moving], yaw_rate_dps[:-1][moving])[0, 1]
    corr_negative = np.corrcoef(gps_yaw_rate[moving], -yaw_rate_dps[:-1][moving])[0, 1]
    
    # 3. Discrepancy & Regression
    # We want to model: gps_yaw_rate = scale * (-yaw_rate_dps) + bias
    A = np.vstack([-yaw_rate_dps[:-1][moving], np.ones(np.sum(moving))]).T
    slope, intercept = np.linalg.lstsq(A, gps_yaw_rate[moving], rcond=None)[0]
    
    # 4. Validations
    est_h_nav = np.zeros(n)
    est_h_nav[0] = np.radians(heading_deg[0])
    est_x_nav = np.zeros(n)
    est_y_nav = np.zeros(n)
    
    for i in range(1, n):
        d_t = dt[i-1]
        
        # Apply optimal scale and bias
        nav_yaw = slope * (-yaw_rate_dps[i-1]) + intercept
        
        est_h_nav[i] = est_h_nav[i-1] + np.radians(nav_yaw) * d_t
        
        est_x_nav[i] = est_x_nav[i-1] + vel_ms[i-1] * np.sin(est_h_nav[i-1]) * d_t
        est_y_nav[i] = est_y_nav[i-1] + vel_ms[i-1] * np.cos(est_h_nav[i-1]) * d_t
        
    err_nav = np.sqrt((est_x_nav - ref_x)**2 + (est_y_nav - ref_y)**2)
    final_err = err_nav[-1]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(ref_x, ref_y, label="GPS Reference")
    ax.plot(est_x_nav, est_y_nav, label="Corrected DR")
    ax.legend()
    fig.savefig(os.path.join(PLOT_DIR, "corrected_dr_test.png"))
    
    res_df = pd.DataFrame({
        "Heading_Is_Nav": [is_nav],
        "Corr_Positive": [corr_positive],
        "Corr_Negative": [corr_negative],
        "Optimal_Scale": [slope],
        "Optimal_Bias_deg_s": [intercept],
        "Final_Error_m": [final_err]
    })
    res_df.to_csv(os.path.join(METRICS_DIR, 'yaw_rate_audit.csv'), index=False)
    
    print("AUDIT RESULTS:")
    print(f"Heading Convention: {'Navigation (0=N, 90=E)' if is_nav else 'Mathematical'}")
    print(f"Yaw Sign: {'Opposite to Heading (CCW)' if corr_negative > corr_positive else 'Matches Heading'}")
    print(f"Scale: {slope:.4f}")
    print(f"Bias: {intercept:.4f}")
    print(f"Final Error: {final_err:.2f} m")

if __name__ == "__main__":
    main()
