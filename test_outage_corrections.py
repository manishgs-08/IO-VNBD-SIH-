import os
import numpy as np
import pandas as pd

CSV_PATH = os.path.join("IO-VNBD-SIH-", "Synchronised V abd S datasets", "Uncategorised IOVNB Dataset", "V-Dataset", "V-Vw4.csv")
R_EARTH = 6_371_000
SAMPLE_RATE = 10

def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    x = R_EARTH * np.cos(np.radians(lat0_deg)) * (np.radians(lon_deg) - np.radians(lon0_deg))
    y = R_EARTH * (np.radians(lat_deg) - np.radians(lat0_deg))
    return x, y

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

def run_outage(timestamps, vel_ms, yaw_rate_dps, ref_x, ref_y, heading_deg, starts, dur_s, fix_sign=True, use_nav_math=True):
    n_samples = dur_s * SAMPLE_RATE
    errors = []
    
    for si in starts:
        est_x = ref_x[si]
        est_y = ref_y[si]
        est_h = np.radians(heading_deg[si])
        
        for k in range(1, n_samples):
            i = si + k
            dt = timestamps[i] - timestamps[i-1]
            
            # Yaw Rate update
            yr = -yaw_rate_dps[i-1] if fix_sign else yaw_rate_dps[i-1]
            est_h += np.radians(yr) * dt
            
            # Position update
            if use_nav_math:
                est_x += vel_ms[i-1] * np.sin(est_h) * dt
                est_y += vel_ms[i-1] * np.cos(est_h) * dt
            else:
                est_x += vel_ms[i-1] * np.cos(est_h) * dt
                est_y += vel_ms[i-1] * np.sin(est_h) * dt
                
        err = np.sqrt((est_x - ref_x[si+n_samples-1])**2 + (est_y - ref_y[si+n_samples-1])**2)
        errors.append(err)
    return np.mean(errors)

def main():
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    timestamps = df["Time Since Start of Day (seconds)"].values
    vel_ms = df["Velocity (km/hr)"].values / 3.6
    yaw_rate_dps = df["Yaw Rate (deg/sec)"].values
    lat_deg = df["Latitude (degrees)"].values
    lon_deg = df["Longitude (degrees)"].values
    heading_deg = df["Heading (degrees)"].values
    
    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat_deg[0], lon_deg[0])
    
    starts = select_window_starts(df["Velocity (km/hr)"].values, len(df), 300*10, 10.0, 60*10, 50)
    
    durations = [10, 30, 60, 120, 300]
    
    print("Outage Duration | Original Buggy DR | Corrected Sign+Math | Corrected + Bias Removed")
    for dur in durations:
        err_orig = run_outage(timestamps, vel_ms, yaw_rate_dps, ref_x, ref_y, heading_deg, starts, dur, fix_sign=False, use_nav_math=False)
        err_fix = run_outage(timestamps, vel_ms, yaw_rate_dps, ref_x, ref_y, heading_deg, starts, dur, fix_sign=True, use_nav_math=True)
        # Apply bias correction (subtract -0.0313 from actual_yaw_rate, so add 0.0313 to -yaw_rate_dps)
        # Let's just create a modified yaw_rate array
        yaw_mod = yaw_rate_dps - 0.0313  # Wait: actual = -yaw_rate. corrected = actual - (-0.0313) = -yaw_rate + 0.0313 = -(yaw_rate - 0.0313)
        err_bias = run_outage(timestamps, vel_ms, yaw_mod, ref_x, ref_y, heading_deg, starts, dur, fix_sign=True, use_nav_math=True)
        
        print(f"{dur:15d} | {err_orig:15.1f} m | {err_fix:17.1f} m | {err_bias:23.1f} m")

if __name__ == "__main__":
    main()
