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
PLOT_DIR = os.path.join("outputs", "plots", "ai_training")
os.makedirs(PLOT_DIR, exist_ok=True)

R_EARTH = 6_371_000
EFFECTIVE_RADIUS = 0.27881

def gps_to_local_xy(lat_deg, lon_deg, lat0_deg, lon0_deg):
    x = R_EARTH * np.cos(np.radians(lat0_deg)) * (np.radians(lon_deg) - np.radians(lon0_deg))
    y = R_EARTH * (np.radians(lat_deg) - np.radians(lat0_deg))
    return x, y

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def create_sequences(X, Y, seq_length):
    num_samples = len(X) - seq_length + 1
    if num_samples <= 0:
        return np.array([]), np.array([])
    # Use strides for efficient windowing (if memory is an issue, otherwise a simple loop)
    # simple loop is fine for 126k rows
    seq_X = np.lib.stride_tricks.sliding_window_view(X, (seq_length, X.shape[1])).squeeze(1)
    seq_Y = Y[seq_length-1:]
    return seq_X, seq_Y

def main():
    print("=" * 70)
    print("AI Dataset Validation:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    
    n_total = len(df)
    print(f"Total rows in dataset: {n_total}")
    
    # 1. Feature Extraction (No GPS)
    # --------------------------------
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
    
    # Create the Feature Matrix (X)
    X = np.stack([
        v_wheel,
        yaw_rate_rps,
        acc_long_g,
        acc_lat_g,
        steering_angle
    ], axis=1)
    
    feature_names = [
        "Wheel-Speed Velocity (m/s)",
        "Yaw Rate (rad/s)",
        "Longitudinal Accel (g)",
        "Lateral Accel (g)",
        "Steering Angle (deg)"
    ]
    
    print("\n--- GPS Leakage Audit ---")
    print("The following input features will be used for AI inference:")
    for fn in feature_names:
        print(f"  [AVAILABLE DURING GPS OUTAGE] - {fn}")
    print("  (No GPS-derived features are present in the input matrix X)")
    
    # 2. Target Extraction (GPS-Derived Labels)
    # -----------------------------------------
    lat_deg = df["Latitude (degrees)"].values
    lon_deg = df["Longitude (degrees)"].values
    heading_deg = df["Heading (degrees)"].values
    heading_rad = np.radians(heading_deg)
    
    ref_x, ref_y = gps_to_local_xy(lat_deg, lon_deg, lat_deg[0], lon_deg[0])
    
    gps_v_east = np.diff(ref_x, prepend=ref_x[0]) / dt
    gps_v_north = np.diff(ref_y, prepend=ref_y[0]) / dt
    gps_yaw_rate = wrap_angle(np.diff(heading_rad, prepend=heading_rad[0])) / dt
    
    # Simple DR logic for residual calculation
    # Using baseline DR to find the errors
    dr_h = np.zeros(n_total)
    dr_h[0] = heading_rad[0]
    for i in range(1, n_total):
        dr_h[i] = wrap_angle(dr_h[i-1] + (-yaw_rate_rps[i-1]) * dt[i])
        
    dr_v_east = v_wheel * np.sin(dr_h)
    dr_v_north = v_wheel * np.cos(dr_h)
    
    # Target residuals (Y)
    y_v_east = gps_v_east - dr_v_east
    y_v_north = gps_v_north - dr_v_north
    y_yaw_rate = gps_yaw_rate - (-yaw_rate_rps) # True yaw rate - Sensor yaw rate (CCW sign matched)
    
    Y = np.stack([y_v_east, y_v_north, y_yaw_rate], axis=1)
    target_names = ["East Velocity Error (m/s)", "North Velocity Error (m/s)", "Yaw Rate Error (rad/s)"]
    
    # 3. Chronological Splitting
    # --------------------------
    # Train: 0-60%, Val: 60-80%, Test: 80-100%
    idx_val = int(0.60 * n_total)
    idx_test = int(0.80 * n_total)
    
    X_train_raw, Y_train_raw = X[:idx_val], Y[:idx_val]
    X_val_raw, Y_val_raw = X[idx_val:idx_test], Y[idx_val:idx_test]
    X_test_raw, Y_test_raw = X[idx_test:], Y[idx_test:]
    
    # 4. Normalization (Fit on Train ONLY)
    # ------------------------------------
    # We standardize features: (x - mean) / std
    X_mean = np.nanmean(X_train_raw, axis=0)
    X_std = np.nanstd(X_train_raw, axis=0)
    # Prevent div by zero
    X_std[X_std == 0] = 1.0
    
    X_train = (X_train_raw - X_mean) / X_std
    X_val = (X_val_raw - X_mean) / X_std
    X_test = (X_test_raw - X_mean) / X_std
    
    # For targets, we also often normalize or just scale, 
    # but for validation report we can look at raw Y or scaled Y. 
    # We will compute stats on un-normalized Y for physical meaning.
    
    # 5. Sequence Length Analysis
    # ---------------------------
    print("\n--- Sequence Dataset Analysis ---")
    seq_lengths = [10, 30, 50, 100]  # 1s, 3s, 5s, 10s
    
    for seq_len in seq_lengths:
        seq_X_train, seq_Y_train = create_sequences(X_train, Y_train_raw, seq_len)
        seq_X_val, seq_Y_val = create_sequences(X_val, Y_val_raw, seq_len)
        seq_X_test, seq_Y_test = create_sequences(X_test, Y_test_raw, seq_len)
        
        print(f"History Length: {seq_len} samples ({seq_len/10.0} seconds)")
        print(f"  Train sequences: {len(seq_X_train)}")
        print(f"  Val sequences:   {len(seq_X_val)}")
        print(f"  Test sequences:  {len(seq_X_test)}")
        print(f"  Input Shape:     {seq_X_train.shape if len(seq_X_train)>0 else 'N/A'}")
        print(f"  Target Shape:    {seq_Y_train.shape if len(seq_Y_train)>0 else 'N/A'}")
        print(f"  ---")

    # Data quality check on entire X and Y
    nan_count = np.isnan(X).sum() + np.isnan(Y).sum()
    inf_count = np.isinf(X).sum() + np.isinf(Y).sum()
    print(f"\n--- Data Quality Audit ---")
    print(f"NaN count across dataset: {nan_count}")
    print(f"Infinity count across dataset: {inf_count}")

    # Stats on Train Features (Raw)
    print("\n--- Train Feature Statistics (Raw) ---")
    for i, name in enumerate(feature_names):
        print(f"  {name:25s} | Mean: {X_mean[i]:7.4f}, Std: {X_std[i]:7.4f}, Min: {np.min(X_train_raw[:,i]):7.4f}, Max: {np.max(X_train_raw[:,i]):7.4f}")
        
    print("\n--- Train Target Statistics (Raw) ---")
    for i, name in enumerate(target_names):
        mean_y = np.mean(Y_train_raw[:, i])
        std_y = np.std(Y_train_raw[:, i])
        min_y = np.min(Y_train_raw[:, i])
        max_y = np.max(Y_train_raw[:, i])
        print(f"  {name:25s} | Mean: {mean_y:7.4f}, Std: {std_y:7.4f}, Min: {min_y:7.4f}, Max: {max_y:7.4f}")

    # 6. Diagnostic Plots
    # -------------------
    # Target Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axes[i].hist(Y_train_raw[:, i], bins=100, alpha=0.7, color='crimson')
        axes[i].set_title(target_names[i])
        axes[i].set_yscale('log')
    fig.suptitle("Training Target Distributions (Log Scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "target_distributions.png"), dpi=150)
    plt.close(fig)
    
    # Feature Distributions
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].hist(X_train_raw[:, i], bins=50, alpha=0.7, color='steelblue')
        axes[i].set_title(feature_names[i])
    fig.suptitle("Training Feature Distributions")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "feature_distributions.png"), dpi=150)
    plt.close(fig)

    # Input-Target Correlation
    corr_matrix = np.zeros((5, 3))
    for i in range(5):
        for j in range(3):
            corr_matrix[i, j] = np.corrcoef(X_train_raw[:, i], Y_train_raw[:, j])[0, 1]
            
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["V_East", "V_North", "YawRate"])
    ax.set_yticklabels(feature_names)
    for i in range(5):
        for j in range(3):
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha='center', va='center', color='black' if abs(corr_matrix[i,j]) < 0.5 else 'white')
    ax.set_title("Feature-Target Correlation Matrix (Train)", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "input_target_correlation.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
