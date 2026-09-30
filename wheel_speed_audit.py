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
PLOT_DIR = os.path.join("outputs", "plots", "wheel_speed_audit")
METRICS_DIR = os.path.join("outputs", "metrics")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("Loading:", CSV_PATH)
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    
    velocity_ms = df["Velocity (km/hr)"].values / 3.6
    ws_fl = df["Wheel Speed Front Left (rad/sec)"].values
    ws_fr = df["Wheel Speed Front Right (rad/sec)"].values
    ws_rl = df["Wheel Speed Rear Left (rad/sec)"].values
    ws_rr = df["Wheel Speed Rear Right (rad/sec)"].values
    
    # 1. Basic stats
    stats = []
    for name, arr in [("FL", ws_fl), ("FR", ws_fr), ("RL", ws_rl), ("RR", ws_rr)]:
        stats.append({
            "Wheel": name,
            "Mean": np.mean(arr),
            "Median": np.median(arr),
            "Std": np.std(arr),
            "Max": np.max(arr),
            "Min": np.min(arr)
        })
    stats_df = pd.DataFrame(stats)
    print("\n--- Wheel Speed Basic Stats ---")
    print(stats_df.to_string(index=False))
    
    # 2. Consistency check (variance between wheels)
    ws_mean = np.mean([ws_fl, ws_fr, ws_rl, ws_rr], axis=0)
    ws_std = np.std([ws_fl, ws_fr, ws_rl, ws_rr], axis=0)
    
    print(f"\nMean std dev across 4 wheels at same timestamp: {np.mean(ws_std):.4f} rad/s")
    print(f"Max diff across 4 wheels: {np.max(np.max([ws_fl, ws_fr, ws_rl, ws_rr], axis=0) - np.min([ws_fl, ws_fr, ws_rl, ws_rr], axis=0)):.4f} rad/s")

    # 3. Linear relationship (effective wheel radius)
    # v = r * w
    # We want to regress v against w to find r.
    # We should only use moving, straight-line segments to avoid differential turning speeds
    yaw_rate = df["Yaw Rate (deg/sec)"].values
    straight_mask = (velocity_ms > 2.0) & (np.abs(yaw_rate) < 2.0)
    
    radii = {}
    for name, arr in [("FL", ws_fl), ("FR", ws_fr), ("RL", ws_rl), ("RR", ws_rr)]:
        w = arr[straight_mask]
        v = velocity_ms[straight_mask]
        # v = r * w  => r = v / w
        r_mean = np.mean(v / w)
        r_median = np.median(v / w)
        radii[name] = {"mean_r": r_mean, "median_r": r_median}
        
    print("\n--- Effective Wheel Radius (v / w) [straight driving > 2 m/s] ---")
    for name, res in radii.items():
        print(f"{name}: Mean = {res['mean_r']:.5f} m, Median = {res['median_r']:.5f} m")

    # Let's do a least squares fit for the mean wheel speed
    w_mean_straight = ws_mean[straight_mask]
    v_straight = velocity_ms[straight_mask]
    A = w_mean_straight[:, np.newaxis]
    radius_lstsq, _, _, _ = np.linalg.lstsq(A, v_straight, rcond=None)
    eff_radius = radius_lstsq[0]
    
    print(f"\nOptimal Effective Radius (Least Squares on Mean WS): {eff_radius:.5f} m")
    
    # Calculate R-squared
    v_pred = eff_radius * w_mean_straight
    ss_res = np.sum((v_straight - v_pred) ** 2)
    ss_tot = np.sum((v_straight - np.mean(v_straight)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R-squared of linear fit: {r2:.6f}")

    # 4. Save metrics
    metrics = {
        "eff_radius_lstsq": [eff_radius],
        "r_squared": [r2],
        "mean_std_across_wheels": [np.mean(ws_std)],
        "max_diff_across_wheels": [np.max(np.max([ws_fl, ws_fr, ws_rl, ws_rr], axis=0) - np.min([ws_fl, ws_fr, ws_rl, ws_rr], axis=0))]
    }
    pd.DataFrame(metrics).to_csv(os.path.join(METRICS_DIR, "wheel_speed_audit.csv"), index=False)
    
    # 5. Plots
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ws_mean[::10], velocity_ms[::10], alpha=0.1, s=1)
    x_line = np.array([0, np.max(ws_mean)])
    ax.plot(x_line, eff_radius * x_line, "r-", label=f"Fit: v = {eff_radius:.4f} * w")
    ax.set_xlabel("Mean Wheel Speed (rad/s)")
    ax.set_ylabel("Vehicle Velocity (m/s)")
    ax.set_title("Vehicle Velocity vs Wheel Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOT_DIR, "velocity_vs_wheelspeed.png"))
    plt.close(fig)
    
    # Check for anomalies (dropouts, negative values)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = df["Time Since Start of Day (seconds)"].values
    axes[0].plot(t, ws_fl, label="FL", linewidth=0.5, alpha=0.8)
    axes[0].plot(t, ws_fr, label="FR", linewidth=0.5, alpha=0.8)
    axes[0].plot(t, ws_rl, label="RL", linewidth=0.5, alpha=0.8)
    axes[0].plot(t, ws_rr, label="RR", linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("Wheel Speed (rad/s)")
    axes[0].set_title("Raw Wheel Speeds over Time")
    axes[0].legend(loc="upper right")
    
    diff = ws_fl - ws_fr
    axes[1].plot(t, diff, color="purple", linewidth=0.5)
    axes[1].set_ylabel("FL - FR (rad/s)")
    axes[1].set_title("Front Axle Differential (turns)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "wheel_speed_timeseries.png"))
    plt.close(fig)
    
    print("\nAudit Complete.")

if __name__ == "__main__":
    main()
