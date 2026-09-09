"""
=============================================================
ANALYZE V3 PRE-ATTACK FEATURE TRAJECTORIES
=============================================================
Evaluates the richer 43-feature V3 representation to determine
if pre-attack signals emerge, particularly for the Bot onset.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================
# CONFIGURATION
# =============================================================

STATES_DIR = Path("data/processed/states_v3")
OUTPUT_DIR = Path("models/latent_world_model_v4/pre_attack_onset_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# New features to focus on + a few base features for reference
FOCUS_FEATURES = [
    # New V3 Packet length stats
    "var_pkt_len", "std_fwd_pkt_len", "std_bwd_pkt_len",
    # New V3 TCP Windows
    "mean_init_fwd_win", "mean_init_bwd_win",
    # New V3 Header/Data ratios
    "mean_fwd_data_pkt_ratio", "mean_fwd_header_ratio", "mean_bwd_header_ratio",
    # New V3 Burstiness
    "log_active_mean", "active_std", "log_idle_mean", "idle_std",
    # New V3 Short flows
    "short_flow_ratio",
    # New V3 Subflow
    "mean_subflow_fwd_pkts", "mean_subflow_fwd_byts",
    "mean_subflow_bwd_pkts", "mean_subflow_bwd_byts",
    # Reference V2 features
    "syn_rate", "dst_port_entropy", "log_mean_bwd_iat"
]

WINDOWS_BEFORE = 10   # 100 seconds before
WINDOWS_AFTER = 5     # 50 seconds after

ONSET_TIMESTAMPS = [
    "2018-03-02 03:54:50", # Brute Force
    "2018-03-02 10:13:20", # Bot (Clean onset)
    "2018-03-02 12:13:30", # Infiltration
]

print("=" * 90)
print("V3 PRE-ATTACK FEATURE TRAJECTORY ANALYSIS")
print("=" * 90)

# =============================================================
# LOAD V3 STATES
# =============================================================

print("\nLoading March 2 V3 states...")
df = pd.read_csv(STATES_DIR / "Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v3.csv", low_memory=False)
df["window_start"] = pd.to_datetime(df["window_start"])
df = df.sort_values("window_start").reset_index(drop=True)
ts_to_idx = {ts: idx for idx, ts in enumerate(df["window_start"])}

print(f"  Total windows: {len(df)}")

# =============================================================
# BASELINE STATISTICS (benign-only)
# =============================================================

benign_df = df[df["attack_present"] == 0]
benign_means = benign_df[FOCUS_FEATURES].mean()
benign_stds = benign_df[FOCUS_FEATURES].std()

# =============================================================
# CATEGORISATION LOGIC
# =============================================================

def categorize_trajectory(z_scores_near, z_onset, z_scores_after):
    """
    z_scores_near: list of z-scores for [t-50s, t-40s, t-30s, t-20s, t-10s]
    z_onset: z-score at t=0
    z_scores_after: list of z-scores for [t+10s, t+20s, ...]
    """
    z_near_abs = np.abs(z_scores_near)
    z_onset_abs = abs(z_onset)
    z_after_abs = np.abs(z_scores_after)
    
    # 1. Monotonic drift
    # Check if z-scores are strictly increasing or strictly decreasing for at least 3 consecutive windows 
    # leading up to onset, and final value is > 1.5
    is_monotonic = False
    if len(z_scores_near) >= 3:
        for i in range(len(z_scores_near) - 2):
            seq = z_scores_near[i:]
            if len(seq) >= 3:
                diffs = np.diff(seq)
                if (np.all(diffs > 0.3) or np.all(diffs < -0.3)) and abs(seq[-1]) > 1.5:
                    is_monotonic = True
                    break
                    
    if is_monotonic:
        return "Monotonic drift"
        
    # 2. Consistent multi-window shift
    # Check if last 3 windows before onset are consistently > 1.5 sigma in same direction
    if len(z_scores_near) >= 3:
        last3 = z_scores_near[-3:]
        if (np.all(last3 > 1.5) or np.all(last3 < -1.5)):
            return "Consistent multi-window shift"
            
    # 3. Isolated spike
    # Check if only 1 window before onset > 2 sigma, others normal
    if np.sum(z_near_abs > 2.0) == 1 and np.sum(z_near_abs > 1.0) <= 2:
        return "Isolated spike"
        
    # 4. Changes only at onset
    if z_onset_abs > 2.0 and np.all(z_near_abs < 1.5):
        return "Changes only at onset"
        
    # 5. Changes only after onset
    if np.any(z_after_abs > 2.0) and z_onset_abs < 1.5 and np.all(z_near_abs < 1.5):
        return "Changes only after onset"
        
    # 6. No meaningful change
    return "No meaningful change"

# =============================================================
# ANALYSIS PER ONSET
# =============================================================

all_trajectory_rows = []

for onset_ts_str in ONSET_TIMESTAMPS:
    onset_ts = pd.to_datetime(onset_ts_str)
    
    if onset_ts not in ts_to_idx:
        continue
        
    onset_idx = ts_to_idx[onset_ts]
    
    print(f"\n{'=' * 90}")
    print(f"ONSET: {onset_ts} ({df.iloc[onset_idx]['dominant_attack']})")
    print(f"{'=' * 90}")

    # "Near" context is t-50s to t-10s (5 windows)
    # Check preceding 300 seconds for any attack windows
    context_start = max(0, onset_idx - 30)
    attack_history = df.iloc[context_start:onset_idx]["attack_present"].values
    n_attack_before = int(attack_history.sum())
    
    print(f"  Preceding 5min context: {'CLEAN' if n_attack_before == 0 else f'CONTAMINATED ({n_attack_before} attack windows)'}")
    
    # Extract window sequences
    near_idxs = list(range(max(0, onset_idx - 5), onset_idx))
    after_idxs = list(range(onset_idx + 1, min(len(df), onset_idx + 6)))
    
    print(f"\n  {'Feature':<25} {'t-50s':>8} {'t-40s':>8} {'t-30s':>8} {'t-20s':>8} {'t-10s':>8} {'ONSET':>8} {'Category':<30}")
    print("  " + "-" * 105)
    
    summary_counts = {
        "Monotonic drift": [],
        "Consistent multi-window shift": [],
        "Isolated spike": [],
        "Changes only at onset": [],
        "Changes only after onset": [],
        "No meaningful change": []
    }
    
    for feat in FOCUS_FEATURES:
        mean_val = benign_means[feat]
        std_val = benign_stds[feat] + 1e-10
        
        # Get raw values
        near_vals = df.iloc[near_idxs][feat].values
        onset_val = df.iloc[onset_idx][feat]
        after_vals = df.iloc[after_idxs][feat].values
        
        # Compute z-scores
        z_near = (near_vals - mean_val) / std_val
        z_onset = (onset_val - mean_val) / std_val
        z_after = (after_vals - mean_val) / std_val
        
        # Pad z_near if less than 5 windows available
        if len(z_near) < 5:
            pad = np.zeros(5 - len(z_near))
            z_near_padded = np.concatenate([pad, z_near])
        else:
            z_near_padded = z_near
            
        category = categorize_trajectory(z_near, z_onset, z_after)
        summary_counts[category].append(feat)
        
        # Only print features that aren't "No meaningful change" OR if it's the Bot onset
        if category != "No meaningful change" or "10:13:20" in onset_ts_str:
            z_strs = [f"{z:>8.2f}" for z in z_near_padded]
            onset_str = f"{z_onset:>8.2f}"
            if abs(z_onset) > 2:
                onset_str += "*"
            print(f"  {feat:<25} {' '.join(z_strs)} {onset_str} {category}")

    print("\n  Summary:")
    for cat, feats in summary_counts.items():
        if len(feats) > 0:
            print(f"    {cat+':':<30} {len(feats)} features")
            if cat in ["Monotonic drift", "Consistent multi-window shift", "Isolated spike", "Changes only at onset"]:
                print(f"      {feats}")

# =============================================================
# SAVE TO CSV
# =============================================================

# (Skipping full CSV dump for brevity in script output, just generating the plot)

# =============================================================
# PLOT FOCUS FEATURES FOR BOT ONSET
# =============================================================

print("\nGenerating trajectory plots for Bot Onset (#2)...")

PLOT_FEATURES = [
    "mean_init_fwd_win", "mean_init_bwd_win", "short_flow_ratio",
    "std_fwd_pkt_len", "mean_fwd_data_pkt_ratio", "mean_fwd_header_ratio",
    "log_active_mean", "log_idle_mean", "mean_subflow_fwd_pkts"
]

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("V3 Pre-Attack Feature Trajectories — Bot Onset (10:13:20)", fontsize=16, fontweight="bold")

onset_ts = pd.to_datetime("2018-03-02 10:13:20")
onset_idx = ts_to_idx[onset_ts]
start_idx = max(0, onset_idx - WINDOWS_BEFORE)
end_idx = min(len(df), onset_idx + WINDOWS_AFTER + 1)
deltas = [(i - onset_idx) * 10 for i in range(start_idx, end_idx)]

for feat_idx, feat in enumerate(PLOT_FEATURES):
    row = feat_idx // 3
    col = feat_idx % 3
    ax = axes[row, col]
    
    values = [df.iloc[i][feat] for i in range(start_idx, end_idx)]
    
    # Plot z-scores instead of raw values for comparability
    z_scores = [(v - benign_means[feat]) / (benign_stds[feat] + 1e-10) for v in values]
    
    ax.plot(deltas, z_scores, marker="o", markersize=5, linewidth=2, color="#2980b9")
    
    # Band for normal
    ax.axhspan(-1, 1, color="#bdc3c7", alpha=0.3)
    ax.axhspan(-2, 2, color="#bdc3c7", alpha=0.15)
    ax.axhline(0, color="#7f8c8d", linestyle="--", alpha=0.5, linewidth=1)
    
    ax.axvline(0, color="black", linestyle=":", alpha=0.8, linewidth=1.5)
    
    ax.set_title(feat, fontsize=12, fontweight="bold")
    ax.set_ylabel("Z-score")
    ax.set_xlabel("Seconds relative to onset")
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plot_path = OUTPUT_DIR / "bot_onset_v3_trajectories.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {plot_path}")

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
