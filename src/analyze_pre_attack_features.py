"""
=============================================================
ANALYZE PRE-ATTACK FEATURE TRAJECTORIES
=============================================================

For each attack onset on March 2, inspects the 26 base network
features in the ~80 seconds before and ~30 seconds after onset.

Goal: Determine whether precursor patterns exist in the aggregate
flow features that could provide genuine early warning signal,
versus features that only change at/after onset or that are
already abnormal from a preceding attack episode.

=============================================================
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

STATES_DIR = Path("data/processed/states_v2")
OUTPUT_DIR = Path("models/latent_world_model_v4/pre_attack_onset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Features the user specifically wants to focus on
FOCUS_FEATURES = [
    "flow_count",
    "window_packet_rate",
    "window_byte_rate",
    "syn_rate",
    "ack_rate",
    "rst_rate",
    "log_unique_dst_ports",
    "dst_port_entropy",
    "protocol_entropy",
    "log_mean_fwd_iat",
    "log_mean_bwd_iat",
    "mean_pkt_len",
    "std_pkt_len",
    "fwd_packet_ratio",
    "bwd_packet_ratio",
    "fwd_byte_ratio",
    "bwd_byte_ratio",
]

ALL_STATE_FEATURES = [
    "flow_count", "log_flow_count", "log_total_packets", "log_total_bytes",
    "window_packet_rate", "window_byte_rate", "syn_rate", "ack_rate",
    "rst_rate", "fin_rate", "psh_rate", "urg_rate", "mean_pkt_len",
    "std_pkt_len", "log_mean_flow_duration", "log_mean_fwd_iat",
    "log_mean_bwd_iat", "fwd_packet_ratio", "bwd_packet_ratio",
    "fwd_byte_ratio", "bwd_byte_ratio", "mean_down_up_ratio",
    "log_unique_dst_ports", "unique_protocols", "dst_port_entropy",
    "protocol_entropy",
]

# How many windows to show before and after onset
WINDOWS_BEFORE = 10   # 100 seconds before
WINDOWS_AFTER = 5     # 50 seconds after

ONSET_TIMESTAMPS = [
    "2018-03-02 03:54:50",
    "2018-03-02 10:13:20",
    "2018-03-02 12:13:30",
]

print("=" * 90)
print("PRE-ATTACK FEATURE TRAJECTORY ANALYSIS")
print("=" * 90)

# =============================================================
# LOAD MARCH 2 DATA
# =============================================================

print("\nLoading March 2 states...")
df = pd.read_csv(
    STATES_DIR / "Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    low_memory=False
)
df["window_start"] = pd.to_datetime(df["window_start"])
df = df.sort_values("window_start").reset_index(drop=True)

# Create a timestamp-to-index lookup
ts_to_idx = {ts: idx for idx, ts in enumerate(df["window_start"])}

print(f"  Total windows: {len(df)}")
print(f"  Time range: {df['window_start'].iloc[0]} to {df['window_start'].iloc[-1]}")

# =============================================================
# COMPUTE BASELINE STATISTICS (benign-only)
# =============================================================

benign_df = df[df["attack_present"] == 0]
print(f"  Benign windows: {len(benign_df)}")

benign_means = benign_df[ALL_STATE_FEATURES].mean()
benign_stds = benign_df[ALL_STATE_FEATURES].std()

# =============================================================
# EXTRACT TRAJECTORIES AROUND EACH ONSET
# =============================================================

all_trajectory_rows = []

for onset_ts_str in ONSET_TIMESTAMPS:
    onset_ts = pd.to_datetime(onset_ts_str)

    if onset_ts not in ts_to_idx:
        print(f"\n  WARNING: Onset {onset_ts} not found in states CSV!")
        continue

    onset_idx = ts_to_idx[onset_ts]

    # Verify this is actually an onset
    assert df.iloc[onset_idx]["attack_present"] == 1, \
        f"Expected attack_present=1 at onset {onset_ts}"
    if onset_idx > 0:
        # The window immediately before onset should be benign
        # (or at least the onset should be a 0->1 transition)
        prev_attack = df.iloc[onset_idx - 1]["attack_present"]
    else:
        prev_attack = 0

    print(f"\n{'=' * 90}")
    print(f"ONSET: {onset_ts}")
    print(f"{'=' * 90}")
    print(f"  Index: {onset_idx}")
    print(f"  Attack type: {df.iloc[onset_idx]['dominant_attack']}")
    print(f"  Attack ratio at onset: {df.iloc[onset_idx]['attack_ratio']:.4f}")
    print(f"  Previous window attack_present: {prev_attack}")

    # Determine context range
    start_idx = max(0, onset_idx - WINDOWS_BEFORE)
    end_idx = min(len(df), onset_idx + WINDOWS_AFTER + 1)

    # Check for preceding attack episode
    print(f"\n  Attack context (30 windows before onset):")
    context_start = max(0, onset_idx - 30)
    attack_history = df.iloc[context_start:onset_idx]["attack_present"].values
    n_attack_before = int(attack_history.sum())
    n_benign_before = len(attack_history) - n_attack_before
    print(f"    Attack windows in preceding 30: {n_attack_before}")
    print(f"    Benign windows in preceding 30: {n_benign_before}")

    if n_attack_before > 0:
        # Find the last attack window before onset
        last_attack_idx = None
        for j in range(onset_idx - 1, context_start - 1, -1):
            if df.iloc[j]["attack_present"] == 1:
                last_attack_idx = j
                break
        if last_attack_idx is not None:
            gap = onset_idx - last_attack_idx
            print(f"    Last attack before onset: {df.iloc[last_attack_idx]['window_start']} "
                  f"({gap * 10}s before onset)")
            print(f"    Attack type: {df.iloc[last_attack_idx]['dominant_attack']}")

    # =============================================================
    # FEATURE TABLE
    # =============================================================
    print(f"\n  FOCUS FEATURE TRAJECTORIES (raw values):")

    # Build header
    time_labels = []
    for i in range(start_idx, end_idx):
        delta = (i - onset_idx) * 10
        if delta < 0:
            label = f"t{delta}s"
        elif delta == 0:
            label = "t=ONSET"
        else:
            label = f"t+{delta}s"
        time_labels.append(label)

    # Print table header
    header = f"  {'Feature':<25}"
    for label in time_labels:
        header += f" {label:>10}"
    header += "  A(t)"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Print attack_present row first
    ap_row = f"  {'attack_present':<25}"
    for i in range(start_idx, end_idx):
        ap_row += f" {int(df.iloc[i]['attack_present']):>10}"
    print(ap_row)

    # Print attack_ratio row
    ar_row = f"  {'attack_ratio':<25}"
    for i in range(start_idx, end_idx):
        ar_row += f" {df.iloc[i]['attack_ratio']:>10.3f}"
    print(ar_row)

    print("  " + "-" * (len(header) - 2))

    # Print each focus feature
    for feat in FOCUS_FEATURES:
        row = f"  {feat:<25}"
        values = []
        for i in range(start_idx, end_idx):
            val = df.iloc[i][feat]
            values.append(val)
            if abs(val) >= 1000:
                row += f" {val:>10.0f}"
            elif abs(val) >= 10:
                row += f" {val:>10.1f}"
            elif abs(val) >= 1:
                row += f" {val:>10.3f}"
            else:
                row += f" {val:>10.4f}"
        row_values = np.array(values)
        print(row)

    # =============================================================
    # Z-SCORE ANALYSIS (relative to benign baseline)
    # =============================================================
    print(f"\n  Z-SCORES RELATIVE TO BENIGN BASELINE:")

    z_header = f"  {'Feature':<25}"
    for label in time_labels:
        z_header += f" {label:>10}"
    print(z_header)
    print("  " + "-" * (len(z_header) - 2))

    for feat in FOCUS_FEATURES:
        row = f"  {feat:<25}"
        for i in range(start_idx, end_idx):
            val = df.iloc[i][feat]
            z = (val - benign_means[feat]) / (benign_stds[feat] + 1e-10)
            if abs(z) >= 2.0:
                row += f" {z:>9.1f}*"
            else:
                row += f" {z:>10.2f}"
        print(row)

    # =============================================================
    # STORE ROWS FOR CSV
    # =============================================================
    for i in range(start_idx, end_idx):
        delta = (i - onset_idx) * 10
        row_data = {
            "onset_ts": str(onset_ts),
            "window_ts": str(df.iloc[i]["window_start"]),
            "delta_seconds": delta,
            "attack_present": int(df.iloc[i]["attack_present"]),
            "attack_ratio": float(df.iloc[i]["attack_ratio"]),
            "dominant_attack": str(df.iloc[i]["dominant_attack"]),
        }
        for feat in ALL_STATE_FEATURES:
            row_data[feat] = float(df.iloc[i][feat])
            z = (df.iloc[i][feat] - benign_means[feat]) / (benign_stds[feat] + 1e-10)
            row_data[f"{feat}_zscore"] = float(z)
        all_trajectory_rows.append(row_data)

# =============================================================
# SAVE CSV
# =============================================================

traj_df = pd.DataFrame(all_trajectory_rows)
csv_path = OUTPUT_DIR / "pre_attack_feature_trajectories.csv"
traj_df.to_csv(csv_path, index=False)
print(f"\nTrajectory CSV saved to: {csv_path}")

# =============================================================
# CHANGE ANALYSIS — GRADUAL vs ABRUPT vs ALREADY-ABNORMAL
# =============================================================

print("\n" + "=" * 90)
print("FEATURE CHANGE ANALYSIS PER ONSET")
print("=" * 90)

for onset_ts_str in ONSET_TIMESTAMPS:
    onset_ts = pd.to_datetime(onset_ts_str)
    onset_idx = ts_to_idx[onset_ts]

    print(f"\n--- Onset: {onset_ts} ---")

    # Windows for analysis
    # "Far benign": t-100s to t-60s (baseline, if available and benign)
    # "Near benign": t-50s to t-10s (pre-onset, should be benign)
    # "Onset": t

    far_start = max(0, onset_idx - 10)
    far_end = max(0, onset_idx - 5)
    near_start = max(0, onset_idx - 5)
    near_end = onset_idx

    # Check if far/near windows are actually benign
    far_benign = df.iloc[far_start:far_end]["attack_present"].values
    near_benign = df.iloc[near_start:near_end]["attack_present"].values

    far_has_attack = (far_benign.sum() > 0)
    near_has_attack = (near_benign.sum() > 0)

    print(f"  Far context (t-100s to t-60s): {'CONTAINS ATTACK' if far_has_attack else 'ALL BENIGN'}")
    print(f"  Near context (t-50s to t-10s): {'CONTAINS ATTACK' if near_has_attack else 'ALL BENIGN'}")

    # Compute feature means for each period
    far_vals = df.iloc[far_start:far_end][FOCUS_FEATURES].mean()
    near_vals = df.iloc[near_start:near_end][FOCUS_FEATURES].mean()
    onset_vals = df.iloc[onset_idx][FOCUS_FEATURES]

    gradually_changing = []
    abrupt_at_onset = []
    already_abnormal = []

    print(f"\n  {'Feature':<25} {'Far Mean':>10} {'Near Mean':>10} {'At Onset':>10} {'Near/Far':>10} {'Onset/Near':>10} {'Category':<20}")
    print("  " + "-" * 100)

    for feat in FOCUS_FEATURES:
        far_v = far_vals[feat]
        near_v = near_vals[feat]
        onset_v = onset_vals[feat]

        # Relative changes
        if abs(far_v) > 1e-6:
            near_far_ratio = near_v / far_v
        else:
            near_far_ratio = float("nan")

        if abs(near_v) > 1e-6:
            onset_near_ratio = onset_v / near_v
        else:
            onset_near_ratio = float("nan")

        # Z-scores relative to benign
        far_z = abs((far_v - benign_means[feat]) / (benign_stds[feat] + 1e-10))
        near_z = abs((near_v - benign_means[feat]) / (benign_stds[feat] + 1e-10))
        onset_z = abs((onset_v - benign_means[feat]) / (benign_stds[feat] + 1e-10))

        # Classification logic
        category = ""
        if far_z > 2.0 and near_z > 2.0:
            category = "ALREADY ABNORMAL"
            already_abnormal.append(feat)
        elif near_z > 1.5 and far_z < 1.0:
            category = "GRADUAL CHANGE"
            gradually_changing.append(feat)
        elif onset_z > 2.0 and near_z < 1.5:
            category = "ABRUPT AT ONSET"
            abrupt_at_onset.append(feat)
        elif abs(near_z - far_z) > 0.5 and near_z > far_z:
            category = "GRADUAL DRIFT"
            gradually_changing.append(feat)
        else:
            category = "STABLE"

        nfr_str = f"{near_far_ratio:.3f}" if not np.isnan(near_far_ratio) else "N/A"
        onr_str = f"{onset_near_ratio:.3f}" if not np.isnan(onset_near_ratio) else "N/A"

        print(f"  {feat:<25} {far_v:>10.3f} {near_v:>10.3f} {onset_v:>10.3f} "
              f"{nfr_str:>10} {onr_str:>10} {category:<20}")

    print(f"\n  Summary:")
    print(f"    Gradually changing: {gradually_changing if gradually_changing else 'None'}")
    print(f"    Abrupt at onset:    {abrupt_at_onset if abrupt_at_onset else 'None'}")
    print(f"    Already abnormal:   {already_abnormal if already_abnormal else 'None'}")


# =============================================================
# PLOT
# =============================================================

print("\nGenerating trajectory plots...")

# Select 12 key features for a 4x3 subplot grid
PLOT_FEATURES = [
    "flow_count", "window_packet_rate", "window_byte_rate",
    "syn_rate", "ack_rate", "rst_rate",
    "log_unique_dst_ports", "dst_port_entropy", "protocol_entropy",
    "mean_pkt_len", "log_mean_fwd_iat", "fwd_packet_ratio",
]

fig, axes = plt.subplots(4, 3, figsize=(22, 20))
fig.suptitle("Pre-Attack Feature Trajectories — March 2 Onsets", fontsize=16, fontweight="bold")

onset_colors = ["#e74c3c", "#2980b9", "#27ae60"]
onset_labels = ["03:54:50 (Brute Force)", "10:13:20 (?)", "12:13:30 (Infiltration)"]

for feat_idx, feat in enumerate(PLOT_FEATURES):
    row = feat_idx // 3
    col = feat_idx % 3
    ax = axes[row, col]

    for onset_i, onset_ts_str in enumerate(ONSET_TIMESTAMPS):
        onset_ts = pd.to_datetime(onset_ts_str)
        onset_idx = ts_to_idx[onset_ts]

        start_idx = max(0, onset_idx - WINDOWS_BEFORE)
        end_idx = min(len(df), onset_idx + WINDOWS_AFTER + 1)

        deltas = [(i - onset_idx) * 10 for i in range(start_idx, end_idx)]
        values = [df.iloc[i][feat] for i in range(start_idx, end_idx)]
        attacks = [int(df.iloc[i]["attack_present"]) for i in range(start_idx, end_idx)]

        ax.plot(deltas, values, marker="o", markersize=4, linewidth=1.5,
                color=onset_colors[onset_i], label=onset_labels[onset_i], alpha=0.85)

        # Mark attack windows with filled markers
        for j, (d, v, a) in enumerate(zip(deltas, values, attacks)):
            if a == 1:
                ax.plot(d, v, marker="s", markersize=6, color=onset_colors[onset_i],
                        alpha=0.4, zorder=1)

    # Add benign baseline band
    bm = benign_means[feat]
    bs = benign_stds[feat]
    ax.axhspan(bm - bs, bm + bs, color="#bdc3c7", alpha=0.2, label="Benign ±1σ" if feat_idx == 0 else None)
    ax.axhline(bm, color="#7f8c8d", linestyle="--", alpha=0.5, linewidth=0.8)

    # Onset line
    ax.axvline(0, color="black", linestyle=":", alpha=0.6, linewidth=1)

    ax.set_title(feat, fontsize=11, fontweight="bold")
    ax.set_xlabel("Seconds relative to onset", fontsize=9)
    ax.grid(True, alpha=0.3)

    if feat_idx == 0:
        ax.legend(fontsize=7, loc="upper left")

plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Add a note at the bottom
fig.text(0.5, 0.005,
         "Square markers = attack_present=1  |  Dashed line = benign mean  |  Gray band = benign ±1σ  |  Vertical line = onset",
         ha="center", fontsize=9, style="italic", color="#555")

plot_path = OUTPUT_DIR / "pre_attack_feature_trajectories.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Plot saved to: {plot_path}")

# =============================================================
# FINAL SUMMARY
# =============================================================

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
print(f"\nOutputs:")
print(f"  CSV: {csv_path}")
print(f"  Plot: {plot_path}")
print("=" * 90)
