"""
=============================================================
PREPARE MULTI-DAY BENCHMARK DATA — V4 WORLD MODEL
=============================================================

This script replaces the legacy temporal_benchmark and multistep
data preparation scripts. It implements a strict multi-day
generalization setup:
1. Loads specified Train and Test days independently.
2. Computes state transition features per-day to prevent leakage.
3. Splits Train days chronologically into Train (85%) and Val (15%).
4. Fits scalers strictly on the Train partition.
5. Builds K-step sequences with rigorous 10-second adjacency validation.
6. Exports compatible V4 arrays.

"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data/processed/states_v2")
OUTPUT_DIR = Path("data/processed/multiday_benchmark_K5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states_v2.csv"
]

TEST_DAYS = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv"
]

VAL_RATIO = 0.15
SEQUENCE_LENGTH = 10
MAX_K = 5

STATE_FEATURES = [
    "flow_count", "log_flow_count", "log_total_packets", "log_total_bytes",
    "window_packet_rate", "window_byte_rate", "syn_rate", "ack_rate",
    "rst_rate", "fin_rate", "psh_rate", "urg_rate", "mean_pkt_len",
    "std_pkt_len", "log_mean_flow_duration", "log_mean_fwd_iat",
    "log_mean_bwd_iat", "fwd_packet_ratio", "bwd_packet_ratio",
    "fwd_byte_ratio", "bwd_byte_ratio", "mean_down_up_ratio",
    "log_unique_dst_ports", "unique_protocols", "dst_port_entropy",
    "protocol_entropy",
]

# =========================================================
# DATA PROCESSING FUNCTIONS
# =========================================================

def load_and_prep_day(filename):
    file_path = DATA_DIR / filename
    print(f"Loading {filename}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
    df = df.dropna(subset=["window_start"])
    df = df.sort_values("window_start").reset_index(drop=True)
    
    # Extract source day (e.g. Wednesday-14-02-2018)
    source_day = filename.split("_")[0]
    df["source_day"] = source_day

    for feature in STATE_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    
    df[STATE_FEATURES] = df[STATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # =========================================================
    # RELATIVE CHANGE DESIGN DOCUMENTATION
    # =========================================================
    # Relative-change features are intentionally computed 
    # chronologically within each source day BEFORE the 85/15 
    # split because the immediately preceding observed network 
    # state is legitimate historical information available at 
    # inference time. Do not silently reset validation's first 
    # relative-change row unless the methodology is explicitly 
    # changed.
    # =========================================================
    state = df[STATE_FEATURES].copy()
    delta = state.diff().fillna(0.0)
    previous_state = state.shift(1)
    relative_change = delta / (previous_state.abs() + 1e-8)
    relative_change.iloc[0] = 0.0
    relative_change = relative_change.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
    relative_change.columns = [f"{f}_rel_change" for f in STATE_FEATURES]
    
    df = pd.concat([df, relative_change], axis=1)
    
    return df, list(relative_change.columns)

# =========================================================
# MAIN PIPELINE
# =========================================================
print("=" * 90)
print("MULTI-DAY BENCHMARK PREPARATION")
print("=" * 90)

train_dfs = []
val_dfs = []
test_dfs = []
TRANSITION_FEATURES = None
INPUT_FEATURES = None

print("\n--- Processing Train Days ---")
for day in TRAIN_DAYS:
    df, trans_feats = load_and_prep_day(day)
    if TRANSITION_FEATURES is None:
        TRANSITION_FEATURES = trans_feats
        INPUT_FEATURES = STATE_FEATURES + TRANSITION_FEATURES
        
    n_states = len(df)
    train_end = int(n_states * (1.0 - VAL_RATIO))
    print(f"  -> Total rows: {n_states} | Train rows: {train_end} | Val rows: {n_states - train_end}")
    train_dfs.append(df.iloc[:train_end].copy())
    val_dfs.append(df.iloc[train_end:].copy())
    
print("\n--- Processing Test Days ---")
for day in TEST_DAYS:
    df, _ = load_and_prep_day(day)
    print(f"  -> Total rows: {len(df)}")
    test_dfs.append(df.copy())

# Concatenate partitions (no temporal mixing is possible during sequence gen due to strict timestamp check)
train_df = pd.concat(train_dfs, ignore_index=True)
val_df = pd.concat(val_dfs, ignore_index=True)
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"\nTotal Train rows: {len(train_df)}")
print(f"Total Val rows: {len(val_df)}")
print(f"Total Test rows: {len(test_df)}")

print("\nFitting scalers on final Train partition only...")
input_scaler = StandardScaler()
input_scaler.fit(train_df[INPUT_FEATURES])

target_scaler = StandardScaler()
target_scaler.fit(train_df[STATE_FEATURES])

def transform_df(df_split):
    inputs = input_scaler.transform(df_split[INPUT_FEATURES])
    inputs = np.nan_to_num(inputs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    
    targets = target_scaler.transform(df_split[STATE_FEATURES])
    targets = np.nan_to_num(targets, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
    return inputs, targets

train_inputs, train_targets = transform_df(train_df)
val_inputs, val_targets = transform_df(val_df)
test_inputs, test_targets = transform_df(test_df)

def create_sequences(inputs, targets, dataframe):
    X, Y_multi, A_soft, A_binary, meta_records = [], [], [], [], []
    
    timestamps = dataframe["window_start"].values
    attack_present = dataframe["attack_present"].values
    attack_ratio = dataframe["attack_ratio"].values
    dominant_attack = dataframe["dominant_attack"].values
    source_days = dataframe["source_day"].values
    
    valid_count = 0
    rejected_count = 0
    
    # We need SEQ_LEN inputs + MAX_K targets = SEQ_LEN + MAX_K contiguous timestamps
    for start_idx in range(len(dataframe) - SEQUENCE_LENGTH - MAX_K + 1):
        end_idx = start_idx + SEQUENCE_LENGTH
        
        # Check strict 10s contiguity for the entire block (inputs + targets)
        block_times = timestamps[start_idx : end_idx + MAX_K]
        diffs = np.diff(block_times).astype('timedelta64[s]').astype(int)
        
        if np.all(diffs == 10):
            valid_count += 1
            X.append(inputs[start_idx:end_idx])
            
            y_block = []
            a_soft_block = []
            a_bin_block = []
            
            for k in range(MAX_K):
                idx = end_idx + k
                y_block.append(targets[idx])
                a_soft_block.append(float(attack_ratio[idx]))
                a_bin_block.append(int(attack_present[idx]))
                
            Y_multi.append(y_block)
            A_soft.append(a_soft_block)
            A_binary.append(a_bin_block)
            
            meta_records.append({
                "timestamp": timestamps[end_idx], # Metadata of target t+1
                "attack_present": attack_present[end_idx],
                "attack_ratio": attack_ratio[end_idx],
                "dominant_attack": dominant_attack[end_idx],
                "source_day": source_days[end_idx],
            })
        else:
            rejected_count += 1
            
    print(f"  Valid sequences: {valid_count} | Rejected sequences (gaps/boundaries): {rejected_count}")
    
    X = np.asarray(X, dtype=np.float32)
    Y_multi = np.asarray(Y_multi, dtype=np.float32)
    A_soft = np.asarray(A_soft, dtype=np.float32)
    A_binary = np.asarray(A_binary, dtype=np.float32)
    meta_df = pd.DataFrame(meta_records)
    
    return X, Y_multi, A_soft, A_binary, meta_df

print("\nBuilding Train sequences...")
X_train, Y_train, A_soft_train, A_binary_train, meta_train = create_sequences(train_inputs, train_targets, train_df)

print("\nBuilding Validation sequences...")
X_val, Y_val, A_soft_val, A_binary_val, meta_val = create_sequences(val_inputs, val_targets, val_df)

print("\nBuilding Test sequences...")
X_test, Y_test, A_soft_test, A_binary_test, meta_test = create_sequences(test_inputs, test_targets, test_df)

# =========================================================
# TEST ONSET VALIDATION REPORTING
# =========================================================
print("\n" + "=" * 90)
print("TEST SET ONSET VALIDATION")
print("=" * 90)

# Raw test dataframe stats
raw_test_attacks = test_df["attack_present"].astype(int).values
raw_test_timestamps = test_df["window_start"].values
raw_onset_indices = np.where(np.diff(raw_test_attacks) == 1)[0] + 1 # +1 because diff is shifted

print(f"Raw Mar 2 windows: {len(test_df)}")
print(f"Raw attack onsets: {len(raw_onset_indices)}")
print(f"Raw total attack windows: {raw_test_attacks.sum()}")

if len(raw_onset_indices) > 0:
    print("Raw attack onset timestamps:")
    for idx in raw_onset_indices:
        print(f"  -> {raw_test_timestamps[idx]}")

# Evaluable test stats
print(f"\nFinal evaluable test windows (sequences): {len(A_binary_test)}")
if len(A_binary_test) > 0:
    # First target window t+1
    eval_t1_attacks = A_binary_test[:, 0].astype(int)
    eval_onset_indices = np.where(np.diff(eval_t1_attacks) == 1)[0] + 1
    
    print(f"Final observable attack onsets (t+1): {len(eval_onset_indices)}")
    print(f"Final total attack sequences (t+1): {eval_t1_attacks.sum()}")
    
    if len(eval_onset_indices) > 0:
        print("Final observable attack onset timestamps:")
        for idx in eval_onset_indices:
            print(f"  -> {meta_test.iloc[idx]['timestamp']}")
            
    is_active = (eval_t1_attacks[0] == 1)
    print(f"\nIs attack already active at first evaluable target window? {'YES' if is_active else 'NO'}")
else:
    print("\nNO VALID TEST SEQUENCES!")

# =========================================================
# SAVE TO DISK
# =========================================================
print("\nSaving arrays...")
for split_name, (x, y, asoft, abin, m) in zip(
    ["train", "val", "test"],
    [(X_train, Y_train, A_soft_train, A_binary_train, meta_train),
     (X_val, Y_val, A_soft_val, A_binary_val, meta_val),
     (X_test, Y_test, A_soft_test, A_binary_test, meta_test)]
):
    np.save(OUTPUT_DIR / f"X_{split_name}.npy", x)
    np.save(OUTPUT_DIR / f"Y_multistep_{split_name}.npy", y)
    np.save(OUTPUT_DIR / f"A_soft_multistep_{split_name}.npy", asoft)
    np.save(OUTPUT_DIR / f"A_binary_multistep_{split_name}.npy", abin)
    m.to_csv(OUTPUT_DIR / f"{split_name}_metadata.csv", index=False)
    print(f"Saved {split_name} (X shape: {x.shape}, Y_multistep shape: {y.shape}, A_binary shape: {abin.shape})")

with open(OUTPUT_DIR / "input_scaler.pkl", "wb") as f:
    pickle.dump(input_scaler, f)
with open(OUTPUT_DIR / "target_scaler.pkl", "wb") as f:
    pickle.dump(target_scaler, f)
with open(OUTPUT_DIR / "input_feature_names.txt", "w") as f:
    f.write("\n".join(INPUT_FEATURES) + "\n")
with open(OUTPUT_DIR / "state_feature_names.txt", "w") as f:
    f.write("\n".join(STATE_FEATURES) + "\n")

print("\n" + "=" * 90)
print(f"MULTI-DAY BENCHMARK DATA READY IN: {OUTPUT_DIR}")
print("=" * 90)
