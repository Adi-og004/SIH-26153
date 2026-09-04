"""
=============================================================
PREPARE MULTI-STEP TRAINING DATA — V4 WORLD MODEL
=============================================================

Constructs multi-step training targets from the existing
temporal_benchmark_v2 data for V4 curriculum training.

For each window t that has K future windows available,
this script creates:
  - X_t:       input sequence (10 timesteps x 52 features)
  - Y_targets: K future output states [Y_{t+1}, ..., Y_{t+K}]
  - A_targets: K future attack labels [A_{t+1}, ..., A_{t+K}]

The existing temporal_benchmark_v2 contains CONSECUTIVE
10-second windows from Feb 28, 2018. Window i+1 is the
genuine next state after window i.

IMPORTANT: This script drops the last K windows from each
split (they don't have K future targets). This is negligible
for K=5 (~5 samples lost from each split).

=============================================================
"""

import shutil

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================
# CONFIGURATION
# =============================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

MAX_K = 5  # Maximum rollout horizon

OUTPUT_DIR = DATA_DIR / f"multistep_K{MAX_K}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================
# MAIN
# =============================================================

print("=" * 90)
print(f"PREPARE MULTI-STEP DATA (K={MAX_K})")
print("=" * 90)


for split in ["train", "val", "test"]:

    print(f"\n--- Processing {split} ---")

    X = np.load(DATA_DIR / f"X_{split}.npy")
    Y = np.load(DATA_DIR / f"Y_{split}.npy")
    meta = pd.read_csv(DATA_DIR / f"{split}_metadata.csv")

    n_samples = X.shape[0]
    n_valid = n_samples - MAX_K + 1

    print(f"  Original samples: {n_samples}")
    print(f"  Valid samples (with {MAX_K} future steps): {n_valid}")

    if n_valid <= 0:
        raise ValueError(
            f"{split} has only {n_samples} samples, "
            f"need at least {MAX_K + 1} for K={MAX_K}."
        )

    # ----- Build multi-step targets -----
    # X_out[i] = X[i]  (input doesn't change)
    # Y[i] is already t+1 (the immediate next step for X[i])
    # Y_out[i] = [Y[i], Y[i+1], ..., Y[i+K-1]]
    # A_out[i] = [A[i], A[i+1], ..., A[i+K-1]]

    X_out = X[:n_valid]

    Y_multistep = np.zeros(
        (n_valid, MAX_K, Y.shape[1]),
        dtype=np.float32
    )

    # Get attack_ratio (soft target) from metadata
    attack_ratio = (
        meta["attack_ratio"]
        .to_numpy(dtype=np.float32)
    )

    A_multistep = np.zeros(
        (n_valid, MAX_K),
        dtype=np.float32
    )

    # Get binary labels for evaluation
    attack_binary = (
        meta["attack_present"]
        .to_numpy(dtype=np.float32)
    )

    A_binary_multistep = np.zeros(
        (n_valid, MAX_K),
        dtype=np.float32
    )

    for k in range(MAX_K):
        offset = k
        Y_multistep[:, k, :] = Y[offset:offset + n_valid]
        A_multistep[:, k] = attack_ratio[offset:offset + n_valid]
        A_binary_multistep[:, k] = attack_binary[offset:offset + n_valid]

    # ----- Truncated metadata -----
    meta_out = meta.iloc[:n_valid].copy()

    # ----- Save -----
    np.save(OUTPUT_DIR / f"X_{split}.npy", X_out)
    np.save(OUTPUT_DIR / f"Y_multistep_{split}.npy", Y_multistep)
    np.save(OUTPUT_DIR / f"A_soft_multistep_{split}.npy", A_multistep)
    np.save(OUTPUT_DIR / f"A_binary_multistep_{split}.npy", A_binary_multistep)
    meta_out.to_csv(OUTPUT_DIR / f"{split}_metadata.csv", index=False)

    print(f"  X_{split}:              {X_out.shape}")
    print(f"  Y_multistep_{split}:    {Y_multistep.shape}")
    print(f"  A_soft_multistep_{split}: {A_multistep.shape}")
    print(f"  A_binary_multistep_{split}: {A_binary_multistep.shape}")

    # Quick sanity check
    print(f"  Attack ratio stats (step 1): "
          f"mean={A_multistep[:, 0].mean():.4f}, "
          f"max={A_multistep[:, 0].max():.4f}")
    print(f"  Attack ratio stats (step {MAX_K}): "
          f"mean={A_multistep[:, -1].mean():.4f}, "
          f"max={A_multistep[:, -1].max():.4f}")

    # Sanity check timestamps for the first sequence
    print("\n  [Sanity Check] Timestamp Alignment for Sequence 0:")
    timestamps = pd.to_datetime(meta["timestamp"].values)
    
    x_end_time = timestamps[0] - pd.Timedelta(seconds=10)
    print(f"    X[0] end timestamp: {x_end_time}")
    
    for k in range(MAX_K):
        y_time = timestamps[k]
        expected_time = x_end_time + pd.Timedelta(seconds=(k+1)*10)
        print(f"    Y{k+1} (t+{k+1}) target timestamp: {y_time} (Expected: {expected_time})")
        
        if y_time != expected_time:
            raise ValueError(f"Timestamp misalignment at step {k+1}: Expected {expected_time}, got {y_time}")


# ----- Copy shared files -----

for shared_file in [
    "input_feature_names.txt",
    "state_feature_names.txt",
    "input_scaler.pkl",
    "target_scaler.pkl",
]:
    src = DATA_DIR / shared_file
    if src.exists():
        shutil.copy2(src, OUTPUT_DIR / shared_file)
        print(f"\nCopied: {shared_file}")


print(f"\n{'=' * 90}")
print(f"MULTI-STEP DATA READY: {OUTPUT_DIR}")
print(f"{'=' * 90}")
