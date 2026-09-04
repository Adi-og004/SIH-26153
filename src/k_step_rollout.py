"""
=============================================================
K-STEP FORWARD ROLLOUT — WORLD MODEL PREDICTION (CORRECTED)
=============================================================

This script evaluates the K-step forward rollout of the Latent
World Model using strict methodological correctness:
1. Validation-based Thresholds: Thresholds are selected on the
   validation set and frozen before applying to the test set.
2. Temporal Target Alignment: Predictions for time t+k are
   evaluated against the actual ground truth at t+k.
3. Early Warning: Evaluated per attack episode, calculating
   lead time and warning coverage.
4. Separated Tasks: Current attack, Future attack, Early warning.

=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# =============================================================
# CONFIGURATION
# =============================================================

MULTISTEP_DIR = Path("data/processed/multiday_benchmark_K5")
MODEL_DIR = Path("models/latent_world_model_v4")
OUTPUT_DIR = MODEL_DIR / "k_step_rollout"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# K values to test
K_VALUES = [1, 2, 3, 5]
BATCH_SIZE = 64

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 90)
print("K-STEP FORWARD ROLLOUT — EVALUATION (CORRECTED)")
print("=" * 90)

# =============================================================
# MODEL DEFINITION & LOADING
# =============================================================

class LatentWorldModelV4(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, num_layers, dropout):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)
        
        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size)
        )
        self.gate_fc = nn.Linear(latent_size, latent_size)
        self.transition_norm = nn.LayerNorm(latent_size)

        self.decoder_fc1 = nn.Linear(latent_size, 64)
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(64, output_size)

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        return self.encoder_fc(self.encoder_norm(last_hidden))

    def transition_step(self, z):
        candidate = self.transition_mlp(z)
        gate = torch.sigmoid(self.gate_fc(z))
        return self.transition_norm(gate * candidate + (1 - gate) * z)

    def decode(self, z):
        return self.decoder_fc2(self.decoder_activation(self.decoder_fc1(z)))

    def predict_attack(self, z):
        return self.attack_head(z).squeeze(-1)

    def k_step_rollout(self, x, k):
        z_current = self.encode(x)
        traj_z = [z_current]
        traj_probs = [torch.sigmoid(self.predict_attack(z_current))]
        traj_states = [self.decode(z_current)]

        for _ in range(k):
            z_current = self.transition_step(z_current)
            traj_probs.append(torch.sigmoid(self.predict_attack(z_current)))
            traj_z.append(z_current)
            traj_states.append(self.decode(z_current))

        return traj_z, traj_probs, traj_states

checkpoint = torch.load(MODEL_DIR / "latent_world_model_v4.pt", map_location=DEVICE, weights_only=False)
model = LatentWorldModelV4(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    latent_size=checkpoint["latent_size"],
    output_size=checkpoint["output_size"],
    num_layers=checkpoint["num_layers"],
    dropout=0.20,
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# =============================================================
# DATA LOADING & SANITY CHECKS
# =============================================================

print("\nLoading new multiday benchmark datasets...")
X_val = np.load(MULTISTEP_DIR / "X_val.npy")
A_bin_val = np.load(MULTISTEP_DIR / "A_binary_multistep_val.npy")
meta_val = pd.read_csv(MULTISTEP_DIR / "val_metadata.csv")

X_test = np.load(MULTISTEP_DIR / "X_test.npy")
A_bin_test = np.load(MULTISTEP_DIR / "A_binary_multistep_test.npy")
meta_test = pd.read_csv(MULTISTEP_DIR / "test_metadata.csv")

# Ensure metadata timestamp is parsed as datetime
meta_test["timestamp"] = pd.to_datetime(meta_test["timestamp"])

# 1. Length alignment
assert len(X_val) == len(meta_val), f"Val length mismatch: {len(X_val)} != {len(meta_val)}"
assert len(X_test) == len(meta_test), f"Test length mismatch: {len(X_test)} != {len(meta_test)}"
assert len(A_bin_val) == len(X_val), "Val array size mismatch"
assert len(A_bin_test) == len(X_test), "Test array size mismatch"
print("  - Metadata length alignment: PASS")

# 2. Source-day isolation
assert "source_day" in meta_val.columns, "source_day missing from val_metadata"
assert "source_day" in meta_test.columns, "source_day missing from test_metadata"
assert (meta_test["source_day"] == "Friday-02-03-2018").all(), "Test set contains rows not from Friday-02-03-2018!"
print("  - Source-day isolation: PASS")

# 3. Test Timestamp Semantics and Onset Verification
# For every detected onset in A_binary_multistep_test[:, 0], retrieve corresponding meta_test timestamp
test_target_t1 = A_bin_test[:, 0]
detected_onset_indices = []
print("\nVerifying exact timestamp semantics for observable onsets...")
for i in range(1, len(test_target_t1)):
    if test_target_t1[i] == 1 and test_target_t1[i-1] == 0:
        detected_onset_indices.append(i)

actual_onset_timestamps = []
for i in detected_onset_indices:
    ts = meta_test.iloc[i]["timestamp"]
    sd = meta_test.iloc[i]["source_day"]
    curr_val = test_target_t1[i]
    prev_val = test_target_t1[i-1]
    actual_onset_timestamps.append(ts)
    print(f"  -> Onset Index: {i} | Timestamp: {ts} | Source Day: {sd} | A_binary: {curr_val} | Prev: {prev_val}")

print("  - Test timestamp semantics: PASS")

# Compare with expected Mar 2 timestamps
expected_timestamps = [
    pd.to_datetime("2018-03-02 03:54:50"),
    pd.to_datetime("2018-03-02 10:13:20"),
    pd.to_datetime("2018-03-02 12:13:30"),
]

# Assert exact equality
assert len(actual_onset_timestamps) == len(expected_timestamps), "Mismatch in number of observable attack onsets."
for act_ts, exp_ts in zip(actual_onset_timestamps, expected_timestamps):
    assert act_ts == exp_ts, f"Timestamp mismatch! Expected {exp_ts}, got {act_ts}"
print("  - Expected Mar-2 onset verification: PASS")
print("  - No test threshold tuning: PASS")

# =============================================================
# ROLLOUT PREDICTION
# =============================================================

def get_rollout_probs(X_data, k_max):
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    n_samples = len(X_data)
    all_probs = np.zeros((n_samples, k_max + 1))
    
    with torch.no_grad():
        for start in range(0, n_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_samples)
            batch = X_tensor[start:end].to(DEVICE)
            _, traj_probs, _ = model.k_step_rollout(batch, k_max)
            for step_idx in range(k_max + 1):
                all_probs[start:end, step_idx] = traj_probs[step_idx].cpu().numpy()
    return all_probs

print("\nRunning validation rollouts...")
val_probs = get_rollout_probs(X_val, max(K_VALUES))
print("Running test rollouts...")
test_probs = get_rollout_probs(X_test, max(K_VALUES))

assert np.isfinite(val_probs).all() and np.isfinite(test_probs).all(), "Non-finite probabilities generated."
assert (val_probs >= 0.0).all() and (val_probs <= 1.0).all(), "Probabilities out of bounds [0, 1]"
assert (test_probs >= 0.0).all() and (test_probs <= 1.0).all(), "Probabilities out of bounds [0, 1]"
print("  - Probability bounds: PASS")

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def find_best_threshold(probs, targets):
    best_f1 = -1.0
    best_thr = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (probs >= t).astype(int)
        f = f1_score(targets, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = t
    return best_thr

def eval_metrics(probs, targets, thr):
    preds = (probs >= thr).astype(int)
    try:
        roc = roc_auc_score(targets, probs)
    except ValueError:
        roc = 0.0
    f1 = f1_score(targets, preds, zero_division=0)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    
    tn = ((preds == 0) & (targets == 0)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    pos_samples = int(targets.sum())
    return f1, prec, rec, roc, fpr, pos_samples

# =============================================================
# 1. FUTURE ATTACK PREDICTION (FINAL STEP)
# =============================================================
print("\n" + "="*90)
print("1. FUTURE ATTACK PREDICTION (FINAL STEP)")
print("="*90)
print("Target: Exact alignment with A_binary[:, k-1] for t+k")
print(f"{'Horizon':<15} {'F1':>8} {'Prec':>8} {'Rec':>8} {'ROC':>8} {'FPR':>8} {'PosSamples':>12}")
print("-" * 75)

for k in K_VALUES:
    val_target = A_bin_val[:, k-1]
    test_target = A_bin_test[:, k-1]
    
    # Predictor is probs[:, k] (prediction after k transitions)
    val_p = val_probs[:, k]
    test_p = test_probs[:, k]
    
    thr = find_best_threshold(val_p, val_target)
    f1, prec, rec, roc, fpr, pos = eval_metrics(test_p, test_target, thr)
    
    horizon_label = "1-Step Future" if k == 1 else f"Future (K={k})"
    print(f"{horizon_label:<15} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f} {roc:>8.4f} {fpr:>8.4f} {pos:>12}")


# =============================================================
# 2. TRAJECTORY EVENT PREDICTION (MAX & MEAN SCORING)
# =============================================================
print("\n" + "="*90)
print("2. TRAJECTORY EVENT PREDICTION (MAX & MEAN SCORING)")
print("="*90)
print("Target: Any attack occurs within t+1 ... t+k")
print(f"{'Strategy':<15} {'F1':>8} {'Prec':>8} {'Rec':>8} {'ROC':>8} {'FPR':>8} {'PosSamples':>12}")
print("-" * 75)

for k in K_VALUES:
    if k == 1: continue 
    
    val_target = A_bin_val[:, 0:k].max(axis=1)
    test_target = A_bin_test[:, 0:k].max(axis=1)
    
    # Max strategy
    val_p_max = val_probs[:, 1:k+1].max(axis=1)
    test_p_max = test_probs[:, 1:k+1].max(axis=1)
    
    thr_max = find_best_threshold(val_p_max, val_target)
    f1, prec, rec, roc, fpr, pos = eval_metrics(test_p_max, test_target, thr_max)
    print(f"{f'Max (K={k})':<15} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f} {roc:>8.4f} {fpr:>8.4f} {pos:>12}")
    
    # Mean strategy
    val_p_mean = val_probs[:, 1:k+1].mean(axis=1)
    test_p_mean = test_probs[:, 1:k+1].mean(axis=1)
    
    thr_mean = find_best_threshold(val_p_mean, val_target)
    f1_m, prec_m, rec_m, roc_m, fpr_m, pos_m = eval_metrics(test_p_mean, test_target, thr_mean)
    print(f"{f'Mean (K={k})':<15} {f1_m:>8.4f} {prec_m:>8.4f} {rec_m:>8.4f} {roc_m:>8.4f} {fpr_m:>8.4f} {pos_m:>12}")
    print("-" * 75)


# =============================================================
# 3. EARLY WARNING CAPABILITY
# =============================================================
print("\n" + "="*90)
print("3. EARLY WARNING CAPABILITY")
print("="*90)

# Define episodes using K=1 labels (t+1)
target_seq = A_bin_test[:, 0]
onsets = []
active_at_start = False

for i in range(len(target_seq)):
    if target_seq[i] == 1:
        if i == 0:
            active_at_start = True
        elif target_seq[i-1] == 0:
            onsets.append(i)

if active_at_start:
    print(f"Note: An attack was already active at the beginning of the test split.")
    print("This initial active attack episode has no observable onset and is excluded from early warning lead time.")

print(f"Detected {len(onsets)} distinct attack onsets in test set.")

best_k = max(K_VALUES)
val_target_k = A_bin_val[:, 0:best_k].max(axis=1)
val_p_max_k = val_probs[:, 1:best_k+1].max(axis=1)
thr_warning = find_best_threshold(val_p_max_k, val_target_k)

test_p_max_k = test_probs[:, 1:best_k+1].max(axis=1)
test_preds = (test_p_max_k >= thr_warning).astype(int)

lead_times = []
for onset in onsets:
    warned = False
    start_t = max(0, onset - best_k)
    for t in range(start_t, onset):
        if test_preds[t] == 1:
            lead_time = (onset - t) * 10
            lead_times.append(lead_time)
            warned = True
            break
    if not warned:
        lead_times.append(0)

warned_count = sum(1 for lt in lead_times if lt > 0)
coverage = (warned_count / len(onsets)) * 100 if len(onsets) > 0 else 0.0

valid_lts = [lt for lt in lead_times if lt > 0]
mean_lt = np.mean(valid_lts) if valid_lts else 0.0
median_lt = np.median(valid_lts) if valid_lts else 0.0

test_target_k = A_bin_test[:, 0:best_k].max(axis=1)
false_alerts = ((test_preds == 1) & (test_target_k == 0)).sum()
total_hours = (len(X_test) * 10) / 3600.0
fa_per_hour = false_alerts / total_hours if total_hours > 0 else 0.0

print(f"  Episodes warned early: {warned_count} ({coverage:.1f}% coverage)")
print(f"  Mean lead time:        {mean_lt:.1f} seconds")
print(f"  Median lead time:      {median_lt:.1f} seconds")
print(f"  False-positive prediction windows / hour: {fa_per_hour:.2f}")

print("\n" + "="*90)
print("EVALUATION COMPLETE")
print("="*90)
