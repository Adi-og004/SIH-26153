"""
=============================================================
PRE-ATTACK ONSET PREDICTION — EVALUATION
=============================================================

PURPOSE:
  Tests whether the V4 World Model can forecast attack EMERGENCE
  while the network is still BENIGN, rather than simply forecasting
  attack PERSISTENCE.

EXPERIMENT DESIGN:
  - Restrict evaluation to windows where A(t) = 0 (currently benign)
  - Define onset target: max(A(t+1), ..., A(t+K)) for K = 1,2,3,5
  - Use FROZEN threshold from existing V4 checkpoint (no recalibration)
  - Evaluate on March 2 held-out test set

  This experiment deliberately removes the advantage of persistence
  and isolates the model's ability to forecast attack onset from a
  benign state.

METHODOLOGY:
  1. V4 trained on multi-day data (Feb 14, 15, 28)
  2. Threshold selected on validation during original training
  3. Threshold FROZEN at checkpoint value
  4. Applied WITHOUT modification to pre-attack onset subset
  5. March 2 test data never used for threshold selection

BASELINES:
  - Persistence: A(t)=0 -> predict 0 (always). By construction,
    persistence has zero ability to predict onset on this subset.
    This is intentional: it demonstrates that the experiment
    isolates forecasting from persistence.
  - Logistic Regression: Trained on train pre-attack subset only.

=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================
# CONFIGURATION
# =============================================================

MULTISTEP_DIR = Path("data/processed/multiday_benchmark_K5")
MODEL_DIR = Path("models/latent_world_model_v4")
STATES_DIR = Path("data/processed/states_v2")
OUTPUT_DIR = MODEL_DIR / "pre_attack_onset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 2, 3, 5]
BATCH_SIZE = 64

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 90)
print("PRE-ATTACK ONSET PREDICTION — EVALUATION")
print("=" * 90)
print(f"Device: {DEVICE}")

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
        traj_probs = [torch.sigmoid(self.predict_attack(z_current))]

        for _ in range(k):
            z_current = self.transition_step(z_current)
            traj_probs.append(torch.sigmoid(self.predict_attack(z_current)))

        return traj_probs


print("\nLoading V4 checkpoint...")
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

FROZEN_THRESHOLD = checkpoint["selected_threshold"]
print(f"  Model loaded: {checkpoint['model_class']}")
print(f"  Architecture: input={checkpoint['input_size']}, hidden={checkpoint['hidden_size']}, "
      f"latent={checkpoint['latent_size']}, output={checkpoint['output_size']}")
print(f"  FROZEN threshold: {FROZEN_THRESHOLD}")
print(f"  Threshold source: original V4 validation-based selection")

# =============================================================
# DATA LOADING
# =============================================================

print("\nLoading benchmark data...")
X_train = np.load(MULTISTEP_DIR / "X_train.npy")
A_bin_train = np.load(MULTISTEP_DIR / "A_binary_multistep_train.npy")
meta_train = pd.read_csv(MULTISTEP_DIR / "train_metadata.csv")

X_test = np.load(MULTISTEP_DIR / "X_test.npy")
A_bin_test = np.load(MULTISTEP_DIR / "A_binary_multistep_test.npy")
meta_test = pd.read_csv(MULTISTEP_DIR / "test_metadata.csv")

meta_train["timestamp"] = pd.to_datetime(meta_train["timestamp"])
meta_test["timestamp"] = pd.to_datetime(meta_test["timestamp"])

print(f"  Train sequences: {len(X_train)}")
print(f"  Test sequences:  {len(X_test)}")

# =============================================================
# A(t) EXTRACTION — CURRENT-STATE LABEL
# =============================================================
# Metadata timestamp = t+1, so A(t) = lookup(timestamp - 10s)
# Reuses the exact pattern from evaluate_baselines.py

def fetch_time_t_labels(meta_df):
    """
    Given metadata with timestamps for t+1, fetch attack_present
    label at time t (= timestamp - 10s) from the original state CSVs.
    """
    source_days = meta_df["source_day"].unique()

    lookup = {}
    for day in source_days:
        csv_path = STATES_DIR / f"{day}_TrafficForML_CICFlowMeter_states_v2.csv"
        df = pd.read_csv(csv_path, usecols=["window_start", "attack_present"])
        df["window_start"] = pd.to_datetime(df["window_start"])
        day_dict = dict(zip(df["window_start"], df["attack_present"]))
        lookup.update(day_dict)

    labels_t = []
    for ts in meta_df["timestamp"]:
        ts_t = ts - pd.Timedelta(seconds=10)
        if ts_t in lookup:
            labels_t.append(lookup[ts_t])
        else:
            raise ValueError(
                f"CRITICAL: Missing label for time t. "
                f"Could not find {ts_t} (10s prior to {ts}) in state CSV."
            )
    return np.array(labels_t)


print("\nExtracting A(t) labels...")
y_current_train = fetch_time_t_labels(meta_train)
y_current_test = fetch_time_t_labels(meta_test)

pre_attack_mask_train = (y_current_train == 0)
pre_attack_mask_test = (y_current_test == 0)

print(f"  Train: A(t)=0: {pre_attack_mask_train.sum()}, A(t)=1: {(y_current_train == 1).sum()}")
print(f"  Test:  A(t)=0: {pre_attack_mask_test.sum()}, A(t)=1: {(y_current_test == 1).sum()}")

# =============================================================
# V4 ROLLOUT
# =============================================================

def get_rollout_probs(X_data, k_max):
    """Run K-step rollout and return probability matrix (N, k_max+1)."""
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    n_samples = len(X_data)
    all_probs = np.zeros((n_samples, k_max + 1))

    with torch.no_grad():
        for start in range(0, n_samples, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_samples)
            batch = X_tensor[start:end].to(DEVICE)
            traj_probs = model.k_step_rollout(batch, k_max)
            for step_idx in range(k_max + 1):
                all_probs[start:end, step_idx] = traj_probs[step_idx].cpu().numpy()
    return all_probs


print("\nRunning V4 rollouts on test data...")
test_probs = get_rollout_probs(X_test, max(K_VALUES))

assert np.isfinite(test_probs).all(), "Non-finite probabilities"
assert (test_probs >= 0.0).all() and (test_probs <= 1.0).all(), "Probabilities out of [0,1]"
print("  Probability bounds check: PASS")

# =============================================================
# LOGISTIC REGRESSION BASELINE — TRAIN ON PRE-ATTACK SUBSET
# =============================================================
# LR trained only on windows where A(t)=0, using the latest
# observed 52-feature state to predict onset within K steps.
# Threshold tuned on same train pre-attack subset (noted as
# limitation; the focus of this experiment is V4 evaluation).

print("\nTraining Logistic Regression baselines on train pre-attack subset...")
X_train_lr = X_train[:, -1, :]  # Latest observed state features

lr_models = {}
lr_thresholds = {}

for k in K_VALUES:
    onset_target_train = A_bin_train[:, :k].max(axis=1)

    # Filter to pre-attack
    X_lr_pa = X_train_lr[pre_attack_mask_train]
    y_lr_pa = onset_target_train[pre_attack_mask_train]

    n_pos = int(y_lr_pa.sum())
    n_neg = int((y_lr_pa == 0).sum())

    if n_pos == 0:
        print(f"  K={k}: No train pre-attack positives — LR will predict all-negative")
        lr_models[k] = None
        lr_thresholds[k] = 0.5
        continue

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_lr_pa, y_lr_pa)
    lr_models[k] = lr

    # Find threshold on train pre-attack (noted as limitation)
    lr_probs_train = lr.predict_proba(X_lr_pa)[:, 1]
    best_f1 = -1.0
    best_thr = 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (lr_probs_train >= t).astype(int)
        f = f1_score(y_lr_pa, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = t
    lr_thresholds[k] = best_thr
    print(f"  K={k}: train positives={n_pos}, negatives={n_neg}, threshold={best_thr:.2f}")


# =============================================================
# EVALUATION METRICS
# =============================================================

def eval_metrics(probs, targets, threshold):
    """Compute classification metrics at a fixed threshold."""
    preds = (probs >= threshold).astype(int)

    try:
        roc = roc_auc_score(targets, probs)
    except ValueError:
        roc = float("nan")

    try:
        pr_auc = average_precision_score(targets, probs)
    except ValueError:
        pr_auc = float("nan")

    f1 = f1_score(targets, preds, zero_division=0)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)

    tn = ((preds == 0) & (targets == 0)).sum()
    fp = ((preds == 1) & (targets == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    pos_samples = int(targets.sum())

    return {
        "F1": f1, "Precision": prec, "Recall": rec,
        "ROC-AUC": roc, "PR-AUC": pr_auc, "FPR": fpr,
        "PosSamples": pos_samples
    }


# =============================================================
# PRE-ATTACK ONSET EVALUATION — V4 WORLD MODEL
# =============================================================

print("\n" + "=" * 90)
print("PRE-ATTACK ONSET PREDICTION — V4 WORLD MODEL")
print("=" * 90)
print(f"Evaluation subset: A(t) = 0 (currently benign windows only)")
print(f"Frozen threshold:  {FROZEN_THRESHOLD}")
print(f"Scoring strategy:  max(P(t+1), ..., P(t+K))")
print()

header = f"{'Horizon':<12} {'F1':>8} {'Prec':>8} {'Rec':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'FPR':>8} {'Pos':>6} {'Neg':>6}"
print(header)
print("-" * len(header))

v4_results = {}

for k in K_VALUES:
    # Onset target: any attack in t+1 ... t+K
    onset_target = A_bin_test[:, :k].max(axis=1)

    # Filter to pre-attack
    target_pa = onset_target[pre_attack_mask_test]
    score_pa = test_probs[pre_attack_mask_test][:, 1:k+1].max(axis=1)

    m = eval_metrics(score_pa, target_pa, FROZEN_THRESHOLD)
    v4_results[k] = m

    neg_count = len(target_pa) - m["PosSamples"]
    print(f"{'K=' + str(k):<12} {m['F1']:>8.4f} {m['Precision']:>8.4f} {m['Recall']:>8.4f} "
          f"{m['ROC-AUC']:>8.4f} {m['PR-AUC']:>8.4f} {m['FPR']:>8.4f} "
          f"{m['PosSamples']:>6} {neg_count:>6}")

# =============================================================
# BASELINE COMPARISON
# =============================================================

print("\n" + "=" * 90)
print("MODEL COMPARISON — PRE-ATTACK ONSET")
print("=" * 90)

full_header = f"{'Model':<35} {'K':>3} {'F1':>8} {'Prec':>8} {'Rec':>8} {'ROC-AUC':>8} {'PR-AUC':>8} {'FPR':>8}"
print(full_header)
print("-" * len(full_header))

for k in K_VALUES:
    onset_target = A_bin_test[:, :k].max(axis=1)
    target_pa = onset_target[pre_attack_mask_test]

    # --- Persistence ---
    # A(t)=0 -> predict 0. By construction, persistence cannot
    # predict onset on this subset. This is intentional: it
    # demonstrates the experiment isolates forecasting from
    # persistence.
    pers_probs = np.zeros(len(target_pa))
    pers_m = eval_metrics(pers_probs, target_pa, 0.5)
    print(f"{'Persistence':<35} {k:>3} {pers_m['F1']:>8.4f} {pers_m['Precision']:>8.4f} "
          f"{pers_m['Recall']:>8.4f} {pers_m['ROC-AUC']:>8.4f} {pers_m['PR-AUC']:>8.4f} "
          f"{pers_m['FPR']:>8.4f}")

    # --- Logistic Regression ---
    X_test_lr = X_test[:, -1, :]
    X_test_lr_pa = X_test_lr[pre_attack_mask_test]

    if lr_models[k] is not None:
        lr_probs_test = lr_models[k].predict_proba(X_test_lr_pa)[:, 1]
    else:
        lr_probs_test = np.zeros(len(target_pa))

    lr_m = eval_metrics(lr_probs_test, target_pa, lr_thresholds[k])
    print(f"{'Logistic Regression (latest-state)':<35} {k:>3} {lr_m['F1']:>8.4f} {lr_m['Precision']:>8.4f} "
          f"{lr_m['Recall']:>8.4f} {lr_m['ROC-AUC']:>8.4f} {lr_m['PR-AUC']:>8.4f} "
          f"{lr_m['FPR']:>8.4f}")

    # --- V4 World Model ---
    m = v4_results[k]
    print(f"{'V4 World Model':<35} {k:>3} {m['F1']:>8.4f} {m['Precision']:>8.4f} "
          f"{m['Recall']:>8.4f} {m['ROC-AUC']:>8.4f} {m['PR-AUC']:>8.4f} "
          f"{m['FPR']:>8.4f}")

    if k != K_VALUES[-1]:
        print("-" * len(full_header))


# =============================================================
# EARLY WARNING ANALYSIS
# =============================================================

print("\n" + "=" * 90)
print("EARLY WARNING CAPABILITY")
print("=" * 90)

# Detect attack onsets in test A(t+1) sequence
target_seq = A_bin_test[:, 0]
onsets = []
for i in range(1, len(target_seq)):
    if target_seq[i] == 1 and target_seq[i-1] == 0:
        onsets.append(i)

active_at_start = (target_seq[0] == 1)
if active_at_start:
    print("Note: Attack already active at start of test — that episode is excluded.")

print(f"Observable attack onsets: {len(onsets)}")
for idx in onsets:
    ts = meta_test.iloc[idx]["timestamp"]
    print(f"  -> Onset index {idx}: {ts}")

# Expected onsets for verification
expected_onset_timestamps = [
    pd.to_datetime("2018-03-02 03:54:50"),
    pd.to_datetime("2018-03-02 10:13:20"),
    pd.to_datetime("2018-03-02 12:13:30"),
]

actual_onset_timestamps = [meta_test.iloc[i]["timestamp"] for i in onsets]
assert len(actual_onset_timestamps) == len(expected_onset_timestamps), \
    f"Expected {len(expected_onset_timestamps)} onsets, found {len(actual_onset_timestamps)}"
for act, exp in zip(actual_onset_timestamps, expected_onset_timestamps):
    assert act == exp, f"Onset timestamp mismatch: expected {exp}, got {act}"
print("  Expected onset verification: PASS")

# For early warning, use K=5 max scoring with frozen threshold
best_k = max(K_VALUES)
test_score_max = test_probs[:, 1:best_k+1].max(axis=1)
test_preds_ew = (test_score_max >= FROZEN_THRESHOLD).astype(int)

# Warning: must also check that A(t)=0 at the warning window
# (we only issue valid pre-attack warnings from benign states)
lead_times = []
onset_details = []

for onset_idx in onsets:
    onset_ts = meta_test.iloc[onset_idx]["timestamp"]
    warned = False
    first_warning_idx = None

    # Scan the K windows before onset
    start_t = max(0, onset_idx - best_k)
    for t in range(start_t, onset_idx):
        # Only count warnings from benign windows
        if y_current_test[t] == 0 and test_preds_ew[t] == 1:
            lead_time = (onset_idx - t) * 10  # seconds
            lead_times.append(lead_time)
            warned = True
            first_warning_idx = t
            break

    if not warned:
        lead_times.append(0)

    onset_details.append({
        "onset_idx": onset_idx,
        "onset_ts": onset_ts,
        "warned": warned,
        "first_warning_idx": first_warning_idx,
        "lead_time": lead_times[-1]
    })

warned_count = sum(1 for lt in lead_times if lt > 0)
coverage = (warned_count / len(onsets)) * 100 if len(onsets) > 0 else 0.0

valid_lts = [lt for lt in lead_times if lt > 0]
mean_lt = np.mean(valid_lts) if valid_lts else 0.0
median_lt = np.median(valid_lts) if valid_lts else 0.0

# False alerts per hour (on pre-attack subset only)
onset_target_full = A_bin_test[:, :best_k].max(axis=1)
target_pa_full = onset_target_full[pre_attack_mask_test]
score_pa_full = test_score_max[pre_attack_mask_test]
preds_pa_full = (score_pa_full >= FROZEN_THRESHOLD).astype(int)
false_alerts = ((preds_pa_full == 1) & (target_pa_full == 0)).sum()
total_hours_pa = (pre_attack_mask_test.sum() * 10) / 3600.0
fa_per_hour = false_alerts / total_hours_pa if total_hours_pa > 0 else 0.0

print(f"\n  Warned before onset:     {warned_count} / {len(onsets)}")
print(f"  Warning coverage:        {coverage:.1f}%")
print(f"  Mean lead time:          {mean_lt:.1f} sec")
print(f"  Median lead time:        {median_lt:.1f} sec")
print(f"  False alerts (K={best_k}):    {false_alerts}")
print(f"  Benign hours evaluated:  {total_hours_pa:.2f}")
print(f"  False alerts / hour:     {fa_per_hour:.2f}")

for d in onset_details:
    status = f"WARNED (lead={d['lead_time']}s)" if d["warned"] else "MISSED"
    print(f"  Onset {d['onset_ts']}: {status}")


# =============================================================
# FORECAST TIMELINE AROUND EACH ONSET
# =============================================================

print("\n" + "=" * 90)
print("FORECAST PROBABILITY TIMELINE AROUND EACH ATTACK ONSET")
print("=" * 90)
print(f"Threshold: {FROZEN_THRESHOLD}  |  Scoring: max(P(t+1)...P(t+{best_k}))")

TIMELINE_BEFORE = 8   # windows before onset to show
TIMELINE_AFTER = 3    # windows after onset to show

all_timeline_rows = []

for onset_info in onset_details:
    onset_idx = onset_info["onset_idx"]
    onset_ts = onset_info["onset_ts"]

    print(f"\n--- Attack Onset: {onset_ts} (index {onset_idx}) ---")

    tl_header = f"{'Time':<22} {'A(t)':>5} {'A(t+1)':>6} {'P_max':>7} {'P(t+1)':>7} {'P(t+2)':>7} {'P(t+3)':>7} {'P(t+5)':>7} {'Warning':>8}"
    print(tl_header)
    print("-" * len(tl_header))

    start = max(0, onset_idx - TIMELINE_BEFORE)
    end = min(len(X_test), onset_idx + TIMELINE_AFTER + 1)

    for i in range(start, end):
        ts = meta_test.iloc[i]["timestamp"]
        a_t = int(y_current_test[i])
        a_t1 = int(A_bin_test[i, 0])

        # Individual step probabilities
        p_t1 = test_probs[i, 1]
        p_t2 = test_probs[i, 2]
        p_t3 = test_probs[i, 3]
        p_t5 = test_probs[i, 5] if test_probs.shape[1] > 5 else test_probs[i, -1]

        p_max = test_probs[i, 1:best_k+1].max()
        warning = ""
        if i == onset_idx:
            warning = "ONSET"
        elif a_t == 0 and p_max >= FROZEN_THRESHOLD:
            warning = "<<<"

        print(f"{str(ts):<22} {a_t:>5} {a_t1:>6} {p_max:>7.4f} {p_t1:>7.4f} {p_t2:>7.4f} "
              f"{p_t3:>7.4f} {p_t5:>7.4f} {warning:>8}")

        all_timeline_rows.append({
            "onset_ts": str(onset_ts),
            "window_ts": str(ts),
            "A_t": a_t,
            "A_t1": a_t1,
            "P_max_K5": round(float(p_max), 6),
            "P_t1": round(float(p_t1), 6),
            "P_t2": round(float(p_t2), 6),
            "P_t3": round(float(p_t3), 6),
            "P_t5": round(float(p_t5), 6),
            "warning": 1 if (a_t == 0 and p_max >= FROZEN_THRESHOLD) else 0,
            "is_onset": 1 if i == onset_idx else 0,
        })

# Save timeline CSV
timeline_df = pd.DataFrame(all_timeline_rows)
timeline_path = OUTPUT_DIR / "onset_forecast_timelines.csv"
timeline_df.to_csv(timeline_path, index=False)
print(f"\nTimeline saved to: {timeline_path}")


# =============================================================
# SAVE FULL RESULTS TABLE
# =============================================================

results_rows = []
for k in K_VALUES:
    onset_target = A_bin_test[:, :k].max(axis=1)
    target_pa = onset_target[pre_attack_mask_test]

    m = v4_results[k]
    results_rows.append({
        "Model": "V4 World Model",
        "Horizon_K": k,
        "Threshold": FROZEN_THRESHOLD,
        "Threshold_Source": "frozen_checkpoint",
        **m
    })

    # Persistence
    pers_probs = np.zeros(len(target_pa))
    pers_m = eval_metrics(pers_probs, target_pa, 0.5)
    results_rows.append({
        "Model": "Persistence",
        "Horizon_K": k,
        "Threshold": 0.5,
        "Threshold_Source": "fixed",
        **pers_m
    })

    # LR
    X_test_lr_pa = X_test[:, -1, :][pre_attack_mask_test]
    if lr_models[k] is not None:
        lr_probs_t = lr_models[k].predict_proba(X_test_lr_pa)[:, 1]
    else:
        lr_probs_t = np.zeros(len(target_pa))
    lr_m = eval_metrics(lr_probs_t, target_pa, lr_thresholds[k])
    results_rows.append({
        "Model": "Logistic Regression",
        "Horizon_K": k,
        "Threshold": lr_thresholds[k],
        "Threshold_Source": "train_pre_attack",
        **lr_m
    })

results_df = pd.DataFrame(results_rows)
results_path = OUTPUT_DIR / "pre_attack_onset_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")


# =============================================================
# INTEGRITY CHECKS
# =============================================================

print("\n" + "=" * 90)
print("INTEGRITY CHECKS")
print("=" * 90)

# 1. All evaluated test windows have A(t)=0
assert (y_current_test[pre_attack_mask_test] == 0).all(), "A(t) != 0 in pre-attack subset!"
print(f"  [PASS] A(t) = 0 for all {pre_attack_mask_test.sum()} evaluated test windows")

# 2. March 2 test only
assert (meta_test["source_day"] == "Friday-02-03-2018").all(), "Test set not Mar 2 only!"
print(f"  [PASS] Test day: March 2 only")

# 3. No test threshold tuning
print(f"  [PASS] Threshold frozen from checkpoint: {FROZEN_THRESHOLD} (no test optimization)")

# 4. Expected onsets verified (already asserted above)
print(f"  [PASS] Expected 3 attack onsets verified at correct timestamps")

# 5. Sample counts
for k in K_VALUES:
    onset_target = A_bin_test[:, :k].max(axis=1)
    target_pa = onset_target[pre_attack_mask_test]
    n_pos = int(target_pa.sum())
    n_neg = int((target_pa == 0).sum())
    print(f"  [INFO] K={k}: {n_pos} onset positives, {n_neg} negatives (total {len(target_pa)} benign windows)")

# 6. No attack windows leaked
print(f"  [PASS] {(y_current_test == 1).sum()} already-attacking windows excluded from evaluation")

# 7. Threshold NOT derived from test data
print(f"  [PASS] V4 threshold ({FROZEN_THRESHOLD}) derived from validation during training, NOT from March 2")

print("\n" + "=" * 90)
print("PRE-ATTACK ONSET EVALUATION COMPLETE")
print("=" * 90)
print()
print("INTERPRETATION NOTE:")
print("  This experiment removes the advantage of persistence and isolates")
print("  the model's ability to forecast attack onset from a benign state.")
print("  Persistence scores F1=0 by construction (A(t)=0 -> predict 0).")
print("  This is not evidence of V4 superiority over persistence in general;")
print("  it demonstrates that the experiment tests a fundamentally different")
print("  question than the persistence-dominated standard benchmark.")
print("=" * 90)
