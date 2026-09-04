"""
=============================================================
BASELINE EVALUATION — MULTI-DAY BENCHMARK
=============================================================

This script evaluates two baselines on the multi-day benchmark:
1. Persistence Baseline: Predicts that the label at t+k will be identical to the observed label at time t.
2. Logistic Regression (latest-state) Baseline: Direct forecasting models trained on the 52-feature input at time t to predict t+k.

Constraints followed:
- Uses the exact same multiday_benchmark_K5 dataset.
- Chronological train/val/test splits (Mar 2 test set untouched until end).
- Models fit only on Training data. Uses pre-scaled features.
- Validation data used for optimal threshold selection.
- Threshold frozen before Test evaluation.
- No future labels used in Persistence.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================
# CONFIGURATION
# =============================================================

MULTISTEP_DIR = Path("data/processed/multiday_benchmark_K5")
STATES_DIR = Path("data/processed/states_v2")

K_VALUES = [1, 2, 3, 5]

print("=" * 90)
print("BASELINE EVALUATION — MULTI-DAY BENCHMARK")
print("=" * 90)

# =============================================================
# DATA LOADING
# =============================================================
print("\nLoading datasets from:", MULTISTEP_DIR)
X_train = np.load(MULTISTEP_DIR / "X_train.npy")
A_bin_train = np.load(MULTISTEP_DIR / "A_binary_multistep_train.npy")
meta_train = pd.read_csv(MULTISTEP_DIR / "train_metadata.csv")

X_val = np.load(MULTISTEP_DIR / "X_val.npy")
A_bin_val = np.load(MULTISTEP_DIR / "A_binary_multistep_val.npy")
meta_val = pd.read_csv(MULTISTEP_DIR / "val_metadata.csv")

X_test = np.load(MULTISTEP_DIR / "X_test.npy")
A_bin_test = np.load(MULTISTEP_DIR / "A_binary_multistep_test.npy")
meta_test = pd.read_csv(MULTISTEP_DIR / "test_metadata.csv")

meta_train["timestamp"] = pd.to_datetime(meta_train["timestamp"])
meta_val["timestamp"] = pd.to_datetime(meta_val["timestamp"])
meta_test["timestamp"] = pd.to_datetime(meta_test["timestamp"])

print(f"Train sequences: {len(X_train)}")
print(f"Val sequences:   {len(X_val)}")
print(f"Test sequences:  {len(X_test)}")

# =============================================================
# PERSISTENCE BASELINE DATA PREP (EXTRACTING LABEL AT time t)
# =============================================================
# The benchmark provides the label for t+1 in A_bin[:, 0].
# To implement a strict persistence rule without leaking future labels,
# we need the true label at time t. We fetch this by looking up the 
# state 10 seconds prior to t+1 in the original CSVs.

def fetch_time_t_labels(meta_df):
    """
    Given a metadata dataframe containing timestamps for t+1, 
    fetches the attack_present label for time t (t+1 - 10s).
    """
    source_days = meta_df["source_day"].unique()
    
    # Pre-load all required CSVs into a fast lookup dict
    lookup = {}
    print("  Loading source CSVs to retrieve ground truth for time t...")
    for day in source_days:
        csv_path = STATES_DIR / f"{day}_TrafficForML_CICFlowMeter_states_v2.csv"
        df = pd.read_csv(csv_path, usecols=["window_start", "attack_present"])
        df["window_start"] = pd.to_datetime(df["window_start"])
        # Create a dictionary mapping timestamp -> attack_present
        day_dict = dict(zip(df["window_start"], df["attack_present"]))
        lookup.update(day_dict)
        
    labels_t = []
    missing_count = 0
    for ts in meta_df["timestamp"]:
        ts_t = ts - pd.Timedelta(seconds=10)
        if ts_t in lookup:
            labels_t.append(lookup[ts_t])
        else:
            raise ValueError(
                f"CRITICAL ERROR: Missing historical label for time t. "
                f"Could not find timestamp {ts_t} (10s prior to {ts}) in original CSV."
            )
            
    assert missing_count == 0, "Lookup failures are not permitted."
    return np.array(labels_t)

print("\nExtracting labels for time t (Persistence Baseline)...")
y_t_val = fetch_time_t_labels(meta_val)
y_t_test = fetch_time_t_labels(meta_test)


# =============================================================
# LOGISTIC REGRESSION DATA PREP
# =============================================================
# Use the most recent observed step's 52 features: X[:, -1, :]
# Benchmark is already scaled, so no StandardScaler is fitted here.
X_train_lr = X_train[:, -1, :]
X_val_lr   = X_val[:, -1, :]
X_test_lr  = X_test[:, -1, :]

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

def eval_metrics(preds, probs, targets):
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

results = []

# =============================================================
# EVALUATION LOOP
# =============================================================
print("\nEvaluating Baselines...")

for k in K_VALUES:
    # Ground truth for t+k
    target_train = A_bin_train[:, k-1]
    target_val   = A_bin_val[:, k-1]
    target_test  = A_bin_test[:, k-1]
    
    # -------------------------------------------------------------
    # 1. PERSISTENCE BASELINE
    # Rule: y_hat(t+k) = y(t). 
    # Probabilities are discrete (0.0 or 1.0). Threshold is effectively 0.5.
    # -------------------------------------------------------------
    preds_pers_test = y_t_test
    probs_pers_test = y_t_test.astype(float)
    
    f1, prec, rec, roc, fpr, pos = eval_metrics(preds_pers_test, probs_pers_test, target_test)
    results.append({
        "Model": "Persistence",
        "Horizon": f"t+{k}",
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC-AUC": roc,
        "FPR": fpr,
        "PosSamples": pos,
        "Threshold": "-"
    })
    
    # -------------------------------------------------------------
    # 2. LOGISTIC REGRESSION (latest-state) BASELINE
    # -------------------------------------------------------------
    # Train a direct forecasting model for horizon k
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_lr, target_train)
    
    # Tune threshold on Validation
    probs_val_lr = lr.predict_proba(X_val_lr)[:, 1]
    best_thr = find_best_threshold(probs_val_lr, target_val)
    
    # Freeze threshold and evaluate on Test
    probs_test_lr = lr.predict_proba(X_test_lr)[:, 1]
    preds_test_lr = (probs_test_lr >= best_thr).astype(int)
    
    f1, prec, rec, roc, fpr, pos = eval_metrics(preds_test_lr, probs_test_lr, target_test)
    results.append({
        "Model": "Logistic Regression (latest-state)",
        "Horizon": f"t+{k}",
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC-AUC": roc,
        "FPR": fpr,
        "PosSamples": pos,
        "Threshold": f"{best_thr:.2f}"
    })


# =============================================================
# PRINT COMPARISON TABLE
# =============================================================
print("\n" + "="*115)
print("FINAL COMPARISON ON HELD-OUT TEST SET (MAR 2)")
print("="*115)
print(f"{'Model':<40} | {'Horizon':<7} | {'F1':>6} | {'Precision':>9} | {'Recall':>6} | {'ROC-AUC':>7} | {'FPR':>6} | {'Threshold':>9} | {'PosSamples':>10}")
print("-" * 115)

for r in results:
    f1_str = f"{r['F1']:.4f}"
    pr_str = f"{r['Precision']:.4f}"
    rc_str = f"{r['Recall']:.4f}"
    ra_str = f"{r['ROC-AUC']:.4f}"
    fp_str = f"{r['FPR']:.4f}"
    print(f"{r['Model']:<40} | {r['Horizon']:<7} | {f1_str:>6} | {pr_str:>9} | {rc_str:>6} | {ra_str:>7} | {fp_str:>6} | {r['Threshold']:>9} | {r['PosSamples']:>10}")
print("=" * 115)

# =============================================================
# TRAJECTORY-EVENT INTERPRETATION (MAX SCORING OVER WINDOW)
# =============================================================
print("\n" + "="*115)
print("TRAJECTORY-EVENT PREDICTION (MAX SCORING over t+1 to t+k)")
print("Target: Any attack occurs within t+1 ... t+k")
print("="*115)
print(f"{'Model':<40} | {'Horizon':<7} | {'F1':>6} | {'Precision':>9} | {'Recall':>6} | {'ROC-AUC':>7} | {'FPR':>6} | {'Threshold':>9} | {'PosSamples':>10}")
print("-" * 115)

traj_results = []
for k in K_VALUES:
    if k == 1:
        continue
        
    # Ground truth: is there any attack in t+1 ... t+k?
    target_val_max = A_bin_val[:, 0:k].max(axis=1)
    target_test_max = A_bin_test[:, 0:k].max(axis=1)
    
    # -------------------------------------------------------------
    # 1. PERSISTENCE BASELINE (Trajectory)
    # -------------------------------------------------------------
    # The rule is: if an attack was present at t, we predict an attack 
    # will be present at ANY time between t+1 and t+k. This is exactly y(t).
    preds_pers_test = y_t_test
    probs_pers_test = y_t_test.astype(float)
    
    f1, prec, rec, roc, fpr, pos = eval_metrics(preds_pers_test, probs_pers_test, target_test_max)
    traj_results.append({
        "Model": "Persistence",
        "Horizon": f"1 to {k}",
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC-AUC": roc,
        "FPR": fpr,
        "PosSamples": pos,
        "Threshold": "-"
    })
    
    # -------------------------------------------------------------
    # 2. LOGISTIC REGRESSION (latest-state) BASELINE (Trajectory)
    # -------------------------------------------------------------
    # We train a model to predict the multi-step max target directly.
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_lr, A_bin_train[:, 0:k].max(axis=1))
    
    probs_val_lr = lr.predict_proba(X_val_lr)[:, 1]
    best_thr = find_best_threshold(probs_val_lr, target_val_max)
    
    probs_test_lr = lr.predict_proba(X_test_lr)[:, 1]
    preds_test_lr = (probs_test_lr >= best_thr).astype(int)
    
    f1, prec, rec, roc, fpr, pos = eval_metrics(preds_test_lr, probs_test_lr, target_test_max)
    traj_results.append({
        "Model": "Logistic Regression (latest-state)",
        "Horizon": f"1 to {k}",
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC-AUC": roc,
        "FPR": fpr,
        "PosSamples": pos,
        "Threshold": f"{best_thr:.2f}"
    })

for r in traj_results:
    f1_str = f"{r['F1']:.4f}"
    pr_str = f"{r['Precision']:.4f}"
    rc_str = f"{r['Recall']:.4f}"
    ra_str = f"{r['ROC-AUC']:.4f}"
    fp_str = f"{r['FPR']:.4f}"
    print(f"{r['Model']:<40} | {r['Horizon']:<7} | {f1_str:>6} | {pr_str:>9} | {rc_str:>6} | {ra_str:>7} | {fp_str:>6} | {r['Threshold']:>9} | {r['PosSamples']:>10}")
print("=" * 115)


# =============================================================
# INTEGRITY CHECKS
# =============================================================
print("\n" + "="*90)
print("INTEGRITY CHECKS")
print("="*90)
print(f"  - Persistence t-label lookup failures: 0")
assert len(meta_test["source_day"].unique()) == 1 and meta_test["source_day"].unique()[0] == "Friday-02-03-2018", "Test day contains multiple days or wrong day!"
print(f"  - Test day: Mar 2 only")
print(f"  - Test labels used for threshold tuning: NO")
print("="*90)
