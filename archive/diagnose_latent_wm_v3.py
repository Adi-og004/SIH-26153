"""
=============================================================
LATENT WORLD MODEL V3 — QUICK DIAGNOSTIC
=============================================================

Runs the same diagnosis as V2 but for the V3 model.
Produces the same 7 diagnostic outputs.
=============================================================
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================
# PATHS
# =============================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

MODEL_DIR = Path(
    "models/latent_world_model_v3"
)

OUTPUT_DIR = (
    MODEL_DIR / "diagnosis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================
# DEVICE
# =============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 90)
print("LATENT WORLD MODEL V3 — DIAGNOSTIC")
print("=" * 90)
print(f"Device: {DEVICE}")


# =============================================================
# MODEL DEFINITION (V3 with residual)
# =============================================================

class LatentWorldModelV3(nn.Module):

    def __init__(
        self, input_size, hidden_size,
        latent_size, output_size,
        num_layers, dropout
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1
                else 0.0
            )
        )
        self.encoder_norm = nn.LayerNorm(
            hidden_size
        )
        self.encoder_fc = nn.Linear(
            hidden_size, latent_size
        )

        # Residual transition
        self.transition_delta = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size)
        )

        self.decoder_fc1 = nn.Linear(
            latent_size, 64
        )
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(
            64, output_size
        )

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(last_hidden)
        return self.encoder_fc(last_hidden)

    def transition_step(self, z):
        return z + self.transition_delta(z)

    def decode(self, z):
        h = self.decoder_fc1(z)
        h = self.decoder_activation(h)
        return self.decoder_fc2(h)

    def predict_attack(self, z):
        logits = self.attack_head(z)
        return logits.squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        z_next = self.transition_step(z)
        state = self.decode(z_next)
        attack = self.predict_attack(z_next)
        return state, z_next, attack, z


# =============================================================
# LOAD MODEL
# =============================================================

print("\n--- Loading V3 model ---")

checkpoint = torch.load(
    MODEL_DIR / "latent_world_model_v3.pt",
    map_location=DEVICE,
    weights_only=False
)

model = LatentWorldModelV3(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    latent_size=checkpoint["latent_size"],
    output_size=checkpoint["output_size"],
    num_layers=checkpoint["num_layers"],
    dropout=0.20
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model = model.to(DEVICE)
model.eval()

saved_threshold = checkpoint["selected_threshold"]
print(f"Saved threshold: {saved_threshold:.2f}")


# =============================================================
# LOAD DATA
# =============================================================

print("\n--- Loading data ---")

X_val = np.load(DATA_DIR / "X_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")

val_meta = pd.read_csv(DATA_DIR / "val_metadata.csv")
test_meta = pd.read_csv(DATA_DIR / "test_metadata.csv")

y_val_attack = val_meta["attack_present"].to_numpy(dtype=np.float32)
y_test_attack = test_meta["attack_present"].to_numpy(dtype=np.float32)

print(f"Val:  {X_val.shape[0]} samples, {int(y_val_attack.sum())} attack")
print(f"Test: {X_test.shape[0]} samples, {int(y_test_attack.sum())} attack")


# =============================================================
# GENERATE PROBABILITIES
# =============================================================

BATCH_SIZE = 64

def get_probabilities(X_data):
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[start:start+BATCH_SIZE].to(DEVICE)
            _, _, attack_logits, _ = model(batch)
            probs = torch.sigmoid(attack_logits)
            all_probs.extend(probs.cpu().numpy())
    return np.asarray(all_probs)

val_probs = get_probabilities(X_val)
test_probs = get_probabilities(X_test)


# =============================================================
# DIAGNOSTIC 1: ROC-AUC and PR-AUC
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 1: ROC-AUC and PR-AUC")
print("=" * 90)

val_roc_auc = roc_auc_score(y_val_attack, val_probs)
val_p_curve, val_r_curve, _ = precision_recall_curve(y_val_attack, val_probs)
val_pr_auc = auc(val_r_curve, val_p_curve)

test_roc_auc = roc_auc_score(y_test_attack, test_probs)
test_p_curve, test_r_curve, _ = precision_recall_curve(y_test_attack, test_probs)
test_pr_auc = auc(test_r_curve, test_p_curve)

print(f"\nValidation  ROC-AUC: {val_roc_auc:.4f}")
print(f"Validation  PR-AUC:  {val_pr_auc:.4f}")
print(f"Test        ROC-AUC: {test_roc_auc:.4f}")
print(f"Test        PR-AUC:  {test_pr_auc:.4f}")

# Save ROC/PR plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

val_fpr_c, val_tpr_c, _ = roc_curve(y_val_attack, val_probs)
test_fpr_c, test_tpr_c, _ = roc_curve(y_test_attack, test_probs)

axes[0].plot(val_fpr_c, val_tpr_c, label=f"Val (AUC={val_roc_auc:.3f})", linewidth=2)
axes[0].plot(test_fpr_c, test_tpr_c, label=f"Test (AUC={test_roc_auc:.3f})", linewidth=2)
axes[0].plot([0,1],[0,1],"k--",alpha=0.3,label="Random")
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve — V3"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(val_r_curve, val_p_curve, label=f"Val (AUC={val_pr_auc:.3f})", linewidth=2)
axes[1].plot(test_r_curve, test_p_curve, label=f"Test (AUC={test_pr_auc:.3f})", linewidth=2)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("PR Curve — V3"); axes[1].legend(); axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_roc_pr_curves.png", dpi=150)
plt.close(fig)


# =============================================================
# DIAGNOSTIC 2: PROBABILITY DISTRIBUTIONS
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 2: Probability Distributions")
print("=" * 90)

val_benign_probs = val_probs[y_val_attack == 0]
val_attack_probs = val_probs[y_val_attack == 1]
test_benign_probs = test_probs[y_test_attack == 0]
test_attack_probs = test_probs[y_test_attack == 1]

print(f"\nVal attack probs:  mean={val_attack_probs.mean():.4f}")
print(f"Test attack probs: mean={test_attack_probs.mean():.4f}")
print(f"Ratio: {val_attack_probs.mean() / max(test_attack_probs.mean(), 1e-8):.1f}x")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(val_benign_probs, bins=50, alpha=0.7, label="Benign", color="steelblue", density=True)
axes[0].hist(val_attack_probs, bins=50, alpha=0.7, label="Attack", color="crimson", density=True)
axes[0].axvline(x=saved_threshold, color="black", linestyle="--", label=f"Thr={saved_threshold:.2f}")
axes[0].set_title("V3 VALIDATION"); axes[0].legend()

axes[1].hist(test_benign_probs, bins=50, alpha=0.7, label="Benign", color="steelblue", density=True)
axes[1].hist(test_attack_probs, bins=50, alpha=0.7, label="Attack", color="crimson", density=True)
axes[1].axvline(x=saved_threshold, color="black", linestyle="--", label=f"Thr={saved_threshold:.2f}")
axes[1].set_title("V3 TEST"); axes[1].legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_probability_distributions.png", dpi=150)
plt.close(fig)


# =============================================================
# DIAGNOSTIC 3: ORACLE THRESHOLD SWEEP
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 3: Oracle Threshold Sweep (Test)")
print("=" * 90)

thresh_results = []
for t in np.arange(0.01, 0.99, 0.01):
    preds = (test_probs >= t).astype(int)
    if preds.sum() == 0:
        continue
    p = precision_score(y_test_attack, preds, zero_division=0)
    r = recall_score(y_test_attack, preds, zero_division=0)
    f = f1_score(y_test_attack, preds, zero_division=0)
    cm = confusion_matrix(y_test_attack, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
    thresh_results.append({
        "threshold": round(t, 2), "precision": p,
        "recall": r, "f1": f, "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    })

thresh_df = pd.DataFrame(thresh_results)
best_row = thresh_df.loc[thresh_df["f1"].idxmax()]

print(f"\nOracle best threshold: {best_row['threshold']:.2f}")
print(f"Oracle best F1:        {best_row['f1']:.4f}")
print(f"  Precision: {best_row['precision']:.4f}")
print(f"  Recall:    {best_row['recall']:.4f}")
print(f"  FPR:       {best_row['fpr']:.4f}")

thresh_df.to_csv(OUTPUT_DIR / "03_test_threshold_sweep.csv", index=False)


# =============================================================
# DIAGNOSTIC 4: TIME-SERIES
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 4: Time-Series Plot")
print("=" * 90)

test_timestamps = pd.to_datetime(test_meta["timestamp"])

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

axes[0].fill_between(range(len(y_test_attack)), y_test_attack, alpha=0.4, color="crimson", label="Attack")
axes[0].set_ylabel("Ground Truth"); axes[0].legend()
axes[0].set_title("V3: Ground Truth vs Model Predictions Over Time")

axes[1].plot(range(len(test_probs)), test_probs, color="steelblue", linewidth=0.8, label="Probability")
axes[1].axhline(y=saved_threshold, color="red", linestyle="--", label=f"Thr={saved_threshold:.2f}")
for i in range(len(y_test_attack)):
    if y_test_attack[i] == 1:
        axes[1].axvspan(i-0.5, i+0.5, alpha=0.1, color="crimson")
axes[1].set_ylabel("Predicted Prob"); axes[1].legend()

axes[2].plot(range(len(test_meta)), test_meta["attack_ratio"], color="darkorange", linewidth=0.8, label="Attack ratio")
axes[2].set_ylabel("Attack Ratio")
axes[2].set_xlabel(f"Window index ({test_timestamps.iloc[0].strftime('%H:%M')} to {test_timestamps.iloc[-1].strftime('%H:%M')})")
axes[2].legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_timeseries_predictions.png", dpi=150)
plt.close(fig)


# =============================================================
# DIAGNOSTIC 5: CLASSIFICATION ANALYSIS
# =============================================================

test_preds = (test_probs >= saved_threshold).astype(int)

analysis_df = test_meta.copy()
analysis_df["probability"] = test_probs
analysis_df["predicted"] = test_preds

conditions = []
for _, row in analysis_df.iterrows():
    a = int(row["attack_present"])
    p = int(row["predicted"])
    if a == 1 and p == 1: conditions.append("TP")
    elif a == 1 and p == 0: conditions.append("FN")
    elif a == 0 and p == 1: conditions.append("FP")
    else: conditions.append("TN")

analysis_df["classification"] = conditions

fn_df = analysis_df[analysis_df["classification"] == "FN"]
tp_df = analysis_df[analysis_df["classification"] == "TP"]

print(f"\nFalse Negatives: {len(fn_df)}")
print(f"True Positives:  {len(tp_df)}")
print(f"FN prob mean: {fn_df['probability'].mean():.6f}")
print(f"FN prob max:  {fn_df['probability'].max():.6f}")

if len(tp_df) > 0:
    print(f"TP window range: {tp_df.index[0]} to {tp_df.index[-1]}")
    print(f"TP timestamps: {tp_df['timestamp'].iloc[0]} to {tp_df['timestamp'].iloc[-1]}")

analysis_df.to_csv(OUTPUT_DIR / "05_classification_analysis.csv", index=False)


# =============================================================
# SUMMARY: V3 vs V2 COMPARISON
# =============================================================

print("\n" + "=" * 90)
print("V3 vs V2 DIAGNOSTIC COMPARISON")
print("=" * 90)

print(f"\n{'Metric':<30} {'V2':>10} {'V3':>10} {'Change':>10}")
print("-" * 62)
print(f"{'Test ROC-AUC':<30} {'0.7229':>10} {test_roc_auc:>10.4f} {test_roc_auc - 0.7229:>+10.4f}")
print(f"{'Test PR-AUC':<30} {'0.6096':>10} {test_pr_auc:>10.4f} {test_pr_auc - 0.6096:>+10.4f}")
print(f"{'Test F1':<30} {'0.4118':>10} {'0.5116':>10} {'+0.0998':>10}")
print(f"{'Test Recall':<30} {'0.2849':>10} {'0.3837':>10} {'+0.0988':>10}")
print(f"{'Val attack prob mean':<30} {'0.7598':>10} {val_attack_probs.mean():>10.4f}")
print(f"{'Test attack prob mean':<30} {'0.2838':>10} {test_attack_probs.mean():>10.4f}")
print(f"{'Prob ratio (val/test)':<30} {'2.7x':>10} {val_attack_probs.mean()/max(test_attack_probs.mean(),1e-8):>9.1f}x")
print(f"{'Oracle F1':<30} {'0.5593':>10} {best_row['f1']:>10.4f}")
print(f"{'FN count':<30} {'123':>10} {len(fn_df):>10}")
print(f"{'TP count':<30} {'49':>10} {len(tp_df):>10}")

print("\n" + "=" * 90)
print("V3 DIAGNOSIS COMPLETE")
print(f"Outputs saved to: {OUTPUT_DIR}")
print("=" * 90)
