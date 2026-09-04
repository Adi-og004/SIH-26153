"""
=============================================================
LATENT WORLD MODEL V2 — DIAGNOSTIC ANALYSIS
=============================================================

PURPOSE:
    Understand WHY the model has strong validation performance
    (F1=0.83) but weak test performance (F1=0.41) on the
    28-Feb infiltration benchmark.

    This script does NOT retrain or modify the model.
    It loads the saved model and predictions, and produces
    evidence-based diagnostics.

WHAT IT COMPUTES:
    1. ROC-AUC and PR-AUC  (val + test)
    2. Probability distribution histograms
    3. Oracle threshold sweep on test data
    4. Time-series plot of predictions vs ground truth
    5. Feature distribution shift  (val vs test)
    6. False negative temporal analysis
    7. Summary report printed to console

OUTPUT:
    models/latent_world_model_v2/diagnosis/
=============================================================
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

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
    "models/latent_world_model_v2"
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
print("LATENT WORLD MODEL V2 — DIAGNOSTIC ANALYSIS")
print("=" * 90)
print(f"Device: {DEVICE}")


# =============================================================
# MODEL DEFINITION  (must match training script)
# =============================================================

class LatentWorldModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        latent_size,
        output_size,
        num_layers,
        dropout
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.encoder_norm = nn.LayerNorm(
            hidden_size
        )

        self.encoder_fc = nn.Linear(
            hidden_size,
            latent_size
        )

        self.transition = nn.Sequential(
            nn.Linear(
                latent_size,
                latent_size * 2
            ),
            nn.ReLU(),
            nn.Linear(
                latent_size * 2,
                latent_size
            )
        )

        self.decoder_fc1 = nn.Linear(
            latent_size,
            64
        )

        self.decoder_activation = nn.ReLU()

        self.decoder_fc2 = nn.Linear(
            64,
            output_size
        )

        self.attack_head = nn.Sequential(
            nn.Linear(
                latent_size,
                32
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )


    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(
            last_hidden
        )
        latent_state = self.encoder_fc(
            last_hidden
        )
        return latent_state


    def transition_step(self, latent_state):
        return self.transition(
            latent_state
        )


    def decode(self, latent_state):
        hidden = self.decoder_fc1(
            latent_state
        )
        hidden = self.decoder_activation(
            hidden
        )
        return self.decoder_fc2(hidden)


    def predict_attack(self, latent_state):
        logits = self.attack_head(
            latent_state
        )
        return logits.squeeze(-1)


    def forward(self, x):
        latent_state = self.encode(x)

        predicted_latent = (
            self.transition_step(
                latent_state
            )
        )

        predicted_state = self.decode(
            predicted_latent
        )

        attack_logits = self.predict_attack(
            predicted_latent
        )

        return (
            predicted_state,
            predicted_latent,
            attack_logits,
            latent_state
        )


# =============================================================
# LOAD MODEL
# =============================================================

print("\n--- Loading model checkpoint ---")

checkpoint = torch.load(
    MODEL_DIR / "latent_world_model_v2.pt",
    map_location=DEVICE,
    weights_only=False
)

model = LatentWorldModel(
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

saved_threshold = checkpoint[
    "selected_threshold"
]

print(f"Model loaded. Saved threshold: {saved_threshold:.2f}")


# =============================================================
# LOAD DATA
# =============================================================

print("\n--- Loading data ---")

X_val = np.load(DATA_DIR / "X_val.npy")
Y_val = np.load(DATA_DIR / "Y_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
Y_test = np.load(DATA_DIR / "Y_test.npy")

val_meta = pd.read_csv(
    DATA_DIR / "val_metadata.csv"
)
test_meta = pd.read_csv(
    DATA_DIR / "test_metadata.csv"
)

y_val_attack = (
    val_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)
y_test_attack = (
    test_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)

print(f"Val:  {X_val.shape[0]} samples, "
      f"{int(y_val_attack.sum())} attack")
print(f"Test: {X_test.shape[0]} samples, "
      f"{int(y_test_attack.sum())} attack")


# =============================================================
# GENERATE FRESH PROBABILITIES
# =============================================================

print("\n--- Generating predictions ---")

BATCH_SIZE = 64


def get_probabilities(X_data):
    """Run model on data and return attack probabilities."""

    X_tensor = torch.tensor(
        X_data, dtype=torch.float32
    )

    all_probs = []

    with torch.no_grad():
        for start in range(
            0, len(X_tensor), BATCH_SIZE
        ):
            batch = X_tensor[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            (
                _pred_state,
                _pred_latent,
                attack_logits,
                _latent_state
            ) = model(batch)

            probs = torch.sigmoid(
                attack_logits
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    return np.asarray(all_probs)


val_probs = get_probabilities(X_val)
test_probs = get_probabilities(X_test)


# =============================================================
# DIAGNOSTIC 1: ROC-AUC and PR-AUC
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 1: ROC-AUC and PR-AUC")
print("=" * 90)

# --- Validation ---
val_roc_auc = roc_auc_score(
    y_val_attack, val_probs
)

val_precision_curve, val_recall_curve, _ = (
    precision_recall_curve(
        y_val_attack, val_probs
    )
)

val_pr_auc = auc(
    val_recall_curve,
    val_precision_curve
)

# --- Test ---
test_roc_auc = roc_auc_score(
    y_test_attack, test_probs
)

test_precision_curve, test_recall_curve, _ = (
    precision_recall_curve(
        y_test_attack, test_probs
    )
)

test_pr_auc = auc(
    test_recall_curve,
    test_precision_curve
)

print(f"\nValidation  ROC-AUC: {val_roc_auc:.4f}")
print(f"Validation  PR-AUC:  {val_pr_auc:.4f}")
print(f"Test        ROC-AUC: {test_roc_auc:.4f}")
print(f"Test        PR-AUC:  {test_pr_auc:.4f}")

# Interpretation
print("\n--- Interpretation ---")

if test_roc_auc > 0.70:
    print("The model provides USEFUL ranking "
          "of attack probability on test data.")
    print("The problem may be threshold transfer, "
          "not model quality.")
elif test_roc_auc > 0.55:
    print("The model provides WEAK ranking. "
          "Some signal exists but it is noisy.")
else:
    print("The model ranking is NEAR RANDOM on "
          "test data. The learned representation "
          "does not generalize.")


# --- Save ROC curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
val_fpr_curve, val_tpr_curve, _ = roc_curve(
    y_val_attack, val_probs
)
test_fpr_curve, test_tpr_curve, _ = roc_curve(
    y_test_attack, test_probs
)

axes[0].plot(
    val_fpr_curve, val_tpr_curve,
    label=f"Val (AUC={val_roc_auc:.3f})",
    linewidth=2
)
axes[0].plot(
    test_fpr_curve, test_tpr_curve,
    label=f"Test (AUC={test_roc_auc:.3f})",
    linewidth=2
)
axes[0].plot(
    [0, 1], [0, 1], "k--",
    alpha=0.3, label="Random"
)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(alpha=0.3)

# PR
axes[1].plot(
    val_recall_curve, val_precision_curve,
    label=f"Val (AUC={val_pr_auc:.3f})",
    linewidth=2
)
axes[1].plot(
    test_recall_curve, test_precision_curve,
    label=f"Test (AUC={test_pr_auc:.3f})",
    linewidth=2
)

# Baseline = positive class ratio
val_pos_ratio = y_val_attack.mean()
test_pos_ratio = y_test_attack.mean()

axes[1].axhline(
    y=val_pos_ratio,
    color="blue", linestyle="--",
    alpha=0.3, label=f"Val baseline={val_pos_ratio:.2f}"
)
axes[1].axhline(
    y=test_pos_ratio,
    color="orange", linestyle="--",
    alpha=0.3, label=f"Test baseline={test_pos_ratio:.2f}"
)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "01_roc_pr_curves.png",
    dpi=150
)
plt.close(fig)

print("\nSaved: 01_roc_pr_curves.png")


# =============================================================
# DIAGNOSTIC 2: PROBABILITY DISTRIBUTIONS
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 2: Probability Distribution Comparison")
print("=" * 90)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Val — benign vs attack
val_benign_probs = val_probs[
    y_val_attack == 0
]
val_attack_probs = val_probs[
    y_val_attack == 1
]

axes[0, 0].hist(
    val_benign_probs, bins=50,
    alpha=0.7, label="Benign",
    color="steelblue", density=True
)
axes[0, 0].hist(
    val_attack_probs, bins=50,
    alpha=0.7, label="Attack",
    color="crimson", density=True
)
axes[0, 0].axvline(
    x=saved_threshold, color="black",
    linestyle="--",
    label=f"Threshold={saved_threshold:.2f}"
)
axes[0, 0].set_title(
    "VALIDATION — Attack Probabilities"
)
axes[0, 0].set_xlabel("Predicted Probability")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()

# Test — benign vs attack
test_benign_probs = test_probs[
    y_test_attack == 0
]
test_attack_probs = test_probs[
    y_test_attack == 1
]

axes[0, 1].hist(
    test_benign_probs, bins=50,
    alpha=0.7, label="Benign",
    color="steelblue", density=True
)
axes[0, 1].hist(
    test_attack_probs, bins=50,
    alpha=0.7, label="Attack",
    color="crimson", density=True
)
axes[0, 1].axvline(
    x=saved_threshold, color="black",
    linestyle="--",
    label=f"Threshold={saved_threshold:.2f}"
)
axes[0, 1].set_title(
    "TEST — Attack Probabilities"
)
axes[0, 1].set_xlabel("Predicted Probability")
axes[0, 1].legend()

# Overall comparison
axes[1, 0].hist(
    val_probs, bins=50, alpha=0.7,
    label="Validation", color="steelblue",
    density=True
)
axes[1, 0].hist(
    test_probs, bins=50, alpha=0.7,
    label="Test", color="darkorange",
    density=True
)
axes[1, 0].set_title(
    "All Probabilities — Val vs Test"
)
axes[1, 0].set_xlabel("Predicted Probability")
axes[1, 0].set_ylabel("Density")
axes[1, 0].legend()

# Summary statistics
summary_text = (
    f"=== Validation ===\n"
    f"Benign probs: "
    f"mean={val_benign_probs.mean():.4f}, "
    f"max={val_benign_probs.max():.4f}\n"
    f"Attack probs: "
    f"mean={val_attack_probs.mean():.4f}, "
    f"min={val_attack_probs.min():.4f}, "
    f"max={val_attack_probs.max():.4f}\n"
    f"\n"
    f"=== Test ===\n"
    f"Benign probs: "
    f"mean={test_benign_probs.mean():.4f}, "
    f"max={test_benign_probs.max():.4f}\n"
    f"Attack probs: "
    f"mean={test_attack_probs.mean():.4f}, "
    f"min={test_attack_probs.min():.4f}, "
    f"max={test_attack_probs.max():.4f}\n"
    f"\n"
    f"=== Threshold={saved_threshold:.2f} ===\n"
    f"Val: {(val_probs >= saved_threshold).sum()}"
    f" predicted positive\n"
    f"Test: {(test_probs >= saved_threshold).sum()}"
    f" predicted positive"
)

axes[1, 1].text(
    0.05, 0.95,
    summary_text,
    transform=axes[1, 1].transAxes,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round",
        facecolor="lightyellow",
        alpha=0.8
    )
)
axes[1, 1].axis("off")
axes[1, 1].set_title("Summary Statistics")

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "02_probability_distributions.png",
    dpi=150
)
plt.close(fig)

# Print summary
print(f"\nValidation benign probs:  "
      f"mean={val_benign_probs.mean():.4f}, "
      f"max={val_benign_probs.max():.4f}")
print(f"Validation attack probs:  "
      f"mean={val_attack_probs.mean():.4f}, "
      f"min={val_attack_probs.min():.4f}, "
      f"max={val_attack_probs.max():.4f}")
print(f"\nTest benign probs:        "
      f"mean={test_benign_probs.mean():.4f}, "
      f"max={test_benign_probs.max():.4f}")
print(f"Test attack probs:        "
      f"mean={test_attack_probs.mean():.4f}, "
      f"min={test_attack_probs.min():.4f}, "
      f"max={test_attack_probs.max():.4f}")

print("\nSaved: 02_probability_distributions.png")


# =============================================================
# DIAGNOSTIC 3: ORACLE THRESHOLD SWEEP ON TEST
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 3: Oracle Threshold Sweep on Test Data")
print("=" * 90)

print("\n(What is the BEST POSSIBLE F1 on test data?)")

threshold_results = []

for threshold in np.arange(0.01, 0.99, 0.01):

    preds = (
        test_probs >= threshold
    ).astype(int)

    if preds.sum() == 0:
        continue

    p = precision_score(
        y_test_attack, preds,
        zero_division=0
    )
    r = recall_score(
        y_test_attack, preds,
        zero_division=0
    )
    f = f1_score(
        y_test_attack, preds,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test_attack, preds,
        labels=[0, 1]
    )
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    threshold_results.append({
        "threshold": round(threshold, 2),
        "precision": p,
        "recall": r,
        "f1": f,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    })

threshold_df = pd.DataFrame(threshold_results)

# Find oracle best F1
best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

print(f"\nOracle best test threshold: "
      f"{best_row['threshold']:.2f}")
print(f"Oracle best test F1:        "
      f"{best_row['f1']:.4f}")
print(f"At that threshold:")
print(f"  Precision: {best_row['precision']:.4f}")
print(f"  Recall:    {best_row['recall']:.4f}")
print(f"  FPR:       {best_row['fpr']:.4f}")
print(f"  TP={int(best_row['tp'])}, "
      f"FP={int(best_row['fp'])}, "
      f"FN={int(best_row['fn'])}, "
      f"TN={int(best_row['tn'])}")

print(f"\nFor comparison, val-selected threshold "
      f"({saved_threshold:.2f}) gives test F1: "
      f"{threshold_df[threshold_df['threshold'] == round(saved_threshold, 2)]['f1'].values}")

# Save threshold sweep
threshold_df.to_csv(
    OUTPUT_DIR / "03_test_threshold_sweep.csv",
    index=False
)

# Plot threshold sweep
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    label="F1", linewidth=2,
    color="steelblue"
)
ax.plot(
    threshold_df["threshold"],
    threshold_df["precision"],
    label="Precision", linewidth=1.5,
    color="green", alpha=0.7
)
ax.plot(
    threshold_df["threshold"],
    threshold_df["recall"],
    label="Recall", linewidth=1.5,
    color="orange", alpha=0.7
)

ax.axvline(
    x=saved_threshold, color="red",
    linestyle="--", linewidth=1.5,
    label=f"Val threshold={saved_threshold:.2f}"
)
ax.axvline(
    x=best_row["threshold"], color="purple",
    linestyle="--", linewidth=1.5,
    label=f"Oracle best={best_row['threshold']:.2f}"
)

ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title(
    "Test Data: Threshold vs F1/Precision/Recall"
)
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "03_test_threshold_sweep.png",
    dpi=150
)
plt.close(fig)

print("\nSaved: 03_test_threshold_sweep.png")
print("Saved: 03_test_threshold_sweep.csv")


# =============================================================
# DIAGNOSTIC 4: TIME-SERIES PREDICTION PLOT
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 4: Time-Series Prediction Plot")
print("=" * 90)

test_timestamps = pd.to_datetime(
    test_meta["timestamp"]
)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Panel 1: Ground truth
axes[0].fill_between(
    range(len(y_test_attack)),
    y_test_attack,
    alpha=0.4, color="crimson",
    label="Attack present"
)
axes[0].set_ylabel("Ground Truth")
axes[0].set_title(
    "Test Period: Ground Truth vs "
    "Model Predictions Over Time"
)
axes[0].legend()
axes[0].set_ylim(-0.05, 1.05)

# Panel 2: Predicted probability
axes[1].plot(
    range(len(test_probs)),
    test_probs,
    color="steelblue",
    linewidth=0.8,
    label="Attack probability"
)
axes[1].axhline(
    y=saved_threshold, color="red",
    linestyle="--", linewidth=1.5,
    label=f"Val threshold={saved_threshold:.2f}"
)
axes[1].axhline(
    y=best_row["threshold"],
    color="purple",
    linestyle="--", linewidth=1.0,
    label=f"Oracle threshold={best_row['threshold']:.2f}"
)

# Shade attack regions
for i in range(len(y_test_attack)):
    if y_test_attack[i] == 1:
        axes[1].axvspan(
            i - 0.5, i + 0.5,
            alpha=0.1, color="crimson"
        )

axes[1].set_ylabel("Predicted Probability")
axes[1].legend(fontsize=9)
axes[1].set_ylim(-0.02, 1.02)

# Panel 3: Attack ratio over time
axes[2].plot(
    range(len(test_meta)),
    test_meta["attack_ratio"],
    color="darkorange",
    linewidth=0.8,
    label="Attack ratio in window"
)
axes[2].set_ylabel("Attack Ratio")
axes[2].set_xlabel(
    f"Window index "
    f"({test_timestamps.iloc[0].strftime('%H:%M')} "
    f"to {test_timestamps.iloc[-1].strftime('%H:%M')})"
)
axes[2].legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "04_timeseries_predictions.png",
    dpi=150
)
plt.close(fig)

print("Saved: 04_timeseries_predictions.png")


# =============================================================
# DIAGNOSTIC 5: FEATURE DISTRIBUTION SHIFT
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 5: Feature Distribution Shift")
print("=" * 90)

# Compare the LAST timestep of each sequence
# (the most recent observation) between val and test

# X shape = (N, 10, 52)
# Take timestep index 9 (the latest)

val_last = X_val[:, -1, :]   # (N_val, 52)
test_last = X_test[:, -1, :] # (N_test, 52)

# Load feature names
feature_names = []
with open(
    DATA_DIR / "input_feature_names.txt"
) as f:
    for line in f:
        line = line.strip()
        if line:
            feature_names.append(line)

# Only compare the first 26 (original features,
# not the rel_change derivatives)
n_original = 26

shift_results = []

for i in range(n_original):

    val_feat = val_last[:, i]
    test_feat = test_last[:, i]

    val_mean = val_feat.mean()
    val_std = val_feat.std()
    test_mean = test_feat.mean()
    test_std = test_feat.std()

    # Standardized mean difference
    pooled_std = np.sqrt(
        (val_std ** 2 + test_std ** 2) / 2
    )

    if pooled_std > 1e-8:
        cohen_d = abs(
            val_mean - test_mean
        ) / pooled_std
    else:
        cohen_d = 0.0

    shift_results.append({
        "feature": feature_names[i],
        "val_mean": val_mean,
        "val_std": val_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "mean_diff": abs(
            val_mean - test_mean
        ),
        "cohen_d": cohen_d,
    })

shift_df = pd.DataFrame(shift_results)
shift_df = shift_df.sort_values(
    "cohen_d", ascending=False
)

print("\nTop features with largest "
      "distribution shift (Cohen's d):\n")

print(
    shift_df[
        ["feature", "val_mean", "test_mean",
         "cohen_d"]
    ].head(15).to_string(index=False)
)

shift_df.to_csv(
    OUTPUT_DIR / "05_feature_distribution_shift.csv",
    index=False
)

# Plot top 8 shifted features
top_shifted = shift_df.head(8)

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for idx, (_, row) in enumerate(
    top_shifted.iterrows()
):
    feat_idx = feature_names.index(
        row["feature"]
    )

    axes[idx].hist(
        val_last[:, feat_idx], bins=40,
        alpha=0.6, label="Val",
        color="steelblue", density=True
    )
    axes[idx].hist(
        test_last[:, feat_idx], bins=40,
        alpha=0.6, label="Test",
        color="darkorange", density=True
    )
    axes[idx].set_title(
        f"{row['feature']}\n"
        f"(d={row['cohen_d']:.2f})",
        fontsize=9
    )
    axes[idx].legend(fontsize=7)
    axes[idx].tick_params(labelsize=7)

fig.suptitle(
    "Feature Distribution Shift: "
    "Validation vs Test (top 8 by Cohen's d)",
    fontsize=12
)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "05_feature_shift_histograms.png",
    dpi=150
)
plt.close(fig)

print("\nSaved: 05_feature_distribution_shift.csv")
print("Saved: 05_feature_shift_histograms.png")


# =============================================================
# DIAGNOSTIC 6: FALSE NEGATIVE ANALYSIS
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 6: False Negative Analysis")
print("=" * 90)

# Which attack windows does the model miss?

test_preds_at_saved = (
    test_probs >= saved_threshold
).astype(int)

# Build analysis dataframe
analysis_df = test_meta.copy()
analysis_df["probability"] = test_probs
analysis_df["predicted"] = test_preds_at_saved
analysis_df["window_idx"] = range(
    len(analysis_df)
)

# Classify each row
conditions = []
for _, row in analysis_df.iterrows():
    actual = int(row["attack_present"])
    pred = int(row["predicted"])

    if actual == 1 and pred == 1:
        conditions.append("TP")
    elif actual == 1 and pred == 0:
        conditions.append("FN")
    elif actual == 0 and pred == 1:
        conditions.append("FP")
    else:
        conditions.append("TN")

analysis_df["classification"] = conditions

# How many FN are there?
fn_df = analysis_df[
    analysis_df["classification"] == "FN"
]
tp_df = analysis_df[
    analysis_df["classification"] == "TP"
]

print(f"\nFalse Negatives: {len(fn_df)}")
print(f"True Positives:  {len(tp_df)}")

print(f"\nFN probability stats:")
print(f"  mean: {fn_df['probability'].mean():.6f}")
print(f"  max:  {fn_df['probability'].max():.6f}")
print(f"  min:  {fn_df['probability'].min():.6f}")

if len(tp_df) > 0:
    print(f"\nTP probability stats:")
    print(f"  mean: "
          f"{tp_df['probability'].mean():.6f}")
    print(f"  min:  "
          f"{tp_df['probability'].min():.6f}")

# Where do TPs cluster?
if len(tp_df) > 0:
    print(f"\nTP window indices: "
          f"{tp_df['window_idx'].min()} "
          f"to {tp_df['window_idx'].max()}")
    print(f"TP timestamps: "
          f"{tp_df['timestamp'].iloc[0]} "
          f"to {tp_df['timestamp'].iloc[-1]}")

# Save full analysis
analysis_df.to_csv(
    OUTPUT_DIR / "06_classification_analysis.csv",
    index=False
)

print("\nSaved: 06_classification_analysis.csv")


# =============================================================
# DIAGNOSTIC 7: OVERALL SUMMARY
# =============================================================

print("\n" + "=" * 90)
print("DIAGNOSTIC 7: SUMMARY REPORT")
print("=" * 90)

# Gather all results into a summary
summary = {
    "val_roc_auc": val_roc_auc,
    "val_pr_auc": val_pr_auc,
    "test_roc_auc": test_roc_auc,
    "test_pr_auc": test_pr_auc,

    "val_attack_prob_mean":
        val_attack_probs.mean(),
    "val_benign_prob_mean":
        val_benign_probs.mean(),
    "test_attack_prob_mean":
        test_attack_probs.mean(),
    "test_benign_prob_mean":
        test_benign_probs.mean(),

    "saved_threshold": saved_threshold,
    "oracle_best_threshold":
        best_row["threshold"],
    "oracle_best_f1": best_row["f1"],

    "test_fn_count": len(fn_df),
    "test_tp_count": len(tp_df),

    "top_shifted_feature_1":
        shift_df.iloc[0]["feature"],
    "top_shifted_cohen_d_1":
        shift_df.iloc[0]["cohen_d"],
    "top_shifted_feature_2":
        shift_df.iloc[1]["feature"],
    "top_shifted_cohen_d_2":
        shift_df.iloc[1]["cohen_d"],
}

summary_df = pd.DataFrame([
    {"metric": k, "value": v}
    for k, v in summary.items()
])

summary_df.to_csv(
    OUTPUT_DIR / "07_diagnosis_summary.csv",
    index=False
)

# Print the key conclusions
print()
print("KEY FINDINGS:")
print("-" * 60)

print(f"\n1. RANKING QUALITY:")
print(f"   Val  ROC-AUC = {val_roc_auc:.4f}")
print(f"   Test ROC-AUC = {test_roc_auc:.4f}")

if test_roc_auc > 0.70:
    print("   → Model provides useful ranking "
          "on test data")
elif test_roc_auc > 0.55:
    print("   → Model provides weak but "
          "non-random ranking")
else:
    print("   → Model ranking is near random "
          "on test data")

print(f"\n2. THRESHOLD TRANSFER:")
print(f"   Val threshold = {saved_threshold:.2f}")
print(f"   Oracle test threshold = "
      f"{best_row['threshold']:.2f}")
print(f"   Oracle test F1 = "
      f"{best_row['f1']:.4f}")
print(f"   Saved threshold test F1 = 0.4118")

if best_row["f1"] > 0.60:
    print("   → Significant F1 uplift possible "
          "with better threshold")
else:
    print("   → Even oracle threshold "
          "doesn't fix the problem")

print(f"\n3. PROBABILITY CALIBRATION:")
print(f"   Val attack prob mean = "
      f"{val_attack_probs.mean():.4f}")
print(f"   Test attack prob mean = "
      f"{test_attack_probs.mean():.4f}")
print(f"   Ratio = "
      f"{val_attack_probs.mean() / max(test_attack_probs.mean(), 1e-8):.1f}x")

print(f"\n4. DISTRIBUTION SHIFT:")
print(f"   Top shifted: "
      f"{shift_df.iloc[0]['feature']} "
      f"(d={shift_df.iloc[0]['cohen_d']:.2f})")

print(f"\n5. FALSE NEGATIVES:")
print(f"   {len(fn_df)} of {int(y_test_attack.sum())} "
      f"attack windows missed "
      f"({100*len(fn_df)/max(int(y_test_attack.sum()),1):.1f}%)")

print()
print("=" * 90)
print("DIAGNOSIS COMPLETE")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("=" * 90)
