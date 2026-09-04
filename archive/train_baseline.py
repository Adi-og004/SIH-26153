import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path("data/processed/final")

X_TRAIN_PATH = DATA_DIR / "X_train.npy"
X_VAL_PATH = DATA_DIR / "X_val.npy"
X_TEST_PATH = DATA_DIR / "X_test.npy"

TRAIN_META_PATH = DATA_DIR / "train_metadata.csv"
VAL_META_PATH = DATA_DIR / "val_metadata.csv"
TEST_META_PATH = DATA_DIR / "test_metadata.csv"


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 90)
print("LOGISTIC REGRESSION BASELINE")
print("=" * 90)

print("\nLoading data...")

X_train = np.load(X_TRAIN_PATH)
X_val = np.load(X_VAL_PATH)
X_test = np.load(X_TEST_PATH)

train_meta = pd.read_csv(TRAIN_META_PATH)
val_meta = pd.read_csv(VAL_META_PATH)
test_meta = pd.read_csv(TEST_META_PATH)

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)


# =========================================================
# TARGET
# =========================================================

y_train = train_meta["attack_present"].to_numpy()
y_val = val_meta["attack_present"].to_numpy()
y_test = test_meta["attack_present"].to_numpy()


print("\n========== CLASS DISTRIBUTION ==========")

print("Train:")
print(pd.Series(y_train).value_counts())

print("\nValidation:")
print(pd.Series(y_val).value_counts())

print("\nTest:")
print(pd.Series(y_test).value_counts())


# =========================================================
# FLATTEN TEMPORAL SEQUENCES
# =========================================================
#
# Original:
# (samples, 10 timesteps, 38 features)
#
# Logistic Regression requires:
# (samples, 10*38)
#
# Therefore:
#
# 10 × 38 = 380 features
# =========================================================

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_val_flat = X_val.reshape(
    X_val.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)

print("\n========== FLATTENED SHAPES ==========")

print("X_train:", X_train_flat.shape)
print("X_val:  ", X_val_flat.shape)
print("X_test: ", X_test_flat.shape)


# =========================================================
# TRAIN MODEL
# =========================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

model.fit(
    X_train_flat,
    y_train
)

print("Training complete.")


# =========================================================
# PREDICTIONS
# =========================================================

val_prob = model.predict_proba(
    X_val_flat
)[:, 1]

test_prob = model.predict_proba(
    X_test_flat
)[:, 1]


# =========================================================
# CHOOSE THRESHOLD USING VALIDATION DATA
# =========================================================

print("\nFinding threshold using validation data...")

best_threshold = 0.5
best_f1 = -1

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

for threshold in thresholds:

    val_pred = (
        val_prob >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(
    f"Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1: "
    f"{best_f1:.4f}"
)


# =========================================================
# FINAL TEST PREDICTIONS
# =========================================================

test_pred = (
    test_prob >= best_threshold
).astype(int)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    test_pred
)

tn, fp, fn, tp = cm.ravel()

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 90)
print("BASELINE RESULTS")
print("=" * 90)

print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision:           {precision:.4f}")
print(f"Recall:              {recall:.4f}")
print(f"F1 Score:            {f1:.4f}")
print(f"False Positive Rate: {false_positive_rate:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_pred,
        target_names=[
            "Benign",
            "Attack"
        ],
        zero_division=0
    )
)


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Benign", "Attack"]
)

plt.yticks(
    [0, 1],
    ["Benign", "Attack"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    DATA_DIR / "baseline_confusion_matrix.png",
    dpi=150
)

plt.close()


# =========================================================
# SAVE MODEL PREDICTIONS
# =========================================================

results = test_meta.copy()

results["predicted_probability"] = test_prob
results["predicted_attack"] = test_pred

results.to_csv(
    DATA_DIR / "baseline_test_predictions.csv",
    index=False
)

print(
    "\nSaved:",
    DATA_DIR / "baseline_test_predictions.csv"
)

print(
    "Saved:",
    DATA_DIR / "baseline_confusion_matrix.png"
)

print("\nBaseline evaluation complete.")