import numpy as np
import pandas as pd

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

DATA_DIR = Path("data/processed/temporal_benchmark")


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 90)
print("TEMPORAL LOGISTIC REGRESSION BASELINE")
print("=" * 90)

print("\nLoading temporal benchmark...")

X_train = np.load(
    DATA_DIR / "X_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_val.npy"
)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

train_meta = pd.read_csv(
    DATA_DIR / "train_metadata.csv"
)

val_meta = pd.read_csv(
    DATA_DIR / "val_metadata.csv"
)

test_meta = pd.read_csv(
    DATA_DIR / "test_metadata.csv"
)


print("\n========== ORIGINAL SHAPES ==========")

print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)


# =========================================================
# TARGET
# =========================================================
#
# We predict whether the NEXT state contains attack traffic.
#
# attack_present:
#   0 = benign
#   1 = attack
# =========================================================

y_train = train_meta[
    "attack_present"
].to_numpy(dtype=np.int32)

y_val = val_meta[
    "attack_present"
].to_numpy(dtype=np.int32)

y_test = test_meta[
    "attack_present"
].to_numpy(dtype=np.int32)


print("\n========== CLASS DISTRIBUTION ==========")

print("\nTRAIN")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)

print("\nVALIDATION")
print(
    pd.Series(y_val)
    .value_counts()
    .sort_index()
)

print("\nTEST")
print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)


# =========================================================
# CHECK DATA
# =========================================================

for name, array in [
    ("X_train", X_train),
    ("X_val", X_val),
    ("X_test", X_test)
]:

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# =========================================================
# FLATTEN TEMPORAL INPUT
# =========================================================
#
# X:
#   (samples, 10 time steps, 38 features)
#
# Logistic Regression cannot directly consume the
# temporal tensor, so we flatten:
#
#   10 × 38 = 380 features
#
# This deliberately ignores the temporal structure.
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
# TRAIN LOGISTIC REGRESSION
# =========================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=3000,
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
# VALIDATION PROBABILITIES
# =========================================================

val_probability = model.predict_proba(
    X_val_flat
)[:, 1]


# =========================================================
# SELECT THRESHOLD USING VALIDATION SET
# =========================================================

print(
    "\nFinding best classification threshold "
    "using validation data..."
)

best_threshold = 0.50
best_f1 = -1.0

thresholds = np.arange(
    0.05,
    0.96,
    0.01
)

for threshold in thresholds:

    prediction = (
        val_probability >= threshold
    ).astype(np.int32)

    score = f1_score(
        y_val,
        prediction,
        zero_division=0
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = threshold


print(
    f"Best validation threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best validation F1: "
    f"{best_f1:.4f}"
)


# =========================================================
# TEST PREDICTIONS
# =========================================================

test_probability = model.predict_proba(
    X_test_flat
)[:, 1]

test_prediction = (
    test_probability >= best_threshold
).astype(np.int32)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    test_prediction
)

precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    test_prediction
)

if cm.shape == (2, 2):

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

else:

    tn = fp = fn = tp = 0
    false_positive_rate = 0.0


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 90)
print("TEMPORAL BASELINE RESULTS")
print("=" * 90)

print(
    f"Accuracy:            {accuracy:.4f}"
)

print(
    f"Precision:           {precision:.4f}"
)

print(
    f"Recall:              {recall:.4f}"
)

print(
    f"F1 Score:            {f1:.4f}"
)

print(
    f"False Positive Rate: {false_positive_rate:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_prediction,
        target_names=[
            "Benign",
            "Attack"
        ],
        zero_division=0
    )
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

results = test_meta.copy()

results[
    "predicted_probability"
] = test_probability

results[
    "predicted_attack"
] = test_prediction

output_path = (
    DATA_DIR /
    "temporal_baseline_predictions.csv"
)

results.to_csv(
    output_path,
    index=False
)


# =========================================================
# SAVE SUMMARY
# =========================================================

summary = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "threshold",
    ],
    "value": [
        accuracy,
        precision,
        recall,
        f1,
        false_positive_rate,
        best_threshold,
    ]
})

summary.to_csv(
    DATA_DIR /
    "temporal_baseline_metrics.csv",
    index=False
)


print("\nSaved:")
print(output_path)

print(
    DATA_DIR /
    "temporal_baseline_metrics.csv"
)

print("\nTemporal baseline complete.")