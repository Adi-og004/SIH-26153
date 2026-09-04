import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark"
)

OUTPUT_DIR = Path(
    "models/world_model/persistence_baseline"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

# Our input sequence has:
# 10 previous states
# Each state has 38 input features:
#   19 original state features
#   19 relative-change features
#
# The target, however, contains only the 19 original
# network-state features.
#
# For persistence, we need the ORIGINAL state from the
# final timestep of the input sequence.
#
# The first 19 features are the original state features.
# The remaining 19 are relative-change features.

NUM_STATE_FEATURES = 19

STATE_FEATURES = [
    "flow_count",
    "total_packets",
    "total_bytes",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
    "urg_count",
    "mean_flow_duration",
    "mean_pkt_len",
    "std_pkt_len",
    "mean_fwd_iat",
    "mean_bwd_iat",
    "mean_down_up_ratio",
    "unique_dst_ports",
    "unique_protocols",
    "window_packet_rate",
    "window_byte_rate",
]


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 90)
print("PERSISTENCE BASELINE")
print("=" * 90)

print("\nLoading temporal benchmark...")

X_train = np.load(
    DATA_DIR / "X_train.npy"
)

Y_train = np.load(
    DATA_DIR / "Y_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_val.npy"
)

Y_val = np.load(
    DATA_DIR / "Y_val.npy"
)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

Y_test = np.load(
    DATA_DIR / "Y_test.npy"
)


print("\n========== DATA SHAPES ==========")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_val:  ", X_val.shape)
print("Y_val:  ", Y_val.shape)

print("X_test: ", X_test.shape)
print("Y_test: ", Y_test.shape)


# =========================================================
# CHECK DATA
# =========================================================

for name, array in [
    ("X_train", X_train),
    ("Y_train", Y_train),
    ("X_val", X_val),
    ("Y_val", Y_val),
    ("X_test", X_test),
    ("Y_test", Y_test),
]:

    if not np.isfinite(array).all():

        raise ValueError(
            f"{name} contains NaN or Inf values."
        )


# =========================================================
# PERSISTENCE PREDICTION
# =========================================================
#
# Input:
#
# [S(t-9), ..., S(t)]
#
# Persistence prediction:
#
# predicted S(t+1) = S(t)
#
# Since X contains:
#
# [19 state features + 19 change features]
#
# the first 19 values of the FINAL timestep are S(t).
# =========================================================

print("\nCreating persistence predictions...")

predicted_train = X_train[
    :, -1, :NUM_STATE_FEATURES
]

predicted_val = X_val[
    :, -1, :NUM_STATE_FEATURES
]

predicted_test = X_test[
    :, -1, :NUM_STATE_FEATURES
]


# =========================================================
# HELPER: CALCULATE METRICS
# =========================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# =========================================================
# VALIDATION
# =========================================================

val_mae, val_rmse, val_r2 = calculate_metrics(
    Y_val,
    predicted_val
)

print("\n========== VALIDATION RESULTS ==========")

print(
    f"MAE:  {val_mae:.6f}"
)

print(
    f"RMSE: {val_rmse:.6f}"
)

print(
    f"R²:   {val_r2:.6f}"
)


# =========================================================
# TEST
# =========================================================

test_mae, test_rmse, test_r2 = calculate_metrics(
    Y_test,
    predicted_test
)

print("\n========== TEST RESULTS ==========")

print(
    f"MAE:  {test_mae:.6f}"
)

print(
    f"RMSE: {test_rmse:.6f}"
)

print(
    f"R²:   {test_r2:.6f}"
)


# =========================================================
# PER-FEATURE TEST METRICS
# =========================================================

print(
    "\n========== PER-FEATURE TEST METRICS =========="
)

feature_results = []

for i, feature in enumerate(
    STATE_FEATURES
):

    actual = Y_test[:, i]

    predicted = predicted_test[:, i]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    feature_results.append({
        "feature": feature,
        "normalized_mae": mae,
        "normalized_rmse": rmse,
        "normalized_r2": r2,
    })


feature_df = pd.DataFrame(
    feature_results
)

print(
    feature_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE METRICS
# =========================================================

summary_df = pd.DataFrame({

    "split": [
        "validation",
        "test"
    ],

    "MAE": [
        val_mae,
        test_mae
    ],

    "RMSE": [
        val_rmse,
        test_rmse
    ],

    "R2": [
        val_r2,
        test_r2
    ]
})


summary_df.to_csv(
    OUTPUT_DIR /
    "persistence_global_metrics.csv",
    index=False
)

feature_df.to_csv(
    OUTPUT_DIR /
    "persistence_feature_metrics.csv",
    index=False
)


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

np.save(
    OUTPUT_DIR /
    "persistence_test_predictions.npy",
    predicted_test
)

np.save(
    OUTPUT_DIR /
    "persistence_test_actual.npy",
    Y_test
)


# =========================================================
# LOAD METADATA
# =========================================================

test_metadata = pd.read_csv(
    DATA_DIR /
    "test_metadata.csv"
)


# =========================================================
# SAVE METADATA + BASIC PREDICTION INFORMATION
# =========================================================

prediction_summary = test_metadata.copy()

prediction_summary[
    "persistence_mae"
] = np.mean(
    np.abs(
        Y_test - predicted_test
    ),
    axis=1
)

prediction_summary[
    "persistence_rmse"
] = np.sqrt(
    np.mean(
        (Y_test - predicted_test) ** 2,
        axis=1
    )
)

prediction_summary.to_csv(
    OUTPUT_DIR /
    "persistence_test_predictions.csv",
    index=False
)


# =========================================================
# COMPARE WITH LSTM
# =========================================================

print("\n========== COMPARISON WITH LSTM V1 ==========")

print(
    f"Persistence Test MAE:  {test_mae:.6f}"
)

print(
    f"Persistence Test RMSE: {test_rmse:.6f}"
)

print(
    f"Persistence Test R²:   {test_r2:.6f}"
)

print(
    "\nYour LSTM V1:"
)

print(
    "LSTM Test Loss: 0.169973"
)

print(
    "LSTM R²: approximately 0.0087"
)


# =========================================================
# FINISH
# =========================================================

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 90)

print("PERSISTENCE BASELINE COMPLETE")

print("=" * 90)