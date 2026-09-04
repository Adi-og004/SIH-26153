import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

MODEL_DIR = Path(
    "models/world_model_v2"
)

OUTPUT_DIR = (
    MODEL_DIR /
    "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 64

DROPOUT = 0.20

STATE_FEATURES = [
    "flow_count",
    "log_flow_count",
    "log_total_packets",
    "log_total_bytes",
    "window_packet_rate",
    "window_byte_rate",
    "syn_rate",
    "ack_rate",
    "rst_rate",
    "fin_rate",
    "psh_rate",
    "urg_rate",
    "mean_pkt_len",
    "std_pkt_len",
    "log_mean_flow_duration",
    "log_mean_fwd_iat",
    "log_mean_bwd_iat",
    "fwd_packet_ratio",
    "bwd_packet_ratio",
    "fwd_byte_ratio",
    "bwd_byte_ratio",
    "mean_down_up_ratio",
    "log_unique_dst_ports",
    "unique_protocols",
    "dst_port_entropy",
    "protocol_entropy",
]


# =========================================================
# DEVICE
# =========================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 90)
print("WORLD MODEL V2 EVALUATION")
print("=" * 90)

print(
    f"Using device: {DEVICE}"
)


# =========================================================
# LOAD TEST DATA
# =========================================================

print("\nLoading V2 test data...")

X_test = np.load(
    DATA_DIR /
    "X_test.npy"
)

Y_test = np.load(
    DATA_DIR /
    "Y_test.npy"
)

print("\nTest shapes:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# =========================================================
# LOAD TARGET SCALER
# =========================================================

with open(
    DATA_DIR /
    "target_scaler.pkl",
    "rb"
) as file:

    target_scaler = pickle.load(
        file
    )


# =========================================================
# MODEL DEFINITION
# =========================================================

class WorldModelLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        dropout
    ):

        super().__init__()

        self.lstm = nn.LSTM(
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

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.fc1 = nn.Linear(
            hidden_size,
            64
        )

        self.activation = nn.ReLU()

        self.fc2 = nn.Linear(
            64,
            output_size
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_state = output[:, -1, :]

        last_state = self.norm(
            last_state
        )

        hidden = self.fc1(
            last_state
        )

        hidden = self.activation(
            hidden
        )

        return self.fc2(hidden)


# =========================================================
# LOAD MODEL CHECKPOINT
# =========================================================

checkpoint_path = (
    MODEL_DIR /
    "lstm_world_model_v2.pt"
)

print("\nLoading checkpoint...")

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

model = WorldModelLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    output_size=checkpoint["output_size"],
    dropout=DROPOUT
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()

print("Model loaded.")


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

predictions = []

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

with torch.no_grad():

    for start in range(
        0,
        len(X_tensor),
        BATCH_SIZE
    ):

        batch = X_tensor[
            start:start + BATCH_SIZE
        ].to(DEVICE)

        output = model(
            batch
        )

        predictions.append(
            output.cpu().numpy()
        )

predictions = np.concatenate(
    predictions,
    axis=0
)


# =========================================================
# SAFETY CHECK
# =========================================================

if not np.isfinite(
    predictions
).all():

    raise ValueError(
        "V2 predictions contain "
        "NaN or Inf values."
    )


# =========================================================
# NORMALIZED METRICS
# =========================================================

normalized_mae = mean_absolute_error(
    Y_test,
    predictions
)

normalized_rmse = np.sqrt(
    mean_squared_error(
        Y_test,
        predictions
    )
)

normalized_r2 = r2_score(
    Y_test,
    predictions
)


print(
    "\n========== NORMALIZED METRICS =========="
)

print(
    f"Normalized MAE:  "
    f"{normalized_mae:.6f}"
)

print(
    f"Normalized RMSE: "
    f"{normalized_rmse:.6f}"
)

print(
    f"Overall R²:      "
    f"{normalized_r2:.6f}"
)


# =========================================================
# ORIGINAL UNITS
# =========================================================

actual_original = target_scaler.inverse_transform(
    Y_test
)

predicted_original = target_scaler.inverse_transform(
    predictions
)

original_mae = mean_absolute_error(
    actual_original,
    predicted_original
)

original_rmse = np.sqrt(
    mean_squared_error(
        actual_original,
        predicted_original
    )
)

original_r2 = r2_score(
    actual_original,
    predicted_original
)


print(
    "\n========== ORIGINAL-UNIT METRICS =========="
)

print(
    f"Original MAE:  "
    f"{original_mae:.6f}"
)

print(
    f"Original RMSE: "
    f"{original_rmse:.6f}"
)

print(
    f"Overall R²:    "
    f"{original_r2:.6f}"
)


# =========================================================
# PER-FEATURE METRICS
# =========================================================

print(
    "\n========== PER-FEATURE METRICS =========="
)

feature_results = []

for i, feature in enumerate(
    STATE_FEATURES
):

    actual_norm = Y_test[:, i]

    predicted_norm = predictions[:, i]

    actual_orig = actual_original[:, i]

    predicted_orig = predicted_original[:, i]

    mae_norm = mean_absolute_error(
        actual_norm,
        predicted_norm
    )

    rmse_norm = np.sqrt(
        mean_squared_error(
            actual_norm,
            predicted_norm
        )
    )

    r2_norm = r2_score(
        actual_norm,
        predicted_norm
    )

    mae_orig = mean_absolute_error(
        actual_orig,
        predicted_orig
    )

    rmse_orig = np.sqrt(
        mean_squared_error(
            actual_orig,
            predicted_orig
        )
    )

    r2_orig = r2_score(
        actual_orig,
        predicted_orig
    )

    feature_results.append({
        "feature": feature,
        "normalized_mae": mae_norm,
        "normalized_rmse": rmse_norm,
        "normalized_r2": r2_norm,
        "original_mae": mae_orig,
        "original_rmse": rmse_orig,
        "original_r2": r2_orig,
        "actual_mean": np.mean(actual_orig),
        "predicted_mean": np.mean(predicted_orig),
    })


feature_df = pd.DataFrame(
    feature_results
)

feature_df = feature_df.sort_values(
    "normalized_mae"
)

print(
    feature_df[
        [
            "feature",
            "normalized_mae",
            "normalized_rmse",
            "normalized_r2",
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# SAVE METRICS
# =========================================================

feature_df.to_csv(
    OUTPUT_DIR /
    "per_feature_metrics.csv",
    index=False
)

global_df = pd.DataFrame({
    "metric": [
        "normalized_mae",
        "normalized_rmse",
        "normalized_r2",
        "original_mae",
        "original_rmse",
        "original_r2",
    ],
    "value": [
        normalized_mae,
        normalized_rmse,
        normalized_r2,
        original_mae,
        original_rmse,
        original_r2,
    ],
})

global_df.to_csv(
    OUTPUT_DIR /
    "global_metrics.csv",
    index=False
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

np.save(
    OUTPUT_DIR /
    "actual_states.npy",
    actual_original
)

np.save(
    OUTPUT_DIR /
    "predicted_states.npy",
    predicted_original
)


# =========================================================
# COMPLETE
# =========================================================

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\n" + "=" * 90
)

print(
    "WORLD MODEL V2 EVALUATION COMPLETE"
)

print(
    "=" * 90
)