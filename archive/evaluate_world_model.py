import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark"
)

MODEL_DIR = Path(
    "models/world_model"
)

OUTPUT_DIR = Path(
    "models/world_model/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 64

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


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
            dropout=dropout if num_layers > 1 else 0.0
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
# LOAD TEST DATA
# =========================================================

print("=" * 90)
print("WORLD MODEL EVALUATION")
print("=" * 90)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

Y_test = np.load(
    DATA_DIR / "Y_test.npy"
)

print("\nTest shape:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# =========================================================
# LOAD SCALER
# =========================================================

with open(
    DATA_DIR / "target_scaler.pkl",
    "rb"
) as f:

    target_scaler = pickle.load(f)


# =========================================================
# LOAD MODEL
# =========================================================

checkpoint = torch.load(
    MODEL_DIR / "lstm_world_model.pt",
    map_location=DEVICE,
    weights_only=False
)

model = WorldModelLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    output_size=checkpoint["output_size"],
    dropout=0.20
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

predictions = []

print("\nGenerating predictions...")

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
# INVERSE TRANSFORM
# =========================================================

actual_original = target_scaler.inverse_transform(
    Y_test
)

predicted_original = target_scaler.inverse_transform(
    predictions
)


# =========================================================
# GLOBAL METRICS
# =========================================================

mae = mean_absolute_error(
    actual_original,
    predicted_original
)

rmse = np.sqrt(
    mean_squared_error(
        actual_original,
        predicted_original
    )
)


print("\n========== GLOBAL METRICS ==========")

print(
    f"Overall MAE:  {mae:.6f}"
)

print(
    f"Overall RMSE: {rmse:.6f}"
)


# =========================================================
# PER-FEATURE METRICS
# =========================================================

feature_results = []

for i, feature in enumerate(
    STATE_FEATURES
):

    actual = actual_original[:, i]

    predicted = predicted_original[:, i]

    feature_mae = mean_absolute_error(
        actual,
        predicted
    )

    feature_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    feature_results.append({
        "feature": feature,
        "MAE": feature_mae,
        "RMSE": feature_rmse,
        "actual_mean": np.mean(actual),
        "predicted_mean": np.mean(predicted),
    })


results_df = pd.DataFrame(
    feature_results
)

results_df = results_df.sort_values(
    "MAE"
)


print("\n========== PER-FEATURE RESULTS ==========")

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE METRICS
# =========================================================

results_df.to_csv(
    OUTPUT_DIR / "per_feature_metrics.csv",
    index=False
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

np.save(
    OUTPUT_DIR / "actual_test_states.npy",
    actual_original
)

np.save(
    OUTPUT_DIR / "predicted_test_states.npy",
    predicted_original
)


# =========================================================
# SAVE SUMMARY
# =========================================================

summary = pd.DataFrame({
    "metric": [
        "overall_mae",
        "overall_rmse"
    ],
    "value": [
        mae,
        rmse
    ]
})

summary.to_csv(
    OUTPUT_DIR / "global_metrics.csv",
    index=False
)


# =========================================================
# COMPLETE
# =========================================================

print("\nSaved evaluation results to:")
print(OUTPUT_DIR)

print("\n" + "=" * 90)
print("WORLD MODEL EVALUATION COMPLETE")
print("=" * 90)