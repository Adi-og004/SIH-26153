import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PATHS
# =========================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark"
)

MODEL_DIR = Path(
    "models/world_model"
)

OUTPUT_DIR = MODEL_DIR / "evaluation_v2"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

BATCH_SIZE = 64

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
# DEVICE
# =========================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 90)
print("WORLD MODEL EVALUATION V2")
print("=" * 90)

print(
    f"Using device: {DEVICE}"
)


# =========================================================
# LOAD TEST DATA
# =========================================================

print("\nLoading test data...")

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

Y_test = np.load(
    DATA_DIR / "Y_test.npy"
)

print("\nTest shapes:")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


# =========================================================
# LOAD TARGET SCALER
# =========================================================

with open(
    DATA_DIR / "target_scaler.pkl",
    "rb"
) as f:

    target_scaler = pickle.load(f)


# =========================================================
# MODEL
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
# LOAD CHECKPOINT
# =========================================================

checkpoint_path = (
    MODEL_DIR /
    "lstm_world_model.pt"
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
    dropout=0.20
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
# VALIDATION
# =========================================================

if not np.isfinite(predictions).all():

    raise ValueError(
        "Model produced NaN or Inf predictions."
    )


# =========================================================
# 1. NORMALIZED METRICS
# =========================================================

print("\n========== NORMALIZED METRICS ==========")

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
    f"Normalized MAE:  {normalized_mae:.6f}"
)

print(
    f"Normalized RMSE: {normalized_rmse:.6f}"
)

print(
    f"Overall R²:      {normalized_r2:.6f}"
)


# =========================================================
# 2. ORIGINAL-UNIT METRICS
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


print("\n========== ORIGINAL-UNIT METRICS ==========")

print(
    f"Original MAE:  {original_mae:.6f}"
)

print(
    f"Original RMSE: {original_rmse:.6f}"
)

print(
    f"Overall R²:    {original_r2:.6f}"
)


# =========================================================
# 3. PER-FEATURE METRICS
# =========================================================

print("\n========== PER-FEATURE METRICS ==========")

feature_results = []

for i, feature in enumerate(
    STATE_FEATURES
):

    actual_norm = Y_test[:, i]

    predicted_norm = predictions[:, i]

    actual_orig = actual_original[:, i]

    predicted_orig = predicted_original[:, i]

    feature_mae_norm = mean_absolute_error(
        actual_norm,
        predicted_norm
    )

    feature_rmse_norm = np.sqrt(
        mean_squared_error(
            actual_norm,
            predicted_norm
        )
    )

    feature_r2_norm = r2_score(
        actual_norm,
        predicted_norm
    )

    feature_mae_orig = mean_absolute_error(
        actual_orig,
        predicted_orig
    )

    feature_rmse_orig = np.sqrt(
        mean_squared_error(
            actual_orig,
            predicted_orig
        )
    )

    feature_r2_orig = r2_score(
        actual_orig,
        predicted_orig
    )

    feature_results.append({

        "feature":
            feature,

        "normalized_mae":
            feature_mae_norm,

        "normalized_rmse":
            feature_rmse_norm,

        "normalized_r2":
            feature_r2_norm,

        "original_mae":
            feature_mae_orig,

        "original_rmse":
            feature_rmse_orig,

        "original_r2":
            feature_r2_orig,

        "actual_mean":
            np.mean(actual_orig),

        "predicted_mean":
            np.mean(predicted_orig),
    })


results_df = pd.DataFrame(
    feature_results
)

results_df = results_df.sort_values(
    "normalized_mae"
)

print(
    results_df[
        [
            "feature",
            "normalized_mae",
            "normalized_rmse",
            "normalized_r2"
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# SAVE FEATURE METRICS
# =========================================================

results_df.to_csv(
    OUTPUT_DIR /
    "per_feature_metrics.csv",
    index=False
)


# =========================================================
# SAVE GLOBAL METRICS
# =========================================================

global_metrics = pd.DataFrame({

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
    ]
})

global_metrics.to_csv(
    OUTPUT_DIR /
    "global_metrics.csv",
    index=False
)


# =========================================================
# 4. ACTUAL VS PREDICTED PLOTS
# =========================================================

# We will plot a manageable subset of the test sequence.

PLOT_COUNT = min(
    300,
    len(Y_test)
)

x_axis = np.arange(
    PLOT_COUNT
)

PLOT_FEATURES = [
    "flow_count",
    "syn_count",
    "ack_count",
    "rst_count",
    "unique_dst_ports",
    "window_packet_rate",
    "window_byte_rate",
]


print(
    "\nCreating actual-vs-predicted plots..."
)

for feature in PLOT_FEATURES:

    index = STATE_FEATURES.index(
        feature
    )

    actual = actual_original[
        :PLOT_COUNT,
        index
    ]

    predicted = predicted_original[
        :PLOT_COUNT,
        index
    ]

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        x_axis,
        actual,
        label="Actual"
    )

    plt.plot(
        x_axis,
        predicted,
        label="Predicted"
    )

    plt.title(
        f"Actual vs Predicted: {feature}"
    )

    plt.xlabel(
        "Test sample"
    )

    plt.ylabel(
        feature
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"{feature}_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# =========================================================
# SAVE ARRAYS
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
# FINISH
# =========================================================

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\n" + "=" * 90)
print("WORLD MODEL EVALUATION V2 COMPLETE")
print("=" * 90)