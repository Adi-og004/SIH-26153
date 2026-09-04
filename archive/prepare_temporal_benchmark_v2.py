import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================

STATE_FILE = Path(
    "data/processed/states_v2/"
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states_v2.csv"
)

OUTPUT_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEQUENCE_LENGTH = 10

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =========================================================
# V2 STATE FEATURES
# =========================================================

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
# LOAD DATA
# =========================================================

print("=" * 90)
print("V2 TEMPORAL BENCHMARK PREPARATION")
print("=" * 90)

print("\nLoading V2 state dataset...")

if not STATE_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{STATE_FILE}"
    )

df = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

print(
    "Total states:",
    len(df)
)


# =========================================================
# TIMESTAMP
# =========================================================

df["window_start"] = pd.to_datetime(
    df["window_start"],
    errors="coerce"
)

df = df.dropna(
    subset=["window_start"]
)

df = (
    df.sort_values("window_start")
      .reset_index(drop=True)
)


# =========================================================
# NUMERICAL CLEANING
# =========================================================

for feature in STATE_FEATURES:

    if feature not in df.columns:
        raise KeyError(
            f"Missing V2 state feature: {feature}"
        )

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

df[STATE_FEATURES] = (
    df[STATE_FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)


# =========================================================
# CREATE TRANSITION FEATURES
# =========================================================
#
# For each state:
#
# relative_change_t =
#     (S_t - S_(t-1))
#     ----------------
#     |S_(t-1)| + eps
#
# =========================================================

state = df[STATE_FEATURES].copy()

delta = state.diff().fillna(0.0)

previous_state = state.shift(1)

relative_change = (
    delta /
    (
        previous_state.abs()
        + 1e-8
    )
)

# First state has no previous state.
relative_change.iloc[0] = 0.0

relative_change = (
    relative_change
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
    .clip(-10.0, 10.0)
)

relative_change.columns = [
    f"{feature}_rel_change"
    for feature in STATE_FEATURES
]


# =========================================================
# COMBINE ORIGINAL + TRANSITION FEATURES
# =========================================================

TRANSITION_FEATURES = list(
    relative_change.columns
)

INPUT_FEATURES = (
    STATE_FEATURES
    +
    TRANSITION_FEATURES
)

df = pd.concat(
    [
        df,
        relative_change
    ],
    axis=1
)


# =========================================================
# FINAL NUMERICAL CHECK
# =========================================================

input_values = df[
    INPUT_FEATURES
].to_numpy(
    dtype=np.float32
)

if not np.isfinite(
    input_values
).all():

    raise ValueError(
        "Invalid values remain in V2 input features."
    )


# =========================================================
# TEMPORAL SPLIT
# =========================================================
#
# IMPORTANT:
# We split chronologically.
#
# We do NOT randomly shuffle states.
#
# 70% → TRAIN
# 15% → VALIDATION
# 15% → TEST
# =========================================================

n_states = len(df)

train_end = int(
    n_states * TRAIN_RATIO
)

val_end = train_end + int(
    n_states * VAL_RATIO
)

train_df = df.iloc[
    :train_end
].copy()

val_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()


print("\n========== TEMPORAL SPLIT ==========")

print(
    "Training states:",
    len(train_df)
)

print(
    "Validation states:",
    len(val_df)
)

print(
    "Test states:",
    len(test_df)
)

print("\nTraining period:")
print(
    train_df["window_start"].min(),
    "→",
    train_df["window_start"].max()
)

print("\nValidation period:")
print(
    val_df["window_start"].min(),
    "→",
    val_df["window_start"].max()
)

print("\nTest period:")
print(
    test_df["window_start"].min(),
    "→",
    test_df["window_start"].max()
)


# =========================================================
# FIT INPUT SCALER ON TRAINING DATA ONLY
# =========================================================

print(
    "\nFitting INPUT scaler on TRAINING data only..."
)

input_scaler = StandardScaler()

input_scaler.fit(
    train_df[INPUT_FEATURES]
)


# =========================================================
# FIT TARGET SCALER ON TRAINING TARGETS ONLY
# =========================================================

print(
    "Fitting TARGET scaler on TRAINING data only..."
)

target_scaler = StandardScaler()

target_scaler.fit(
    train_df[STATE_FEATURES]
)


# =========================================================
# TRANSFORM INPUT FEATURES
# =========================================================

def transform_inputs(dataframe):

    values = input_scaler.transform(
        dataframe[INPUT_FEATURES]
    )

    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0
    )

    return values


# =========================================================
# TRANSFORM TARGET FEATURES
# =========================================================

def transform_targets(dataframe):

    values = target_scaler.transform(
        dataframe[STATE_FEATURES]
    )

    values = np.asarray(
        values,
        dtype=np.float32
    )

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0
    )

    return values


train_inputs = transform_inputs(
    train_df
)

val_inputs = transform_inputs(
    val_df
)

test_inputs = transform_inputs(
    test_df
)

train_targets = transform_targets(
    train_df
)

val_targets = transform_targets(
    val_df
)

test_targets = transform_targets(
    test_df
)


# =========================================================
# CREATE SEQUENCES
# =========================================================
#
# Input:
#
# [S(t-9), ..., S(t)]
#
# Target:
#
# S(t+1)
#
# =========================================================

def create_sequences(
    inputs,
    targets,
    dataframe
):

    X = []
    Y = []

    timestamps = []
    attack_present = []
    attack_ratio = []
    dominant_attack = []

    for end_idx in range(
        SEQUENCE_LENGTH,
        len(inputs)
    ):

        start_idx = (
            end_idx -
            SEQUENCE_LENGTH
        )

        # 10 previous states
        sequence = inputs[
            start_idx:end_idx
        ]

        # Next state
        target = targets[
            end_idx
        ]

        X.append(
            sequence
        )

        Y.append(
            target
        )

        timestamps.append(
            dataframe.iloc[
                end_idx
            ]["window_start"]
        )

        attack_present.append(
            int(
                dataframe.iloc[
                    end_idx
                ]["attack_present"]
            )
        )

        attack_ratio.append(
            float(
                dataframe.iloc[
                    end_idx
                ]["attack_ratio"]
            )
        )

        dominant_attack.append(
            str(
                dataframe.iloc[
                    end_idx
                ]["dominant_attack"]
            )
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    Y = np.asarray(
        Y,
        dtype=np.float32
    )

    metadata = pd.DataFrame({
        "timestamp": timestamps,
        "attack_present": attack_present,
        "attack_ratio": attack_ratio,
        "dominant_attack": dominant_attack,
    })

    return X, Y, metadata


# =========================================================
# CREATE FINAL SEQUENCES
# =========================================================

X_train, Y_train, train_meta = create_sequences(
    train_inputs,
    train_targets,
    train_df
)

X_val, Y_val, val_meta = create_sequences(
    val_inputs,
    val_targets,
    val_df
)

X_test, Y_test, test_meta = create_sequences(
    test_inputs,
    test_targets,
    test_df
)


# =========================================================
# VALIDATE
# =========================================================

print("\n========== FINAL V2 SEQUENCES ==========")

print(
    "X_train:",
    X_train.shape
)

print(
    "Y_train:",
    Y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "Y_val:",
    Y_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "Y_test:",
    Y_test.shape
)


print(
    "\nInput features:",
    len(INPUT_FEATURES)
)

print(
    "State features:",
    len(STATE_FEATURES)
)

print(
    "Transition features:",
    len(TRANSITION_FEATURES)
)


# =========================================================
# NUMERICAL VALIDATION
# =========================================================

print(
    "\n========== NUMERICAL VALIDATION =========="
)

for name, array in [
    ("X_train", X_train),
    ("Y_train", Y_train),
    ("X_val", X_val),
    ("Y_val", Y_val),
    ("X_test", X_test),
    ("Y_test", Y_test),
]:

    print(
        f"{name}: "
        f"NaN={np.isnan(array).sum()}, "
        f"Inf={np.isinf(array).sum()}"
    )

    if not np.isfinite(array).all():

        raise ValueError(
            f"{name} contains invalid values."
        )


# =========================================================
# SAVE ARRAYS
# =========================================================

print(
    "\nSaving V2 benchmark data..."
)

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "Y_train.npy",
    Y_train
)

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "Y_val.npy",
    Y_val
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "Y_test.npy",
    Y_test
)


# =========================================================
# SAVE METADATA
# =========================================================

train_meta.to_csv(
    OUTPUT_DIR / "train_metadata.csv",
    index=False
)

val_meta.to_csv(
    OUTPUT_DIR / "val_metadata.csv",
    index=False
)

test_meta.to_csv(
    OUTPUT_DIR / "test_metadata.csv",
    index=False
)


# =========================================================
# SAVE SCALERS
# =========================================================

with open(
    OUTPUT_DIR / "input_scaler.pkl",
    "wb"
) as file:

    pickle.dump(
        input_scaler,
        file
    )

with open(
    OUTPUT_DIR / "target_scaler.pkl",
    "wb"
) as file:

    pickle.dump(
        target_scaler,
        file
    )


# =========================================================
# SAVE FEATURE NAMES
# =========================================================

with open(
    OUTPUT_DIR / "input_feature_names.txt",
    "w"
) as file:

    for feature in INPUT_FEATURES:

        file.write(
            feature + "\n"
        )

with open(
    OUTPUT_DIR / "state_feature_names.txt",
    "w"
) as file:

    for feature in STATE_FEATURES:

        file.write(
            feature + "\n"
        )


# =========================================================
# FINAL REPORT
# =========================================================

print("\n========== FINAL REPORT ==========")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)

print(
    "Test samples:",
    len(X_test)
)

print(
    "Sequence length:",
    X_train.shape[1]
)

print(
    "Input feature count:",
    X_train.shape[2]
)

print(
    "Target feature count:",
    Y_train.shape[1]
)

print(
    "\nSaved everything to:"
)

print(
    OUTPUT_DIR
)

print("\n" + "=" * 90)
print("V2 TEMPORAL BENCHMARK PREPARATION COMPLETE")
print("=" * 90)