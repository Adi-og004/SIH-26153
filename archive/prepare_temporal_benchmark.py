import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


# =========================================================
# CONFIGURATION
# =========================================================

STATE_FILE = (
    "data/processed/states/"
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states.csv"
)

OUTPUT_DIR = Path(
    "data/processed/temporal_benchmark"
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
# NETWORK STATE FEATURES
# =========================================================

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
print("TEMPORAL BENCHMARK PREPARATION")
print("=" * 90)

print("\nLoading dataset...")

df = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

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

print(
    "Total states:",
    len(df)
)


# =========================================================
# CLEAN FEATURES
# =========================================================

for feature in STATE_FEATURES:

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
# CREATE RELATIVE CHANGE FEATURES
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

relative_change.iloc[0] = 0.0

relative_change = (
    relative_change
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)

relative_change = relative_change.clip(
    -10.0,
    10.0
)

relative_change.columns = [
    f"{feature}_rel_change"
    for feature in STATE_FEATURES
]


# =========================================================
# COMBINE FEATURES
# =========================================================

INPUT_FEATURES = (
    STATE_FEATURES
    +
    list(relative_change.columns)
)

df = pd.concat(
    [
        df,
        relative_change
    ],
    axis=1
)


# =========================================================
# DETERMINE TEMPORAL SPLIT
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
# FIT SCALER ONLY ON TRAINING DATA
# =========================================================

print("\nFitting scaler on TRAINING data only...")

scaler = StandardScaler()

scaler.fit(
    train_df[INPUT_FEATURES]
)


# =========================================================
# TRANSFORM DATA
# =========================================================

def transform_features(
    dataframe
):

    transformed = scaler.transform(
        dataframe[INPUT_FEATURES]
    )

    transformed = np.asarray(
        transformed,
        dtype=np.float32
    )

    transformed = np.nan_to_num(
        transformed,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0
    )

    return transformed


train_features = transform_features(
    train_df
)

val_features = transform_features(
    val_df
)

test_features = transform_features(
    test_df
)


# =========================================================
# TARGET FEATURES
# =========================================================

target_scaler = StandardScaler()

target_scaler.fit(
    train_df[STATE_FEATURES]
)


def transform_targets(
    dataframe
):

    transformed = target_scaler.transform(
        dataframe[STATE_FEATURES]
    )

    transformed = np.asarray(
        transformed,
        dtype=np.float32
    )

    transformed = np.nan_to_num(
        transformed,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0
    )

    return transformed


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
# SEQUENCE CREATION
# =========================================================

def create_sequences(
    features,
    targets,
    dataframe
):

    X = []
    Y = []

    metadata = []

    for end_idx in range(
        SEQUENCE_LENGTH,
        len(features)
    ):

        start_idx = (
            end_idx
            -
            SEQUENCE_LENGTH
        )

        sequence = features[
            start_idx:end_idx
        ]

        target = targets[
            end_idx
        ]

        X.append(
            sequence
        )

        Y.append(
            target
        )

        metadata.append({
            "timestamp":
                dataframe.iloc[
                    end_idx
                ]["window_start"],

            "attack_present":
                int(
                    dataframe.iloc[
                        end_idx
                    ]["attack_present"]
                ),

            "attack_ratio":
                float(
                    dataframe.iloc[
                        end_idx
                    ]["attack_ratio"]
                ),

            "dominant_attack":
                dataframe.iloc[
                    end_idx
                ]["dominant_attack"]
        })

    X = np.asarray(
        X,
        dtype=np.float32
    )

    Y = np.asarray(
        Y,
        dtype=np.float32
    )

    metadata = pd.DataFrame(
        metadata
    )

    return X, Y, metadata


# =========================================================
# CREATE FINAL SPLIT DATA
# =========================================================

X_train, Y_train, train_meta = create_sequences(
    train_features,
    train_targets,
    train_df
)

X_val, Y_val, val_meta = create_sequences(
    val_features,
    val_targets,
    val_df
)

X_test, Y_test, test_meta = create_sequences(
    test_features,
    test_targets,
    test_df
)


# =========================================================
# REPORT
# =========================================================

print("\n========== FINAL SEQUENCES ==========")

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


# =========================================================
# CHECK FOR INVALID NUMBERS
# =========================================================

print("\n========== NUMERICAL VALIDATION ==========")

for name, array in [
    ("X_train", X_train),
    ("Y_train", Y_train),
    ("X_val", X_val),
    ("Y_val", Y_val),
    ("X_test", X_test),
    ("Y_test", Y_test)
]:

    print(
        f"{name}: "
        f"NaN={np.isnan(array).sum()}, "
        f"Inf={np.isinf(array).sum()}"
    )


# =========================================================
# SAVE ARRAYS
# =========================================================

print("\nSaving benchmark data...")

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
        scaler,
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
# FINISH
# =========================================================

print("\nSaved to:")

print(
    OUTPUT_DIR
)

print("\n" + "=" * 90)
print("TEMPORAL BENCHMARK PREPARATION COMPLETE")
print("=" * 90)