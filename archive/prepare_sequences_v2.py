import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

STATE_DIR = Path("data/processed/states_v2")

OUTPUT_DIR = Path(
    "data/processed/sequences_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv",
]

# 10 × 10-second states = 100 seconds of history
SEQUENCE_LENGTH = 10


# =========================================================
# V2 NETWORK STATE FEATURES
# =========================================================
#
# IMPORTANT:
# These are telemetry-derived features only.
#
# We deliberately exclude:
#   attack_count
#   attack_ratio
#   attack_present
#   dominant_attack
#
# Those are ground-truth metadata and must NOT become
# model input features.
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
# LOAD ONE EPISODE
# =========================================================

def load_episode(filename: str) -> pd.DataFrame:

    path = STATE_DIR / filename

    print("\n" + "=" * 90)
    print(f"Loading: {filename}")
    print("=" * 90)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    # -----------------------------------------------------
    # Timestamp
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Convert state features to numeric
    # -----------------------------------------------------

    for feature in STATE_FEATURES:

        if feature not in df.columns:

            raise KeyError(
                f"Missing required feature: "
                f"{feature}"
            )

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Clean numerical values
    # -----------------------------------------------------

    df[STATE_FEATURES] = (
        df[STATE_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    # -----------------------------------------------------
    # Final numeric check
    # -----------------------------------------------------

    values = df[
        STATE_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(values).all():

        raise ValueError(
            f"Invalid values remain in {filename}"
        )

    return df


# =========================================================
# CREATE RELATIVE-CHANGE FEATURES
# =========================================================

def create_transition_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create temporal change features:

        relative_change_t =
        (S_t - S_(t-1))
        ----------------
        |S_(t-1)| + eps

    These describe how the network state is changing.
    """

    state = df[
        STATE_FEATURES
    ].copy()

    # -----------------------------------------------------
    # Difference between consecutive states
    # -----------------------------------------------------

    delta = state.diff()

    # First state has no previous state.
    delta = delta.fillna(0.0)

    # -----------------------------------------------------
    # Previous state
    # -----------------------------------------------------

    previous_state = state.shift(1)

    # -----------------------------------------------------
    # Relative state change
    # -----------------------------------------------------

    relative_change = (
        delta /
        (
            previous_state.abs()
            + 1e-8
        )
    )

    # First state has no history.
    relative_change.iloc[0] = 0.0

    # -----------------------------------------------------
    # Numerical cleanup
    # -----------------------------------------------------

    relative_change = (
        relative_change
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    # -----------------------------------------------------
    # Clip extreme ratios.
    # -----------------------------------------------------

    relative_change = relative_change.clip(
        lower=-10.0,
        upper=10.0
    )

    # -----------------------------------------------------
    # Rename columns
    # -----------------------------------------------------

    relative_change.columns = [
        f"{feature}_rel_change"
        for feature in STATE_FEATURES
    ]

    # -----------------------------------------------------
    # Combine with original state
    # -----------------------------------------------------

    result = pd.concat(
        [
            df,
            relative_change
        ],
        axis=1
    )

    return result


# =========================================================
# CREATE SEQUENCES
# =========================================================

def create_sequences(
    df: pd.DataFrame,
    episode_name: str
):

    transition_features = [
        f"{feature}_rel_change"
        for feature in STATE_FEATURES
    ]

    input_features = (
        STATE_FEATURES
        +
        transition_features
    )

    # -----------------------------------------------------
    # Convert to arrays
    # -----------------------------------------------------

    feature_data = df[
        input_features
    ].to_numpy(
        dtype=np.float32
    )

    target_data = df[
        STATE_FEATURES
    ].to_numpy(
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Final safety cleanup
    # -----------------------------------------------------

    feature_data = np.nan_to_num(
        feature_data,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0
    )

    target_data = np.nan_to_num(
        target_data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X = []
    Y = []

    target_timestamps = []
    target_attack_present = []
    target_attack_ratio = []
    target_attack_type = []

    # -----------------------------------------------------
    # Sliding temporal window
    # -----------------------------------------------------

    for end_idx in range(
        SEQUENCE_LENGTH,
        len(df)
    ):

        start_idx = (
            end_idx -
            SEQUENCE_LENGTH
        )

        # -------------------------------------------------
        # Input:
        #
        # S(t-9), ..., S(t)
        # -------------------------------------------------

        sequence = feature_data[
            start_idx:end_idx
        ]

        # -------------------------------------------------
        # Target:
        #
        # S(t+1)
        # -------------------------------------------------

        target = target_data[
            end_idx
        ]

        X.append(
            sequence
        )

        Y.append(
            target
        )

        # -------------------------------------------------
        # Ground-truth metadata
        # -------------------------------------------------

        target_timestamps.append(
            df.loc[
                end_idx,
                "window_start"
            ]
        )

        target_attack_present.append(
            int(
                df.loc[
                    end_idx,
                    "attack_present"
                ]
            )
        )

        target_attack_ratio.append(
            float(
                df.loc[
                    end_idx,
                    "attack_ratio"
                ]
            )
        )

        target_attack_type.append(
            str(
                df.loc[
                    end_idx,
                    "dominant_attack"
                ]
            )
        )

    # -----------------------------------------------------
    # Convert lists to NumPy
    # -----------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32
    )

    Y = np.asarray(
        Y,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    if not np.isfinite(X).all():

        raise ValueError(
            "X contains NaN or Inf values."
        )

    if not np.isfinite(Y).all():

        raise ValueError(
            "Y contains NaN or Inf values."
        )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = pd.DataFrame({

        "timestamp":
            target_timestamps,

        "attack_present":
            target_attack_present,

        "attack_ratio":
            target_attack_ratio,

        "dominant_attack":
            target_attack_type,
    })

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("\n========== SEQUENCE RESULT ==========")

    print(
        "Episode:",
        episode_name
    )

    print(
        "Sequences:",
        X.shape[0]
    )

    print(
        "Sequence shape:",
        X.shape
    )

    print(
        "Target shape:",
        Y.shape
    )

    print(
        "Original state features:",
        len(STATE_FEATURES)
    )

    print(
        "Transition features:",
        len(transition_features)
    )

    print(
        "Total input features:",
        len(input_features)
    )

    print(
        "NaN in X:",
        np.isnan(X).sum()
    )

    print(
        "Inf in X:",
        np.isinf(X).sum()
    )

    print(
        "NaN in Y:",
        np.isnan(Y).sum()
    )

    print(
        "Inf in Y:",
        np.isinf(Y).sum()
    )

    return X, Y, metadata


# =========================================================
# SAVE EPISODE
# =========================================================

def save_episode(
    filename: str,
    X: np.ndarray,
    Y: np.ndarray,
    metadata: pd.DataFrame
):

    stem = filename.replace(
        "_states_v2.csv",
        ""
    )

    x_path = (
        OUTPUT_DIR /
        f"{stem}_X.npy"
    )

    y_path = (
        OUTPUT_DIR /
        f"{stem}_Y.npy"
    )

    metadata_path = (
        OUTPUT_DIR /
        f"{stem}_metadata.csv"
    )

    np.save(
        x_path,
        X
    )

    np.save(
        y_path,
        Y
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    print("\nSaved:")
    print(x_path)
    print(y_path)
    print(metadata_path)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 90)
    print("V2 TEMPORAL SEQUENCE PREPARATION")
    print("=" * 90)

    print(
        f"Sequence length: "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"History represented: "
        f"{SEQUENCE_LENGTH * 10} seconds"
    )

    for filename in FILES:

        df = load_episode(
            filename
        )

        df = create_transition_features(
            df
        )

        X, Y, metadata = create_sequences(
            df,
            filename
        )

        save_episode(
            filename,
            X,
            Y,
            metadata
        )

    print("\n" + "=" * 90)
    print("V2 SEQUENCE PREPARATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()