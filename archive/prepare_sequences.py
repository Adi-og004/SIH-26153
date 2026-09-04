import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

STATE_DIR = Path("data/processed/states")
OUTPUT_DIR = Path("data/processed/sequences")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_states.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter_states.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter_states.csv",
]

# Number of previous states given to the model.
# 10 states × 10 seconds = 100 seconds of history.
SEQUENCE_LENGTH = 10

# =========================================================
# NETWORK STATE FEATURES
# =========================================================
#
# IMPORTANT:
# These are the features the model is allowed to observe.
#
# Ground-truth columns such as:
# attack_present
# attack_ratio
# dominant_attack
#
# are deliberately NOT included here.
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
# LOAD ONE EPISODE
# =========================================================

def load_episode(filename: str) -> pd.DataFrame:
    """
    Load one network-state episode.

    Returns:
        Clean, chronologically sorted DataFrame.
    """

    path = STATE_DIR / filename

    print(f"\nLoading: {filename}")

    if not path.exists():
        raise FileNotFoundError(
            f"State file not found: {path}"
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

    # Remove rows with invalid timestamps.
    df = df.dropna(
        subset=["window_start"]
    )

    # Sort chronologically.
    df = (
        df.sort_values("window_start")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Make sure state features are numeric.
    # -----------------------------------------------------

    for feature in STATE_FEATURES:

        if feature not in df.columns:
            raise KeyError(
                f"Required feature '{feature}' "
                f"not found in {filename}"
            )

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Replace invalid values.
    # -----------------------------------------------------

    df[STATE_FEATURES] = (
        df[STATE_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    return df


# =========================================================
# CREATE TEMPORAL / TRANSITION FEATURES
# =========================================================

def create_transition_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    state = df[STATE_FEATURES].copy()

    # -----------------------------------------------------
    # First difference:
    #
    # delta_t = S_t - S_(t-1)
    # -----------------------------------------------------

    delta = state.diff()

    # The first state has no previous state.
    delta = delta.fillna(0.0)

    # -----------------------------------------------------
    # Previous state
    # -----------------------------------------------------

    previous_state = state.shift(1)

    # -----------------------------------------------------
    # Relative change:
    #
    # (S_t - S_(t-1))
    # -----------------
    # |S_(t-1)| + eps
    # -----------------------------------------------------

    relative_change = (
        delta /
        (
            previous_state.abs() +
            1e-8
        )
    )

    # -----------------------------------------------------
    # First state has no previous state.
    # -----------------------------------------------------

    relative_change.iloc[0] = 0.0

    # -----------------------------------------------------
    # Remove invalid values.
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
    # Prevent extreme relative changes from dominating.
    # -----------------------------------------------------

    relative_change = relative_change.clip(
        lower=-10.0,
        upper=10.0
    )

    # -----------------------------------------------------
    # Rename transition features.
    # -----------------------------------------------------

    relative_change.columns = [
        f"{column}_rel_change"
        for column in STATE_FEATURES
    ]

    # -----------------------------------------------------
    # Combine original state + transition features.
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
# CREATE TEMPORAL SEQUENCES
# =========================================================

def create_sequences(
    df: pd.DataFrame,
    episode_name: str
):
    """
    Create:

        X:
        [S(t-9), ..., S(t)]

        Y:
        S(t+1)

    Each input state contains:
        19 original state features
        + 19 relative-change features
        = 38 input features.

    Target contains only:
        19 state features.
    """

    transition_columns = [
        f"{feature}_rel_change"
        for feature in STATE_FEATURES
    ]

    input_features = (
        STATE_FEATURES +
        transition_columns
    )

    # -----------------------------------------------------
    # Convert to NumPy.
    # -----------------------------------------------------

    feature_data = (
        df[input_features]
        .to_numpy(dtype=np.float32)
    )

    target_data = (
        df[STATE_FEATURES]
        .to_numpy(dtype=np.float32)
    )

    # -----------------------------------------------------
    # Final safety cleanup before sequence generation.
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

    target_attack_present = []
    target_attack_ratio = []
    target_attack_type = []
    target_timestamp = []

    # -----------------------------------------------------
    # Sliding temporal window.
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
        # Input history:
        #
        # S(t-9), S(t-8), ..., S(t)
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

        X.append(sequence)
        Y.append(target)

        # -------------------------------------------------
        # Ground-truth metadata for the target state.
        # -------------------------------------------------

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

        target_timestamp.append(
            df.loc[
                end_idx,
                "window_start"
            ]
        )

    # -----------------------------------------------------
    # Convert lists to NumPy arrays.
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
    # Final numerical validation.
    # -----------------------------------------------------

    if not np.isfinite(X).all():

        print(
            "Warning: invalid values found in X. "
            "Replacing invalid values."
        )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )

    if not np.isfinite(Y).all():

        print(
            "Warning: invalid values found in Y. "
            "Replacing invalid values."
        )

        Y = np.nan_to_num(
            Y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    # -----------------------------------------------------
    # Metadata.
    # -----------------------------------------------------

    metadata = pd.DataFrame({
        "timestamp": target_timestamp,
        "attack_present": target_attack_present,
        "attack_ratio": target_attack_ratio,
        "dominant_attack": target_attack_type,
    })

    # -----------------------------------------------------
    # Report.
    # -----------------------------------------------------

    print(
        f"\nEpisode: {episode_name}"
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
        "Input features:",
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
# SAVE ONE EPISODE
# =========================================================

def save_episode(
    filename: str,
    X: np.ndarray,
    Y: np.ndarray,
    metadata: pd.DataFrame
):

    stem = filename.replace(
        "_states.csv",
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
    print("TEMPORAL SEQUENCE PREPARATION")
    print("=" * 90)

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
    print("SEQUENCE PREPARATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()