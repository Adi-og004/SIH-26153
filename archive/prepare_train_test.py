import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================

SEQUENCE_DIR = Path("data/processed/sequences")
OUTPUT_DIR = Path("data/processed/final")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_EPISODES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter",
]

VALIDATION_EPISODES = [
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter",
]

TEST_EPISODES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter",
]


# =========================================================
# LOAD EPISODE
# =========================================================

def load_episode(stem):
    """
    Load X, Y and metadata for one temporal episode.
    """

    X_path = SEQUENCE_DIR / f"{stem}_X.npy"
    Y_path = SEQUENCE_DIR / f"{stem}_Y.npy"
    metadata_path = SEQUENCE_DIR / f"{stem}_metadata.csv"

    X = np.load(X_path)
    Y = np.load(Y_path)

    metadata = __import__("pandas").read_csv(
        metadata_path
    )

    return X, Y, metadata


# =========================================================
# LOAD MULTIPLE EPISODES
# =========================================================

def load_split(episodes):
    """
    Concatenate several independent episodes.
    """

    X_parts = []
    Y_parts = []
    metadata_parts = []

    for stem in episodes:

        print(f"Loading: {stem}")

        X, Y, metadata = load_episode(stem)

        X_parts.append(X)
        Y_parts.append(Y)
        metadata_parts.append(metadata)

        print(
            f"  X shape: {X.shape}"
        )

        print(
            f"  Y shape: {Y.shape}"
        )

    X = np.concatenate(
        X_parts,
        axis=0
    )

    Y = np.concatenate(
        Y_parts,
        axis=0
    )

    metadata = __import__("pandas").concat(
        metadata_parts,
        ignore_index=True
    )

    return X, Y, metadata


# =========================================================
# FIT SCALERS ONLY ON TRAINING DATA
# =========================================================

def fit_scalers(X_train, Y_train):

    print("\nFitting scalers using TRAINING data only...")

    # X shape:
    # (samples, sequence_length, features)

    n_samples, sequence_length, n_features = X_train.shape

    # Flatten temporal dimension temporarily
    X_train_flat = X_train.reshape(
        -1,
        n_features
    )

    input_scaler = StandardScaler()

    input_scaler.fit(
        X_train_flat
    )

    # -----------------------------------------------------
    # Target scaler
    # -----------------------------------------------------

    n_target_features = Y_train.shape[1]

    target_scaler = StandardScaler()

    target_scaler.fit(
        Y_train.reshape(
            -1,
            n_target_features
        )
    )

    return input_scaler, target_scaler


# =========================================================
# APPLY SCALER
# =========================================================

def scale_X(X, scaler):

    samples, sequence_length, features = X.shape

    X_flat = X.reshape(
        -1,
        features
    )

    X_scaled = scaler.transform(
        X_flat
    )

    return X_scaled.reshape(
        samples,
        sequence_length,
        features
    ).astype(np.float32)


def scale_Y(Y, scaler):

    Y_scaled = scaler.transform(
        Y
    )

    return Y_scaled.astype(
        np.float32
    )


# =========================================================
# SAVE ARRAY
# =========================================================

def save_array(name, array):

    path = OUTPUT_DIR / f"{name}.npy"

    np.save(
        path,
        array
    )

    print(
        f"Saved {name}: {array.shape}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 90)
    print("LEAKAGE-FREE TRAIN / VALIDATION / TEST PREPARATION")
    print("=" * 90)

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print("\n========== TRAIN DATA ==========")

    X_train_raw, Y_train_raw, train_metadata = load_split(
        TRAIN_EPISODES
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    print("\n========== VALIDATION DATA ==========")

    X_val_raw, Y_val_raw, val_metadata = load_split(
        VALIDATION_EPISODES
    )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    print("\n========== TEST DATA ==========")

    X_test_raw, Y_test_raw, test_metadata = load_split(
        TEST_EPISODES
    )

    # -----------------------------------------------------
    # Print raw shapes
    # -----------------------------------------------------

    print("\n========== RAW SHAPES ==========")

    print("X_train:", X_train_raw.shape)
    print("Y_train:", Y_train_raw.shape)

    print("X_val:", X_val_raw.shape)
    print("Y_val:", Y_val_raw.shape)

    print("X_test:", X_test_raw.shape)
    print("Y_test:", Y_test_raw.shape)

    # -----------------------------------------------------
    # FIT SCALERS ONLY ON TRAINING DATA
    # -----------------------------------------------------

    input_scaler, target_scaler = fit_scalers(
        X_train_raw,
        Y_train_raw
    )

    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    X_train = scale_X(
        X_train_raw,
        input_scaler
    )

    X_val = scale_X(
        X_val_raw,
        input_scaler
    )

    X_test = scale_X(
        X_test_raw,
        input_scaler
    )

    Y_train = scale_Y(
        Y_train_raw,
        target_scaler
    )

    Y_val = scale_Y(
        Y_val_raw,
        target_scaler
    )

    Y_test = scale_Y(
        Y_test_raw,
        target_scaler
    )

    # -----------------------------------------------------
    # SAVE ARRAYS
    # -----------------------------------------------------

    print("\n========== SAVING FINAL DATA ==========")

    save_array(
        "X_train",
        X_train
    )

    save_array(
        "Y_train",
        Y_train
    )

    save_array(
        "X_val",
        X_val
    )

    save_array(
        "Y_val",
        Y_val
    )

    save_array(
        "X_test",
        X_test
    )

    save_array(
        "Y_test",
        Y_test
    )

    # -----------------------------------------------------
    # SAVE METADATA
    # -----------------------------------------------------

    train_metadata.to_csv(
        OUTPUT_DIR / "train_metadata.csv",
        index=False
    )

    val_metadata.to_csv(
        OUTPUT_DIR / "val_metadata.csv",
        index=False
    )

    test_metadata.to_csv(
        OUTPUT_DIR / "test_metadata.csv",
        index=False
    )

    # -----------------------------------------------------
    # SAVE SCALERS
    # -----------------------------------------------------

    with open(
        OUTPUT_DIR / "input_scaler.pkl",
        "wb"
    ) as f:

        pickle.dump(
            input_scaler,
            f
        )

    with open(
        OUTPUT_DIR / "target_scaler.pkl",
        "wb"
    ) as f:

        pickle.dump(
            target_scaler,
            f
        )

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    print("\n========== FINAL DATASET ==========")

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
        "Input features:",
        X_train.shape[2]
    )

    print(
        "Target features:",
        Y_train.shape[1]
    )

    print("\nSaved everything under:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 90)
    print("TRAIN / VALIDATION / TEST PREPARATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()