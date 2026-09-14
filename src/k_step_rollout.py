"""
=============================================================
K-STEP FORWARD ROLLOUT — V4 WORLD MODEL EVALUATION
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Evaluation-only script.

This script:
    1. Loads the EXISTING trained V4 checkpoint.
    2. Reconstructs the EXACT V4 architecture used during training.
    3. Verifies the checkpoint loads with strict=True.
    4. Rolls the latent world model forward up to K=5 steps.
    5. Evaluates future attack prediction at t+1, t+2, t+3, t+5.
    6. Evaluates trajectory-event prediction for K=2,3,5.
    7. Evaluates early-warning performance per observed attack onset.
    8. Selects all thresholds using VALIDATION data only.
    9. Applies frozen thresholds to TEST data.
   10. Saves machine-readable results.

IMPORTANT:
    - No training.
    - No fine-tuning.
    - No model-weight modification.
    - No test-set threshold tuning.
    - No architecture modification.
    - No use of test labels for threshold optimization.

Model:
    models/latent_world_model_v4/latent_world_model_v4.pt

Dataset:
    data/processed/temporal_benchmark_v2/multistep_K5/
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================
# 0. CONFIGURATION
# =============================================================

PROJECT_ROOT = Path(".")

MODEL_DIR = PROJECT_ROOT / "models" / "latent_world_model_v4"
MODEL_PATH = MODEL_DIR / "latent_world_model_v4.pt"

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_benchmark_v2"
    / "multistep_K5"
)

OUTPUT_DIR = MODEL_DIR / "k_step_rollout"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prepared multistep data supports K=5.
K_MAX = 5

# Future horizons to evaluate individually.
FUTURE_HORIZONS = [1, 2, 3, 5]

# Trajectory-event horizons.
TRAJECTORY_HORIZONS = [2, 3, 5]

# Each network state window represents 10 seconds.
SECONDS_PER_STEP = 10

# Inference batch size.
BATCH_SIZE = 256

# Validation threshold search.
THRESHOLD_LO = 0.01
THRESHOLD_HI = 0.98
THRESHOLD_STEP = 0.01

# Expected dimensions from the V4 model/data pipeline.
EXPECTED_INPUT_SIZE = 52
EXPECTED_OUTPUT_SIZE = 26
EXPECTED_HIDDEN_SIZE = 128
EXPECTED_LATENT_SIZE = 32
EXPECTED_NUM_LAYERS = 2
EXPECTED_DROPOUT = 0.20


# =============================================================
# 1. EXACT V4 MODEL DEFINITION
# =============================================================

class LatentWorldModelV4(nn.Module):
    """
    EXACT V4 architecture used by the training script.

    Architecture:

        Input
          |
          v
        2-layer LSTM
          |
          v
        LayerNorm
          |
          v
        Linear(128 -> 32)
          |
          v
        latent z_t
          |
          +-----------------------------+
          |                             |
          v                             |
      transition MLP                    |
      32 -> 128 -> 32                  |
          |                             |
          v                             |
        candidate                       |
          |                             |
          +------ gated residual <------+
                     |
                     v
                  LayerNorm
                     |
                     v
                  z_next
                 /      \
                /        \
               v          v
           Decoder     Attack Head
          32->64->26   32->32->1
                           |
                         sigmoid
                           |
                     attack probability

    This architecture intentionally matches the V4 training code.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        latent_size,
        output_size,
        num_layers,
        dropout,
    ):
        super().__init__()

        # -----------------------------------------------------
        # Encoder
        # -----------------------------------------------------

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.encoder_norm = nn.LayerNorm(
            hidden_size
        )

        self.encoder_fc = nn.Linear(
            hidden_size,
            latent_size,
        )

        # -----------------------------------------------------
        # Gated residual transition
        # -----------------------------------------------------

        self.transition_mlp = nn.Sequential(
            nn.Linear(
                latent_size,
                latent_size * 4,
            ),
            nn.ReLU(),
            nn.Linear(
                latent_size * 4,
                latent_size,
            ),
        )

        self.gate_fc = nn.Linear(
            latent_size,
            latent_size,
        )

        self.transition_norm = nn.LayerNorm(
            latent_size
        )

        # -----------------------------------------------------
        # Decoder
        # -----------------------------------------------------

        self.decoder_fc1 = nn.Linear(
            latent_size,
            64,
        )

        self.decoder_activation = nn.ReLU()

        self.decoder_fc2 = nn.Linear(
            64,
            output_size,
        )

        # -----------------------------------------------------
        # Attack head
        # -----------------------------------------------------

        self.attack_head = nn.Sequential(
            nn.Linear(
                latent_size,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(
                32,
                1,
            ),
        )

    def encode(self, x):
        """
        X_t -> z_t
        """

        output, _ = self.encoder_lstm(x)

        last_hidden = output[:, -1, :]

        last_hidden = self.encoder_norm(
            last_hidden
        )

        return self.encoder_fc(
            last_hidden
        )

    def transition_step(self, latent_state):
        """
        Deterministic latent transition:

            candidate = transition_mlp(z)
            gate      = sigmoid(gate_fc(z))
            z_next    = gate*candidate
                        +(1-gate)*z
            z_next    = LayerNorm(z_next)
        """

        candidate = self.transition_mlp(
            latent_state
        )

        gate = torch.sigmoid(
            self.gate_fc(latent_state)
        )

        z_next = (
            gate * candidate
            + (1.0 - gate) * latent_state
        )

        return self.transition_norm(
            z_next
        )

    def decode(self, latent_state):
        """
        Latent state -> 26-dimensional network state.
        """

        hidden = self.decoder_fc1(
            latent_state
        )

        hidden = self.decoder_activation(
            hidden
        )

        return self.decoder_fc2(
            hidden
        )

    def predict_attack(self, latent_state):
        """
        Latent state -> raw attack logit.
        """

        logits = self.attack_head(
            latent_state
        )

        return logits.squeeze(-1)

    def attack_probability(self, latent_state):
        """
        Latent state -> attack probability.
        """

        logits = self.predict_attack(
            latent_state
        )

        return torch.sigmoid(logits)

    def k_step_rollout(self, x, k):
        """
        Full forward rollout.

        Returns:

            trajectory_z:
                [z_t, z_t+1, ..., z_t+k]

            trajectory_probs:
                [
                    P(A_t),
                    P(A_t+1),
                    ...,
                    P(A_t+k)
                ]

            trajectory_states:
                decoded states for each latent state.
        """

        z_current = self.encode(x)

        trajectory_z = [
            z_current
        ]

        trajectory_probs = [
            self.attack_probability(
                z_current
            )
        ]

        trajectory_states = [
            self.decode(z_current)
        ]

        for _ in range(k):

            z_current = (
                self.transition_step(
                    z_current
                )
            )

            probability = (
                self.attack_probability(
                    z_current
                )
            )

            state = self.decode(
                z_current
            )

            trajectory_z.append(
                z_current
            )

            trajectory_probs.append(
                probability
            )

            trajectory_states.append(
                state
            )

        return (
            trajectory_z,
            trajectory_probs,
            trajectory_states,
        )


# =============================================================
# 2. DEVICE
# =============================================================

def get_device():
    """
    Select inference device.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================
# 3. LOAD MODEL — STRICTLY
# =============================================================

def load_model(model_path, device):
    """
    Load the trained V4 model.

    STRICT requirement:
        checkpoint must match the architecture exactly.

    If even one state_dict key is missing or unexpected,
    execution stops instead of evaluating a partially loaded model.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    print("\n--- Loading V4 checkpoint ---")
    print(f"Path: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint is not a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    # Read architecture parameters stored by training.
    input_size = checkpoint.get(
        "input_size",
        EXPECTED_INPUT_SIZE,
    )

    hidden_size = checkpoint.get(
        "hidden_size",
        EXPECTED_HIDDEN_SIZE,
    )

    latent_size = checkpoint.get(
        "latent_size",
        EXPECTED_LATENT_SIZE,
    )

    output_size = checkpoint.get(
        "output_size",
        EXPECTED_OUTPUT_SIZE,
    )

    num_layers = checkpoint.get(
        "num_layers",
        EXPECTED_NUM_LAYERS,
    )

    # Validate architecture metadata.
    if input_size != EXPECTED_INPUT_SIZE:
        raise ValueError(
            f"Unexpected input_size={input_size}; "
            f"expected {EXPECTED_INPUT_SIZE}."
        )

    if hidden_size != EXPECTED_HIDDEN_SIZE:
        raise ValueError(
            f"Unexpected hidden_size={hidden_size}; "
            f"expected {EXPECTED_HIDDEN_SIZE}."
        )

    if latent_size != EXPECTED_LATENT_SIZE:
        raise ValueError(
            f"Unexpected latent_size={latent_size}; "
            f"expected {EXPECTED_LATENT_SIZE}."
        )

    if output_size != EXPECTED_OUTPUT_SIZE:
        raise ValueError(
            f"Unexpected output_size={output_size}; "
            f"expected {EXPECTED_OUTPUT_SIZE}."
        )

    if num_layers != EXPECTED_NUM_LAYERS:
        raise ValueError(
            f"Unexpected num_layers={num_layers}; "
            f"expected {EXPECTED_NUM_LAYERS}."
        )

    print("\nCheckpoint architecture:")
    print(f"  input_size  = {input_size}")
    print(f"  hidden_size = {hidden_size}")
    print(f"  latent_size = {latent_size}")
    print(f"  output_size = {output_size}")
    print(f"  num_layers  = {num_layers}")
    print(f"  dropout    = {EXPECTED_DROPOUT}")

    # Reconstruct EXACT V4 architecture.
    model = LatentWorldModelV4(
        input_size=input_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=EXPECTED_DROPOUT,
    )

    # IMPORTANT:
    # strict=True intentionally prevents partial loading.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)

    model.eval()

    print("\nCHECKPOINT LOAD: PASS")
    print("All model parameters matched exactly.")
    print("Missing keys   : 0")
    print("Unexpected keys: 0")

    return model, checkpoint


# =============================================================
# 4. LOAD MULTISTEP DATA
# =============================================================

def load_data():
    """
    Load validation/test features and future attack labels.
    """

    required_files = [
        "X_val.npy",
        "A_binary_multistep_val.npy",
        "X_test.npy",
        "A_binary_multistep_test.npy",
    ]

    print("\n--- Loading multistep dataset ---")

    for filename in required_files:

        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        print(f"Found: {filename}")

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    )

    A_bin_val = np.load(
        DATA_DIR / "A_binary_multistep_val.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    A_bin_test = np.load(
        DATA_DIR / "A_binary_multistep_test.npy"
    )

    return (
        X_val,
        A_bin_val,
        X_test,
        A_bin_test,
    )


# =============================================================
# 5. DATA SAFETY CHECKS
# =============================================================

def validate_input_data(
    X_val,
    A_bin_val,
    X_test,
    A_bin_test,
):
    """
    Validate data shapes before model inference.
    """

    print("\n--- Data validation ---")

    # Sample counts.
    assert len(X_val) == len(A_bin_val), (
        "Validation X/label count mismatch."
    )

    assert len(X_test) == len(A_bin_test), (
        "Test X/label count mismatch."
    )

    # Feature count.
    assert X_val.ndim == 3, (
        f"Expected X_val to have 3 dimensions, "
        f"got {X_val.shape}."
    )

    assert X_test.ndim == 3, (
        f"Expected X_test to have 3 dimensions, "
        f"got {X_test.shape}."
    )

    assert X_val.shape[2] == EXPECTED_INPUT_SIZE, (
        f"Expected {EXPECTED_INPUT_SIZE} input features, "
        f"got {X_val.shape[2]}."
    )

    assert X_test.shape[2] == EXPECTED_INPUT_SIZE, (
        f"Expected {EXPECTED_INPUT_SIZE} input features, "
        f"got {X_test.shape[2]}."
    )

    # Multistep labels.
    assert A_bin_val.ndim == 2, (
        "A_bin_val must be 2-dimensional."
    )

    assert A_bin_test.ndim == 2, (
        "A_bin_test must be 2-dimensional."
    )

    assert A_bin_val.shape[1] >= K_MAX, (
        "Validation data does not contain 5 future labels."
    )

    assert A_bin_test.shape[1] >= K_MAX, (
        "Test data does not contain 5 future labels."
    )

    # Finite values.
    assert np.all(
        np.isfinite(X_val)
    ), "X_val contains NaN/Inf."

    assert np.all(
        np.isfinite(X_test)
    ), "X_test contains NaN/Inf."

    assert np.all(
        np.isfinite(A_bin_val)
    ), "Validation labels contain NaN/Inf."

    assert np.all(
        np.isfinite(A_bin_test)
    ), "Test labels contain NaN/Inf."

    # Labels must be binary.
    unique_val = np.unique(
        A_bin_val[:, :K_MAX]
    )

    unique_test = np.unique(
        A_bin_test[:, :K_MAX]
    )

    assert np.all(
        np.isin(unique_val, [0, 1])
    ), (
        f"Validation labels are not binary: "
        f"{unique_val}"
    )

    assert np.all(
        np.isin(unique_test, [0, 1])
    ), (
        f"Test labels are not binary: "
        f"{unique_test}"
    )

    print(
        f"X_val       : {X_val.shape}"
    )

    print(
        f"A_bin_val   : {A_bin_val.shape}"
    )

    print(
        f"X_test      : {X_test.shape}"
    )

    print(
        f"A_bin_test  : {A_bin_test.shape}"
    )

    print("Data validation: PASS")


# =============================================================
# 6. ROLLOUT INFERENCE
# =============================================================

def get_rollout_probs(
    model,
    X_data,
    k_max,
    device,
    batch_size=BATCH_SIZE,
):
    """
    Generate attack probabilities for:

        step 0 -> current latent state z_t
        step 1 -> t+1
        step 2 -> t+2
        ...
        step k -> t+k

    Returns:
        numpy array with shape:
        (N, k_max + 1)
    """

    N = X_data.shape[0]

    all_probs = np.zeros(
        (N, k_max + 1),
        dtype=np.float32,
    )

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            N,
            batch_size,
        ):

            end = min(
                start + batch_size,
                N,
            )

            X_batch = torch.as_tensor(
                X_data[start:end],
                dtype=torch.float32,
                device=device,
            )

            (
                _trajectory_z,
                trajectory_probs,
                _trajectory_states,
            ) = model.k_step_rollout(
                X_batch,
                k_max,
            )

            for step_idx in range(
                k_max + 1
            ):

                probs = (
                    trajectory_probs[
                        step_idx
                    ]
                    .squeeze(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                all_probs[
                    start:end,
                    step_idx
                ] = probs

    assert np.all(
        np.isfinite(all_probs)
    ), (
        "Rollout probabilities contain "
        "NaN/Inf."
    )

    assert np.all(
        (all_probs >= 0.0)
        & (all_probs <= 1.0)
    ), (
        "Rollout probabilities outside [0,1]."
    )

    return all_probs


# =============================================================
# 7. THRESHOLD SELECTION — VALIDATION ONLY
# =============================================================

def find_best_threshold(
    probs,
    targets,
):
    """
    Find threshold maximizing F1.

    IMPORTANT:
        This function must ONLY be called with
        validation predictions and validation labels.

    Test labels must NEVER be used here.
    """

    probs = np.asarray(
        probs,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=int,
    )

    best_threshold = 0.5
    best_f1 = -1.0

    threshold = (
        THRESHOLD_LO
    )

    while threshold <= (
        THRESHOLD_HI + 1e-9
    ):

        predictions = (
            probs >= threshold
        ).astype(int)

        current_f1 = f1_score(
            targets,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:

            best_f1 = current_f1

            best_threshold = float(
                threshold
            )

        threshold += THRESHOLD_STEP

    return (
        best_threshold,
        best_f1,
    )


# =============================================================
# 8. METRICS
# =============================================================

def compute_metrics(
    probs,
    targets,
    threshold,
):
    """
    Compute test metrics using a frozen threshold.
    """

    probs = np.asarray(
        probs,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=int,
    )

    predictions = (
        probs >= threshold
    ).astype(int)

    f1 = f1_score(
        targets,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        targets,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        zero_division=0,
    )

    if len(np.unique(targets)) >= 2:

        roc_auc = roc_auc_score(
            targets,
            probs,
        )

    else:

        roc_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()

    denominator = fp + tn

    if denominator > 0:

        fpr = fp / denominator

    else:

        fpr = float("nan")

    return {
        "f1": float(f1),
        "precision": float(
            precision
        ),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "fpr": float(fpr),
        "pos_samples": int(
            targets.sum()
        ),
        "threshold": float(
            threshold
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# =============================================================
# 9. FUTURE ATTACK PREDICTION
# =============================================================

def evaluate_future_prediction(
    val_probs,
    A_bin_val,
    test_probs,
    A_bin_test,
):
    """
    Evaluate point predictions:

        prediction at t+k
            vs
        actual A(t+k)

    IMPORTANT INDEXING:

        val_probs[:, k]
            corresponds to
        A_bin_val[:, k-1]

    Same for test.
    """

    print("\n")
    print("=" * 90)
    print("1. FUTURE ATTACK PREDICTION")
    print("=" * 90)

    print(
        "Prediction t+k is evaluated against actual A(t+k)."
    )

    print()

    results = []

    header = (
        f"{'Horizon':<18}"
        f"{'F1':>8}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'ROC-AUC':>10}"
        f"{'FPR':>10}"
        f"{'PosSamples':>12}"
        f"{'Threshold':>12}"
    )

    print(header)
    print("-" * len(header))

    for k in FUTURE_HORIZONS:

        # ---------------------------------------------
        # Correct future target
        # ---------------------------------------------

        val_target = (
            A_bin_val[:, k - 1]
        )

        test_target = (
            A_bin_test[:, k - 1]
        )

        # ---------------------------------------------
        # Correct future prediction
        # ---------------------------------------------

        val_prediction = (
            val_probs[:, k]
        )

        test_prediction = (
            test_probs[:, k]
        )

        # ---------------------------------------------
        # Threshold from VALIDATION ONLY
        # ---------------------------------------------

        threshold, validation_f1 = (
            find_best_threshold(
                val_prediction,
                val_target,
            )
        )

        # ---------------------------------------------
        # Apply frozen threshold to TEST
        # ---------------------------------------------

        metrics = compute_metrics(
            test_prediction,
            test_target,
            threshold,
        )

        metrics.update(
            {
                "k": k,
                "horizon": (
                    "1-Step Future"
                    if k == 1
                    else f"Future (K={k})"
                ),
                "validation_f1_for_threshold": float(
                    validation_f1
                ),
            }
        )

        results.append(metrics)

        print(
            f"{metrics['horizon']:<18}"
            f"{metrics['f1']:>8.4f}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>10.4f}"
            f"{metrics['roc_auc']:>10.4f}"
            f"{metrics['fpr']:>10.4f}"
            f"{metrics['pos_samples']:>12d}"
            f"{metrics['threshold']:>12.2f}"
        )

    return results


# =============================================================
# 10. TRAJECTORY EVENT PREDICTION
# =============================================================

def evaluate_trajectory_events(
    val_probs,
    A_bin_val,
    test_probs,
    A_bin_test,
):
    """
    Horizon event prediction.

    Target:
        Did ANY attack occur within:
            t+1 ... t+k ?

    Target:
        max(A[t+1], ..., A[t+k])

    Strategies:
        1. MAX probability
        2. MEAN probability (heuristic)

    Thresholds are selected on validation only.
    """

    print("\n")
    print("=" * 90)
    print("2. TRAJECTORY EVENT PREDICTION")
    print("=" * 90)

    print(
        "Target: any attack during t+1 ... t+k"
    )

    print()

    results = []

    header = (
        f"{'Strategy':<20}"
        f"{'K':>4}"
        f"{'F1':>8}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'ROC-AUC':>10}"
        f"{'FPR':>10}"
        f"{'PosSamples':>12}"
        f"{'Threshold':>12}"
    )

    print(header)
    print("-" * len(header))

    for k in TRAJECTORY_HORIZONS:

        # ---------------------------------------------
        # Event target
        # ---------------------------------------------

        val_target = (
            A_bin_val[:, :k]
            .max(axis=1)
        )

        test_target = (
            A_bin_test[:, :k]
            .max(axis=1)
        )

        # ---------------------------------------------
        # MAX strategy
        # ---------------------------------------------

        val_score_max = (
            val_probs[:, 1:k + 1]
            .max(axis=1)
        )

        test_score_max = (
            test_probs[:, 1:k + 1]
            .max(axis=1)
        )

        threshold_max, validation_f1_max = (
            find_best_threshold(
                val_score_max,
                val_target,
            )
        )

        max_metrics = compute_metrics(
            test_score_max,
            test_target,
            threshold_max,
        )

        max_metrics.update(
            {
                "strategy": "Max",
                "k": k,
                "validation_f1_for_threshold": float(
                    validation_f1_max
                ),
            }
        )

        results.append(
            max_metrics
        )

        print(
            f"{'Max':<20}"
            f"{k:>4d}"
            f"{max_metrics['f1']:>8.4f}"
            f"{max_metrics['precision']:>12.4f}"
            f"{max_metrics['recall']:>10.4f}"
            f"{max_metrics['roc_auc']:>10.4f}"
            f"{max_metrics['fpr']:>10.4f}"
            f"{max_metrics['pos_samples']:>12d}"
            f"{max_metrics['threshold']:>12.2f}"
        )

        # ---------------------------------------------
        # MEAN heuristic
        # ---------------------------------------------

        val_score_mean = (
            val_probs[:, 1:k + 1]
            .mean(axis=1)
        )

        test_score_mean = (
            test_probs[:, 1:k + 1]
            .mean(axis=1)
        )

        threshold_mean, validation_f1_mean = (
            find_best_threshold(
                val_score_mean,
                val_target,
            )
        )

        mean_metrics = compute_metrics(
            test_score_mean,
            test_target,
            threshold_mean,
        )

        mean_metrics.update(
            {
                "strategy": "Mean (heuristic)",
                "k": k,
                "validation_f1_for_threshold": float(
                    validation_f1_mean
                ),
            }
        )

        results.append(
            mean_metrics
        )

        print(
            f"{'Mean (heuristic)':<20}"
            f"{k:>4d}"
            f"{mean_metrics['f1']:>8.4f}"
            f"{mean_metrics['precision']:>12.4f}"
            f"{mean_metrics['recall']:>10.4f}"
            f"{mean_metrics['roc_auc']:>10.4f}"
            f"{mean_metrics['fpr']:>10.4f}"
            f"{mean_metrics['pos_samples']:>12d}"
            f"{mean_metrics['threshold']:>12.2f}"
        )

    return results


# =============================================================
# 11. EARLY WARNING
# =============================================================

def evaluate_early_warning(
    val_probs,
    A_bin_val,
    test_probs,
    A_bin_test,
):
    """
    Early warning evaluation.

    We define an attack onset as:

        A[i] == 1
        and
        A[i-1] == 0

    The first test window may already be inside an ongoing
    attack. Such a case has no observable onset in the test
    interval and is excluded from onset-based lead-time
    calculation.

    Warning strategy:
        K=5 horizon MAX probability

    A warning at time t means:
        max P(A(t+1)...A(t+5))
        >= validation-selected threshold

    We then determine whether the warning happened before
    the observed attack onset.
    """

    print("\n")
    print("=" * 90)
    print("3. EARLY WARNING CAPABILITY")
    print("=" * 90)

    # ---------------------------------------------------------
    # Test attack sequence
    # ---------------------------------------------------------

    A0_test = (
        A_bin_test[:, 0]
        .astype(int)
    )

    N = len(A0_test)

    active_at_start = bool(
        N > 0
        and A0_test[0] == 1
    )

    # ---------------------------------------------------------
    # Detect attack onsets inside test interval
    # ---------------------------------------------------------

    onsets = []

    for i in range(1, N):

        if (
            A0_test[i] == 1
            and A0_test[i - 1] == 0
        ):

            onsets.append(i)

    # ---------------------------------------------------------
    # Report initial active attack
    # ---------------------------------------------------------

    print(
        f"Attack active at test start : "
        f"{active_at_start}"
    )

    if active_at_start:

        print(
            "NOTE: attack already active at first "
            "test window; its true onset occurred "
            "before the observable test interval."
        )

    print(
        f"Observed attack onsets        : "
        f"{len(onsets)}"
    )

    # ---------------------------------------------------------
    # Validation-selected warning threshold
    # ---------------------------------------------------------

    val_event_target = (
        A_bin_val[:, :K_MAX]
        .max(axis=1)
    )

    val_warning_score = (
        val_probs[:, 1:K_MAX + 1]
        .max(axis=1)
    )

    warning_threshold, validation_f1 = (
        find_best_threshold(
            val_warning_score,
            val_event_target,
        )
    )

    print(
        f"Warning threshold (validation): "
        f"{warning_threshold:.2f}"
    )

    print(
        f"Validation F1 used to choose "
        f"threshold: {validation_f1:.4f}"
    )

    # ---------------------------------------------------------
    # Test warning scores
    # ---------------------------------------------------------

    test_warning_score = (
        test_probs[:, 1:K_MAX + 1]
        .max(axis=1)
    )

    alerts = (
        test_warning_score
        >= warning_threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Per-episode lead time
    # ---------------------------------------------------------

    lead_times = []

    warned_count = 0

    for onset in onsets:

        # Search preceding K=5 windows.
        window_start = max(
            0,
            onset - K_MAX,
        )

        earliest_alert = None

        for i in range(
            window_start,
            onset,
        ):

            if alerts[i] == 1:

                earliest_alert = i

                break

        if earliest_alert is not None:

            warned_count += 1

            lead_time = (
                onset - earliest_alert
            ) * SECONDS_PER_STEP

            lead_times.append(
                lead_time
            )

        else:

            lead_times.append(0)

    # ---------------------------------------------------------
    # Warning statistics
    # ---------------------------------------------------------

    positive_lead_times = [
        lt
        for lt in lead_times
        if lt > 0
    ]

    if positive_lead_times:

        mean_lead = float(
            np.mean(
                positive_lead_times
            )
        )

        median_lead = float(
            np.median(
                positive_lead_times
            )
        )

    else:

        mean_lead = float("nan")
        median_lead = float("nan")

    if onsets:

        coverage_pct = (
            warned_count
            / len(onsets)
            * 100.0
        )

    else:

        coverage_pct = float("nan")

    # ---------------------------------------------------------
    # False-positive prediction windows/hour
    # ---------------------------------------------------------

    test_event_target = (
        A_bin_test[:, :K_MAX]
        .max(axis=1)
    )

    false_prediction_windows = int(
        (
            (alerts == 1)
            & (test_event_target == 0)
        ).sum()
    )

    total_hours = (
        len(test_probs)
        * SECONDS_PER_STEP
        / 3600.0
    )

    if total_hours > 0:

        fp_windows_per_hour = (
            false_prediction_windows
            / total_hours
        )

    else:

        fp_windows_per_hour = float("nan")

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print()
    print(
        f"Episodes warned early       : "
        f"{warned_count}"
    )

    if onsets:

        print(
            f"Warning coverage            : "
            f"{coverage_pct:.2f}%"
        )

    else:

        print(
            "Warning coverage            : n/a"
        )

    print(
        f"Mean lead time (positive)    : "
        f"{mean_lead:.2f} seconds"
    )

    print(
        f"Median lead time (positive)  : "
        f"{median_lead:.2f} seconds"
    )

    print(
        f"False-positive prediction "
        f"windows/hour                 : "
        f"{fp_windows_per_hour:.4f}"
    )

    return {
        "active_at_test_start": (
            active_at_start
        ),
        "observed_attack_onsets": (
            len(onsets)
        ),
        "episodes_warned_early": (
            warned_count
        ),
        "warning_coverage_pct": (
            coverage_pct
        ),
        "mean_lead_time_seconds": (
            mean_lead
        ),
        "median_lead_time_seconds": (
            median_lead
        ),
        "warning_threshold": (
            warning_threshold
        ),
        "validation_f1_for_warning_threshold": (
            validation_f1
        ),
        "false_positive_prediction_windows": (
            false_prediction_windows
        ),
        "false_positive_prediction_windows_per_hour": (
            fp_windows_per_hour
        ),
        "total_test_hours": total_hours,
        "lead_times_seconds": lead_times,
    }


# =============================================================
# 12. EVALUATION INTEGRITY
# =============================================================

def print_integrity_checks():
    """
    Print methodological guarantees.
    """

    print("\n")
    print("=" * 90)
    print("4. EVALUATION INTEGRITY")
    print("=" * 90)

    print(
        "✓ Validation-only threshold selection"
    )

    print(
        "✓ Test thresholds are frozen"
    )

    print(
        "✓ Future target alignment: "
        "prediction t+k ↔ A(t+k)"
    )

    print(
        "✓ No test-set threshold optimization"
    )

    print(
        "✓ No model training"
    )

    print(
        "✓ No model fine-tuning"
    )

    print(
        "✓ Exact V4 checkpoint loading"
    )

    print(
        "✓ Strict checkpoint compatibility"
    )

    print(
        "✓ K=1,2,3,5 future prediction evaluation"
    )

    print(
        "✓ K=2,3,5 horizon-event evaluation"
    )

    print(
        "✓ Per-onset early-warning evaluation"
    )

    print()
    print(
        "TEST LABELS ARE NOT USED FOR "
        "THRESHOLD SELECTION."
    )


# =============================================================
# 13. SAVE RESULTS
# =============================================================

def make_json_safe(value):
    """
    Convert numpy values into JSON-safe values.
    """

    if isinstance(
        value,
        np.integer,
    ):

        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(value):

            return None

        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):

        return [
            make_json_safe(v)
            for v in value.tolist()
        ]

    if isinstance(
        value,
        dict,
    ):

        return {
            key: make_json_safe(
                val
            )
            for key, val in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            make_json_safe(v)
            for v in value
        ]

    return value


def save_results(
    future_results,
    trajectory_results,
    early_warning_results,
):
    """
    Save CSV + JSON result files.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Future prediction.
    future_df = pd.DataFrame(
        future_results
    )

    future_df.to_csv(
        OUTPUT_DIR
        / "future_prediction_results.csv",
        index=False,
    )

    # Trajectory event prediction.
    trajectory_df = pd.DataFrame(
        trajectory_results
    )

    trajectory_df.to_csv(
        OUTPUT_DIR
        / "trajectory_event_results.csv",
        index=False,
    )

    # Early warning.
    early_warning_to_save = {
        key: value
        for key, value
        in early_warning_results.items()
        if key != "lead_times_seconds"
    }

    early_warning_df = pd.DataFrame(
        [early_warning_to_save]
    )

    early_warning_df.to_csv(
        OUTPUT_DIR
        / "early_warning_results.csv",
        index=False,
    )

    # Full summary JSON.
    summary = {
        "future_prediction": (
            future_results
        ),
        "trajectory_event": (
            trajectory_results
        ),
        "early_warning": (
            early_warning_results
        ),
        "evaluation_integrity": {
            "validation_only_thresholds": True,
            "test_threshold_tuning": False,
            "temporal_target_alignment": True,
            "model_training": False,
            "model_finetuning": False,
            "strict_checkpoint_loading": True,
        },
    }

    summary = make_json_safe(
        summary
    )

    with open(
        OUTPUT_DIR
        / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print(
        f"Results saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"  - future_prediction_results.csv"
    )

    print(
        f"  - trajectory_event_results.csv"
    )

    print(
        f"  - early_warning_results.csv"
    )

    print(
        f"  - evaluation_summary.json"
    )


# =============================================================
# 14. MAIN
# =============================================================

def main():

    print()
    print("=" * 90)
    print(
        "K-STEP FORWARD ROLLOUT — "
        "V4 WORLD MODEL EVALUATION"
    )
    print("=" * 90)

    print()
    print(
        "SIH 2026 — Predictive Cyber Defence "
        "Using World Models"
    )

    print()
    print(
        "IMPORTANT: evaluation only."
    )

    print(
        "No training / no fine-tuning / "
        "no test threshold tuning."
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = get_device()

    print()
    print(
        f"Device: {device}"
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    (
        X_val,
        A_bin_val,
        X_test,
        A_bin_test,
    ) = load_data()

    # ---------------------------------------------------------
    # Validate input data
    # ---------------------------------------------------------

    validate_input_data(
        X_val,
        A_bin_val,
        X_test,
        A_bin_test,
    )

    # ---------------------------------------------------------
    # Load exact model
    # ---------------------------------------------------------

    model, checkpoint = load_model(
        MODEL_PATH,
        device,
    )

    # ---------------------------------------------------------
    # Print checkpoint metadata
    # ---------------------------------------------------------

    print()
    print("--- Checkpoint metadata ---")

    if "selected_threshold" in checkpoint:

        print(
            "Saved training threshold : "
            f"{checkpoint['selected_threshold']:.4f}"
        )

        print(
            "NOTE: saved threshold is NOT "
            "used for this evaluation."
        )

    if "best_validation_f1" in checkpoint:

        print(
            "Saved validation F1       : "
            f"{checkpoint['best_validation_f1']:.4f}"
        )

    # ---------------------------------------------------------
    # Rollout validation data
    # ---------------------------------------------------------

    print()
    print(
        "Running validation rollout..."
    )

    val_probs = get_rollout_probs(
        model=model,
        X_data=X_val,
        k_max=K_MAX,
        device=device,
    )

    print(
        f"Validation rollout shape: "
        f"{val_probs.shape}"
    )

    # ---------------------------------------------------------
    # Rollout test data
    # ---------------------------------------------------------

    print()
    print(
        "Running test rollout..."
    )

    test_probs = get_rollout_probs(
        model=model,
        X_data=X_test,
        k_max=K_MAX,
        device=device,
    )

    print(
        f"Test rollout shape: "
        f"{test_probs.shape}"
    )

    # ---------------------------------------------------------
    # Final prediction safety checks
    # ---------------------------------------------------------

    assert val_probs.shape == (
        len(X_val),
        K_MAX + 1,
    ), (
        "Unexpected validation "
        "rollout shape."
    )

    assert test_probs.shape == (
        len(X_test),
        K_MAX + 1,
    ), (
        "Unexpected test "
        "rollout shape."
    )

    assert np.all(
        np.isfinite(val_probs)
    ), (
        "Validation rollout contains "
        "NaN/Inf."
    )

    assert np.all(
        np.isfinite(test_probs)
    ), (
        "Test rollout contains "
        "NaN/Inf."
    )

    # ---------------------------------------------------------
    # Future prediction
    # ---------------------------------------------------------

    future_results = (
        evaluate_future_prediction(
            val_probs,
            A_bin_val,
            test_probs,
            A_bin_test,
        )
    )

    # ---------------------------------------------------------
    # Trajectory event
    # ---------------------------------------------------------

    trajectory_results = (
        evaluate_trajectory_events(
            val_probs,
            A_bin_val,
            test_probs,
            A_bin_test,
        )
    )

    # ---------------------------------------------------------
    # Early warning
    # ---------------------------------------------------------

    early_warning_results = (
        evaluate_early_warning(
            val_probs,
            A_bin_val,
            test_probs,
            A_bin_test,
        )
    )

    # ---------------------------------------------------------
    # Integrity
    # ---------------------------------------------------------

    print_integrity_checks()

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_results(
        future_results,
        trajectory_results,
        early_warning_results,
    )

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "EVALUATION COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        "No model weights were changed."
    )

    print(
        "No training was performed."
    )

    print(
        "No threshold was optimized "
        "using test labels."
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()