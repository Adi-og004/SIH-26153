"""
=============================================================
LATENT WORLD MODEL V4 — MULTI-STEP OBJECTIVE
=============================================================

FORMAL STATE DEFINITION:

  Network State S_t:
    The aggregate statistical summary of all network flows
    observed in a 10-second window centered at time t.
    Composed of 26 base features (flow_count, syn_rate,
    mean_pkt_len, dst_port_entropy, etc.)

  X_t (Input, shape: (10, 52)):
    A sequence of 10 consecutive state vectors:
    [S_{t-9}, S_{t-8}, ..., S_t]
    Each S_t has 26 base features + 26 relative-change
    features = 52 total.

  Z_t (Latent State, 32-dim):
    A compressed representation of X_t learned by the
    LSTM encoder: Z_t = Encoder(X_t)

  Y_{t+1} (Decoded Output, 26-dim):
    The predicted base features of S_{t+1}:
    Y_{t+1} = Decoder(Transition(Z_t))

  Transition Function T (DETERMINISTIC):
    A learned deterministic transition function that
    predicts the expected next latent state:
    Z_{t+1} = T(Z_t) = gate * MLP(Z_t) + (1 - gate) * Z_t
    This is NOT a probability distribution P(S_{t+1}|S_t).
    It produces a single-point estimate.


CHANGES FROM V3 (each driven by V3 diagnosis):

1. MULTI-STEP TRAINING OBJECTIVE:
   V3: Loss at Z_{t+1} only (single step).
   V4: Loss accumulated over K steps:
       L = sum_{k=1}^{K} gamma^{k-1} * L_k
       where gamma=0.8 (exponential decay).

   WHY: V3's transition diverges at K>3 because it
   was never trained on its own outputs. V4 trains
   the transition to remain stable under K=5 chained
   applications.

2. GATED RESIDUAL TRANSITION:
   V3: Z' = Z + MLP(Z)          (simple residual)
   V4: Z' = gate*MLP(Z) + (1-gate)*Z  (gated)
       where gate = sigmoid(Linear(Z))

   WHY: The gate learns HOW MUCH to change the state
   at each step. During long rollouts, the gate can
   close (near 0) to preserve the current state,
   preventing divergence.

3. LATENT LAYER NORMALIZATION:
   V3: No normalization on latent transitions.
   V4: LayerNorm applied after each transition step.

   WHY: Prevents latent norm explosion/collapse during
   multi-step rollouts by keeping the latent vector
   in a consistent scale.

4. CURRICULUM TRAINING:
   V3: Train with K=1 always.
   V4: Phase 1 (epochs 1-20): K=1
       Phase 2 (epochs 21-40): K=2
       Phase 3 (epochs 41-60): K=3
       Phase 4 (epochs 61-80): K=4
       Phase 5 (epochs 81-120): K=5

   WHY: Gradually increasing K lets the model first
   learn good single-step dynamics, then progressively
   extend its stable horizon.

EVERYTHING ELSE IS IDENTICAL TO V3:
- Same data (V2 temporal benchmark, 28-Feb)
- Same encoder (2-layer LSTM, 128 hidden)
- Same latent size (32)
- Same decoder architecture
- Same attack head architecture
- Same loss functions (SmoothL1 + weighted BCE)
- Same evaluation methodology


FEATURE AUDIT (52 input features):

  COVERED:
  - TCP flags: syn_rate, ack_rate, rst_rate, fin_rate,
    psh_rate, urg_rate
  - Packet size: mean_pkt_len, std_pkt_len
  - Flow/packet counts: flow_count, log_total_packets,
    log_total_bytes
  - Timing/IAT: log_mean_fwd_iat, log_mean_bwd_iat,
    log_mean_flow_duration
  - Port scanning: log_unique_dst_ports, dst_port_entropy
  - Direction: fwd_packet_ratio, bwd_packet_ratio,
    fwd_byte_ratio, bwd_byte_ratio
  - Temporal change: all 26 *_rel_change features

  NOT COVERED (CICFlowMeter limitation):
  - TTL: Not in CICFlowMeter flow-level output
  - IP Fragmentation: Not in CICFlowMeter output
  - TCP Window Size: Available in raw data (Init Fwd/Bwd
    Win Byts) but not currently extracted. Adding would
    require full pipeline re-processing (future work).
  - Retransmission: Partially captured by rst_rate.
    Direct retransmission counts require packet-level
    parsing beyond CICFlowMeter (future work).

=============================================================
"""

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression


# =============================================================
# CONFIGURATION
# =============================================================

# Multi-step data directory
MULTISTEP_DIR = Path(
    "data/processed/temporal_benchmark_v2/multistep_K5"
)

# Original data directory (for single-step Y targets
# and for the logistic regression baseline)
ORIGINAL_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

MODEL_DIR = Path(
    "models/latent_world_model_v4"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_SEED = 42

BATCH_SIZE = 64

INPUT_SIZE = 52
OUTPUT_SIZE = 26

HIDDEN_SIZE = 128
LATENT_SIZE = 32

NUM_LAYERS = 2
DROPOUT = 0.20

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5

EPOCHS = 120
PATIENCE = 20

STATE_LOSS_WEIGHT = 1.0
ATTACK_LOSS_WEIGHT = 0.5

# Multi-step config
MAX_K = 5
GAMMA = 0.8  # Exponential decay for future step losses

# Curriculum schedule: (start_epoch, K_value)
CURRICULUM = [
    (1, 1),
    (21, 2),
    (41, 3),
    (61, 4),
    (81, 5),
]


# =============================================================
# REPRODUCIBILITY
# =============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =============================================================
# DEVICE
# =============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("=" * 90)
print("LATENT WORLD MODEL V4 — MULTI-STEP OBJECTIVE")
print("=" * 90)
print(f"Using device: {DEVICE}")

print("\n--- Changes from V3 ---")
print("1. Multi-step loss: L = sum gamma^(k-1) * L_k")
print("2. Gated residual: Z' = gate*MLP(Z) + (1-gate)*Z")
print("3. Latent LayerNorm after transition")
print("4. Curriculum: K=1→2→3→4→5")
print(f"5. gamma={GAMMA}, max_K={MAX_K}")
print("-" * 40)


# =============================================================
# LOAD MULTI-STEP DATA
# =============================================================

print("\nLoading multi-step data...")

X_train = np.load(MULTISTEP_DIR / "X_train.npy")
Y_ms_train = np.load(MULTISTEP_DIR / "Y_multistep_train.npy")
A_soft_train = np.load(MULTISTEP_DIR / "A_soft_multistep_train.npy")
A_binary_train = np.load(MULTISTEP_DIR / "A_binary_multistep_train.npy")

X_val = np.load(MULTISTEP_DIR / "X_val.npy")
Y_ms_val = np.load(MULTISTEP_DIR / "Y_multistep_val.npy")
A_soft_val = np.load(MULTISTEP_DIR / "A_soft_multistep_val.npy")
A_binary_val = np.load(MULTISTEP_DIR / "A_binary_multistep_val.npy")

X_test = np.load(MULTISTEP_DIR / "X_test.npy")
Y_ms_test = np.load(MULTISTEP_DIR / "Y_multistep_test.npy")
A_soft_test = np.load(MULTISTEP_DIR / "A_soft_multistep_test.npy")
A_binary_test = np.load(MULTISTEP_DIR / "A_binary_multistep_test.npy")

train_meta = pd.read_csv(MULTISTEP_DIR / "train_metadata.csv")
val_meta = pd.read_csv(MULTISTEP_DIR / "val_metadata.csv")
test_meta = pd.read_csv(MULTISTEP_DIR / "test_metadata.csv")

print(f"\nX_train: {X_train.shape}")
print(f"Y_ms_train: {Y_ms_train.shape}  (K={MAX_K} future steps)")
print(f"A_soft_train: {A_soft_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")


# =============================================================
# NUMERICAL VALIDATION
# =============================================================

for name, array in [
    ("X_train", X_train),
    ("Y_ms_train", Y_ms_train),
    ("A_soft_train", A_soft_train),
    ("X_val", X_val),
    ("Y_ms_val", Y_ms_val),
    ("A_soft_val", A_soft_val),
    ("X_test", X_test),
    ("Y_ms_test", Y_ms_test),
    ("A_soft_test", A_soft_test),
]:
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


# =============================================================
# ATTACK DISTRIBUTION
# =============================================================

print("\n========== ATTACK DISTRIBUTION ==========")

for name, a_soft, a_binary in [
    ("TRAIN", A_soft_train, A_binary_train),
    ("VAL", A_soft_val, A_binary_val),
    ("TEST", A_soft_test, A_binary_test),
]:
    # Step 1 labels
    benign = int((a_binary[:, 0] == 0).sum())
    attack = int((a_binary[:, 0] == 1).sum())
    mean_ratio = (
        a_soft[:, 0][a_binary[:, 0] == 1].mean()
        if attack > 0 else 0
    )
    print(
        f"{name} (step 1): Benign={benign}, "
        f"Attack={attack}, "
        f"Mean attack_ratio={mean_ratio:.3f}"
    )


# =============================================================
# POSITIVE CLASS WEIGHT
# =============================================================

train_benign = int(
    (A_binary_train[:, 0] == 0).sum()
)
train_attack = int(
    (A_binary_train[:, 0] == 1).sum()
)

if train_attack == 0:
    raise ValueError(
        "Training set has no attack samples."
    )

positive_weight_value = (
    train_benign / train_attack
)

print(f"\nPositive class weight: "
      f"{positive_weight_value:.4f}")


# =============================================================
# PYTORCH TENSORS
# =============================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_ms_train_t = torch.tensor(Y_ms_train, dtype=torch.float32)
A_soft_train_t = torch.tensor(A_soft_train, dtype=torch.float32)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
Y_ms_val_t = torch.tensor(Y_ms_val, dtype=torch.float32)
A_soft_val_t = torch.tensor(A_soft_val, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_ms_test_t = torch.tensor(Y_ms_test, dtype=torch.float32)
A_soft_test_t = torch.tensor(A_soft_test, dtype=torch.float32)


# =============================================================
# DATASETS AND LOADERS
# =============================================================

train_dataset = TensorDataset(
    X_train_t, Y_ms_train_t, A_soft_train_t
)
val_dataset = TensorDataset(
    X_val_t, Y_ms_val_t, A_soft_val_t
)
test_dataset = TensorDataset(
    X_test_t, Y_ms_test_t, A_soft_test_t
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =============================================================
# MODEL — V4: GATED RESIDUAL + LATENT LAYERNORM
# =============================================================

class LatentWorldModelV4(nn.Module):
    """
    Latent World Model V4 — Multi-Step Objective.

    Architecture:
      Encoder: 2-layer LSTM → LayerNorm → Linear(128→32)
      Transition: GATED residual with LayerNorm
        candidate = MLP(Z)        [32 → 128 → 32]
        gate = sigmoid(Linear(Z)) [32 → 32]
        Z' = LayerNorm(gate * candidate + (1 - gate) * Z)
      Decoder: Linear(32→64) → ReLU → Linear(64→26)
      Attack Head: Linear(32→32) → ReLU → Dropout → Linear(32→1)

    The transition function is a LEARNED DETERMINISTIC
    function. It predicts the expected next latent state,
    NOT a probability distribution.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        latent_size,
        output_size,
        num_layers,
        dropout
    ):
        super().__init__()

        # ----- Encoder (SAME as V3) -----

        self.encoder_lstm = nn.LSTM(
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

        self.encoder_norm = nn.LayerNorm(
            hidden_size
        )

        self.encoder_fc = nn.Linear(
            hidden_size,
            latent_size
        )

        # ----- Transition (CHANGED: Gated Residual) -----

        # Candidate transition MLP (same width as V3)
        self.transition_mlp = nn.Sequential(
            nn.Linear(
                latent_size,
                latent_size * 4   # 32*4=128
            ),
            nn.ReLU(),
            nn.Linear(
                latent_size * 4,
                latent_size
            )
        )

        # Gate: learns how much to change the state
        self.gate_fc = nn.Linear(
            latent_size,
            latent_size
        )

        # LayerNorm after transition (NEW in V4)
        self.transition_norm = nn.LayerNorm(
            latent_size
        )

        # ----- Decoder (SAME as V3) -----

        self.decoder_fc1 = nn.Linear(
            latent_size, 64
        )
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(
            64, output_size
        )

        # ----- Attack head (SAME as V3) -----

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )


    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(
            last_hidden
        )
        return self.encoder_fc(last_hidden)


    def transition_step(self, latent_state):
        """
        Gated residual transition (DETERMINISTIC).

        Z_next = LayerNorm(gate * candidate + (1 - gate) * Z)

        The gate learns to control how much the latent
        state changes at each step. During long rollouts,
        the gate can close (→0) to preserve the current
        state, preventing divergence.
        """
        candidate = self.transition_mlp(
            latent_state
        )
        gate = torch.sigmoid(
            self.gate_fc(latent_state)
        )
        z_next = (
            gate * candidate
            + (1 - gate) * latent_state
        )
        return self.transition_norm(z_next)


    def decode(self, latent_state):
        hidden = self.decoder_fc1(
            latent_state
        )
        hidden = self.decoder_activation(
            hidden
        )
        return self.decoder_fc2(hidden)


    def predict_attack(self, latent_state):
        logits = self.attack_head(
            latent_state
        )
        return logits.squeeze(-1)


    def forward(self, x):
        """Single-step forward for compatibility."""
        latent_state = self.encode(x)

        predicted_latent = (
            self.transition_step(latent_state)
        )

        predicted_state = self.decode(
            predicted_latent
        )

        attack_logits = self.predict_attack(
            predicted_latent
        )

        return (
            predicted_state,
            predicted_latent,
            attack_logits,
            latent_state
        )


    def attack_probability(self, x):
        """
        End-to-end: input window -> attack probability.
        Used by Captum explainer (backward-compatible
        with V3's FeatureExplainer).
        """
        z = self.encode(x)
        z_next = self.transition_step(z)
        logit = self.predict_attack(z_next)
        return torch.sigmoid(logit).unsqueeze(-1)


    def multi_step_forward(self, x, k_steps):
        """
        Multi-step forward rollout for training.

        Returns lists of length k_steps:
          - predicted_states: decoded network state at each step
          - attack_logits: raw logits at each step
        """
        z = self.encode(x)

        predicted_states = []
        attack_logits_list = []

        for step in range(k_steps):
            z = self.transition_step(z)
            state = self.decode(z)
            logits = self.predict_attack(z)

            predicted_states.append(state)
            attack_logits_list.append(logits)

        return predicted_states, attack_logits_list


    def k_step_rollout(self, x, k):
        """
        Full K-step rollout for evaluation.

        Returns trajectory of K+1 states (including Z_0).
        Compatible with the rollout analysis scripts.
        """
        z_current = self.encode(x)

        trajectory_z = [z_current]
        trajectory_probs = [
            torch.sigmoid(
                self.predict_attack(z_current)
            )
        ]
        trajectory_states = [
            self.decode(z_current)
        ]

        for step in range(k):
            z_current = self.transition_step(
                z_current
            )
            prob = torch.sigmoid(
                self.predict_attack(z_current)
            )
            state = self.decode(z_current)

            trajectory_z.append(z_current)
            trajectory_probs.append(prob)
            trajectory_states.append(state)

        return (
            trajectory_z,
            trajectory_probs,
            trajectory_states,
        )


# =============================================================
# INITIALIZE MODEL
# =============================================================

model = LatentWorldModelV4(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    latent_size=LATENT_SIZE,
    output_size=OUTPUT_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model = model.to(DEVICE)

print("\n========== MODEL ==========")
print(model)

param_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
print(f"\nTrainable parameters: {param_count}")


# =============================================================
# LOSS FUNCTIONS
# =============================================================

state_criterion = nn.SmoothL1Loss()

positive_weight = torch.tensor(
    [positive_weight_value],
    dtype=torch.float32,
    device=DEVICE
)

attack_criterion = nn.BCEWithLogitsLoss(
    pos_weight=positive_weight.squeeze(0)
)


# =============================================================
# OPTIMIZER AND SCHEDULER
# =============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# =============================================================
# CURRICULUM HELPER
# =============================================================

def get_current_k(epoch):
    """
    Returns the number of rollout steps for the
    current epoch based on the curriculum schedule.
    """
    current_k = 1
    for start_epoch, k_value in CURRICULUM:
        if epoch >= start_epoch:
            current_k = k_value
    return current_k


# =============================================================
# TRAIN ONE EPOCH (MULTI-STEP)
# =============================================================

def train_one_epoch(current_k):

    model.train()

    total_loss = 0.0
    total_state_loss = 0.0
    total_attack_loss = 0.0
    sample_count = 0

    for X_batch, Y_ms_batch, A_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        Y_ms_batch = Y_ms_batch.to(DEVICE)
        A_batch = A_batch.to(DEVICE)

        optimizer.zero_grad()

        # Multi-step forward
        predicted_states, attack_logits_list = (
            model.multi_step_forward(
                X_batch, current_k
            )
        )

        # Accumulate loss over K steps with decay
        batch_state_loss = 0.0
        batch_attack_loss = 0.0

        for k in range(current_k):
            weight = GAMMA ** k

            step_state_loss = state_criterion(
                predicted_states[k],
                Y_ms_batch[:, k, :].contiguous()
            )

            step_attack_loss = attack_criterion(
                attack_logits_list[k],
                A_batch[:, k].contiguous()
            )

            batch_state_loss += (
                weight * step_state_loss
            )
            batch_attack_loss += (
                weight * step_attack_loss
            )

        loss = (
            STATE_LOSS_WEIGHT * batch_state_loss
            + ATTACK_LOSS_WEIGHT * batch_attack_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = X_batch.size(0)
        total_loss += (
            loss.item() * batch_size
        )
        total_state_loss += (
            batch_state_loss.item() * batch_size
        )
        total_attack_loss += (
            batch_attack_loss.item() * batch_size
        )
        sample_count += batch_size

    return (
        total_loss / sample_count,
        total_state_loss / sample_count,
        total_attack_loss / sample_count
    )


# =============================================================
# VALIDATION (uses step-1 for checkpointing, full K for info)
# =============================================================

def collect_validation_outputs():

    model.eval()

    total_state_loss = 0.0
    total_attack_loss = 0.0
    sample_count = 0

    attack_probabilities = []
    attack_targets = []

    with torch.no_grad():

        for X_batch, Y_ms_batch, A_batch in val_loader:

            X_batch = X_batch.to(DEVICE)
            Y_ms_batch = Y_ms_batch.to(DEVICE)
            A_batch = A_batch.to(DEVICE)

            # Single-step evaluation for checkpointing
            (
                predicted_state,
                predicted_latent,
                attack_logits,
                latent_state
            ) = model(X_batch)

            state_loss = state_criterion(
                predicted_state,
                Y_ms_batch[:, 0, :].contiguous()  # Step-1 target
            )

            attack_loss = attack_criterion(
                attack_logits,
                A_batch[:, 0].contiguous()  # Step-1 target
            )

            batch_size = X_batch.size(0)
            total_state_loss += (
                state_loss.item() * batch_size
            )
            total_attack_loss += (
                attack_loss.item() * batch_size
            )

            probabilities = torch.sigmoid(
                attack_logits
            )

            attack_probabilities.extend(
                probabilities.cpu().numpy()
            )

            attack_targets.extend(
                A_batch[:, 0].cpu().numpy()
            )

            sample_count += batch_size

    return (
        total_state_loss / sample_count,
        total_attack_loss / sample_count,
        np.asarray(attack_probabilities),
        np.asarray(attack_targets)
    )


# =============================================================
# THRESHOLD SEARCH
# =============================================================

def find_best_threshold(
    probabilities,
    binary_targets
):
    best_threshold = 0.50
    best_f1 = -1.0

    threshold_results = []

    for threshold in np.arange(0.05, 0.96, 0.01):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            binary_targets, predictions,
            zero_division=0
        )

        recall_val = recall_score(
            binary_targets, predictions,
            zero_division=0
        )

        f1 = f1_score(
            binary_targets, predictions,
            zero_division=0
        )

        threshold_results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall_val,
            "f1": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    threshold_df = pd.DataFrame(
        threshold_results
    )

    return (
        best_threshold,
        best_f1,
        threshold_df
    )


# =============================================================
# TRAINING LOOP
# =============================================================

print("\n========== TRAINING ==========")

best_validation_f1 = -1.0
best_state = None
best_threshold = 0.50
epochs_without_improvement = 0

history = []

# Binary labels for threshold search
y_val_binary = (
    val_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)

y_test_binary = (
    test_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)


for epoch in range(1, EPOCHS + 1):

    current_k = get_current_k(epoch)

    # Print phase transitions
    for start_epoch, k_value in CURRICULUM:
        if epoch == start_epoch:
            print(f"\n{'='*40}")
            print(f"CURRICULUM PHASE: K={k_value} "
                  f"(epoch {start_epoch})")
            print(f"{'='*40}")

            # Reset early stopping counter on
            # phase transitions — the loss will
            # temporarily spike when K increases,
            # which is expected, not a sign to stop.
            epochs_without_improvement = 0

            # Reset best F1 so the model can adapt
            # to the harder objective without being
            # anchored to the easier phase's peak.
            best_validation_f1 = -1.0

            # Reset learning rate so the model
            # can adapt to the harder objective.
            for param_group in optimizer.param_groups:
                param_group["lr"] = LEARNING_RATE

            # Reset the LR scheduler state
            scheduler = (
                torch.optim.lr_scheduler
                .ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    factor=0.5,
                    patience=5,
                )
            )

    (
        train_loss,
        train_state_loss,
        train_attack_loss
    ) = train_one_epoch(current_k)

    (
        val_state_loss,
        val_attack_loss,
        val_probabilities,
        val_soft_targets
    ) = collect_validation_outputs()

    # Threshold search uses BINARY labels
    (
        epoch_threshold,
        val_f1,
        threshold_df
    ) = find_best_threshold(
        val_probabilities,
        y_val_binary
    )

    val_predictions = (
        val_probabilities >= epoch_threshold
    ).astype(int)

    val_precision = precision_score(
        y_val_binary, val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        y_val_binary, val_predictions,
        zero_division=0
    )

    # ROC-AUC on validation
    try:
        val_roc_auc = roc_auc_score(
            y_val_binary, val_probabilities
        )
    except ValueError:
        val_roc_auc = 0.0

    scheduler.step(val_f1)

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    history.append({
        "epoch": epoch,
        "curriculum_k": current_k,
        "train_loss": train_loss,
        "train_state_loss": train_state_loss,
        "train_attack_loss": train_attack_loss,
        "val_state_loss": val_state_loss,
        "val_attack_loss": val_attack_loss,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_roc_auc": val_roc_auc,
        "threshold": epoch_threshold,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch:03d}/{EPOCHS} K={current_k} | "
        f"Train: {train_loss:.4f} | "
        f"Val State: {val_state_loss:.4f} | "
        f"Val Atk: {val_attack_loss:.4f} | "
        f"P: {val_precision:.3f} | "
        f"R: {val_recall:.3f} | "
        f"F1: {val_f1:.3f} | "
        f"AUC: {val_roc_auc:.3f} | "
        f"Thr: {epoch_threshold:.2f} | "
        f"LR: {current_lr:.6g}"
    )

    # ----- Checkpoint on best val F1 -----

    if val_f1 > best_validation_f1:

        best_validation_f1 = val_f1
        best_threshold = epoch_threshold

        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if (
        epochs_without_improvement >= PATIENCE
        and current_k == MAX_K
    ):
        print(
            f"\nEarly stopping at epoch {epoch} "
            f"(final phase K={MAX_K})."
        )
        break


# =============================================================
# RESTORE BEST MODEL
# =============================================================

if best_state is None:
    raise RuntimeError(
        "No checkpoint was recorded."
    )

model.load_state_dict(best_state)

print("\n========== BEST VALIDATION ==========")
print(f"Best val F1:        {best_validation_f1:.4f}")
print(f"Selected threshold: {best_threshold:.2f}")


# =============================================================
# TEST EVALUATION (SINGLE-STEP)
# =============================================================

model.eval()

test_state_losses = []
test_probabilities = []

with torch.no_grad():

    for X_batch, Y_ms_batch, A_batch in test_loader:

        X_batch = X_batch.to(DEVICE)
        Y_ms_batch = Y_ms_batch.to(DEVICE)

        (
            predicted_state,
            predicted_latent,
            attack_logits,
            latent_state
        ) = model(X_batch)

        state_loss = state_criterion(
            predicted_state,
            Y_ms_batch[:, 0, :].contiguous()
        )

        probabilities = torch.sigmoid(
            attack_logits
        )

        test_state_losses.append(
            state_loss.item()
            * X_batch.size(0)
        )

        test_probabilities.extend(
            probabilities.cpu().numpy()
        )


test_state_loss = (
    sum(test_state_losses)
    / len(test_dataset)
)

test_probabilities = np.asarray(
    test_probabilities
)


# =============================================================
# TEST METRICS
# =============================================================

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test_binary, test_predictions
)

test_precision = precision_score(
    y_test_binary, test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test_binary, test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test_binary, test_predictions,
    zero_division=0
)

try:
    test_roc_auc = roc_auc_score(
        y_test_binary, test_probabilities
    )
except ValueError:
    test_roc_auc = 0.0

cm = confusion_matrix(
    y_test_binary,
    test_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

test_fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)


# =============================================================
# MULTI-STEP TRAJECTORY EVALUATION
# =============================================================

print("\n========== K-STEP TRAJECTORY EVALUATION ==========")

trajectory_results = []

for k_eval in [1, 2, 3, 5]:
    all_max_probs = []
    all_norms = []

    X_test_tensor = torch.tensor(
        X_test, dtype=torch.float32
    )

    with torch.no_grad():
        for start in range(0, len(X_test), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_test))
            batch = X_test_tensor[start:end].to(DEVICE)

            traj_z, traj_probs, _ = (
                model.k_step_rollout(batch, k_eval)
            )

            # Stack probs: (batch, k+1)
            probs = torch.stack(
                traj_probs, dim=1
            ).cpu().numpy()
            max_probs = probs.max(axis=1)
            all_max_probs.extend(max_probs)

            # Latent norms
            init_norm = (
                torch.norm(traj_z[0], dim=1)
                .cpu().numpy()
            )
            final_norm = (
                torch.norm(traj_z[-1], dim=1)
                .cpu().numpy()
            )
            all_norms.extend(
                final_norm / (init_norm + 1e-8)
            )

    all_max_probs = np.asarray(all_max_probs)
    all_norms = np.asarray(all_norms)

    try:
        traj_roc = roc_auc_score(
            y_test_binary, all_max_probs
        )
    except ValueError:
        traj_roc = 0.0

    traj_f1, traj_thr = find_best_threshold(
        all_max_probs, y_test_binary
    )[:2]

    norm_ratio = all_norms.mean()

    print(
        f"  K={k_eval}: Traj ROC-AUC={traj_roc:.4f}, "
        f"Best F1={traj_f1:.4f}, "
        f"Norm ratio={norm_ratio:.3f}x"
    )

    trajectory_results.append({
        "k": k_eval,
        "trajectory_roc_auc": traj_roc,
        "trajectory_best_f1": traj_f1,
        "trajectory_threshold": traj_thr,
        "norm_ratio": norm_ratio,
    })


# =============================================================
# LOGISTIC REGRESSION BASELINE (SAME SPLITS)
# =============================================================

print("\n========== LOGISTIC REGRESSION BASELINE ==========")
print("(Re-running on same temporal_benchmark_v2 splits)")

# Flatten temporal sequences: (N, 10, 52) -> (N, 520)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Use step-1 binary labels
y_train_lr = A_binary_train[:, 0]
y_val_lr = A_binary_val[:, 0]

lr_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

lr_model.fit(X_train_flat, y_train_lr)

# Find best threshold on val
lr_val_prob = lr_model.predict_proba(
    X_val_flat
)[:, 1]

lr_best_thr = 0.5
lr_best_f1 = -1

for t in np.arange(0.05, 0.96, 0.01):
    preds = (lr_val_prob >= t).astype(int)
    f = f1_score(y_val_lr, preds, zero_division=0)
    if f > lr_best_f1:
        lr_best_f1 = f
        lr_best_thr = t

# Test metrics
lr_test_prob = lr_model.predict_proba(
    X_test_flat
)[:, 1]

lr_test_pred = (
    lr_test_prob >= lr_best_thr
).astype(int)

lr_test_f1 = f1_score(
    y_test_binary, lr_test_pred, zero_division=0
)
lr_test_precision = precision_score(
    y_test_binary, lr_test_pred, zero_division=0
)
lr_test_recall = recall_score(
    y_test_binary, lr_test_pred, zero_division=0
)

try:
    lr_test_roc = roc_auc_score(
        y_test_binary, lr_test_prob
    )
except ValueError:
    lr_test_roc = 0.0

lr_cm = confusion_matrix(
    y_test_binary, lr_test_pred, labels=[0, 1]
)
lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()
lr_test_fpr = (
    lr_fp / (lr_fp + lr_tn)
    if (lr_fp + lr_tn) > 0
    else 0.0
)

print(f"LR best threshold: {lr_best_thr:.2f}")
print(f"LR test F1:        {lr_test_f1:.4f}")
print(f"LR test ROC-AUC:   {lr_test_roc:.4f}")
print(f"LR test FPR:       {lr_test_fpr:.4f}")


# =============================================================
# RESULTS — THREE-WAY COMPARISON
# =============================================================

print("\n" + "=" * 90)
print("LATENT WORLD MODEL V4 RESULTS")
print("=" * 90)

print(f"Best val F1:        {best_validation_f1:.4f}")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Test state loss:    {test_state_loss:.6f}")
print(f"\nTest attack metrics:")
print(f"  Accuracy:   {test_accuracy:.4f}")
print(f"  Precision:  {test_precision:.4f}")
print(f"  Recall:     {test_recall:.4f}")
print(f"  F1:         {test_f1:.4f}")
print(f"  ROC-AUC:    {test_roc_auc:.4f}")
print(f"  FPR:        {test_fpr:.4f}")
print(f"\nConfusion Matrix:")
print(cm)


# =============================================================
# COMPARISON TABLE: LR vs V3 vs V4
# =============================================================

print("\n" + "=" * 90)
print("THREE-WAY COMPARISON (same temporal_benchmark_v2 splits)")
print("=" * 90)

print(f"\n{'Model':<25} {'F1':>8} {'Precision':>10} "
      f"{'Recall':>8} {'FPR':>8} {'ROC-AUC':>9}")
print("-" * 70)

# Logistic Regression (just computed)
print(f"{'Logistic Regression':<25} "
      f"{lr_test_f1:>8.4f} "
      f"{lr_test_precision:>10.4f} "
      f"{lr_test_recall:>8.4f} "
      f"{lr_test_fpr:>8.4f} "
      f"{lr_test_roc:>9.4f}")

# V3 (hardcoded from V3 training output)
print(f"{'Latent WM V3':<25} "
      f"{'0.5116':>8} "
      f"{'0.7674':>10} "
      f"{'0.3837':>8} "
      f"{'0.0606':>8} "
      f"{'0.7513':>9}")

# V4 (just computed)
print(f"{'Latent WM V4':<25} "
      f"{test_f1:>8.4f} "
      f"{test_precision:>10.4f} "
      f"{test_recall:>8.4f} "
      f"{test_fpr:>8.4f} "
      f"{test_roc_auc:>9.4f}")


# Verdict
print("\n--- Verdict ---")

if test_f1 > 0.5116:
    print("✓ V4 BEATS V3 on F1!")
elif test_f1 > lr_test_f1:
    print("△ V4 beats LR baseline but not V3.")
else:
    print("✗ V4 does NOT beat V3.")

if test_roc_auc > 0.7513:
    print("✓ V4 has better ranking (ROC-AUC) than V3!")
else:
    print("△ V4 ranking is not better than V3.")

if test_fpr <= 0.0606:
    print("✓ V4 FPR is equal or better than V3!")
else:
    print("△ V4 FPR is worse than V3.")

# Trajectory improvement
print("\n--- Trajectory Stability ---")
for r in trajectory_results:
    if r["k"] == 5:
        if 0.7 <= r["norm_ratio"] <= 1.3:
            print(f"✓ K=5 latent norm stable "
                  f"(ratio={r['norm_ratio']:.3f})")
        else:
            print(f"✗ K=5 latent norm unstable "
                  f"(ratio={r['norm_ratio']:.3f})")

        if r["trajectory_roc_auc"] > 0.744:
            print(f"✓ K=5 trajectory ROC-AUC improved "
                  f"({r['trajectory_roc_auc']:.4f} vs 0.744)")
        else:
            print(f"△ K=5 trajectory ROC-AUC: "
                  f"{r['trajectory_roc_auc']:.4f}")


# =============================================================
# SAVE MODEL
# =============================================================

model_path = (
    MODEL_DIR / "latent_world_model_v4.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),
        "model_class": "LatentWorldModelV4",
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "latent_size": LATENT_SIZE,
        "output_size": OUTPUT_SIZE,
        "num_layers": NUM_LAYERS,
        "sequence_length": X_train.shape[1],
        "positive_class_weight":
            positive_weight_value,
        "best_validation_f1":
            best_validation_f1,
        "selected_threshold":
            best_threshold,
        "test_state_loss": test_state_loss,
        "test_attack_f1": test_f1,
        "test_roc_auc": test_roc_auc,
        "test_fpr": test_fpr,
        "max_k": MAX_K,
        "gamma": GAMMA,
        "curriculum": CURRICULUM,
        "changes_from_v3": [
            "multi_step_objective",
            "gated_residual_transition",
            "latent_layer_norm",
            "curriculum_training",
        ],
        "transition_type": "deterministic",
    },
    model_path
)


# =============================================================
# SAVE TRAINING HISTORY
# =============================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    MODEL_DIR / "training_history.csv",
    index=False
)


# =============================================================
# SAVE TEST PREDICTIONS
# =============================================================

prediction_df = test_meta.copy()

prediction_df[
    "attack_probability"
] = test_probabilities

prediction_df[
    "predicted_attack"
] = test_predictions

prediction_df[
    "threshold"
] = best_threshold

prediction_df.to_csv(
    MODEL_DIR / "test_attack_predictions.csv",
    index=False
)


# =============================================================
# SAVE METRICS
# =============================================================

metrics_df = pd.DataFrame({
    "metric": [
        "best_validation_f1",
        "selected_threshold",
        "test_state_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "test_false_positive_rate",
        "positive_class_weight",
        "max_k",
        "gamma",
        "lr_baseline_f1",
        "lr_baseline_roc_auc",
        "lr_baseline_fpr",
    ],
    "value": [
        best_validation_f1,
        best_threshold,
        test_state_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_roc_auc,
        test_fpr,
        positive_weight_value,
        MAX_K,
        GAMMA,
        lr_test_f1,
        lr_test_roc,
        lr_test_fpr,
    ]
})

metrics_df.to_csv(
    MODEL_DIR / "metrics.csv",
    index=False
)


# =============================================================
# SAVE CONFUSION MATRIX
# =============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Benign",
        "Actual_Attack"
    ],
    columns=[
        "Predicted_Benign",
        "Predicted_Attack"
    ]
)

cm_df.to_csv(
    MODEL_DIR / "confusion_matrix.csv"
)


# =============================================================
# SAVE TRAJECTORY RESULTS
# =============================================================

traj_df = pd.DataFrame(trajectory_results)
traj_df.to_csv(
    MODEL_DIR / "trajectory_evaluation.csv",
    index=False
)


# =============================================================
# SAVE VALIDATION THRESHOLD SEARCH
# =============================================================

(
    final_val_state_loss,
    final_val_attack_loss,
    final_val_probs,
    final_val_targets
) = collect_validation_outputs()

(
    _, _,
    final_threshold_df
) = find_best_threshold(
    final_val_probs,
    y_val_binary
)

final_threshold_df.to_csv(
    MODEL_DIR / "validation_threshold_search.csv",
    index=False
)


# =============================================================
# FINISH
# =============================================================

print(f"\nSaved model:       {model_path}")
print(f"Saved history:     {MODEL_DIR / 'training_history.csv'}")
print(f"Saved predictions: {MODEL_DIR / 'test_attack_predictions.csv'}")
print(f"Saved metrics:     {MODEL_DIR / 'metrics.csv'}")
print(f"Saved trajectory:  {MODEL_DIR / 'trajectory_evaluation.csv'}")

print("\n" + "=" * 90)
print("LATENT WORLD MODEL V4 COMPLETE")
print("=" * 90)
