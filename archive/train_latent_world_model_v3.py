"""
=============================================================
LATENT WORLD MODEL V3 — EVIDENCE-BASED IMPROVEMENT
=============================================================

CHANGES FROM V2 (each driven by V2 diagnosis):

1. RESIDUAL TRANSITION:
   V2: Z_(t+1) = MLP(Z_t)
   V3: Z_(t+1) = Z_t + MLP(Z_t)

   WHY: V2 diagnosis showed the absolute transition
   fails under distribution shift. Learning residual
   deltas is more robust because even if the delta
   predictor is imperfect, the output stays near Z_t
   rather than collapsing to a wrong absolute value.

2. SOFT ATTACK TARGET:
   V2: binary attack_present (0 or 1)
   V3: continuous attack_ratio (0.0 to 1.0)

   WHY: V2 diagnosis showed probabilities poorly
   calibrated (val mean 0.76 vs test mean 0.28).
   A window with 6% attack flows shouldn't get the
   same label as 49%. Soft targets give the model
   proportional signal.

3. WIDER TRANSITION MLP:
   V2: 32 -> 64 -> 32
   V3: 32 -> 128 -> 32

   WHY: The transition network is the core "world model"
   component. More capacity may help it learn richer
   dynamics.

4. TRAINING HYPERPARAMETERS:
   V2: epochs=50, patience=8, lr=1e-3
   V3: epochs=100, patience=15, lr=5e-4

   WHY: V2 best checkpoint was epoch 2 of 10
   (early stopped). The model was undertrained.

EVERYTHING ELSE IS IDENTICAL TO V2:
- Same data (V2 temporal benchmark, 28-Feb)
- Same encoder (2-layer LSTM, 128 hidden)
- Same latent size (32)
- Same decoder architecture
- Same attack head architecture
- Same loss functions (SmoothL1 + weighted BCE)
- Same evaluation methodology

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
)


# =============================================================
# CONFIGURATION
# =============================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

MODEL_DIR = Path(
    "models/latent_world_model_v3"
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

# --- CHANGED from V2 ---
LEARNING_RATE = 5e-4      # V2: 1e-3
WEIGHT_DECAY = 1e-5

EPOCHS = 100              # V2: 50
PATIENCE = 15             # V2: 8
# ------------------------

STATE_LOSS_WEIGHT = 1.0
ATTACK_LOSS_WEIGHT = 0.5


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
print("LATENT WORLD MODEL V3 — EVIDENCE-BASED IMPROVEMENT")
print("=" * 90)
print(f"Using device: {DEVICE}")

print("\n--- Changes from V2 ---")
print("1. Residual transition: Z' = Z + MLP(Z)")
print("2. Soft attack target: attack_ratio (0-1)")
print("3. Wider transition: 32->128->32")
print("4. LR=5e-4, epochs=100, patience=15")
print("-" * 40)


# =============================================================
# LOAD DATA
# =============================================================

print("\nLoading V2 temporal benchmark...")

X_train = np.load(DATA_DIR / "X_train.npy")
Y_train = np.load(DATA_DIR / "Y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
Y_val = np.load(DATA_DIR / "Y_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
Y_test = np.load(DATA_DIR / "Y_test.npy")

train_meta = pd.read_csv(
    DATA_DIR / "train_metadata.csv"
)
val_meta = pd.read_csv(
    DATA_DIR / "val_metadata.csv"
)
test_meta = pd.read_csv(
    DATA_DIR / "test_metadata.csv"
)

print(f"\nX_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")


# =============================================================
# NUMERICAL VALIDATION
# =============================================================

for name, array in [
    ("X_train", X_train),
    ("Y_train", Y_train),
    ("X_val", X_val),
    ("Y_val", Y_val),
    ("X_test", X_test),
    ("Y_test", Y_test),
]:
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )


# =============================================================
# ATTACK TARGETS — CHANGE #2: SOFT TARGETS
# =============================================================
#
# V2 used binary: attack_present (0 or 1)
# V3 uses continuous: attack_ratio (0.0 to 1.0)
#
# For evaluation thresholding, we still need
# the binary labels too.
# =============================================================

# Soft targets for training
y_train_soft = (
    train_meta["attack_ratio"]
    .to_numpy(dtype=np.float32)
)

y_val_soft = (
    val_meta["attack_ratio"]
    .to_numpy(dtype=np.float32)
)

y_test_soft = (
    test_meta["attack_ratio"]
    .to_numpy(dtype=np.float32)
)

# Binary labels for evaluation
y_train_binary = (
    train_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)

y_val_binary = (
    val_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)

y_test_binary = (
    test_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)


print("\n========== ATTACK DISTRIBUTION ==========")

for name, soft, binary in [
    ("TRAIN", y_train_soft, y_train_binary),
    ("VAL", y_val_soft, y_val_binary),
    ("TEST", y_test_soft, y_test_binary),
]:
    benign = int((binary == 0).sum())
    attack = int((binary == 1).sum())
    mean_ratio = soft[binary == 1].mean() if attack > 0 else 0

    print(
        f"{name}: Benign={benign}, "
        f"Attack={attack}, "
        f"Mean attack_ratio={mean_ratio:.3f}"
    )


# =============================================================
# POSITIVE CLASS WEIGHT
# =============================================================

train_benign = int(
    (y_train_binary == 0).sum()
)
train_attack = int(
    (y_train_binary == 1).sum()
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

X_train_t = torch.tensor(
    X_train, dtype=torch.float32
)
Y_train_t = torch.tensor(
    Y_train, dtype=torch.float32
)
A_train_t = torch.tensor(
    y_train_soft, dtype=torch.float32
)

X_val_t = torch.tensor(
    X_val, dtype=torch.float32
)
Y_val_t = torch.tensor(
    Y_val, dtype=torch.float32
)
A_val_t = torch.tensor(
    y_val_soft, dtype=torch.float32
)

X_test_t = torch.tensor(
    X_test, dtype=torch.float32
)
Y_test_t = torch.tensor(
    Y_test, dtype=torch.float32
)
A_test_t = torch.tensor(
    y_test_soft, dtype=torch.float32
)


# =============================================================
# DATASETS AND LOADERS
# =============================================================

train_dataset = TensorDataset(
    X_train_t, Y_train_t, A_train_t
)
val_dataset = TensorDataset(
    X_val_t, Y_val_t, A_val_t
)
test_dataset = TensorDataset(
    X_test_t, Y_test_t, A_test_t
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
# MODEL — CHANGE #1 & #3: RESIDUAL + WIDER TRANSITION
# =============================================================

class LatentWorldModelV3(nn.Module):
    """
    Same as V2 except:
    - Transition is RESIDUAL: Z' = Z + delta_MLP(Z)
    - Transition MLP is WIDER: 32 -> 128 -> 32
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

        # ----- Encoder (SAME as V2) -----

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

        # ----- Transition (CHANGED) -----
        #
        # V2: self.transition = Sequential(
        #         32 -> 64 -> 32
        #     )
        #     Z' = transition(Z)
        #
        # V3: self.transition_delta = Sequential(
        #         32 -> 128 -> 32
        #     )
        #     Z' = Z + transition_delta(Z)

        self.transition_delta = nn.Sequential(
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

        # ----- Decoder (SAME as V2) -----

        self.decoder_fc1 = nn.Linear(
            latent_size, 64
        )
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(
            64, output_size
        )

        # ----- Attack head (SAME as V2) -----

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
        # CHANGE #1: RESIDUAL
        delta = self.transition_delta(
            latent_state
        )
        return latent_state + delta


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


# =============================================================
# INITIALIZE MODEL
# =============================================================

model = LatentWorldModelV3(
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

# Weighted BCE for attack detection
# (same weighting as V2)
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
    patience=5    # V2: 3
)


# =============================================================
# TRAIN ONE EPOCH
# =============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_state_loss = 0.0
    total_attack_loss = 0.0
    sample_count = 0

    for X_batch, Y_batch, A_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)
        A_batch = A_batch.to(DEVICE)

        optimizer.zero_grad()

        (
            predicted_state,
            predicted_latent,
            attack_logits,
            latent_state
        ) = model(X_batch)

        state_loss = state_criterion(
            predicted_state, Y_batch
        )

        attack_loss = attack_criterion(
            attack_logits, A_batch
        )

        loss = (
            STATE_LOSS_WEIGHT * state_loss
            + ATTACK_LOSS_WEIGHT * attack_loss
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
            state_loss.item() * batch_size
        )
        total_attack_loss += (
            attack_loss.item() * batch_size
        )
        sample_count += batch_size

    return (
        total_loss / sample_count,
        total_state_loss / sample_count,
        total_attack_loss / sample_count
    )


# =============================================================
# VALIDATION
# =============================================================

def collect_validation_outputs():

    model.eval()

    total_state_loss = 0.0
    total_attack_loss = 0.0
    sample_count = 0

    attack_probabilities = []
    attack_targets = []

    with torch.no_grad():

        for X_batch, Y_batch, A_batch in val_loader:

            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)
            A_batch = A_batch.to(DEVICE)

            (
                predicted_state,
                predicted_latent,
                attack_logits,
                latent_state
            ) = model(X_batch)

            state_loss = state_criterion(
                predicted_state, Y_batch
            )

            attack_loss = attack_criterion(
                attack_logits, A_batch
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
                A_batch.cpu().numpy()
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
#
# NOTE: For threshold search, we evaluate against
# BINARY labels (attack_present), not soft targets.
# The model outputs probabilities; we find the
# threshold that maximizes F1 on binary classification.
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

        recall = recall_score(
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
            "recall": recall,
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


for epoch in range(1, EPOCHS + 1):

    (
        train_loss,
        train_state_loss,
        train_attack_loss
    ) = train_one_epoch()

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

    # ROC-AUC on validation (binary labels)
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
        f"Epoch {epoch:03d}/{EPOCHS} | "
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

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}."
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
# TEST EVALUATION
# =============================================================

model.eval()

test_state_losses = []
test_probabilities = []
test_targets = []

with torch.no_grad():

    for X_batch, Y_batch, A_batch in test_loader:

        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        (
            predicted_state,
            predicted_latent,
            attack_logits,
            latent_state
        ) = model(X_batch)

        state_loss = state_criterion(
            predicted_state, Y_batch
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

        test_targets.extend(
            A_batch.cpu().numpy()
        )


test_state_loss = (
    sum(test_state_losses)
    / len(test_dataset)
)

test_probabilities = np.asarray(
    test_probabilities
)

test_targets = np.asarray(
    test_targets
)


# =============================================================
# TEST METRICS (using BINARY labels)
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


# =============================================================
# CONFUSION MATRIX
# =============================================================

from sklearn.metrics import confusion_matrix

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
# RESULTS
# =============================================================

print("\n" + "=" * 90)
print("LATENT WORLD MODEL V3 RESULTS")
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
# COMPARISON WITH V2 AND BASELINE
# =============================================================

print("\n" + "=" * 90)
print("COMPARISON WITH PREVIOUS MODELS")
print("=" * 90)

print(f"\n{'Model':<25} {'F1':>8} {'Precision':>10} "
      f"{'Recall':>8} {'FPR':>8} {'ROC-AUC':>9}")
print("-" * 70)

print(f"{'Logistic Regression':<25} {'0.4641':>8} "
      f"{'0.5299':>10} {'0.4128':>8} "
      f"{'0.1909':>8} {'—':>9}")

print(f"{'Latent WM V2':<25} {'0.4118':>8} "
      f"{'0.7424':>10} {'0.2849':>8} "
      f"{'0.0515':>8} {'0.7229':>9}")

print(f"{'Latent WM V3':<25} "
      f"{test_f1:>8.4f} "
      f"{test_precision:>10.4f} "
      f"{test_recall:>8.4f} "
      f"{test_fpr:>8.4f} "
      f"{test_roc_auc:>9.4f}")

# Verdict
print("\n--- Verdict ---")

if test_f1 > 0.4641:
    print("✓ V3 BEATS logistic regression baseline!")
elif test_f1 > 0.4118:
    print("△ V3 beats V2 but not logistic regression.")
else:
    print("✗ V3 does NOT beat V2.")

if test_roc_auc > 0.7229:
    print("✓ V3 has better ranking (ROC-AUC) than V2!")
else:
    print("△ V3 ranking is not better than V2.")


# =============================================================
# SAVE MODEL
# =============================================================

model_path = (
    MODEL_DIR / "latent_world_model_v3.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),
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
        "changes_from_v2": [
            "residual_transition",
            "soft_attack_target",
            "wider_transition_mlp",
            "adjusted_hyperparameters",
        ],
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

print(f"\nSaved model:      {model_path}")
print(f"Saved history:    {MODEL_DIR / 'training_history.csv'}")
print(f"Saved predictions:{MODEL_DIR / 'test_attack_predictions.csv'}")
print(f"Saved metrics:    {MODEL_DIR / 'metrics.csv'}")

print("\n" + "=" * 90)
print("LATENT WORLD MODEL V3 COMPLETE")
print("=" * 90)
