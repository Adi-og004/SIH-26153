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
)


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

MODEL_DIR = Path(
    "models/latent_world_model_v2"
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

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

EPOCHS = 50
PATIENCE = 8

# ---------------------------------------------------------
# Loss weights
# ---------------------------------------------------------

STATE_LOSS_WEIGHT = 1.0
ATTACK_LOSS_WEIGHT = 0.5

# ---------------------------------------------------------
# Positive-class weighting
# ---------------------------------------------------------
#
# This will be calculated from the training set:
#
# attack_weight = benign_count / attack_count
#
# This makes missing an attack more costly.
# ---------------------------------------------------------


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)


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
print("LATENT WORLD MODEL V2")
print("=" * 90)

print(
    f"Using device: {DEVICE}"
)


# =========================================================
# LOAD DATA
# =========================================================

print(
    "\nLoading V2 temporal benchmark..."
)

X_train = np.load(
    DATA_DIR / "X_train.npy"
)

Y_train = np.load(
    DATA_DIR / "Y_train.npy"
)

X_val = np.load(
    DATA_DIR / "X_val.npy"
)

Y_val = np.load(
    DATA_DIR / "Y_val.npy"
)

X_test = np.load(
    DATA_DIR / "X_test.npy"
)

Y_test = np.load(
    DATA_DIR / "Y_test.npy"
)


train_meta = pd.read_csv(
    DATA_DIR / "train_metadata.csv"
)

val_meta = pd.read_csv(
    DATA_DIR / "val_metadata.csv"
)

test_meta = pd.read_csv(
    DATA_DIR / "test_metadata.csv"
)


print("\n========== SHAPES ==========")

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
# NUMERICAL VALIDATION
# =========================================================

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
            f"{name} contains NaN or Inf values."
        )


# =========================================================
# ATTACK TARGETS
# =========================================================

y_train_attack = (
    train_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)

y_val_attack = (
    val_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)

y_test_attack = (
    test_meta["attack_present"]
    .to_numpy(dtype=np.float32)
)


print(
    "\n========== ATTACK DISTRIBUTION =========="
)

for name, labels in [
    ("TRAIN", y_train_attack),
    ("VALIDATION", y_val_attack),
    ("TEST", y_test_attack),
]:

    benign = int(
        (labels == 0).sum()
    )

    attack = int(
        (labels == 1).sum()
    )

    print(
        f"{name}: "
        f"Benign={benign}, "
        f"Attack={attack}"
    )


# =========================================================
# CALCULATE POSITIVE CLASS WEIGHT
# =========================================================

train_benign = int(
    (y_train_attack == 0).sum()
)

train_attack = int(
    (y_train_attack == 1).sum()
)

if train_attack == 0:

    raise ValueError(
        "Training set contains no attack samples."
    )

positive_weight_value = (
    train_benign / train_attack
)

print(
    "\nAttack positive-class weight:",
    round(
        positive_weight_value,
        4
    )
)


# =========================================================
# PYTORCH TENSORS
# =========================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

Y_train_tensor = torch.tensor(
    Y_train,
    dtype=torch.float32
)

A_train_tensor = torch.tensor(
    y_train_attack,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

Y_val_tensor = torch.tensor(
    Y_val,
    dtype=torch.float32
)

A_val_tensor = torch.tensor(
    y_val_attack,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

Y_test_tensor = torch.tensor(
    Y_test,
    dtype=torch.float32
)

A_test_tensor = torch.tensor(
    y_test_attack,
    dtype=torch.float32
)


# =========================================================
# DATASETS
# =========================================================

train_dataset = TensorDataset(
    X_train_tensor,
    Y_train_tensor,
    A_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    Y_val_tensor,
    A_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    Y_test_tensor,
    A_test_tensor
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


# =========================================================
# MODEL
# =========================================================

class LatentWorldModel(nn.Module):

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

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Latent transition
        # -------------------------------------------------

        self.transition = nn.Sequential(

            nn.Linear(
                latent_size,
                latent_size * 2
            ),

            nn.ReLU(),

            nn.Linear(
                latent_size * 2,
                latent_size
            )
        )

        # -------------------------------------------------
        # Decoder
        # -------------------------------------------------

        self.decoder_fc1 = nn.Linear(
            latent_size,
            64
        )

        self.decoder_activation = nn.ReLU()

        self.decoder_fc2 = nn.Linear(
            64,
            output_size
        )

        # -------------------------------------------------
        # Attack head
        # -------------------------------------------------

        self.attack_head = nn.Sequential(

            nn.Linear(
                latent_size,
                32
            ),

            nn.ReLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                32,
                1
            )
        )


    def encode(self, x):

        output, _ = self.encoder_lstm(
            x
        )

        last_hidden = output[:, -1, :]

        last_hidden = self.encoder_norm(
            last_hidden
        )

        latent_state = self.encoder_fc(
            last_hidden
        )

        return latent_state


    def transition_step(
        self,
        latent_state
    ):

        predicted_latent = self.transition(
            latent_state
        )

        return predicted_latent


    def decode(
        self,
        latent_state
    ):

        hidden = self.decoder_fc1(
            latent_state
        )

        hidden = self.decoder_activation(
            hidden
        )

        predicted_state = self.decoder_fc2(
            hidden
        )

        return predicted_state


    def predict_attack(
        self,
        latent_state
    ):

        logits = self.attack_head(
            latent_state
        )

        return logits.squeeze(
            -1
        )


    def forward(self, x):

        latent_state = self.encode(
            x
        )

        predicted_latent = (
            self.transition_step(
                latent_state
            )
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


# =========================================================
# INITIALIZE MODEL
# =========================================================

model = LatentWorldModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    latent_size=LATENT_SIZE,
    output_size=OUTPUT_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

model = model.to(
    DEVICE
)


print("\n========== MODEL ==========")

print(model)

print(
    "\nTrainable parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
)


# =========================================================
# LOSS FUNCTIONS
# =========================================================

state_criterion = nn.SmoothL1Loss()

# ---------------------------------------------------------
# Weighted BCE
# ---------------------------------------------------------

positive_weight = torch.tensor(
    [positive_weight_value],
    dtype=torch.float32,
    device=DEVICE
)

attack_criterion = nn.BCEWithLogitsLoss(
    pos_weight=positive_weight.squeeze(0)
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# LEARNING RATE SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_state_loss = 0.0
    total_attack_loss = 0.0

    sample_count = 0

    for (
        X_batch,
        Y_batch,
        A_batch
    ) in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        A_batch = A_batch.to(
            DEVICE
        )

        optimizer.zero_grad()

        (
            predicted_state,
            predicted_latent,
            attack_logits,
            latent_state
        ) = model(
            X_batch
        )

        state_loss = state_criterion(
            predicted_state,
            Y_batch
        )

        attack_loss = attack_criterion(
            attack_logits,
            A_batch
        )

        loss = (
            STATE_LOSS_WEIGHT * state_loss
            +
            ATTACK_LOSS_WEIGHT * attack_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_state_loss += (
            state_loss.item()
            * batch_size
        )

        total_attack_loss += (
            attack_loss.item()
            * batch_size
        )

        sample_count += batch_size

    return (
        total_loss / sample_count,
        total_state_loss / sample_count,
        total_attack_loss / sample_count
    )


# =========================================================
# VALIDATION
# =========================================================

def collect_validation_outputs():

    model.eval()

    total_state_loss = 0.0
    total_attack_loss = 0.0
    sample_count = 0

    attack_probabilities = []
    attack_targets = []

    with torch.no_grad():

        for (
            X_batch,
            Y_batch,
            A_batch
        ) in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            Y_batch = Y_batch.to(
                DEVICE
            )

            A_batch = A_batch.to(
                DEVICE
            )

            (
                predicted_state,
                predicted_latent,
                attack_logits,
                latent_state
            ) = model(
                X_batch
            )

            state_loss = state_criterion(
                predicted_state,
                Y_batch
            )

            attack_loss = attack_criterion(
                attack_logits,
                A_batch
            )

            batch_size = X_batch.size(0)

            total_state_loss += (
                state_loss.item()
                * batch_size
            )

            total_attack_loss += (
                attack_loss.item()
                * batch_size
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
        np.asarray(
            attack_probabilities
        ),
        np.asarray(
            attack_targets
        )
    )


# =========================================================
# THRESHOLD SEARCH
# =========================================================

def find_best_threshold(
    probabilities,
    targets
):

    best_threshold = 0.50
    best_f1 = -1.0

    threshold_results = []

    for threshold in np.arange(
        0.05,
        0.96,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            targets,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            targets,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            targets,
            predictions,
            zero_division=0
        )

        threshold_results.append({

            "threshold":
                threshold,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1
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


# =========================================================
# TRAINING LOOP
# =========================================================

print("\n========== TRAINING ==========")

best_validation_f1 = -1.0

best_state = None

best_threshold = 0.50

epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    (
        train_loss,
        train_state_loss,
        train_attack_loss
    ) = train_one_epoch()

    (
        val_state_loss,
        val_attack_loss,
        val_probabilities,
        val_targets
    ) = collect_validation_outputs()

    (
        epoch_threshold,
        val_f1,
        threshold_df
    ) = find_best_threshold(
        val_probabilities,
        val_targets
    )

    val_predictions = (
        val_probabilities
        >= epoch_threshold
    ).astype(int)

    val_precision = precision_score(
        val_targets,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        val_targets,
        val_predictions,
        zero_division=0
    )

    scheduler.step(
        val_f1
    )

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_state_loss":
            train_state_loss,

        "train_attack_loss":
            train_attack_loss,

        "val_state_loss":
            val_state_loss,

        "val_attack_loss":
            val_attack_loss,

        "val_precision":
            val_precision,

        "val_recall":
            val_recall,

        "val_f1":
            val_f1,

        "threshold":
            epoch_threshold,

        "learning_rate":
            current_lr
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train: {train_loss:.6f} | "
        f"Val State: {val_state_loss:.6f} | "
        f"Val Attack: {val_attack_loss:.6f} | "
        f"Precision: {val_precision:.4f} | "
        f"Recall: {val_recall:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"Threshold: {epoch_threshold:.2f} | "
        f"LR: {current_lr:.6g}"
    )

    # -----------------------------------------------------
    # Checkpoint selection based on VALIDATION F1
    # -----------------------------------------------------

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
            "\nEarly stopping triggered."
        )

        break


# =========================================================
# RESTORE BEST MODEL
# =========================================================

if best_state is None:

    raise RuntimeError(
        "No best model checkpoint was recorded."
    )

model.load_state_dict(
    best_state
)


print("\n========== BEST VALIDATION MODEL ==========")

print(
    f"Best validation F1: "
    f"{best_validation_f1:.4f}"
)

print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)


# =========================================================
# TEST EVALUATION
# =========================================================

model.eval()

test_state_losses = []

test_probabilities = []
test_targets = []

with torch.no_grad():

    for (
        X_batch,
        Y_batch,
        A_batch
    ) in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        A_batch = A_batch.to(
            DEVICE
        )

        (
            predicted_state,
            predicted_latent,
            attack_logits,
            latent_state
        ) = model(
            X_batch
        )

        state_loss = state_criterion(
            predicted_state,
            Y_batch
        )

        probabilities = torch.sigmoid(
            attack_logits
        )

        test_state_losses.append(
            state_loss.item()
            *
            X_batch.size(0)
        )

        test_probabilities.extend(
            probabilities.cpu().numpy()
        )

        test_targets.extend(
            A_batch.cpu().numpy()
        )


test_state_loss = (
    sum(test_state_losses)
    /
    len(test_dataset)
)

test_probabilities = np.asarray(
    test_probabilities
)

test_targets = np.asarray(
    test_targets
)


# =========================================================
# APPLY VALIDATION-DERIVED THRESHOLD
# =========================================================

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


# =========================================================
# TEST ATTACK METRICS
# =========================================================

test_accuracy = accuracy_score(
    test_targets,
    test_predictions
)

test_precision = precision_score(
    test_targets,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_targets,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_targets,
    test_predictions,
    zero_division=0
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    test_targets,
    test_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

test_fpr = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0.0
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 90)
print("LATENT WORLD MODEL V2 RESULTS")
print("=" * 90)

print(
    f"Best validation F1: "
    f"{best_validation_f1:.4f}"
)

print(
    f"Selected threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Test state loss: "
    f"{test_state_loss:.6f}"
)

print(
    f"Test attack accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test attack precision: "
    f"{test_precision:.4f}"
)

print(
    f"Test attack recall: "
    f"{test_recall:.4f}"
)

print(
    f"Test attack F1: "
    f"{test_f1:.4f}"
)

print(
    f"Test false positive rate: "
    f"{test_fpr:.4f}"
)

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# SAVE MODEL
# =========================================================

model_path = (
    MODEL_DIR /
    "latent_world_model_v2.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_size":
            INPUT_SIZE,

        "hidden_size":
            HIDDEN_SIZE,

        "latent_size":
            LATENT_SIZE,

        "output_size":
            OUTPUT_SIZE,

        "num_layers":
            NUM_LAYERS,

        "sequence_length":
            X_train.shape[1],

        "positive_class_weight":
            positive_weight_value,

        "best_validation_f1":
            best_validation_f1,

        "selected_threshold":
            best_threshold,

        "test_state_loss":
            test_state_loss,

        "test_attack_f1":
            test_f1,
    },
    model_path
)


# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    MODEL_DIR /
    "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# =========================================================
# SAVE TEST PREDICTIONS
# =========================================================

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

prediction_path = (
    MODEL_DIR /
    "test_attack_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics_df = pd.DataFrame({

    "metric": [

        "best_validation_f1",
        "selected_threshold",

        "test_state_loss",

        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
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
        test_fpr,

        positive_weight_value,
    ]
})

metrics_path = (
    MODEL_DIR /
    "metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# =========================================================
# SAVE CONFUSION MATRIX
# =========================================================

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
    MODEL_DIR /
    "confusion_matrix.csv"
)


# =========================================================
# SAVE THRESHOLD SEARCH
# =========================================================

# Re-run threshold search on the final validation outputs
# for record keeping.

(
    final_val_state_loss,
    final_val_attack_loss,
    final_val_probabilities,
    final_val_targets
) = collect_validation_outputs()

(
    _,
    _,
    final_threshold_df
) = find_best_threshold(
    final_val_probabilities,
    final_val_targets
)

final_threshold_df.to_csv(
    MODEL_DIR /
    "validation_threshold_search.csv",
    index=False
)


# =========================================================
# FINISH
# =========================================================

print("\nSaved model:")
print(model_path)

print("\nSaved training history:")
print(history_path)

print("\nSaved predictions:")
print(prediction_path)

print("\nSaved metrics:")
print(metrics_path)

print("\nSaved confusion matrix:")
print(
    MODEL_DIR /
    "confusion_matrix.csv"
)

print("\n" + "=" * 90)
print("LATENT WORLD MODEL V2 COMPLETE")
print("=" * 90)