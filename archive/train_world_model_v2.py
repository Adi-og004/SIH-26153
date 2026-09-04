import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = Path(
    "data/processed/temporal_benchmark_v2"
)

MODEL_DIR = Path(
    "models/world_model_v2"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_SEED = 42

BATCH_SIZE = 64

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.20

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

EPOCHS = 50

PATIENCE = 8


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


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
print("LSTM WORLD MODEL V2")
print("=" * 90)

print(f"Using device: {DEVICE}")


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading V2 temporal benchmark...")

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


print("\n========== DATA SHAPES ==========")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)


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
            f"{name} contains NaN or Inf."
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

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

Y_val_tensor = torch.tensor(
    Y_val,
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


# =========================================================
# DATASETS / DATALOADERS
# =========================================================

train_dataset = TensorDataset(
    X_train_tensor,
    Y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    Y_val_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    Y_test_tensor
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

class WorldModelLSTM(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float
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

        # x shape:
        # [batch, sequence_length, features]

        lstm_output, _ = self.lstm(x)

        # Final temporal representation.
        last_state = lstm_output[:, -1, :]

        last_state = self.norm(
            last_state
        )

        hidden = self.fc1(
            last_state
        )

        hidden = self.activation(
            hidden
        )

        output = self.fc2(
            hidden
        )

        return output


# =========================================================
# MODEL INITIALIZATION
# =========================================================

INPUT_SIZE = X_train.shape[2]
OUTPUT_SIZE = Y_train.shape[1]

model = WorldModelLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE,
    dropout=DROPOUT
)

model = model.to(DEVICE)


print("\n========== MODEL ==========")

print(model)

print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# =========================================================
# LOSS
# =========================================================

criterion = nn.SmoothL1Loss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# =========================================================
# LEARNING-RATE SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    sample_count = 0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(X_batch)

        loss = criterion(
            predictions,
            Y_batch
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

        sample_count += batch_size

    return total_loss / sample_count


# =========================================================
# EVALUATION
# =========================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    sample_count = 0

    with torch.no_grad():

        for X_batch, Y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                Y_batch
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            sample_count += batch_size

    return total_loss / sample_count


# =========================================================
# TRAINING LOOP
# =========================================================

print("\n========== TRAINING ==========")

best_val_loss = float("inf")

best_state = None

epochs_without_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss = train_one_epoch()

    val_loss = evaluate(
        val_loader
    )

    scheduler.step(
        val_loss
    )

    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_lr,
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"LR: {current_lr:.6g}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

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


# =========================================================
# TEST EVALUATION
# =========================================================

test_loss = evaluate(
    test_loader
)


print("\n" + "=" * 90)
print("WORLD MODEL V2 RESULTS")
print("=" * 90)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Test loss: "
    f"{test_loss:.6f}"
)


# =========================================================
# SAVE CHECKPOINT
# =========================================================

model_path = (
    MODEL_DIR /
    "lstm_world_model_v2.pt"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "input_size":
            INPUT_SIZE,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "output_size":
            OUTPUT_SIZE,

        "sequence_length":
            X_train.shape[1],

        "best_val_loss":
            best_val_loss,

        "test_loss":
            test_loss,
    },
    model_path
)


# =========================================================
# SAVE HISTORY
# =========================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    MODEL_DIR /
    "training_history_v2.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# =========================================================
# GENERATE TEST PREDICTIONS
# =========================================================

model.eval()

predictions = []

with torch.no_grad():

    for X_batch, _ in test_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        output = model(
            X_batch
        )

        predictions.append(
            output.cpu().numpy()
        )


predictions = np.concatenate(
    predictions,
    axis=0
)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

np.save(
    MODEL_DIR /
    "test_predictions_v2.npy",
    predictions
)


# =========================================================
# FINISH
# =========================================================

print("\nSaved model:")
print(model_path)

print("\nSaved training history:")
print(history_path)

print("\nSaved test predictions:")
print(
    MODEL_DIR /
    "test_predictions_v2.npy"
)

print("\n" + "=" * 90)
print("WORLD MODEL V2 TRAINING COMPLETE")
print("=" * 90)