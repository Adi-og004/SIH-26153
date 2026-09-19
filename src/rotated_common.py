"""
Shared machinery for the rotated leave-one-day-out generalization experiment (Task 6).

Nothing here touches the frozen V4 reference. Every fold builds its own dataset,
fits its own scalers on its own TRAIN partition only, trains its own model, and
selects its own thresholds on its own VALIDATION partition only.

The V4 architecture and training configuration are copied verbatim from
src/train_latent_world_model_v4.py so the rotated experiment differs from the
frozen reference ONLY in which day is held out and in the fold-isolation fixes.
"""

from __future__ import annotations

import hashlib
import pickle
import subprocess
from pathlib import Path

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
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent

STATES_DIR = REPO_ROOT / "data" / "processed" / "states_v2"
FROZEN_BENCHMARK = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
FEATURE_SCHEMA_JSON = REPO_ROOT / "docs" / "v4_feature_schema.json"

RESULTS_DIR = REPO_ROOT / "results" / "rotated_generalization"
FOLD_DATA_DIR = REPO_ROOT / "data" / "processed" / "rotated_folds"
V4_DIR = REPO_ROOT / "models" / "rotated_v4"
LR_DIR = REPO_ROOT / "models" / "rotated_logistic_regression"

DAYS = {
    "Feb14": "Wednesday-14-02-2018",
    "Feb15": "Thursday-15-02-2018",
    "Feb28": "Wednesday-28-02-2018",
    "Mar02": "Friday-02-03-2018",
}
DAY_ORDER = ["Feb14", "Feb15", "Feb28", "Mar02"]

# ---- Frozen dataset construction constants (from prepare_multiday_benchmark.py) ----
VAL_RATIO = 0.15
SEQUENCE_LENGTH = 10
MAX_K = 5
WINDOW_SECONDS = 10

STATE_FEATURES = [
    "flow_count", "log_flow_count", "log_total_packets", "log_total_bytes",
    "window_packet_rate", "window_byte_rate", "syn_rate", "ack_rate",
    "rst_rate", "fin_rate", "psh_rate", "urg_rate", "mean_pkt_len",
    "std_pkt_len", "log_mean_flow_duration", "log_mean_fwd_iat",
    "log_mean_bwd_iat", "fwd_packet_ratio", "bwd_packet_ratio",
    "fwd_byte_ratio", "bwd_byte_ratio", "mean_down_up_ratio",
    "log_unique_dst_ports", "unique_protocols", "dst_port_entropy",
    "protocol_entropy",
]
INPUT_FEATURES = STATE_FEATURES + [f"{name}_rel_change" for name in STATE_FEATURES]

# ---- Frozen V4 training configuration (from train_latent_world_model_v4.py) ----
V4_CONFIG = {
    "input_size": 52, "output_size": 26, "hidden_size": 128, "latent_size": 32,
    "num_layers": 2, "dropout": 0.20,
    "random_seed": 42, "batch_size": 64,
    "learning_rate": 5e-4, "weight_decay": 1e-5, "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=5), reset each curriculum phase",
    "epochs": 120, "patience": 20,
    "state_loss": "SmoothL1Loss", "state_loss_weight": 1.0,
    "attack_loss": "BCEWithLogitsLoss(pos_weight=train_benign/train_attack)",
    "attack_loss_weight": 0.5,
    "max_k": 5, "gamma": 0.8,
    "curriculum": [(1, 1), (21, 2), (41, 3), (61, 4), (81, 5)],
    "grad_clip_max_norm": 1.0,
    "checkpoint_selection": "best VALIDATION F1 (threshold grid np.arange(0.05, 0.96, 0.01))",
    "early_stopping": "patience 20, only active in the final curriculum phase (K=5)",
}

LR_CONFIG = {
    "estimator": "sklearn.linear_model.LogisticRegression",
    "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4,
    "class_weight": "balanced", "fit_intercept": True, "random_state": 42,
}

POINT_HORIZONS = [1, 2, 3, 5]
TRAJECTORY_HORIZONS = [2, 3, 5]
LR_HORIZONS = [1, 2, 3, 4, 5]

# Evaluation threshold grid — identical to the frozen reference evaluation
EVAL_THRESHOLD_GRID = np.arange(0.01, 0.99, 0.01)
# Training-time checkpoint-selection grid — identical to the frozen training script
TRAIN_THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


# =============================================================
# UTILITIES
# =============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def select_threshold(scores, targets, grid=EVAL_THRESHOLD_GRID):
    """First strict maximum of F1 — the convention shared by every frozen baseline."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in grid:
        value = f1_score(targets, (scores >= threshold).astype(int), zero_division=0)
        if value > best_f1:
            best_f1, best_threshold = float(value), float(threshold)
    return best_threshold, best_f1


def standard_metrics(scores, targets, predictions, roc_auc_available=True):
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    roc_auc = (
        float(roc_auc_score(targets, scores))
        if roc_auc_available and len(np.unique(targets)) >= 2
        else None
    )
    return {
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "pos_samples": int(np.sum(targets)),
        "neg_samples": int(len(targets) - np.sum(targets)),
    }


# =============================================================
# DAY LOADING (verbatim logic from prepare_multiday_benchmark.py)
# =============================================================

def load_day(day_key: str) -> pd.DataFrame:
    filename = f"{DAYS[day_key]}_TrafficForML_CICFlowMeter_states_v2.csv"
    frame = pd.read_csv(STATES_DIR / filename, low_memory=False)

    frame["window_start"] = pd.to_datetime(frame["window_start"], errors="coerce")
    frame = frame.dropna(subset=["window_start"]).sort_values("window_start").reset_index(drop=True)
    frame["source_day"] = DAYS[day_key]
    frame["day_key"] = day_key

    for feature in STATE_FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame[STATE_FEATURES] = frame[STATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Relative-change features: per source day, chronological, before any split.
    state = frame[STATE_FEATURES].copy()
    delta = state.diff().fillna(0.0)
    previous = state.shift(1)
    relative = delta / (previous.abs() + 1e-8)
    relative.iloc[0] = 0.0
    relative = relative.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
    relative.columns = [f"{name}_rel_change" for name in STATE_FEATURES]

    return pd.concat([frame, relative], axis=1)


def build_episodes(frame: pd.DataFrame):
    """Contiguous runs of attack_present == 1 on one raw day."""
    labels = frame["attack_present"].astype(int).to_numpy()
    times = frame["window_start"].to_numpy()

    episodes = []
    index = 0
    while index < len(labels):
        if labels[index] == 1:
            end = index
            while end + 1 < len(labels) and labels[end + 1] == 1:
                end += 1
            windows = end - index + 1
            families = frame.loc[index:end, "dominant_attack"].value_counts()
            episodes.append(
                {
                    "episode_id": len(episodes) + 1,
                    "onset": pd.Timestamp(times[index]),
                    "offset": pd.Timestamp(times[end]),
                    "windows": windows,
                    "duration_sec": windows * WINDOW_SECONDS,
                    "families": "; ".join(f"{name}({count})" for name, count in families.items()),
                    "primary_family": families.index[0],
                    "cold_start_episode": index == 0,
                }
            )
            index = end + 1
        else:
            index += 1
    return episodes


def create_sequences(inputs, targets, frame):
    """Strict 10-second contiguity over inputs + MAX_K targets (frozen rule)."""
    X, Y, A_soft, A_binary, records = [], [], [], [], []

    timestamps = frame["window_start"].values
    attack_present = frame["attack_present"].values
    attack_ratio = frame["attack_ratio"].values
    dominant = frame["dominant_attack"].values
    source_days = frame["source_day"].values

    for start in range(len(frame) - SEQUENCE_LENGTH - MAX_K + 1):
        end = start + SEQUENCE_LENGTH
        block = timestamps[start:end + MAX_K]
        diffs = np.diff(block).astype("timedelta64[s]").astype(int)
        if not np.all(diffs == WINDOW_SECONDS):
            continue

        X.append(inputs[start:end])
        Y.append([targets[end + k] for k in range(MAX_K)])
        A_soft.append([float(attack_ratio[end + k]) for k in range(MAX_K)])
        A_binary.append([int(attack_present[end + k]) for k in range(MAX_K)])
        records.append(
            {
                "timestamp": timestamps[end],
                "attack_present": attack_present[end],
                "attack_ratio": attack_ratio[end],
                "dominant_attack": dominant[end],
                "source_day": source_days[end],
                "label_at_time_t": int(attack_present[end - 1]),
            }
        )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(A_soft, dtype=np.float32),
        np.asarray(A_binary, dtype=np.float32),
        pd.DataFrame(records),
    )


def build_fold(test_day: str, verbose=True):
    """
    Build one leave-one-day-out fold with strict isolation:
      - train  = first 85% (chronological) of each non-held-out day
      - val    = last 15% of each non-held-out day
      - test   = the entire held-out day
      - scalers fitted on the TRAIN partition only
    """
    train_days = [key for key in DAY_ORDER if key != test_day]

    train_frames, val_frames = [], []
    boundaries = []
    for key in train_days:
        frame = load_day(key)
        split = int(len(frame) * (1.0 - VAL_RATIO))
        train_frames.append(frame.iloc[:split].copy())
        val_frames.append(frame.iloc[split:].copy())
        boundaries.append(
            {
                "day_key": key,
                "total_windows": int(len(frame)),
                "train_windows": int(split),
                "validation_windows": int(len(frame) - split),
                "train_last_timestamp": str(frame["window_start"].iloc[split - 1]),
                "validation_first_timestamp": str(frame["window_start"].iloc[split]),
            }
        )

    test_frame = load_day(test_day)

    train_frame = pd.concat(train_frames, ignore_index=True)
    val_frame = pd.concat(val_frames, ignore_index=True)

    input_scaler = StandardScaler().fit(train_frame[INPUT_FEATURES])
    target_scaler = StandardScaler().fit(train_frame[STATE_FEATURES])

    def transform(frame):
        inputs = np.nan_to_num(
            input_scaler.transform(frame[INPUT_FEATURES]), nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)
        targets = np.nan_to_num(
            target_scaler.transform(frame[STATE_FEATURES]), nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)
        return inputs, targets

    splits = {}
    for name, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        inputs, targets = transform(frame)
        X, Y, A_soft, A_binary, meta = create_sequences(inputs, targets, frame)
        splits[name] = {"X": X, "Y": Y, "A_soft": A_soft, "A_binary": A_binary, "meta": meta}
        if verbose:
            print(f"    {name:<5}: X={X.shape}  attack(t+1)={int(A_binary[:, 0].sum())}  "
                  f"benign={int((A_binary[:, 0] == 0).sum())}")

    return {
        "test_day": test_day,
        "train_days": train_days,
        "splits": splits,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "boundaries": boundaries,
        "episodes": build_episodes(test_frame),
        "test_frame": test_frame,
    }


def persist_fold(fold):
    """Write fold arrays (data dir) and scalers (results dir); return provenance hashes."""
    test_day = fold["test_day"]
    data_dir = FOLD_DATA_DIR / f"fold_test_{test_day}"
    scaler_dir = RESULTS_DIR / f"fold_{test_day}"
    data_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)

    for name, split in fold["splits"].items():
        np.save(data_dir / f"X_{name}.npy", split["X"])
        np.save(data_dir / f"Y_multistep_{name}.npy", split["Y"])
        np.save(data_dir / f"A_soft_multistep_{name}.npy", split["A_soft"])
        np.save(data_dir / f"A_binary_multistep_{name}.npy", split["A_binary"])
        split["meta"].to_csv(data_dir / f"{name}_metadata.csv", index=False)

    with open(scaler_dir / "input_scaler.pkl", "wb") as handle:
        pickle.dump(fold["input_scaler"], handle)
    with open(scaler_dir / "target_scaler.pkl", "wb") as handle:
        pickle.dump(fold["target_scaler"], handle)
    (scaler_dir / "input_feature_names.txt").write_text("\n".join(INPUT_FEATURES) + "\n")

    return {
        "data_dir": str(data_dir.relative_to(REPO_ROOT)),
        "scaler_dir": str(scaler_dir.relative_to(REPO_ROOT)),
        "input_scaler_sha256": sha256_file(scaler_dir / "input_scaler.pkl"),
        "target_scaler_sha256": sha256_file(scaler_dir / "target_scaler.pkl"),
        "input_scaler_n_samples_seen": int(fold["input_scaler"].n_samples_seen_),
        "feature_names_sha256": sha256_file(scaler_dir / "input_feature_names.txt"),
        "X_test_sha256": sha256_file(data_dir / "X_test.npy"),
        "A_binary_test_sha256": sha256_file(data_dir / "A_binary_multistep_test.npy"),
    }


def load_fold_arrays(test_day: str):
    data_dir = FOLD_DATA_DIR / f"fold_test_{test_day}"
    splits = {}
    for name in ("train", "val", "test"):
        meta = pd.read_csv(data_dir / f"{name}_metadata.csv")
        meta["timestamp"] = pd.to_datetime(meta["timestamp"])
        splits[name] = {
            "X": np.load(data_dir / f"X_{name}.npy"),
            "Y": np.load(data_dir / f"Y_multistep_{name}.npy"),
            "A_soft": np.load(data_dir / f"A_soft_multistep_{name}.npy"),
            "A_binary": np.load(data_dir / f"A_binary_multistep_{name}.npy"),
            "meta": meta,
        }
    return splits


# =============================================================
# V4 ARCHITECTURE (verbatim from train_latent_world_model_v4.py)
# =============================================================

class LatentWorldModelV4(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, num_layers, dropout):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4), nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size),
        )
        self.gate_fc = nn.Linear(latent_size, latent_size)
        self.transition_norm = nn.LayerNorm(latent_size)

        self.decoder_fc1 = nn.Linear(latent_size, 64)
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(64, output_size)

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32), nn.ReLU(), nn.Dropout(0.10), nn.Linear(32, 1),
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        return self.encoder_fc(self.encoder_norm(output[:, -1, :]))

    def transition_step(self, z):
        candidate = self.transition_mlp(z)
        gate = torch.sigmoid(self.gate_fc(z))
        return self.transition_norm(gate * candidate + (1 - gate) * z)

    def decode(self, z):
        return self.decoder_fc2(self.decoder_activation(self.decoder_fc1(z)))

    def predict_attack(self, z):
        return self.attack_head(z).squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        z_next = self.transition_step(z)
        return self.decode(z_next), z_next, self.predict_attack(z_next), z

    def multi_step_forward(self, x, k_steps):
        z = self.encode(x)
        states, logits = [], []
        for _ in range(k_steps):
            z = self.transition_step(z)
            states.append(self.decode(z))
            logits.append(self.predict_attack(z))
        return states, logits

    def k_step_rollout(self, x, k):
        z = self.encode(x)
        probabilities = [torch.sigmoid(self.predict_attack(z))]
        for _ in range(k):
            z = self.transition_step(z)
            probabilities.append(torch.sigmoid(self.predict_attack(z)))
        return probabilities


def rollout_probabilities(model, X, k_max, device, batch_size=256):
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32)
    output = np.zeros((len(X), k_max + 1))
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            steps = model.k_step_rollout(tensor[start:end].to(device), k_max)
            for index in range(k_max + 1):
                output[start:end, index] = steps[index].cpu().numpy()
    return output
