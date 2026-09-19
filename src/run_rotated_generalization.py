"""
=============================================================
ROTATED HELD-OUT-DAY GENERALIZATION EXPERIMENT (TASK 6)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Leave-one-day-out over the four available source days. For every fold a NEW
V4 model and NEW Logistic Regression models are trained from scratch with
strict fold isolation.

    Fold 1: test Feb-14  train/val Feb-15, Feb-28, Mar-02
    Fold 2: test Feb-15  train/val Feb-14, Feb-28, Mar-02
    Fold 3: test Feb-28  train/val Feb-14, Feb-15, Mar-02
    Fold 4: test Mar-02  train/val Feb-14, Feb-15, Feb-28

These are EXPERIMENTAL models named V4_rotated_<day>. They never touch and never
replace the frozen V4 reference in models/latent_world_model_v4/ — Fold 4 is a
new reproduced experiment, not a substitute for the historical benchmark.

Isolation guarantees, enforced per fold:
    - the held-out day never appears in train or validation
    - within each training day, the split is chronological 85/15
    - both scalers are fitted on that fold's TRAIN partition only
    - the checkpoint is selected on VALIDATION F1 only
    - every reported threshold is selected on VALIDATION only
    - the held-out day is evaluated exactly once
    - no test label is used for training, early stopping or tuning

Usage:
    python src/run_rotated_generalization.py                        # everything
    python src/run_rotated_generalization.py --stages inventory,build
    python src/run_rotated_generalization.py --stages train_v4 --folds Feb14
    python src/run_rotated_generalization.py --stages evaluate,summarize
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import platform
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rotated_common import (  # noqa: E402
    DAY_ORDER, DAYS, EVAL_THRESHOLD_GRID, FOLD_DATA_DIR, INPUT_FEATURES,
    LR_CONFIG, LR_DIR, LR_HORIZONS, MAX_K, POINT_HORIZONS, REPO_ROOT,
    RESULTS_DIR, STATE_FEATURES, TRAIN_THRESHOLD_GRID, TRAJECTORY_HORIZONS,
    V4_CONFIG, V4_DIR, WINDOW_SECONDS, LatentWorldModelV4, build_episodes,
    build_fold, get_device, git_value, load_day, load_fold_arrays,
    persist_fold, rollout_probabilities, select_threshold, sha256_file,
    standard_metrics,
)

FIG_DIR = RESULTS_DIR / "figures"
SUMMARY_JSON = REPO_ROOT / "results" / "rotated_generalization_summary.json"
MASTER_CSV = RESULTS_DIR / "rotated_generalization_results.csv"

MODEL_PERSISTENCE = "persistence"
MODEL_LR = "logistic_regression"
MODEL_V4 = "v4_world_model"
MODELS = [MODEL_PERSISTENCE, MODEL_LR, MODEL_V4]

ONSET_LOOKBACK_WINDOWS = 12  # 120 s, same convention as Tasks 4 and 5


# =============================================================
# STAGE 1 — DAY INVENTORY
# =============================================================

def stage_inventory():
    print("\n" + "=" * 100)
    print("STAGE 1 — DAY INVENTORY")
    print("=" * 100)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for key in DAY_ORDER:
        frame = load_day(key)
        labels = frame["attack_present"].astype(int).to_numpy()
        episodes = build_episodes(frame)

        # Evaluable sequences for this day when it is the held-out test day
        raw_path = REPO_ROOT / "data" / "processed" / "states_v2" / (
            f"{DAYS[key]}_TrafficForML_CICFlowMeter_states_v2.csv"
        )
        families = frame.loc[labels == 1, "dominant_attack"].value_counts()

        observable = [ep for ep in episodes if not ep["cold_start_episode"]]

        rows.append(
            {
                "day_key": key,
                "source_day": DAYS[key],
                "state_file": str(raw_path.relative_to(REPO_ROOT)),
                "state_file_sha256": sha256_file(raw_path),
                "raw_windows": int(len(frame)),
                "window_seconds": WINDOW_SECONDS,
                "first_window": str(frame["window_start"].iloc[0]),
                "last_window": str(frame["window_start"].iloc[-1]),
                "positive_windows": int(labels.sum()),
                "negative_windows": int((labels == 0).sum()),
                "positive_prevalence": float(labels.mean()),
                "attack_episodes": len(episodes),
                "observable_onsets": len(observable),
                "observable_offsets": int(
                    sum(1 for ep in episodes
                        if ep["offset"] < frame["window_start"].iloc[-1])
                ),
                "cold_start_episodes": int(sum(1 for ep in episodes if ep["cold_start_episode"])),
                "attack_active_at_day_start": bool(labels[0] == 1),
                "episode_windows": "; ".join(str(ep["windows"]) for ep in episodes),
                "episode_durations_sec": "; ".join(str(ep["duration_sec"]) for ep in episodes),
                "min_episode_windows": min((ep["windows"] for ep in episodes), default=0),
                "max_episode_windows": max((ep["windows"] for ep in episodes), default=0),
                "attack_families": "; ".join(f"{name}({count})" for name, count in families.items()),
                "n_attack_families": int(len(families)),
                "first_attack_timestamp": str(frame.loc[labels == 1, "window_start"].min())
                if labels.sum() else "",
                "last_attack_timestamp": str(frame.loc[labels == 1, "window_start"].max())
                if labels.sum() else "",
                "families_source": "dominant_attack column of states_v2 (explicit dataset label)",
            }
        )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(RESULTS_DIR / "day_inventory.csv", index=False)
    print(inventory[["day_key", "raw_windows", "positive_windows", "positive_prevalence",
                     "attack_episodes", "observable_onsets", "cold_start_episodes",
                     "n_attack_families", "attack_families"]].to_string(index=False))
    print(f"\nWrote {(RESULTS_DIR / 'day_inventory.csv').relative_to(REPO_ROOT)}")
    return inventory


# =============================================================
# STAGE 2 — BUILD FOLDS
# =============================================================

def stage_build(fold_keys):
    print("\n" + "=" * 100)
    print("STAGE 2 — BUILD FOLD DATASETS (scalers fitted on TRAIN only)")
    print("=" * 100)

    rows = []
    for test_day in fold_keys:
        print(f"\n  Fold: held-out day = {test_day} ({DAYS[test_day]})")
        fold = build_fold(test_day)
        provenance = persist_fold(fold)

        splits = fold["splits"]
        for name in ("train", "val", "test"):
            labels = splits[name]["A_binary"][:, 0]
            rows.append(
                {
                    "test_day": test_day,
                    "partition": name,
                    "days": ";".join(fold["train_days"]) if name != "test" else test_day,
                    "sequences": int(len(labels)),
                    "attack_windows_t1": int(labels.sum()),
                    "benign_windows_t1": int((labels == 0).sum()),
                    "positive_prevalence_t1": float(labels.mean()),
                    "first_target_timestamp": str(splits[name]["meta"]["timestamp"].min()),
                    "last_target_timestamp": str(splits[name]["meta"]["timestamp"].max()),
                    **({"input_scaler_sha256": provenance["input_scaler_sha256"],
                        "target_scaler_sha256": provenance["target_scaler_sha256"],
                        "input_scaler_n_samples_seen": provenance["input_scaler_n_samples_seen"]}
                       if name == "train" else {}),
                    "chronological_boundaries": json.dumps(fold["boundaries"]) if name == "train" else "",
                }
            )

        episodes = pd.DataFrame(fold["episodes"])
        episodes["test_day"] = test_day
        episodes.to_csv(RESULTS_DIR / f"fold_{test_day}" / "episodes.csv", index=False)

    inventory = pd.DataFrame(rows)
    inventory.to_csv(RESULTS_DIR / "fold_inventory.csv", index=False)
    print(f"\nWrote {(RESULTS_DIR / 'fold_inventory.csv').relative_to(REPO_ROOT)}")
    print(inventory[["test_day", "partition", "sequences", "attack_windows_t1",
                     "positive_prevalence_t1"]].to_string(index=False))
    return inventory


# =============================================================
# STAGE 3 — TRAIN ROTATED V4
# =============================================================

def train_v4_fold(test_day, device):
    """
    Fold-isolated V4 training. Architecture and hyperparameters are verbatim from
    src/train_latent_world_model_v4.py; the held-out day is never seen.
    """
    print("\n" + "-" * 100)
    print(f"TRAINING V4_rotated_{test_day}  (held-out day: {DAYS[test_day]})")
    print("-" * 100)

    random.seed(V4_CONFIG["random_seed"])
    np.random.seed(V4_CONFIG["random_seed"])
    torch.manual_seed(V4_CONFIG["random_seed"])

    splits = load_fold_arrays(test_day)
    X_train, Y_train, A_soft_train, A_bin_train = (
        splits["train"][key] for key in ("X", "Y", "A_soft", "A_binary")
    )
    X_val, Y_val, A_soft_val, A_bin_val = (
        splits["val"][key] for key in ("X", "Y", "A_soft", "A_binary")
    )

    benign = int((A_bin_train[:, 0] == 0).sum())
    attack = int((A_bin_train[:, 0] == 1).sum())
    if attack == 0:
        raise ValueError(f"fold {test_day}: no attack samples in training")
    positive_weight = benign / attack
    print(f"  train sequences={len(X_train)}  attack={attack}  benign={benign}  "
          f"pos_weight={positive_weight:.4f}")

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train), torch.tensor(Y_train), torch.tensor(A_soft_train)
        ),
        batch_size=V4_CONFIG["batch_size"], shuffle=True,
    )

    model = LatentWorldModelV4(
        input_size=V4_CONFIG["input_size"], hidden_size=V4_CONFIG["hidden_size"],
        latent_size=V4_CONFIG["latent_size"], output_size=V4_CONFIG["output_size"],
        num_layers=V4_CONFIG["num_layers"], dropout=V4_CONFIG["dropout"],
    ).to(device)

    state_criterion = nn.SmoothL1Loss()
    attack_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=V4_CONFIG["learning_rate"], weight_decay=V4_CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    X_val_tensor = torch.tensor(X_val)
    Y_val_tensor = torch.tensor(Y_val)
    A_val_tensor = torch.tensor(A_soft_val)
    y_val_binary = A_bin_val[:, 0]

    def current_k(epoch):
        value = 1
        for start_epoch, k_value in V4_CONFIG["curriculum"]:
            if epoch >= start_epoch:
                value = k_value
        return value

    best_f1, best_state, best_threshold = -1.0, None, 0.5
    stale = 0
    history = []

    for epoch in range(1, V4_CONFIG["epochs"] + 1):
        k_steps = current_k(epoch)

        for start_epoch, k_value in V4_CONFIG["curriculum"]:
            if epoch == start_epoch:
                stale = 0
                best_f1 = -1.0
                for group in optimizer.param_groups:
                    group["lr"] = V4_CONFIG["learning_rate"]
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=5
                )

        model.train()
        total_loss, seen = 0.0, 0
        for X_batch, Y_batch, A_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            A_batch = A_batch.to(device)

            optimizer.zero_grad()
            states, logits = model.multi_step_forward(X_batch, k_steps)

            state_loss, attack_loss = 0.0, 0.0
            for k in range(k_steps):
                weight = V4_CONFIG["gamma"] ** k
                state_loss = state_loss + weight * state_criterion(
                    states[k], Y_batch[:, k, :].contiguous()
                )
                attack_loss = attack_loss + weight * attack_criterion(
                    logits[k], A_batch[:, k].contiguous()
                )

            loss = (V4_CONFIG["state_loss_weight"] * state_loss
                    + V4_CONFIG["attack_loss_weight"] * attack_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), V4_CONFIG["grad_clip_max_norm"])
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            seen += X_batch.size(0)

        # ----- VALIDATION ONLY (held-out day untouched) -----
        model.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(X_val_tensor), 256):
                end = min(start + 256, len(X_val_tensor))
                _, _, logits_val, _ = model(X_val_tensor[start:end].to(device))
                probabilities.extend(torch.sigmoid(logits_val).cpu().numpy())
        probabilities = np.asarray(probabilities)

        threshold, val_f1 = select_threshold(probabilities, y_val_binary, TRAIN_THRESHOLD_GRID)
        scheduler.step(val_f1)

        history.append(
            {
                "epoch": epoch, "curriculum_k": k_steps,
                "train_loss": total_loss / seen, "val_f1": val_f1,
                "val_threshold": threshold,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        if val_f1 > best_f1:
            best_f1, best_threshold = val_f1, threshold
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  K={k_steps}  loss={total_loss / seen:.4f}  "
                  f"val_F1={val_f1:.4f}  best={best_f1:.4f}")

        if stale >= V4_CONFIG["patience"] and k_steps == MAX_K:
            print(f"    early stop at epoch {epoch} (final curriculum phase)")
            break

    if best_state is None:
        raise RuntimeError(f"fold {test_day}: no checkpoint recorded")
    model.load_state_dict(best_state)

    fold_dir = V4_DIR / f"fold_test_{test_day}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / f"v4_rotated_{test_day}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "LatentWorldModelV4",
            "experiment": f"V4_rotated_{test_day}",
            "is_frozen_reference": False,
            "held_out_day": DAYS[test_day],
            "train_days": [DAYS[key] for key in DAY_ORDER if key != test_day],
            "input_size": V4_CONFIG["input_size"], "hidden_size": V4_CONFIG["hidden_size"],
            "latent_size": V4_CONFIG["latent_size"], "output_size": V4_CONFIG["output_size"],
            "num_layers": V4_CONFIG["num_layers"], "sequence_length": X_train.shape[1],
            "max_k": MAX_K, "gamma": V4_CONFIG["gamma"], "curriculum": V4_CONFIG["curriculum"],
            "positive_class_weight": positive_weight,
            "best_validation_f1": best_f1,
            "validation_selected_threshold_training": best_threshold,
            "transition_type": "deterministic",
            "test_labels_used_in_training": False,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    print(f"  best validation F1 = {best_f1:.4f} (epoch "
          f"{max(history, key=lambda row: row['val_f1'])['epoch']})")
    print(f"  saved {checkpoint_path.relative_to(REPO_ROOT)}")

    return checkpoint_path, best_f1, positive_weight, history


def stage_train_v4(fold_keys):
    print("\n" + "=" * 100)
    print("STAGE 3 — TRAIN ROTATED V4 MODELS (one per held-out day)")
    print("=" * 100)
    device = get_device()
    print(f"Device: {device}")

    for test_day in fold_keys:
        checkpoint_path, best_f1, positive_weight, _ = train_v4_fold(test_day, device)

        provenance = {
            "experiment": f"V4_rotated_{test_day}",
            "is_frozen_reference": False,
            "held_out_day": DAYS[test_day],
            "train_days": [DAYS[key] for key in DAY_ORDER if key != test_day],
            "validation_days": [DAYS[key] for key in DAY_ORDER if key != test_day],
            "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "feature_schema_sha256": sha256_file(
                RESULTS_DIR / f"fold_{test_day}" / "input_feature_names.txt"
            ),
            "input_scaler_sha256": sha256_file(
                RESULTS_DIR / f"fold_{test_day}" / "input_scaler.pkl"
            ),
            "target_scaler_sha256": sha256_file(
                RESULTS_DIR / f"fold_{test_day}" / "target_scaler.pkl"
            ),
            "training_config": {k: (str(v) if isinstance(v, list) else v)
                                for k, v in V4_CONFIG.items()},
            "positive_class_weight": positive_weight,
            "best_validation_f1": best_f1,
            "seed": V4_CONFIG["random_seed"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "test_labels_used": False,
        }
        with open(checkpoint_path.parent / "provenance.json", "w") as handle:
            json.dump(provenance, handle, indent=2)
            handle.write("\n")


# =============================================================
# STAGE 4 — TRAIN ROTATED LOGISTIC REGRESSION
# =============================================================

def stage_train_lr(fold_keys):
    print("\n" + "=" * 100)
    print("STAGE 4 — TRAIN ROTATED LOGISTIC REGRESSION MODELS")
    print("=" * 100)

    for test_day in fold_keys:
        print(f"\n  Fold {test_day}:")
        splits = load_fold_arrays(test_day)
        S_train = splits["train"]["X"][:, -1, :]
        A_train = splits["train"]["A_binary"]

        fold_dir = LR_DIR / f"fold_test_{test_day}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        records = {}
        for k in LR_HORIZONS:
            model = LogisticRegression(
                penalty=LR_CONFIG["penalty"], C=LR_CONFIG["C"], solver=LR_CONFIG["solver"],
                max_iter=LR_CONFIG["max_iter"], tol=LR_CONFIG["tol"],
                class_weight=LR_CONFIG["class_weight"],
                fit_intercept=LR_CONFIG["fit_intercept"],
                random_state=LR_CONFIG["random_state"],
            )
            model.fit(S_train, A_train[:, k - 1])

            path = fold_dir / f"logistic_regression_k{k}.pkl"
            with open(path, "wb") as handle:
                pickle.dump(model, handle)

            records[f"k{k}"] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "n_iter": int(np.max(model.n_iter_)),
                "converged": bool(np.max(model.n_iter_) < LR_CONFIG["max_iter"]),
                "n_features_in": int(model.n_features_in_),
            }
            print(f"    K={k}: converged in {records[f'k{k}']['n_iter']} iterations")

        with open(fold_dir / "provenance.json", "w") as handle:
            json.dump(
                {
                    "experiment": f"LR_rotated_{test_day}",
                    "is_frozen_reference": False,
                    "held_out_day": DAYS[test_day],
                    "train_days": [DAYS[key] for key in DAY_ORDER if key != test_day],
                    "input_representation": "S_t = X[:, -1, :] (52-dim, fold-specific scaler)",
                    "model_config": LR_CONFIG,
                    "models": records,
                    "feature_schema_sha256": sha256_file(
                        RESULTS_DIR / f"fold_{test_day}" / "input_feature_names.txt"
                    ),
                    "input_scaler_sha256": sha256_file(
                        RESULTS_DIR / f"fold_{test_day}" / "input_scaler.pkl"
                    ),
                    "seed": LR_CONFIG["random_state"],
                    "python": platform.python_version(),
                    "sklearn": __import__("sklearn").__version__,
                    "git_commit": git_value("rev-parse", "HEAD"),
                    "test_labels_used": False,
                },
                handle, indent=2,
            )
            handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotated held-out-day generalization experiment.")
    parser.add_argument("--stages", default="inventory,build,train_v4,train_lr,evaluate,summarize")
    parser.add_argument("--folds", default=",".join(DAY_ORDER))
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    fold_keys = [key.strip() for key in args.folds.split(",") if key.strip()]

    print("=" * 100)
    print("ROTATED HELD-OUT-DAY GENERALIZATION EXPERIMENT")
    print("=" * 100)
    print(f"Stages: {stages}")
    print(f"Folds : {fold_keys}")
    print("The frozen V4 reference is never read for modification and never written to.")

    if "inventory" in stages:
        stage_inventory()
    if "build" in stages:
        stage_build(fold_keys)
    if "train_v4" in stages:
        stage_train_v4(fold_keys)
    if "train_lr" in stages:
        stage_train_lr(fold_keys)
    if "evaluate" in stages or "summarize" in stages:
        from rotated_evaluate import stage_evaluate, stage_summarize
        if "evaluate" in stages:
            stage_evaluate(fold_keys)
        if "summarize" in stages:
            stage_summarize(fold_keys, skip_plots=args.skip_plots)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
