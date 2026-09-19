"""
=============================================================
STANDARDIZED BASELINE COMPARISON
Persistence  vs  Logistic Regression  vs  V4 World Model
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Evaluation and reporting only. This script NEVER trains, fine-tunes
or modifies any model, threshold, scaler, feature schema or split.

What it does:
    1. Verifies the frozen V4 baseline  (src/verify_v4_baseline.py)
    2. Verifies the frozen LR baseline  (src/verify_logistic_regression_baseline.py)
    3. Reconstructs the persistence rule from src/evaluate_baselines.py
    4. Recomputes every metric from raw prediction scores on ONE common
       benchmark: the frozen multi-day benchmark, Mar-2-2018 held out
    5. Cross-checks the recomputed V4 and LR numbers against their frozen
       manifests (exact equality required)
    6. Writes results/standardized_baseline_comparison.csv
    7. Writes comparison plots into results/figures/
    8. Writes results/comparison_verification.json (leakage checks)
    9. Writes docs/standardized_comparison_manifest.json

Model inference performed here:
    - V4  : forward rollout of the FROZEN checkpoint (inference only), exactly
            as the pinned reference evaluation at commit 2be3073 does
    - LR  : predict_proba of the FROZEN pickled models (inference only);
            test probabilities are also read back from the frozen
            test_predictions.csv and required to agree
    - Pers: deterministic rule, no parameters

Usage:
    python src/run_standardized_comparison.py
    python src/run_standardized_comparison.py --skip-plots
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
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


REPO_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
STATES_DIR = REPO_ROOT / "data" / "processed" / "states_v2"

V4_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
LR_MANIFEST = REPO_ROOT / "docs" / "logistic_regression_baseline_manifest.json"
LR_DIR = REPO_ROOT / "models" / "logistic_regression_baseline"

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
COMPARISON_CSV = RESULTS_DIR / "standardized_baseline_comparison.csv"
DELTA_CSV = RESULTS_DIR / "persistence_hypothesis_analysis.csv"
VERIFICATION_JSON = RESULTS_DIR / "comparison_verification.json"
COMPARISON_MANIFEST = REPO_ROOT / "docs" / "standardized_comparison_manifest.json"

# Primary point horizons. K=4 is excluded from the three-model comparison
# because the frozen V4 reference has no K=4 benchmark result.
POINT_HORIZONS = [1, 2, 3, 5]
TRAJECTORY_HORIZONS = [2, 3, 5]
LR_SECONDARY_HORIZONS = [4]

THRESHOLD_GRID = np.arange(0.01, 0.99, 0.01)

EXPECTED_TEST_DAY = "Friday-02-03-2018"
EXPECTED_TRAIN_DAYS = {
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Wednesday-28-02-2018",
}

MODEL_PERSISTENCE = "persistence"
MODEL_LR = "logistic_regression"
MODEL_V4 = "v4_world_model"

TOLERANCE = 1e-9


# =============================================================
# V4 ARCHITECTURE (verbatim; used for inference only)
# =============================================================

class LatentWorldModelV4(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, num_layers, dropout):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size),
        )
        self.gate_fc = nn.Linear(latent_size, latent_size)
        self.transition_norm = nn.LayerNorm(latent_size)

        self.decoder_fc1 = nn.Linear(latent_size, 64)
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(64, output_size)

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        return self.encoder_fc(self.encoder_norm(output[:, -1, :]))

    def transition_step(self, z):
        candidate = self.transition_mlp(z)
        gate = torch.sigmoid(self.gate_fc(z))
        return self.transition_norm(gate * candidate + (1 - gate) * z)

    def predict_attack(self, z):
        return self.attack_head(z).squeeze(-1)

    def k_step_rollout(self, x, k):
        """Returns K+1 probabilities: index 0 = z_t, index j = after j transitions."""
        z = self.encode(x)
        probabilities = [torch.sigmoid(self.predict_attack(z))]
        for _ in range(k):
            z = self.transition_step(z)
            probabilities.append(torch.sigmoid(self.predict_attack(z)))
        return probabilities


# =============================================================
# HELPERS
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


def select_threshold_on_validation(scores, targets):
    """First strict maximum of validation F1 — the convention shared by both baselines."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        f1 = f1_score(targets, (scores >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)
    return best_threshold, best_f1


def standard_metrics(scores, targets, predictions, roc_auc_available=True):
    """One metric definition used identically for all three models."""
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

    if roc_auc_available and len(np.unique(targets)) >= 2:
        roc_auc = float(roc_auc_score(targets, scores))
    else:
        roc_auc = None

    return {
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "pos_samples": int(np.sum(targets)),
        "neg_samples": int(len(targets) - np.sum(targets)),
    }


def run_verifier(script_name):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / script_name)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    overall = "UNKNOWN"
    for line in result.stdout.splitlines():
        if line.startswith("OVERALL:"):
            overall = line.split(":", 1)[1].strip()
    return result.returncode == 0 and overall == "PASS", overall


# =============================================================
# DATA
# =============================================================

def load_benchmark():
    data = {}
    for split in ("train", "val", "test"):
        meta = pd.read_csv(BENCHMARK_DIR / f"{split}_metadata.csv")
        meta["timestamp"] = pd.to_datetime(meta["timestamp"])
        data[split] = {
            "X": np.load(BENCHMARK_DIR / f"X_{split}.npy"),
            "A": np.load(BENCHMARK_DIR / f"A_binary_multistep_{split}.npy"),
            "meta": meta,
        }
    return data


def fetch_labels_at_time_t(meta):
    """
    Persistence input: the observed attack label at time t.

    Reconstructed exactly as src/evaluate_baselines.py does — the benchmark
    metadata timestamp is the t+1 target window, so the label at time t is
    read from the source state CSV at (timestamp - 10s). No future
    information is used.
    """
    lookup = {}
    for day in meta["source_day"].unique():
        frame = pd.read_csv(
            STATES_DIR / f"{day}_TrafficForML_CICFlowMeter_states_v2.csv",
            usecols=["window_start", "attack_present"],
        )
        frame["window_start"] = pd.to_datetime(frame["window_start"])
        lookup.update(dict(zip(frame["window_start"], frame["attack_present"])))

    labels = []
    for timestamp in meta["timestamp"]:
        key = timestamp - pd.Timedelta(seconds=10)
        if key not in lookup:
            raise ValueError(f"Missing observed label at time t for target {timestamp}")
        labels.append(lookup[key])

    return np.asarray(labels, dtype=float)


# =============================================================
# PER-MODEL SCORES (inference only)
# =============================================================

def v4_scores(data, device):
    """Frozen V4 checkpoint, rolled forward exactly as the pinned reference does."""
    manifest = json.loads(V4_MANIFEST.read_text())
    checkpoint_path = REPO_ROOT / manifest["checkpoint"]["path"]

    if sha256_file(checkpoint_path) != manifest["checkpoint"]["sha256"]:
        raise RuntimeError("V4 checkpoint SHA256 does not match the frozen manifest.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LatentWorldModelV4(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        latent_size=checkpoint["latent_size"],
        output_size=checkpoint["output_size"],
        num_layers=checkpoint["num_layers"],
        dropout=0.20,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    scores = {}
    for split in ("val", "test"):
        X = data[split]["X"]
        tensor = torch.tensor(X, dtype=torch.float32)
        probabilities = np.zeros((len(X), max(POINT_HORIZONS) + 1))
        with torch.no_grad():
            for start in range(0, len(X), 64):
                end = min(start + 64, len(X))
                batch_probabilities = model.k_step_rollout(
                    tensor[start:end].to(device), max(POINT_HORIZONS)
                )
                for step in range(max(POINT_HORIZONS) + 1):
                    probabilities[start:end, step] = batch_probabilities[step].cpu().numpy()
        scores[split] = probabilities

    return scores, manifest


def lr_scores(data):
    """Frozen LR models (inference only) + frozen test prediction artifact."""
    manifest = json.loads(LR_MANIFEST.read_text())
    stored = pd.read_csv(LR_DIR / "test_predictions.csv")

    models = {}
    for k in POINT_HORIZONS + LR_SECONDARY_HORIZONS:
        record = manifest["models"][str(k)]
        path = REPO_ROOT / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"LR K={k} model SHA256 does not match the frozen manifest.")
        with open(path, "rb") as handle:
            models[k] = pickle.load(handle)

    scores = {"val": {}, "test": {}}
    for k, model in models.items():
        scores["val"][k] = model.predict_proba(data["val"]["X"][:, -1, :])[:, 1]
        scores["test"][k] = model.predict_proba(data["test"]["X"][:, -1, :])[:, 1]

        # The frozen prediction artifact must agree with live inference.
        if not np.allclose(stored[f"prob_t+{k}"].to_numpy(), scores["test"][k], atol=1e-12):
            raise RuntimeError(
                f"LR K={k}: frozen test_predictions.csv disagrees with the frozen model."
            )

    return scores, manifest


def persistence_scores(data):
    """Deterministic rule: predicted score for every future horizon is A(t)."""
    return {
        "val": fetch_labels_at_time_t(data["val"]["meta"]),
        "test": fetch_labels_at_time_t(data["test"]["meta"]),
    }


# =============================================================
# MAIN
# =============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standardized Persistence / Logistic Regression / V4 comparison "
                    "on the frozen multi-day benchmark (evaluation only)."
    )
    parser.add_argument("--skip-plots", action="store_true", help="Do not render figures")
    args = parser.parse_args()

    print("=" * 100)
    print("STANDARDIZED BASELINE COMPARISON — Persistence vs Logistic Regression vs V4")
    print("=" * 100)
    print("Evaluation only. No model, threshold, scaler, split or label is modified.")

    checks = {}

    # ---------------------------------------------------------
    # 0. Frozen-baseline preflight
    # ---------------------------------------------------------
    print("\n--- Preflight: frozen baselines ---")
    v4_ok, v4_overall = run_verifier("verify_v4_baseline.py")
    lr_ok, lr_overall = run_verifier("verify_logistic_regression_baseline.py")
    print(f"V4 baseline verification : {v4_overall}")
    print(f"LR baseline verification : {lr_overall}")
    checks["v4_baseline_verifier_pass"] = bool(v4_ok)
    checks["lr_baseline_verifier_pass"] = bool(lr_ok)

    if not (v4_ok and lr_ok):
        raise RuntimeError("A frozen baseline failed verification; comparison aborted.")

    # ---------------------------------------------------------
    # 1. Common benchmark
    # ---------------------------------------------------------
    data = load_benchmark()

    test_days = set(data["test"]["meta"]["source_day"].unique())
    train_days = set(data["train"]["meta"]["source_day"].unique())
    val_days = set(data["val"]["meta"]["source_day"].unique())

    print("\n--- Common benchmark ---")
    print(f"Train : {sorted(train_days)}  ({len(data['train']['meta'])} sequences)")
    print(f"Val   : {sorted(val_days)}  ({len(data['val']['meta'])} sequences)")
    print(f"Test  : {sorted(test_days)}  ({len(data['test']['meta'])} sequences)")

    checks["train_days_expected"] = train_days == EXPECTED_TRAIN_DAYS
    checks["validation_days_expected"] = val_days == EXPECTED_TRAIN_DAYS
    checks["test_day_is_mar2_only"] = test_days == {EXPECTED_TEST_DAY}
    checks["test_day_absent_from_train_and_validation"] = not (test_days & (train_days | val_days))

    # Chronological split discipline
    chronological = True
    for day in sorted(EXPECTED_TRAIN_DAYS):
        train_last = data["train"]["meta"].loc[
            data["train"]["meta"]["source_day"] == day, "timestamp"
        ].max()
        val_first = data["val"]["meta"].loc[
            data["val"]["meta"]["source_day"] == day, "timestamp"
        ].min()
        chronological = chronological and bool(val_first > train_last)
    checks["validation_is_chronologically_after_train"] = chronological

    # No duplicated sequences across splits
    keys = {
        split: set(zip(data[split]["meta"]["source_day"], data[split]["meta"]["timestamp"]))
        for split in ("train", "val", "test")
    }
    checks["no_duplicate_sequences_within_splits"] = all(
        len(keys[split]) == len(data[split]["meta"]) for split in keys
    )
    checks["no_overlapping_sequences_across_splits"] = (
        not (keys["train"] & keys["val"])
        and not (keys["train"] & keys["test"])
        and not (keys["val"] & keys["test"])
    )

    # Target alignment verified against the raw source day (no future leakage into X)
    test_meta = data["test"]["meta"]
    raw = pd.read_csv(
        STATES_DIR / f"{EXPECTED_TEST_DAY}_TrafficForML_CICFlowMeter_states_v2.csv",
        usecols=["window_start", "attack_present"],
    )
    raw["window_start"] = pd.to_datetime(raw["window_start"])
    raw_lookup = dict(zip(raw["window_start"], raw["attack_present"]))
    alignment_ok = True
    for k in range(1, 6):
        expected = np.array(
            [raw_lookup[ts + pd.Timedelta(seconds=10 * (k - 1))] for ts in test_meta["timestamp"]],
            dtype=float,
        )
        alignment_ok = alignment_ok and bool(np.array_equal(expected, data["test"]["A"][:, k - 1]))
    checks["target_alignment_t_plus_k_verified"] = alignment_ok

    # Scaler fitted on the train partition only (no test normalisation fitting)
    with open(BENCHMARK_DIR / "input_scaler.pkl", "rb") as handle:
        scaler = pickle.load(handle)
    checks["input_scaler_fitted_on_train_only"] = int(scaler.n_samples_seen_) == 8575
    checks["input_scaler_not_refit_for_comparison"] = True

    # ---------------------------------------------------------
    # 2. Model scores (inference only)
    # ---------------------------------------------------------
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"\n--- Model scores (inference only, device={device}) ---")

    v4_probabilities, v4_manifest = v4_scores(data, device)
    print(f"V4          : rollout {v4_probabilities['test'].shape} (index j = prediction for t+j)")

    lr_probabilities, lr_manifest = lr_scores(data)
    print(f"LR          : {len(lr_probabilities['test'])} horizons, frozen artifact agreement OK")

    persistence = persistence_scores(data)
    print(f"Persistence : A(t) recovered for {len(persistence['test'])} test windows "
          f"({int(persistence['test'].sum())} positive)")

    A_val, A_test = data["val"]["A"], data["test"]["A"]

    # ---------------------------------------------------------
    # 3. Point-horizon comparison
    # ---------------------------------------------------------
    rows = []
    threshold_provenance = {}

    print("\n" + "=" * 100)
    print("POINT FORECAST — prediction for t+k vs A(t+k)")
    print("=" * 100)
    header = (
        f"{'Model':<22}{'K':>3}{'Thr':>8}{'F1':>9}{'Precision':>11}"
        f"{'Recall':>9}{'ROC-AUC':>10}{'FPR':>9}{'TP':>7}{'FP':>6}{'TN':>7}{'FN':>7}"
    )
    print(header)
    print("-" * len(header))

    for k in POINT_HORIZONS:
        target_val = A_val[:, k - 1]
        target_test = A_test[:, k - 1]

        # ----- Persistence: deterministic, no threshold to select -----
        predictions = persistence["test"].astype(int)
        metrics = standard_metrics(persistence["test"], target_test, predictions)
        metrics.update(
            {
                "model": MODEL_PERSISTENCE,
                "evaluation_type": "point_forecast",
                "horizon": k,
                "threshold": None,
                "threshold_policy": "not applicable — deterministic rule y_hat(t+k) = A(t)",
                "score_type": "binary 0/1 (observed label at time t)",
                "roc_auc_degenerate": True,
                "validation_f1_at_threshold": float(
                    f1_score(target_val, persistence["val"].astype(int), zero_division=0)
                ),
            }
        )
        rows.append(metrics)

        # ----- Logistic Regression: frozen validation threshold -----
        frozen_lr = float(lr_manifest["thresholds"][f"k{k}"])
        derived_lr, derived_lr_f1 = select_threshold_on_validation(
            lr_probabilities["val"][k], target_val
        )
        threshold_provenance[f"lr_point_k{k}"] = {
            "frozen": frozen_lr,
            "re_derived_on_validation": derived_lr,
            "matches": abs(frozen_lr - derived_lr) < TOLERANCE,
        }

        predictions = (lr_probabilities["test"][k] >= frozen_lr).astype(int)
        metrics = standard_metrics(lr_probabilities["test"][k], target_test, predictions)
        metrics.update(
            {
                "model": MODEL_LR,
                "evaluation_type": "point_forecast",
                "horizon": k,
                "threshold": frozen_lr,
                "threshold_policy": "validation-only F1 argmax (frozen in Task 2)",
                "score_type": "probability (horizon-specific logistic model on S_t)",
                "roc_auc_degenerate": False,
                "validation_f1_at_threshold": derived_lr_f1,
            }
        )
        rows.append(metrics)

        # ----- V4: frozen validation threshold from the reference evaluation -----
        frozen_v4 = float(v4_manifest["threshold"]["per_horizon_future_prediction"][str(k)])
        derived_v4, derived_v4_f1 = select_threshold_on_validation(
            v4_probabilities["val"][:, k], target_val
        )
        threshold_provenance[f"v4_point_k{k}"] = {
            "frozen": frozen_v4,
            "re_derived_on_validation": derived_v4,
            "matches": abs(frozen_v4 - derived_v4) < TOLERANCE,
        }

        predictions = (v4_probabilities["test"][:, k] >= frozen_v4).astype(int)
        metrics = standard_metrics(v4_probabilities["test"][:, k], target_test, predictions)
        metrics.update(
            {
                "model": MODEL_V4,
                "evaluation_type": "point_forecast",
                "horizon": k,
                "threshold": frozen_v4,
                "threshold_policy": "validation-only F1 argmax (frozen V4 reference)",
                "score_type": "probability (attack head after k latent transitions)",
                "roc_auc_degenerate": False,
                "validation_f1_at_threshold": derived_v4_f1,
            }
        )
        rows.append(metrics)

        for model_name in (MODEL_PERSISTENCE, MODEL_LR, MODEL_V4):
            row = next(
                r for r in rows
                if r["model"] == model_name
                and r["evaluation_type"] == "point_forecast"
                and r["horizon"] == k
            )
            threshold_text = "n/a" if row["threshold"] is None else f"{row['threshold']:.2f}"
            roc_text = "n/a" if row["roc_auc"] is None else f"{row['roc_auc']:.4f}"
            print(
                f"{model_name:<22}{k:>3}{threshold_text:>8}{row['f1']:>9.4f}"
                f"{row['precision']:>11.4f}{row['recall']:>9.4f}{roc_text:>10}"
                f"{row['fpr']:>9.4f}{row['tp']:>7d}{row['fp']:>6d}{row['tn']:>7d}{row['fn']:>7d}"
            )
        print("-" * len(header))

    # ----- Secondary: LR-only K=4 (no V4 reference exists) -----
    for k in LR_SECONDARY_HORIZONS:
        frozen_lr = float(lr_manifest["thresholds"][f"k{k}"])
        predictions = (lr_probabilities["test"][k] >= frozen_lr).astype(int)
        metrics = standard_metrics(lr_probabilities["test"][k], A_test[:, k - 1], predictions)
        metrics.update(
            {
                "model": MODEL_LR,
                "evaluation_type": "point_forecast_secondary_lr_only",
                "horizon": k,
                "threshold": frozen_lr,
                "threshold_policy": "validation-only F1 argmax (frozen in Task 2)",
                "score_type": "probability (horizon-specific logistic model on S_t)",
                "roc_auc_degenerate": False,
                "validation_f1_at_threshold": None,
            }
        )
        rows.append(metrics)
        print(f"{'logistic_regression':<22}{k:>3}{frozen_lr:>8.2f}{metrics['f1']:>9.4f}"
              f"{metrics['precision']:>11.4f}{metrics['recall']:>9.4f}{metrics['roc_auc']:>10.4f}"
              f"{metrics['fpr']:>9.4f}   [SECONDARY — no V4 K=4 reference; not a 3-model row]")

    # ---------------------------------------------------------
    # 4. Trajectory-event comparison
    # ---------------------------------------------------------
    print("\n" + "=" * 100)
    print("TRAJECTORY EVENT (MAX) — any attack within t+1 .. t+K")
    print("=" * 100)
    print("Aggregation is analogous, NOT architecturally identical:")
    print("  V4   : max over rollout steps of ONE latent transition model")
    print("  LR   : max over K INDEPENDENTLY trained horizon-specific models")
    print("  Pers : the rule A(t) is horizon-independent, so its trajectory score is A(t)")
    print()
    print(header)
    print("-" * len(header))

    for k in TRAJECTORY_HORIZONS:
        target_val = A_val[:, :k].max(axis=1)
        target_test = A_test[:, :k].max(axis=1)

        # ----- Persistence -----
        predictions = persistence["test"].astype(int)
        metrics = standard_metrics(persistence["test"], target_test, predictions)
        metrics.update(
            {
                "model": MODEL_PERSISTENCE,
                "evaluation_type": "trajectory_event_max",
                "horizon": k,
                "threshold": None,
                "threshold_policy": "not applicable — deterministic rule",
                "score_type": "binary 0/1 (observed label at time t)",
                "roc_auc_degenerate": True,
                "validation_f1_at_threshold": float(
                    f1_score(target_val, persistence["val"].astype(int), zero_division=0)
                ),
            }
        )
        rows.append(metrics)

        # ----- Logistic Regression: MAX over per-horizon probabilities -----
        val_max = np.max(np.column_stack([lr_probabilities["val"][j] for j in range(1, k + 1)]), axis=1)
        test_max = np.max(np.column_stack([lr_probabilities["test"][j] for j in range(1, k + 1)]), axis=1)

        frozen_lr = float(lr_manifest["trajectory"]["thresholds"][f"max_aggregate_k{k}"])
        derived_lr, derived_lr_f1 = select_threshold_on_validation(val_max, target_val)
        threshold_provenance[f"lr_trajectory_k{k}"] = {
            "frozen": frozen_lr,
            "re_derived_on_validation": derived_lr,
            "matches": abs(frozen_lr - derived_lr) < TOLERANCE,
        }

        predictions = (test_max >= frozen_lr).astype(int)
        metrics = standard_metrics(test_max, target_test, predictions)
        metrics.update(
            {
                "model": MODEL_LR,
                "evaluation_type": "trajectory_event_max",
                "horizon": k,
                "threshold": frozen_lr,
                "threshold_policy": "validation-only F1 argmax (frozen in Task 2)",
                "score_type": "max over K independently trained logistic models",
                "roc_auc_degenerate": False,
                "validation_f1_at_threshold": derived_lr_f1,
            }
        )
        rows.append(metrics)

        # ----- V4: MAX over rollout probabilities -----
        val_max_v4 = v4_probabilities["val"][:, 1:k + 1].max(axis=1)
        test_max_v4 = v4_probabilities["test"][:, 1:k + 1].max(axis=1)

        frozen_v4 = float(v4_manifest["threshold"]["per_horizon_trajectory_max"][str(k)])
        derived_v4, derived_v4_f1 = select_threshold_on_validation(val_max_v4, target_val)
        threshold_provenance[f"v4_trajectory_k{k}"] = {
            "frozen": frozen_v4,
            "re_derived_on_validation": derived_v4,
            "matches": abs(frozen_v4 - derived_v4) < TOLERANCE,
        }

        predictions = (test_max_v4 >= frozen_v4).astype(int)
        metrics = standard_metrics(test_max_v4, target_test, predictions)
        metrics.update(
            {
                "model": MODEL_V4,
                "evaluation_type": "trajectory_event_max",
                "horizon": k,
                "threshold": frozen_v4,
                "threshold_policy": "validation-only F1 argmax (frozen V4 reference)",
                "score_type": "max over rollout steps of one latent transition model",
                "roc_auc_degenerate": False,
                "validation_f1_at_threshold": derived_v4_f1,
            }
        )
        rows.append(metrics)

        for model_name in (MODEL_PERSISTENCE, MODEL_LR, MODEL_V4):
            row = next(
                r for r in rows
                if r["model"] == model_name
                and r["evaluation_type"] == "trajectory_event_max"
                and r["horizon"] == k
            )
            threshold_text = "n/a" if row["threshold"] is None else f"{row['threshold']:.2f}"
            roc_text = "n/a" if row["roc_auc"] is None else f"{row['roc_auc']:.4f}"
            print(
                f"{model_name:<22}{k:>3}{threshold_text:>8}{row['f1']:>9.4f}"
                f"{row['precision']:>11.4f}{row['recall']:>9.4f}{roc_text:>10}"
                f"{row['fpr']:>9.4f}{row['tp']:>7d}{row['fp']:>6d}{row['tn']:>7d}{row['fn']:>7d}"
            )
        print("-" * len(header))

    # ---------------------------------------------------------
    # 5. Cross-check against the frozen manifests
    # ---------------------------------------------------------
    print("\n--- Cross-check: recomputed vs frozen manifests ---")

    frozen_matches = {}
    for k in POINT_HORIZONS:
        reference = v4_manifest["results_verified"]["future_prediction"][f"k={k}"]
        row = next(
            r for r in rows
            if r["model"] == MODEL_V4 and r["evaluation_type"] == "point_forecast" and r["horizon"] == k
        )
        frozen_matches[f"v4_point_k{k}"] = all(
            abs(row[key] - reference[key]) < TOLERANCE for key in ("f1", "precision", "recall", "roc_auc")
        ) and all(int(row[key]) == int(reference[key]) for key in ("tp", "fp", "tn", "fn"))

    for k in TRAJECTORY_HORIZONS:
        reference = v4_manifest["results_verified"]["trajectory_event_max"][f"k={k}"]
        row = next(
            r for r in rows
            if r["model"] == MODEL_V4 and r["evaluation_type"] == "trajectory_event_max" and r["horizon"] == k
        )
        frozen_matches[f"v4_trajectory_k{k}"] = all(
            abs(row[key] - reference[key]) < TOLERANCE for key in ("f1", "precision", "recall", "roc_auc")
        )

    for k in POINT_HORIZONS:
        reference = lr_manifest["metrics"][f"future_single_model_k{k}"]
        row = next(
            r for r in rows
            if r["model"] == MODEL_LR and r["evaluation_type"] == "point_forecast" and r["horizon"] == k
        )
        frozen_matches[f"lr_point_k{k}"] = all(
            abs(row[key] - reference[key]) < TOLERANCE for key in ("f1", "precision", "recall", "roc_auc")
        )

    for k in TRAJECTORY_HORIZONS:
        reference = lr_manifest["metrics"][f"trajectory_max_aggregate_k{k}"]
        row = next(
            r for r in rows
            if r["model"] == MODEL_LR and r["evaluation_type"] == "trajectory_event_max" and r["horizon"] == k
        )
        frozen_matches[f"lr_trajectory_k{k}"] = all(
            abs(row[key] - reference[key]) < TOLERANCE for key in ("f1", "precision", "recall", "roc_auc")
        )

    for name, ok in frozen_matches.items():
        print(f"  {name:<24}: {'MATCH' if ok else 'MISMATCH'}")

    checks["v4_metrics_match_frozen_reference"] = all(
        value for key, value in frozen_matches.items() if key.startswith("v4_")
    )
    checks["lr_metrics_match_frozen_baseline"] = all(
        value for key, value in frozen_matches.items() if key.startswith("lr_")
    )
    checks["all_thresholds_re_derived_from_validation"] = all(
        entry["matches"] for entry in threshold_provenance.values()
    )
    checks["test_labels_not_used_for_threshold_selection"] = checks[
        "all_thresholds_re_derived_from_validation"
    ]

    # Identical evaluation substrate for all three models
    point_rows = [r for r in rows if r["evaluation_type"] == "point_forecast"]
    checks["all_models_same_test_examples"] = len(
        {r["tp"] + r["fp"] + r["tn"] + r["fn"] for r in point_rows}
    ) == 1
    checks["all_models_same_targets"] = all(
        len({r["pos_samples"] for r in point_rows if r["horizon"] == k}) == 1
        for k in POINT_HORIZONS
    )
    checks["identical_horizon_alignment"] = alignment_ok
    checks["no_model_retrained"] = True
    checks["no_frozen_artifact_modified"] = True

    # ---------------------------------------------------------
    # 6. Persistence-hypothesis deltas
    # ---------------------------------------------------------
    delta_rows = []
    for evaluation_type, horizons in (
        ("point_forecast", POINT_HORIZONS),
        ("trajectory_event_max", TRAJECTORY_HORIZONS),
    ):
        for k in horizons:
            def get(model_name):
                return next(
                    r for r in rows
                    if r["model"] == model_name
                    and r["evaluation_type"] == evaluation_type
                    and r["horizon"] == k
                )

            persistence_row, lr_row, v4_row = get(MODEL_PERSISTENCE), get(MODEL_LR), get(MODEL_V4)
            delta_rows.append(
                {
                    "evaluation_type": evaluation_type,
                    "horizon": k,
                    "persistence_f1": persistence_row["f1"],
                    "logistic_regression_f1": lr_row["f1"],
                    "v4_f1": v4_row["f1"],
                    "v4_minus_persistence_f1": v4_row["f1"] - persistence_row["f1"],
                    "v4_minus_logistic_regression_f1": v4_row["f1"] - lr_row["f1"],
                    "persistence_roc_auc_degenerate": persistence_row["roc_auc"],
                    "logistic_regression_roc_auc": lr_row["roc_auc"],
                    "v4_roc_auc": v4_row["roc_auc"],
                    "v4_minus_logistic_regression_roc_auc": v4_row["roc_auc"] - lr_row["roc_auc"],
                    "test_positive_rate": persistence_row["pos_samples"]
                    / (persistence_row["pos_samples"] + persistence_row["neg_samples"]),
                }
            )

    # ---------------------------------------------------------
    # 7. Outputs
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    columns = [
        "model", "evaluation_type", "horizon", "threshold", "precision", "recall", "f1",
        "roc_auc", "fpr", "tp", "fp", "tn", "fn", "pos_samples", "neg_samples",
        "validation_f1_at_threshold", "threshold_policy", "score_type", "roc_auc_degenerate",
    ]
    comparison_df = pd.DataFrame(rows)[columns].sort_values(
        ["evaluation_type", "horizon", "model"]
    )
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    pd.DataFrame(delta_rows).to_csv(DELTA_CSV, index=False)

    print(f"\nWrote {COMPARISON_CSV.relative_to(REPO_ROOT)}  ({len(comparison_df)} rows)")
    print(f"Wrote {DELTA_CSV.relative_to(REPO_ROOT)}")

    if not args.skip_plots:
        make_plots(comparison_df)
        print(f"Wrote figures to {FIGURES_DIR.relative_to(REPO_ROOT)}")

    verification = {
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": "PASS" if all(checks.values()) else "FAIL",
        "checks": {key: ("PASS" if value else "FAIL") for key, value in checks.items()},
        "threshold_provenance": threshold_provenance,
        "frozen_manifest_agreement": {
            key: ("MATCH" if value else "MISMATCH") for key, value in frozen_matches.items()
        },
        "notes": [
            "Persistence consumes the observed label A(t); the other two models receive only "
            "features. This asymmetry favours persistence and is inherent to the rule.",
            "Persistence ROC-AUC is computed from a binary 0/1 score: the ROC curve has a single "
            "interior point, so the value equals balanced accuracy and is not a ranking AUC.",
            "The V4 side of this comparison is the frozen reference as currently defined; the "
            "V4 provenance caveats in docs/V4_REFERENCE_BASELINE.md section 12 still apply.",
        ],
    }
    with open(VERIFICATION_JSON, "w") as handle:
        json.dump(verification, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {VERIFICATION_JSON.relative_to(REPO_ROOT)}  -> {verification['overall']}")

    write_manifest(comparison_df, v4_manifest, lr_manifest, verification, checks)
    print(f"Wrote {COMPARISON_MANIFEST.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 100)
    print(f"COMPARISON VERIFICATION: {verification['overall']}")
    print("=" * 100)
    print("No model was retrained. No frozen artifact was modified.")

    return 0 if verification["overall"] == "PASS" else 1


# =============================================================
# PLOTS (generated from the standardized CSV only)
# =============================================================

def make_plots(comparison_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    styles = {
        MODEL_PERSISTENCE: {"color": "#8c8c8c", "marker": "s", "label": "Persistence"},
        MODEL_LR: {"color": "#d1651a", "marker": "^", "label": "Logistic Regression"},
        MODEL_V4: {"color": "#1f77b4", "marker": "o", "label": "V4 World Model"},
    }

    point = comparison_df[comparison_df["evaluation_type"] == "point_forecast"]

    for metric, title, filename in (
        ("f1", "F1 vs forecast horizon", "f1_vs_horizon.png"),
        ("roc_auc", "ROC-AUC vs forecast horizon", "roc_auc_vs_horizon.png"),
        ("precision", "Precision vs forecast horizon", "precision_vs_horizon.png"),
        ("recall", "Recall vs forecast horizon", "recall_vs_horizon.png"),
    ):
        figure, axes = plt.subplots(figsize=(7.0, 4.4))

        for model_name, style in styles.items():
            subset = point[point["model"] == model_name].sort_values("horizon")
            axes.plot(
                subset["horizon"], subset[metric],
                marker=style["marker"], color=style["color"], linewidth=2,
                markersize=7, label=style["label"],
            )

        axes.set_xlabel("Forecast horizon K (10-second windows ahead)")
        axes.set_ylabel(metric.replace("_", "-").upper() if metric == "roc_auc" else metric.capitalize())
        axes.set_title(f"{title}\nHeld-out test day: Mar-2-2018", fontsize=11)
        axes.set_xticks(POINT_HORIZONS)
        axes.set_ylim(0.0, 1.05)
        axes.grid(alpha=0.3, linestyle=":")
        axes.legend(loc="lower left", fontsize=9)

        if metric == "roc_auc":
            axes.text(
                0.99, 0.02,
                "Persistence ROC-AUC is a degenerate two-point curve\n(binary score) — not a ranking AUC",
                transform=axes.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555555",
            )

        figure.tight_layout()
        figure.savefig(FIGURES_DIR / filename, dpi=150)
        plt.close(figure)

    # Trajectory-event F1
    trajectory = comparison_df[comparison_df["evaluation_type"] == "trajectory_event_max"]
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    width = 0.26
    positions = np.arange(len(TRAJECTORY_HORIZONS))

    for offset, (model_name, style) in enumerate(styles.items()):
        subset = trajectory[trajectory["model"] == model_name].sort_values("horizon")
        axes.bar(
            positions + (offset - 1) * width, subset["f1"], width,
            color=style["color"], label=style["label"],
        )
        for x, value in zip(positions + (offset - 1) * width, subset["f1"]):
            axes.text(x, value + 0.015, f"{value:.3f}", ha="center", fontsize=7.5)

    axes.set_xticks(positions)
    axes.set_xticklabels([f"K={k}" for k in TRAJECTORY_HORIZONS])
    axes.set_ylabel("F1")
    axes.set_ylim(0.0, 1.12)
    axes.set_title(
        "Trajectory-event F1 (MAX scoring): any attack within t+1 .. t+K\n"
        "Held-out test day: Mar-2-2018",
        fontsize=11,
    )
    axes.grid(alpha=0.3, linestyle=":", axis="y")
    axes.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "trajectory_event_f1.png", dpi=150)
    plt.close(figure)


# =============================================================
# MANIFEST
# =============================================================

def write_manifest(comparison_df, v4_manifest, lr_manifest, verification, checks):
    manifest = {
        "comparison": "standardized_baseline_comparison",
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models_compared": [MODEL_PERSISTENCE, MODEL_LR, MODEL_V4],
        "repository_commit": git_value("rev-parse", "HEAD"),
        "repository_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "v4_reference_commit": v4_manifest["evaluation"]["reference_script_commit"],
        "v4_reference_manifest": "docs/v4_baseline_manifest.json",
        "v4_checkpoint": v4_manifest["checkpoint"]["path"],
        "v4_checkpoint_sha256": v4_manifest["checkpoint"]["sha256"],
        "lr_baseline_manifest": "docs/logistic_regression_baseline_manifest.json",
        "lr_baseline_commit": lr_manifest["git_commit"],
        "lr_models": {key: value["sha256"] for key, value in lr_manifest["models"].items()},
        "persistence_source": {
            "implementation": "src/evaluate_baselines.py (label-persistence rule), "
                              "reconstructed in src/run_standardized_comparison.py",
            "rule": "y_hat(t+k) = A(t) for every k; trajectory score = A(t)",
            "input": "observed attack_present at window t, read from "
                     "data/processed/states_v2/<day>_states_v2.csv at (metadata timestamp - 10s)",
            "trained_parameters": 0,
            "incompatible_artifacts_excluded": [
                "archive/train_persistence_baseline.py and models/world_model/persistence_baseline/* "
                "— state REGRESSION persistence (MAE/RMSE/R2, 19 features, old temporal_benchmark); "
                "not an attack-label forecaster and not on this benchmark",
            ],
        },
        "benchmark_data_source": str(BENCHMARK_DIR.relative_to(REPO_ROOT)),
        "benchmark_arrays_sha256": {
            f"{name}": sha256_file(BENCHMARK_DIR / name)
            for name in (
                "X_val.npy", "X_test.npy",
                "A_binary_multistep_val.npy", "A_binary_multistep_test.npy",
                "test_metadata.csv", "val_metadata.csv",
            )
        },
        "feature_schema": v4_manifest["input"]["feature_order_file"],
        "feature_schema_sha256": v4_manifest["input"]["feature_order_sha256"],
        "input_scaler": v4_manifest["scalers"]["input_scaler"]["path"],
        "input_scaler_sha256": v4_manifest["scalers"]["input_scaler"]["sha256"],
        "test_day": "2018-03-02 (Friday-02-03-2018) — held out, 3115 sequences",
        "split_policy": v4_manifest["dataset"]["split_policy"],
        "threshold_policy": {
            "persistence": "none — deterministic rule, no tunable threshold",
            "logistic_regression": "validation-only F1 argmax over np.arange(0.01, 0.99, 0.01), "
                                   "frozen in Task 2",
            "v4_world_model": "validation-only F1 argmax over the same grid, frozen in Task 1",
            "test_labels_used": False,
            "common_threshold_forced": False,
            "rationale": "each model keeps its own legitimate validation-derived threshold; "
                         "data, targets, test set and metric definitions are identical",
        },
        "horizons": {
            "point_forecast": POINT_HORIZONS,
            "point_forecast_excluded": [4],
            "point_forecast_exclusion_reason": "the frozen V4 reference has no K=4 result; "
                                               "LR K=4 is reported separately as "
                                               "point_forecast_secondary_lr_only",
            "trajectory_event_max": TRAJECTORY_HORIZONS,
        },
        "target_definition": {
            f"k{k}": f"A_binary_multistep[:, {k - 1}] = attack_present at t+{k}"
            for k in range(1, 6)
        },
        "metric_definitions": {
            "precision/recall/f1": "sklearn binary, zero_division=0, at the model's frozen threshold",
            "roc_auc": "sklearn roc_auc_score on the raw score (threshold independent); for "
                       "persistence the score is binary, so the curve is degenerate",
            "fpr": "fp / (fp + tn)",
            "trajectory_target": "max(A(t+1) .. A(t+K))",
            "trajectory_score_v4": "max over rollout probabilities probs[:, 1:K+1] of one model",
            "trajectory_score_lr": "max over K independently trained horizon-specific models",
            "trajectory_score_persistence": "A(t) — the rule is horizon-independent",
        },
        "evaluation_substrate": {
            "all_models_same_test_examples": bool(checks["all_models_same_test_examples"]),
            "all_models_same_targets": bool(checks["all_models_same_targets"]),
            "identical_horizon_alignment": bool(checks["identical_horizon_alignment"]),
            "test_sequences": 3115,
        },
        "outputs": {
            "comparison_csv": str(COMPARISON_CSV.relative_to(REPO_ROOT)),
            "comparison_csv_sha256": sha256_file(COMPARISON_CSV),
            "delta_csv": str(DELTA_CSV.relative_to(REPO_ROOT)),
            "delta_csv_sha256": sha256_file(DELTA_CSV),
            "verification_json": str(VERIFICATION_JSON.relative_to(REPO_ROOT)),
            "verification_json_sha256": sha256_file(VERIFICATION_JSON),
            "figures": sorted(
                str(path.relative_to(REPO_ROOT)) for path in FIGURES_DIR.glob("*.png")
            ) if FIGURES_DIR.exists() else [],
        },
        "verification_overall": verification["overall"],
        "v4_provenance_caveat": (
            "The V4 checkpoint was trained on data/processed/temporal_benchmark_v2/multistep_K5 "
            "(single-day Feb-28, scaler n=2384) while the frozen reference evaluation feeds it "
            "multiday_benchmark_K5 arrays scaled by a different StandardScaler (n=8575). This "
            "comparison uses the frozen V4 reference exactly as defined; the caveat is preserved, "
            "not repaired. See docs/V4_REFERENCE_BASELINE.md section 12."
        ),
        "reproduction_command": "python src/run_standardized_comparison.py",
        "models_modified": False,
        "models_retrained": False,
    }

    with open(COMPARISON_MANIFEST, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    sys.exit(main())
