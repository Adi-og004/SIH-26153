"""
=============================================================
LOGISTIC REGRESSION BASELINE — MULTI-DAY BENCHMARK
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

A conventional STATIC classifier baseline. This is NOT a world model.

Scientific question:
    How much future-attack prediction performance is achievable
    WITHOUT a learned temporal world model?

Formulation (established by src/evaluate_baselines.py, "Logistic
Regression (latest-state)"):

    Input  : S_t in R^52  -- the most recent observed network state,
             i.e. X[:, -1, :] of the frozen V4 benchmark tensor.
             The classifier never sees the 10-step history that V4
             consumes.
    Target : A(t+k) = A_binary_multistep[:, k-1]   for k = 1..5
    Model  : one independent LogisticRegression per horizon

Frozen-baseline discipline (mirrors docs/V4_REFERENCE_BASELINE.md):
    - Same benchmark arrays, same split, same 52-feature order.
    - Same frozen input scaler: the benchmark arrays are ALREADY
      scaled by data/processed/multiday_benchmark_K5/input_scaler.pkl
      (fit on the train partition only). Nothing is refit here.
    - Thresholds are selected on VALIDATION only, then frozen and
      applied once to the Mar-2 test day.
    - Test labels are never used for threshold, C, class weight,
      solver, feature or model selection.

This script writes only into models/logistic_regression_baseline/
and docs/logistic_regression_baseline_manifest.json. It never reads,
writes or modifies any V4 artifact other than the shared, read-only
benchmark data and feature schema.

Usage:
    python src/train_logistic_regression_baseline.py
    python src/train_logistic_regression_baseline.py --output-dir models/lr_experiment --no-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BENCHMARK_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "logistic_regression_baseline"
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "logistic_regression_baseline_manifest.json"
FEATURE_SCHEMA_JSON = REPO_ROOT / "docs" / "v4_feature_schema.json"
V4_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"

# ----- Frozen baseline constants -------------------------------------------

HORIZONS = [1, 2, 3, 4, 5]
TRAJECTORY_HORIZONS = [2, 3, 5]

EXPECTED_FEATURE_DIM = 52
EXPECTED_BASE_FEATURES = 26
EXPECTED_SEQUENCE_LENGTH = 10
EXPECTED_TEST_SOURCE_DAY = "Friday-02-03-2018"

# Threshold grid — identical convention to the frozen V4 reference
# evaluation (np.arange(0.01, 0.99, 0.01), keep the first strict maximum).
THRESHOLD_LO = 0.01
THRESHOLD_HI = 0.99
THRESHOLD_STEP = 0.01

# Model configuration. class_weight="balanced" and random_state=42 are the
# repository's established baseline convention (src/evaluate_baselines.py,
# archive/train_baseline.py, archive/train_temporal_baseline.py). Chosen
# before looking at any test score and not tuned afterwards.
MODEL_CONFIG = {
    "estimator": "sklearn.linear_model.LogisticRegression",
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "tol": 1e-4,
    "class_weight": "balanced",
    "fit_intercept": True,
    "random_state": 42,
    "multi_class": "binary (single positive class)",
}


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
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def threshold_grid() -> np.ndarray:
    return np.arange(THRESHOLD_LO, THRESHOLD_HI, THRESHOLD_STEP)


def select_threshold_on_validation(probabilities, targets):
    """
    Maximise F1 over the validation grid.

    MUST be called with validation probabilities and validation labels only.
    Returns (threshold, validation_f1, full grid as a list of dicts).
    """
    best_threshold = 0.5
    best_f1 = -1.0
    rows = []

    for threshold in threshold_grid():
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "val_precision": float(precision),
                "val_recall": float(recall),
                "val_f1": float(f1),
            }
        )

        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)

    return best_threshold, best_f1, rows


def compute_metrics(probabilities, targets, threshold):
    """Test metrics under a frozen threshold. ROC-AUC uses raw probabilities."""
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        targets, predictions, labels=[0, 1]
    ).ravel()

    roc_auc = (
        float(roc_auc_score(targets, probabilities))
        if len(np.unique(targets)) >= 2
        else float("nan")
    )

    return {
        "threshold": round(float(threshold), 2),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "pos_samples": int(np.sum(targets)),
        "neg_samples": int(len(targets) - np.sum(targets)),
    }


def save_confusion_matrix(path: Path, metrics: dict) -> None:
    pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Actual_Benign", "Actual_Attack"],
        columns=["Predicted_Benign", "Predicted_Attack"],
    ).to_csv(path)


# =============================================================
# DATA LOADING AND INTEGRITY
# =============================================================

def load_feature_order(benchmark_dir: Path):
    """Load the authoritative 52-feature order and cross-check both sources."""
    names_path = benchmark_dir / "input_feature_names.txt"

    if not names_path.exists():
        raise FileNotFoundError(f"Feature order file not found: {names_path}")

    features = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]

    if len(features) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_DIM} features, found {len(features)} in {names_path}"
        )

    base = features[:EXPECTED_BASE_FEATURES]
    relative = features[EXPECTED_BASE_FEATURES:]

    expected_relative = [f"{name}_rel_change" for name in base]
    if relative != expected_relative:
        raise ValueError(
            "Feature order violated: indices 26-51 are not the base features "
            "suffixed with '_rel_change'."
        )

    schema_agrees = None
    if FEATURE_SCHEMA_JSON.exists():
        schema = json.loads(FEATURE_SCHEMA_JSON.read_text())
        schema_names = [item["name"] for item in schema["features"]]
        schema_agrees = schema_names == features
        if not schema_agrees:
            raise ValueError(
                f"{FEATURE_SCHEMA_JSON} disagrees with {names_path} on feature order."
            )

    return features, base, relative, names_path, schema_agrees


def load_benchmark(benchmark_dir: Path):
    """Load the frozen benchmark arrays and verify the split before use."""
    print("\n--- Loading frozen benchmark ---")
    print(f"Directory: {benchmark_dir}")

    data = {}
    for split in ("train", "val", "test"):
        X = np.load(benchmark_dir / f"X_{split}.npy")
        A = np.load(benchmark_dir / f"A_binary_multistep_{split}.npy")
        meta = pd.read_csv(benchmark_dir / f"{split}_metadata.csv")
        meta["timestamp"] = pd.to_datetime(meta["timestamp"])

        if X.ndim != 3 or X.shape[1] != EXPECTED_SEQUENCE_LENGTH or X.shape[2] != EXPECTED_FEATURE_DIM:
            raise ValueError(
                f"{split}: expected X of shape (N, {EXPECTED_SEQUENCE_LENGTH}, "
                f"{EXPECTED_FEATURE_DIM}), got {X.shape}"
            )
        if A.shape[0] != X.shape[0] or A.shape[1] < max(HORIZONS):
            raise ValueError(f"{split}: label array shape {A.shape} incompatible with X {X.shape}")
        if len(meta) != X.shape[0]:
            raise ValueError(f"{split}: metadata rows {len(meta)} != sequences {X.shape[0]}")
        if not np.isfinite(X).all():
            raise ValueError(f"{split}: X contains NaN/Inf")
        if not np.all(np.isin(np.unique(A[:, : max(HORIZONS)]), [0, 1])):
            raise ValueError(f"{split}: labels are not binary")

        data[split] = {"X": X, "A": A, "meta": meta}
        print(f"  {split:<5}: X={X.shape}  A={A.shape}  days={sorted(meta['source_day'].unique())}")

    # ----- Split integrity (verified, not assumed) -----
    train_days = set(data["train"]["meta"]["source_day"].unique())
    val_days = set(data["val"]["meta"]["source_day"].unique())
    test_days = set(data["test"]["meta"]["source_day"].unique())

    expected_train_days = {
        "Wednesday-14-02-2018",
        "Thursday-15-02-2018",
        "Wednesday-28-02-2018",
    }

    if train_days != expected_train_days:
        raise ValueError(f"Unexpected train days: {sorted(train_days)}")
    if val_days != expected_train_days:
        raise ValueError(f"Unexpected validation days: {sorted(val_days)}")
    if test_days != {EXPECTED_TEST_SOURCE_DAY}:
        raise ValueError(f"Test split is not {EXPECTED_TEST_SOURCE_DAY}-only: {sorted(test_days)}")
    if test_days & (train_days | val_days):
        raise ValueError("Test day leaks into train/validation.")

    # Chronological discipline: validation must follow training within each day.
    for day in sorted(expected_train_days):
        train_last = data["train"]["meta"].loc[
            data["train"]["meta"]["source_day"] == day, "timestamp"
        ].max()
        val_first = data["val"]["meta"].loc[
            data["val"]["meta"]["source_day"] == day, "timestamp"
        ].min()
        if not val_first > train_last:
            raise ValueError(
                f"{day}: validation starts at {val_first}, not after train end {train_last}"
            )

    # Target alignment: A[:, 0] is the label at t+1, which metadata records.
    for split in ("train", "val", "test"):
        stored = data[split]["meta"]["attack_present"].to_numpy(dtype=float)
        if not np.array_equal(stored, data[split]["A"][:, 0]):
            raise ValueError(f"{split}: A[:, 0] does not match metadata attack_present (t+1)")

    print("  Split integrity: PASS (train/val days match, Mar-2 test isolated, "
          "validation is chronologically after train, A[:, 0] == t+1 label)")

    return data


# =============================================================
# MAIN PIPELINE
# =============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Logistic Regression (latest-state) baseline "
                    "on the frozen multi-day benchmark."
    )
    parser.add_argument("--benchmark-dir", default=str(DEFAULT_BENCHMARK_DIR),
                        help="Frozen benchmark directory (default: data/processed/multiday_benchmark_K5)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Artifact directory (default: models/logistic_regression_baseline)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                        help="Manifest path (default: docs/logistic_regression_baseline_manifest.json)")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Skip writing the reproducibility manifest")
    parser.add_argument("--horizons", default=",".join(str(k) for k in HORIZONS),
                        help="Comma-separated horizons (default: 1,2,3,4,5)")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = REPO_ROOT / benchmark_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    horizons = [int(value) for value in args.horizons.split(",") if value.strip()]
    trajectory_horizons = [k for k in TRAJECTORY_HORIZONS if k <= max(horizons)]

    print("=" * 96)
    print("LOGISTIC REGRESSION BASELINE — LATEST-STATE, MULTI-HORIZON")
    print("=" * 96)
    print("Conventional static classifier. NOT a world model.")
    print(f"Repository : {REPO_ROOT}")
    print(f"Output dir : {output_dir}")

    # ----- Feature schema -----
    features, base_features, relative_features, feature_path, schema_agrees = (
        load_feature_order(benchmark_dir)
    )
    print("\n--- Feature schema ---")
    print(f"Source: {feature_path.relative_to(REPO_ROOT)}")
    print(f"{len(base_features)} base + {len(relative_features)} relative = {len(features)} features")
    print(f"docs/v4_feature_schema.json agreement: "
          f"{'PASS' if schema_agrees else 'not present'}")

    # ----- Scaler provenance (read-only; nothing is refit) -----
    scaler_path = benchmark_dir / "input_scaler.pkl"
    with open(scaler_path, "rb") as handle:
        input_scaler = pickle.load(handle)

    if int(input_scaler.n_features_in_) != EXPECTED_FEATURE_DIM:
        raise ValueError(
            f"Frozen input scaler has n_features_in_={input_scaler.n_features_in_}, "
            f"expected {EXPECTED_FEATURE_DIM}"
        )

    print("\n--- Input scaling ---")
    print(f"Frozen scaler: {scaler_path.relative_to(REPO_ROOT)}")
    print(f"  {type(input_scaler).__name__}, n_features_in_={input_scaler.n_features_in_}, "
          f"n_samples_seen_={int(input_scaler.n_samples_seen_)} (train partition only)")
    print("  The benchmark arrays are ALREADY transformed by this scaler at build time")
    print("  (src/prepare_multiday_benchmark.py). No scaler is fitted here; test data")
    print("  never participates in any fit.")

    # ----- Data -----
    data = load_benchmark(benchmark_dir)

    # Latest observed state S_t = X[:, -1, :]. The 10-step history is deliberately discarded.
    S_train = data["train"]["X"][:, -1, :]
    S_val = data["val"]["X"][:, -1, :]
    S_test = data["test"]["X"][:, -1, :]

    A_train, A_val, A_test = (data[s]["A"] for s in ("train", "val", "test"))

    print("\n--- Classifier input ---")
    print(f"S_t = X[:, -1, :] -> train {S_train.shape}, val {S_val.shape}, test {S_test.shape}")
    print("The 10-step temporal sequence used by V4 is NOT provided to this baseline.")

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    threshold_rows = []
    model_records = {}
    val_probabilities = {}
    test_probabilities = {}

    # =========================================================
    # PER-HORIZON MODELS
    # =========================================================
    print("\n" + "=" * 96)
    print("1. FUTURE ATTACK PREDICTION — one model per horizon")
    print("=" * 96)
    print("Input S_t -> target A(t+k) = A_binary_multistep[:, k-1]")
    print()

    header = (
        f"{'Horizon':<10}{'Thr':>6}{'F1':>9}{'Precision':>11}{'Recall':>9}"
        f"{'ROC-AUC':>10}{'FPR':>9}{'TP':>7}{'FP':>6}{'TN':>7}{'FN':>7}{'ValF1':>9}"
    )
    print(header)
    print("-" * len(header))

    for k in horizons:
        y_train = A_train[:, k - 1]
        y_val = A_val[:, k - 1]
        y_test = A_test[:, k - 1]

        model = LogisticRegression(
            penalty=MODEL_CONFIG["penalty"],
            C=MODEL_CONFIG["C"],
            solver=MODEL_CONFIG["solver"],
            max_iter=MODEL_CONFIG["max_iter"],
            tol=MODEL_CONFIG["tol"],
            class_weight=MODEL_CONFIG["class_weight"],
            fit_intercept=MODEL_CONFIG["fit_intercept"],
            random_state=MODEL_CONFIG["random_state"],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(S_train, y_train)
            converged = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )

        probabilities_val = model.predict_proba(S_val)[:, 1]
        probabilities_test = model.predict_proba(S_test)[:, 1]

        val_probabilities[k] = probabilities_val
        test_probabilities[k] = probabilities_test

        # ----- Threshold: VALIDATION ONLY -----
        threshold, validation_f1, grid = select_threshold_on_validation(
            probabilities_val, y_val
        )

        for row in grid:
            threshold_rows.append({"task": "future", "k": k, "strategy": "single_model", **row})

        # ----- Frozen threshold applied once to TEST -----
        metrics = compute_metrics(probabilities_test, y_test, threshold)
        metrics.update(
            {
                "task": "future",
                "k": k,
                "strategy": "single_model",
                "target": f"A(t+{k})",
                "validation_f1_at_threshold": validation_f1,
                "val_pos_samples": int(y_val.sum()),
                "val_neg_samples": int(len(y_val) - y_val.sum()),
                "train_samples": int(len(y_train)),
                "train_pos_samples": int(y_train.sum()),
                "converged": bool(converged),
                "n_iter": int(np.max(model.n_iter_)),
            }
        )
        metrics_rows.append(metrics)

        model_path = output_dir / f"logistic_regression_k{k}.pkl"
        with open(model_path, "wb") as handle:
            pickle.dump(model, handle)

        model_records[k] = {
            "path": str(model_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(model_path),
            "n_iter": int(np.max(model.n_iter_)),
            "converged": bool(converged),
            "n_features_in": int(model.n_features_in_),
            "intercept": float(model.intercept_[0]),
            "coef_l2_norm": float(np.linalg.norm(model.coef_)),
        }

        save_confusion_matrix(output_dir / f"confusion_matrix_k{k}.csv", metrics)

        print(
            f"{'t+' + str(k):<10}{metrics['threshold']:>6.2f}{metrics['f1']:>9.4f}"
            f"{metrics['precision']:>11.4f}{metrics['recall']:>9.4f}{metrics['roc_auc']:>10.4f}"
            f"{metrics['fpr']:>9.4f}{metrics['tp']:>7d}{metrics['fp']:>6d}"
            f"{metrics['tn']:>7d}{metrics['fn']:>7d}{validation_f1:>9.4f}"
        )

    # =========================================================
    # TRAJECTORY-EVENT PREDICTION
    # =========================================================
    print("\n" + "=" * 96)
    print("2. TRAJECTORY-EVENT PREDICTION — target max(A(t+1) .. A(t+k))")
    print("=" * 96)
    print("PRIMARY   'max_aggregate' : max over the per-horizon LR probabilities for t+1..t+k")
    print("                            (same MAX convention as the frozen V4 reference, which")
    print("                             takes max over its rollout probabilities probs[:, 1:k+1])")
    print("SECONDARY 'direct_target' : one LR trained directly on the max target")
    print("                            (the formulation already present in src/evaluate_baselines.py)")
    print()

    print(header.replace("Horizon", "Strategy").replace(f"{'Strategy':<10}", f"{'Strategy':<24}"))
    print("-" * (len(header) + 14))

    for k in trajectory_horizons:
        y_val_event = A_val[:, :k].max(axis=1)
        y_test_event = A_test[:, :k].max(axis=1)
        y_train_event = A_train[:, :k].max(axis=1)

        strategies = {}

        # ----- PRIMARY: MAX aggregation over per-horizon probabilities -----
        strategies["max_aggregate"] = (
            np.max(np.column_stack([val_probabilities[j] for j in range(1, k + 1)]), axis=1),
            np.max(np.column_stack([test_probabilities[j] for j in range(1, k + 1)]), axis=1),
            None,
        )

        # ----- SECONDARY: direct-trained model on the event target -----
        direct_model = LogisticRegression(
            penalty=MODEL_CONFIG["penalty"],
            C=MODEL_CONFIG["C"],
            solver=MODEL_CONFIG["solver"],
            max_iter=MODEL_CONFIG["max_iter"],
            tol=MODEL_CONFIG["tol"],
            class_weight=MODEL_CONFIG["class_weight"],
            fit_intercept=MODEL_CONFIG["fit_intercept"],
            random_state=MODEL_CONFIG["random_state"],
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            direct_model.fit(S_train, y_train_event)
            direct_converged = not any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )

        strategies["direct_target"] = (
            direct_model.predict_proba(S_val)[:, 1],
            direct_model.predict_proba(S_test)[:, 1],
            direct_model,
        )

        direct_path = output_dir / f"logistic_regression_trajectory_direct_k{k}.pkl"
        with open(direct_path, "wb") as handle:
            pickle.dump(direct_model, handle)

        model_records[f"trajectory_direct_k{k}"] = {
            "path": str(direct_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(direct_path),
            "n_iter": int(np.max(direct_model.n_iter_)),
            "converged": bool(direct_converged),
            "n_features_in": int(direct_model.n_features_in_),
            "intercept": float(direct_model.intercept_[0]),
            "coef_l2_norm": float(np.linalg.norm(direct_model.coef_)),
        }

        for strategy, (probabilities_val, probabilities_test, _) in strategies.items():
            threshold, validation_f1, grid = select_threshold_on_validation(
                probabilities_val, y_val_event
            )
            for row in grid:
                threshold_rows.append(
                    {"task": "trajectory", "k": k, "strategy": strategy, **row}
                )

            metrics = compute_metrics(probabilities_test, y_test_event, threshold)
            metrics.update(
                {
                    "task": "trajectory",
                    "k": k,
                    "strategy": strategy,
                    "target": f"max(A(t+1)..A(t+{k}))",
                    "validation_f1_at_threshold": validation_f1,
                    "val_pos_samples": int(y_val_event.sum()),
                    "val_neg_samples": int(len(y_val_event) - y_val_event.sum()),
                    "train_samples": int(len(y_train_event)),
                    "train_pos_samples": int(y_train_event.sum()),
                    "converged": True if strategy == "max_aggregate" else bool(direct_converged),
                    "n_iter": -1 if strategy == "max_aggregate" else int(np.max(direct_model.n_iter_)),
                }
            )
            metrics_rows.append(metrics)

            save_confusion_matrix(
                output_dir / f"confusion_matrix_trajectory_{strategy}_k{k}.csv", metrics
            )

            label = f"{strategy} K={k}"
            print(
                f"{label:<24}{metrics['threshold']:>6.2f}{metrics['f1']:>9.4f}"
                f"{metrics['precision']:>11.4f}{metrics['recall']:>9.4f}{metrics['roc_auc']:>10.4f}"
                f"{metrics['fpr']:>9.4f}{metrics['tp']:>7d}{metrics['fp']:>6d}"
                f"{metrics['tn']:>7d}{metrics['fn']:>7d}{validation_f1:>9.4f}"
            )

    # =========================================================
    # SAVE ARTIFACTS
    # =========================================================
    print("\n" + "=" * 96)
    print("3. ARTIFACTS")
    print("=" * 96)

    metrics_df = pd.DataFrame(metrics_rows)[
        [
            "task", "k", "strategy", "target", "threshold", "precision", "recall", "f1",
            "roc_auc", "fpr", "tp", "fp", "tn", "fn", "pos_samples", "neg_samples",
            "validation_f1_at_threshold", "val_pos_samples", "val_neg_samples",
            "train_samples", "train_pos_samples", "converged", "n_iter",
        ]
    ]
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "validation_threshold_search.csv", index=False
    )

    predictions = data["test"]["meta"][["timestamp", "source_day"]].copy()
    frozen_thresholds = {
        row["k"]: row["threshold"]
        for row in metrics_rows
        if row["task"] == "future"
    }
    for k in horizons:
        predictions[f"target_t+{k}"] = A_test[:, k - 1].astype(int)
        predictions[f"prob_t+{k}"] = test_probabilities[k]
        predictions[f"pred_t+{k}"] = (
            test_probabilities[k] >= frozen_thresholds[k]
        ).astype(int)
    for k in trajectory_horizons:
        predictions[f"target_traj_1..{k}"] = A_test[:, :k].max(axis=1).astype(int)
        predictions[f"prob_traj_max_1..{k}"] = np.max(
            np.column_stack([test_probabilities[j] for j in range(1, k + 1)]), axis=1
        )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    config = {
        "baseline": "logistic_regression",
        "baseline_display_name": "Logistic Regression (latest-state)",
        "is_world_model": False,
        "formulation": {
            "input": "S_t in R^52 = X[:, -1, :] (most recent observed state; no temporal sequence)",
            "target": "A(t+k) = A_binary_multistep[:, k-1]",
            "models": "one independent LogisticRegression per horizon",
        },
        "horizons": horizons,
        "trajectory_horizons": trajectory_horizons,
        "model_config": MODEL_CONFIG,
        "threshold_method": {
            "grid": f"np.arange({THRESHOLD_LO}, {THRESHOLD_HI}, {THRESHOLD_STEP})",
            "objective": "maximise F1 on validation (first strict maximum wins ties)",
            "selection_source": "validation",
            "test_labels_used_for_selection": False,
            "same_convention_as_v4_reference": True,
        },
        "thresholds": {
            f"k{row['k']}": row["threshold"]
            for row in metrics_rows
            if row["task"] == "future"
        },
        "trajectory_thresholds": {
            f"{row['strategy']}_k{row['k']}": row["threshold"]
            for row in metrics_rows
            if row["task"] == "trajectory"
        },
        "models": model_records,
        "feature_order": features,
        "git_commit": git_value("rev-parse", "HEAD"),
    }
    with open(output_dir / "baseline_config.json", "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    for filename in (
        "metrics.csv",
        "validation_threshold_search.csv",
        "test_predictions.csv",
        "baseline_config.json",
    ):
        print(f"  wrote {(output_dir / filename).relative_to(REPO_ROOT)}")
    print(f"  wrote {len(model_records)} model .pkl files")
    print(f"  wrote {len(horizons) + 2 * len(trajectory_horizons)} confusion matrices")

    # =========================================================
    # MANIFEST
    # =========================================================
    if not args.no_manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path

        v4_commit = None
        if V4_MANIFEST.exists():
            v4_commit = json.loads(V4_MANIFEST.read_text())["git"]["commit"]

        def split_entry(split):
            meta = data[split]["meta"]
            return {
                "sequences": int(len(meta)),
                "source_days": sorted(meta["source_day"].unique().tolist()),
                "first_target_timestamp": str(meta["timestamp"].min()),
                "last_target_timestamp": str(meta["timestamp"].max()),
                "X_file": str((benchmark_dir / f"X_{split}.npy").relative_to(REPO_ROOT)),
                "X_sha256": sha256_file(benchmark_dir / f"X_{split}.npy"),
                "labels_file": str(
                    (benchmark_dir / f"A_binary_multistep_{split}.npy").relative_to(REPO_ROOT)
                ),
                "labels_sha256": sha256_file(
                    benchmark_dir / f"A_binary_multistep_{split}.npy"
                ),
            }

        manifest = {
            "baseline": "logistic_regression",
            "baseline_display_name": "Logistic Regression (latest-state)",
            "is_world_model": False,
            "purpose": "Conventional static classifier: how much future-attack prediction "
                       "is achievable without a learned temporal world model.",
            "created_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "benchmark_directory": str(benchmark_dir.relative_to(REPO_ROOT)),
            "feature_dimension": EXPECTED_FEATURE_DIM,
            "base_feature_count": EXPECTED_BASE_FEATURES,
            "relative_feature_count": len(relative_features),
            "feature_schema": str(feature_path.relative_to(REPO_ROOT)),
            "feature_schema_sha256": sha256_file(feature_path),
            "feature_schema_json": str(FEATURE_SCHEMA_JSON.relative_to(REPO_ROOT)),
            "feature_schema_json_sha256": sha256_file(FEATURE_SCHEMA_JSON)
            if FEATURE_SCHEMA_JSON.exists() else None,
            "feature_order": features,
            "input_representation": "S_t in R^52 = X[:, -1, :]; the 10x52 sequence is NOT used",
            "input_scaler": str(scaler_path.relative_to(REPO_ROOT)),
            "input_scaler_sha256": sha256_file(scaler_path),
            "input_scaler_class": type(input_scaler).__name__,
            "input_scaler_n_samples_seen": int(input_scaler.n_samples_seen_),
            "input_scaler_fitted_on": "train partition only (Feb-14/15/28 first 85%); "
                                      "applied at benchmark build time, never refit here",
            "train_split": split_entry("train"),
            "validation_split": split_entry("val"),
            "test_split": split_entry("test"),
            "split_policy": "chronological; train days Feb-14/Feb-15/Feb-28 split 85%/15% "
                            "per day; Mar-2-2018 held out entirely",
            "target_alignment": {f"k{k}": f"t+{k} (A_binary_multistep[:, {k - 1}])" for k in horizons},
            "target_alignment_verified": "A[:, 0] equals metadata attack_present at t+1 for "
                                         "all three splits",
            "model_config": MODEL_CONFIG,
            "models": model_records,
            "threshold_method": config["threshold_method"],
            "thresholds": config["thresholds"],
            "trajectory": {
                "target": "max(A(t+1) .. A(t+k))",
                "primary_strategy": "max_aggregate — max over per-horizon LR probabilities "
                                    "(matches the V4 reference MAX convention)",
                "secondary_strategy": "direct_target — one LR trained on the max target "
                                      "(pre-existing src/evaluate_baselines.py formulation)",
                "horizons": trajectory_horizons,
                "thresholds": config["trajectory_thresholds"],
            },
            "metrics": {
                f"{row['task']}_{row['strategy']}_k{row['k']}": {
                    key: row[key]
                    for key in (
                        "threshold", "precision", "recall", "f1", "roc_auc", "fpr",
                        "tp", "fp", "tn", "fn", "pos_samples", "neg_samples",
                        "validation_f1_at_threshold",
                    )
                }
                for row in metrics_rows
            },
            "artifacts": {
                "directory": str(output_dir.relative_to(REPO_ROOT)),
                "metrics_csv": str((output_dir / "metrics.csv").relative_to(REPO_ROOT)),
                "metrics_csv_sha256": sha256_file(output_dir / "metrics.csv"),
                "threshold_search_csv": str(
                    (output_dir / "validation_threshold_search.csv").relative_to(REPO_ROOT)
                ),
                "test_predictions_csv": str(
                    (output_dir / "test_predictions.csv").relative_to(REPO_ROOT)
                ),
                "config_json": str((output_dir / "baseline_config.json").relative_to(REPO_ROOT)),
            },
            "reproduction_command": "python src/train_logistic_regression_baseline.py",
            "verification_command": "python src/verify_logistic_regression_baseline.py",
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "v4_reference_manifest": str(V4_MANIFEST.relative_to(REPO_ROOT)),
            "v4_reference_commit": v4_commit,
            "v4_artifacts_modified": False,
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        print(f"  wrote {manifest_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 96)
    print("LOGISTIC REGRESSION BASELINE COMPLETE")
    print("=" * 96)
    print("No V4 artifact was read for modification or written to.")
    print("All thresholds were selected on validation only; test labels were used")
    print("exclusively to compute the final reported metrics.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
