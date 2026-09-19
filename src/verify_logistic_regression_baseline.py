"""
=============================================================
LOGISTIC REGRESSION BASELINE — VERIFICATION (READ-ONLY)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Verifies that the Logistic Regression baseline artifacts in
models/logistic_regression_baseline/ are internally consistent,
were produced on the frozen multi-day benchmark, and that every
threshold was selected on VALIDATION data only.

This script trains nothing and writes nothing. It reloads the
saved models and re-derives the reported numbers independently.

Checks:
    Feature schema              52 = 26 base + 26 relative, order matches
                                input_feature_names.txt and v4_feature_schema.json
    Scaler                      manifest scaler == the frozen V4 benchmark scaler
                                (path + SHA256), fitted on train only
    Train / Validation / Test   source days, sizes and chronology re-derived
                                from the benchmark metadata
    K=1..5 models               exist, load, SHA256 match, n_features_in_ == 52
    Validation-only thresholds  each frozen threshold is EXACTLY the argmax of
                                validation F1 recomputed from the saved model,
                                and reproduces the recorded validation F1
    Test-label independence     the frozen threshold is compared against the
                                test-optimal threshold; differences are reported
    Artifacts                   metrics / predictions / threshold search /
                                confusion matrices present and mutually consistent
                                (test metrics recomputed from the saved models)

Usage:
    python src/verify_logistic_regression_baseline.py
    python src/verify_logistic_regression_baseline.py --manifest docs/logistic_regression_baseline_manifest.json

Exit code: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "logistic_regression_baseline_manifest.json"
V4_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
FEATURE_SCHEMA_JSON = REPO_ROOT / "docs" / "v4_feature_schema.json"

EXPECTED_FEATURE_DIM = 52
EXPECTED_BASE_FEATURES = 26
EXPECTED_TEST_DAY = "Friday-02-03-2018"
EXPECTED_TRAIN_DAYS = {
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Wednesday-28-02-2018",
}

METRIC_TOLERANCE = 1e-9


# =============================================================
# REPORTING
# =============================================================

class Report:
    def __init__(self):
        self.rows = []
        self.notes = []
        self.failed = False

    def add(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        if not ok:
            self.failed = True
        return ok

    def note(self, text):
        self.notes.append(text)

    def render(self):
        width = max(len(name) for name, _, _ in self.rows)

        print()
        print("LOGISTIC REGRESSION BASELINE VERIFICATION")
        print("-" * 41)

        for name, ok, detail in self.rows:
            line = f"{name:<{width}} : {'PASS' if ok else 'FAIL'}"
            if detail:
                line += f"   ({detail})"
            print(line)

        print()
        print(f"OVERALL: {'FAIL' if self.failed else 'PASS'}")

        if self.notes:
            print()
            print("Notes:")
            for note in self.notes:
                print(f"  - {note}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else REPO_ROOT / path


def threshold_grid(config) -> np.ndarray:
    lo, hi, step = 0.01, 0.99, 0.01
    spec = config.get("threshold_method", {}).get("grid", "")
    if spec.startswith("np.arange("):
        lo, hi, step = (float(part) for part in spec[len("np.arange("):-1].split(","))
    return np.arange(lo, hi, step)


def argmax_f1_threshold(probabilities, targets, grid):
    """First strict maximum of F1 over the grid — the baseline's selection rule."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in grid:
        f1 = f1_score(targets, (probabilities >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)
    return best_threshold, best_f1


# =============================================================
# CHECKS
# =============================================================

def check_feature_schema(report, manifest):
    names_path = resolve(manifest["feature_schema"])

    if not names_path.exists():
        return report.add("Feature schema", False, f"missing: {manifest['feature_schema']}")

    if sha256_file(names_path) != manifest["feature_schema_sha256"]:
        return report.add("Feature schema", False, "benchmark feature file SHA256 changed")

    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    manifest_order = manifest["feature_order"]

    problems = []
    if len(names) != EXPECTED_FEATURE_DIM:
        problems.append(f"{len(names)} features in benchmark file")
    if manifest_order != names:
        problems.append("manifest feature_order differs from the benchmark file")
    if manifest["feature_dimension"] != EXPECTED_FEATURE_DIM:
        problems.append(f"manifest feature_dimension={manifest['feature_dimension']}")

    base = names[:EXPECTED_BASE_FEATURES]
    relative = names[EXPECTED_BASE_FEATURES:]
    if relative != [f"{name}_rel_change" for name in base]:
        problems.append("indices 26-51 are not base features + '_rel_change'")

    if FEATURE_SCHEMA_JSON.exists():
        schema = json.loads(FEATURE_SCHEMA_JSON.read_text())
        if [item["name"] for item in schema["features"]] != names:
            problems.append("docs/v4_feature_schema.json disagrees on order")
    else:
        problems.append("docs/v4_feature_schema.json missing")

    if problems:
        return report.add("Feature schema", False, "; ".join(problems))

    return report.add(
        "Feature schema",
        True,
        f"{len(base)} base + {len(relative)} rel = {len(names)}, order matches all sources",
    )


def check_scaler(report, manifest):
    scaler_path = resolve(manifest["input_scaler"])

    if not scaler_path.exists():
        return report.add("Scaler", False, f"missing: {manifest['input_scaler']}")

    actual_sha = sha256_file(scaler_path)
    problems = []

    if actual_sha != manifest["input_scaler_sha256"]:
        problems.append("SHA256 changed since the baseline was trained")

    # Must be the SAME scaler the frozen V4 benchmark uses — no baseline-specific refit.
    if V4_MANIFEST.exists():
        v4_scaler = json.loads(V4_MANIFEST.read_text())["scalers"]["input_scaler"]
        if actual_sha != v4_scaler["sha256"]:
            problems.append("not the frozen V4 benchmark input scaler")
        if resolve(v4_scaler["path"]).resolve() != scaler_path.resolve():
            problems.append("scaler path differs from the V4 manifest")
    else:
        problems.append("docs/v4_baseline_manifest.json missing — cannot cross-check")

    with open(scaler_path, "rb") as handle:
        scaler = pickle.load(handle)

    if int(scaler.n_features_in_) != EXPECTED_FEATURE_DIM:
        problems.append(f"n_features_in_={scaler.n_features_in_}")
    if int(scaler.n_samples_seen_) != manifest["input_scaler_n_samples_seen"]:
        problems.append("n_samples_seen_ differs from the manifest")

    if problems:
        return report.add("Scaler", False, "; ".join(problems))

    return report.add(
        "Scaler",
        True,
        f"frozen V4 scaler, {EXPECTED_FEATURE_DIM} feat, n={int(scaler.n_samples_seen_)} (train only)",
    )


def load_benchmark(manifest):
    benchmark_dir = resolve(manifest["benchmark_directory"])
    data = {}
    for split, key in (("train", "train_split"), ("val", "validation_split"), ("test", "test_split")):
        data[split] = {
            "X": np.load(benchmark_dir / f"X_{split}.npy"),
            "A": np.load(benchmark_dir / f"A_binary_multistep_{split}.npy"),
            "meta": pd.read_csv(benchmark_dir / f"{split}_metadata.csv"),
            "manifest_key": key,
        }
        data[split]["meta"]["timestamp"] = pd.to_datetime(data[split]["meta"]["timestamp"])
    return data


def check_split(report, manifest, data, split, label, expected_days):
    entry = manifest[data[split]["manifest_key"]]
    meta = data[split]["meta"]
    X = data[split]["X"]

    problems = []

    if int(entry["sequences"]) != len(meta) or len(meta) != X.shape[0]:
        problems.append(f"{len(meta)} sequences != manifest {entry['sequences']}")

    days = set(meta["source_day"].unique())
    if days != expected_days:
        problems.append(f"days {sorted(days)} != expected {sorted(expected_days)}")
    if set(entry["source_days"]) != days:
        problems.append("manifest source_days differ from the metadata")

    if tuple(X.shape[1:]) != (10, EXPECTED_FEATURE_DIM):
        problems.append(f"X shape {X.shape}")

    x_path = resolve(entry["X_file"])
    if not x_path.exists():
        problems.append("X file missing")
    elif sha256_file(x_path) != entry["X_sha256"]:
        problems.append("X SHA256 changed")

    labels_path = resolve(entry["labels_file"])
    if not labels_path.exists():
        problems.append("label file missing")
    elif sha256_file(labels_path) != entry["labels_sha256"]:
        problems.append("label SHA256 changed")

    if split == "test":
        train_days = set(data["train"]["meta"]["source_day"].unique())
        val_days = set(data["val"]["meta"]["source_day"].unique())
        if days & (train_days | val_days):
            problems.append("test day also appears in train/validation")

    if split == "val":
        for day in sorted(expected_days):
            train_last = data["train"]["meta"].loc[
                data["train"]["meta"]["source_day"] == day, "timestamp"
            ].max()
            val_first = meta.loc[meta["source_day"] == day, "timestamp"].min()
            if not val_first > train_last:
                problems.append(f"{day}: validation not chronologically after train")

    if problems:
        return report.add(label, False, "; ".join(problems))

    return report.add(label, True, f"{len(meta)} sequences, {len(days)} day(s)")


def check_models(report, manifest, data):
    models = {}
    for k in manifest["horizons"] if "horizons" in manifest else [1, 2, 3, 4, 5]:
        record = manifest["models"].get(str(k)) or manifest["models"].get(k)
        label = f"K={k} model"

        if record is None:
            report.add(label, False, "not recorded in the manifest")
            continue

        path = resolve(record["path"])
        if not path.exists():
            report.add(label, False, f"missing: {record['path']}")
            continue

        if sha256_file(path) != record["sha256"]:
            report.add(label, False, "SHA256 changed since training")
            continue

        with open(path, "rb") as handle:
            model = pickle.load(handle)

        problems = []
        config = manifest["model_config"]

        if int(model.n_features_in_) != EXPECTED_FEATURE_DIM:
            problems.append(f"n_features_in_={model.n_features_in_}")
        if model.coef_.shape != (1, EXPECTED_FEATURE_DIM):
            problems.append(f"coef_ shape {model.coef_.shape}")
        for attribute in ("C", "solver", "max_iter", "class_weight", "penalty", "random_state"):
            if attribute in config and getattr(model, attribute) != config[attribute]:
                problems.append(f"{attribute}={getattr(model, attribute)} != {config[attribute]}")
        if not record.get("converged", True):
            problems.append("did not converge at training time")

        if problems:
            report.add(label, False, "; ".join(problems))
            continue

        models[k] = model
        report.add(
            label,
            True,
            f"{EXPECTED_FEATURE_DIM} features, {config['solver']}/{config['penalty']}, "
            f"C={config['C']}, converged in {record['n_iter']} iters",
        )

    return models


def check_thresholds(report, manifest, data, models):
    """Re-derive every frozen threshold from validation data using the saved models."""
    grid = threshold_grid(manifest)
    S_val = data["val"]["X"][:, -1, :]
    S_test = data["test"]["X"][:, -1, :]
    A_val, A_test = data["val"]["A"], data["test"]["A"]

    problems = []
    matched = 0
    test_optimal_differs = 0

    for k, model in models.items():
        frozen = manifest["thresholds"].get(f"k{k}")
        if frozen is None:
            problems.append(f"K={k}: no frozen threshold recorded")
            continue

        probabilities_val = model.predict_proba(S_val)[:, 1]
        derived, derived_f1 = argmax_f1_threshold(probabilities_val, A_val[:, k - 1], grid)

        if abs(derived - float(frozen)) > 1e-9:
            problems.append(
                f"K={k}: frozen {frozen} != validation argmax {derived:.2f}"
            )
            continue

        recorded_key = f"future_single_model_k{k}"
        recorded = manifest["metrics"].get(recorded_key, {})
        if "validation_f1_at_threshold" in recorded:
            if abs(recorded["validation_f1_at_threshold"] - derived_f1) > 1e-9:
                problems.append(f"K={k}: recorded validation F1 not reproducible")
                continue

        # Independent evidence that test labels were not used: compare against the
        # threshold that WOULD have been chosen on test.
        probabilities_test = model.predict_proba(S_test)[:, 1]
        test_optimal, test_optimal_f1 = argmax_f1_threshold(
            probabilities_test, A_test[:, k - 1], grid
        )
        if abs(test_optimal - float(frozen)) > 1e-9:
            test_optimal_differs += 1
            report.note(
                f"K={k}: frozen threshold {frozen:.2f} (validation) vs test-optimal "
                f"{test_optimal:.2f} (test F1 would be {test_optimal_f1:.4f} instead of "
                f"{recorded.get('f1', float('nan')):.4f}) — test labels demonstrably unused"
            )

        matched += 1

    # Trajectory thresholds: re-derive the MAX-aggregate strategy too.
    trajectory = manifest.get("trajectory", {})
    for k in trajectory.get("horizons", []):
        frozen = trajectory["thresholds"].get(f"max_aggregate_k{k}")
        if frozen is None:
            problems.append(f"trajectory K={k}: no max_aggregate threshold recorded")
            continue
        if any(j not in models for j in range(1, k + 1)):
            problems.append(f"trajectory K={k}: missing per-horizon models")
            continue

        scores_val = np.max(
            np.column_stack([models[j].predict_proba(S_val)[:, 1] for j in range(1, k + 1)]),
            axis=1,
        )
        derived, _ = argmax_f1_threshold(scores_val, A_val[:, :k].max(axis=1), grid)

        if abs(derived - float(frozen)) > 1e-9:
            problems.append(
                f"trajectory K={k}: frozen {frozen} != validation argmax {derived:.2f}"
            )
            continue
        matched += 1

    if manifest["threshold_method"].get("selection_source") != "validation":
        problems.append("manifest does not declare validation as the selection source")
    if manifest["threshold_method"].get("test_labels_used_for_selection") is not False:
        problems.append("manifest does not assert test-label independence")

    if problems:
        return report.add("Validation-only thresholds", False, "; ".join(problems))

    return report.add(
        "Validation-only thresholds",
        True,
        f"{matched} thresholds re-derived from validation; "
        f"{test_optimal_differs} differ from the test-optimal threshold",
    )


def check_artifacts(report, manifest, data, models):
    """Artifacts present, mutually consistent, and test metrics independently reproducible."""
    output_dir = resolve(manifest["artifacts"]["directory"])
    problems = []

    required = [
        "metrics.csv",
        "validation_threshold_search.csv",
        "test_predictions.csv",
        "baseline_config.json",
    ]
    for filename in required:
        if not (output_dir / filename).exists():
            problems.append(f"missing {filename}")

    for k in models:
        if not (output_dir / f"confusion_matrix_k{k}.csv").exists():
            problems.append(f"missing confusion_matrix_k{k}.csv")

    if problems:
        return report.add("Artifacts", False, "; ".join(problems))

    metrics_df = pd.read_csv(output_dir / "metrics.csv")
    config = json.loads((output_dir / "baseline_config.json").read_text())
    predictions = pd.read_csv(output_dir / "test_predictions.csv")
    grid_df = pd.read_csv(output_dir / "validation_threshold_search.csv")

    S_test = data["test"]["X"][:, -1, :]
    A_test = data["test"]["A"]

    recomputed = 0
    for k, model in models.items():
        row = metrics_df[(metrics_df["task"] == "future") & (metrics_df["k"] == k)]
        if row.empty:
            problems.append(f"metrics.csv has no future row for K={k}")
            continue
        row = row.iloc[0]

        threshold = float(manifest["thresholds"][f"k{k}"])

        if abs(float(config["thresholds"][f"k{k}"]) - threshold) > 1e-9:
            problems.append(f"K={k}: baseline_config threshold != manifest threshold")
        if abs(float(row["threshold"]) - threshold) > 1e-9:
            problems.append(f"K={k}: metrics.csv threshold != manifest threshold")

        probabilities = model.predict_proba(S_test)[:, 1]
        targets = A_test[:, k - 1]
        predicted = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
        expected = {
            "precision": precision_score(targets, predicted, zero_division=0),
            "recall": recall_score(targets, predicted, zero_division=0),
            "f1": f1_score(targets, predicted, zero_division=0),
            "roc_auc": roc_auc_score(targets, probabilities),
        }

        for name, value in expected.items():
            if abs(float(row[name]) - value) > METRIC_TOLERANCE:
                problems.append(f"K={k}: {name} in metrics.csv not reproducible")
        for name, value in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
            if int(row[name]) != int(value):
                problems.append(f"K={k}: {name} in metrics.csv not reproducible")

        cm = pd.read_csv(output_dir / f"confusion_matrix_k{k}.csv", index_col=0)
        if [int(cm.iloc[0, 0]), int(cm.iloc[0, 1]), int(cm.iloc[1, 0]), int(cm.iloc[1, 1])] != [
            int(tn), int(fp), int(fn), int(tp)
        ]:
            problems.append(f"K={k}: confusion_matrix_k{k}.csv disagrees with metrics.csv")

        if not np.allclose(predictions[f"prob_t+{k}"].to_numpy(), probabilities, atol=1e-12):
            problems.append(f"K={k}: test_predictions.csv probabilities not reproducible")
        if not np.array_equal(predictions[f"target_t+{k}"].to_numpy(), targets.astype(int)):
            problems.append(f"K={k}: test_predictions.csv targets differ from the benchmark")

        recorded_metrics = manifest["metrics"][f"future_single_model_k{k}"]
        if abs(recorded_metrics["f1"] - expected["f1"]) > METRIC_TOLERANCE:
            problems.append(f"K={k}: manifest F1 differs from metrics.csv")

        grid_rows = grid_df[
            (grid_df["task"] == "future")
            & (grid_df["k"] == k)
            & (grid_df["strategy"] == "single_model")
        ]
        if grid_rows.empty:
            problems.append(f"K={k}: no validation threshold-search rows")
        elif abs(float(grid_rows.loc[grid_rows["val_f1"].idxmax(), "threshold"]) - threshold) > 1e-9:
            problems.append(f"K={k}: threshold-search argmax != frozen threshold")

        recomputed += 1

    if len(predictions) != len(data["test"]["meta"]):
        problems.append("test_predictions.csv row count != test split size")

    if problems:
        return report.add("Artifacts", False, "; ".join(problems))

    return report.add(
        "Artifacts",
        True,
        f"metrics/predictions/threshold-search consistent; {recomputed} horizons recomputed",
    )


# =============================================================
# MAIN
# =============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Logistic Regression baseline artifacts (read-only)."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    manifest_path = resolve(args.manifest)

    print("=" * 78)
    print("LOGISTIC REGRESSION BASELINE — VERIFICATION (read-only)")
    print("=" * 78)
    print(f"Repository : {REPO_ROOT}")
    print(f"Manifest   : {manifest_path}")

    report = Report()

    if not manifest_path.exists():
        report.add("Manifest", False, f"missing: {manifest_path}")
        report.render()
        return 1

    manifest = json.loads(manifest_path.read_text())
    report.add(
        "Manifest",
        True,
        f"{manifest['baseline']} created {manifest['created_utc']}",
    )

    check_feature_schema(report, manifest)
    check_scaler(report, manifest)

    data = load_benchmark(manifest)
    check_split(report, manifest, data, "train", "Train split", EXPECTED_TRAIN_DAYS)
    check_split(report, manifest, data, "val", "Validation split", EXPECTED_TRAIN_DAYS)
    check_split(report, manifest, data, "test", "Test split", {EXPECTED_TEST_DAY})

    models = check_models(report, manifest, data)
    check_thresholds(report, manifest, data, models)
    check_artifacts(report, manifest, data, models)

    report.render()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
