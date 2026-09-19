"""
=============================================================
PERSISTENCE HYPOTHESIS — VERIFICATION (READ-ONLY)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Independently re-derives the persistence-hypothesis analysis in
results/persistence_hypothesis/ and confirms it sits on the same frozen
substrate as Tasks 3 and 4.

Checks:
    Test substrate        3,115 windows, Mar-2 only, unchanged benchmark hashes
    V4 integrity          V4 rollout re-run from the frozen checkpoint matches
    LR integrity          stored probabilities match the frozen LR artifact
    Persistence integrity predictions equal A(t) re-read from the raw day CSV
    Threshold integrity   all thresholds equal the frozen Task 1 / Task 2 values
    Label integrity       A(t+k) re-derived from the raw day CSV for k = 1..5
    Transition accounting PP + ONSET + OFFSET + AA == 3,115 for every model/horizon,
                          and the counts agree with persistence_statistics.csv
    Task 3 reproduction   overall F1 per model/horizon matches Task 3 exactly
    Task 4 reproduction   TP/FP/TN/FN per model/horizon matches Task 4 exactly
    No tuning             frozen thresholds differ from the test-optimal ones and
                          no baseline artifact hash changed

Usage:
    python src/verify_persistence_hypothesis.py

Exit code: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_standardized_comparison import (  # noqa: E402
    BENCHMARK_DIR,
    MODEL_LR,
    MODEL_PERSISTENCE,
    MODEL_V4,
    POINT_HORIZONS,
    STATES_DIR,
    fetch_labels_at_time_t,
    load_benchmark,
    sha256_file,
    v4_scores,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "results" / "persistence_hypothesis"
EVENTS_CSV = REPO_ROOT / "results" / "error_analysis" / "error_events.csv"
TASK3_CSV = REPO_ROOT / "results" / "standardized_baseline_comparison.csv"
TASK4_CSV = REPO_ROOT / "results" / "error_analysis" / "horizon_error_summary.csv"
V4_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
LR_MANIFEST = REPO_ROOT / "docs" / "logistic_regression_baseline_manifest.json"
LR_PREDICTIONS = REPO_ROOT / "models" / "logistic_regression_baseline" / "test_predictions.csv"
SUMMARY_JSON = REPO_ROOT / "results" / "persistence_hypothesis_summary.json"

TEST_DAY = "Friday-02-03-2018"
EXPECTED_WINDOWS = 3115
TOLERANCE = 1e-9

REQUIRED_TABLES = [
    "persistence_statistics.csv", "transition_decomposition.csv",
    "transition_model_performance.csv", "persistence_excess.csv",
    "transition_only_metrics.csv", "onset_transition_analysis.csv",
    "offset_transition_analysis.csv", "v4_k1_to_k5_correction_decomposition.csv",
    "pre_transition_signal.csv", "current_state_conditioning.csv",
    "persistence_scorecard.csv",
]


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
        print("PERSISTENCE HYPOTHESIS VERIFICATION")
        print("-" * 35)
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


def main() -> int:
    print("=" * 78)
    print("PERSISTENCE HYPOTHESIS — VERIFICATION (read-only)")
    print("=" * 78)
    print(f"Repository : {REPO_ROOT}")
    print(f"Analysis   : {ANALYSIS_DIR}")

    report = Report()

    missing = [name for name in REQUIRED_TABLES if not (ANALYSIS_DIR / name).exists()]
    if missing or not SUMMARY_JSON.exists():
        report.add("Artifacts present", False, f"missing: {', '.join(missing) or SUMMARY_JSON.name}")
        report.render()
        return 1
    report.add("Artifacts present", True, f"{len(REQUIRED_TABLES)} tables + summary JSON")

    events = pd.read_csv(EVENTS_CSV, parse_dates=["forecast_time", "target_time"])
    events = events[events["model"].isin([MODEL_PERSISTENCE, MODEL_LR, MODEL_V4])]
    data = load_benchmark()
    meta = data["test"]["meta"]
    A_test = data["test"]["A"]

    v4_manifest = json.loads(V4_MANIFEST.read_text())
    lr_manifest = json.loads(LR_MANIFEST.read_text())
    thresholds = {
        MODEL_V4: {k: float(v4_manifest["threshold"]["per_horizon_future_prediction"][str(k)])
                   for k in POINT_HORIZONS},
        MODEL_LR: {k: float(lr_manifest["thresholds"][f"k{k}"]) for k in POINT_HORIZONS},
    }

    stats = pd.read_csv(ANALYSIS_DIR / "persistence_statistics.csv")
    decomposition = pd.read_csv(ANALYSIS_DIR / "transition_decomposition.csv")
    excess = pd.read_csv(ANALYSIS_DIR / "persistence_excess.csv")

    # ---------------------------------------------------------
    # Test substrate
    # ---------------------------------------------------------
    problems = []
    if len(meta) != EXPECTED_WINDOWS:
        problems.append(f"benchmark holds {len(meta)} windows")
    if set(events["source_day"].unique()) != {TEST_DAY}:
        problems.append("events reference a day other than the held-out day")
    for (model, k), group in events.groupby(["model", "horizon"]):
        if len(group) != EXPECTED_WINDOWS:
            problems.append(f"{model} K={k}: {len(group)} rows")
    for name, key in (("X_test.npy", "X"), ("A_binary_multistep_test.npy", "A_binary_multistep")):
        if sha256_file(BENCHMARK_DIR / name) != v4_manifest["dataset"]["splits"]["test"][key]["sha256"]:
            problems.append(f"{name} hash changed")
    report.add("Test substrate", not problems,
               "; ".join(problems) if problems else f"{EXPECTED_WINDOWS} Mar-2 windows, benchmark hashes unchanged")

    # ---------------------------------------------------------
    # V4 integrity
    # ---------------------------------------------------------
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    v4_probabilities, _ = v4_scores(data, device)
    problems = []
    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_V4) & (events["horizon"] == k)].sort_values("sample_index")
        if not np.allclose(group["predicted_probability"].to_numpy(), v4_probabilities["test"][:, k], atol=1e-12):
            problems.append(f"K={k}: probabilities differ from a fresh rollout")
        expected = (v4_probabilities["test"][:, k] >= thresholds[MODEL_V4][k]).astype(int)
        if not np.array_equal(group["predicted_label"].to_numpy(), expected):
            problems.append(f"K={k}: labels are not the frozen-threshold decision")
    report.add("V4 integrity", not problems,
               "; ".join(problems) if problems else "rollout from the frozen checkpoint reproduces every score")

    # ---------------------------------------------------------
    # LR integrity
    # ---------------------------------------------------------
    stored_lr = pd.read_csv(LR_PREDICTIONS)
    problems = []
    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_LR) & (events["horizon"] == k)].sort_values("sample_index")
        frozen = stored_lr[f"prob_t+{k}"].to_numpy()
        if not np.allclose(group["predicted_probability"].to_numpy(), frozen, atol=1e-12):
            problems.append(f"K={k}: probabilities differ from the frozen artifact")
        if not np.array_equal(group["predicted_label"].to_numpy(),
                              (frozen >= thresholds[MODEL_LR][k]).astype(int)):
            problems.append(f"K={k}: labels are not the frozen-threshold decision")
    for key, record in lr_manifest["models"].items():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            problems.append(f"LR model {key} hash changed")
    report.add("LR integrity", not problems,
               "; ".join(problems) if problems else "matches the frozen LR models and predictions")

    # ---------------------------------------------------------
    # Persistence integrity
    # ---------------------------------------------------------
    label_t = fetch_labels_at_time_t(meta).astype(int)
    problems = []
    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_PERSISTENCE) & (events["horizon"] == k)].sort_values("sample_index")
        if not np.array_equal(group["predicted_label"].to_numpy(), label_t):
            problems.append(f"K={k}: prediction != A(t)")
        if not np.array_equal(group["label_at_time_t"].to_numpy(), label_t):
            problems.append(f"K={k}: stored A(t) differs from the raw day CSV")
        row = stats[stats["horizon"] == k].iloc[0]
        recomputed = float((A_test[:, k - 1].astype(int) == label_t).mean())
        if abs(recomputed - float(row["persistence_accuracy"])) > TOLERANCE:
            problems.append(f"K={k}: persistence accuracy not reproducible")
    report.add("Persistence integrity", not problems,
               "; ".join(problems) if problems else "predictions equal A(t); persistence statistics reproduce")

    # ---------------------------------------------------------
    # Threshold integrity
    # ---------------------------------------------------------
    problems = []
    for model in (MODEL_V4, MODEL_LR):
        for k in POINT_HORIZONS:
            stored = events[(events["model"] == model) & (events["horizon"] == k)]["threshold"].unique()
            if len(stored) != 1 or abs(float(stored[0]) - thresholds[model][k]) > TOLERANCE:
                problems.append(f"{model} K={k}: {stored} != {thresholds[model][k]}")
    if not events[events["model"] == MODEL_PERSISTENCE]["threshold"].isna().all():
        problems.append("persistence carries a threshold")
    report.add("Threshold integrity", not problems,
               "; ".join(problems) if problems else "8 frozen thresholds unchanged; persistence has none")

    # ---------------------------------------------------------
    # Label integrity
    # ---------------------------------------------------------
    raw = pd.read_csv(
        STATES_DIR / f"{TEST_DAY}_TrafficForML_CICFlowMeter_states_v2.csv",
        usecols=["window_start", "attack_present"],
    )
    raw["window_start"] = pd.to_datetime(raw["window_start"])
    lookup = dict(zip(raw["window_start"], raw["attack_present"]))
    problems = []
    for k in range(1, 6):
        expected = np.array(
            [lookup[ts + pd.Timedelta(seconds=10 * (k - 1))] for ts in meta["timestamp"]], dtype=int
        )
        if not np.array_equal(expected, A_test[:, k - 1].astype(int)):
            problems.append(f"A(t+{k}) differs from the raw day CSV")
    report.add("Label integrity", not problems,
               "; ".join(problems) if problems else "A(t+1..t+5) re-derived from the raw day CSV")

    # ---------------------------------------------------------
    # Transition accounting
    # ---------------------------------------------------------
    problems = []
    for (model, k), group in decomposition.groupby(["model", "horizon"]):
        total = int(group["count"].sum())
        if total != EXPECTED_WINDOWS:
            problems.append(f"{model} K={k}: categories sum to {total}")

        row = stats[stats["horizon"] == k].iloc[0]
        onset = int(group[group["transition_category"] == "ONSET"]["count"].iloc[0])
        offset = int(group[group["transition_category"] == "OFFSET"]["count"].iloc[0])
        if onset != int(row["n_current_benign_future_attack"]):
            problems.append(f"{model} K={k}: ONSET count disagrees with persistence_statistics")
        if offset != int(row["n_current_attack_future_benign"]):
            problems.append(f"{model} K={k}: OFFSET count disagrees with persistence_statistics")
        if onset + offset != int(row["transition_count"]):
            problems.append(f"{model} K={k}: transition total inconsistent")

        # Persistence must be right on PP/AA and wrong on ONSET/OFFSET, by construction
        if model == MODEL_PERSISTENCE:
            for category in ("PP", "AA"):
                entry = group[group["transition_category"] == category].iloc[0]
                if int(entry["incorrect"]) != 0:
                    problems.append(f"persistence K={k}: {category} has errors")
            for category in ("ONSET", "OFFSET"):
                entry = group[group["transition_category"] == category].iloc[0]
                if int(entry["correct"]) != 0:
                    problems.append(f"persistence K={k}: {category} has correct predictions")
    report.add("Transition accounting", not problems,
               "; ".join(problems[:3]) if problems else
               "PP+ONSET+OFFSET+AA = 3115 everywhere; persistence is exact on PP/AA and wrong on all transitions")

    # ---------------------------------------------------------
    # Task 3 reproduction
    # ---------------------------------------------------------
    task3 = pd.read_csv(TASK3_CSV)
    task3_point = task3[task3["evaluation_type"] == "point_forecast"]
    problems = []
    for _, row in excess.iterrows():
        reference = task3_point[
            (task3_point["model"] == row["model"]) & (task3_point["horizon"] == row["horizon"])
        ]
        if reference.empty:
            problems.append(f"{row['model']} K={row['horizon']} absent from Task 3")
            continue
        if abs(float(row["overall_f1"]) - float(reference.iloc[0]["f1"])) > TOLERANCE:
            problems.append(f"{row['model']} K={row['horizon']}: F1 {row['overall_f1']} vs {reference.iloc[0]['f1']}")
    report.add("Task 3 reproduction", not problems,
               "; ".join(problems[:3]) if problems else "12 overall F1 values match the standardized comparison")

    # ---------------------------------------------------------
    # Task 4 reproduction
    # ---------------------------------------------------------
    task4 = pd.read_csv(TASK4_CSV)
    problems = []
    for (model, k), group in events.groupby(["model", "horizon"]):
        group = group.sort_values("sample_index")
        tn, fp, fn, tp = confusion_matrix(
            group["actual_label"], group["predicted_label"], labels=[0, 1]
        ).ravel()
        reference = task4[(task4["model"] == model) & (task4["horizon"] == k)]
        if reference.empty:
            problems.append(f"{model} K={k} absent from Task 4")
            continue
        reference = reference.iloc[0]
        for name, value in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn)):
            if int(reference[name]) != int(value):
                problems.append(f"{model} K={k} {name}: {value} vs {reference[name]}")
    report.add("Task 4 reproduction", not problems,
               "; ".join(problems[:3]) if problems else "12 confusion matrices match the error analysis")

    # ---------------------------------------------------------
    # No tuning
    # ---------------------------------------------------------
    problems = []
    grid = np.arange(0.01, 0.99, 0.01)
    for model in (MODEL_V4, MODEL_LR):
        for k in POINT_HORIZONS:
            group = events[(events["model"] == model) & (events["horizon"] == k)].sort_values("sample_index")
            score = group["predicted_probability"].to_numpy()
            actual = group["actual_label"].to_numpy()
            best_threshold, best_f1 = 0.5, -1.0
            for threshold in grid:
                value = f1_score(actual, (score >= threshold).astype(int), zero_division=0)
                if value > best_f1:
                    best_f1, best_threshold = float(value), float(threshold)
            if abs(best_threshold - thresholds[model][k]) < TOLERANCE:
                problems.append(f"{model} K={k}: frozen threshold equals the test-optimal threshold")

    if sha256_file(REPO_ROOT / v4_manifest["checkpoint"]["path"]) != v4_manifest["checkpoint"]["sha256"]:
        problems.append("V4 checkpoint hash changed")

    summary = json.loads(SUMMARY_JSON.read_text())
    for flag in ("no_model_retrained", "no_threshold_changed", "no_artifact_modified"):
        if summary["verification"].get(flag) is not True:
            problems.append(f"summary does not assert {flag}")

    report.add("No tuning", not problems,
               "; ".join(problems[:3]) if problems else
               "frozen thresholds differ from test-optimal; checkpoint and model hashes unchanged")

    report.note(
        "Persistence receives A(t) directly; its exactness on PP/AA windows is structural, "
        "not evidence that a feature-based model could match it."
    )
    report.note(
        f"Transition subsets hold {int(stats['transition_count'].min())}-"
        f"{int(stats['transition_count'].max())} windows — descriptive only, no significance testing."
    )

    report.render()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
