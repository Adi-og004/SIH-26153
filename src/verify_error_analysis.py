"""
=============================================================
ERROR ANALYSIS VERIFICATION (READ-ONLY)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Independently re-derives the error analysis in results/error_analysis/
and confirms that it sits on exactly the same frozen substrate as the
Task 3 standardized comparison.

Checks:
    Test substrate          3115 unique windows per model/horizon, matching
                            the frozen benchmark test split
    Target alignment        actual_label == A_binary_multistep[:, k-1] and
                            target_time == metadata timestamp + (k-1)*10s
    Thresholds              every threshold equals the frozen Task 1 / Task 2
                            value; persistence carries none
    V4 prediction integrity V4 rollout re-run from the frozen checkpoint and
                            compared with the stored probabilities/labels
    LR prediction integrity stored probabilities compared with the frozen
                            models/logistic_regression_baseline artifact
    Persistence integrity   A(t) re-read from the raw day CSV; predictions
                            equal A(t) and every error is a label transition
    Aggregate metrics       recomputed from error_events.csv and matched
                            against results/standardized_baseline_comparison.csv
    Episode mapping         every attack target maps to an episode, every
                            benign target maps to none, counts add up
    No tuning               frozen thresholds differ from the test-optimal
                            thresholds; baseline artifact hashes unchanged

Usage:
    python src/verify_error_analysis.py

Exit code: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
OUT_DIR = REPO_ROOT / "results" / "error_analysis"
V4_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
LR_MANIFEST = REPO_ROOT / "docs" / "logistic_regression_baseline_manifest.json"
LR_PREDICTIONS = REPO_ROOT / "models" / "logistic_regression_baseline" / "test_predictions.csv"
TASK3_CSV = REPO_ROOT / "results" / "standardized_baseline_comparison.csv"
SUMMARY_JSON = REPO_ROOT / "results" / "error_analysis_summary.json"
TEST_DAY = "Friday-02-03-2018"

EXPECTED_WINDOWS = 3115
TOLERANCE = 1e-9
PROBABILITY_TOLERANCE = 1e-12  # error_events.csv stores full-precision scores


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
        print("ERROR ANALYSIS VERIFICATION")
        print("-" * 27)
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
    print("ERROR ANALYSIS — VERIFICATION (read-only)")
    print("=" * 78)
    print(f"Repository : {REPO_ROOT}")
    print(f"Analysis   : {OUT_DIR}")

    report = Report()

    required = [
        "error_events.csv", "attack_episodes.csv", "false_positive_summary.csv",
        "false_negative_summary.csv", "onset_analysis.csv", "post_attack_analysis.csv",
        "error_feature_summary.csv", "error_taxonomy.csv", "representative_cases.csv",
        "horizon_error_summary.csv", "lr_probability_shift.csv", "transition_analysis.csv",
    ]
    missing = [name for name in required if not (OUT_DIR / name).exists()]
    if missing or not SUMMARY_JSON.exists():
        report.add("Artifacts present", False, f"missing: {', '.join(missing) or SUMMARY_JSON.name}")
        report.render()
        return 1
    report.add("Artifacts present", True, f"{len(required)} tables + summary JSON")

    events = pd.read_csv(OUT_DIR / "error_events.csv", parse_dates=["forecast_time", "target_time"])
    episodes = pd.read_csv(OUT_DIR / "attack_episodes.csv", parse_dates=["attack_onset", "attack_offset"])
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

    primary = events[events["model"].isin([MODEL_PERSISTENCE, MODEL_LR, MODEL_V4])]

    # ---------------------------------------------------------
    # Test substrate
    # ---------------------------------------------------------
    problems = []
    if len(meta) != EXPECTED_WINDOWS:
        problems.append(f"benchmark test split has {len(meta)} windows")
    for (model, k), group in primary.groupby(["model", "horizon"]):
        if len(group) != EXPECTED_WINDOWS or group["sample_index"].nunique() != EXPECTED_WINDOWS:
            problems.append(f"{model} K={k}: {len(group)} rows")
        counts = group["error_type"].value_counts()
        if int(counts.sum()) != EXPECTED_WINDOWS:
            problems.append(f"{model} K={k}: TP+FP+TN+FN = {int(counts.sum())}")
    if set(primary["source_day"].unique()) != {TEST_DAY}:
        problems.append("source_day is not the held-out day only")
    report.add(
        "Test substrate", not problems,
        "; ".join(problems) if problems else f"{EXPECTED_WINDOWS} windows x 3 models x 4 horizons",
    )

    # ---------------------------------------------------------
    # Target alignment
    # ---------------------------------------------------------
    problems = []
    for (model, k), group in primary.groupby(["model", "horizon"]):
        group = group.sort_values("sample_index")
        if not np.array_equal(group["actual_label"].to_numpy(), A_test[:, k - 1].astype(int)):
            problems.append(f"{model} K={k}: actual_label != A_binary_multistep[:, {k - 1}]")
        expected_target = meta["timestamp"] + pd.Timedelta(seconds=10 * (k - 1))
        if not np.array_equal(group["target_time"].to_numpy(), expected_target.to_numpy()):
            problems.append(f"{model} K={k}: target_time misaligned")
        expected_forecast = meta["timestamp"] - pd.Timedelta(seconds=10)
        if not np.array_equal(group["forecast_time"].to_numpy(), expected_forecast.to_numpy()):
            problems.append(f"{model} K={k}: forecast_time is not t")
    report.add(
        "Target alignment", not problems,
        "; ".join(problems) if problems else "target_time = t+k, forecast_time = t, labels match the benchmark",
    )

    # ---------------------------------------------------------
    # Thresholds
    # ---------------------------------------------------------
    problems = []
    for model in (MODEL_V4, MODEL_LR):
        for k in POINT_HORIZONS:
            stored = primary[(primary["model"] == model) & (primary["horizon"] == k)]["threshold"].unique()
            if len(stored) != 1 or abs(float(stored[0]) - thresholds[model][k]) > TOLERANCE:
                problems.append(f"{model} K={k}: {stored} != frozen {thresholds[model][k]}")
    persistence_thresholds = primary[primary["model"] == MODEL_PERSISTENCE]["threshold"]
    if not persistence_thresholds.isna().all():
        problems.append("persistence carries a threshold value")
    report.add(
        "Thresholds", not problems,
        "; ".join(problems) if problems else "8 frozen thresholds unchanged; persistence has none",
    )

    # ---------------------------------------------------------
    # V4 prediction integrity
    # ---------------------------------------------------------
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    v4_probabilities, _ = v4_scores(data, device)
    problems = []
    for k in POINT_HORIZONS:
        group = primary[(primary["model"] == MODEL_V4) & (primary["horizon"] == k)].sort_values("sample_index")
        recomputed = v4_probabilities["test"][:, k]
        if not np.allclose(group["predicted_probability"].to_numpy(), recomputed, atol=PROBABILITY_TOLERANCE):
            problems.append(f"K={k}: probabilities differ from a fresh rollout")
        expected_labels = (recomputed >= thresholds[MODEL_V4][k]).astype(int)
        if not np.array_equal(group["predicted_label"].to_numpy(), expected_labels):
            problems.append(f"K={k}: predicted_label != (probability >= frozen threshold)")
    report.add(
        "V4 prediction integrity", not problems,
        "; ".join(problems) if problems else "rollout re-run from the frozen checkpoint reproduces every score",
    )

    # ---------------------------------------------------------
    # LR prediction integrity
    # ---------------------------------------------------------
    stored_lr = pd.read_csv(LR_PREDICTIONS)
    problems = []
    for k in POINT_HORIZONS:
        group = primary[(primary["model"] == MODEL_LR) & (primary["horizon"] == k)].sort_values("sample_index")
        frozen = stored_lr[f"prob_t+{k}"].to_numpy()
        if not np.allclose(group["predicted_probability"].to_numpy(), frozen, atol=PROBABILITY_TOLERANCE):
            problems.append(f"K={k}: probabilities differ from the frozen artifact")
        expected_labels = (frozen >= thresholds[MODEL_LR][k]).astype(int)
        if not np.array_equal(group["predicted_label"].to_numpy(), expected_labels):
            problems.append(f"K={k}: predicted_label != (probability >= frozen threshold)")
    report.add(
        "LR prediction integrity", not problems,
        "; ".join(problems) if problems else "matches models/logistic_regression_baseline/test_predictions.csv",
    )

    # ---------------------------------------------------------
    # Persistence integrity
    # ---------------------------------------------------------
    label_t = fetch_labels_at_time_t(meta).astype(int)
    problems = []
    for k in POINT_HORIZONS:
        group = primary[(primary["model"] == MODEL_PERSISTENCE) & (primary["horizon"] == k)].sort_values("sample_index")
        if not np.array_equal(group["predicted_label"].to_numpy(), label_t):
            problems.append(f"K={k}: prediction != A(t)")
        if not np.array_equal(group["label_at_time_t"].to_numpy(), label_t):
            problems.append(f"K={k}: recorded A(t) differs from the raw day CSV")
        errors = group[group["error_type"].isin(["FP", "FN"])]
        transitions = group[group["actual_label"].to_numpy() != label_t]
        if len(errors) != len(transitions) or not np.array_equal(
            np.sort(errors["sample_index"].to_numpy()), np.sort(transitions["sample_index"].to_numpy())
        ):
            problems.append(f"K={k}: persistence errors are not exactly the label transitions")
    report.add(
        "Persistence integrity", not problems,
        "; ".join(problems) if problems else "predictions equal A(t); every error is a label transition",
    )

    # ---------------------------------------------------------
    # Aggregate metrics vs Task 3
    # ---------------------------------------------------------
    task3 = pd.read_csv(TASK3_CSV)
    task3_point = task3[task3["evaluation_type"] == "point_forecast"]
    problems = []
    for (model, k), group in primary.groupby(["model", "horizon"]):
        group = group.sort_values("sample_index")
        actual = group["actual_label"].to_numpy()
        predicted = group["predicted_label"].to_numpy()
        score = group["predicted_probability"].to_numpy()

        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        recomputed = {
            "precision": precision_score(actual, predicted, zero_division=0),
            "recall": recall_score(actual, predicted, zero_division=0),
            "f1": f1_score(actual, predicted, zero_division=0),
            "roc_auc": roc_auc_score(actual, score),
            "fpr": fp / (fp + tn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

        reference = task3_point[(task3_point["model"] == model) & (task3_point["horizon"] == k)]
        if reference.empty:
            problems.append(f"{model} K={k}: absent from the Task 3 CSV")
            continue
        reference = reference.iloc[0]

        for field, value in recomputed.items():
            tolerance = TOLERANCE
            if abs(float(value) - float(reference[field])) > tolerance:
                problems.append(f"{model} K={k} {field}: {value} vs {reference[field]}")
    report.add(
        "Aggregate metrics", not problems,
        "; ".join(problems[:3]) if problems else "12 model/horizon rows reproduce Task 3 exactly",
    )

    # ---------------------------------------------------------
    # Episode mapping
    # ---------------------------------------------------------
    problems = []
    raw = pd.read_csv(
        STATES_DIR / f"{TEST_DAY}_TrafficForML_CICFlowMeter_states_v2.csv",
        usecols=["window_start", "attack_present"],
    )
    raw["window_start"] = pd.to_datetime(raw["window_start"])
    raw_attacks = set(raw.loc[raw["attack_present"].astype(int) == 1, "window_start"])

    if int(episodes["number_of_attack_windows"].sum()) != len(raw_attacks):
        problems.append(
            f"episode windows {int(episodes['number_of_attack_windows'].sum())} != "
            f"raw attack windows {len(raw_attacks)}"
        )
    if int((~episodes["observable_from_test_start"]).sum()) != 1:
        problems.append("expected exactly one episode without an observable onset")

    for (model, k), group in primary.groupby(["model", "horizon"]):
        attack_rows = group[group["actual_label"] == 1]
        benign_rows = group[group["actual_label"] == 0]
        if (attack_rows["attack_episode_id"] <= 0).any():
            problems.append(f"{model} K={k}: attack target without an episode")
        if (benign_rows["attack_episode_id"] != -1).any():
            problems.append(f"{model} K={k}: benign target assigned to an episode")
        if not set(attack_rows["target_time"]).issubset(raw_attacks):
            problems.append(f"{model} K={k}: attack target outside the raw attack windows")
    report.add(
        "Episode mapping", not problems,
        "; ".join(problems[:3]) if problems else
        f"{len(episodes)} episodes, {int(episodes['number_of_attack_windows'].sum())} attack windows, "
        f"{int(episodes['observable_from_test_start'].sum())} observable onsets",
    )

    # ---------------------------------------------------------
    # No tuning
    # ---------------------------------------------------------
    problems = []
    grid = np.arange(0.01, 0.99, 0.01)

    for model in (MODEL_V4, MODEL_LR):
        for k in POINT_HORIZONS:
            group = primary[(primary["model"] == model) & (primary["horizon"] == k)].sort_values("sample_index")
            score = group["predicted_probability"].to_numpy()
            actual = group["actual_label"].to_numpy()

            best_threshold, best_f1 = 0.5, -1.0
            for threshold in grid:
                value = f1_score(actual, (score >= threshold).astype(int), zero_division=0)
                if value > best_f1:
                    best_f1, best_threshold = float(value), float(threshold)

            if abs(best_threshold - thresholds[model][k]) < TOLERANCE:
                problems.append(
                    f"{model} K={k}: frozen threshold equals the test-optimal threshold"
                )
            else:
                report.note(
                    f"{model} K={k}: frozen {thresholds[model][k]:.2f} (validation) vs test-optimal "
                    f"{best_threshold:.2f} — no test tuning"
                )

    checkpoint = REPO_ROOT / v4_manifest["checkpoint"]["path"]
    if sha256_file(checkpoint) != v4_manifest["checkpoint"]["sha256"]:
        problems.append("V4 checkpoint hash changed")
    for key, record in lr_manifest["models"].items():
        path = REPO_ROOT / record["path"]
        if not path.exists() or sha256_file(path) != record["sha256"]:
            problems.append(f"LR model {key} hash changed")
    for name in ("X_test.npy", "A_binary_multistep_test.npy"):
        expected = v4_manifest["dataset"]["splits"]["test"][
            "X" if name.startswith("X") else "A_binary_multistep"
        ]["sha256"]
        if sha256_file(BENCHMARK_DIR / name) != expected:
            problems.append(f"benchmark {name} hash changed")

    report.add(
        "No tuning", not problems,
        "; ".join(problems[:3]) if problems else
        "frozen thresholds differ from test-optimal; all baseline artifact hashes unchanged",
    )

    report.render()
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
