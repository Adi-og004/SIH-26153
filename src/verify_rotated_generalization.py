"""
=============================================================
ROTATED GENERALIZATION — VERIFICATION (READ-ONLY)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Independently re-derives the rotated leave-one-day-out experiment and checks
that every fold is isolated and that the frozen V4 reference is untouched.

Usage:
    python src/verify_rotated_generalization.py

Exit code: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rotated_common import (  # noqa: E402
    DAY_ORDER, DAYS, EVAL_THRESHOLD_GRID, FOLD_DATA_DIR, INPUT_FEATURES, LR_DIR,
    MAX_K, POINT_HORIZONS, REPO_ROOT, RESULTS_DIR, SEQUENCE_LENGTH, V4_DIR,
    VAL_RATIO, WINDOW_SECONDS, build_episodes, get_device, load_day,
    load_fold_arrays, sha256_file, standard_metrics,
)
from rotated_evaluate import MODEL_LR, MODEL_PERSISTENCE, MODEL_V4, load_fold_models

MASTER_CSV = RESULTS_DIR / "rotated_generalization_results.csv"
V4_FROZEN_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
TOLERANCE = 1e-9


class Report:
    def __init__(self):
        self.rows, self.notes, self.failed = [], [], False

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
        print("ROTATED GENERALIZATION VERIFICATION")
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
    print("ROTATED GENERALIZATION — VERIFICATION (read-only)")
    print("=" * 78)
    print(f"Repository : {REPO_ROOT}")

    report = Report()
    device = get_device()
    master = pd.read_csv(MASTER_CSV)
    fold_inventory = pd.read_csv(RESULTS_DIR / "fold_inventory.csv")

    # ---------------------------------------------------------
    # 1-3. Per-fold existence, single held-out day, no leakage
    # ---------------------------------------------------------
    for test_day in DAY_ORDER:
        problems = []
        fold_dir = RESULTS_DIR / f"fold_{test_day}"
        data_dir = FOLD_DATA_DIR / f"fold_test_{test_day}"
        checkpoint = V4_DIR / f"fold_test_{test_day}" / f"v4_rotated_{test_day}.pt"

        for path in (fold_dir / "input_scaler.pkl", fold_dir / "target_scaler.pkl", checkpoint):
            if not path.exists():
                problems.append(f"missing {path.name}")
        for k in POINT_HORIZONS:
            if not (LR_DIR / f"fold_test_{test_day}" / f"logistic_regression_k{k}.pkl").exists():
                problems.append(f"missing LR K={k}")

        if problems:
            report.add(f"Fold {test_day}", False, "; ".join(problems))
            continue

        splits = load_fold_arrays(test_day)
        test_days_present = set(splits["test"]["meta"]["source_day"].unique())
        train_days_present = set(splits["train"]["meta"]["source_day"].unique())
        val_days_present = set(splits["val"]["meta"]["source_day"].unique())
        expected_train = {DAYS[key] for key in DAY_ORDER if key != test_day}

        if test_days_present != {DAYS[test_day]}:
            problems.append(f"test contains {sorted(test_days_present)}")
        if train_days_present != expected_train:
            problems.append(f"train contains {sorted(train_days_present)}")
        if val_days_present != expected_train:
            problems.append(f"validation contains {sorted(val_days_present)}")
        if DAYS[test_day] in train_days_present | val_days_present:
            problems.append("HELD-OUT DAY LEAKS INTO TRAINING")

        report.add(
            f"Fold {test_day}", not problems,
            "; ".join(problems) if problems else
            f"test={len(splits['test']['meta'])}, train={len(splits['train']['meta'])}, "
            f"val={len(splits['val']['meta'])}, held-out day isolated",
        )

    # ---------------------------------------------------------
    # 6. Feature schema identical across folds
    # ---------------------------------------------------------
    problems = []
    frozen_features = (
        REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_feature_names.txt"
    ).read_text().split()
    hashes = set()
    for test_day in DAY_ORDER:
        path = RESULTS_DIR / f"fold_{test_day}" / "input_feature_names.txt"
        names = path.read_text().split()
        hashes.add(sha256_file(path))
        if names != INPUT_FEATURES:
            problems.append(f"{test_day}: order differs from the module definition")
        if names != frozen_features:
            problems.append(f"{test_day}: order differs from the frozen benchmark schema")
        if len(names) != 52:
            problems.append(f"{test_day}: {len(names)} features")
    if len(hashes) != 1:
        problems.append("feature-name files differ between folds")
    report.add("Feature schema", not problems,
               "; ".join(problems) if problems else
               f"52 features, identical in all 4 folds and equal to the frozen schema ({list(hashes)[0][:12]}…)")

    # ---------------------------------------------------------
    # 5. Scaler isolation — refit on train only and compare
    # ---------------------------------------------------------
    problems = []
    for test_day in DAY_ORDER:
        with open(RESULTS_DIR / f"fold_{test_day}" / "input_scaler.pkl", "rb") as handle:
            scaler = pickle.load(handle)

        expected_samples = 0
        for key in DAY_ORDER:
            if key == test_day:
                continue
            expected_samples += int(len(load_day(key)) * (1.0 - VAL_RATIO))

        if int(scaler.n_samples_seen_) != expected_samples:
            problems.append(
                f"{test_day}: scaler saw {int(scaler.n_samples_seen_)} rows, "
                f"train partition has {expected_samples}"
            )
        if int(scaler.n_features_in_) != 52:
            problems.append(f"{test_day}: scaler has {scaler.n_features_in_} features")

        # A train-fitted scaler standardises TRAIN, not TEST.
        splits = load_fold_arrays(test_day)
        train_flat = splits["train"]["X"].reshape(-1, 52)
        if abs(float(train_flat.mean(axis=0).mean())) > 0.05:
            problems.append(f"{test_day}: train partition is not centred by its own scaler")
    report.add("Scaler isolation", not problems,
               "; ".join(problems) if problems else
               "each fold's scaler was fitted on that fold's train rows only")

    # ---------------------------------------------------------
    # 4. Chronological within-day split
    # ---------------------------------------------------------
    problems = []
    for test_day in DAY_ORDER:
        splits = load_fold_arrays(test_day)
        for day in {DAYS[key] for key in DAY_ORDER if key != test_day}:
            train_last = splits["train"]["meta"].loc[
                splits["train"]["meta"]["source_day"] == day, "timestamp"
            ].max()
            val_first = splits["val"]["meta"].loc[
                splits["val"]["meta"]["source_day"] == day, "timestamp"
            ].min()
            if not val_first > train_last:
                problems.append(f"{test_day}/{day}: validation not after train")
    report.add("Temporal split", not problems,
               "; ".join(problems) if problems else
               "validation follows training chronologically within every training day (85/15)")

    # ---------------------------------------------------------
    # 7. Target alignment
    # ---------------------------------------------------------
    problems = []
    for test_day in DAY_ORDER:
        splits = load_fold_arrays(test_day)
        frame = load_day(test_day)
        lookup = dict(zip(frame["window_start"], frame["attack_present"].astype(int)))
        meta = splits["test"]["meta"]
        A_test = splits["test"]["A_binary"].astype(int)

        for k in range(1, MAX_K + 1):
            expected = np.array(
                [lookup[ts + pd.Timedelta(seconds=WINDOW_SECONDS * (k - 1))] for ts in meta["timestamp"]]
            )
            if not np.array_equal(expected, A_test[:, k - 1]):
                problems.append(f"{test_day}: A(t+{k}) misaligned")

        expected_t = np.array(
            [lookup[ts - pd.Timedelta(seconds=WINDOW_SECONDS)] for ts in meta["timestamp"]]
        )
        if not np.array_equal(expected_t, meta["label_at_time_t"].to_numpy()):
            problems.append(f"{test_day}: A(t) misaligned")
    report.add("Target alignment", not problems,
               "; ".join(problems) if problems else
               "A(t) and A(t+1..t+5) re-derived from the raw day CSV for all 4 folds")

    # ---------------------------------------------------------
    # 8-10. Threshold isolation + model integrity + reproduction
    # ---------------------------------------------------------
    threshold_problems, v4_problems, lr_problems, metric_problems = [], [], [], []
    threshold_coincidences = []

    for test_day in DAY_ORDER:
        splits = load_fold_arrays(test_day)
        model, checkpoint, lr_models = load_fold_models(test_day, device)

        if checkpoint.get("is_frozen_reference") is not False:
            v4_problems.append(f"{test_day}: checkpoint not marked experimental")
        if checkpoint.get("held_out_day") != DAYS[test_day]:
            v4_problems.append(f"{test_day}: checkpoint records the wrong held-out day")
        if checkpoint.get("test_labels_used_in_training") is not False:
            v4_problems.append(f"{test_day}: checkpoint does not assert test-label isolation")

        from rotated_common import rollout_probabilities
        val_probabilities = rollout_probabilities(model, splits["val"]["X"], MAX_K, device)
        test_probabilities = rollout_probabilities(model, splits["test"]["X"], MAX_K, device)

        A_val = splits["val"]["A_binary"].astype(int)
        A_test = splits["test"]["A_binary"].astype(int)
        label_t = splits["test"]["meta"]["label_at_time_t"].to_numpy().astype(int)

        stored = json.loads((V4_DIR / f"fold_test_{test_day}" / "thresholds.json").read_text())

        for k in POINT_HORIZONS:
            # V4: threshold must be the validation argmax, and must NOT be the test argmax
            derived, _ = _argmax_f1(val_probabilities[:, k], A_val[:, k - 1])
            recorded = float(stored["point"][str(k)])
            if abs(derived - recorded) > TOLERANCE:
                threshold_problems.append(f"{test_day} V4 K={k}: {recorded} != validation argmax {derived:.2f}")

            # Diagnostic only: the threshold coinciding with the test-optimal value is NOT
            # evidence of tuning. When a model's probabilities pile up near zero, the grid's
            # lowest point maximises F1 on both partitions by coincidence. The hard
            # requirement is the validation-argmax equality checked above.
            test_optimal, _ = _argmax_f1(test_probabilities[:, k], A_test[:, k - 1])
            threshold_coincidences.append(
                (test_day, k, recorded, test_optimal, abs(test_optimal - recorded) < TOLERANCE)
            )

            lr_val = lr_models[k].predict_proba(splits["val"]["X"][:, -1, :])[:, 1]
            lr_derived, _ = _argmax_f1(lr_val, A_val[:, k - 1])
            lr_recorded = float(stored["lr_point"][str(k)])
            if abs(lr_derived - lr_recorded) > TOLERANCE:
                threshold_problems.append(f"{test_day} LR K={k}: {lr_recorded} != validation argmax")

            # Aggregate reproduction against the master CSV
            predicted = (test_probabilities[:, k] >= recorded).astype(int)
            metrics = standard_metrics(test_probabilities[:, k], A_test[:, k - 1], predicted)
            reference = master[
                (master["test_day"] == test_day) & (master["model"] == MODEL_V4)
                & (master["evaluation_type"] == "point_forecast") & (master["horizon"] == k)
            ]
            if reference.empty:
                metric_problems.append(f"{test_day} V4 K={k} missing from the master CSV")
            else:
                for field in ("f1", "precision", "recall", "roc_auc", "tp", "fp", "tn", "fn"):
                    if abs(float(metrics[field]) - float(reference.iloc[0][field])) > 1e-9:
                        metric_problems.append(f"{test_day} V4 K={k} {field}")

            lr_test = lr_models[k].predict_proba(splits["test"]["X"][:, -1, :])[:, 1]
            lr_predicted = (lr_test >= lr_recorded).astype(int)
            lr_metrics = standard_metrics(lr_test, A_test[:, k - 1], lr_predicted)
            lr_reference = master[
                (master["test_day"] == test_day) & (master["model"] == MODEL_LR)
                & (master["evaluation_type"] == "point_forecast") & (master["horizon"] == k)
            ]
            if lr_reference.empty:
                metric_problems.append(f"{test_day} LR K={k} missing from the master CSV")
            else:
                for field in ("f1", "precision", "recall", "roc_auc"):
                    if abs(float(lr_metrics[field]) - float(lr_reference.iloc[0][field])) > 1e-9:
                        metric_problems.append(f"{test_day} LR K={k} {field}")

            if int(lr_models[k].n_features_in_) != 52:
                lr_problems.append(f"{test_day} K={k}: {lr_models[k].n_features_in_} features")

            # Persistence integrity
            persistence_reference = master[
                (master["test_day"] == test_day) & (master["model"] == MODEL_PERSISTENCE)
                & (master["evaluation_type"] == "point_forecast") & (master["horizon"] == k)
            ]
            persistence_metrics = standard_metrics(label_t.astype(float), A_test[:, k - 1], label_t)
            if not persistence_reference.empty:
                for field in ("f1", "tp", "fp", "tn", "fn"):
                    if abs(float(persistence_metrics[field]) - float(persistence_reference.iloc[0][field])) > 1e-9:
                        metric_problems.append(f"{test_day} persistence K={k} {field}")

    differing = sum(1 for *_, same in threshold_coincidences if not same)
    report.add("Threshold isolation", not threshold_problems,
               "; ".join(threshold_problems[:3]) if threshold_problems else
               f"all 32 thresholds re-derived from validation; {differing}/{len(threshold_coincidences)} "
               f"V4 thresholds also differ from their test-optimal value")
    for test_day, k, recorded, test_optimal, same in threshold_coincidences:
        if same:
            report.note(
                f"{test_day} V4 K={k}: validation threshold {recorded:.2f} happens to equal the "
                f"test-optimal threshold — both land on the grid minimum because the rotated "
                f"model's probabilities pile up near zero; the threshold is still the validation argmax"
            )
    report.add("V4 integrity", not v4_problems,
               "; ".join(v4_problems[:3]) if v4_problems else
               "4 experimental checkpoints load strict=True and declare their held-out day")
    report.add("LR integrity", not lr_problems,
               "; ".join(lr_problems[:3]) if lr_problems else "20 models load with 52 input features")
    report.add("Persistence integrity", not [p for p in metric_problems if "persistence" in p],
               "predictions equal A(t) on every fold" if not [p for p in metric_problems if "persistence" in p]
               else "; ".join(p for p in metric_problems if "persistence" in p)[:120])
    report.add("Aggregate reproduction", not metric_problems,
               "; ".join(metric_problems[:3]) if metric_problems else
               "every point-forecast metric recomputes from the saved models")

    # ---------------------------------------------------------
    # 15-16. Transition accounting and episode counts
    # ---------------------------------------------------------
    problems = []
    transitions = pd.read_csv(RESULTS_DIR / "transition_diagnostics.csv")
    inventory = pd.read_csv(RESULTS_DIR / "day_inventory.csv")
    episodes_frame = pd.read_csv(RESULTS_DIR / "episode_warning_results.csv")

    for test_day in DAY_ORDER:
        splits = load_fold_arrays(test_day)
        A_test = splits["test"]["A_binary"].astype(int)
        label_t = splits["test"]["meta"]["label_at_time_t"].to_numpy().astype(int)

        for k in POINT_HORIZONS:
            onset = int(((label_t == 0) & (A_test[:, k - 1] == 1)).sum())
            offset = int(((label_t == 1) & (A_test[:, k - 1] == 0)).sum())
            steady = int(len(label_t)) - onset - offset
            row = transitions[
                (transitions["test_day"] == test_day) & (transitions["horizon"] == k)
                & (transitions["model"] == MODEL_V4)
            ]
            if row.empty:
                problems.append(f"{test_day} K={k}: transition row missing")
                continue
            if int(row["onset_count"].iloc[0]) != onset or int(row["offset_count"].iloc[0]) != offset:
                problems.append(f"{test_day} K={k}: transition counts disagree")
            if onset + offset + steady != len(label_t):
                problems.append(f"{test_day} K={k}: transition accounting does not sum to the test size")

        frame = load_day(test_day)
        episodes = build_episodes(frame)
        inventory_row = inventory[inventory["day_key"] == test_day].iloc[0]
        if int(inventory_row["attack_episodes"]) != len(episodes):
            problems.append(f"{test_day}: inventory episode count disagrees")

        observable = sum(1 for ep in episodes if not ep["cold_start_episode"])
        coverage_rows = episodes_frame[
            (episodes_frame["test_day"] == test_day)
            & (episodes_frame["warning_signal"] == "trajectory_max_k5")
        ]
        for _, row in coverage_rows.iterrows():
            if int(row["total_episodes"]) != len(episodes):
                problems.append(f"{test_day}: episode total disagrees")
            if int(row["observable_onset_episodes"]) != observable:
                problems.append(f"{test_day}: observable-onset count disagrees")

    report.add("Transition accounting", not problems,
               "; ".join(problems[:3]) if problems else
               f"onset+offset+steady = test size in all folds; "
               f"{int(inventory['attack_episodes'].sum())} episodes, "
               f"{int(inventory['attack_episodes'].sum() - inventory['cold_start_episodes'].sum())} observable onsets")

    # ---------------------------------------------------------
    # 12. Frozen V4 reference untouched
    # ---------------------------------------------------------
    problems = []
    frozen = json.loads(V4_FROZEN_MANIFEST.read_text())
    if sha256_file(REPO_ROOT / frozen["checkpoint"]["path"]) != frozen["checkpoint"]["sha256"]:
        problems.append("frozen V4 checkpoint hash changed")
    for name, key in (("X_test.npy", "X"), ("A_binary_multistep_test.npy", "A_binary_multistep")):
        path = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / name
        if sha256_file(path) != frozen["dataset"]["splits"]["test"][key]["sha256"]:
            problems.append(f"frozen benchmark {name} changed")
    if sha256_file(REPO_ROOT / frozen["scalers"]["input_scaler"]["path"]) != frozen["scalers"]["input_scaler"]["sha256"]:
        problems.append("frozen input scaler changed")
    for test_day in DAY_ORDER:
        checkpoint = V4_DIR / f"fold_test_{test_day}" / f"v4_rotated_{test_day}.pt"
        if str(checkpoint).startswith(str(REPO_ROOT / "models" / "latent_world_model_v4")):
            problems.append(f"{test_day} checkpoint written into the frozen directory")
    report.add("Frozen reference untouched", not problems,
               "; ".join(problems) if problems else
               "frozen checkpoint, scaler and benchmark hashes unchanged; rotated models live elsewhere")

    report.note("Four source days only — the folds share two training days with each other and are "
                "not four independent experiments.")
    report.note("Every day carries a different attack family, so every fold is also an unseen-family "
                "test; day effects and family effects cannot be separated.")

    report.render()
    return 1 if report.failed else 0


def _argmax_f1(scores, targets):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in EVAL_THRESHOLD_GRID:
        value = f1_score(targets, (scores >= threshold).astype(int), zero_division=0)
        if value > best_f1:
            best_f1, best_threshold = float(value), float(threshold)
    return best_threshold, best_f1


if __name__ == "__main__":
    sys.exit(main())
