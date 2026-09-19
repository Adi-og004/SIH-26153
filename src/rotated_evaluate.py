"""
Evaluation and summarisation for the rotated leave-one-day-out experiment (Task 6).

Every threshold is selected on that fold's VALIDATION partition only; each held-out
day is scored exactly once. The frozen V4 reference is never touched.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rotated_common import (  # noqa: E402
    DAY_ORDER, DAYS, INPUT_FEATURES, LR_CONFIG, LR_DIR, MAX_K, POINT_HORIZONS,
    REPO_ROOT, RESULTS_DIR, TRAJECTORY_HORIZONS, V4_CONFIG, V4_DIR,
    WINDOW_SECONDS, LatentWorldModelV4, build_episodes, get_device, git_value,
    load_day, load_fold_arrays, rollout_probabilities, select_threshold,
    sha256_file, standard_metrics,
)

FIG_DIR = RESULTS_DIR / "figures"
MASTER_CSV = RESULTS_DIR / "rotated_generalization_results.csv"
SUMMARY_JSON = REPO_ROOT / "results" / "rotated_generalization_summary.json"

MODEL_PERSISTENCE = "persistence"
MODEL_LR = "logistic_regression"
MODEL_V4 = "v4_world_model"
MODELS = [MODEL_PERSISTENCE, MODEL_LR, MODEL_V4]

ONSET_LOOKBACK_WINDOWS = 12

SHIFT_FEATURES = [
    "flow_count", "window_packet_rate", "window_byte_rate", "syn_rate", "ack_rate",
    "rst_rate", "log_unique_dst_ports", "dst_port_entropy", "protocol_entropy",
    "flow_count_rel_change", "syn_rate_rel_change", "window_packet_rate_rel_change",
]


# =============================================================
# PER-FOLD SCORING
# =============================================================

def load_fold_models(test_day, device):
    checkpoint_path = V4_DIR / f"fold_test_{test_day}" / f"v4_rotated_{test_day}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = LatentWorldModelV4(
        input_size=checkpoint["input_size"], hidden_size=checkpoint["hidden_size"],
        latent_size=checkpoint["latent_size"], output_size=checkpoint["output_size"],
        num_layers=checkpoint["num_layers"], dropout=V4_CONFIG["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    lr_models = {}
    for k in (1, 2, 3, 4, 5):
        path = LR_DIR / f"fold_test_{test_day}" / f"logistic_regression_k{k}.pkl"
        with open(path, "rb") as handle:
            lr_models[k] = pickle.load(handle)

    return model, checkpoint, lr_models


def score_fold(test_day, device):
    """Produce validation and test scores for all three models on one fold."""
    splits = load_fold_arrays(test_day)
    model, checkpoint, lr_models = load_fold_models(test_day, device)

    scores = {"val": {}, "test": {}}
    for partition in ("val", "test"):
        X = splits[partition]["X"]
        scores[partition][MODEL_V4] = rollout_probabilities(model, X, MAX_K, device)
        scores[partition][MODEL_LR] = {
            k: lr_models[k].predict_proba(X[:, -1, :])[:, 1] for k in lr_models
        }
        scores[partition][MODEL_PERSISTENCE] = splits[partition]["meta"][
            "label_at_time_t"
        ].to_numpy().astype(float)

    return splits, scores, checkpoint


def evaluate_fold(test_day, device):
    splits, scores, checkpoint = score_fold(test_day, device)

    A_val = splits["val"]["A_binary"].astype(int)
    A_test = splits["test"]["A_binary"].astype(int)
    meta_test = splits["test"]["meta"]
    label_t_test = scores["test"][MODEL_PERSISTENCE].astype(int)
    hours = len(meta_test) * WINDOW_SECONDS / 3600.0

    rows, transition_rows = [], []
    thresholds = {MODEL_V4: {}, MODEL_LR: {}}
    trajectory_thresholds = {MODEL_V4: {}, MODEL_LR: {}}

    # ---------- point forecast ----------
    for k in POINT_HORIZONS:
        target_val, target_test = A_val[:, k - 1], A_test[:, k - 1]

        for model_name in MODELS:
            if model_name == MODEL_PERSISTENCE:
                score = scores["test"][MODEL_PERSISTENCE]
                predicted = score.astype(int)
                threshold = None
            else:
                if model_name == MODEL_V4:
                    val_score = scores["val"][MODEL_V4][:, k]
                    score = scores["test"][MODEL_V4][:, k]
                else:
                    val_score = scores["val"][MODEL_LR][k]
                    score = scores["test"][MODEL_LR][k]
                threshold, _ = select_threshold(val_score, target_val)
                thresholds[model_name][k] = threshold
                predicted = (score >= threshold).astype(int)

            metrics = standard_metrics(score, target_test, predicted)
            metrics.update(
                {
                    "test_day": test_day, "source_day": DAYS[test_day], "model": model_name,
                    "evaluation_type": "point_forecast", "horizon": k, "threshold": threshold,
                    "positive_prevalence": float(target_test.mean()),
                    "transition_fraction": float((target_test != label_t_test).mean()),
                    "sample_count": int(len(target_test)),
                    "fp_per_hour": metrics["fp"] / hours,
                    "roc_auc_degenerate": model_name == MODEL_PERSISTENCE,
                }
            )
            rows.append(metrics)

            # ---------- transition-conditioned diagnostic ----------
            onset_mask = (label_t_test == 0) & (target_test == 1)
            offset_mask = (label_t_test == 1) & (target_test == 0)
            transition_mask = onset_mask | offset_mask

            transition_rows.append(
                {
                    "test_day": test_day, "model": model_name, "horizon": k,
                    "evaluation": "transition-conditioned diagnostic (NOT an overall benchmark)",
                    "transition_count": int(transition_mask.sum()),
                    "onset_count": int(onset_mask.sum()),
                    "offset_count": int(offset_mask.sum()),
                    "onset_detected": int(predicted[onset_mask].sum()),
                    "onset_recall": float(predicted[onset_mask].mean()) if onset_mask.any() else None,
                    "offset_correct": int((predicted[offset_mask] == 0).sum()),
                    "offset_recall": float((predicted[offset_mask] == 0).mean())
                    if offset_mask.any() else None,
                    "onset_mean_probability": float(score[onset_mask].mean()) if onset_mask.any() else None,
                    "onset_median_probability": float(np.median(score[onset_mask]))
                    if onset_mask.any() else None,
                    "offset_mean_probability": float(score[offset_mask].mean())
                    if offset_mask.any() else None,
                    "threshold_crossing_rate": float(predicted[transition_mask].mean())
                    if transition_mask.any() else None,
                    "transition_roc_auc": float(
                        __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                            target_test[transition_mask], score[transition_mask]
                        )
                    ) if transition_mask.sum() > 1
                    and len(np.unique(target_test[transition_mask])) > 1 else None,
                    "predicted_positive_rate": float(predicted.mean()),
                    "expected_onset_hits_at_base_rate": round(
                        float(predicted.mean()) * int(onset_mask.sum()), 3
                    ),
                    "persistence_accuracy": float((target_test == label_t_test).mean()),
                }
            )

    # ---------- trajectory events ----------
    for k in TRAJECTORY_HORIZONS:
        target_val = A_val[:, :k].max(axis=1)
        target_test = A_test[:, :k].max(axis=1)

        for model_name in MODELS:
            if model_name == MODEL_PERSISTENCE:
                score = scores["test"][MODEL_PERSISTENCE]
                predicted = score.astype(int)
                threshold = None
                mechanism = "deterministic A(t); the rule is horizon-independent"
            elif model_name == MODEL_V4:
                val_score = scores["val"][MODEL_V4][:, 1:k + 1].max(axis=1)
                score = scores["test"][MODEL_V4][:, 1:k + 1].max(axis=1)
                threshold, _ = select_threshold(val_score, target_val)
                trajectory_thresholds[MODEL_V4][k] = threshold
                predicted = (score >= threshold).astype(int)
                mechanism = "max over rollout steps of ONE latent transition model"
            else:
                val_score = np.max(
                    np.column_stack([scores["val"][MODEL_LR][j] for j in range(1, k + 1)]), axis=1
                )
                score = np.max(
                    np.column_stack([scores["test"][MODEL_LR][j] for j in range(1, k + 1)]), axis=1
                )
                threshold, _ = select_threshold(val_score, target_val)
                trajectory_thresholds[MODEL_LR][k] = threshold
                predicted = (score >= threshold).astype(int)
                mechanism = "max over K INDEPENDENTLY trained horizon-specific models"

            metrics = standard_metrics(score, target_test, predicted)
            metrics.update(
                {
                    "test_day": test_day, "source_day": DAYS[test_day], "model": model_name,
                    "evaluation_type": "trajectory_event_max", "horizon": k, "threshold": threshold,
                    "positive_prevalence": float(target_test.mean()),
                    "transition_fraction": float((A_test[:, k - 1] != label_t_test).mean()),
                    "sample_count": int(len(target_test)),
                    "fp_per_hour": metrics["fp"] / hours,
                    "roc_auc_degenerate": model_name == MODEL_PERSISTENCE,
                    "aggregation_mechanism": mechanism,
                }
            )
            rows.append(metrics)

    return {
        "rows": rows,
        "transition_rows": transition_rows,
        "thresholds": thresholds,
        "trajectory_thresholds": trajectory_thresholds,
        "splits": splits,
        "scores": scores,
        "checkpoint": checkpoint,
        "hours": hours,
    }


# =============================================================
# EARLY WARNING (episode level)
# =============================================================

def early_warning(test_day, fold_result):
    splits = fold_result["splits"]
    scores = fold_result["scores"]
    meta = splits["test"]["meta"]
    A_test = splits["test"]["A_binary"].astype(int)

    frame = load_day(test_day)
    episodes = build_episodes(frame)

    signals = {}
    for model_name in MODELS:
        if model_name == MODEL_PERSISTENCE:
            point = scores["test"][MODEL_PERSISTENCE]
            trajectory = scores["test"][MODEL_PERSISTENCE]
            point_threshold = trajectory_threshold = None
            point_predictions = point.astype(int)
            trajectory_predictions = trajectory.astype(int)
        else:
            if model_name == MODEL_V4:
                point = scores["test"][MODEL_V4][:, 5]
                trajectory = scores["test"][MODEL_V4][:, 1:6].max(axis=1)
            else:
                point = scores["test"][MODEL_LR][5]
                trajectory = np.max(
                    np.column_stack([scores["test"][MODEL_LR][j] for j in range(1, 6)]), axis=1
                )
            point_threshold = fold_result["thresholds"][model_name][5]
            trajectory_threshold = fold_result["trajectory_thresholds"][model_name][5]
            point_predictions = (point >= point_threshold).astype(int)
            trajectory_predictions = (trajectory >= trajectory_threshold).astype(int)

        signals[model_name] = {
            "point_k5": (point, point_predictions, point_threshold),
            "trajectory_max_k5": (trajectory, trajectory_predictions, trajectory_threshold),
        }

    onset_rows, episode_rows = [], []
    evaluable_start = meta["timestamp"].min()

    for model_name in MODELS:
        for signal_name, (score, predicted, threshold) in signals[model_name].items():
            warned, lead_times = 0, []
            observable_count = 0

            for episode in episodes:
                onset = episode["onset"]
                cold_start = bool(episode["cold_start_episode"] or onset < evaluable_start)

                window_start = onset - pd.Timedelta(seconds=WINDOW_SECONDS * ONSET_LOOKBACK_WINDOWS)
                mask = (meta["timestamp"] >= window_start) & (meta["timestamp"] < onset)
                at_onset_mask = meta["timestamp"] == onset

                pre_scores = score[mask.to_numpy()]
                pre_predictions = predicted[mask.to_numpy()]
                pre_labels = A_test[mask.to_numpy(), 0]

                alert_indices = np.where(pre_predictions == 1)[0]
                first_alert_time = (
                    meta.loc[mask, "timestamp"].iloc[alert_indices[0]] if len(alert_indices) else None
                )
                lead = float((onset - first_alert_time).total_seconds()) if first_alert_time is not None else None

                if not cold_start:
                    observable_count += 1
                    if lead is not None:
                        warned += 1
                        lead_times.append(lead)

                onset_rows.append(
                    {
                        "test_day": test_day, "model": model_name, "warning_signal": signal_name,
                        "episode_id": episode["episode_id"],
                        "attack_onset": str(onset),
                        "attack_offset": str(episode["offset"]),
                        "episode_windows": episode["windows"],
                        "attack_family": episode["primary_family"],
                        "cold_start_episode": cold_start,
                        "threshold": threshold,
                        "pre_onset_windows": int(mask.sum()),
                        "pre_onset_windows_that_are_attacks": int(pre_labels.sum()),
                        "pre_onset_lookback_is_clean": bool(pre_labels.sum() == 0) and bool(mask.sum() > 0),
                        "warning_detected": bool(len(alert_indices) > 0) and not cold_start,
                        "first_warning_time": str(first_alert_time) if first_alert_time is not None else "",
                        "lead_time_seconds": lead if not cold_start else None,
                        "number_of_pre_onset_warnings": int(len(alert_indices)),
                        "probability_at_onset": float(score[at_onset_mask.to_numpy()][0])
                        if at_onset_mask.any() else None,
                        "max_pre_onset_probability": float(pre_scores.max()) if len(pre_scores) else None,
                        "mean_pre_onset_probability": float(pre_scores.mean()) if len(pre_scores) else None,
                        "note": "cold-start episode — no observable onset, excluded from lead-time statistics"
                        if cold_start else "",
                    }
                )

            # ---- episode-level coverage (the primary early-warning unit) ----
            benign_far = (A_test[:, 0] == 0)
            false_alarm_windows = int(((predicted == 1) & benign_far).sum())
            episode_rows.append(
                {
                    "test_day": test_day, "model": model_name, "warning_signal": signal_name,
                    "threshold": threshold,
                    "total_episodes": len(episodes),
                    "cold_start_episodes": int(sum(1 for ep in episodes if ep["cold_start_episode"])),
                    "observable_onset_episodes": observable_count,
                    "episodes_warned": warned,
                    "episodes_missed": observable_count - warned,
                    "episode_coverage": warned / observable_count if observable_count else None,
                    "median_lead_time_sec": float(np.median(lead_times)) if lead_times else None,
                    "mean_lead_time_sec": float(np.mean(lead_times)) if lead_times else None,
                    "earliest_warning_sec": float(max(lead_times)) if lead_times else None,
                    "false_alarm_windows": false_alarm_windows,
                    "false_alarms_per_hour": false_alarm_windows / fold_result["hours"],
                }
            )

    return pd.DataFrame(onset_rows), pd.DataFrame(episode_rows), episodes


# =============================================================
# ATTACK FAMILY / DISTRIBUTION SHIFT
# =============================================================

def attack_family_generalization(fold_results, master):
    inventory = pd.read_csv(RESULTS_DIR / "day_inventory.csv")
    families_by_day = {}
    for _, row in inventory.iterrows():
        families = {}
        for entry in str(row["attack_families"]).split(";"):
            entry = entry.strip()
            if not entry:
                continue
            name = entry.split("(")[0]
            count = int(entry.split("(")[1].rstrip(")"))
            families[name] = count
        families_by_day[row["day_key"]] = families

    rows = []
    for test_day, result in fold_results.items():
        training_families = set()
        for key in DAY_ORDER:
            if key != test_day:
                training_families |= set(families_by_day[key])

        onsets = result["onset_frame"]
        episodes = result["episodes"]

        for family, windows in families_by_day[test_day].items():
            family_episodes = [ep for ep in episodes if ep["primary_family"] == family]
            point = master[
                (master["test_day"] == test_day) & (master["evaluation_type"] == "point_forecast")
                & (master["horizon"] == 5)
            ]

            def metric(model_name, column):
                subset = point[point["model"] == model_name]
                return None if subset.empty else float(subset[column].iloc[0])

            def onset_recall(model_name):
                subset = result["transition_frame"][
                    (result["transition_frame"]["model"] == model_name)
                    & (result["transition_frame"]["horizon"] == 5)
                ]
                return None if subset.empty else subset["onset_recall"].iloc[0]

            rows.append(
                {
                    "test_day": test_day,
                    "source_day": DAYS[test_day],
                    "attack_family": family,
                    "seen_in_training": family in training_families,
                    "training_families": "; ".join(sorted(training_families)),
                    "num_test_episodes": len(family_episodes),
                    "num_test_windows": windows,
                    "V4_F1": metric(MODEL_V4, "f1"),
                    "V4_ROC_AUC": metric(MODEL_V4, "roc_auc"),
                    "V4_onset_recall": onset_recall(MODEL_V4),
                    "LR_F1": metric(MODEL_LR, "f1"),
                    "LR_ROC_AUC": metric(MODEL_LR, "roc_auc"),
                    "LR_onset_recall": onset_recall(MODEL_LR),
                    "note": "day-level metrics; a day's families are not separable window-by-window "
                            "when a day contains more than one family",
                    "metrics_are_family_specific": len(families_by_day[test_day]) == 1,
                }
            )
    return pd.DataFrame(rows)


def distribution_shift(fold_results):
    rows = []
    for test_day, result in fold_results.items():
        splits = result["splits"]
        indices = {name: INPUT_FEATURES.index(name) for name in SHIFT_FEATURES}

        partitions = {
            name: splits[name]["X"][:, -1, :] for name in ("train", "val", "test")
        }
        prevalences = {
            name: float(splits[name]["A_binary"][:, 0].mean()) for name in ("train", "val", "test")
        }

        rows.append(
            {
                "test_day": test_day, "feature": "positive_prevalence_t1",
                "space": "label",
                "train_mean": prevalences["train"], "val_mean": prevalences["val"],
                "test_mean": prevalences["test"],
                "test_minus_train_mean": prevalences["test"] - prevalences["train"],
                "test_over_train_ratio": prevalences["test"] / prevalences["train"]
                if prevalences["train"] else None,
                "train_std": None, "test_std": None, "std_difference": None,
                "p10_difference": None, "p50_difference": None, "p90_difference": None,
            }
        )

        for name, index in indices.items():
            train_values = partitions["train"][:, index]
            test_values = partitions["test"][:, index]
            val_values = partitions["val"][:, index]
            rows.append(
                {
                    "test_day": test_day, "feature": name,
                    "space": "fold-scaled (as the model sees it: train-fitted StandardScaler)",
                    "train_mean": float(train_values.mean()),
                    "val_mean": float(val_values.mean()),
                    "test_mean": float(test_values.mean()),
                    "test_minus_train_mean": float(test_values.mean() - train_values.mean()),
                    "test_over_train_ratio": None,
                    "train_std": float(train_values.std()),
                    "test_std": float(test_values.std()),
                    "std_difference": float(test_values.std() - train_values.std()),
                    "p10_difference": float(np.quantile(test_values, 0.10) - np.quantile(train_values, 0.10)),
                    "p50_difference": float(np.quantile(test_values, 0.50) - np.quantile(train_values, 0.50)),
                    "p90_difference": float(np.quantile(test_values, 0.90) - np.quantile(train_values, 0.90)),
                }
            )
    return pd.DataFrame(rows)


# =============================================================
# STAGES
# =============================================================

def stage_evaluate(fold_keys):
    print("\n" + "=" * 100)
    print("STAGE 5 — EVALUATE EVERY FOLD (validation-only thresholds, one test pass)")
    print("=" * 100)

    device = get_device()
    all_rows, all_transitions, all_onsets, all_episodes = [], [], [], []
    fold_results = {}

    for test_day in fold_keys:
        print(f"\n  Fold {test_day} ({DAYS[test_day]})")
        result = evaluate_fold(test_day, device)
        onset_frame, episode_frame, episodes = early_warning(test_day, result)

        result["onset_frame"] = onset_frame
        result["transition_frame"] = pd.DataFrame(result["transition_rows"])
        result["episodes"] = episodes
        fold_results[test_day] = result

        all_rows.extend(result["rows"])
        all_transitions.extend(result["transition_rows"])
        all_onsets.append(onset_frame)
        all_episodes.append(episode_frame)

        point = pd.DataFrame(result["rows"])
        point = point[(point["evaluation_type"] == "point_forecast") & (point["horizon"] == 5)]
        for _, row in point.iterrows():
            print(f"    K=5 {row['model']:<20} F1={row['f1']:.4f}  AUC="
                  f"{row['roc_auc'] if row['roc_auc'] is None else round(row['roc_auc'], 4)}  "
                  f"thr={row['threshold']}")

    master = pd.DataFrame(all_rows)
    columns = [
        "test_day", "source_day", "model", "evaluation_type", "horizon", "threshold",
        "precision", "recall", "f1", "roc_auc", "fpr", "tp", "fp", "tn", "fn",
        "pos_samples", "neg_samples", "positive_prevalence", "transition_fraction",
        "sample_count", "fp_per_hour", "roc_auc_degenerate",
    ]
    master = master[columns + [c for c in master.columns if c not in columns]]
    master.to_csv(MASTER_CSV, index=False)

    pd.DataFrame(all_transitions).to_csv(RESULTS_DIR / "transition_diagnostics.csv", index=False)
    pd.concat(all_onsets, ignore_index=True).to_csv(RESULTS_DIR / "onset_results.csv", index=False)
    pd.concat(all_episodes, ignore_index=True).to_csv(RESULTS_DIR / "episode_warning_results.csv", index=False)

    attack_family_generalization(fold_results, master).to_csv(
        RESULTS_DIR / "attack_family_generalization.csv", index=False
    )
    distribution_shift(fold_results).to_csv(RESULTS_DIR / "distribution_shift.csv", index=False)

    # per-fold test predictions
    for test_day, result in fold_results.items():
        meta = result["splits"]["test"]["meta"].copy()
        A_test = result["splits"]["test"]["A_binary"].astype(int)
        predictions = meta[["timestamp", "source_day", "label_at_time_t"]].copy()
        for k in POINT_HORIZONS:
            predictions[f"target_t+{k}"] = A_test[:, k - 1]
            predictions[f"v4_prob_t+{k}"] = result["scores"]["test"][MODEL_V4][:, k]
            predictions[f"v4_pred_t+{k}"] = (
                result["scores"]["test"][MODEL_V4][:, k] >= result["thresholds"][MODEL_V4][k]
            ).astype(int)
            predictions[f"lr_prob_t+{k}"] = result["scores"]["test"][MODEL_LR][k]
            predictions[f"lr_pred_t+{k}"] = (
                result["scores"]["test"][MODEL_LR][k] >= result["thresholds"][MODEL_LR][k]
            ).astype(int)
            predictions[f"persistence_pred_t+{k}"] = result["scores"]["test"][MODEL_PERSISTENCE].astype(int)
        (V4_DIR / f"fold_test_{test_day}").mkdir(parents=True, exist_ok=True)
        predictions.to_csv(V4_DIR / f"fold_test_{test_day}" / "test_predictions.csv", index=False)

        with open(V4_DIR / f"fold_test_{test_day}" / "thresholds.json", "w") as handle:
            json.dump(
                {
                    "selection_source": "validation",
                    "selection_method": "F1 argmax over np.arange(0.01, 0.99, 0.01)",
                    "test_labels_used": False,
                    "point": {str(k): v for k, v in result["thresholds"][MODEL_V4].items()},
                    "trajectory_max": {str(k): v for k, v in result["trajectory_thresholds"][MODEL_V4].items()},
                    "lr_point": {str(k): v for k, v in result["thresholds"][MODEL_LR].items()},
                    "lr_trajectory_max": {str(k): v for k, v in result["trajectory_thresholds"][MODEL_LR].items()},
                },
                handle, indent=2,
            )
            handle.write("\n")

    print(f"\nWrote {MASTER_CSV.relative_to(REPO_ROOT)} ({len(master)} rows)")
    return fold_results


def stage_summarize(fold_keys, skip_plots=False):
    print("\n" + "=" * 100)
    print("STAGE 6 — CROSS-FOLD SUMMARY")
    print("=" * 100)

    master = pd.read_csv(MASTER_CSV)
    transitions = pd.read_csv(RESULTS_DIR / "transition_diagnostics.csv")
    episodes = pd.read_csv(RESULTS_DIR / "episode_warning_results.csv")
    inventory = pd.read_csv(RESULTS_DIR / "day_inventory.csv")

    point = master[master["evaluation_type"] == "point_forecast"]
    rows = []
    for (model_name, k), group in point.groupby(["model", "horizon"]):
        entry = {"model": model_name, "horizon": k, "folds": int(len(group))}
        for column, prefix in (("f1", "f1"), ("roc_auc", "roc_auc")):
            values = group[column].dropna()
            entry.update(
                {
                    f"mean_{prefix}": float(values.mean()), f"median_{prefix}": float(values.median()),
                    f"std_{prefix}": float(values.std(ddof=1)) if len(values) > 1 else None,
                    f"min_{prefix}": float(values.min()), f"max_{prefix}": float(values.max()),
                    f"min_{prefix}_day": group.loc[values.idxmin(), "test_day"] if len(values) else None,
                    f"max_{prefix}_day": group.loc[values.idxmax(), "test_day"] if len(values) else None,
                }
            )
        entry.update(
            {
                "mean_precision": float(group["precision"].mean()),
                "mean_recall": float(group["recall"].mean()),
                "mean_fpr": float(group["fpr"].mean()),
                "std_precision": float(group["precision"].std(ddof=1)),
                "std_recall": float(group["recall"].std(ddof=1)),
            }
        )
        rows.append(entry)
    cross_fold = pd.DataFrame(rows)
    cross_fold.to_csv(RESULTS_DIR / "cross_fold_summary.csv", index=False)

    rows = []
    for (model_name, k), group in transitions.groupby(["model", "horizon"]):
        coverage = episodes[
            (episodes["model"] == model_name) & (episodes["warning_signal"] == "trajectory_max_k5")
        ]
        rows.append(
            {
                "model": model_name, "horizon": k, "folds": int(len(group)),
                "total_onsets": int(group["onset_count"].sum()),
                "total_offsets": int(group["offset_count"].sum()),
                "total_onsets_detected": int(group["onset_detected"].sum()),
                "pooled_onset_recall": float(group["onset_detected"].sum() / group["onset_count"].sum())
                if group["onset_count"].sum() else None,
                "mean_onset_recall": float(group["onset_recall"].dropna().mean()),
                "min_onset_recall": float(group["onset_recall"].dropna().min()),
                "max_onset_recall": float(group["onset_recall"].dropna().max()),
                "pooled_offset_recall": float(group["offset_correct"].sum() / group["offset_count"].sum())
                if group["offset_count"].sum() else None,
                "mean_transition_count": float(group["transition_count"].mean()),
                "expected_onset_hits_at_base_rate": float(group["expected_onset_hits_at_base_rate"].sum()),
                "mean_persistence_accuracy": float(group["persistence_accuracy"].mean()),
                "episode_coverage_mean": float(coverage["episode_coverage"].dropna().mean())
                if not coverage.empty else None,
                "median_lead_time_sec": float(coverage["median_lead_time_sec"].dropna().median())
                if not coverage.empty and coverage["median_lead_time_sec"].notna().any() else None,
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "transition_cross_fold_summary.csv", index=False)

    # ---- Mar-02 historical comparison ----
    historical = json.loads((REPO_ROOT / "docs" / "v4_baseline_manifest.json").read_text())
    reference = historical["results_verified"]["future_prediction"]
    rotated = master[
        (master["test_day"] == "Mar02") & (master["model"] == MODEL_V4)
        & (master["evaluation_type"] == "point_forecast")
    ]
    comparison_rows = []
    for k in POINT_HORIZONS:
        row = rotated[rotated["horizon"] == k]
        if row.empty:
            continue
        row = row.iloc[0]
        ref = reference[f"k={k}"]
        for metric in ("f1", "precision", "recall", "roc_auc", "threshold"):
            historical_value = float(ref[metric])
            rotated_value = float(row[metric])
            comparison_rows.append(
                {
                    "metric": f"K={k} {metric}",
                    "historical_frozen_v4": historical_value,
                    "rotated_fold_v4": rotated_value,
                    "absolute_difference": rotated_value - historical_value,
                    "interpretation": (
                        "Both evaluate the same Mar-2 windows with the same architecture and the same "
                        "validation-only threshold rule. They differ in training provenance: the frozen "
                        "checkpoint was trained on the single-day temporal_benchmark_v2 arrays and scored "
                        "under a different scaler, while the rotated model is trained and scored entirely "
                        "inside this fold. Neither is ground truth."
                    ),
                }
            )
    pd.DataFrame(comparison_rows).to_csv(RESULTS_DIR / "mar02_reference_comparison.csv", index=False)

    print("\nCross-fold point-forecast summary (K=5):")
    print(cross_fold[cross_fold["horizon"] == 5][
        ["model", "mean_f1", "median_f1", "std_f1", "min_f1", "max_f1", "min_f1_day",
         "max_f1_day", "mean_roc_auc", "std_roc_auc"]
    ].round(4).to_string(index=False))

    print("\nPer-day K=5 point forecast:")
    print(point[point["horizon"] == 5][
        ["test_day", "model", "threshold", "f1", "precision", "recall", "roc_auc",
         "positive_prevalence", "transition_fraction"]
    ].round(4).to_string(index=False))

    print("\nMar-02: historical frozen vs rotated fold")
    print(pd.DataFrame(comparison_rows)[
        ["metric", "historical_frozen_v4", "rotated_fold_v4", "absolute_difference"]
    ].round(4).to_string(index=False))

    if not skip_plots:
        make_figures(master, transitions, episodes, inventory, cross_fold)
        print(f"\nWrote figures to {FIG_DIR.relative_to(REPO_ROOT)}")

    summary = {
        "experiment": "rotated_leave_one_day_out_generalization",
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "is_frozen_reference": False,
        "folds": {
            key: {
                "test_day": DAYS[key],
                "train_days": [DAYS[other] for other in DAY_ORDER if other != key],
                "checkpoint": f"models/rotated_v4/fold_test_{key}/v4_rotated_{key}.pt",
                "checkpoint_sha256": sha256_file(
                    V4_DIR / f"fold_test_{key}" / f"v4_rotated_{key}.pt"
                ),
                "input_scaler_sha256": sha256_file(RESULTS_DIR / f"fold_{key}" / "input_scaler.pkl"),
                "target_scaler_sha256": sha256_file(RESULTS_DIR / f"fold_{key}" / "target_scaler.pkl"),
            }
            for key in fold_keys
        },
        "configuration": {"v4": {k: str(v) for k, v in V4_CONFIG.items()}, "logistic_regression": LR_CONFIG},
        "day_inventory": json.loads(inventory.to_json(orient="records")),
        "point_metrics": json.loads(point.to_json(orient="records")),
        "trajectory_metrics": json.loads(
            master[master["evaluation_type"] == "trajectory_event_max"].to_json(orient="records")
        ),
        "transition_diagnostics": json.loads(transitions.to_json(orient="records")),
        "episode_diagnostics": json.loads(episodes.to_json(orient="records")),
        "cross_fold_summary": json.loads(cross_fold.to_json(orient="records")),
        "mar02_reference_comparison": comparison_rows,
        "git_commit": git_value("rev-parse", "HEAD"),
        "caveats": [
            "Four source days only. The folds are not four independent statistical experiments: "
            "each training set shares two days with every other training set.",
            "Every day carries a different attack family, so every fold is simultaneously an "
            "unseen-family test. Cross-day variation and cross-family variation cannot be separated.",
            "Persistence receives the observed label A(t); V4 and LR receive only features.",
            "Fold 4 (Mar-02) is a NEW experiment, not a replacement for the frozen V4 reference.",
            "Windows inside one episode are temporally dependent; episode counts per day are 2-5.",
        ],
    }
    with open(SUMMARY_JSON, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {SUMMARY_JSON.relative_to(REPO_ROOT)}")


def make_figures(master, transitions, episodes, inventory, cross_fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    point = master[master["evaluation_type"] == "point_forecast"]
    colours = {MODEL_PERSISTENCE: "#8c8c8c", MODEL_LR: "#d1651a", MODEL_V4: "#1f77b4"}
    days = [key for key in DAY_ORDER if key in set(point["test_day"])]
    positions = np.arange(len(days))
    width = 0.26

    # 1 & 2: F1 and ROC-AUC by held-out day (K=5)
    for column, filename, title in (
        ("f1", "f1_by_heldout_day.png", "F1 by held-out day (K=5, frozen per-fold validation thresholds)"),
        ("roc_auc", "roc_auc_by_heldout_day.png", "ROC-AUC by held-out day (K=5)"),
    ):
        figure, axes = plt.subplots(figsize=(8.6, 4.4))
        for index, (model_name, colour) in enumerate(colours.items()):
            values = [
                float(point[(point["test_day"] == day) & (point["model"] == model_name)
                            & (point["horizon"] == 5)][column].iloc[0])
                for day in days
            ]
            axes.bar(positions + (index - 1) * width, values, width, color=colour, label=model_name)
            for x, value in zip(positions + (index - 1) * width, values):
                axes.text(x, value + 0.015, f"{value:.3f}", ha="center", fontsize=7.5)
        axes.set_xticks(positions)
        axes.set_xticklabels([f"{day}\n{DAYS[day].split('_')[0]}" for day in days], fontsize=8)
        axes.set_ylim(0, 1.12)
        axes.set_ylabel(column.upper() if column == "f1" else "ROC-AUC")
        axes.set_title(title + "\nPersistence ROC-AUC is a degenerate two-point curve", fontsize=10.5)
        axes.grid(alpha=0.3, linestyle=":", axis="y")
        axes.legend(fontsize=8, ncol=3)
        figure.tight_layout()
        figure.savefig(FIG_DIR / filename, dpi=150)
        plt.close(figure)

    # 3: F1 vs horizon across days
    figure, axes_list = plt.subplots(1, len(days), figsize=(3.6 * len(days), 3.8), squeeze=False)
    for axes, day in zip(axes_list[0], days):
        for model_name, colour in colours.items():
            subset = point[(point["test_day"] == day) & (point["model"] == model_name)].sort_values("horizon")
            axes.plot(subset["horizon"], subset["f1"], marker="o", color=colour, label=model_name)
        axes.set_xticks(POINT_HORIZONS)
        axes.set_ylim(0, 1.05)
        axes.set_title(f"held-out {day}", fontsize=10)
        axes.set_xlabel("horizon K")
        axes.grid(alpha=0.3, linestyle=":")
    axes_list[0][0].set_ylabel("F1")
    axes_list[0][-1].legend(fontsize=7)
    figure.suptitle("F1 vs forecast horizon, by held-out day", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "f1_vs_horizon_by_day.png", dpi=150)
    plt.close(figure)

    # 4 & 5: persistence accuracy and transition fraction by day
    persistence = transitions[transitions["model"] == MODEL_V4]
    figure, axes_pair = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for k in POINT_HORIZONS:
        subset = persistence[persistence["horizon"] == k]
        values = [float(subset[subset["test_day"] == day]["persistence_accuracy"].iloc[0]) for day in days]
        axes_pair[0].plot(positions, values, marker="o", label=f"K={k}")
        counts = [int(subset[subset["test_day"] == day]["transition_count"].iloc[0]) for day in days]
        axes_pair[1].plot(positions, counts, marker="s", label=f"K={k}")
    for axes, title, ylabel in (
        (axes_pair[0], "Label persistence P(A(t+k)=A(t)) by held-out day", "persistence accuracy"),
        (axes_pair[1], "Transition windows by held-out day", "transition windows"),
    ):
        axes.set_xticks(positions)
        axes.set_xticklabels(days)
        axes.set_title(title, fontsize=10.5)
        axes.set_ylabel(ylabel)
        axes.grid(alpha=0.3, linestyle=":")
        axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "persistence_and_transitions_by_day.png", dpi=150)
    plt.close(figure)

    # 6: V4 onset recall by day
    figure, axes = plt.subplots(figsize=(8.0, 4.2))
    for index, (model_name, colour) in enumerate([(MODEL_V4, "#1f77b4"), (MODEL_LR, "#d1651a")]):
        subset = transitions[(transitions["model"] == model_name) & (transitions["horizon"] == 5)]
        values, counts = [], []
        for day in days:
            row = subset[subset["test_day"] == day]
            values.append(float(row["onset_recall"].iloc[0]) if not row.empty
                          and row["onset_recall"].notna().iloc[0] else 0.0)
            counts.append(int(row["onset_count"].iloc[0]) if not row.empty else 0)
        axes.bar(positions + (index - 0.5) * width, values, width, color=colour, label=model_name)
        expected = [
            float(subset[subset["test_day"] == day]["expected_onset_hits_at_base_rate"].iloc[0])
            / max(counts[i], 1) for i, day in enumerate(days)
        ]
        axes.plot(positions + (index - 0.5) * width, expected, "k_", markersize=14,
                  label="chance at own base rate" if index == 0 else None)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{day}\n{counts[i]} onset windows" for i, day in enumerate(days)], fontsize=8)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("ONSET recall (K=5)")
    axes.set_title("Transition-conditioned diagnostic: recall on 0→1 windows\n"
                   "black dashes = each model's own chance rate", fontsize=10.5)
    axes.grid(alpha=0.3, linestyle=":", axis="y")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "onset_recall_by_day.png", dpi=150)
    plt.close(figure)

    # 7: episode-level coverage
    coverage = episodes[episodes["warning_signal"] == "trajectory_max_k5"]
    figure, axes = plt.subplots(figsize=(8.4, 4.2))
    for index, (model_name, colour) in enumerate(colours.items()):
        values = []
        for day in days:
            row = coverage[(coverage["test_day"] == day) & (coverage["model"] == model_name)]
            values.append(float(row["episode_coverage"].iloc[0])
                          if not row.empty and row["episode_coverage"].notna().iloc[0] else 0.0)
        axes.bar(positions + (index - 1) * width, values, width, color=colour, label=model_name)
        for x, value in zip(positions + (index - 1) * width, values):
            axes.text(x, value + 0.02, f"{value:.2f}", ha="center", fontsize=7.5)
    axes.set_xticks(positions)
    axes.set_xticklabels(days)
    axes.set_ylim(0, 1.15)
    axes.set_ylabel("episodes warned / observable onsets")
    axes.set_title("Episode-level early-warning coverage (trajectory MAX K=5, 120 s lookback)", fontsize=10.5)
    axes.grid(alpha=0.3, linestyle=":", axis="y")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "episode_warning_coverage.png", dpi=150)
    plt.close(figure)

    # 8: prevalence by day
    fold_inventory = pd.read_csv(RESULTS_DIR / "fold_inventory.csv")
    figure, axes = plt.subplots(figsize=(8.0, 4.2))
    for index, partition in enumerate(("train", "val", "test")):
        values = [
            float(fold_inventory[(fold_inventory["test_day"] == day)
                                 & (fold_inventory["partition"] == partition)]
                  ["positive_prevalence_t1"].iloc[0]) for day in days
        ]
        axes.bar(positions + (index - 1) * width, values, width, label=partition)
    axes.set_xticks(positions)
    axes.set_xticklabels(days)
    axes.set_ylabel("positive prevalence at t+1")
    axes.set_title("Attack prevalence by partition and held-out day", fontsize=10.5)
    axes.grid(alpha=0.3, linestyle=":", axis="y")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "prevalence_by_day.png", dpi=150)
    plt.close(figure)

    # 9 & 10: performance vs prevalence shift, performance vs transition fraction
    shift = pd.read_csv(RESULTS_DIR / "distribution_shift.csv")
    prevalence_shift = shift[shift["feature"] == "positive_prevalence_t1"]
    figure, axes_pair = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for model_name, colour in ((MODEL_V4, "#1f77b4"), (MODEL_LR, "#d1651a")):
        subset = point[(point["model"] == model_name) & (point["horizon"] == 5)]
        x_shift = [float(prevalence_shift[prevalence_shift["test_day"] == day]
                         ["test_minus_train_mean"].iloc[0]) for day in subset["test_day"]]
        axes_pair[0].scatter(x_shift, subset["f1"], color=colour, s=60, label=model_name)
        for x, y, day in zip(x_shift, subset["f1"], subset["test_day"]):
            axes_pair[0].annotate(day, (x, y), fontsize=7.5, xytext=(3, 3), textcoords="offset points")

        axes_pair[1].scatter(subset["transition_fraction"], subset["f1"], color=colour, s=60, label=model_name)
        for x, y, day in zip(subset["transition_fraction"], subset["f1"], subset["test_day"]):
            axes_pair[1].annotate(day, (x, y), fontsize=7.5, xytext=(3, 3), textcoords="offset points")

    axes_pair[0].set_xlabel("test − train positive prevalence")
    axes_pair[1].set_xlabel("transition fraction on the held-out day (K=5)")
    for axes in axes_pair:
        axes.set_ylabel("F1 (K=5)")
        axes.grid(alpha=0.3, linestyle=":")
        axes.legend(fontsize=8)
        axes.set_ylim(0, 1.0)
    axes_pair[0].set_title("F1 vs prevalence shift (n=4 folds)", fontsize=10.5)
    axes_pair[1].set_title("F1 vs transition density (n=4 folds)", fontsize=10.5)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "performance_vs_shift.png", dpi=150)
    plt.close(figure)
