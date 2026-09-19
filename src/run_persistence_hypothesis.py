"""
=============================================================
PERSISTENCE HYPOTHESIS TEST
How much of V4's future-attack performance is label persistence?
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Analysis only. Nothing is trained, retrained, re-thresholded or modified.

Central question:
    The benchmark target A(t+k) is highly autocorrelated with A(t).
    Persistence (y_hat(t+k) = A(t)) is therefore correct on every window
    that does NOT change state. This script separates

        persistent-state prediction   (A(t+k) == A(t))
    from
        transition prediction         (A(t+k) != A(t))

    and measures every model on each part separately.

Inputs (frozen, read-only):
    results/error_analysis/error_events.csv        verified per-window predictions
    results/error_analysis/attack_episodes.csv     verified ground-truth episodes
    results/standardized_baseline_comparison.csv   Task 3 cross-check
    docs/v4_baseline_manifest.json, docs/logistic_regression_baseline_manifest.json

Outputs: results/persistence_hypothesis/*, results/persistence_hypothesis_summary.json

Usage:
    python src/run_persistence_hypothesis.py
    python src/run_persistence_hypothesis.py --skip-plots --skip-preflight
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_standardized_comparison import (  # noqa: E402
    MODEL_LR,
    MODEL_PERSISTENCE,
    MODEL_V4,
    POINT_HORIZONS,
    git_value,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

EVENTS_CSV = REPO_ROOT / "results" / "error_analysis" / "error_events.csv"
EPISODES_CSV = REPO_ROOT / "results" / "error_analysis" / "attack_episodes.csv"
TASK3_CSV = REPO_ROOT / "results" / "standardized_baseline_comparison.csv"
TASK4_SUMMARY = REPO_ROOT / "results" / "error_analysis" / "horizon_error_summary.csv"
V4_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
LR_MANIFEST = REPO_ROOT / "docs" / "logistic_regression_baseline_manifest.json"

OUT_DIR = REPO_ROOT / "results" / "persistence_hypothesis"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_JSON = REPO_ROOT / "results" / "persistence_hypothesis_summary.json"

MODELS = [MODEL_PERSISTENCE, MODEL_LR, MODEL_V4]
FEATURE_MODELS = [MODEL_LR, MODEL_V4]
WINDOW_SECONDS = 10
PRE_TRANSITION_WINDOWS = 12          # 120 s
SHUFFLE_SEED = 42
SHUFFLE_DRAWS = 1000
TOLERANCE = 1e-9

UNAVAILABLE = "unavailable (single-class subset)"


# =============================================================
# HELPERS
# =============================================================

def safe_f1(actual, predicted):
    if len(np.unique(actual)) < 2:
        return None
    return float(f1_score(actual, predicted, zero_division=0))


def safe_auc(actual, score):
    if len(np.unique(actual)) < 2:
        return None
    return float(roc_auc_score(actual, score))


def preflight():
    """Re-verify the frozen chain before consuming any of it."""
    results = {}
    for script in (
        "verify_v4_baseline.py",
        "verify_logistic_regression_baseline.py",
        "verify_error_analysis.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / script)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        overall = "UNKNOWN"
        for line in completed.stdout.splitlines():
            if line.startswith("OVERALL:"):
                overall = line.split(":", 1)[1].strip()
        results[script] = overall
        print(f"  {script:<42}: {overall}")
        if overall != "PASS":
            raise RuntimeError(f"{script} did not pass; analysis aborted.")
    return results


# =============================================================
# PART 1 — LABEL PERSISTENCE
# =============================================================

def persistence_statistics(events):
    rows = []
    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_PERSISTENCE) & (events["horizon"] == k)]
        label_t = group["label_at_time_t"].to_numpy()
        label_future = group["actual_label"].to_numpy()

        current_attack = label_t == 1
        current_benign = label_t == 0

        rows.append(
            {
                "horizon": k,
                "n_evaluated": int(len(group)),
                "persistence_accuracy": float((label_future == label_t).mean()),
                "attack_persistence_rate": float(label_future[current_attack].mean()),
                "benign_persistence_rate": float((label_future[current_benign] == 0).mean()),
                "onset_rate": float(label_future[current_benign].mean()),
                "offset_rate": float((label_future[current_attack] == 0).mean()),
                "n_current_attack": int(current_attack.sum()),
                "n_current_benign": int(current_benign.sum()),
                "n_current_benign_future_attack": int((current_benign & (label_future == 1)).sum()),
                "n_current_attack_future_benign": int((current_attack & (label_future == 0)).sum()),
                "transition_count": int((label_future != label_t).sum()),
                "transition_fraction": float((label_future != label_t).mean()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================
# PART 2/3 — TRANSITION DECOMPOSITION AND PER-CATEGORY PERFORMANCE
# =============================================================

def category_of(label_t, label_future):
    category = np.empty(len(label_t), dtype=object)
    category[(label_t == 0) & (label_future == 0)] = "PP"
    category[(label_t == 0) & (label_future == 1)] = "ONSET"
    category[(label_t == 1) & (label_future == 0)] = "OFFSET"
    category[(label_t == 1) & (label_future == 1)] = "AA"
    return category


def transition_decomposition(events, n_test):
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("sample_index")
        categories = category_of(
            group["label_at_time_t"].to_numpy(), group["actual_label"].to_numpy()
        )
        group = group.assign(transition_category=categories)

        for category in ("PP", "ONSET", "OFFSET", "AA"):
            subset = group[group["transition_category"] == category]
            if subset.empty:
                continue
            correct = int((subset["predicted_label"] == subset["actual_label"]).sum())
            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "transition_category": category,
                    "count": int(len(subset)),
                    "percentage_of_test": 100.0 * len(subset) / n_test,
                    "correct": correct,
                    "incorrect": int(len(subset)) - correct,
                    "prediction_rate": float(subset["predicted_label"].mean()),
                    "mean_probability": float(subset["predicted_probability"].mean()),
                    "correct_by_construction": model == MODEL_PERSISTENCE
                    and category in ("PP", "AA"),
                    "incorrect_by_construction": model == MODEL_PERSISTENCE
                    and category in ("ONSET", "OFFSET"),
                }
            )
    return pd.DataFrame(rows)


def transition_model_performance(events):
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("sample_index")
        categories = category_of(
            group["label_at_time_t"].to_numpy(), group["actual_label"].to_numpy()
        )
        group = group.assign(transition_category=categories)

        for category in ("PP", "ONSET", "OFFSET", "AA"):
            subset = group[group["transition_category"] == category]
            if subset.empty:
                continue

            correct = int((subset["predicted_label"] == subset["actual_label"]).sum())
            single_class = subset["actual_label"].nunique() < 2
            positives = subset[subset["actual_label"] == 1]
            negatives = subset[subset["actual_label"] == 0]

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "transition_category": category,
                    "sample_count": int(len(subset)),
                    "correct_count": correct,
                    "incorrect_count": int(len(subset)) - correct,
                    "accuracy": correct / len(subset),
                    "mean_probability": float(subset["predicted_probability"].mean()),
                    "median_probability": float(subset["predicted_probability"].median()),
                    "mean_probability_positive_target": float(positives["predicted_probability"].mean())
                    if len(positives) else None,
                    "mean_probability_negative_target": float(negatives["predicted_probability"].mean())
                    if len(negatives) else None,
                    "f1_within_category": UNAVAILABLE if single_class else safe_f1(
                        subset["actual_label"], subset["predicted_label"]
                    ),
                    "roc_auc_within_category": UNAVAILABLE if single_class else safe_auc(
                        subset["actual_label"], subset["predicted_probability"]
                    ),
                    "note": "single-class category — F1/AUC are undefined here; accuracy is the "
                            "meaningful statistic" if single_class else "",
                }
            )
    return pd.DataFrame(rows)


# =============================================================
# PART 4/5 — PERSISTENCE EXCESS AND TRANSITION-ONLY DIAGNOSTIC
# =============================================================

def persistence_excess(events):
    reference = {}
    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_PERSISTENCE) & (events["horizon"] == k)].sort_values(
            "sample_index"
        )
        reference[k] = group

    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("sample_index")
        is_transition = group["actual_label"].to_numpy() != group["label_at_time_t"].to_numpy()

        persistent = group[~is_transition]
        transition = group[is_transition]

        persistence_group = reference[k]
        persistence_f1 = f1_score(
            persistence_group["actual_label"], persistence_group["predicted_label"], zero_division=0
        )
        persistence_transition_f1 = f1_score(
            persistence_group[is_transition]["actual_label"],
            persistence_group[is_transition]["predicted_label"],
            zero_division=0,
        )

        overall_f1 = f1_score(group["actual_label"], group["predicted_label"], zero_division=0)
        transition_f1 = safe_f1(transition["actual_label"], transition["predicted_label"])

        rows.append(
            {
                "model": model,
                "horizon": k,
                "overall_f1": float(overall_f1),
                "persistent_subset_f1": safe_f1(persistent["actual_label"], persistent["predicted_label"]),
                "transition_subset_f1": transition_f1,
                "overall_auc": safe_auc(group["actual_label"], group["predicted_probability"]),
                "persistent_subset_auc": safe_auc(
                    persistent["actual_label"], persistent["predicted_probability"]
                ),
                "transition_subset_auc": safe_auc(
                    transition["actual_label"], transition["predicted_probability"]
                ),
                "persistent_subset_n": int(len(persistent)),
                "transition_subset_n": int(len(transition)),
                "persistent_subset_accuracy": float(
                    (persistent["predicted_label"] == persistent["actual_label"]).mean()
                ),
                "transition_subset_accuracy": float(
                    (transition["predicted_label"] == transition["actual_label"]).mean()
                ),
                "v4_minus_persistence_overall": None,
                "v4_minus_persistence_transition": None,
                "persistence_overall_f1_reference": float(persistence_f1),
                "persistence_transition_f1_reference": float(persistence_transition_f1),
                "note": "transition subset holds only %d windows — F1/AUC there are descriptive, "
                        "not statistically reliable" % int(len(transition)),
            }
        )

    frame = pd.DataFrame(rows)
    for k in POINT_HORIZONS:
        v4 = frame[(frame["model"] == MODEL_V4) & (frame["horizon"] == k)]
        persistence = frame[(frame["model"] == MODEL_PERSISTENCE) & (frame["horizon"] == k)]
        if v4.empty or persistence.empty:
            continue
        frame.loc[(frame["model"] == MODEL_V4) & (frame["horizon"] == k), "v4_minus_persistence_overall"] = (
            v4["overall_f1"].iloc[0] - persistence["overall_f1"].iloc[0]
        )
        frame.loc[
            (frame["model"] == MODEL_V4) & (frame["horizon"] == k), "v4_minus_persistence_transition"
        ] = (v4["transition_subset_accuracy"].iloc[0] - persistence["transition_subset_accuracy"].iloc[0])
    return frame


def transition_only_metrics(events):
    """Transition-conditioned DIAGNOSTIC — not a standalone benchmark score."""
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("sample_index")
        label_t = group["label_at_time_t"].to_numpy()
        actual = group["actual_label"].to_numpy()

        onset = group[(label_t == 0) & (actual == 1)]
        offset = group[(label_t == 1) & (actual == 0)]
        transition = pd.concat([onset, offset])

        positive_predictions = int(transition["predicted_label"].sum())
        onset_hits = int(onset["predicted_label"].sum())

        rows.append(
            {
                "model": model,
                "horizon": k,
                "evaluation": "transition-conditioned diagnostic (NOT an overall benchmark)",
                "onset_n": int(len(onset)),
                "onset_detected": onset_hits,
                "onset_recall": onset_hits / len(onset) if len(onset) else None,
                "onset_mean_probability": float(onset["predicted_probability"].mean()) if len(onset) else None,
                "onset_max_probability": float(onset["predicted_probability"].max()) if len(onset) else None,
                "offset_n": int(len(offset)),
                "offset_correct": int((offset["predicted_label"] == 0).sum()),
                "offset_recall_negative": float((offset["predicted_label"] == 0).mean()) if len(offset) else None,
                "offset_mean_probability": float(offset["predicted_probability"].mean()) if len(offset) else None,
                "offset_min_probability": float(offset["predicted_probability"].min()) if len(offset) else None,
                "transition_precision": onset_hits / positive_predictions if positive_predictions else None,
                "transition_precision_note": "undefined — no positive prediction on the transition subset"
                if not positive_predictions else "",
                "transition_n": int(len(transition)),
                "transition_accuracy": float(
                    (transition["predicted_label"] == transition["actual_label"]).mean()
                ),
                "predicted_positive_rate_overall": float(group["predicted_label"].mean()),
                "expected_onset_hits_at_base_rate": round(
                    float(group["predicted_label"].mean()) * len(onset), 3
                ),
                "expected_offset_correct_at_base_rate": round(
                    (1.0 - float(group["predicted_label"].mean())) * len(offset), 3
                ),
            }
        )
    return pd.DataFrame(rows)


# =============================================================
# PART 6/7 — ONSET AND OFFSET TRANSITION ANALYSIS
# =============================================================

def onset_transition_analysis(events, episodes):
    observable = episodes[episodes["observable_from_test_start"]]
    rows = []

    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("target_time")
        for _, ep in observable.iterrows():
            onset = ep["attack_onset"]
            window_start = onset - pd.Timedelta(seconds=WINDOW_SECONDS * PRE_TRANSITION_WINDOWS)
            pre = group[(group["target_time"] >= window_start) & (group["target_time"] < onset)]
            at_onset = group[group["target_time"] == onset]

            alerts = pre[pre["predicted_label"] == 1]
            first_alert = alerts["target_time"].min() if len(alerts) else None
            fired_at_onset = bool(at_onset["predicted_label"].iloc[0]) if len(at_onset) else None

            if len(alerts):
                note = ("warning raised before onset; %d of %d lookback windows are themselves "
                        "ground-truth attack windows" % (int(pre["actual_label"].sum()), len(pre)))
                status = "warning_before_onset"
            elif fired_at_onset:
                note = "silent before onset; fired on the onset window itself"
                status = "warning_only_after_onset"
            else:
                note = "silent before and at the onset window"
                status = "silent"

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "episode_id": int(ep["episode_id"]),
                    "attack_onset": str(onset),
                    "warning_detected": bool(len(alerts) > 0),
                    "warning_status": status,
                    "first_warning_time": str(first_alert) if first_alert is not None else "",
                    "lead_time_seconds": float((onset - first_alert).total_seconds())
                    if first_alert is not None else None,
                    "probability_at_onset": float(at_onset["predicted_probability"].iloc[0])
                    if len(at_onset) else None,
                    "fired_on_onset_window": fired_at_onset,
                    "max_pre_onset_probability": float(pre["predicted_probability"].max()) if len(pre) else None,
                    "mean_pre_onset_probability": float(pre["predicted_probability"].mean()) if len(pre) else None,
                    "pre_onset_windows": int(len(pre)),
                    "pre_onset_windows_that_are_attacks": int(pre["actual_label"].sum()),
                    "pre_onset_lookback_is_clean": bool(pre["actual_label"].sum() == 0),
                    "note": note,
                }
            )
    return pd.DataFrame(rows)


def offset_transition_analysis(events, episodes):
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("target_time")
        for _, ep in episodes.iterrows():
            offset = ep["attack_offset"]
            before = group[
                (group["target_time"] > offset - pd.Timedelta(seconds=WINDOW_SECONDS * PRE_TRANSITION_WINDOWS))
                & (group["target_time"] <= offset)
            ]
            after = group[
                (group["target_time"] > offset)
                & (group["target_time"] <= offset + pd.Timedelta(seconds=WINDOW_SECONDS * PRE_TRANSITION_WINDOWS))
            ]

            if after.empty:
                rows.append(
                    {
                        "model": model, "horizon": k, "episode_id": int(ep["episode_id"]),
                        "attack_offset": str(offset), "offset_observable": False,
                        "mean_probability_before_offset": float(before["predicted_probability"].mean())
                        if len(before) else None,
                        "mean_probability_after_offset": None,
                        "probability_drop": None,
                        "time_until_below_threshold_sec": None,
                        "positive_predictions_after_offset": None,
                        "windows_after_offset": 0,
                        "note": "offset falls at the end of the evaluated day — no post-offset windows",
                    }
                )
                continue

            below = after[after["predicted_label"] == 0]
            first_below = below["target_time"].min() if len(below) else None
            mean_before = float(before["predicted_probability"].mean()) if len(before) else None
            mean_after = float(after["predicted_probability"].mean())

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "episode_id": int(ep["episode_id"]),
                    "attack_offset": str(offset),
                    "offset_observable": True,
                    "mean_probability_before_offset": mean_before,
                    "mean_probability_after_offset": mean_after,
                    "probability_drop": (mean_before - mean_after) if mean_before is not None else None,
                    "probability_10s_after_offset": float(after["predicted_probability"].iloc[0]),
                    "time_until_below_threshold_sec": float((first_below - offset).total_seconds())
                    if first_below is not None else None,
                    "positive_predictions_after_offset": int(after["predicted_label"].sum()),
                    "windows_after_offset": int(len(after)),
                    "note": "prediction continues through the attack tail"
                    if int(after["predicted_label"].sum()) > 0 else "prediction stops at the offset",
                }
            )
    return pd.DataFrame(rows)


# =============================================================
# PART 9 — V4 K=1 -> K=5 CORRECTION DECOMPOSITION
# =============================================================

# Categories are defined HERE, before any counting.
#   onset_related : the onset window itself, or a benign target within 120 s before an onset
#   early         : attack target 10-60 s after its episode onset
#   late          : attack target within 120 s before its episode offset
#   mid           : attack target > 60 s after onset and > 120 s before offset
#   post_attack   : benign target within 120 s after an episode offset
#   benign        : any other benign target
#   short_episode : attack target in an episode shorter than 180 s (phases indistinguishable)

SHORT_EPISODE_SECONDS = 180


def phase_category(row, short_episodes):
    if row["actual_label"] == 1:
        if row["attack_episode_id"] in short_episodes:
            return "short_episode"
        if row["time_from_attack_onset_sec"] == 0:
            return "onset_related"
        if row["time_from_attack_onset_sec"] <= 60:
            return "early"
        if row["time_to_attack_offset_sec"] <= 120:
            return "late"
        return "mid"

    distance = row["signed_distance_to_attack_window_sec"]
    if -120 <= distance < 0:
        return "onset_related"
    if 0 < distance <= 120:
        return "post_attack"
    return "benign"


def correction_decomposition(events, episodes):
    short_episodes = set(
        episodes.loc[episodes["duration_sec"] < SHORT_EPISODE_SECONDS, "episode_id"]
    )

    rows = []
    # --- A: category-wise FN counts per horizon (well defined; targets differ per horizon) ---
    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_V4) & (events["horizon"] == k)].copy()
        group["phase"] = group.apply(lambda row: phase_category(row, short_episodes), axis=1)
        for phase, subset in group.groupby("phase"):
            attacks = subset[subset["actual_label"] == 1]
            if attacks.empty:
                continue
            rows.append(
                {
                    "analysis": "A_category_fn_counts",
                    "model": MODEL_V4,
                    "horizon": k,
                    "phase": phase,
                    "attack_windows": int(len(attacks)),
                    "fn": int((attacks["error_type"] == "FN").sum()),
                    "tp": int((attacks["error_type"] == "TP").sum()),
                    "recall": float((attacks["error_type"] == "TP").mean()),
                    "mean_probability": float(attacks["predicted_probability"].mean()),
                }
            )

    frame_a = pd.DataFrame(rows)

    # Delta between K=1 and K=5 per phase
    delta_rows = []
    k1 = frame_a[frame_a["horizon"] == 1].set_index("phase")
    k5 = frame_a[frame_a["horizon"] == 5].set_index("phase")
    total_delta = int(k1["fn"].sum() - k5["fn"].sum())
    for phase in sorted(set(k1.index) | set(k5.index)):
        fn1 = int(k1.loc[phase, "fn"]) if phase in k1.index else 0
        fn5 = int(k5.loc[phase, "fn"]) if phase in k5.index else 0
        delta_rows.append(
            {
                "analysis": "B_k1_minus_k5_fn_delta",
                "model": MODEL_V4,
                "horizon": "1->5",
                "phase": phase,
                "fn_k1": fn1,
                "fn_k5": fn5,
                "fn_reduction": fn1 - fn5,
                "fraction_of_total_reduction": (fn1 - fn5) / total_delta if total_delta else None,
                "attack_windows_k1": int(k1.loc[phase, "attack_windows"]) if phase in k1.index else 0,
                "attack_windows_k5": int(k5.loc[phase, "attack_windows"]) if phase in k5.index else 0,
            }
        )
    frame_b = pd.DataFrame(delta_rows)

    # --- C: target-aligned pairing (same target window, lead time 1 vs 5) ---
    v4_k1 = events[(events["model"] == MODEL_V4) & (events["horizon"] == 1)][
        ["target_time", "actual_label", "predicted_label", "predicted_probability",
         "attack_episode_id", "time_from_attack_onset_sec", "time_to_attack_offset_sec",
         "signed_distance_to_attack_window_sec"]
    ].rename(columns={"predicted_label": "label_k1", "predicted_probability": "prob_k1"})
    v4_k5 = events[(events["model"] == MODEL_V4) & (events["horizon"] == 5)][
        ["target_time", "actual_label", "predicted_label", "predicted_probability"]
    ].rename(columns={"predicted_label": "label_k5", "predicted_probability": "prob_k5"})

    paired = v4_k1.merge(v4_k5, on=["target_time", "actual_label"], how="inner")
    paired["phase"] = paired.apply(lambda row: phase_category(row, short_episodes), axis=1)

    attacks = paired[paired["actual_label"] == 1]
    pair_rows = []
    for phase, subset in attacks.groupby("phase"):
        pair_rows.append(
            {
                "analysis": "C_target_aligned_pairs",
                "model": MODEL_V4,
                "horizon": "1 vs 5 (same target window)",
                "phase": phase,
                "paired_attack_windows": int(len(subset)),
                "missed_at_k1_caught_at_k5": int(((subset["label_k1"] == 0) & (subset["label_k5"] == 1)).sum()),
                "caught_at_k1_missed_at_k5": int(((subset["label_k1"] == 1) & (subset["label_k5"] == 0)).sum()),
                "caught_at_both": int(((subset["label_k1"] == 1) & (subset["label_k5"] == 1)).sum()),
                "missed_at_both": int(((subset["label_k1"] == 0) & (subset["label_k5"] == 0)).sum()),
                "mean_prob_k1": float(subset["prob_k1"].mean()),
                "mean_prob_k5": float(subset["prob_k5"].mean()),
            }
        )
    frame_c = pd.DataFrame(pair_rows)

    return pd.concat([frame_a, frame_b, frame_c], ignore_index=True), total_delta


# =============================================================
# PART 10 — PRE-TRANSITION SIGNAL
# =============================================================

def pre_transition_signal(events, episodes):
    observable_onsets = episodes[episodes["observable_from_test_start"]]
    rows = []

    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        if model == MODEL_PERSISTENCE:
            continue
        group = group.sort_values("target_time")

        # Baselines: deep-persistent windows only
        deep_benign = group[
            (group["label_at_time_t"] == 0)
            & (group["actual_label"] == 0)
            & (group["signed_distance_to_attack_window_sec"].abs() > 600)
        ]
        deep_attack = group[
            (group["label_at_time_t"] == 1)
            & (group["actual_label"] == 1)
            & (group["time_from_attack_onset_sec"] > 600)
            & (group["time_to_attack_offset_sec"] > 600)
        ]
        benign_baseline = float(deep_benign["predicted_probability"].mean())
        attack_baseline = float(deep_attack["predicted_probability"].mean())

        for _, ep in observable_onsets.iterrows():
            onset = ep["attack_onset"]
            pre = group[
                (group["target_time"] >= onset - pd.Timedelta(seconds=WINDOW_SECONDS * PRE_TRANSITION_WINDOWS))
                & (group["target_time"] < onset)
            ]
            if pre.empty:
                continue
            clean = bool(pre["actual_label"].sum() == 0)
            rows.append(
                {
                    "model": model, "horizon": k, "transition_type": "ONSET",
                    "episode_id": int(ep["episode_id"]), "transition_time": str(onset),
                    "pre_windows": int(len(pre)),
                    "pre_window_is_clean_benign": clean,
                    "mean_pre_transition_probability": float(pre["predicted_probability"].mean()),
                    "max_pre_transition_probability": float(pre["predicted_probability"].max()),
                    "baseline_probability": benign_baseline,
                    "baseline_definition": "mean probability on persistent-benign windows >600 s from any attack",
                    "delta_mean_from_baseline": float(pre["predicted_probability"].mean()) - benign_baseline,
                    "delta_max_from_baseline": float(pre["predicted_probability"].max()) - benign_baseline,
                    "threshold": float(group["threshold"].iloc[0]),
                    "max_reaches_threshold": bool(
                        pre["predicted_probability"].max() >= group["threshold"].iloc[0]
                    ),
                    "interpretable": clean,
                    "note": "" if clean else
                            "lookback overlaps the preceding episode — elevation cannot be attributed "
                            "to pre-onset anticipation",
                }
            )

        for _, ep in episodes.iterrows():
            offset = ep["attack_offset"]
            pre = group[
                (group["target_time"] > offset - pd.Timedelta(seconds=WINDOW_SECONDS * PRE_TRANSITION_WINDOWS))
                & (group["target_time"] <= offset)
            ]
            after_exists = not group[group["target_time"] > offset].empty
            if pre.empty:
                continue
            rows.append(
                {
                    "model": model, "horizon": k, "transition_type": "OFFSET",
                    "episode_id": int(ep["episode_id"]), "transition_time": str(offset),
                    "pre_windows": int(len(pre)),
                    "pre_window_is_clean_benign": False,
                    "mean_pre_transition_probability": float(pre["predicted_probability"].mean()),
                    "max_pre_transition_probability": float(pre["predicted_probability"].max()),
                    "baseline_probability": attack_baseline,
                    "baseline_definition": "mean probability on persistent-attack windows >600 s from onset and offset",
                    "delta_mean_from_baseline": float(pre["predicted_probability"].mean()) - attack_baseline,
                    "delta_max_from_baseline": float(pre["predicted_probability"].max()) - attack_baseline,
                    "threshold": float(group["threshold"].iloc[0]),
                    "max_reaches_threshold": bool(
                        pre["predicted_probability"].max() >= group["threshold"].iloc[0]
                    ),
                    "interpretable": bool(after_exists),
                    "note": "" if after_exists else "offset at end of day — no post-offset evidence",
                }
            )

    return pd.DataFrame(rows)


# =============================================================
# PART 11 — SHUFFLE DIAGNOSTIC (exploratory null reference)
# =============================================================

def shuffle_diagnostic(events):
    rng = np.random.default_rng(SHUFFLE_SEED)
    rows = []

    for k in POINT_HORIZONS:
        persistence_group = events[
            (events["model"] == MODEL_PERSISTENCE) & (events["horizon"] == k)
        ].sort_values("sample_index")
        label_t = persistence_group["label_at_time_t"].to_numpy()
        actual = persistence_group["actual_label"].to_numpy()

        observed_accuracy = float((actual == label_t).mean())
        prevalence = float(actual.mean())
        analytic = prevalence**2 + (1 - prevalence) ** 2

        accuracies = []
        for _ in range(SHUFFLE_DRAWS):
            shuffled = rng.permutation(actual)
            accuracies.append(float((shuffled == label_t).mean()))

        rows.append(
            {
                "diagnostic": "EXPLORATORY — prevalence-preserving target shuffle. NOT a benchmark score.",
                "horizon": k,
                "seed": SHUFFLE_SEED,
                "draws": SHUFFLE_DRAWS,
                "observed_persistence_accuracy": observed_accuracy,
                "shuffled_persistence_accuracy_mean": float(np.mean(accuracies)),
                "shuffled_persistence_accuracy_std": float(np.std(accuracies)),
                "analytic_expectation_p2_plus_q2": analytic,
                "temporal_continuity_gain": observed_accuracy - float(np.mean(accuracies)),
                "note": "Shuffling destroys temporal order, so it isolates how much of persistence "
                        "accuracy comes from label continuity rather than prevalence. It is NOT run "
                        "for V4/LR: shuffling their targets only produces a prevalence-driven chance "
                        "level and says nothing about their temporal skill.",
            }
        )
    return pd.DataFrame(rows)


# =============================================================
# PART 12 — CURRENT-STATE CONDITIONING
# =============================================================

def current_state_conditioning(events):
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        for condition, value in (("A(t)=0", 0), ("A(t)=1", 1)):
            subset = group[group["label_at_time_t"] == value]
            if subset.empty:
                continue
            actual = subset["actual_label"].to_numpy()
            predicted = subset["predicted_label"].to_numpy()
            negatives = int((actual == 0).sum())
            false_positives = int(((predicted == 1) & (actual == 0)).sum())

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "condition": condition,
                    "sample_count": int(len(subset)),
                    "positive_target_rate": float(actual.mean()),
                    "mean_probability": float(subset["predicted_probability"].mean()),
                    "median_probability": float(subset["predicted_probability"].median()),
                    "f1": safe_f1(actual, predicted),
                    "f1_note": UNAVAILABLE if len(np.unique(actual)) < 2 else "",
                    "recall": float(recall_score(actual, predicted, zero_division=0)) if actual.sum() else None,
                    "precision": float(precision_score(actual, predicted, zero_division=0))
                    if predicted.sum() else None,
                    "false_positive_rate": false_positives / negatives if negatives else None,
                    "negatives_in_condition": negatives,
                    "fpr_note": "denominator is only %d windows" % negatives if 0 < negatives < 30 else "",
                    "structurally_determined": model == MODEL_PERSISTENCE,
                }
            )
    return pd.DataFrame(rows)


# =============================================================
# PART 13 — SCORECARD
# =============================================================

def scorecard(stats, excess, transition_only, onsets, decomposition_b, events):
    def value(frame, column, k, **filters):
        subset = frame[frame["horizon"] == k]
        for key, item in filters.items():
            subset = subset[subset[key] == item]
        return None if subset.empty else subset[column].iloc[0]

    metrics = []

    def add(name, function, note=""):
        metrics.append({"metric": name, **{f"K{k}": function(k) for k in POINT_HORIZONS}, "note": note})

    add("persistence_accuracy", lambda k: round(value(stats, "persistence_accuracy", k), 6))
    add("transition_fraction", lambda k: round(value(stats, "transition_fraction", k), 6))
    add("transition_count", lambda k: int(value(stats, "transition_count", k)))
    add("V4_F1", lambda k: round(value(excess, "overall_f1", k, model=MODEL_V4), 4))
    add("LR_F1", lambda k: round(value(excess, "overall_f1", k, model=MODEL_LR), 4))
    add("persistence_F1", lambda k: round(value(excess, "overall_f1", k, model=MODEL_PERSISTENCE), 4))
    add("V4_minus_persistence_F1",
        lambda k: round(value(excess, "overall_f1", k, model=MODEL_V4)
                        - value(excess, "overall_f1", k, model=MODEL_PERSISTENCE), 4))
    add("V4_transition_recall",
        lambda k: round(value(transition_only, "onset_recall", k, model=MODEL_V4), 4),
        "recall on 0->1 transitions only")
    add("LR_transition_recall",
        lambda k: round(value(transition_only, "onset_recall", k, model=MODEL_LR), 4),
        "recall on 0->1 transitions only")
    add("V4_onset_count", lambda k: int(value(transition_only, "onset_n", k, model=MODEL_V4)),
        "number of 0->1 transition windows at this horizon")
    add("V4_onset_detected", lambda k: int(value(transition_only, "onset_detected", k, model=MODEL_V4)))
    add("V4_expected_onset_hits_at_base_rate",
        lambda k: value(transition_only, "expected_onset_hits_at_base_rate", k, model=MODEL_V4),
        "what V4's own positive rate would produce by chance")

    def median_lead(k):
        subset = onsets[
            (onsets["model"] == MODEL_V4) & (onsets["horizon"] == k)
            & (onsets["lead_time_seconds"].notna())
        ]
        return None if subset.empty else float(subset["lead_time_seconds"].median())

    add("V4_median_observed_lead_time_sec", median_lead,
        "only 3 observable onsets, 2 of them preceded by an active episode — descriptive only")

    def clean_onset_warnings(k):
        subset = onsets[
            (onsets["model"] == MODEL_V4) & (onsets["horizon"] == k)
            & (onsets["pre_onset_lookback_is_clean"])
        ]
        return int(subset["warning_detected"].sum())

    add("V4_warnings_before_clean_onsets", clean_onset_warnings,
        "onsets whose 120 s lookback contains no attack window (n=1: ep3)")

    # Window-count decomposition instead of an F1 decomposition (see note)
    def correct_share(model, k):
        group = events[(events["model"] == model) & (events["horizon"] == k)]
        persistent = group[group["actual_label"] == group["label_at_time_t"]]
        correct = (group["predicted_label"] == group["actual_label"]).sum()
        correct_persistent = (persistent["predicted_label"] == persistent["actual_label"]).sum()
        return round(float(correct_persistent / correct), 6) if correct else None

    add("fraction_of_V4_correct_windows_that_are_persistent",
        lambda k: correct_share(MODEL_V4, k),
        "window-count decomposition — F1 is not additively decomposable, so no F1 share is reported")
    add("fraction_of_V4_TP_in_AA_windows",
        lambda k: round(
            float(
                len(events[(events["model"] == MODEL_V4) & (events["horizon"] == k)
                           & (events["error_type"] == "TP") & (events["label_at_time_t"] == 1)])
                / max(len(events[(events["model"] == MODEL_V4) & (events["horizon"] == k)
                                 & (events["error_type"] == "TP")]), 1)
            ), 6))
    add("fraction_of_V4_FN_reduction_K1_to_K5_in_mid_attack",
        lambda k: round(
            float(decomposition_b[decomposition_b["phase"] == "mid"]["fraction_of_total_reduction"].iloc[0]), 6
        ) if k == 5 else None,
        "reported at K=5 only; see v4_k1_to_k5_correction_decomposition.csv")

    return pd.DataFrame(metrics)


# =============================================================
# FIGURES
# =============================================================

def make_figures(stats, decomposition, performance, events, episodes, correction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    horizons = stats["horizon"].tolist()

    # 1. persistence vs horizon
    figure, axes = plt.subplots(figsize=(7.2, 4.3))
    axes.plot(horizons, stats["persistence_accuracy"], marker="o", color="#8c8c8c",
              linewidth=2, label="P(A(t+k) = A(t))")
    axes.plot(horizons, stats["attack_persistence_rate"], marker="^", color="#d62728",
              linewidth=1.6, label="P(A(t+k)=1 | A(t)=1)")
    axes.plot(horizons, stats["benign_persistence_rate"], marker="s", color="#1f77b4",
              linewidth=1.6, label="P(A(t+k)=0 | A(t)=0)")
    axes.set_xticks(horizons)
    axes.set_ylim(0.985, 1.001)
    axes.set_xlabel("Forecast horizon K (10-second windows)")
    axes.set_ylabel("Probability")
    axes.set_title("Label persistence vs forecast horizon\nMar-2-2018 held-out day, 3,115 windows", fontsize=11)
    axes.grid(alpha=0.3, linestyle=":")
    axes.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "persistence_vs_horizon.png", dpi=150)
    plt.close(figure)

    # 2. transition fraction vs horizon
    figure, axes = plt.subplots(figsize=(7.2, 4.3))
    axes.bar([str(h) for h in horizons], stats["transition_count"], color="#7b4173", width=0.55)
    for x, (count, fraction) in enumerate(zip(stats["transition_count"], stats["transition_fraction"])):
        axes.text(x, count + 0.25, f"{int(count)}\n({fraction * 100:.2f}%)", ha="center", fontsize=9)
    axes.set_xlabel("Forecast horizon K")
    axes.set_ylabel("Transition windows (A(t+k) != A(t))")
    axes.set_ylim(0, max(stats["transition_count"]) * 1.45)
    axes.set_title("How few windows actually change state\n"
                   "Every other window is correct for persistence by construction", fontsize=11)
    axes.grid(alpha=0.3, linestyle=":", axis="y")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "transition_fraction_vs_horizon.png", dpi=150)
    plt.close(figure)

    # 3. model performance by transition type (K=5)
    figure, axes = plt.subplots(figsize=(9.0, 4.6))
    categories = ["PP", "AA", "ONSET", "OFFSET"]
    colours = {MODEL_PERSISTENCE: "#8c8c8c", MODEL_LR: "#d1651a", MODEL_V4: "#1f77b4"}
    width = 0.26
    positions = np.arange(len(categories))

    for index, (model, colour) in enumerate(colours.items()):
        values, counts = [], []
        for category in categories:
            row = performance[
                (performance["model"] == model) & (performance["horizon"] == 5)
                & (performance["transition_category"] == category)
            ]
            values.append(float(row["accuracy"].iloc[0]) if not row.empty else 0.0)
            counts.append(int(row["sample_count"].iloc[0]) if not row.empty else 0)
        axes.bar(positions + (index - 1) * width, values, width, color=colour, label=model)
        for x, value in zip(positions + (index - 1) * width, values):
            axes.text(x, value + 0.02, f"{value:.2f}", ha="center", fontsize=7.5)

    axes.set_xticks(positions)
    axes.set_xticklabels([f"{category}\nn={count}" for category, count in zip(categories, counts)])
    axes.set_ylim(0, 1.15)
    axes.set_ylabel("Accuracy within category")
    axes.set_title("K=5 accuracy by transition category (frozen thresholds)\n"
                   "PP/AA are persistent windows; ONSET/OFFSET are the only state changes", fontsize=11)
    axes.grid(alpha=0.3, linestyle=":", axis="y")
    axes.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "model_performance_by_transition_type.png", dpi=150)
    plt.close(figure)

    # 4 & 5. V4 probability around onsets / offsets
    for transition_type, filename in (("onset", "v4_probability_around_onsets.png"),
                                      ("offset", "v4_probability_around_offsets.png")):
        if transition_type == "onset":
            markers = episodes[episodes["observable_from_test_start"]][["episode_id", "attack_onset"]]
            markers = markers.rename(columns={"attack_onset": "time"})
        else:
            markers = episodes[["episode_id", "attack_offset"]].rename(columns={"attack_offset": "time"})

        figure, axes_list = plt.subplots(1, len(markers), figsize=(4.3 * len(markers), 3.9), squeeze=False)
        for axes, (_, marker) in zip(axes_list[0], markers.iterrows()):
            centre = pd.Timestamp(marker["time"])
            group = events[(events["model"] == MODEL_V4) & (events["horizon"] == 5)]
            window = group[
                (group["target_time"] >= centre - pd.Timedelta(seconds=120))
                & (group["target_time"] <= centre + pd.Timedelta(seconds=120))
            ].sort_values("target_time")
            if window.empty:
                axes.text(0.5, 0.5, "no evaluated windows", ha="center", va="center", transform=axes.transAxes)
                axes.set_title(f"ep{int(marker['episode_id'])} {transition_type}", fontsize=9)
                continue
            relative = (window["target_time"] - centre).dt.total_seconds()
            axes.plot(relative, window["predicted_probability"], marker="o", markersize=3,
                      color="#1f77b4", linewidth=1.2)
            axes.scatter(relative[window["actual_label"] == 1],
                         window.loc[window["actual_label"] == 1, "predicted_probability"],
                         color="#d62728", s=14, zorder=3, label="attack window")
            axes.axvline(0, color="#d62728", linestyle="--", linewidth=1.3)
            axes.axhline(0.72, color="black", linestyle=":", linewidth=1.1)
            axes.set_ylim(-0.05, 1.05)
            axes.set_xlabel(f"seconds relative to {transition_type}")
            axes.set_ylabel("V4 P(attack at t+5)")
            axes.set_title(f"ep{int(marker['episode_id'])} — {centre:%H:%M:%S}", fontsize=9.5)
            axes.grid(alpha=0.3, linestyle=":")
            axes.legend(fontsize=7, loc="lower right")
        figure.suptitle(
            f"V4 K=5 probability around attack {transition_type}s "
            f"(dotted line = frozen threshold 0.72)", fontsize=11,
        )
        figure.tight_layout()
        figure.savefig(FIG_DIR / filename, dpi=150)
        plt.close(figure)

    # 6. K=1 -> K=5 correction decomposition
    delta = correction[correction["analysis"] == "B_k1_minus_k5_fn_delta"].copy()
    delta = delta.sort_values("fn_reduction", ascending=False)
    figure, axes = plt.subplots(figsize=(8.4, 4.4))
    axes.bar(delta["phase"], delta["fn_reduction"], color="#1f77b4", width=0.55)
    for x, (reduction, fraction) in enumerate(zip(delta["fn_reduction"], delta["fraction_of_total_reduction"])):
        axes.text(x, reduction + 4, f"{int(reduction)}\n({fraction * 100:.1f}%)" if fraction is not None
                  else str(int(reduction)), ha="center", fontsize=8.5)
    axes.set_ylabel("Missed attack windows recovered (FN at K=1 − FN at K=5)")
    axes.set_xlabel("Episode phase of the attack window")
    axes.set_title("Where V4's K=1 → K=5 false-negative reduction occurs\n"
                   "Phases defined before counting; see methodology", fontsize=11)
    axes.grid(alpha=0.3, linestyle=":", axis="y")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "v4_k1_vs_k5_correction_decomposition.png", dpi=150)
    plt.close(figure)


# =============================================================
# MAIN
# =============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Persistence hypothesis test (analysis only).")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print("PERSISTENCE HYPOTHESIS TEST — Mar-2-2018 held-out day")
    print("=" * 100)
    print("Analysis only: no training, no threshold change, no artifact modification.")

    preflight_results = {}
    if not args.skip_preflight:
        print("\n--- Preflight: re-verifying the frozen chain ---")
        preflight_results = preflight()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(EVENTS_CSV, parse_dates=["forecast_time", "target_time"])
    events = events[events["model"].isin(MODELS)]
    episodes = pd.read_csv(EPISODES_CSV, parse_dates=["attack_onset", "attack_offset"])
    n_test = int(events["sample_index"].nunique())
    print(f"\nEvaluated windows: {n_test}   models: {MODELS}   horizons: {POINT_HORIZONS}")

    # ----- Part 1 -----
    stats = persistence_statistics(events)
    stats.to_csv(OUT_DIR / "persistence_statistics.csv", index=False)
    print("\n" + "=" * 100)
    print("PART 1 — LABEL PERSISTENCE")
    print("=" * 100)
    print(stats[["horizon", "persistence_accuracy", "attack_persistence_rate",
                 "benign_persistence_rate", "onset_rate", "offset_rate",
                 "n_current_benign_future_attack", "n_current_attack_future_benign",
                 "transition_count"]].round(6).to_string(index=False))

    # ----- Parts 2 & 3 -----
    decomposition = transition_decomposition(events, n_test)
    decomposition.to_csv(OUT_DIR / "transition_decomposition.csv", index=False)
    performance = transition_model_performance(events)
    performance.to_csv(OUT_DIR / "transition_model_performance.csv", index=False)

    print("\n" + "=" * 100)
    print("PART 2/3 — TRANSITION DECOMPOSITION (K=5)")
    print("=" * 100)
    print(decomposition[decomposition["horizon"] == 5][
        ["model", "transition_category", "count", "percentage_of_test", "correct",
         "incorrect", "prediction_rate", "mean_probability"]
    ].round(4).to_string(index=False))

    # ----- Parts 4 & 5 -----
    excess = persistence_excess(events)
    excess.to_csv(OUT_DIR / "persistence_excess.csv", index=False)
    transition_only = transition_only_metrics(events)
    transition_only.to_csv(OUT_DIR / "transition_only_metrics.csv", index=False)

    print("\n" + "=" * 100)
    print("PART 4/5 — PERSISTENT vs TRANSITION SUBSETS")
    print("=" * 100)
    print(excess[["model", "horizon", "overall_f1", "persistent_subset_f1", "persistent_subset_accuracy",
                  "transition_subset_n", "transition_subset_accuracy"]].round(4).to_string(index=False))
    print()
    print(transition_only[["model", "horizon", "onset_n", "onset_detected", "onset_recall",
                           "expected_onset_hits_at_base_rate", "offset_n", "offset_correct",
                           "expected_offset_correct_at_base_rate"]].round(4).to_string(index=False))

    # ----- Parts 6 & 7 -----
    onsets = onset_transition_analysis(events, episodes)
    onsets.to_csv(OUT_DIR / "onset_transition_analysis.csv", index=False)
    offsets = offset_transition_analysis(events, episodes)
    offsets.to_csv(OUT_DIR / "offset_transition_analysis.csv", index=False)

    print("\n" + "=" * 100)
    print("PART 6 — ONSET ANALYSIS (K=5)")
    print("=" * 100)
    print(onsets[onsets["horizon"] == 5][
        ["model", "episode_id", "attack_onset", "warning_status", "lead_time_seconds",
         "probability_at_onset", "max_pre_onset_probability", "pre_onset_windows_that_are_attacks",
         "pre_onset_lookback_is_clean"]
    ].round(4).to_string(index=False))

    # ----- Part 9 -----
    correction, total_delta = correction_decomposition(events, episodes)
    correction.to_csv(OUT_DIR / "v4_k1_to_k5_correction_decomposition.csv", index=False)
    print("\n" + "=" * 100)
    print(f"PART 9 — V4 K=1 -> K=5 FALSE-NEGATIVE REDUCTION ({total_delta} windows)")
    print("=" * 100)
    print(correction[correction["analysis"] == "B_k1_minus_k5_fn_delta"][
        ["phase", "fn_k1", "fn_k5", "fn_reduction", "fraction_of_total_reduction"]
    ].round(4).to_string(index=False))
    print()
    print(correction[correction["analysis"] == "C_target_aligned_pairs"][
        ["phase", "paired_attack_windows", "missed_at_k1_caught_at_k5",
         "caught_at_k1_missed_at_k5", "caught_at_both", "missed_at_both"]
    ].to_string(index=False))

    # ----- Part 10 -----
    signal = pre_transition_signal(events, episodes)
    signal.to_csv(OUT_DIR / "pre_transition_signal.csv", index=False)
    print("\n" + "=" * 100)
    print("PART 10 — PRE-TRANSITION SIGNAL (K=5, ONSET)")
    print("=" * 100)
    print(signal[(signal["horizon"] == 5) & (signal["transition_type"] == "ONSET")][
        ["model", "episode_id", "pre_window_is_clean_benign", "mean_pre_transition_probability",
         "max_pre_transition_probability", "baseline_probability", "delta_max_from_baseline",
         "max_reaches_threshold"]
    ].round(4).to_string(index=False))

    # ----- Part 11 -----
    shuffle = shuffle_diagnostic(events)
    shuffle.to_csv(OUT_DIR / "shuffle_diagnostic.csv", index=False)
    print("\n" + "=" * 100)
    print("PART 11 — SHUFFLE DIAGNOSTIC (exploratory null reference)")
    print("=" * 100)
    print(shuffle[["horizon", "observed_persistence_accuracy", "shuffled_persistence_accuracy_mean",
                   "analytic_expectation_p2_plus_q2", "temporal_continuity_gain"]].round(6).to_string(index=False))

    # ----- Part 12 -----
    conditioning = current_state_conditioning(events)
    conditioning.to_csv(OUT_DIR / "current_state_conditioning.csv", index=False)
    print("\n" + "=" * 100)
    print("PART 12 — CURRENT-STATE CONDITIONING (K=5)")
    print("=" * 100)
    print(conditioning[conditioning["horizon"] == 5][
        ["model", "condition", "sample_count", "positive_target_rate", "mean_probability",
         "recall", "false_positive_rate"]
    ].round(4).to_string(index=False))

    # ----- Part 13 -----
    delta_frame = correction[correction["analysis"] == "B_k1_minus_k5_fn_delta"]
    card = scorecard(stats, excess, transition_only, onsets, delta_frame, events)
    card.to_csv(OUT_DIR / "persistence_scorecard.csv", index=False)
    print("\n" + "=" * 100)
    print("PART 13 — SCORECARD")
    print("=" * 100)
    print(card.to_string(index=False))

    # ----- Part 14 -----
    if not args.skip_plots:
        make_figures(stats, decomposition, performance, events, episodes, correction)
        print(f"\nWrote figures to {FIG_DIR.relative_to(REPO_ROOT)}")

    # ----- Task 3 / Task 4 consistency -----
    task3 = pd.read_csv(TASK3_CSV)
    task3_point = task3[task3["evaluation_type"] == "point_forecast"]
    mismatches = []
    for _, row in excess.iterrows():
        reference = task3_point[
            (task3_point["model"] == row["model"]) & (task3_point["horizon"] == row["horizon"])
        ]
        if reference.empty:
            mismatches.append(f"{row['model']} K={row['horizon']} missing in Task 3")
            continue
        if abs(float(row["overall_f1"]) - float(reference.iloc[0]["f1"])) > TOLERANCE:
            mismatches.append(f"{row['model']} K={row['horizon']} F1 mismatch")
    if mismatches:
        raise RuntimeError("Task 3 consistency failed: " + "; ".join(mismatches))
    print("\nTask 3 consistency: all 12 F1 values reproduce exactly")

    # ----- Summary JSON -----
    v4_manifest = json.loads(V4_MANIFEST.read_text())
    summary = {
        "analysis": "persistence_hypothesis_test",
        "test_day": "2018-03-02 (Friday-02-03-2018)",
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluated_windows": n_test,
        "horizons": POINT_HORIZONS,
        "persistence_statistics": json.loads(stats.to_json(orient="records")),
        "transition_decomposition": json.loads(decomposition.to_json(orient="records")),
        "transition_only_metrics": json.loads(transition_only.to_json(orient="records")),
        "persistence_excess": json.loads(excess.to_json(orient="records")),
        "onset_analysis_k5": json.loads(
            onsets[onsets["horizon"] == 5].to_json(orient="records")
        ),
        "k1_to_k5_fn_reduction": {
            "total_windows_recovered": total_delta,
            "by_phase": json.loads(delta_frame.to_json(orient="records")),
        },
        "shuffle_diagnostic": json.loads(shuffle.to_json(orient="records")),
        "scorecard": json.loads(card.to_json(orient="records")),
        "verification": {
            "preflight": preflight_results,
            "task3_f1_reproduced": True,
            "test_substrate_windows": n_test,
            "no_model_retrained": True,
            "no_threshold_changed": True,
            "no_artifact_modified": True,
        },
        "provenance": {
            "repository_commit": git_value("rev-parse", "HEAD"),
            "v4_reference_commit": v4_manifest["evaluation"]["reference_script_commit"],
            "v4_checkpoint_sha256": v4_manifest["checkpoint"]["sha256"],
            "error_events_sha256": sha256_file(EVENTS_CSV),
            "task3_csv_sha256": sha256_file(TASK3_CSV),
        },
        "interpretation_caveats": [
            "Persistence receives the observed label A(t); V4 and LR receive only features. "
            "Its accuracy is not achievable by a traffic model for free.",
            "Only 6-14 transition windows exist per horizon; transition-conditioned numbers are "
            "descriptive and cannot support significance testing.",
            "Windows inside one attack episode are temporally dependent — thousands of attack "
            "windows are not thousands of independent episodes.",
            "Two of the three observable onsets are preceded by an active episode, so only one "
            "cold-start onset exists on this day.",
            "The V4 training/evaluation normalisation provenance caveat from Task 1 is unresolved.",
        ],
    }
    with open(SUMMARY_JSON, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {SUMMARY_JSON.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 100)
    print("PERSISTENCE HYPOTHESIS ANALYSIS GENERATED")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
