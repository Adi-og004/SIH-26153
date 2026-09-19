"""
=============================================================
FALSE-POSITIVE / FALSE-NEGATIVE ERROR ANALYSIS
Persistence  vs  Logistic Regression  vs  V4 World Model
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Analysis only. Nothing is trained, retrained, re-thresholded or modified.
Every threshold is the frozen validation-selected value from Tasks 1-2, and
every aggregate metric is cross-checked against Task 3's standardized
comparison before any output is written.

Inputs (all frozen, read-only):
    docs/v4_baseline_manifest.json                     thresholds + checkpoint
    docs/logistic_regression_baseline_manifest.json    thresholds
    models/logistic_regression_baseline/test_predictions.csv   LR probabilities
    data/processed/multiday_benchmark_K5/*             benchmark arrays
    data/processed/states_v2/Friday-02-03-2018_*.csv   raw labels + raw features
    results/standardized_baseline_comparison.csv       Task 3 cross-check

Outputs: results/error_analysis/*, results/error_analysis_summary.json

Usage:
    python src/run_error_analysis.py
    python src/run_error_analysis.py --skip-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_standardized_comparison import (  # noqa: E402  (frozen inference path, reused verbatim)
    BENCHMARK_DIR,
    MODEL_LR,
    MODEL_PERSISTENCE,
    MODEL_V4,
    POINT_HORIZONS,
    STATES_DIR,
    fetch_labels_at_time_t,
    git_value,
    load_benchmark,
    sha256_file,
    standard_metrics,
    v4_scores,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

V4_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
LR_MANIFEST = REPO_ROOT / "docs" / "logistic_regression_baseline_manifest.json"
LR_PREDICTIONS = REPO_ROOT / "models" / "logistic_regression_baseline" / "test_predictions.csv"
TASK3_CSV = REPO_ROOT / "results" / "standardized_baseline_comparison.csv"

OUT_DIR = REPO_ROOT / "results" / "error_analysis"
FIG_DIR = OUT_DIR / "figures"
SUMMARY_JSON = REPO_ROOT / "results" / "error_analysis_summary.json"

TEST_DAY = "Friday-02-03-2018"
TEST_DAY_CSV = STATES_DIR / f"{TEST_DAY}_TrafficForML_CICFlowMeter_states_v2.csv"

WINDOW_SECONDS = 10
CONTEXT_BUCKETS = [10, 30, 60, 120]  # seconds; fixed before looking at any result
ONSET_LOOKBACK_WINDOWS = 12          # 120 s — covers the largest context bucket

BASE_FEATURES = [
    "flow_count", "log_flow_count", "log_total_packets", "log_total_bytes",
    "window_packet_rate", "window_byte_rate", "syn_rate", "ack_rate",
    "rst_rate", "fin_rate", "psh_rate", "urg_rate", "mean_pkt_len",
    "std_pkt_len", "log_mean_flow_duration", "log_mean_fwd_iat",
    "log_mean_bwd_iat", "fwd_packet_ratio", "bwd_packet_ratio",
    "fwd_byte_ratio", "bwd_byte_ratio", "mean_down_up_ratio",
    "log_unique_dst_ports", "unique_protocols", "dst_port_entropy",
    "protocol_entropy",
]

FEATURE_ANALYSIS_SUBSET = [
    "flow_count", "window_packet_rate", "window_byte_rate", "syn_rate",
    "ack_rate", "rst_rate", "fin_rate", "mean_pkt_len", "std_pkt_len",
    "log_mean_fwd_iat", "log_mean_bwd_iat", "log_mean_flow_duration",
    "log_unique_dst_ports", "dst_port_entropy", "unique_protocols",
    "protocol_entropy",
]

TOLERANCE = 1e-9


# =============================================================
# EPISODES
# =============================================================

def build_episodes(raw):
    """Contiguous runs of attack_present == 1 on the raw test day."""
    labels = raw["attack_present"].astype(int).to_numpy()
    times = raw["window_start"].to_numpy()

    episodes = []
    index = 0
    while index < len(labels):
        if labels[index] == 1:
            end = index
            while end + 1 < len(labels) and labels[end + 1] == 1:
                end += 1

            window_count = end - index + 1
            onset = pd.Timestamp(times[index])
            offset = pd.Timestamp(times[end])
            types = raw.loc[index:end, "dominant_attack"].value_counts()

            episodes.append(
                {
                    "episode_id": len(episodes) + 1,
                    "attack_onset": onset,
                    "attack_offset": offset,
                    "duration_sec": window_count * WINDOW_SECONDS,
                    "duration_windows": window_count,
                    "number_of_attack_windows": window_count,
                    "attack_type_if_available": "; ".join(
                        f"{name}({count})" for name, count in types.items()
                    ),
                    "observable_from_test_start": index > 0,
                    "time_contiguous": (offset - onset).total_seconds()
                    == (window_count - 1) * WINDOW_SECONDS,
                    "notes": (
                        "already active at the first raw test window — no observable onset"
                        if index == 0
                        else "onset observable inside the test interval"
                    ),
                }
            )
            index = end + 1
        else:
            index += 1

    return episodes


def episode_context(target_times, episodes, attack_times):
    """Signed episode-relative timing for every evaluated target window."""
    onsets = np.array([ep["attack_onset"].value for ep in episodes], dtype=np.int64)
    offsets = np.array([ep["attack_offset"].value for ep in episodes], dtype=np.int64)
    attack_values = np.sort(np.array([t.value for t in attack_times], dtype=np.int64))

    target_values = target_times.astype("datetime64[ns]").astype(np.int64)

    nearest_onset_index = np.abs(target_values[:, None] - onsets[None, :]).argmin(axis=1)
    nearest_offset_index = np.abs(target_values[:, None] - offsets[None, :]).argmin(axis=1)

    # signed seconds: negative = target is BEFORE the onset
    time_from_onset = (target_values - onsets[nearest_onset_index]) / 1e9
    # signed seconds: positive = target is BEFORE the offset
    time_to_offset = (offsets[nearest_offset_index] - target_values) / 1e9

    # signed distance to the nearest attack window (0 when the target is an attack)
    insertion = np.searchsorted(attack_values, target_values)
    distance = np.full(len(target_values), np.nan)
    for position, (value, slot) in enumerate(zip(target_values, insertion)):
        candidates = []
        if slot < len(attack_values):
            candidates.append((attack_values[slot] - value) / 1e9 * -1.0)  # attack in future -> negative
        if slot > 0:
            candidates.append((value - attack_values[slot - 1]) / 1e9)     # attack in past -> positive
        distance[position] = min(candidates, key=abs)

    # episode id of the target window itself (-1 when benign)
    episode_of_target = np.full(len(target_values), -1, dtype=int)
    for ep in episodes:
        inside = (target_values >= ep["attack_onset"].value) & (
            target_values <= ep["attack_offset"].value
        )
        episode_of_target[inside] = ep["episode_id"]

    # For a window INSIDE an episode the timing must be measured against that
    # episode, not against whichever onset happens to be nearest in absolute
    # time (deep inside a long episode the nearest onset can be the NEXT one).
    for position, ep in enumerate(episodes):
        inside = episode_of_target == ep["episode_id"]
        if not inside.any():
            continue
        time_from_onset[inside] = (
            target_values[inside] - ep["attack_onset"].value
        ) / 1e9
        time_to_offset[inside] = (
            ep["attack_offset"].value - target_values[inside]
        ) / 1e9
        nearest_onset_index[inside] = position

    return {
        "episode_id": episode_of_target,
        "nearest_episode_id": np.array([episodes[i]["episode_id"] for i in nearest_onset_index]),
        "time_from_attack_onset_sec": time_from_onset,
        "time_to_attack_offset_sec": time_to_offset,
        "signed_distance_to_attack_window_sec": distance,
    }


# =============================================================
# ERROR EVENT TABLE
# =============================================================

def classify(actual, predicted):
    out = np.empty(len(actual), dtype=object)
    out[(predicted == 1) & (actual == 1)] = "TP"
    out[(predicted == 0) & (actual == 0)] = "TN"
    out[(predicted == 1) & (actual == 0)] = "FP"
    out[(predicted == 0) & (actual == 1)] = "FN"
    return out


def runs_of(mask):
    """Lengths of contiguous True runs (by benchmark sample index)."""
    lengths = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def build_error_events(data, scores, thresholds, episodes, raw):
    meta = data["test"]["meta"]
    A_test = data["test"]["A"]

    target_t1 = meta["timestamp"].to_numpy()
    forecast_time = meta["timestamp"] - pd.Timedelta(seconds=WINDOW_SECONDS)

    raw_indexed = raw.set_index("window_start")
    features_at_t = raw_indexed.loc[forecast_time, BASE_FEATURES].reset_index(drop=True)

    attack_times = raw.loc[raw["attack_present"].astype(int) == 1, "window_start"].tolist()

    frames = []
    for k in POINT_HORIZONS:
        target_time = meta["timestamp"] + pd.Timedelta(seconds=WINDOW_SECONDS * (k - 1))
        context = episode_context(target_time.to_numpy(), episodes, attack_times)
        actual = A_test[:, k - 1].astype(int)

        for model in (MODEL_PERSISTENCE, MODEL_LR, MODEL_V4):
            if model == MODEL_PERSISTENCE:
                probability = scores["persistence"]
                threshold = np.nan
                predicted = probability.astype(int)
            elif model == MODEL_LR:
                probability = scores["lr"][k]
                threshold = thresholds["lr"][k]
                predicted = (probability >= threshold).astype(int)
            else:
                probability = scores["v4"][:, k]
                threshold = thresholds["v4"][k]
                predicted = (probability >= threshold).astype(int)

            frame = pd.DataFrame(
                {
                    "model": model,
                    "horizon": k,
                    "sample_index": np.arange(len(meta)),
                    "forecast_time": forecast_time.to_numpy(),
                    "target_time": target_time.to_numpy(),
                    "actual_label": actual,
                    "predicted_probability": probability,  # full precision: V4 scores are so tightly
                    # packed that 6-decimal rounding creates ~800 ties and shifts ROC-AUC by ~3e-4
                    "threshold": threshold,
                    "predicted_label": predicted,
                    "error_type": classify(actual, predicted),
                    "source_day": TEST_DAY,
                    "attack_episode_id": context["episode_id"],
                    "nearest_episode_id": context["nearest_episode_id"],
                    "time_from_attack_onset_sec": context["time_from_attack_onset_sec"],
                    "time_to_attack_offset_sec": context["time_to_attack_offset_sec"],
                    "signed_distance_to_attack_window_sec": context[
                        "signed_distance_to_attack_window_sec"
                    ],
                    "label_at_time_t": scores["persistence"].astype(int),
                    "is_target_transition": (actual != scores["persistence"].astype(int)).astype(int),
                }
            )
            frames.append(pd.concat([frame, features_at_t.round(6)], axis=1))

    # Secondary: LR K=4 only (no V4 reference exists at K=4)
    k = 4
    target_time = meta["timestamp"] + pd.Timedelta(seconds=WINDOW_SECONDS * (k - 1))
    context = episode_context(target_time.to_numpy(), episodes, attack_times)
    actual = A_test[:, k - 1].astype(int)
    probability = scores["lr"][k]
    threshold = thresholds["lr"][k]
    predicted = (probability >= threshold).astype(int)
    frame = pd.DataFrame(
        {
            "model": "logistic_regression_secondary_k4",
            "horizon": k,
            "sample_index": np.arange(len(meta)),
            "forecast_time": forecast_time.to_numpy(),
            "target_time": target_time.to_numpy(),
            "actual_label": actual,
            "predicted_probability": probability,  # full precision: V4 scores are so tightly
                    # packed that 6-decimal rounding creates ~800 ties and shifts ROC-AUC by ~3e-4
            "threshold": threshold,
            "predicted_label": predicted,
            "error_type": classify(actual, predicted),
            "source_day": TEST_DAY,
            "attack_episode_id": context["episode_id"],
            "nearest_episode_id": context["nearest_episode_id"],
            "time_from_attack_onset_sec": context["time_from_attack_onset_sec"],
            "time_to_attack_offset_sec": context["time_to_attack_offset_sec"],
            "signed_distance_to_attack_window_sec": context[
                "signed_distance_to_attack_window_sec"
            ],
            "label_at_time_t": scores["persistence"].astype(int),
            "is_target_transition": (actual != scores["persistence"].astype(int)).astype(int),
        }
    )
    frames.append(pd.concat([frame, features_at_t.round(6)], axis=1))

    return pd.concat(frames, ignore_index=True)


# =============================================================
# SUMMARIES
# =============================================================

def false_positive_summary(events, hours):
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("sample_index")
        is_fp = (group["error_type"] == "FP").to_numpy()
        benign = int((group["actual_label"] == 0).sum())
        fp_count = int(is_fp.sum())
        lengths = runs_of(is_fp)
        distance = group.loc[is_fp, "signed_distance_to_attack_window_sec"].to_numpy()

        rows.append(
            {
                "model": model,
                "horizon": k,
                "fp_count": fp_count,
                "benign_target_windows": benign,
                "fp_rate": fp_count / benign if benign else np.nan,
                "fp_percent_of_benign_windows": 100.0 * fp_count / benign if benign else np.nan,
                "fp_windows_per_hour": fp_count / hours,
                "fp_runs": len(lengths),
                "median_run_length": float(np.median(lengths)) if lengths else 0.0,
                "max_run_length": int(max(lengths)) if lengths else 0,
                "fp_before_attack_count": int((distance < 0).sum()),
                "fp_after_attack_count": int((distance > 0).sum()),
                "isolated_fp_count": int(sum(1 for length in lengths if length == 1)),
                **{
                    f"fp_before_attack_within_{bucket}s": int(
                        ((distance < 0) & (distance >= -bucket)).sum()
                    )
                    for bucket in CONTEXT_BUCKETS
                },
                **{
                    f"fp_after_attack_within_{bucket}s": int(
                        ((distance > 0) & (distance <= bucket)).sum()
                    )
                    for bucket in CONTEXT_BUCKETS
                },
                "fp_far_from_attack_gt120s": int((np.abs(distance) > 120).sum()),
            }
        )
    return pd.DataFrame(rows)


def false_negative_summary(events, episodes):
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        attack_rows = group[group["actual_label"] == 1]
        for ep in episodes:
            episode_rows = attack_rows[attack_rows["attack_episode_id"] == ep["episode_id"]]
            if episode_rows.empty:
                continue
            missed = episode_rows[episode_rows["error_type"] == "FN"]
            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "episode_id": ep["episode_id"],
                    "episode_attack_windows": int(len(episode_rows)),
                    "fn_count": int(len(missed)),
                    "fn_rate": len(missed) / len(episode_rows),
                    "first_missed_time": str(missed["target_time"].min()) if len(missed) else "",
                    "last_missed_time": str(missed["target_time"].max()) if len(missed) else "",
                    "missed_windows": int(len(missed)),
                    "median_time_from_onset_of_missed_sec": float(
                        missed["time_from_attack_onset_sec"].median()
                    )
                    if len(missed)
                    else np.nan,
                    "fn_in_first_60s_of_episode": int(
                        (missed["time_from_attack_onset_sec"] <= 60).sum()
                    ),
                    "median_probability_of_missed": float(
                        missed["predicted_probability"].median()
                    )
                    if len(missed)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def onset_analysis(events, episodes):
    rows = []
    observable = [ep for ep in episodes if ep["observable_from_test_start"]]

    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("target_time")
        for ep in observable:
            onset = ep["attack_onset"]
            lookback_start = onset - pd.Timedelta(
                seconds=WINDOW_SECONDS * ONSET_LOOKBACK_WINDOWS
            )
            before = group[
                (group["target_time"] >= lookback_start) & (group["target_time"] < onset)
            ]

            alerts = before[before["predicted_label"] == 1]
            first_alert = alerts["target_time"].min() if len(alerts) else None

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "episode_id": ep["episode_id"],
                    "attack_onset": str(onset),
                    "lookback_windows_available": int(len(before)),
                    "warning_detected": bool(len(alerts) > 0),
                    "first_warning_time": str(first_alert) if first_alert is not None else "",
                    "lead_time_sec": float((onset - first_alert).total_seconds())
                    if first_alert is not None
                    else np.nan,
                    "prediction_probability_at_warning": float(
                        alerts.loc[alerts["target_time"].idxmin(), "predicted_probability"]
                    )
                    if len(alerts)
                    else np.nan,
                    "number_of_warning_windows_before_onset": int(len(alerts)),
                    **{
                        f"warnings_within_{bucket}s": int(
                            (
                                (onset - alerts["target_time"]).dt.total_seconds() <= bucket
                            ).sum()
                        )
                        if len(alerts)
                        else 0
                        for bucket in CONTEXT_BUCKETS
                    },
                    "missed_onset": bool(len(alerts) == 0),
                    "max_probability_in_lookback": float(before["predicted_probability"].max())
                    if len(before)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def post_attack_analysis(events, episodes):
    rows = []
    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("target_time")
        for ep in episodes:
            offset = ep["attack_offset"]
            after = group[group["target_time"] > offset].head(30)  # up to 300 s after offset
            if after.empty:
                continue

            positives = after[after["predicted_label"] == 1]
            below = after[after["predicted_label"] == 0]
            first_below = below["target_time"].min() if len(below) else None

            inside = group[group["attack_episode_id"] == ep["episode_id"]]

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "episode_id": ep["episode_id"],
                    "attack_offset": str(offset),
                    "windows_examined_after_offset": int(len(after)),
                    "positive_predictions_after_offset": int(len(positives)),
                    "consecutive_positive_after_offset": int(
                        runs_of((after["predicted_label"] == 1).to_numpy())[0]
                    )
                    if len(after) and after["predicted_label"].iloc[0] == 1
                    else 0,
                    "time_until_prediction_below_threshold_sec": float(
                        (first_below - offset).total_seconds()
                    )
                    if first_below is not None
                    else np.nan,
                    "mean_probability_inside_episode": float(
                        inside["predicted_probability"].mean()
                    )
                    if len(inside)
                    else np.nan,
                    "mean_probability_10s_after_offset": float(
                        after["predicted_probability"].iloc[0]
                    ),
                    "mean_probability_first_60s_after_offset": float(
                        after.head(6)["predicted_probability"].mean()
                    ),
                    "mean_probability_first_300s_after_offset": float(
                        after["predicted_probability"].mean()
                    ),
                    "false_positives_in_tail": int((after["error_type"] == "FP").sum()),
                }
            )
    return pd.DataFrame(rows)


def feature_summary(events):
    rows = []
    subset = events[events["model"].isin([MODEL_LR, MODEL_V4])]
    for (model, k, error_type), group in subset.groupby(
        ["model", "horizon", "error_type"], sort=False
    ):
        for feature in FEATURE_ANALYSIS_SUBSET:
            values = group[feature]
            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "error_type": error_type,
                    "feature": feature,
                    "feature_kind": "base (raw, unscaled, at forecast time t)",
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "p10": float(values.quantile(0.10)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def error_taxonomy(events, episodes):
    """
    Operational (not semantic) rules, fixed in advance. Every rule is a statement
    about measured timing or probability, never about attacker intent.

      FP-B  false positive whose target is within 120 s BEFORE an attack window
      FP-C  false positive whose target is within 120 s AFTER an attack window
      FP-A  isolated false positive (run length 1) more than 120 s from any attack
      FP-D  false positive inside a run of >= 3 more than 120 s from any attack
      FN-A  missed attack within the first 60 s of an episode
      FN-B  missed attack inside an episode shorter than 60 s
      FN-C  missed attack that is a target transition (A(t)=0, A(t+k)=1)
      FN-E  missed attack whose probability is within 0.10 below the frozen threshold
      FN-D  missed attack deep inside a long episode (> 300 s after onset)
      uncertain  anything the rules above do not cover
    """
    short_episodes = {ep["episode_id"] for ep in episodes if ep["duration_sec"] < 60}

    rows = []
    errors = events[events["error_type"].isin(["FP", "FN"])]

    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("sample_index")
        is_fp = (group["error_type"] == "FP").to_numpy()

        run_length = np.zeros(len(group), dtype=int)
        start = None
        for position, value in enumerate(list(is_fp) + [False]):
            if value and start is None:
                start = position
            elif not value and start is not None:
                run_length[start:position] = position - start
                start = None
        group = group.assign(fp_run_length=run_length)

        for _, row in group[group["error_type"].isin(["FP", "FN"])].iterrows():
            distance = row["signed_distance_to_attack_window_sec"]
            category, evidence = "uncertain", ""

            if row["error_type"] == "FP":
                if -120 <= distance < 0:
                    category = "FP-B"
                    evidence = f"target is {abs(distance):.0f}s before the nearest attack window"
                elif 0 < distance <= 120:
                    category = "FP-C"
                    evidence = f"target is {distance:.0f}s after the nearest attack window"
                elif abs(distance) > 120 and row["fp_run_length"] == 1:
                    category = "FP-A"
                    evidence = f"isolated single-window FP, {abs(distance):.0f}s from any attack"
                elif abs(distance) > 120 and row["fp_run_length"] >= 3:
                    category = "FP-D"
                    evidence = (
                        f"FP inside a run of {int(row['fp_run_length'])} windows, "
                        f"{abs(distance):.0f}s from any attack"
                    )
                else:
                    evidence = f"FP {abs(distance):.0f}s from any attack, run={int(row['fp_run_length'])}"
            else:
                gap = row["threshold"] - row["predicted_probability"]
                if row["attack_episode_id"] in short_episodes:
                    category = "FN-B"
                    evidence = f"episode {int(row['attack_episode_id'])} is shorter than 60s"
                elif row["is_target_transition"] == 1:
                    category = "FN-C"
                    evidence = "target is a 0->1 transition relative to A(t)"
                elif 0 <= row["time_from_attack_onset_sec"] <= 60:
                    category = "FN-A"
                    evidence = (
                        f"{row['time_from_attack_onset_sec']:.0f}s after episode onset"
                    )
                elif not np.isnan(gap) and 0 < gap <= 0.10:
                    category = "FN-E"
                    evidence = f"probability {row['predicted_probability']:.3f} is {gap:.3f} below the frozen threshold"
                elif row["time_from_attack_onset_sec"] > 300:
                    category = "FN-D"
                    evidence = (
                        f"{row['time_from_attack_onset_sec']:.0f}s after onset, deep inside an episode"
                    )
                else:
                    evidence = (
                        f"prob={row['predicted_probability']:.3f}, "
                        f"{row['time_from_attack_onset_sec']:.0f}s from onset"
                    )

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "sample_index": int(row["sample_index"]),
                    "target_time": str(row["target_time"]),
                    "error_type": row["error_type"],
                    "taxonomy_category": category,
                    "evidence_summary": evidence,
                }
            )

    return pd.DataFrame(rows), len(errors)


def representative_cases(events, taxonomy):
    """
    Explicit, stated-in-advance selection rule (no cherry-picking):
    for each model in {V4, LR} and each error type, take the K=5 errors and
    pick one case from each of the three most common taxonomy categories;
    within a category pick the MEDIAN-probability case, not the extreme one.
    Persistence cases: every persistence error at K=5 (there are few).
    """
    merged = events.merge(
        taxonomy[["model", "horizon", "sample_index", "taxonomy_category"]],
        on=["model", "horizon", "sample_index"],
        how="left",
    )

    rows = []
    case_id = 0

    for model in (MODEL_V4, MODEL_LR):
        for error_type in ("FP", "FN"):
            subset = merged[
                (merged["model"] == model)
                & (merged["horizon"] == 5)
                & (merged["error_type"] == error_type)
            ]
            if subset.empty:
                continue

            for category in subset["taxonomy_category"].value_counts().head(3).index:
                pool = subset[subset["taxonomy_category"] == category].sort_values(
                    "predicted_probability"
                )
                chosen = pool.iloc[len(pool) // 2]
                case_id += 1
                rows.append(
                    {
                        "case_id": f"C{case_id:02d}",
                        "model": model,
                        "horizon": 5,
                        "sample_index": int(chosen["sample_index"]),
                        "time": str(chosen["target_time"]),
                        "actual": int(chosen["actual_label"]),
                        "predicted": int(chosen["predicted_label"]),
                        "probability": float(chosen["predicted_probability"]),
                        "error_type": error_type,
                        "taxonomy_category": category,
                        "reason_selected": (
                            f"median-probability {error_type} of the {category} group "
                            f"({len(pool)} cases) — most common failure mode #"
                            f"{list(subset['taxonomy_category'].value_counts().head(3).index).index(category) + 1}"
                        ),
                        "context_before": context_string(merged, model, 5, int(chosen["sample_index"]), -3),
                        "context_after": context_string(merged, model, 5, int(chosen["sample_index"]), 3),
                        "flow_count": float(chosen["flow_count"]),
                        "syn_rate": float(chosen["syn_rate"]),
                        "window_packet_rate": float(chosen["window_packet_rate"]),
                        "time_from_attack_onset_sec": float(chosen["time_from_attack_onset_sec"]),
                        "signed_distance_to_attack_window_sec": float(
                            chosen["signed_distance_to_attack_window_sec"]
                        ),
                    }
                )

    persistence_errors = merged[
        (merged["model"] == MODEL_PERSISTENCE)
        & (merged["horizon"] == 5)
        & (merged["error_type"].isin(["FP", "FN"]))
    ]
    for _, chosen in persistence_errors.iterrows():
        case_id += 1
        rows.append(
            {
                "case_id": f"C{case_id:02d}",
                "model": MODEL_PERSISTENCE,
                "horizon": 5,
                "sample_index": int(chosen["sample_index"]),
                "time": str(chosen["target_time"]),
                "actual": int(chosen["actual_label"]),
                "predicted": int(chosen["predicted_label"]),
                "probability": float(chosen["predicted_probability"]),
                "error_type": chosen["error_type"],
                "taxonomy_category": chosen["taxonomy_category"],
                "reason_selected": "every persistence K=5 error is listed (they are few and "
                                   "each marks an attack-state transition)",
                "context_before": context_string(merged, MODEL_PERSISTENCE, 5, int(chosen["sample_index"]), -3),
                "context_after": context_string(merged, MODEL_PERSISTENCE, 5, int(chosen["sample_index"]), 3),
                "flow_count": float(chosen["flow_count"]),
                "syn_rate": float(chosen["syn_rate"]),
                "window_packet_rate": float(chosen["window_packet_rate"]),
                "time_from_attack_onset_sec": float(chosen["time_from_attack_onset_sec"]),
                "signed_distance_to_attack_window_sec": float(
                    chosen["signed_distance_to_attack_window_sec"]
                ),
            }
        )

    return pd.DataFrame(rows)


def context_string(merged, model, horizon, sample_index, offset):
    lo, hi = (sample_index + offset, sample_index) if offset < 0 else (sample_index + 1, sample_index + offset + 1)
    window = merged[
        (merged["model"] == model)
        & (merged["horizon"] == horizon)
        & (merged["sample_index"] >= max(lo, 0))
        & (merged["sample_index"] < hi)
    ].sort_values("sample_index")
    return " | ".join(
        f"p={row.predicted_probability:.3f},y={int(row.actual_label)}" for row in window.itertuples()
    )


def horizon_error_summary(events, scores, data, thresholds, hours):
    rows = []
    A_test = data["test"]["A"]

    for (model, k), group in events.groupby(["model", "horizon"], sort=False):
        group = group.sort_values("sample_index")
        actual = A_test[:, k - 1].astype(int)
        predicted = group["predicted_label"].to_numpy()

        if model == MODEL_PERSISTENCE:
            score = scores["persistence"]
            threshold = np.nan
        elif model == MODEL_LR:
            score = scores["lr"][k]
            threshold = thresholds["lr"][k]
        elif model == MODEL_V4:
            score = scores["v4"][:, k]
            threshold = thresholds["v4"][k]
        else:
            score = scores["lr"][k]
            threshold = thresholds["lr"][k]

        metrics = standard_metrics(score, actual, predicted)
        fp_runs = runs_of((group["error_type"] == "FP").to_numpy())
        fn_runs = runs_of((group["error_type"] == "FN").to_numpy())

        rows.append(
            {
                "model": model,
                "horizon": k,
                "threshold": threshold,
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "fpr": metrics["fpr"],
                "fp_per_hour": metrics["fp"] / hours,
                "fn_per_hour": metrics["fn"] / hours,
                "median_fp_run": float(np.median(fp_runs)) if fp_runs else 0.0,
                "median_fn_run": float(np.median(fn_runs)) if fn_runs else 0.0,
                "max_fp_run": int(max(fp_runs)) if fp_runs else 0,
                "max_fn_run": int(max(fn_runs)) if fn_runs else 0,
                "total_evaluated": metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"],
            }
        )
    return pd.DataFrame(rows)


def lr_probability_shift(events, data, thresholds):
    rows = []
    bins = np.arange(0.0, 1.01, 0.10)

    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_LR) & (events["horizon"] == k)]
        threshold = thresholds["lr"][k]

        for lower, upper in zip(bins[:-1], bins[1:]):
            in_bin = (group["predicted_probability"] >= lower) & (
                group["predicted_probability"] < upper + (1e-9 if upper == 1.0 else 0)
            )
            positives = group[in_bin & (group["actual_label"] == 1)]
            negatives = group[in_bin & (group["actual_label"] == 0)]

            rows.append(
                {
                    "model": MODEL_LR,
                    "horizon": k,
                    "threshold": threshold,
                    "probability_bin": f"{lower:.2f}-{upper:.2f}",
                    "bin_lower": float(lower),
                    "positives_in_bin": int(len(positives)),
                    "negatives_in_bin": int(len(negatives)),
                    "positives_below_threshold": int((positives["predicted_label"] == 0).sum()),
                    "bin_is_below_threshold": bool(upper <= threshold),
                }
            )

    summary_rows = []
    for k in POINT_HORIZONS:
        group = events[(events["model"] == MODEL_LR) & (events["horizon"] == k)]
        threshold = thresholds["lr"][k]
        positives = group[group["actual_label"] == 1]
        negatives = group[group["actual_label"] == 0]
        false_negatives = group[group["error_type"] == "FN"]

        summary_rows.append(
            {
                "model": MODEL_LR,
                "horizon": k,
                "threshold": threshold,
                "probability_bin": "SUMMARY",
                "bin_lower": np.nan,
                "positives_in_bin": int(len(positives)),
                "negatives_in_bin": int(len(negatives)),
                "positives_below_threshold": int(len(false_negatives)),
                "bin_is_below_threshold": False,
                "fn_count": int(len(false_negatives)),
                "fn_caused_by_probability_below_threshold": int(
                    (false_negatives["predicted_probability"] < threshold).sum()
                ),
                "median_positive_probability": float(positives["predicted_probability"].median()),
                "median_negative_probability": float(negatives["predicted_probability"].median()),
                "p90_negative_probability": float(
                    negatives["predicted_probability"].quantile(0.90)
                ),
                "positives_above_p90_of_negatives": int(
                    (
                        positives["predicted_probability"]
                        > negatives["predicted_probability"].quantile(0.90)
                    ).sum()
                ),
            }
        )

    return pd.concat([pd.DataFrame(rows), pd.DataFrame(summary_rows)], ignore_index=True)


def horizon_evolution(events, episodes):
    """
    Step 16: where do the per-horizon changes in error counts actually occur?
    Attack windows are bucketed by time since THEIR OWN episode onset.
    """
    buckets = [(0, 60), (60, 300), (300, 1800), (1800, 10 ** 9)]
    labels = ["0-60s", "60-300s", "300-1800s", ">1800s"]

    rows = []
    for model in (MODEL_LR, MODEL_V4):
        for k in POINT_HORIZONS:
            group = events[(events["model"] == model) & (events["horizon"] == k)]
            attacks = group[group["actual_label"] == 1]

            for (low, high), label in zip(buckets, labels):
                in_bucket = attacks[
                    (attacks["time_from_attack_onset_sec"] >= low)
                    & (attacks["time_from_attack_onset_sec"] < high)
                ]
                if in_bucket.empty:
                    continue
                detected = int((in_bucket["predicted_label"] == 1).sum())
                rows.append(
                    {
                        "model": model,
                        "horizon": k,
                        "time_since_episode_onset": label,
                        "attack_windows": int(len(in_bucket)),
                        "detected_tp": detected,
                        "missed_fn": int(len(in_bucket)) - detected,
                        "recall_in_bucket": detected / len(in_bucket),
                        "mean_probability": float(in_bucket["predicted_probability"].mean()),
                        "median_probability": float(in_bucket["predicted_probability"].median()),
                    }
                )
    return pd.DataFrame(rows)


def transition_analysis(events, scores, data):
    """
    Target transitions relative to the current observed label:
        0 -> 1 : A(t) = 0 and A(t+k) = 1   (an onset within the horizon)
        1 -> 0 : A(t) = 1 and A(t+k) = 0   (an offset within the horizon)
    Persistence is wrong on every transition by construction; the question is
    whether the feature-based models anticipate them.
    """
    label_t = scores["persistence"].astype(int)
    A_test = data["test"]["A"]

    rows = []
    for k in POINT_HORIZONS:
        actual = A_test[:, k - 1].astype(int)
        rising = (label_t == 0) & (actual == 1)
        falling = (label_t == 1) & (actual == 0)
        steady = ~(rising | falling)

        for model in (MODEL_PERSISTENCE, MODEL_LR, MODEL_V4):
            group = events[(events["model"] == model) & (events["horizon"] == k)].sort_values(
                "sample_index"
            )
            predicted = group["predicted_label"].to_numpy()

            positive_rate = float((predicted == 1).mean())

            rows.append(
                {
                    "model": model,
                    "horizon": k,
                    "predicted_positive_rate": positive_rate,
                    "expected_0to1_correct_at_base_rate": round(
                        positive_rate * int(rising.sum()), 3
                    ),
                    "expected_1to0_correct_at_base_rate": round(
                        (1.0 - positive_rate) * int(falling.sum()), 3
                    ),
                    "transition_0to1_count": int(rising.sum()),
                    "transition_0to1_correct": int((predicted[rising] == 1).sum()),
                    "transition_0to1_missed": int((predicted[rising] == 0).sum()),
                    "transition_1to0_count": int(falling.sum()),
                    "transition_1to0_correct": int((predicted[falling] == 0).sum()),
                    "transition_1to0_missed": int((predicted[falling] == 1).sum()),
                    "steady_windows": int(steady.sum()),
                    "steady_correct": int((predicted[steady] == actual[steady]).sum()),
                    "steady_accuracy": float(
                        (predicted[steady] == actual[steady]).mean()
                    ),
                    "all_transitions": int(rising.sum() + falling.sum()),
                    "all_transitions_correct": int(
                        (predicted[rising] == 1).sum() + (predicted[falling] == 0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


# =============================================================
# FIGURES
# =============================================================

def make_figures(events, episodes, raw, thresholds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def break_gaps(times, values):
        """Insert NaN where the window spacing exceeds 10 s so lines do not bridge data gaps."""
        times = pd.Series(times).reset_index(drop=True)
        values = pd.Series(values).reset_index(drop=True)
        gaps = times.diff().dt.total_seconds() > WINDOW_SECONDS
        values = values.mask(gaps.shift(-1).fillna(False) & False)
        out_times, out_values = [], []
        for i in range(len(times)):
            if i and gaps.iloc[i]:
                out_times.append(times.iloc[i] - pd.Timedelta(seconds=1))
                out_values.append(np.nan)
            out_times.append(times.iloc[i])
            out_values.append(values.iloc[i])
        return out_times, out_values

    def shade_attacks(axes):
        for ep in episodes:
            axes.axvspan(
                ep["attack_onset"], ep["attack_offset"] + pd.Timedelta(seconds=WINDOW_SECONDS),
                color="#d62728", alpha=0.12, zorder=0,
            )
            if ep["observable_from_test_start"]:
                axes.axvline(ep["attack_onset"], color="#d62728", linestyle="--", linewidth=1.2, zorder=1)

    # ---- 1 & 2: probability timelines ----
    for model, colour, filename, title in (
        (MODEL_V4, "#1f77b4", "timeline_v4_k5.png", "V4 World Model — K=5 attack probability"),
        (MODEL_LR, "#d1651a", "timeline_lr_k5.png", "Logistic Regression — K=5 attack probability"),
    ):
        group = events[(events["model"] == model) & (events["horizon"] == 5)].sort_values("target_time")
        figure, axes = plt.subplots(figsize=(13, 4.6))
        shade_attacks(axes)

        times, values = break_gaps(group["target_time"], group["predicted_probability"])
        axes.plot(times, values, color=colour, linewidth=0.8, label=f"{model} P(attack at t+5)")
        axes.axhline(
            group["threshold"].iloc[0], color="black", linestyle=":", linewidth=1.3,
            label=f"frozen threshold = {group['threshold'].iloc[0]:.2f}",
        )
        axes.set_ylim(-0.03, 1.03)
        axes.set_ylabel("Predicted probability")
        axes.set_xlabel("Target window time (Mar-2-2018)")
        axes.set_title(
            f"{title}\nred shading = ground-truth attack windows; dashed = observable onset",
            fontsize=11,
        )
        axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes.grid(alpha=0.25, linestyle=":")
        axes.legend(loc="center left", fontsize=8)
        figure.tight_layout()
        figure.savefig(FIG_DIR / filename, dpi=150)
        plt.close(figure)

    # ---- 3 & 4: FP/FN annotated timelines ----
    for model, filename, title in (
        (MODEL_V4, "errors_v4_k5.png", "V4 World Model — K=5 error timeline"),
        (MODEL_LR, "errors_lr_k5.png", "Logistic Regression — K=5 error timeline"),
    ):
        group = events[(events["model"] == model) & (events["horizon"] == 5)].sort_values("target_time")
        figure, axes = plt.subplots(figsize=(13, 3.4))
        shade_attacks(axes)

        for error_type, y, colour, marker in (
            ("TP", 3, "#2ca02c", "|"), ("TN", 2, "#7f7f7f", "|"),
            ("FP", 1, "#ff7f0e", "|"), ("FN", 0, "#d62728", "|"),
        ):
            subset = group[group["error_type"] == error_type]
            axes.scatter(
                subset["target_time"], np.full(len(subset), y),
                marker=marker, s=28, color=colour, linewidths=0.8,
                label=f"{error_type} ({len(subset)})",
            )

        axes.set_yticks([0, 1, 2, 3])
        axes.set_yticklabels(["FN", "FP", "TN", "TP"])
        axes.set_ylim(-0.6, 3.6)
        axes.set_xlabel("Target window time (Mar-2-2018)")
        axes.set_title(f"{title}  (frozen threshold {group['threshold'].iloc[0]:.2f})", fontsize=11)
        axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes.grid(alpha=0.25, linestyle=":", axis="x")
        axes.legend(loc="upper center", ncol=4, fontsize=8, framealpha=0.9)
        figure.tight_layout()
        figure.savefig(FIG_DIR / filename, dpi=150)
        plt.close(figure)

    # ---- 5: per-episode prediction timeline (around each observable onset) ----
    observable = [ep for ep in episodes if ep["observable_from_test_start"]]
    figure, axes_list = plt.subplots(len(observable), 1, figsize=(11, 3.1 * len(observable)))
    if len(observable) == 1:
        axes_list = [axes_list]

    for axes, ep in zip(axes_list, observable):
        onset = ep["attack_onset"]
        lo = onset - pd.Timedelta(seconds=300)
        hi = onset + pd.Timedelta(seconds=300)

        for model, colour in ((MODEL_V4, "#1f77b4"), (MODEL_LR, "#d1651a"), (MODEL_PERSISTENCE, "#8c8c8c")):
            group = events[
                (events["model"] == model) & (events["horizon"] == 5)
                & (events["target_time"] >= lo) & (events["target_time"] <= hi)
            ].sort_values("target_time")
            if group.empty:
                continue
            axes.plot(
                group["target_time"], group["predicted_probability"],
                marker="o", markersize=3, linewidth=1.1, color=colour, label=model,
            )

        axes.axvline(onset, color="#d62728", linestyle="--", linewidth=1.4)
        axes.axvspan(onset, ep["attack_offset"] + pd.Timedelta(seconds=WINDOW_SECONDS),
                     color="#d62728", alpha=0.12)
        axes.axhline(thresholds["v4"][5], color="#1f77b4", linestyle=":", linewidth=1.0)
        axes.axhline(thresholds["lr"][5], color="#d1651a", linestyle=":", linewidth=1.0)
        axes.set_ylim(-0.05, 1.05)
        axes.set_ylabel("P(attack)")
        axes.set_title(
            f"Episode {ep['episode_id']} — onset {onset:%H:%M:%S}, "
            f"{ep['duration_windows']} attack windows (K=5 scores, ±300 s)",
            fontsize=10,
        )
        axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        axes.grid(alpha=0.25, linestyle=":")
        axes.legend(fontsize=8, loc="center left")

    figure.tight_layout()
    figure.savefig(FIG_DIR / "episode_prediction_timelines.png", dpi=150)
    plt.close(figure)

    # ---- 6, 7, 8: aggregate error charts ----
    summary = events[events["model"].isin([MODEL_PERSISTENCE, MODEL_LR, MODEL_V4])]
    counts = (
        summary.groupby(["model", "horizon", "error_type"]).size().unstack(fill_value=0).reset_index()
    )
    styles = {MODEL_PERSISTENCE: "#8c8c8c", MODEL_LR: "#d1651a", MODEL_V4: "#1f77b4"}

    figure, axes_pair = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.26
    positions = np.arange(len(POINT_HORIZONS))
    for index, (model, colour) in enumerate(styles.items()):
        subset = counts[counts["model"] == model].sort_values("horizon")
        axes_pair[0].bar(positions + (index - 1) * width, subset["FP"], width, color=colour, label=model)
        axes_pair[1].bar(positions + (index - 1) * width, subset["FN"], width, color=colour, label=model)
    for axes, name in zip(axes_pair, ("False positives", "False negatives")):
        axes.set_xticks(positions)
        axes.set_xticklabels([f"K={k}" for k in POINT_HORIZONS])
        axes.set_title(f"{name} by horizon (Mar-2 test day)", fontsize=11)
        axes.set_ylabel("Windows")
        axes.grid(alpha=0.25, linestyle=":", axis="y")
        axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "error_counts_by_horizon.png", dpi=150)
    plt.close(figure)

    hours = len(events["sample_index"].unique()) * WINDOW_SECONDS / 3600.0
    figure, axes = plt.subplots(figsize=(7.4, 4.3))
    for index, (model, colour) in enumerate(styles.items()):
        subset = counts[counts["model"] == model].sort_values("horizon")
        axes.bar(positions + (index - 1) * width, subset["FP"] / hours, width, color=colour, label=model)
        for x, value in zip(positions + (index - 1) * width, subset["FP"] / hours):
            axes.text(x, value + 0.15, f"{value:.1f}", ha="center", fontsize=7.5)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"K={k}" for k in POINT_HORIZONS])
    axes.set_ylabel("False-positive windows / hour")
    axes.set_title("False-alarm rate by model and horizon\nMar-2 test day (8.65 evaluated hours)", fontsize=11)
    axes.grid(alpha=0.25, linestyle=":", axis="y")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "fp_per_hour.png", dpi=150)
    plt.close(figure)

    episode_counts = (
        summary[(summary["error_type"] == "FN") & (summary["horizon"] == 5)]
        .groupby(["model", "attack_episode_id"]).size().unstack(fill_value=0)
    )
    figure, axes = plt.subplots(figsize=(8.2, 4.3))
    episode_ids = [ep["episode_id"] for ep in episodes]
    positions = np.arange(len(episode_ids))
    for index, (model, colour) in enumerate(styles.items()):
        values = [episode_counts.loc[model].get(ep_id, 0) if model in episode_counts.index else 0
                  for ep_id in episode_ids]
        axes.bar(positions + (index - 1) * width, values, width, color=colour, label=model)
        for x, value in zip(positions + (index - 1) * width, values):
            if value:
                axes.text(x, value + 6, str(int(value)), ha="center", fontsize=7.5)
    axes.set_xticks(positions)
    axes.set_xticklabels(
        [f"ep{ep['episode_id']}\n{ep['attack_onset']:%H:%M}\n{ep['duration_windows']}w" for ep in episodes],
        fontsize=8,
    )
    axes.set_ylabel("Missed attack windows (FN)")
    axes.set_title("False negatives by attack episode (K=5, frozen thresholds)", fontsize=11)
    axes.grid(alpha=0.25, linestyle=":", axis="y")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "fn_by_episode.png", dpi=150)
    plt.close(figure)


# =============================================================
# MAIN
# =============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Error analysis of the frozen three-model comparison.")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print("FALSE-POSITIVE / FALSE-NEGATIVE ERROR ANALYSIS — Mar-2-2018 held-out day")
    print("=" * 100)
    print("Analysis only: no training, no threshold change, no artifact modification.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v4_manifest = json.loads(V4_MANIFEST.read_text())
    lr_manifest = json.loads(LR_MANIFEST.read_text())

    thresholds = {
        "v4": {k: float(v4_manifest["threshold"]["per_horizon_future_prediction"][str(k)])
               for k in POINT_HORIZONS},
        "lr": {k: float(lr_manifest["thresholds"][f"k{k}"]) for k in POINT_HORIZONS + [4]},
    }
    print(f"\nFrozen thresholds  V4: {thresholds['v4']}")
    print(f"Frozen thresholds  LR: {thresholds['lr']}")

    data = load_benchmark()
    raw = pd.read_csv(TEST_DAY_CSV, low_memory=False)
    raw["window_start"] = pd.to_datetime(raw["window_start"])
    raw = raw.sort_values("window_start").reset_index(drop=True)

    episodes = build_episodes(raw)
    print(f"\nAttack episodes on the raw test day: {len(episodes)}")
    for ep in episodes:
        print(f"  ep{ep['episode_id']}: {ep['attack_onset']} -> {ep['attack_offset']}  "
              f"{ep['duration_windows']} windows  observable_onset={ep['observable_from_test_start']}")

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    v4_probabilities, _ = v4_scores(data, device)

    stored_lr = pd.read_csv(LR_PREDICTIONS)
    scores = {
        "v4": v4_probabilities["test"],
        "lr": {k: stored_lr[f"prob_t+{k}"].to_numpy() for k in POINT_HORIZONS + [4]},
        "persistence": fetch_labels_at_time_t(data["test"]["meta"]),
    }

    hours = len(data["test"]["meta"]) * WINDOW_SECONDS / 3600.0
    print(f"\nEvaluated windows: {len(data['test']['meta'])}  ({hours:.2f} hours of 10 s windows)")

    # ----- Error events -----
    events = build_error_events(data, scores, thresholds, episodes, raw)
    events.to_csv(OUT_DIR / "error_events.csv", index=False)
    print(f"\nWrote error_events.csv  ({len(events)} rows)")

    primary = events[events["model"].isin([MODEL_PERSISTENCE, MODEL_LR, MODEL_V4])]

    # ----- Task 3 cross-check (STOP on mismatch) -----
    horizon_summary = horizon_error_summary(primary, scores, data, thresholds, hours)
    task3 = pd.read_csv(TASK3_CSV)
    task3_point = task3[task3["evaluation_type"] == "point_forecast"]

    mismatches = []
    for _, row in horizon_summary.iterrows():
        reference = task3_point[
            (task3_point["model"] == row["model"]) & (task3_point["horizon"] == row["horizon"])
        ]
        if reference.empty:
            mismatches.append(f"{row['model']} K={row['horizon']}: missing in Task 3 CSV")
            continue
        reference = reference.iloc[0]
        for field in ("precision", "recall", "f1", "roc_auc", "fpr", "tp", "fp", "tn", "fn"):
            if abs(float(row[field]) - float(reference[field])) > TOLERANCE:
                mismatches.append(
                    f"{row['model']} K={row['horizon']} {field}: {row[field]} vs {reference[field]}"
                )

    if mismatches:
        raise RuntimeError(
            "Aggregate metrics do not reproduce results/standardized_baseline_comparison.csv:\n  "
            + "\n  ".join(mismatches)
        )
    print("Task 3 cross-check: all point-forecast metrics reproduce exactly")

    # ----- Tables -----
    pd.DataFrame(episodes).to_csv(OUT_DIR / "attack_episodes.csv", index=False)
    fp_summary = false_positive_summary(primary, hours)
    fp_summary.to_csv(OUT_DIR / "false_positive_summary.csv", index=False)
    fn_summary = false_negative_summary(primary, episodes)
    fn_summary.to_csv(OUT_DIR / "false_negative_summary.csv", index=False)
    onsets = onset_analysis(primary, episodes)
    onsets.to_csv(OUT_DIR / "onset_analysis.csv", index=False)
    post_attack = post_attack_analysis(primary, episodes)
    post_attack.to_csv(OUT_DIR / "post_attack_analysis.csv", index=False)
    feature_stats = feature_summary(primary)
    feature_stats.to_csv(OUT_DIR / "error_feature_summary.csv", index=False)
    taxonomy, error_total = error_taxonomy(primary, episodes)
    taxonomy.to_csv(OUT_DIR / "error_taxonomy.csv", index=False)
    cases = representative_cases(primary, taxonomy)
    cases.to_csv(OUT_DIR / "representative_cases.csv", index=False)
    horizon_summary.to_csv(OUT_DIR / "horizon_error_summary.csv", index=False)
    shift = lr_probability_shift(events, data, thresholds)
    shift.to_csv(OUT_DIR / "lr_probability_shift.csv", index=False)
    transitions = transition_analysis(primary, scores, data)
    transitions.to_csv(OUT_DIR / "transition_analysis.csv", index=False)
    evolution = horizon_evolution(primary, episodes)
    evolution.to_csv(OUT_DIR / "horizon_evolution.csv", index=False)

    for name in (
        "attack_episodes.csv", "false_positive_summary.csv", "false_negative_summary.csv",
        "onset_analysis.csv", "post_attack_analysis.csv", "error_feature_summary.csv",
        "error_taxonomy.csv", "representative_cases.csv", "horizon_error_summary.csv",
        "lr_probability_shift.csv", "transition_analysis.csv", "horizon_evolution.csv",
    ):
        print(f"Wrote {name}")

    if not args.skip_plots:
        make_figures(primary, episodes, raw, thresholds)
        print(f"Wrote figures to {FIG_DIR.relative_to(REPO_ROOT)}")

    # ----- Console summary -----
    print("\n" + "=" * 100)
    print("HORIZON ERROR SUMMARY")
    print("=" * 100)
    print(
        horizon_summary[
            ["model", "horizon", "threshold", "tp", "fp", "tn", "fn", "precision", "recall",
             "f1", "fp_per_hour", "fn_per_hour", "max_fp_run", "max_fn_run"]
        ].round(4).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("TRANSITION ANALYSIS (targets where A(t+k) != A(t))")
    print("=" * 100)
    print(
        transitions[
            ["model", "horizon", "transition_0to1_count", "transition_0to1_correct",
             "transition_1to0_count", "transition_1to0_correct", "steady_accuracy"]
        ].round(4).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("ONSET ANALYSIS (K=5, 120 s lookback)")
    print("=" * 100)
    print(
        onsets[onsets["horizon"] == 5][
            ["model", "episode_id", "attack_onset", "warning_detected", "first_warning_time",
             "lead_time_sec", "number_of_warning_windows_before_onset", "max_probability_in_lookback"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("HORIZON EVOLUTION — recall on attack windows by time since their episode onset")
    print("=" * 100)
    print(
        evolution.pivot_table(
            index=["model", "time_since_episode_onset"], columns="horizon",
            values="recall_in_bucket",
        ).round(4).to_string()
    )

    # ----- Summary JSON -----
    summary = {
        "test_day": "2018-03-02 (Friday-02-03-2018)",
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluated_windows": int(len(data["test"]["meta"])),
        "evaluated_hours": round(hours, 4),
        "horizons": POINT_HORIZONS,
        "frozen_thresholds": {"v4_world_model": thresholds["v4"], "logistic_regression":
                              {k: v for k, v in thresholds["lr"].items() if k in POINT_HORIZONS},
                              "persistence": "none (deterministic rule)"},
        "models": {
            model: {
                f"K={int(row['horizon'])}": {
                    field: (None if pd.isna(row[field]) else
                            (int(row[field]) if field in ("tp", "fp", "tn", "fn") else float(row[field])))
                    for field in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "roc_auc",
                                  "fpr", "fp_per_hour", "fn_per_hour", "median_fp_run",
                                  "max_fp_run", "median_fn_run", "max_fn_run")
                }
                for _, row in horizon_summary[horizon_summary["model"] == model].iterrows()
            }
            for model in (MODEL_PERSISTENCE, MODEL_LR, MODEL_V4)
        },
        "episodes": [
            {
                "episode_id": ep["episode_id"],
                "attack_onset": str(ep["attack_onset"]),
                "attack_offset": str(ep["attack_offset"]),
                "duration_windows": ep["duration_windows"],
                "duration_sec": ep["duration_sec"],
                "attack_type": ep["attack_type_if_available"],
                "observable_onset": ep["observable_from_test_start"],
            }
            for ep in episodes
        ],
        "transition_counts": {
            f"K={k}": {
                "transition_0to1": int(
                    transitions[(transitions["horizon"] == k)]["transition_0to1_count"].iloc[0]
                ),
                "transition_1to0": int(
                    transitions[(transitions["horizon"] == k)]["transition_1to0_count"].iloc[0]
                ),
                "correct_by_model": {
                    row["model"]: {
                        "0to1": int(row["transition_0to1_correct"]),
                        "1to0": int(row["transition_1to0_correct"]),
                    }
                    for _, row in transitions[transitions["horizon"] == k].iterrows()
                },
            }
            for k in POINT_HORIZONS
        },
        "horizon_summary": json.loads(horizon_summary.to_json(orient="records")),
        "horizon_evolution": json.loads(evolution.to_json(orient="records")),
        "taxonomy_counts": json.loads(
            taxonomy.groupby(["model", "horizon", "taxonomy_category"]).size()
            .rename("count").reset_index().to_json(orient="records")
        ),
        "verification": {
            "task3_aggregate_metrics_reproduced": True,
            "thresholds_frozen_not_reselected": True,
            "test_set_unchanged": int(len(data["test"]["meta"])) == 3115,
            "no_model_retrained": True,
            "no_artifact_modified": True,
        },
        "provenance": {
            "repository_commit": git_value("rev-parse", "HEAD"),
            "v4_reference_commit": v4_manifest["evaluation"]["reference_script_commit"],
            "v4_checkpoint_sha256": v4_manifest["checkpoint"]["sha256"],
            "lr_predictions_sha256": sha256_file(LR_PREDICTIONS),
            "task3_csv_sha256": sha256_file(TASK3_CSV),
        },
        "caveats": [
            "Persistence receives the observed label A(t); V4 and LR receive only features.",
            "Only one held-out day and three observable attack onsets exist — onset findings "
            "are descriptive, not statistically conclusive.",
            "All four episodes on this day are labelled Bot.",
            "A false positive near an attack onset remains a false positive under the benchmark label.",
            "The V4 training/evaluation normalisation provenance caveat from Task 1 is unresolved.",
        ],
    }

    with open(SUMMARY_JSON, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"\nWrote {SUMMARY_JSON.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 100)
    print("ERROR ANALYSIS GENERATED")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
