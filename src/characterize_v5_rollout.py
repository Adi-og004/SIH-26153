"""
=============================================================
V5 ROLLOUT CHARACTERIZATION — HOST-LEVEL REAL REPLAY
=============================================================

Read-only characterization of the EXISTING V5 checkpoint over all real
host-level sequences in `results/v5_demo/host_172.31.69.30_X.npy`.

-------------------------------------------------------------
WHAT THIS IS NOT
-------------------------------------------------------------
This artifact has NO ground-truth labels, so nothing here is a
performance evaluation. No accuracy, F1, precision, recall, ROC or
calibration figure is computed, and none can be. The question answered is
narrow and purely behavioural:

    does this checkpoint produce a usable dynamic range of threat
    probabilities on this real host-level input, or does it collapse /
    saturate?

Every threshold that appears (0.1 / 0.25 / 0.5 / 0.75 / 0.9) is a
DIAGNOSTIC probe for describing a distribution. None is a validated
operating point.

-------------------------------------------------------------
SCOPE SHIFT — APPLIES TO EVERY NUMBER BELOW
-------------------------------------------------------------
Observed output behavior may be affected by train/inference distribution
shift caused by host-level versus organization-wide feature scope. The V5
model's 52-D V4 prefix was fitted on organization-wide flow state; this
replay is a single host. Nothing here is corrected, renormalized or
refitted to compensate — the shift is reported, not removed.

Usage:
    python src/characterize_v5_rollout.py
    python src/characterize_v5_rollout.py --batch-size 512

=============================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer_latent_world_model_v5 import V5Inference

REPO_ROOT = SRC_DIR.parent
DEMO_DIR = REPO_ROOT / "results" / "v5_demo"

HOST = "172.31.69.30"
X_PATH = DEMO_DIR / f"host_{HOST}_X.npy"
METADATA_PATH = DEMO_DIR / f"host_{HOST}_metadata.csv"

SUMMARY_CSV = DEMO_DIR / "rollout_summary.csv"
DISTRIBUTION_JSON = DEMO_DIR / "rollout_distribution.json"
STATE_CSV = DEMO_DIR / "state_rollout_summary.csv"
TOP_CSV = DEMO_DIR / "top_rollout_sequences.csv"
FIG_DIST = DEMO_DIR / "threat_probability_distribution.png"
FIG_QUANT = DEMO_DIR / "threat_probability_quantiles.png"
FIG_TOP = DEMO_DIR / "top_trajectory_examples.png"

K = 5
DIAGNOSTIC_THRESHOLDS = (0.1, 0.25, 0.5, 0.75, 0.9)
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)

SCOPE_SHIFT_NOTE = (
    "Observed output behavior may be affected by train/inference "
    "distribution shift caused by host-level versus organization-wide "
    "feature scope."
)
NOT_A_BENCHMARK = (
    "No labels exist for this replay. These are descriptive model-output "
    "statistics only — not accuracy, not calibration, not early-warning "
    "or attack-detection claims. Diagnostic thresholds are not validated "
    "operating thresholds."
)

# Ordinal 5-step single-hue ramp (t+1 .. t+5 are ORDERED, not categorical).
HORIZON_COLORS = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
# Fixed categorical order, used only for the handful of example trajectories.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"


def _rule(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def _style(ax) -> None:
    """Recessive grid and axes; text in ink tokens, never series color."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


# =============================================================
# INFERENCE
# =============================================================


def run_batched_inference(X: np.ndarray, batch_size: int) -> dict:
    """Run every sequence through V5Inference at K=5, in batches.

    The engine is used exactly as-is; nothing about the model, its
    weights, the scalers or the threshold is touched.
    """
    _rule("STEP 1 — batched V5 inference")

    engine = V5Inference()
    engine.load_model()
    print(f"  checkpoint   : {engine.checkpoint_path.relative_to(REPO_ROOT)}")
    print(f"  params       : {engine.parameter_count}")
    print(f"  threshold    : {engine.threshold} (frozen, from checkpoint)")

    n = X.shape[0]
    probabilities = np.empty((n, K), dtype=np.float64)
    latents = np.empty((n, K, engine.latent_dim), dtype=np.float64)
    states = np.empty((n, K, engine.output_dim), dtype=np.float64)
    states_scaled = np.empty((n, K, engine.output_dim), dtype=np.float64)
    current = np.empty((n, engine.latent_dim), dtype=np.float64)

    print(f"\nTotal sequences  : {n}")
    print(f"Batch size       : {batch_size}")

    started = time.perf_counter()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        result = engine.k_step_rollout(X[start:stop], steps=K)
        probabilities[start:stop] = result.threat_probabilities
        latents[start:stop] = result.future_latents
        states[start:stop] = result.future_states
        states_scaled[start:stop] = result.future_states_scaled
        current[start:stop] = result.current_latent
    elapsed = time.perf_counter() - started

    print(f"Total inference time: {elapsed:.2f} s")
    print(f"Sequences/sec       : {n / elapsed:,.1f}")

    non_finite = sum(
        int((~np.isfinite(a)).sum())
        for a in (probabilities, latents, states, states_scaled, current)
    )
    print(f"Non-finite values across all outputs: {non_finite}")

    return {
        "engine": engine,
        "probabilities": probabilities,
        "latents": latents,
        "states": states,
        "states_scaled": states_scaled,
        "current_latent": current,
        "elapsed_s": elapsed,
        "batch_size": batch_size,
        "non_finite": non_finite,
    }


# =============================================================
# OUTPUT 1 — PER-SEQUENCE SUMMARY
# =============================================================


def build_summary(metadata: pd.DataFrame, P: np.ndarray) -> pd.DataFrame:
    """One row per sequence, with trajectory-SHAPE descriptors.

    `monotonic_increase` means exactly p1 <= p2 <= p3 <= p4 <= p5. It is a
    shape descriptor and nothing more — it is NOT attack progression.
    """
    _rule("STEP 2 — per-sequence rollout summary")

    diffs = np.diff(P, axis=1)
    frame = pd.DataFrame(
        {
            "sequence_index": metadata["sequence_index"].to_numpy(),
            "window_idx_start": metadata["window_idx_start"].to_numpy(),
            "window_idx_end": metadata["window_idx_end"].to_numpy(),
            "timestamp_start": metadata["timestamp_start"].to_numpy(),
            "timestamp_end": metadata["timestamp_end"].to_numpy(),
            "timestamp_start_utc": metadata["timestamp_start_utc"].to_numpy(),
            "timestamp_end_utc": metadata["timestamp_end_utc"].to_numpy(),
        }
    )
    for k in range(K):
        frame[f"p_t+{k + 1}"] = P[:, k]

    frame["max_probability"] = P.max(axis=1)
    frame["mean_probability"] = P.mean(axis=1)
    frame["min_probability"] = P.min(axis=1)
    frame["delta_p_1_to_5"] = P[:, -1] - P[:, 0]
    frame["monotonic_increase"] = (diffs >= 0).all(axis=1)
    frame["monotonic_decrease"] = (diffs <= 0).all(axis=1)
    frame["argmax_horizon"] = P.argmax(axis=1) + 1

    frame.to_csv(SUMMARY_CSV, index=False)
    print(f"  wrote {SUMMARY_CSV.relative_to(REPO_ROOT)} ({len(frame)} rows)")
    return frame


# =============================================================
# OUTPUT 2 + 3 — DISTRIBUTION AND TRAJECTORY BEHAVIOUR
# =============================================================


def build_distribution(P: np.ndarray, summary: pd.DataFrame) -> dict:
    _rule("STEP 3 — probability distribution by horizon")

    per_horizon = {}
    for k in range(K):
        column = P[:, k]
        entry = {
            "min": float(column.min()),
            "max": float(column.max()),
            "mean": float(column.mean()),
            "median": float(np.median(column)),
            "std": float(column.std(ddof=1)),
        }
        for q in PERCENTILES:
            entry[f"p{q}"] = float(np.percentile(column, q))
        for threshold in DIAGNOSTIC_THRESHOLDS:
            entry[f"fraction_ge_{threshold}"] = float((column >= threshold).mean())
        per_horizon[f"t+{k + 1}"] = entry

    header = f"  {'horizon':<8}{'min':>10}{'median':>11}{'mean':>11}{'p95':>11}{'p99':>11}{'max':>11}"
    print(header)
    for name, entry in per_horizon.items():
        print(
            f"  {name:<8}{entry['min']:>10.6f}{entry['median']:>11.6f}"
            f"{entry['mean']:>11.6f}{entry['p95']:>11.6f}"
            f"{entry['p99']:>11.6f}{entry['max']:>11.6f}"
        )

    print(f"\n  fraction >= diagnostic threshold (NOT operating points):")
    print(f"  {'horizon':<8}" + "".join(f"{f'>={t}':>11}" for t in DIAGNOSTIC_THRESHOLDS))
    for name, entry in per_horizon.items():
        print(
            f"  {name:<8}"
            + "".join(
                f"{entry[f'fraction_ge_{t}']:>11.5f}" for t in DIAGNOSTIC_THRESHOLDS
            )
        )

    delta = summary["delta_p_1_to_5"].to_numpy()
    diffs = np.diff(P, axis=1)
    trajectory = {
        "fraction_monotonic_increase": float(summary["monotonic_increase"].mean()),
        "fraction_monotonic_decrease": float(summary["monotonic_decrease"].mean()),
        "fraction_any_increase_after_t1": float((diffs > 0).any(axis=1).mean()),
        "fraction_max_at_t5": float((summary["argmax_horizon"] == 5).mean()),
        "fraction_max_at_t1": float((summary["argmax_horizon"] == 1).mean()),
        "mean_change_t1_to_t5": float(delta.mean()),
        "median_change_t1_to_t5": float(np.median(delta)),
        "std_change_t1_to_t5": float(delta.std(ddof=1)),
    }

    max_probability = summary["max_probability"].to_numpy()
    max_distribution = {
        "min": float(max_probability.min()),
        "max": float(max_probability.max()),
        "mean": float(max_probability.mean()),
        "median": float(np.median(max_probability)),
        "std": float(max_probability.std(ddof=1)),
    }
    for q in PERCENTILES:
        max_distribution[f"p{q}"] = float(np.percentile(max_probability, q))
    for threshold in DIAGNOSTIC_THRESHOLDS:
        max_distribution[f"fraction_ge_{threshold}"] = float(
            (max_probability >= threshold).mean()
        )

    print("\n  trajectory behaviour:")
    for key, value in trajectory.items():
        print(f"    {key:<34}{value: .6f}")

    return {
        "per_horizon": per_horizon,
        "trajectory_behaviour": trajectory,
        "max_probability_distribution": max_distribution,
    }


# =============================================================
# OUTPUT 4 — LATENT STABILITY
# =============================================================


def latent_stability(latents: np.ndarray, current: np.ndarray) -> dict:
    _rule("STEP 4 — latent stability")

    norms = np.linalg.norm(latents, axis=2)                       # (N, 5)
    current_norm = np.linalg.norm(current, axis=1)                # (N,)
    # ||z_k - z_(k-1)|| for k = 2..5
    deltas = np.linalg.norm(np.diff(latents, axis=1), axis=2)     # (N, 4)
    # extra, clearly separate: the first transition out of the encoder
    first_step = np.linalg.norm(latents[:, 0, :] - current, axis=1)

    report: dict = {
        "encoder_latent_norm": {
            "mean": float(current_norm.mean()),
            "median": float(np.median(current_norm)),
            "p95": float(np.percentile(current_norm, 95)),
        },
        "norm_by_horizon": {},
        "transition_delta_by_horizon": {},
        "extra_first_transition_z0_to_z1": {
            "mean": float(first_step.mean()),
            "median": float(np.median(first_step)),
            "p95": float(np.percentile(first_step, 95)),
        },
        "n_nan": int(np.isnan(latents).sum()),
        "n_inf": int(np.isinf(latents).sum()),
    }

    for k in range(K):
        column = norms[:, k]
        report["norm_by_horizon"][f"t+{k + 1}"] = {
            "mean": float(column.mean()),
            "median": float(np.median(column)),
            "p95": float(np.percentile(column, 95)),
            "min": float(column.min()),
            "max": float(column.max()),
        }
    for k in range(K - 1):
        column = deltas[:, k]
        report["transition_delta_by_horizon"][f"t+{k + 1}->t+{k + 2}"] = {
            "mean": float(column.mean()),
            "median": float(np.median(column)),
            "p95": float(np.percentile(column, 95)),
            "max": float(column.max()),
        }

    print(f"  {'horizon':<10}{'mean |z|':>11}{'median':>11}{'p95':>11}{'max':>11}")
    for name, entry in report["norm_by_horizon"].items():
        print(
            f"  {name:<10}{entry['mean']:>11.4f}{entry['median']:>11.4f}"
            f"{entry['p95']:>11.4f}{entry['max']:>11.4f}"
        )
    print(f"\n  {'step':<14}{'mean d':>11}{'median':>11}{'p95':>11}{'max':>11}")
    for name, entry in report["transition_delta_by_horizon"].items():
        print(
            f"  {name:<14}{entry['mean']:>11.4f}{entry['median']:>11.4f}"
            f"{entry['p95']:>11.4f}{entry['max']:>11.4f}"
        )

    # Explicit stability verdicts, computed rather than asserted.
    norm_growth = (
        report["norm_by_horizon"]["t+5"]["mean"]
        / max(report["norm_by_horizon"]["t+1"]["mean"], 1e-12)
    )
    delta_names = list(report["transition_delta_by_horizon"])
    delta_decay = (
        report["transition_delta_by_horizon"][delta_names[-1]]["mean"]
        / max(report["transition_delta_by_horizon"][delta_names[0]]["mean"], 1e-12)
    )
    report["norm_growth_t1_to_t5"] = float(norm_growth)
    report["delta_ratio_last_over_first"] = float(delta_decay)
    report["exploding"] = bool(norms.max() > 1e3)
    report["non_finite"] = bool(report["n_nan"] or report["n_inf"])
    print(f"\n  mean-norm growth t+1 -> t+5      : {norm_growth:.4f}x")
    print(f"  transition delta last/first      : {delta_decay:.4f}x")
    print(f"  exploding (any |z| > 1e3)        : {report['exploding']}")
    print(f"  NaN / Inf in latents             : {report['n_nan']} / {report['n_inf']}")
    return report


# =============================================================
# OUTPUT 5 — PREDICTED STATE BEHAVIOUR
# =============================================================


def state_behaviour(
    states: np.ndarray, states_scaled: np.ndarray, names: tuple[str, ...]
) -> tuple[pd.DataFrame, list[dict]]:
    _rule("STEP 5 — predicted 26-D state behaviour")

    rows = []
    for j, name in enumerate(names):
        for k in range(K):
            column = states[:, k, j]
            rows.append(
                {
                    "feature": name,
                    "feature_index": j,
                    "horizon": f"t+{k + 1}",
                    "mean": float(column.mean()),
                    "std": float(column.std(ddof=1)),
                    "min": float(column.min()),
                    "max": float(column.max()),
                    "mean_scaled": float(states_scaled[:, k, j].mean()),
                    "std_scaled": float(states_scaled[:, k, j].std(ddof=1)),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(STATE_CSV, index=False)
    print(f"  wrote {STATE_CSV.relative_to(REPO_ROOT)} ({len(frame)} rows)")

    # Ranking uses TARGET-SCALED space: the 26 features have wildly
    # different natural units, so an original-unit ranking would just
    # rank by magnitude. Both are reported.
    change_scaled = np.abs(states_scaled[:, -1, :] - states_scaled[:, 0, :]).mean(axis=0)
    change_original = np.abs(states[:, -1, :] - states[:, 0, :]).mean(axis=0)
    order = np.argsort(change_scaled)[::-1][:10]

    top = [
        {
            "rank": int(r + 1),
            "feature": names[j],
            "mean_abs_change_t1_to_t5_scaled": float(change_scaled[j]),
            "mean_abs_change_t1_to_t5_original_units": float(change_original[j]),
        }
        for r, j in enumerate(order)
    ]
    print("\n  top 10 by mean |state(t+5) - state(t+1)| in TARGET-SCALED space:")
    print(f"  {'#':<4}{'feature':<30}{'scaled':>12}{'original':>16}")
    for entry in top:
        print(
            f"  {entry['rank']:<4}{entry['feature']:<30}"
            f"{entry['mean_abs_change_t1_to_t5_scaled']:>12.5f}"
            f"{entry['mean_abs_change_t1_to_t5_original_units']:>16.5f}"
        )
    print("\n  (descriptive model-output behaviour only — NOT attack indicators)")
    return frame, top


# =============================================================
# OUTPUT 6 — HIGHEST MODEL-RISK-OUTPUT SEQUENCES
# =============================================================


def top_sequences(summary: pd.DataFrame) -> pd.DataFrame:
    """Top 20 by each of three rankings.

    These are the highest model-risk-output sequences in the host-level
    replay. They are NOT attacks — no label exists to make that claim.
    """
    _rule("STEP 6 — highest model-risk-output sequences")

    columns = [
        "sequence_index",
        "window_idx_start",
        "window_idx_end",
        "timestamp_start",
        "timestamp_end",
        "timestamp_start_utc",
        "timestamp_end_utc",
    ] + [f"p_t+{k + 1}" for k in range(K)] + ["max_probability", "mean_probability"]

    frames = []
    for label, key in (
        ("max_probability", "max_probability"),
        ("mean_probability", "mean_probability"),
        ("p_t+5", "p_t+5"),
    ):
        block = summary.nlargest(20, key)[columns].copy()
        block.insert(0, "ranking", label)
        block.insert(1, "rank", range(1, len(block) + 1))
        frames.append(block)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(TOP_CSV, index=False)
    print(f"  wrote {TOP_CSV.relative_to(REPO_ROOT)} ({len(combined)} rows)")

    head = frames[0].head(10)
    print("\n  top 10 by max_probability (highest model-risk-output windows):")
    print(
        f"  {'#':<4}{'seq':>7}{'win_start':>11}{'UTC start':>21}"
        f"{'p_t+1':>10}{'p_t+5':>10}{'max':>10}"
    )
    for _, row in head.iterrows():
        print(
            f"  {int(row['rank']):<4}{int(row['sequence_index']):>7}"
            f"{int(row['window_idx_start']):>11}{row['timestamp_start_utc']:>21}"
            f"{row['p_t+1']:>10.5f}{row['p_t+5']:>10.5f}"
            f"{row['max_probability']:>10.5f}"
        )
    return combined


# =============================================================
# FIGURES (diagnostic only)
# =============================================================


def make_figures(P: np.ndarray, summary: pd.DataFrame) -> None:
    _rule("STEP 7 — diagnostic figures")

    floor = max(P[P > 0].min() * 0.5, 1e-9) if (P > 0).any() else 1e-9
    clipped = np.clip(P, floor, 1.0)

    # ---- Figure 1: distribution per horizon, small multiples ----
    fig, axes = plt.subplots(K, 1, figsize=(9, 8.5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    # Bound the axis to where data actually lives; a fixed 0..1 range
    # would squeeze every distribution into the left fifth of the frame.
    top_edge = float(P.max()) * 1.6
    bins = np.logspace(np.log10(floor), np.log10(top_edge), 60)
    for k, ax in enumerate(axes):
        _style(ax)
        ax.hist(clipped[:, k], bins=bins, color=HORIZON_COLORS[k], edgecolor="none")
        ax.set_xscale("log")
        ax.set_xlim(floor, top_edge)
        ax.set_ylabel(f"t+{k + 1}", color=INK, fontsize=10, rotation=0,
                      ha="right", va="center", labelpad=18)
        median = np.median(P[:, k])
        ax.axvline(max(median, floor), color=INK_MUTED, linewidth=1.2,
                   linestyle="--", alpha=0.8)
        ax.text(max(median, floor), ax.get_ylim()[1] * 0.82,
                f"  median {median:.4f}", color=INK_MUTED, fontsize=8, va="top")
    axes[-1].set_xlabel("threat probability (log scale)", color=INK, fontsize=10)
    fig.suptitle(
        "V5 threat-probability distribution by rollout horizon\n"
        f"host {HOST} replay, {len(P):,} real sequences — unlabelled, descriptive only",
        color=INK, fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIST, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {FIG_DIST.relative_to(REPO_ROOT)}")

    # ---- Figure 2: quantile bands across horizons ----
    horizons = np.arange(1, K + 1)
    q = {p: np.percentile(P, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.fill_between(horizons, q[5], q[95], color=HORIZON_COLORS[0], alpha=0.45,
                    linewidth=0, label="p5–p95")
    ax.fill_between(horizons, q[10], q[90], color=HORIZON_COLORS[1], alpha=0.55,
                    linewidth=0, label="p10–p90")
    ax.fill_between(horizons, q[25], q[75], color=HORIZON_COLORS[2], alpha=0.65,
                    linewidth=0, label="p25–p75")
    ax.plot(horizons, q[50], color=HORIZON_COLORS[4], linewidth=2,
            marker="o", markersize=8, label="median")
    for x, y in zip(horizons, q[50]):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", color=INK_MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"t+{k}" for k in horizons])
    ax.set_xlabel("rollout horizon (10 s per step)", color=INK, fontsize=10)
    ax.set_ylabel("threat probability (log scale)", color=INK, fontsize=10)
    ax.set_title(
        "V5 threat-probability quantiles across the 5-step rollout\n"
        f"host {HOST} replay, {len(P):,} real sequences — unlabelled, descriptive only",
        color=INK, fontsize=12,
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_MUTED)
    fig.tight_layout()
    fig.savefig(FIG_QUANT, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {FIG_QUANT.relative_to(REPO_ROOT)}")

    # ---- Figure 3: highest model-risk-output trajectories ----
    top = summary.nlargest(6, "max_probability")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    sample = np.random.default_rng(0).choice(
        len(P), size=min(400, len(P)), replace=False
    )
    for i in sample:
        ax.plot(horizons, clipped[i], color=INK_MUTED, alpha=0.05, linewidth=0.8)
    # Direct labels only — a legend box here would sit on top of the
    # t+1 points, which are the highest marks in the frame.
    endpoints = []
    for slot, (_, row) in enumerate(top.iterrows()):
        values = [row[f"p_t+{k}"] for k in horizons]
        ax.plot(horizons, values, color=SERIES_COLORS[slot], linewidth=2,
                marker="o", markersize=8)
        endpoints.append([values[-1], int(row["sequence_index"]), SERIES_COLORS[slot]])

    # Push apart labels whose endpoints are too close to read, working up
    # the log axis from the lowest.
    endpoints.sort(key=lambda e: e[0])
    min_ratio = 1.13
    for i in range(1, len(endpoints)):
        if endpoints[i][0] < endpoints[i - 1][0] * min_ratio:
            endpoints[i][0] = endpoints[i - 1][0] * min_ratio
    for label_y, sequence_index, color in endpoints:
        ax.annotate(
            f"seq {sequence_index}", (horizons[-1], label_y),
            textcoords="offset points", xytext=(10, 0), va="center",
            color=color, fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"t+{k}" for k in horizons])
    ax.set_xlim(0.9, K + 0.55)
    ax.set_xlabel("rollout horizon (10 s per step)", color=INK, fontsize=10)
    ax.set_ylabel("threat probability (log scale)", color=INK, fontsize=10)
    ax.set_title(
        "Highest model-risk-output trajectories in the host-level replay\n"
        "top 6 by max probability, over 400 sampled sequences (grey) — NOT labelled attacks",
        color=INK, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_TOP, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {FIG_TOP.relative_to(REPO_ROOT)}")


# =============================================================
# MAIN
# =============================================================


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Characterize the V5 rollout over the real host-level replay."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    X = np.load(X_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    if X.shape[0] != len(metadata):
        raise RuntimeError(
            f"X has {X.shape[0]} sequences but metadata has {len(metadata)} rows"
        )
    print(f"Loaded {X_PATH.relative_to(REPO_ROOT)}  shape={X.shape}")

    outputs = run_batched_inference(X, args.batch_size)
    P = outputs["probabilities"]

    summary = build_summary(metadata, P)
    distribution = build_distribution(P, summary)
    latent = latent_stability(outputs["latents"], outputs["current_latent"])
    _, top_state = state_behaviour(
        outputs["states"], outputs["states_scaled"],
        outputs["engine"].state_feature_names,
    )
    top_sequences(summary)
    make_figures(P, summary)

    payload = {
        "source_X": str(X_PATH.relative_to(REPO_ROOT)),
        "host": HOST,
        "n_sequences": int(X.shape[0]),
        "rollout_steps": K,
        "model_path": str(outputs["engine"].checkpoint_path.relative_to(REPO_ROOT)),
        "threshold_from_checkpoint": float(outputs["engine"].threshold),
        "batch_size": outputs["batch_size"],
        "inference_seconds": float(outputs["elapsed_s"]),
        "sequences_per_second": float(X.shape[0] / outputs["elapsed_s"]),
        "non_finite_output_values": outputs["non_finite"],
        "diagnostic_thresholds": list(DIAGNOSTIC_THRESHOLDS),
        "per_horizon": distribution["per_horizon"],
        "trajectory_behaviour": distribution["trajectory_behaviour"],
        "max_probability_distribution": distribution["max_probability_distribution"],
        "latent_stability": latent,
        "top_changing_state_features": top_state,
        "labels_available": False,
        "scope_shift_note": SCOPE_SHIFT_NOTE,
        "interpretation_caveat": NOT_A_BENCHMARK,
    }
    DISTRIBUTION_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  wrote {DISTRIBUTION_JSON.relative_to(REPO_ROOT)}")
    print(f"\n{SCOPE_SHIFT_NOTE}\n{NOT_A_BENCHMARK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
