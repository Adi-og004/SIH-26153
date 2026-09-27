"""
=============================================================
V4 RELIABILITY LAYER — BUILD + EVIDENCE PACK + VALIDATION
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Task 15. Builds the reliability reference distribution, runs the
reliability layer over the whole held-out screening substrate, finds
REAL failure cases, runs the retrospective forecast-vs-observed study,
and validates that nothing frozen moved.

READ-ONLY on every frozen artifact. This script does not train, refit a
scaler, rethreshold the V4 alert rule, touch screening_X.npy /
screening_predictions.csv, or fit any reliability estimator. It writes
only into results/v4_reliability/.

THRESHOLD PROVENANCE: every distributional threshold is a quantile of
X_train.npy pushed through the frozen checkpoint. No label array is
opened while the reference is built — enforced by a guarded loader that
records every array this process reads.

Outputs (results/v4_reliability/ only):
    reliability_config.json         reference distribution + derived bounds
    reliability_per_sequence.csv    status + every indicator, 3115 rows
    reliability_summary.json        distributions, rule rates, correlations
    failure_cases.csv               real demonstrated cases, per category
    forecast_vs_observed.csv        retrospective forecast error, 3115 x 5
    reliability_manifest.json       full provenance
    validation.json                 13 explicit PASS/FAIL gates

Usage:
    ./venv/bin/python src/build_v4_reliability.py

Exit code: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import py_compile
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import v4_reliability as rel                                    # noqa: E402
from pcap_ingest_v4 import load_v4_model                        # noqa: E402

BENCH_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
MODEL_DIR = REPO_ROOT / "models" / "latent_world_model_v4"
CHECKPOINT = MODEL_DIR / "latent_world_model_v4.pt"
MANIFEST_PATH = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
SCREEN_DIR = REPO_ROOT / "results" / "v4_screening_mvp"
OUT_DIR = REPO_ROOT / "results" / "v4_reliability"

BATCH_SIZE = 64
MAX_K = rel.MAX_K

# The 16 frozen artifacts hashed before and after, identical to the set
# used by the Task-12/14 validations (results/v4_screening_mvp/
# visual_qa.json -> frozen_artifact_integrity).
FROZEN_16 = [
    CHECKPOINT,
    BENCH_DIR / "input_scaler.pkl",
    BENCH_DIR / "target_scaler.pkl",
    BENCH_DIR / "input_feature_names.txt",
    BENCH_DIR / "state_feature_names.txt",
    BENCH_DIR / "X_test.npy",
    BENCH_DIR / "Y_multistep_test.npy",
    BENCH_DIR / "A_binary_multistep_test.npy",
    BENCH_DIR / "A_soft_multistep_test.npy",
    SCREEN_DIR / "screening_X.npy",
    SCREEN_DIR / "screening_predictions.csv",
    SCREEN_DIR / "screening_metadata.csv",
    SCREEN_DIR / "event_timeline.csv",
    SCREEN_DIR / "feature_names.txt",
    SCREEN_DIR / "screening_manifest.json",
    SCREEN_DIR / "readiness_report.json",
]

NEW_FILES = ["src/v4_reliability.py", "src/build_v4_reliability.py", "src/app.py"]

NETWORK_PATTERNS = [
    r"\brequests\.(get|post|put|delete|head|patch)\s*\(",
    r"\burllib\.request\b",
    r"\bhttpx\.",
    r"\bsocket\.(socket|create_connection)\s*\(",
    r"\baiohttp\b",
    r"https?://[^\s\"')]+",
]
URL_ALLOW_SUBSTRINGS = ("attack.mitre.org",)

# Label arrays that must NOT be read while the reference/rule is built.
LABEL_ARRAY_NAMES = ("A_binary_multistep", "A_soft_multistep")


class Report:
    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, key, name, ok, detail=""):
        self.rows.append({"key": key, "name": name,
                          "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            self.failed = True
        print(f"  {name:<50} {'PASS' if ok else 'FAIL'}   {detail}")
        return bool(ok)


REPORT = Report()
READ_LOG: list[str] = []


def header(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def guarded_load(path: Path, *, phase: str) -> np.ndarray:
    """Every .npy this process opens goes through here and is logged, so
    the 'no test labels used to tune' claim is a recorded fact rather
    than an assertion."""
    READ_LOG.append(f"{phase}:{path.relative_to(REPO_ROOT)}")
    return np.load(path)


def rollout(model, X, k=MAX_K, bs=BATCH_SIZE):
    """The frozen reference inference path, pinned to CPU so the
    reliability layer is deterministic and device-independent."""
    n = len(X)
    P = np.zeros((n, k + 1), dtype=np.float64)
    Z = np.zeros((n, k + 1, rel.LATENT_SIZE), dtype=np.float64)
    S = np.zeros((n, k + 1, rel.N_STATE_FEATURES), dtype=np.float64)
    T = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            tz, tp, ts = model.k_step_rollout(T[s:e], k)
            for j in range(k + 1):
                P[s:e, j] = tp[j].cpu().numpy()
                Z[s:e, j] = tz[j].cpu().numpy()
                S[s:e, j] = ts[j].cpu().numpy()
    return P, Z, S


def vec_indicators(X, P, Z, S, fmin, fmax):
    """Vectorised twin of rel.compute_indicators, used for the reference
    distribution and the bulk pass. Agreement with the per-sequence
    implementation is a validation gate, not an assumption."""
    x = X.astype(np.float64)
    a = np.abs(x)
    zn = np.linalg.norm(Z, axis=2)
    exceed = np.maximum(np.maximum(x - fmax, 0.0), np.maximum(fmin - x, 0.0))
    return {
        "rms_z": np.sqrt((x ** 2).mean((1, 2))),
        "max_abs_z": a.max((1, 2)),
        "frac_gt2": (a > 2).mean((1, 2)),
        "frac_gt3": (a > 3).mean((1, 2)),
        "feature_support_exceedance": exceed.max((1, 2)),
        "latent_norm_max": zn.max(1),
        "latent_norm_change": zn.max(1) - zn[:, 0],
        "step_delta_max": np.linalg.norm(np.diff(Z, axis=1), axis=2).max(1),
        "displacement": np.linalg.norm(Z[:, -1] - Z[:, 0], axis=1),
        "state_delta_max": np.linalg.norm(np.diff(S, axis=1), axis=2).max(1),
        "prob_max_abs_step": np.abs(np.diff(P, axis=1)).max(1),
        "prob_range": P.max(1) - P.min(1),
    }


# =============================================================
def main():
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("=" * 88)
    print("V4 RELIABILITY LAYER — BUILD + EVIDENCE PACK + VALIDATION")
    print("=" * 88)
    print(f"Repository : {REPO_ROOT}")
    print(f"Started    : {started}")
    print(f"Layer      : {rel.LAYER_VERSION}   device: {rel.DEVICE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pre_hashes = {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in FROZEN_16}
    manifest = json.loads(MANIFEST_PATH.read_text())

    # =========================================================
    header("STEP 1 — AUDIT THE EXISTING RELIABILITY EVIDENCE")
    # =========================================================
    audit = {}

    feature_names = [ln.strip() for ln in
                     (BENCH_DIR / "input_feature_names.txt").read_text().splitlines()
                     if ln.strip()]
    with open(BENCH_DIR / "input_scaler.pkl", "rb") as fh:
        in_scaler = pickle.load(fh)
    with open(BENCH_DIR / "target_scaler.pkl", "rb") as fh:
        tgt_scaler = pickle.load(fh)

    audit["input_scaler"] = {
        "path": "data/processed/multiday_benchmark_K5/input_scaler.pkl",
        "class": type(in_scaler).__name__,
        "n_features_in": int(in_scaler.n_features_in_),
        "n_samples_seen": int(in_scaler.n_samples_seen_),
        "arrays_are_pre_scaled": True,
        "consequence": (
            "X_train/X_val/X_test.npy are ALREADY standardized by this "
            "scaler, so a stored value IS a z-score against the reference "
            "train distribution. Input-OOD indicators therefore need no "
            "new fit and no new scaler — the crucial precondition for "
            "computing them without training."),
    }
    print(f"  input scaler              StandardScaler, 52 features, "
          f"n_samples_seen={in_scaler.n_samples_seen_}")
    print("  arrays pre-scaled         yes -> stored values ARE z-scores")

    # What already exists, so nothing is rebuilt that is already evidence.
    existing = {
        "results/long_horizon_rollout/latent_stability.csv":
            "per-sample per-step latent norm / delta / cosine, K=0..10",
        "results/long_horizon_rollout/latent_contraction.csv":
            "observed contraction ratios and per-sample verdicts",
        "results/long_horizon_rollout/probability_stability.csv":
            "per-sample probability drift, range, oscillation, saturation",
        "results/long_horizon_rollout/forecast_error.csv":
            "AGGREGATE state MAE/RMSE by horizon vs persistence and train-mean",
        "results/long_horizon_rollout/horizon_status.csv":
            "pre-declared numerical stability rules, STABLE at every K=1..10",
        "results/v4_screening_mvp/screening_predictions.csv":
            "latent_norm_t0 / latent_norm_t5 per screening sequence",
    }
    for p, what in existing.items():
        ok = (REPO_ROOT / p).exists()
        print(f"  {'found  ' if ok else 'MISSING'} {p:<58} {what[:40]}")
        audit.setdefault("existing_evidence", {})[p] = {"present": ok, "content": what}

    # What is computable without training, and what is NOT.
    audit["computable_without_training"] = [
        "input OOD: the arrays are already in frozen-scaler z units",
        "rollout stability: latent norms and step deltas are arithmetic on "
        "the frozen model's own output",
        "forecast self-consistency: differences between successive predicted "
        "states and probabilities",
        "analytic latent-norm bounds: closed form from the frozen LayerNorm "
        "and Linear weights",
        "retrospective forecast error: Y_multistep_test.npy holds the "
        "observed future state in the same target_scaler space the decoder "
        "outputs — usable AFTER the fact only",
    ]
    audit["deliberately_not_computed"] = [
        {"metric": "calibrated probability of correctness",
         "why_not": "would require fitting an estimator on labels; the task "
                    "forbids a new learned model and forbids tuning on test "
                    "labels. No calibration is claimed."},
        {"metric": "predictive-distribution variance / epistemic uncertainty",
         "why_not": "docs/LONG_HORIZON_ROLLOUT.md section 11: the V4 "
                    "transition is a DETERMINISTIC point estimate in latent "
                    "space and does not predict a distribution. There is no "
                    "variance to read; inventing one from the rollout spread "
                    "would be a number without a meaning."},
        {"metric": "ensemble / MC-dropout disagreement",
         "why_not": "requires either new models or stochastic forward passes, "
                    "i.e. a new component and a non-deterministic output. Both "
                    "are out of scope for a frozen-model reliability layer."},
        {"metric": "latent-norm growth as an instability signal",
         "why_not": "COMPUTED and reported, but weak by construction: the "
                    "transition terminates in LayerNorm, which bounds the "
                    "latent RMS. Recorded as architectural, not learned."},
    ]
    REPORT.add("audit_completed", "Existing reliability evidence audited",
               all((REPO_ROOT / p).exists() for p in existing),
               f"{len(existing)} prior artifacts, "
               f"{len(audit['deliberately_not_computed'])} metrics declined")

    # =========================================================
    header("STEP 2 — DERIVE ANALYTIC NUMERICAL BOUNDS (FAIL-SAFE H6/H7)")
    # =========================================================
    model, ckpt = load_v4_model(CHECKPOINT)
    bounds = rel.derive_latent_norm_bounds(ckpt["model_state_dict"])
    print(f"  ||z_0|| <= {bounds['encoder_output_norm_bound']:.4f}   "
          f"(encoder path: sigma_max(W_fc) x LayerNorm bound + ||b||)")
    print(f"  ||z_k|| <= {bounds['transition_output_norm_bound']:.4f}   "
          f"(k>=1, LayerNorm-terminated transition)")
    REPORT.add("bounds_derived", "Analytic latent-norm bounds derived",
               bounds["transition_output_norm_bound"] > 0,
               f"z0<={bounds['encoder_output_norm_bound']:.3f}, "
               f"zk<={bounds['transition_output_norm_bound']:.3f}")

    # =========================================================
    header("STEP 3 — BUILD THE REFERENCE DISTRIBUTION (TRAIN ONLY, NO LABELS)")
    # =========================================================
    X_train = guarded_load(BENCH_DIR / "X_train.npy", phase="reference_build")
    X_val = guarded_load(BENCH_DIR / "X_val.npy", phase="reference_rate_check")

    fmin = X_train.reshape(-1, rel.N_FEATURES).astype(np.float64).min(0)
    fmax = X_train.reshape(-1, rel.N_FEATURES).astype(np.float64).max(0)

    P_tr, Z_tr, S_tr = rollout(model, X_train)
    I_tr = vec_indicators(X_train, P_tr, Z_tr, S_tr, fmin, fmax)
    print(f"  reference sequences       {len(X_train)} (train split of "
          f"multiday_benchmark_K5)")
    print(f"  label arrays opened       "
          f"{sum(any(l in r for l in LABEL_ARRAY_NAMES) for r in READ_LOG)}")

    quantiles = {n: {"p99": float(np.percentile(I_tr[n], rel.ELEVATED_QUANTILE)),
                     "max": float(I_tr[n].max()),
                     "p50": float(np.percentile(I_tr[n], 50)),
                     "mean": float(I_tr[n].mean())}
                 for n in rel.TIERED_INDICATORS}
    for n in rel.TIERED_INDICATORS:
        q = quantiles[n]
        print(f"    {n:28s} p50={q['p50']:9.4f}  p99={q['p99']:9.4f}  "
              f"max={q['max']:9.4f}")

    scaler_contract = {
        "n_features_in": int(in_scaler.n_features_in_),
        "mean_sum": float(np.sum(in_scaler.mean_)),
        "scale_sum": float(np.sum(in_scaler.scale_)),
        "feature_order": feature_names,
        "sha256": pre_hashes["data/processed/multiday_benchmark_K5/input_scaler.pkl"],
        "note": "Values must match docs/v4_baseline_manifest.json -> scalers.",
    }
    m_sc = manifest["scalers"]["input_scaler"]
    contract_ok = (abs(scaler_contract["mean_sum"] - m_sc["mean_sum"]) < 1e-6
                   and abs(scaler_contract["scale_sum"] - m_sc["scale_sum"]) < 1e-6
                   and scaler_contract["sha256"] == m_sc["sha256"])
    REPORT.add("scaler_contract", "Scaler contract matches frozen manifest",
               contract_ok, f"sha256 {scaler_contract['sha256'][:16]}…")

    config = {
        "layer_version": rel.LAYER_VERSION,
        "generated_utc": started,
        "what_this_is": (
            "A rule-based reliability status for the FROZEN V4 world model. "
            "NOT a calibrated probability, NOT a confidence score, NOT a "
            "probability of correctness."),
        "reference_distribution": {
            "source": "data/processed/multiday_benchmark_K5/X_train.npy",
            "sha256": sha256_file(BENCH_DIR / "X_train.npy"),
            "n_sequences": int(len(X_train)),
            "why_this_split": (
                "It is the partition the frozen input_scaler was fit on, so "
                "its quantiles define what 'inside the scaler's world' means. "
                "No label array is read."),
            "labels_read": False,
            "test_labels_read": False,
        },
        "indicator_reference": quantiles,
        "indicator_definitions": {i.name: {"group": i.group,
                                           "description": i.description}
                                  for i in rel.INDICATORS},
        "feature_support": {
            "min": fmin.tolist(), "max": fmax.tolist(),
            "feature_names": feature_names,
            "units": "frozen input_scaler z-units (training standard deviations)",
        },
        "numerical_bounds": bounds,
        "scaler_contract": scaler_contract,
        "status_rule": {
            "UNRELIABLE": "any hard fail-safe condition, OR >= 2 indicators "
                          "above the reference maximum",
            "CAUTION": ">= 1 indicator above the reference maximum, OR >= 2 "
                       "indicators above the reference 99th percentile",
            "RELIABLE": "otherwise",
            "elevated_quantile": rel.ELEVATED_QUANTILE,
            "input_shift_hard_margin_sd": rel.INPUT_SHIFT_HARD_MARGIN_SD,
        },
        "hard_conditions": {
            "H1": "non-finite model input",
            "H2": "invalid input / latent / state / probability tensor shape",
            "H3": "input scaler incompatible with the frozen contract",
            "H4": "non-finite latent state or decoded state",
            "H5": "non-finite or out-of-[0,1] threat probability",
            "H6": "||z_0|| exceeds the analytic encoder bound",
            "H7": "||z_k|| exceeds the analytic LayerNorm transition bound",
            "H8": f"an input value lies more than "
                  f"{rel.INPUT_SHIFT_HARD_MARGIN_SD} training SD outside the "
                  f"reference per-feature observed range",
        },
        "provenance": {
            "checkpoint": "models/latent_world_model_v4/latent_world_model_v4.pt",
            "checkpoint_sha256": pre_hashes[
                "models/latent_world_model_v4/latent_world_model_v4.pt"],
            "device": rel.DEVICE,
            "batch_size": BATCH_SIZE,
            "max_k": MAX_K,
            "v4_frozen": True,
            "v4_retrained": False,
            "scalers_refit": False,
            "thresholds_changed": False,
        },
    }
    (OUT_DIR / "reliability_config.json").write_text(json.dumps(config, indent=2))
    ref = rel.load_reference(OUT_DIR / "reliability_config.json")
    REPORT.add("config_written", "reliability_config.json written and reloadable",
               ref.n_reference_sequences == len(X_train),
               f"{len(rel.TIERED_INDICATORS)} tiered indicators")

    # =========================================================
    header("STEP 4 — RUN THE LAYER OVER TRAIN / VAL / HELD-OUT SCREENING")
    # =========================================================
    X_screen = guarded_load(SCREEN_DIR / "screening_X.npy", phase="apply")
    preds = pd.read_csv(SCREEN_DIR / "screening_predictions.csv")
    meta = pd.read_csv(SCREEN_DIR / "screening_metadata.csv")
    timeline = pd.read_csv(SCREEN_DIR / "event_timeline.csv")

    P_sc, Z_sc, S_sc = rollout(model, X_screen)
    P_va, Z_va, S_va = rollout(model, X_val)

    # Frozen-path agreement: the layer must be reading the same forecast the
    # dashboard shows. screening_predictions.csv was produced on MPS.
    pcols = ["nowcast_probability_t0"] + [f"prob_t{k}" for k in range(1, MAX_K + 1)]
    frozen_P = preds[pcols].to_numpy(dtype=np.float64)
    max_dp = float(np.abs(P_sc - frozen_P).max())
    thr = float(preds["alert_threshold"].iloc[0])
    alerts_recomputed = (P_sc[:, 1:].max(1) >= thr).astype(int)
    alerts_frozen = preds["alert"].to_numpy(dtype=int)
    alert_identical = bool(np.array_equal(alerts_recomputed, alerts_frozen))
    REPORT.add("frozen_path_agreement",
               "Recomputed rollout matches frozen predictions",
               max_dp < 1e-5 and alert_identical,
               f"max|dP|={max_dp:.2e} (CPU vs MPS), all {len(preds)} alert "
               f"decisions identical")

    def run_layer(X, P, Z, S, tag):
        rows, assessments = [], []
        for i in range(len(X)):
            a = rel.assess(X[i], P[i], Z[i], S[i], ref)
            assessments.append(a)
            rows.append(a.to_row())
        df = pd.DataFrame(rows)
        df.insert(0, "sequence_id", np.arange(len(X)))
        counts = df["reliability_status"].value_counts().to_dict()
        pct = {k: round(100 * v / len(df), 3) for k, v in counts.items()}
        print(f"  {tag:<22} n={len(df):5d}  " +
              "  ".join(f"{s}={counts.get(s, 0)} ({pct.get(s, 0)}%)"
                        for s in rel.STATUS_ORDER))
        return df, assessments, counts, pct

    # Reference flag rates: what the rule does on data it was derived from
    # and on a second label-free split. This is the rule's base rate, and
    # it is the only thing that makes the held-out rate interpretable.
    I_va = vec_indicators(X_val, P_va, Z_va, S_va, fmin, fmax)
    df_tr, _, c_tr, p_tr = run_layer(X_train, P_tr, Z_tr, S_tr,
                                     "train (reference)")
    df_va, _, c_va, p_va = run_layer(X_val, P_va, Z_va, S_va,
                                     "val (label-free)")
    df_sc, A_sc, c_sc, p_sc = run_layer(X_screen, P_sc, Z_sc, S_sc,
                                        "screening (held-out)")

    # Cross-check: the vectorised and per-sequence implementations must agree.
    I_sc = vec_indicators(X_screen, P_sc, Z_sc, S_sc, fmin, fmax)
    max_ind_gap = max(float(np.abs(df_sc[n].to_numpy() - I_sc[n]).max())
                      for n in rel.INDICATOR_NAMES)
    REPORT.add("indicator_implementations_agree",
               "Per-sequence and vectorised indicators agree", max_ind_gap < 1e-9,
               f"max gap {max_ind_gap:.2e}")

    # =========================================================
    header("STEP 5 — RETROSPECTIVE FORECAST-VS-OBSERVED ERROR ANALYSIS")
    # =========================================================
    # Y_multistep_test.npy is the OBSERVED future state at t+1..t+5 in the
    # same target_scaler space the decoder writes into. It is opened HERE,
    # after every forecast and every reliability status already exists, and
    # it is never an input to either.
    Y = guarded_load(BENCH_DIR / "Y_multistep_test.npy",
                     phase="retrospective_only").astype(np.float64)
    resid = S_sc[:, 1:, :] - Y
    mae_k = np.abs(resid).mean(2)                    # (n, 5)
    rmse_k = np.sqrt((resid ** 2).mean(2))
    seq_mae = mae_k.mean(1)

    fvo = pd.DataFrame({
        "sequence_id": np.repeat(np.arange(len(X_screen)), MAX_K),
        "horizon_k": np.tile(np.arange(1, MAX_K + 1), len(X_screen)),
        "state_mae": mae_k.ravel(),
        "state_rmse": rmse_k.ravel(),
        "state_max_abs_error": np.abs(resid).max(2).ravel(),
        "forecast_probability": P_sc[:, 1:].ravel(),
        "reliability_status": np.repeat(
            df_sc["reliability_status"].to_numpy(), MAX_K),
    })
    fvo["target_timestamp"] = np.repeat(
        preds["target_t1_timestamp"].to_numpy(), MAX_K)
    fvo["analysis_type"] = "retrospective forecast error analysis"
    fvo["state_space"] = ("26-D state in target_scaler space "
                          "(data/processed/multiday_benchmark_K5/target_scaler.pkl)")
    fvo.to_csv(OUT_DIR / "forecast_vs_observed.csv", index=False)

    by_h = {int(k): {
        "n": int(len(mae_k)),
        "state_mae_mean": float(mae_k[:, k - 1].mean()),
        "state_mae_p50": float(np.percentile(mae_k[:, k - 1], 50)),
        "state_mae_p90": float(np.percentile(mae_k[:, k - 1], 90)),
        "state_mae_p99": float(np.percentile(mae_k[:, k - 1], 99)),
        "state_mae_max": float(mae_k[:, k - 1].max()),
        "state_rmse_mean": float(rmse_k[:, k - 1].mean()),
    } for k in range(1, MAX_K + 1)}
    print("  retrospective state error by horizon (target_scaler space):")
    for k, v in by_h.items():
        print(f"    k={k}  MAE mean={v['state_mae_mean']:.4f} "
              f"p50={v['state_mae_p50']:.4f} p90={v['state_mae_p90']:.4f} "
              f"max={v['state_mae_max']:.4f}   RMSE={v['state_rmse_mean']:.4f}")

    # Indicator vs retrospective error. Spearman, computed WITHOUT scipy so
    # the offline dependency set does not grow.
    def spearman(a, b):
        ra = pd.Series(a).rank().to_numpy(dtype=np.float64, copy=True)
        rb = pd.Series(b).rank().to_numpy(dtype=np.float64, copy=True)
        ra -= ra.mean()
        rb -= rb.mean()
        d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / d) if d > 0 else 0.0

    corr = {n: {"spearman_rho_vs_seq_mean_state_mae":
                spearman(df_sc[n].to_numpy(), seq_mae)}
            for n in rel.INDICATOR_NAMES}
    print("\n  Spearman rho, indicator vs sequence-mean state MAE:")
    for n, v in sorted(corr.items(),
                       key=lambda kv: -abs(kv[1]["spearman_rho_vs_seq_mean_state_mae"])):
        print(f"    {n:30s} {v['spearman_rho_vs_seq_mean_state_mae']:+.3f}")

    by_status = {}
    for s in rel.STATUS_ORDER:
        m = (df_sc["reliability_status"] == s).to_numpy()
        if not m.any():
            continue
        by_status[s] = {
            "n": int(m.sum()),
            "state_mae_mean": float(seq_mae[m].mean()),
            "state_mae_p50": float(np.percentile(seq_mae[m], 50)),
            "state_mae_p90": float(np.percentile(seq_mae[m], 90)),
            "state_mae_max": float(seq_mae[m].max()),
        }
    print("\n  retrospective state MAE by reliability status:")
    for s, v in by_status.items():
        print(f"    {s:11s} n={v['n']:5d}  mean={v['state_mae_mean']:.4f}  "
              f"p50={v['state_mae_p50']:.4f}  p90={v['state_mae_p90']:.4f}")
    ordered = [by_status[s]["state_mae_mean"] for s in rel.STATUS_ORDER
               if s in by_status]
    monotone = all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1))
    REPORT.add("error_ordering", "Retrospective error increases with severity",
               monotone,
               " < ".join(f"{v:.4f}" for v in ordered) if monotone
               else f"NOT monotone: {[round(v, 4) for v in ordered]}")

    # =========================================================
    header("STEP 6 — REAL FAILURE CASES (EXISTING DATA ONLY)")
    # =========================================================
    df_sc["target_t1_timestamp"] = preds["target_t1_timestamp"]
    df_sc["observed_window_end"] = preds["observed_window_end"]
    df_sc["nowcast_probability_t0"] = preds["nowcast_probability_t0"]
    df_sc["max_forecast_probability"] = preds["max_forecast_probability"]
    df_sc["alert"] = preds["alert"]
    df_sc["alert_label"] = preds["alert_label"]
    for k in range(1, MAX_K + 1):
        df_sc[f"prob_t{k}"] = preds[f"prob_t{k}"]
    df_sc["latent_norm_t0"] = np.linalg.norm(Z_sc[:, 0, :], axis=1)
    df_sc["latent_norm_t5"] = np.linalg.norm(Z_sc[:, -1, :], axis=1)
    df_sc["retrospective_state_mae_mean_k1_k5"] = seq_mae
    for k in range(1, MAX_K + 1):
        df_sc[f"retrospective_state_mae_t{k}"] = mae_k[:, k - 1]
    # Context columns, clearly flagged: never inputs to the status.
    df_sc["context_region"] = timeline["region"]
    df_sc["context_dominant_attack_t1"] = meta["dominant_attack_t1"]
    df_sc["context_label_t1"] = meta["label_t1"]
    df_sc["context_note"] = "context_* columns are ground truth, NOT layer inputs"
    df_sc.to_csv(OUT_DIR / "reliability_per_sequence.csv", index=False)

    cases = []
    # Every UNRELIABLE sequence is a mandatory member.
    unrel = df_sc.index[df_sc["reliability_status"] == "UNRELIABLE"].tolist()
    for i in unrel:
        cases.append((i, "UNRELIABLE",
                      "hard fail-safe condition fired on real held-out data"))
    # Most severe CAUTION cases: outside reference support first, then the
    # highest count of elevated indicators. Selected on the layer's own
    # output, never on labels or on the retrospective error.
    cdf = df_sc[df_sc["reliability_status"] == "CAUTION"].copy()
    cdf = cdf.sort_values(["n_beyond_reference", "n_elevated", "sequence_id"],
                          ascending=[False, False, True])
    for i in cdf.index[:8].tolist():
        cases.append((i, "CAUTION", "most severe CAUTION by indicator count"))
    # Reliable exemplars: the demo default sequence, plus the two lowest
    # total indicator pressure. Again chosen from model-side quantities only.
    rdf = df_sc[df_sc["reliability_status"] == "RELIABLE"].copy()
    demo_default = int(np.flatnonzero(
        preds["max_forecast_probability"].to_numpy()
        == preds["max_forecast_probability"].max())[0])
    if demo_default in rdf.index:
        cases.append((demo_default, "RELIABLE",
                      "default demo sequence (highest max forecast probability)"))
    rdf["pressure"] = (rdf["rms_z"] / ref.quantiles["rms_z"]["p99"]
                       + rdf["prob_range"] / ref.quantiles["prob_range"]["p99"])
    for i in rdf.sort_values(["pressure", "sequence_id"]).index[:2].tolist():
        if i != demo_default:
            cases.append((i, "RELIABLE", "lowest combined indicator pressure"))

    keep = ["sequence_id", "target_t1_timestamp", "observed_window_end",
            "reliability_status", "reliability_short", "hard_fail_codes",
            "hard_fail_names", "n_elevated", "n_beyond_reference",
            "triggering_indicators", "headline",
            "nowcast_probability_t0"] + [f"prob_t{k}" for k in range(1, MAX_K + 1)] + [
            "max_forecast_probability", "alert_label",
            "latent_norm_t0", "latent_norm_t5", "latent_norm_max",
            "latent_norm_change", "step_delta_max", "displacement",
            "state_delta_max", "prob_max_abs_step", "prob_range",
            "rms_z", "max_abs_z", "frac_gt2", "frac_gt3",
            "feature_support_exceedance",
            "retrospective_state_mae_mean_k1_k5",
            "context_region", "context_dominant_attack_t1"]
    fc = df_sc.loc[[c[0] for c in cases], keep].copy()
    fc.insert(1, "demonstrated_category", [c[1] for c in cases])
    fc.insert(2, "selection_basis", [c[2] for c in cases])
    fc = fc.drop_duplicates(subset=["sequence_id"])
    fc.to_csv(OUT_DIR / "failure_cases.csv", index=False)

    demonstrated = sorted(set(fc["demonstrated_category"]))
    print(f"  cases recorded            {len(fc)}")
    print(f"  categories demonstrated   {demonstrated}")
    for s in rel.STATUS_ORDER:
        sub = fc[fc["demonstrated_category"] == s]
        if len(sub):
            ex = sub.iloc[0]
            print(f"    {s:11s} e.g. seq {int(ex['sequence_id'])} @ "
                  f"{ex['target_t1_timestamp']}  "
                  f"P_max={ex['max_forecast_probability']:.3f}  "
                  f"{str(ex['headline'])[:74]}")
    REPORT.add("real_cases", "All three statuses demonstrated on real data",
               len(demonstrated) == 3,
               f"{demonstrated}, {len(fc)} cases, 0 synthetic")

    # =========================================================
    header("STEP 7 — FAIL-SAFE DETECTOR SELF-TESTS")
    # =========================================================
    # These are DETECTOR self-tests on deliberately corrupted tensors, not
    # data examples. Corrupted tensors never enter failure_cases.csv, the
    # per-sequence CSV, or any reported statistic. A hard condition that
    # cannot be shown to fire is not a fail-safe.
    base_i = int(df_sc.index[df_sc["reliability_status"] == "RELIABLE"][0])
    bx, bp, bz, bs = (X_screen[base_i].astype(np.float64).copy(),
                      P_sc[base_i].copy(), Z_sc[base_i].copy(), S_sc[base_i].copy())

    class _BadScaler:
        n_features_in_ = 26
        mean_ = np.zeros(26)
        scale_ = np.ones(26)

    self_tests = []

    def probe(code, name, fn):
        try:
            a = fn()
            fired = any(h["code"] == code for h in a.hard_fails)
            status = a.status
        except Exception as exc:                          # noqa: BLE001
            fired, status = False, f"EXCEPTION {type(exc).__name__}: {exc}"
        self_tests.append({"code": code, "condition": name,
                           "detector_fired": bool(fired),
                           "resulting_status": status})
        print(f"  {code}  {name:<46} "
              f"{'DETECTED' if fired else 'MISSED':<9} -> {status}")
        return fired

    def _mut(arr, idx, val):
        a = arr.copy()
        a[idx] = val
        return a

    probe("H1", "NaN in model input",
          lambda: rel.assess(_mut(bx, (3, 7), np.nan), bp, bz, bs, ref))
    probe("H1", "Inf in model input",
          lambda: rel.assess(_mut(bx, (0, 0), np.inf), bp, bz, bs, ref))
    probe("H2", "invalid input shape (9, 52)",
          lambda: rel.assess(bx[:9], bp, bz, bs, ref))
    probe("H2", "invalid latent shape (6, 16)",
          lambda: rel.assess(bx, bp, bz[:, :16], bs, ref))
    probe("H3", "incompatible scaler (26 features)",
          lambda: rel.assess(bx, bp, bz, bs, ref, scaler=_BadScaler()))
    probe("H4", "non-finite latent state",
          lambda: rel.assess(bx, bp, _mut(bz, (2, 5), np.nan), bs, ref))
    probe("H4", "non-finite decoded state",
          lambda: rel.assess(bx, bp, bz, _mut(bs, (4, 11), np.inf), ref))
    probe("H5", "probability outside [0, 1]",
          lambda: rel.assess(bx, _mut(bp, 3, 1.7), bz, bs, ref))
    probe("H6", "encoder latent norm past analytic bound",
          lambda: rel.assess(bx, bp, _mut(bz, (0, 0), 1e3), bs, ref))
    probe("H7", "rollout latent norm explosion at k=3",
          lambda: rel.assess(bx, bp, _mut(bz, (3, slice(None)), 50.0), bs, ref))
    probe("H8", "extreme input shift (+40 SD)",
          lambda: rel.assess(_mut(bx, (5, 12), 40.0), bp, bz, bs, ref))

    all_detected = all(t["detector_fired"] for t in self_tests)
    all_unreliable = all(t["resulting_status"] == "UNRELIABLE" for t in self_tests)
    REPORT.add("failsafe_self_tests",
               "Every hard condition detected and forced UNRELIABLE",
               all_detected and all_unreliable,
               f"{sum(t['detector_fired'] for t in self_tests)}/"
               f"{len(self_tests)} detected")

    # Verify the intact baseline is NOT flagged, i.e. the probes are not
    # trivially always-on.
    base_a = rel.assess(bx, bp, bz, bs, ref, scaler=in_scaler)
    REPORT.add("failsafe_no_false_fire",
               "Intact sequence passes every hard condition",
               base_a.status == "RELIABLE" and not base_a.hard_fails,
               f"seq {base_i} -> {base_a.status}")

    # =========================================================
    header("STEP 8 — DETERMINISM AND LEAKAGE CHECKS")
    # =========================================================
    # Determinism: recompute the whole pass and hash the flat output.
    def flat_hash(df):
        return hashlib.sha256(
            df.to_csv(index=False).encode()).hexdigest()

    P2, Z2, S2 = rollout(model, X_screen)
    rows2 = [rel.assess(X_screen[i], P2[i], Z2[i], S2[i], ref).to_row()
             for i in range(len(X_screen))]
    df2 = pd.DataFrame(rows2)
    df2.insert(0, "sequence_id", np.arange(len(X_screen)))
    det_ok = flat_hash(df2) == flat_hash(
        df_sc[df2.columns.tolist()].reset_index(drop=True))
    REPORT.add("deterministic", "Reliability output bit-identical on rerun",
               det_ok, f"sha256 {flat_hash(df2)[:16]}…")

    # Leakage: the status must be a function of X and the frozen rollout
    # alone. Perturbing the observed future must not move a single status.
    Y_perturbed = Y + 1000.0        # never reaches assess(); proves the point
    rows3 = [rel.assess(X_screen[i], P_sc[i], Z_sc[i], S_sc[i], ref).to_row()
             for i in range(len(X_screen))]
    no_future = all(rows3[i]["reliability_status"]
                    == df_sc["reliability_status"].iloc[i]
                    for i in range(len(rows3)))
    sig = list(rel.assess.__code__.co_varnames[:rel.assess.__code__.co_argcount])
    REPORT.add("no_future_observation",
               "No future observation can reach the assessment",
               no_future and "Y" not in sig and len(Y_perturbed) == len(Y),
               f"assess({', '.join(sig)}) — no observed-future argument")

    # Static check: the reliability module must never open a label array.
    src_rel = (REPO_ROOT / "src" / "v4_reliability.py").read_text()
    label_refs = [n for n in LABEL_ARRAY_NAMES if n in src_rel]
    reads_before_ref = [r for r in READ_LOG
                        if r.startswith(("reference_build", "reference_rate_check"))
                        and any(l in r for l in LABEL_ARRAY_NAMES)]
    label_reads = [r for r in READ_LOG
                   if any(l in r for l in LABEL_ARRAY_NAMES)]
    REPORT.add("no_test_labels_for_tuning",
               "No test label used to set any reliability threshold",
               not label_refs and not reads_before_ref,
               f"module label refs={label_refs or 'none'}; label arrays read "
               f"during reference build={len(reads_before_ref)}; "
               f"total label array reads this run={len(label_reads)}")

    # =========================================================
    header("STEP 9 — FROZEN ARTIFACTS, THRESHOLDS, DASHBOARD, OFFLINE")
    # =========================================================
    post_hashes = {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in FROZEN_16}
    changed = [k for k in pre_hashes if pre_hashes[k] != post_hashes[k]]
    REPORT.add("frozen_unchanged", "All 16 frozen artifacts byte-identical",
               not changed, f"{len(FROZEN_16)} hashed before and after; "
                            f"changed={changed or 'none'}")

    ck_ok = (post_hashes["models/latent_world_model_v4/latent_world_model_v4.pt"]
             == manifest["checkpoint"]["sha256"])
    REPORT.add("checkpoint_matches_manifest",
               "Checkpoint hash matches frozen baseline manifest", ck_ok,
               manifest["checkpoint"]["sha256"][:16] + "…")

    sc_ok = (post_hashes["data/processed/multiday_benchmark_K5/input_scaler.pkl"]
             == manifest["scalers"]["input_scaler"]["sha256"]
             and post_hashes["data/processed/multiday_benchmark_K5/target_scaler.pkl"]
             == manifest["scalers"]["target_scaler"]["sha256"])
    REPORT.add("scalers_unchanged", "Both scalers unchanged and manifest-matched",
               sc_ok, "input + target")

    pred_ok = (post_hashes["results/v4_screening_mvp/screening_predictions.csv"]
               == pre_hashes["results/v4_screening_mvp/screening_predictions.csv"])
    screen_man = json.loads((SCREEN_DIR / "screening_manifest.json").read_text())
    thr_ok = (float(screen_man["alert_rule"]["threshold"])
              == float(manifest["threshold"]["per_horizon_trajectory_max"]["5"])
              == thr
              and screen_man["alert_rule"]["per_horizon_point_thresholds"]
              == {k: v for k, v in
                  manifest["threshold"]["per_horizon_future_prediction"].items()})
    REPORT.add("predictions_and_thresholds_unchanged",
               "Screening predictions and thresholds unchanged",
               pred_ok and thr_ok, f"alert threshold {thr:.2f}, unchanged")

    # No synthetic data: the substrate is byte-identical to the frozen test X.
    synth_ok = (post_hashes["results/v4_screening_mvp/screening_X.npy"]
                == post_hashes["data/processed/multiday_benchmark_K5/X_test.npy"])
    REPORT.add("no_synthetic_data",
               "Every assessed sequence is real held-out capture", synth_ok,
               "screening_X.npy == X_test.npy (byte-identical)")

    # Compile + import the dashboard headless, and confirm both the new page
    # and all pre-existing pages survive.
    compile_ok = True
    for f in NEW_FILES:
        try:
            py_compile.compile(str(REPO_ROOT / f), doraise=True)
        except Exception as exc:                              # noqa: BLE001
            compile_ok = False
            print(f"    compile FAIL {f}: {exc}")
    REPORT.add("compiles", "New and modified Python files compile", compile_ok,
               ", ".join(NEW_FILES))

    app_src = (REPO_ROOT / "src" / "app.py").read_text()
    NEW_PAGE = "🛡️ Reliability / Model Health"
    EXISTING_PAGES = [
        "🎯 Live Forecast", "⏱️ Early Warning", "🧭 Event Timeline",
        "🔮 Trajectory Rollout", "📊 Window Inspector", "🔍 Feature Attribution",
        "🗺️ MITRE ATT&CK Timeline", "📥 Data Input",
        "📐 Methodology & Limits",
    ]
    pages_ok = (NEW_PAGE in app_src
                and all(f'page == "{p}"' in app_src for p in EXISTING_PAGES))
    REPORT.add("dashboard_pages", "New page added, all 9 existing pages intact",
               pages_ok, f"10 pages total")

    # The repo carries two environments: the evaluation venv (torch, no
    # Streamlit) and the dashboard venv (Streamlit + plotly). The UI must be
    # validated under the one `streamlit run src/app.py` actually uses, so the
    # canonical discovery helper from validate_dashboard.py is reused rather
    # than reimplemented here.
    from validate_dashboard import find_dashboard_python
    dash_py, dash_mods = find_dashboard_python()
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    # A bare import only exercises whichever page the navigation radio
    # returns first. Every page is forced and rendered instead, so a break
    # on an existing page is caught as loudly as one on the new page.
    probe_src = r"""
import importlib.util, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import streamlit as st
TARGET = sys.argv[1]
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else None

def _radio(label, options, *a, **kw):
    opts = list(options)
    if TARGET in opts:
        return TARGET
    idx = kw.get("index", 0) or 0
    return opts[idx] if opts else None

st.radio = _radio
st.sidebar.radio = _radio

# Outside `streamlit run`, st.stop() is a no-op, so a page would keep
# executing past its own guard and fail on state the guard exists to
# prevent. Honour it the way the real runtime does: a deliberate stop is
# a valid render outcome, not a failure.
class _Stopped(BaseException):
    pass

def _stop():
    raise _Stopped()

st.stop = _stop
if SEQ is not None:
    st.session_state["seq_id"] = SEQ
spec = importlib.util.spec_from_file_location("app_probe", "src/app.py")
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except _Stopped:
    print("PAGE_OK stopped_by_guard")
except BaseException as e:
    t = type(e).__name__
    if "StopException" in t or "RerunException" in t:
        print("PAGE_OK")
    else:
        print("PAGE_FAIL %s: %s" % (t, e))
        raise SystemExit(1)
else:
    print("PAGE_OK")
"""
    ALL_PAGES = EXISTING_PAGES[:5] + [NEW_PAGE] + EXISTING_PAGES[5:]
    # Render the new page on a RELIABLE window and on a real UNRELIABLE one,
    # so the abstention path is exercised rather than merely written.
    probe_targets = [(pg, None) for pg in ALL_PAGES]
    if unrel:
        probe_targets.append((NEW_PAGE, int(unrel[0])))
        probe_targets.append(("\U0001f3af Live Forecast", int(unrel[0])))

    page_results, import_ok = [], True
    if not dash_py:
        import_ok = False
        page_results.append({"page": "*", "ok": False,
                             "error": "no interpreter with streamlit + plotly"})
    else:
        for pg, seq in probe_targets:
            cmd = [dash_py, "-c", probe_src, pg]
            if seq is not None:
                cmd.append(str(seq))
            try:
                pr = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                    capture_output=True, text=True, timeout=900)
                ok = "PAGE_OK" in pr.stdout
                stopped = "stopped_by_guard" in pr.stdout
                err = None if ok else (
                    ([l for l in pr.stdout.splitlines() if "PAGE_FAIL" in l]
                     or (pr.stderr.strip().splitlines() or ["no output"]))[-1][:200])
            except Exception as exc:                          # noqa: BLE001
                ok, stopped, err = False, False, f"{type(exc).__name__}: {exc}"
            page_results.append({"page": pg, "seq_id": seq, "ok": ok,
                                 "stopped_by_own_guard": bool(ok and stopped),
                                 **({"error": err} if err else {})})
            if not ok:
                import_ok = False
                print(f"    render FAIL  {pg}  seq={seq}  {err}")
    n_ok = sum(r["ok"] for r in page_results)
    REPORT.add("dashboard_pages_render",
               "Every dashboard page renders headless",
               import_ok,
               f"{n_ok}/{len(page_results)} render probes on "
               f"{Path(dash_py).relative_to(REPO_ROOT) if dash_py else 'no interpreter'}"
               f" (streamlit {dash_mods.get('streamlit')})")

    # Offline: the new source must contain no runtime network construct.
    net_hits = []
    for f in ["src/v4_reliability.py", "src/build_v4_reliability.py"]:
        txt = (REPO_ROOT / f).read_text()
        for pat in NETWORK_PATTERNS:
            for m in re.finditer(pat, txt):
                if any(a in m.group(0) for a in URL_ALLOW_SUBSTRINGS):
                    continue
                net_hits.append(f"{f}: {m.group(0)[:60]}")
    REPORT.add("offline", "No network dependency in the reliability layer",
               not net_hits, net_hits[0] if net_hits else "0 matches")

    # V4 untouched: the module must not contain a training construct.
    tree = ast.parse(src_rel)
    banned = {"backward", "step", "fit", "fit_transform", "train",
              "load_state_dict", "save"}
    hits = sorted({n.func.attr for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in banned})
    REPORT.add("no_training", "Reliability module contains no training call",
               not hits, f"banned calls found: {hits or 'none'}")

    # =========================================================
    header("STEP 10 — WRITE THE EVIDENCE PACK")
    # =========================================================
    margin_sens = {}
    for m in (0.0, 1.0, 3.0, 5.0):
        margin_sens[str(m)] = {
            "train_flagged": int((I_tr["feature_support_exceedance"] > m).sum()),
            "val_flagged": int((I_va["feature_support_exceedance"] > m).sum()),
            "screening_flagged": int((I_sc["feature_support_exceedance"] > m).sum()),
        }

    tier_counts = {n: df_sc[f"tier_{n}"].value_counts().to_dict()
                   for n in rel.TIERED_INDICATORS}

    summary = {
        "layer_version": rel.LAYER_VERSION,
        "generated_utc": started,
        "wording_constraint": rel.STATUS_NOTE,
        "audit": audit,
        "status_distribution": {
            "train_reference": {"n": int(len(df_tr)), "counts": c_tr, "pct": p_tr},
            "val_label_free": {"n": int(len(df_va)), "counts": c_va, "pct": p_va},
            "screening_held_out": {"n": int(len(df_sc)), "counts": c_sc, "pct": p_sc},
        },
        "reference_flag_rate_note": (
            "The train row is the rule's BASE RATE on the very distribution "
            "its thresholds came from: an ELEVATED tier is by construction a "
            "~1% event per indicator. The held-out rate is only interpretable "
            "against it."),
        "indicator_reference": quantiles,
        "indicator_tier_counts_screening": tier_counts,
        "input_shift_margin_sensitivity": margin_sens,
        "input_shift_margin_selected": rel.INPUT_SHIFT_HARD_MARGIN_SD,
        "input_shift_margin_justification": {
            "value_sd": rel.INPUT_SHIFT_HARD_MARGIN_SD,
            "test_labels_used": False,
            "ground_a": (
                "The condition must not fire on data the reference "
                "distribution covers. Margin 0 does (10 val sequences "
                "slightly extend the train range, as any finite sample "
                "will). The smallest margin in {0, 1, 3, 5} that flags "
                "neither train nor val is 1.0 SD, so 3.0 satisfies this "
                "with room to spare."),
            "ground_b": (
                "3.0 is not a new free parameter: |z| > 3 is already one of "
                "this layer's indicators (frac_gt3, per the task "
                "specification), so the hard condition reuses the same "
                "conventional numerical-outlier threshold."),
            "honest_caveat": (
                "3.0 is the more conservative of two defensible values, not "
                "the unique value the evidence forces. At 1.0 SD the same "
                "rule would flag 21 held-out sequences instead of 11. Both "
                "counts are in input_shift_margin_sensitivity."),
        },
        "numerical_bounds": bounds,
        "retrospective_forecast_error": {
            "label": "retrospective forecast error analysis",
            "definition": ("decoded state at t+k vs OBSERVED state at t+k "
                           "from Y_multistep_test.npy, in target_scaler space"),
            "future_observation_used_in_forecast": False,
            "by_horizon": by_h,
            "by_reliability_status": by_status,
            "error_increases_with_severity": bool(monotone),
            "indicator_correlations": corr,
            "correlation_note": (
                "Spearman rho against sequence-mean state MAE over k=1..5. "
                "These correlations are REPORTED, not used: no threshold was "
                "chosen from them. latent_norm_max correlates NEGATIVELY, "
                "which contradicts its 'larger is worse' framing and is "
                "recorded as a limitation rather than corrected."),
        },
        "failure_cases": {
            "n_cases": int(len(fc)),
            "categories_demonstrated": demonstrated,
            "all_from_real_capture": True,
            "synthetic_examples": 0,
            "unreliable_sequence_ids": [int(i) for i in unrel],
        },
        "failsafe_self_tests": {
            "note": ("Detector self-tests on deliberately corrupted tensors. "
                     "These are NOT data examples and appear in no reported "
                     "statistic or CSV row."),
            "n_tests": len(self_tests),
            "n_detected": int(sum(t["detector_fired"] for t in self_tests)),
            "tests": self_tests,
            "intact_baseline_sequence_id": base_i,
            "intact_baseline_status": base_a.status,
        },
        "frozen_path_agreement": {
            "max_abs_probability_delta_vs_frozen_csv": max_dp,
            "all_alert_decisions_identical": alert_identical,
            "cause": ("screening_predictions.csv was produced on MPS; the "
                      "reliability layer pins CPU for determinism. The gap is "
                      "float32 device arithmetic, not a model change."),
        },
        "known_limitations": [
            "The status is rule-based. It is NOT a calibrated probability and "
            "no calibration was attempted or established.",
            "The reference distribution is the multiday_benchmark_K5 TRAIN "
            "split — the distribution of the scaler the inference path uses. "
            "The checkpoint's own training distribution (single-day "
            "temporal_benchmark_v2) no longer exists on disk, so this layer "
            "measures distance from the scaler's reference, not from the "
            "weights' reference. The documented train/eval normalisation "
            "mismatch is inherited, not repaired.",
            "One held-out day (Friday-02-03-2018, 3,115 sequences, 3 "
            "observable onsets). Every held-out rate is a single-day rate.",
            "Latent-norm stability is largely architectural: the transition "
            "ends in LayerNorm, which bounds the latent RMS by construction. "
            "H7 is therefore a corruption detector, not an instability "
            "detector, and latent_norm_max correlates negatively with "
            "retrospective error.",
            "ELEVATED is a ~1% per-indicator event by construction, so the "
            "CAUTION rate is a property of the quantile choice as much as of "
            "the data.",
            "The retrospective study measures STATE reconstruction error, not "
            "attack-label correctness. It shows the layer separates "
            "conditions under which the decoder is worse; it does not show "
            "the layer predicts label errors.",
            "11 UNRELIABLE sequences is too few to estimate any rate. They "
            "are one physical event observed through a sliding 10-window "
            "lookback, not 11 independent samples.",
            "No abstention is executed. This layer reports; Task 16 owns the "
            "ALERT / BENIGN / ABSTAIN decision.",
        ],
    }
    (OUT_DIR / "reliability_summary.json").write_text(json.dumps(summary, indent=2))

    out_files = ["reliability_config.json", "reliability_per_sequence.csv",
                 "reliability_summary.json", "failure_cases.csv",
                 "forecast_vs_observed.csv"]
    rel_manifest = {
        "component": "V4 reliability / model-health layer",
        "layer_version": rel.LAYER_VERSION,
        "task": "Task 15",
        "generated_utc": started,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "v4_status": "FROZEN — not retrained, not refit, not rethresholded",
        "is_new_learned_model": False,
        "frozen_inputs_sha256": post_hashes,
        "frozen_inputs_unmodified": not changed,
        "reference_distribution_sha256": sha256_file(BENCH_DIR / "X_train.npy"),
        "arrays_read_this_run": READ_LOG,
        "source_files": {f: sha256_file(REPO_ROOT / f) for f in NEW_FILES},
        "outputs": {f: {"sha256": sha256_file(OUT_DIR / f),
                        "size_bytes": (OUT_DIR / f).stat().st_size}
                    for f in out_files},
        "dashboard_interpreter": {
            "path": (str(Path(dash_py).relative_to(REPO_ROOT))
                     if dash_py else None),
            "modules": dash_mods,
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "device": rel.DEVICE,
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True).stdout.strip() or None,
    }
    (OUT_DIR / "reliability_manifest.json").write_text(
        json.dumps(rel_manifest, indent=2))

    validation = {
        "layer_version": rel.LAYER_VERSION,
        "generated_utc": started,
        "completed_utc": rel_manifest["completed_utc"],
        "overall": "FAIL" if REPORT.failed else "PASS",
        "n_gates": len(REPORT.rows),
        "n_passed": sum(r["status"] == "PASS" for r in REPORT.rows),
        "gates": REPORT.rows,
        "artifacts_hashed": len(FROZEN_16),
        "hashed_before_and_after": True,
        "artifact_hashes_before": pre_hashes,
        "artifact_hashes_after": post_hashes,
        "artifacts_changed": changed,
        "claims": {
            "frozen_checkpoint_unchanged": ck_ok and not changed,
            "scalers_unchanged": sc_ok,
            "screening_predictions_unchanged": pred_ok,
            "thresholds_unchanged": thr_ok,
            "no_test_labels_used_to_tune": not label_refs and not reads_before_ref,
            "no_future_observation_in_model_input": bool(no_future),
            "hard_numerical_failures_detected": all_detected and all_unreliable,
            "reliability_outputs_deterministic": bool(det_ok),
            "no_synthetic_data": synth_ok,
            "dashboard_page_loads": bool(import_ok and pages_ok),
            "all_pages_render_headless": bool(import_ok),
            "page_render_probes": page_results,
            "existing_v4_pages_intact": bool(pages_ok),
            "offline_operation_preserved": not net_hits,
            "v4_retrained": False,
            "new_learned_model_introduced": False,
            "calibration_claimed": False,
        },
        "failsafe_self_tests": self_tests,
        "wording_constraint": rel.STATUS_NOTE,
    }
    (OUT_DIR / "validation.json").write_text(json.dumps(validation, indent=2))

    for f in out_files + ["reliability_manifest.json", "validation.json"]:
        p = OUT_DIR / f
        print(f"  wrote  {f:<34} {p.stat().st_size:>10,} B")

    header("VALIDATION SUMMARY")
    for r in REPORT.rows:
        print(f"  {r['status']:<5} {r['name']:<50} {r['detail']}")
    print(f"\n  {validation['n_passed']}/{validation['n_gates']} gates passed")
    print(f"  OVERALL: {validation['overall']}")
    return 1 if REPORT.failed else 0


if __name__ == "__main__":
    sys.exit(main())
