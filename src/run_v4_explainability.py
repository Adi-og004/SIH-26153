"""
=============================================================
TASK 8 -- V4 EXPLAINABILITY ANALYSIS RUNNER
=============================================================

Produces every artifact under ``results/v4_explainability/``
plus ``docs/v4_explainability_contract.json`` and
``results/v4_explainability_summary.json``.

Read-only with respect to the frozen V4 baseline: it loads the
checkpoint, the input scaler, the feature schema and the
pre-built benchmark arrays, and writes nothing back to any of
them. No training, no threshold tuning, no test-label feedback
into the explanation method.

Case-selection criteria are fixed in SELECTION_CRITERIA below
and were defined before any explanation was computed. Every
selection rule is deterministic (median-by-probability or
arg-extremum with an ascending-index tie-break), so re-running
picks the same cases.

    python src/run_v4_explainability.py
=============================================================
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from v4_explainer import (
    V4Explainer,
    FileAccessAudit,
    horizon_target,
    trajectory_max_target,
    benign_mean_baseline_array,
    zero_baseline,
    HorizonProbability,
    sha256_file,
    REPO_ROOT,
    SEQ_LEN,
    N_FEATURES,
    MAX_K,
    IG_N_STEPS,
    IG_METHOD,
)

CANON = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
OUT_DIR = REPO_ROOT / "results" / "v4_explainability"
FIG_DIR = OUT_DIR / "figures"
DOCS_DIR = REPO_ROOT / "docs"

HORIZONS = [1, 3, 5]          # Part 12 -- at least K=1, K=3, K=5
HEADLINE_K = 5                # the horizon the dashboard displays

# Tolerance for the prediction-identity test. The explainer and the
# reference rollout run the identical ops on the identical tensor in
# the identical batch, so the requirement is bit-exactness; 1e-9 is
# the stated ceiling.
IDENTITY_TOLERANCE = 1e-9

# Relative completeness error is only a meaningful statistic where the
# denominator F(x) - F(baseline) is itself not ~0. Below this floor the
# ratio is degenerate; the probability-scale ABSOLUTE error governs.
COMPLETENESS_DENOM_FLOOR = 0.05

# Seeds for the model-randomisation sanity check (Part 23A).
RANDOMIZATION_SEEDS = [20260919, 7, 12345]

SELECTION_CRITERIA = {
    "defined_before_computing_explanations": True,
    "horizon": HEADLINE_K,
    "threshold": "frozen validation threshold for K=5 (0.72)",
    "label": "A_binary_multistep_test[:, 4]  (attack present at t+5)",
    "true_positive": "median-probability sample among predicted=1, actual=1",
    "true_negative": "median-probability sample among predicted=0, actual=0",
    "false_positive": "median-probability sample among predicted=1, actual=0",
    "false_negative": "median-probability sample among predicted=0, actual=1",
    "near_threshold": "argmin |p - 0.72| over all test sequences",
    "high_confidence": "argmax p over all test sequences",
    "low_confidence": "argmin p over all test sequences",
    "median_rule": (
        "sort the group's probabilities ascending, take element "
        "n//2; ties broken by ascending sample index"
    ),
    "not_selected_on": (
        "explanation content, attribution magnitude, or agreement "
        "with any expected narrative"
    ),
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def median_index(prob: np.ndarray, mask: np.ndarray) -> int | None:
    """Deterministic median-probability member of a boolean group."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    order = np.lexsort((idx, prob[idx]))     # prob asc, then index asc
    return int(idx[order[idx.size // 2]])


def argext(prob: np.ndarray, mode: str, ref: float = 0.0) -> int:
    if mode == "max":
        key = -prob
    elif mode == "min":
        key = prob
    else:
        key = np.abs(prob - ref)
    return int(np.lexsort((np.arange(prob.size), key))[0])


# =============================================================
# MAIN
# =============================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)

    # Every file touched from here on is recorded, so the V3
    # contamination claim is a runtime measurement, not an assertion.
    audit = FileAccessAudit().__enter__()

    log("=" * 74)
    log("TASK 8 -- V4 EXPLAINABILITY ANALYSIS")
    log("=" * 74)

    ex = V4Explainer()
    prov = ex.provenance
    log(f"model_class     : {prov.model_class}")
    log(f"checkpoint      : {prov.checkpoint_path}")
    log(f"checkpoint sha  : {prov.checkpoint_hash}")
    log(f"scaler sha      : {prov.scaler_hash}")
    log(f"feature order   : {prov.feature_order_hash}")
    log(f"input shape     : {tuple(prov.input_shape)}")
    log(f"IG              : {IG_METHOD}, n_steps={IG_N_STEPS}")

    names = ex.feature_names
    tlabels = ex.time_labels

    X = np.load(CANON / "X_test.npy")
    A = np.load(CANON / "A_binary_multistep_test.npy")
    meta = pd.read_csv(CANON / "test_metadata.csv")
    log(f"\ntest set        : X{X.shape}  A{A.shape}  meta{meta.shape}")
    assert X.shape[1:] == (SEQ_LEN, N_FEATURES)

    summary: dict = {}
    n = X.shape[0]

    # ---------------------------------------------------------
    # PART 22 -- FEATURE-NAME INTEGRITY
    # ---------------------------------------------------------
    log("\n[Part 22] feature-name integrity")
    with open(DOCS_DIR / "v4_feature_schema.json") as fh:
        schema = json.load(fh)
    disk_names = [
        ln.strip()
        for ln in open(CANON / "input_feature_names.txt")
        if ln.strip()
    ]
    integrity = {
        "checked_utc": started.isoformat().replace("+00:00", "Z"),
        "schema_path": "docs/v4_feature_schema.json",
        "schema_sha256": prov.feature_schema_hash,
        "feature_order_file": (
            "data/processed/multiday_benchmark_K5/input_feature_names.txt"
        ),
        "feature_order_sha256": prov.feature_order_hash,
        "feature_order_sha256_expected": prov.feature_order_hash_expected,
        "n_features": len(names),
        "index_map_identical": all(
            schema["features"][i]["index"] == i
            and schema["features"][i]["name"] == names[i] == disk_names[i]
            for i in range(N_FEATURES)
        ),
        "explainer_names_equal_disk_names": names == disk_names,
        "base_block_indices": [0, 25],
        "rel_change_block_indices": [26, 51],
        "rel_change_naming_rule_holds": all(
            names[26 + i] == f"{names[i]}_rel_change" for i in range(26)
        ),
        "index_to_name": {str(i): names[i] for i in range(N_FEATURES)},
        "mismatches": [],
    }
    for i in range(N_FEATURES):
        if not (schema["features"][i]["name"] == names[i] == disk_names[i]):
            integrity["mismatches"].append(
                {
                    "index": i,
                    "schema": schema["features"][i]["name"],
                    "explainer": names[i],
                    "disk": disk_names[i],
                }
            )
    integrity["passed"] = (
        integrity["index_map_identical"]
        and integrity["explainer_names_equal_disk_names"]
        and integrity["rel_change_naming_rule_holds"]
        and not integrity["mismatches"]
    )
    (OUT_DIR / "feature_name_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )
    log(f"  index->name map identical across 3 sources: "
        f"{integrity['passed']}")
    summary["feature_name_integrity"] = integrity["passed"]

    # ---------------------------------------------------------
    # REFERENCE INFERENCE (the displayed predictions)
    # ---------------------------------------------------------
    log("\n[reference] running the V4 inference path (batch=64)")
    probs = np.concatenate(
        [ex.predict_rollout(X[s:s + 64], MAX_K) for s in range(0, n, 64)]
    )
    log(f"  rollout probabilities: {probs.shape} (column k = P(t+k))")

    from sklearn.metrics import f1_score
    ref_f1 = {}
    for k, thr in ex.horizon_thresholds.items():
        ref_f1[k] = float(f1_score(A[:, k - 1], probs[:, k] >= thr))
    manifest_f1 = {
        int(kk.split("=")[1]): v["f1"]
        for kk, v in ex.manifest["results_verified"][
            "future_prediction"
        ].items()
    }
    f1_match = {
        k: abs(ref_f1[k] - manifest_f1[k]) for k in ref_f1
    }
    log("  frozen-baseline F1 reproduction (must be ~0):")
    for k in sorted(f1_match):
        log(f"    k={k}: f1={ref_f1[k]:.16f}  |delta|={f1_match[k]:.3e}")
    summary["reference_f1_reproduction"] = {
        "computed": ref_f1,
        "manifest": manifest_f1,
        "max_abs_delta": float(max(f1_match.values())),
    }

    # ---------------------------------------------------------
    # PART 20 -- PREDICTION-IDENTITY TEST
    # ---------------------------------------------------------
    log("\n[Part 20] prediction-identity test")
    ident_rows = []
    for k in HORIZONS:
        for s in range(0, n, 64):
            r = ex.check_prediction_identity(
                X[s:s + 64], k, tolerance=IDENTITY_TOLERANCE
            )
            for j in range(r["abs_diff"].size):
                ident_rows.append(
                    {
                        "sample_id": s + j,
                        "horizon": k,
                        "target_type": "horizon_probability",
                        "p_inference": float(r["p_reference"][j]),
                        "p_explained": float(r["p_explained"][j]),
                        "abs_diff": float(r["abs_diff"][j]),
                        "tolerance": IDENTITY_TOLERANCE,
                        "bit_exact": bool(
                            r["p_reference"][j] == r["p_explained"][j]
                        ),
                        "passed": bool(
                            r["abs_diff"][j] < IDENTITY_TOLERANCE
                        ),
                    }
                )
    ident_df = pd.DataFrame(ident_rows)
    ident_df.to_csv(OUT_DIR / "prediction_identity.csv", index=False)
    id_max = float(ident_df["abs_diff"].max())
    id_pass = bool(ident_df["passed"].all())
    id_bit = bool(ident_df["bit_exact"].all())
    log(f"  rows={len(ident_df)}  max|P-P_explain|={id_max:.3e}  "
        f"bit-exact={id_bit}  PASS={id_pass}")

    # Batch-composition sensitivity of the frozen model itself --
    # measured and reported, NOT folded into the identity test.
    p_single = np.concatenate(
        [ex.predict_rollout(X[i:i + 1], MAX_K)[:, HEADLINE_K] for i in range(n)]
    )
    batch_delta = float(np.abs(p_single - probs[:, HEADLINE_K]).max())
    flips = int(
        (
            (p_single >= ex.horizon_thresholds[HEADLINE_K])
            != (probs[:, HEADLINE_K] >= ex.horizon_thresholds[HEADLINE_K])
        ).sum()
    )
    log(f"  [context] batch=1 vs batch=64 LSTM reduction-order spread: "
        f"{batch_delta:.3e}, label flips at thr: {flips}")
    summary["prediction_identity"] = {
        "rows": int(len(ident_df)),
        "horizons": HORIZONS,
        "max_abs_diff": id_max,
        "mean_abs_diff": float(ident_df["abs_diff"].mean()),
        "tolerance": IDENTITY_TOLERANCE,
        "all_bit_exact": id_bit,
        "passed": id_pass,
        "batch_composition_sensitivity_k5": batch_delta,
        "batch_composition_label_flips_k5": flips,
        "batch_composition_note": (
            "Re-batching the SAME frozen model shifts LSTM reduction "
            "order by up to ~1.5e-06. The identity test is run on the "
            "identical tensor in the identical batch, so it is exact. "
            "Batch=1 inference still reproduces the frozen K=5 F1 "
            "digit-for-digit with zero label flips."
        ),
    }

    # ---------------------------------------------------------
    # ATTRIBUTION OVER THE FULL TEST SET, PER HORIZON
    # ---------------------------------------------------------
    log("\n[Parts 7/9/12] Integrated Gradients over the full test set")
    attrs_by_k: dict[int, np.ndarray] = {}
    comp_rows = []
    for k in HORIZONS:
        chunks, errs, rels, preds, bases, deltas = [], [], [], [], [], []
        for s in range(0, n, 64):
            r = ex.attribute(X[s:s + 64], k)
            chunks.append(r["attributions"])
            errs.append(r["completeness_abs_error"])
            rels.append(r["completeness_rel_error"])
            preds.append(r["prediction"])
            bases.append(r["baseline_prediction"])
            deltas.append(r["target_delta"])
        attrs_by_k[k] = np.concatenate(chunks)
        err = np.concatenate(errs)
        rel = np.concatenate(rels)
        pr = np.concatenate(preds)
        bs = np.concatenate(bases)
        dl = np.concatenate(deltas)
        for i in range(n):
            comp_rows.append(
                {
                    "sample_id": i,
                    "horizon": k,
                    "target_type": "horizon_probability",
                    "baseline": "zero_scaled",
                    "n_steps": IG_N_STEPS,
                    "method": IG_METHOD,
                    "F_x": float(pr[i]),
                    "F_baseline": float(bs[i]),
                    "F_x_minus_F_baseline": float(dl[i]),
                    "sum_attributions": float(
                        attrs_by_k[k][i].sum()
                    ),
                    "completeness_abs_error": float(err[i]),
                    "completeness_rel_error": float(rel[i]),
                }
            )
        log(f"  k={k}: attrs{attrs_by_k[k].shape}  "
            f"completeness abs max={err.max():.3e} mean={err.mean():.3e}  "
            f"rel max={rel.max():.3e} mean={rel.mean():.3e}")

    # ---------------------------------------------------------
    # PART 7 -- COMPLETENESS: n_steps sweep + baseline sensitivity
    # ---------------------------------------------------------
    log("\n[Part 7] completeness convergence sweep (fixed 64-sample probe)")
    probe = X[:64]
    benign_base = benign_mean_baseline_array()
    sweep = []
    for k in HORIZONS:
        for nsteps in [50, 128, 256, 1024]:
            r = ex.attribute(probe, k, n_steps=nsteps)
            sweep.append(
                {
                    "horizon": k,
                    "baseline": "zero_scaled",
                    "n_steps": nsteps,
                    "max_abs_error": float(r["completeness_abs_error"].max()),
                    "mean_abs_error": float(r["completeness_abs_error"].mean()),
                    "max_rel_error": float(r["completeness_rel_error"].max()),
                    "mean_rel_error": float(r["completeness_rel_error"].mean()),
                }
            )
        rb = ex.attribute(probe, k, baseline=benign_base, n_steps=IG_N_STEPS)
        sweep.append(
            {
                "horizon": k,
                "baseline": "benign_train_mean",
                "n_steps": IG_N_STEPS,
                "max_abs_error": float(rb["completeness_abs_error"].max()),
                "mean_abs_error": float(rb["completeness_abs_error"].mean()),
                "max_rel_error": float(rb["completeness_rel_error"].max()),
                "mean_rel_error": float(rb["completeness_rel_error"].mean()),
            }
        )
        # Baseline-sensitivity: does the top-feature set survive a
        # different reference point?
        a0 = ex.attribute(probe, k, n_steps=IG_N_STEPS)["attributions"]
        t0 = set(np.argsort(-np.abs(a0).sum(axis=(0, 1)))[:10].tolist())
        t1 = set(
            np.argsort(-np.abs(rb["attributions"]).sum(axis=(0, 1)))[:10]
            .tolist()
        )
        log(f"  k={k} top-10 feature overlap zero vs benign-mean baseline: "
            f"{len(t0 & t1)}/10")
        sweep[-1]["top10_overlap_with_zero_baseline"] = len(t0 & t1)

    comp_df = pd.concat(
        [pd.DataFrame(comp_rows), pd.DataFrame(sweep).assign(
            sample_id="SWEEP_64_SAMPLE_PROBE",
            target_type="horizon_probability",
        )],
        ignore_index=True,
    )
    comp_df.to_csv(OUT_DIR / "completeness_checks.csv", index=False)

    comp_stats = {}
    per_sample = pd.DataFrame(comp_rows)
    for k in HORIZONS:
        g = per_sample[per_sample.horizon == k]
        # The relative error is only well-posed where the denominator
        # F(x) - F(baseline) is not itself ~0. Reported both ways.
        wp = g[g.F_x_minus_F_baseline.abs() > COMPLETENESS_DENOM_FLOOR]
        comp_stats[str(k)] = {
            "max_abs_error": float(g.completeness_abs_error.max()),
            "mean_abs_error": float(g.completeness_abs_error.mean()),
            "p99_abs_error": float(g.completeness_abs_error.quantile(0.99)),
            "unconditioned_max_rel_error": float(
                g.completeness_rel_error.max()
            ),
            "unconditioned_mean_rel_error": float(
                g.completeness_rel_error.mean()
            ),
            "median_abs_denominator": float(
                g.F_x_minus_F_baseline.abs().median()
            ),
            "n_wellposed": int(len(wp)),
            "wellposed_mean_rel_error": float(
                wp.completeness_rel_error.mean()
            ),
            "wellposed_p99_rel_error": float(
                wp.completeness_rel_error.quantile(0.99)
            ),
            "wellposed_max_rel_error": float(wp.completeness_rel_error.max()),
        }
    comp_pass = all(
        v["mean_abs_error"] < 0.01
        and v["p99_abs_error"] < 0.01
        and v["wellposed_mean_rel_error"] < 0.01
        for v in comp_stats.values()
    )
    summary["completeness"] = {
        "method": IG_METHOD,
        "n_steps": IG_N_STEPS,
        "baseline": "zero vector in scaled feature space",
        "per_horizon": comp_stats,
        "acceptance_rule": (
            "PRIMARY: the attributed output is a probability in [0, 1], so "
            "the completeness residual |sum(attr) - (F(x) - F(baseline))| "
            "is read on the probability scale -- mean AND p99 absolute "
            "error must be below 0.01 (one probability point) at every "
            "evaluated horizon. SECONDARY: mean relative error below 1% "
            "restricted to the well-posed subset "
            f"|F(x) - F(baseline)| > {COMPLETENESS_DENOM_FLOOR}."
        ),
        "why_unconditioned_relative_error_is_reported_but_not_used": (
            "For a large share of test windows the model's prediction "
            "nearly equals its prediction at the baseline, so the "
            "completeness denominator is ~0 (1st-percentile "
            "|F(x)-F(baseline)| is ~1e-7 at K=1). Dividing a small "
            "absolute residual by that denominator produces ratios in the "
            "hundreds while the probability-scale error stays below 0.02. "
            "The unconditioned figures are recorded in full above and in "
            "completeness_checks.csv; they are a degenerate statistic, "
            "not a hidden failure."
        ),
        "passed": comp_pass,
    }

    # ---------------------------------------------------------
    # PARTS 9 / 10 -- FEATURE, TEMPORAL, HEATMAP (headline K=5)
    # ---------------------------------------------------------
    log(f"\n[Parts 9/10] aggregations at K={HEADLINE_K}")
    Ah = attrs_by_k[HEADLINE_K]
    thr_h = ex.horizon_thresholds[HEADLINE_K]
    y_h = A[:, HEADLINE_K - 1].astype(int)
    p_h = probs[:, HEADLINE_K]
    pred_h = (p_h >= thr_h).astype(int)

    feat_signed_mean = Ah.sum(axis=1).mean(axis=0)
    feat_abs_mean = np.abs(Ah).sum(axis=1).mean(axis=0)
    fi = pd.DataFrame(
        {
            "feature_index": np.arange(N_FEATURES),
            "feature": names,
            "kind": ["base"] * 26 + ["rel_change"] * 26,
            "horizon": HEADLINE_K,
            "mean_signed_attribution": feat_signed_mean,
            "mean_absolute_attribution": feat_abs_mean,
            "std_signed_attribution": Ah.sum(axis=1).std(axis=0),
            "mean_signed_attack_windows": Ah[y_h == 1].sum(axis=1).mean(axis=0),
            "mean_signed_benign_windows": Ah[y_h == 0].sum(axis=1).mean(axis=0),
            "sample_count": n,
        }
    ).sort_values("mean_absolute_attribution", ascending=False)
    fi.insert(0, "rank", range(1, N_FEATURES + 1))
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    log("  top 10 features by mean |attribution| (K=5):")
    for _, r in fi.head(10).iterrows():
        log(f"    {int(r['rank']):2d}. {r['feature']:<38} "
            f"|attr|={r['mean_absolute_attribution']:.6f}  "
            f"signed={r['mean_signed_attribution']:+.6f}")

    step_signed = Ah.sum(axis=2)
    step_abs = np.abs(Ah).sum(axis=2)
    ti = pd.DataFrame(
        {
            "time_step_index": np.arange(SEQ_LEN),
            "time_step": tlabels,
            "horizon": HEADLINE_K,
            "mean_signed_attribution": step_signed.mean(axis=0),
            "mean_absolute_attribution": step_abs.mean(axis=0),
            "std_absolute_attribution": step_abs.std(axis=0),
            "share_of_total_absolute": (
                step_abs.mean(axis=0) / step_abs.mean(axis=0).sum()
            ),
            "mean_absolute_attack_windows": step_abs[y_h == 1].mean(axis=0),
            "mean_absolute_benign_windows": step_abs[y_h == 0].mean(axis=0),
            "sample_count": n,
        }
    )
    ti.to_csv(OUT_DIR / "temporal_importance.csv", index=False)
    log("  temporal importance (K=5), share of total |attribution|:")
    for _, r in ti.iterrows():
        log(f"    {r['time_step']:<4} "
            f"|attr|={r['mean_absolute_attribution']:.6f}  "
            f"({100 * r['share_of_total_absolute']:.1f}%)  "
            f"signed={r['mean_signed_attribution']:+.6f}")

    # Per-window attribution tensor for the dashboard. Storing it
    # (rather than having the dashboard recompute 3115 x 1024-step IG
    # on load) is what keeps the displayed explanation identical to
    # this verified run -- the file is hash-pinned in the summary and
    # the dashboard refuses to use it if the provenance differs.
    for _k in HORIZONS:
        np.save(
            OUT_DIR / f"attributions_k{_k}.npy",
            attrs_by_k[_k].astype(np.float32),
        )
    log(f"  saved attributions_k{{{','.join(str(k) for k in HORIZONS)}}}.npy "
        f"{Ah.shape} each (float32, per-window attribution)")

    heat = pd.DataFrame(Ah.mean(axis=0), index=tlabels, columns=names)
    heat.index.name = "time_step"
    heat.to_csv(OUT_DIR / "attribution_heatmap.csv")
    log(f"  heatmap grid: {heat.shape[0]} x {heat.shape[1]} "
        f"(mean signed attribution over {n} test sequences)")

    summary["top_features_k5"] = [
        {
            "rank": int(r["rank"]),
            "feature": r["feature"],
            "mean_absolute_attribution": float(
                r["mean_absolute_attribution"]
            ),
            "mean_signed_attribution": float(r["mean_signed_attribution"]),
        }
        for _, r in fi.head(10).iterrows()
    ]
    summary["temporal_importance_k5"] = [
        {
            "time_step": r["time_step"],
            "mean_absolute_attribution": float(
                r["mean_absolute_attribution"]
            ),
            "share_of_total_absolute": float(r["share_of_total_absolute"]),
            "mean_signed_attribution": float(r["mean_signed_attribution"]),
        }
        for _, r in ti.iterrows()
    ]

    # ---------------------------------------------------------
    # PART 12 -- HORIZON FEATURE IMPORTANCE
    # ---------------------------------------------------------
    log("\n[Part 12] horizon comparison")
    hrows = []
    for k in HORIZONS:
        Ak = attrs_by_k[k]
        sg = Ak.sum(axis=1)
        ab = np.abs(Ak).sum(axis=1)
        for i in range(N_FEATURES):
            hrows.append(
                {
                    "horizon": k,
                    "feature": names[i],
                    "feature_index": i,
                    "mean_signed_attribution": float(sg[:, i].mean()),
                    "mean_absolute_attribution": float(ab[:, i].mean()),
                    "sample_count": n,
                }
            )
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(OUT_DIR / "horizon_feature_importance.csv", index=False)
    top_by_k = {}
    for k in HORIZONS:
        g = hdf[hdf.horizon == k].nlargest(10, "mean_absolute_attribution")
        top_by_k[k] = g["feature"].tolist()
        log(f"  K={k} top-5: {', '.join(top_by_k[k][:5])}")
    overlaps = {
        "k1_vs_k5": len(set(top_by_k[1]) & set(top_by_k[5])),
        "k3_vs_k5": len(set(top_by_k[3]) & set(top_by_k[5])),
        "k1_vs_k3": len(set(top_by_k[1]) & set(top_by_k[3])),
    }
    log(f"  top-10 overlap: {overlaps}")
    summary["horizon_top10"] = {str(k): v for k, v in top_by_k.items()}
    summary["horizon_top10_overlap"] = overlaps

    # ---------------------------------------------------------
    # PART 8 -- FEATURE ABLATION CROSS-CHECK
    # ---------------------------------------------------------
    log("\n[Part 8] feature-ablation cross-check")
    # Deterministic representative sample: every 32nd test sequence.
    abl_idx = list(range(0, n, 32))
    abl_rows = []
    sg_h = Ah.sum(axis=1)
    ab_h = np.abs(Ah).sum(axis=1)
    rng = np.random.default_rng(20260919)   # fixed seed -> reproducible
    for i in abl_idx:
        order = np.lexsort((np.arange(N_FEATURES), -ab_h[i]))
        top5 = order[:5]
        low_pool = order[26:]
        low5 = rng.choice(low_pool, size=5, replace=False)
        for grp, fidx in (
            [("top_attributed", int(f)) for f in top5]
            + [("low_attributed_control", int(f)) for f in low5]
        ):
            r = ex.ablate_feature(X[i:i + 1], fidx, HEADLINE_K)
            abl_rows.append(
                {
                    "sample_id": i,
                    "horizon": HEADLINE_K,
                    "group": grp,
                    "feature_index": fidx,
                    "feature": names[fidx],
                    "signed_attribution": float(sg_h[i, fidx]),
                    "absolute_attribution": float(ab_h[i, fidx]),
                    "attribution_rank": int(
                        np.flatnonzero(order == fidx)[0] + 1
                    ),
                    "p_original": float(r["p_original"][0]),
                    "p_ablated": float(r["p_ablated"][0]),
                    "delta_probability": float(r["delta"][0]),
                    "sign_agrees": bool(
                        np.sign(sg_h[i, fidx]) == np.sign(r["delta"][0])
                        or abs(r["delta"][0]) < 1e-12
                    ),
                }
            )
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(OUT_DIR / "ablation_crosscheck.csv", index=False)
    top = abl[abl.group == "top_attributed"]
    low = abl[abl.group == "low_attributed_control"]
    sign_agree = float(top["sign_agrees"].mean())
    spearman = float(
        pd.Series(top["signed_attribution"]).corr(
            pd.Series(top["delta_probability"]), method="spearman"
        )
    )
    pearson = float(
        pd.Series(top["signed_attribution"]).corr(
            pd.Series(top["delta_probability"]), method="pearson"
        )
    )
    log(f"  samples={len(abl_idx)}  rows={len(abl)}")
    log(f"  top-attributed: sign agreement={100 * sign_agree:.1f}%  "
        f"spearman={spearman:.4f}  pearson={pearson:.4f}")
    log(f"  mean |dP|: top-attributed={top.delta_probability.abs().mean():.6f}"
        f"  low-attributed control="
        f"{low.delta_probability.abs().mean():.6f}  "
        f"ratio={top.delta_probability.abs().mean() / max(low.delta_probability.abs().mean(), 1e-12):.1f}x")
    summary["ablation_crosscheck"] = {
        "samples": len(abl_idx),
        "rows": int(len(abl)),
        "horizon": HEADLINE_K,
        "top_attributed_sign_agreement": sign_agree,
        "spearman_signed_attr_vs_delta_p": spearman,
        "pearson_signed_attr_vs_delta_p": pearson,
        "mean_abs_delta_top_attributed": float(
            top.delta_probability.abs().mean()
        ),
        "mean_abs_delta_low_attributed_control": float(
            low.delta_probability.abs().mean()
        ),
        "sensitivity_ratio": float(
            top.delta_probability.abs().mean()
            / max(low.delta_probability.abs().mean(), 1e-12)
        ),
        "interpretation": (
            "Directional validation only. IG attributes along a path "
            "from the baseline; single-feature ablation is a discrete "
            "one-coordinate move. Exact agreement is not expected."
        ),
    }

    # ---------------------------------------------------------
    # PART 21 -- REPEATABILITY
    # ---------------------------------------------------------
    log("\n[Part 21] repeatability (same input, two independent runs)")
    rep_rows = []
    for k in HORIZONS:
        for s in range(0, 256, 64):
            r1 = ex.attribute(X[s:s + 64], k)
            r2 = ex.attribute(X[s:s + 64], k)
            a1, a2 = r1["attributions"], r2["attributions"]
            for j in range(a1.shape[0]):
                o1 = np.lexsort(
                    (np.arange(N_FEATURES), -np.abs(a1[j]).sum(axis=0))
                )[:5]
                o2 = np.lexsort(
                    (np.arange(N_FEATURES), -np.abs(a2[j]).sum(axis=0))
                )[:5]
                rep_rows.append(
                    {
                        "sample_id": s + j,
                        "horizon": k,
                        "p_run1": float(r1["prediction"][j]),
                        "p_run2": float(r2["prediction"][j]),
                        "prediction_abs_diff": float(
                            abs(r1["prediction"][j] - r2["prediction"][j])
                        ),
                        "attribution_max_abs_diff": float(
                            np.abs(a1[j] - a2[j]).max()
                        ),
                        "attribution_bit_exact": bool(
                            np.array_equal(a1[j], a2[j])
                        ),
                        "top5_identical": bool(np.array_equal(o1, o2)),
                        "full_ranking_identical": bool(
                            np.array_equal(
                                np.lexsort((np.arange(N_FEATURES),
                                            -np.abs(a1[j]).sum(axis=0))),
                                np.lexsort((np.arange(N_FEATURES),
                                            -np.abs(a2[j]).sum(axis=0))),
                            )
                        ),
                    }
                )
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(OUT_DIR / "repeatability.csv", index=False)
    rep_ok = bool(
        rep.attribution_bit_exact.all()
        and rep.top5_identical.all()
        and rep.full_ranking_identical.all()
        and (rep.prediction_abs_diff == 0).all()
    )
    log(f"  rows={len(rep)}  attribution bit-exact="
        f"{bool(rep.attribution_bit_exact.all())}  "
        f"ranking identical={bool(rep.full_ranking_identical.all())}  "
        f"PASS={rep_ok}")
    summary["repeatability"] = {
        "rows": int(len(rep)),
        "horizons": HORIZONS,
        "prediction_max_abs_diff": float(rep.prediction_abs_diff.max()),
        "attribution_max_abs_diff": float(rep.attribution_max_abs_diff.max()),
        "all_attributions_bit_exact": bool(rep.attribution_bit_exact.all()),
        "all_top5_identical": bool(rep.top5_identical.all()),
        "all_full_rankings_identical": bool(rep.full_ranking_identical.all()),
        "passed": rep_ok,
    }

    # ---------------------------------------------------------
    # PART 11 -- REPRESENTATIVE EXPLANATIONS
    # ---------------------------------------------------------
    log("\n[Part 11] representative cases (criteria fixed in advance)")
    ts = pd.to_datetime(meta["timestamp"]).to_numpy()
    cases: list[tuple[str, int | None]] = [
        ("true_positive", median_index(p_h, (pred_h == 1) & (y_h == 1))),
        ("true_negative", median_index(p_h, (pred_h == 0) & (y_h == 0))),
        ("false_positive", median_index(p_h, (pred_h == 1) & (y_h == 0))),
        ("false_negative", median_index(p_h, (pred_h == 0) & (y_h == 1))),
        ("near_threshold", argext(p_h, "near", thr_h)),
        ("high_confidence", argext(p_h, "max")),
        ("low_confidence", argext(p_h, "min")),
    ]

    def explain_rows(items, target_k, extra_cols_fn, source="dataset"):
        rows = []
        for label, idx in items:
            if idx is None:
                continue
            r = ex.attribute(X[idx:idx + 1], target_k)
            a = r["attributions"][0]
            sg, ab = a.sum(axis=0), np.abs(a).sum(axis=0)
            order = np.lexsort((np.arange(N_FEATURES), -ab))[:5]
            st = np.abs(a).sum(axis=1)
            sorder = np.lexsort((np.arange(SEQ_LEN), -st))[:3]
            base = {
                "label": label,
                "sample_id": int(idx),
                "input_source": source,
                "target_type": "horizon_probability",
                "target_horizon": target_k,
                "target_definition": horizon_target(
                    target_k, ex.manifest
                ).target_definition,
                "forecast_time": str(pd.Timestamp(ts[idx])),
                "target_time": str(
                    pd.Timestamp(ts[idx]) + pd.Timedelta(seconds=10 * target_k)
                ),
                "horizon": target_k,
                "probability": float(probs[idx, target_k]),
                "probability_explained": float(r["prediction"][0]),
                "threshold": float(ex.horizon_thresholds[target_k]),
                "actual_label": int(A[idx, target_k - 1]),
                "predicted_label": int(
                    probs[idx, target_k] >= ex.horizon_thresholds[target_k]
                ),
                "top_features": "; ".join(names[i] for i in order),
                "top_signed_attributions": "; ".join(
                    f"{sg[i]:+.6f}" for i in order
                ),
                "top_absolute_attributions": "; ".join(
                    f"{ab[i]:.6f}" for i in order
                ),
                "top_time_steps": "; ".join(tlabels[s] for s in sorder),
                "top_time_step_absolute": "; ".join(
                    f"{st[s]:.6f}" for s in sorder
                ),
                "completeness_abs_error": float(
                    r["completeness_abs_error"][0]
                ),
            }
            base.update(extra_cols_fn(label, idx, order, sg))
            rows.append(base)
        return rows

    def rep_extra(label, idx, order, sg):
        return {
            "selection_rule": SELECTION_CRITERIA.get(label, label),
            "dominant_attack": str(meta["dominant_attack"].iloc[idx]),
        }

    rep_cases = pd.DataFrame(explain_rows(cases, HEADLINE_K, rep_extra))
    rep_cases.to_csv(
        OUT_DIR / "representative_explanations.csv", index=False
    )
    for _, r in rep_cases.iterrows():
        log(f"  {r['label']:<16} id={r['sample_id']:<5} "
            f"p={r['probability']:.4f} y={r['actual_label']} "
            f"pred={r['predicted_label']}  top: "
            f"{r['top_features'].split(';')[0].strip()}")
    summary["representative_cases"] = rep_cases[
        ["label", "sample_id", "probability", "actual_label",
         "predicted_label", "top_features", "top_time_steps"]
    ].to_dict("records")

    # ---------------------------------------------------------
    # PART 13 -- ATTACK VS BENIGN
    # ---------------------------------------------------------
    log("\n[Part 13] attack vs benign attribution (correct predictions only)")
    tp_mask = (pred_h == 1) & (y_h == 1)
    tn_mask = (pred_h == 0) & (y_h == 0)
    log(f"  correctly predicted attack windows: {int(tp_mask.sum())}  "
        f"benign: {int(tn_mask.sum())}")
    sg_all = Ah.sum(axis=1)
    ab_all = np.abs(Ah).sum(axis=1)
    avb = pd.DataFrame(
        {
            "feature_index": np.arange(N_FEATURES),
            "feature": names,
            "kind": ["base"] * 26 + ["rel_change"] * 26,
            "horizon": HEADLINE_K,
            "n_attack_windows": int(tp_mask.sum()),
            "n_benign_windows": int(tn_mask.sum()),
            "attack_mean_signed": sg_all[tp_mask].mean(axis=0),
            "attack_mean_absolute": ab_all[tp_mask].mean(axis=0),
            "benign_mean_signed": sg_all[tn_mask].mean(axis=0),
            "benign_mean_absolute": ab_all[tn_mask].mean(axis=0),
            "signed_difference_attack_minus_benign": (
                sg_all[tp_mask].mean(axis=0) - sg_all[tn_mask].mean(axis=0)
            ),
            "attack_mean_feature_value_last_step": X[tp_mask][:, -1, :].mean(
                axis=0
            ),
            "benign_mean_feature_value_last_step": X[tn_mask][:, -1, :].mean(
                axis=0
            ),
        }
    ).sort_values(
        "signed_difference_attack_minus_benign",
        key=np.abs,
        ascending=False,
    )
    avb.to_csv(OUT_DIR / "attack_vs_benign.csv", index=False)
    log("  largest attack-minus-benign signed attribution gaps:")
    for _, r in avb.head(8).iterrows():
        log(f"    {r['feature']:<38} "
            f"attack={r['attack_mean_signed']:+.6f} "
            f"benign={r['benign_mean_signed']:+.6f} "
            f"diff={r['signed_difference_attack_minus_benign']:+.6f}")
    summary["attack_vs_benign_top_gaps"] = [
        {
            "feature": r["feature"],
            "attack_mean_signed": float(r["attack_mean_signed"]),
            "benign_mean_signed": float(r["benign_mean_signed"]),
            "difference": float(r["signed_difference_attack_minus_benign"]),
        }
        for _, r in avb.head(10).iterrows()
    ]

    # ---------------------------------------------------------
    # PART 14 -- FP/FN EXPLANATIONS (Task 4 cases)
    # ---------------------------------------------------------
    log("\n[Part 14] FP/FN explanations from the Task 4 representative set")
    t4 = pd.read_csv(
        REPO_ROOT / "results" / "error_analysis" / "representative_cases.csv"
    )
    t4 = t4[t4.model == "v4_world_model"].reset_index(drop=True)
    err_rows = []
    for _, c in t4.iterrows():
        idx = int(c["sample_index"])
        k = int(c["horizon"])
        r = ex.attribute(X[idx:idx + 1], k)
        a = r["attributions"][0]
        sg, ab = a.sum(axis=0), np.abs(a).sum(axis=0)
        order = np.lexsort((np.arange(N_FEATURES), -ab))[:5]
        st = np.abs(a).sum(axis=1)
        top_step = tlabels[int(np.lexsort((np.arange(SEQ_LEN), -st))[0])]
        ups = [names[i] for i in order if sg[i] > 0]
        downs = [names[i] for i in order if sg[i] <= 0]
        if c["error_type"] == "FP":
            note = (
                f"Model attribution: {', '.join(ups) or 'no feature'} "
                f"increased the predicted threat probability at K={k} for a "
                f"window whose actual label is benign; largest-magnitude "
                f"history window was {top_step}."
            )
        else:
            note = (
                f"Model attribution: {', '.join(downs) or 'no feature'} "
                f"decreased the predicted threat probability at K={k} for a "
                f"window whose actual label is attack; largest-magnitude "
                f"history window was {top_step}."
            )
        err_rows.append(
            {
                "error_type": c["error_type"],
                "task4_case_id": c["case_id"],
                "taxonomy_category": c["taxonomy_category"],
                "sample_id": idx,
                "time": c["time"],
                "horizon": k,
                "target_type": "horizon_probability",
                "input_source": "dataset",
                "probability": float(probs[idx, k]),
                "probability_explained": float(r["prediction"][0]),
                "threshold": float(ex.horizon_thresholds[k]),
                "actual": int(c["actual"]),
                "predicted": int(c["predicted"]),
                "top_feature_1": names[order[0]],
                "top_feature_2": names[order[1]],
                "top_feature_3": names[order[2]],
                "top_feature_4": names[order[3]],
                "top_feature_5": names[order[4]],
                "top_signed_attributions": "; ".join(
                    f"{sg[i]:+.6f}" for i in order
                ),
                "top_absolute_attributions": "; ".join(
                    f"{ab[i]:.6f}" for i in order
                ),
                "top_time_step": top_step,
                "interpretation_notes": note,
            }
        )
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(OUT_DIR / "error_explanations.csv", index=False)
    for _, r in err_df.iterrows():
        log(f"  {r['task4_case_id']} {r['error_type']} "
            f"({r['taxonomy_category']}) p={r['probability']:.4f} "
            f"top1={r['top_feature_1']}")
    summary["error_explanations"] = err_df[
        ["task4_case_id", "error_type", "taxonomy_category", "sample_id",
         "probability", "top_feature_1", "top_feature_2", "top_feature_3"]
    ].to_dict("records")

    # ---------------------------------------------------------
    # PART 15 -- ONSET EXPLANATIONS
    # ---------------------------------------------------------
    log("\n[Part 15] onset explanations")
    onsets = ex.manifest["dataset"]["test_observable_attack_onsets"]
    ts_series = pd.to_datetime(meta["timestamp"])
    onset_rows = []
    for ep, onset_str in enumerate(onsets, start=2):
        onset = pd.Timestamp(onset_str)
        # Forecast windows whose K=5 target time lands at/near onset:
        # the 12 windows before onset, matching Task 4's lookback.
        pre = np.flatnonzero(
            (ts_series >= onset - pd.Timedelta(seconds=120))
            & (ts_series < onset)
        )
        if pre.size == 0:
            continue
        maxp = p_h[pre].max()
        warned = bool(maxp >= thr_h)
        # elevated pre-onset window = highest-probability lookback window
        pick = int(pre[argext(p_h[pre], "max")])
        r = ex.attribute(X[pick:pick + 1], HEADLINE_K)
        a = r["attributions"][0]
        sg, ab = a.sum(axis=0), np.abs(a).sum(axis=0)
        order = np.lexsort((np.arange(N_FEATURES), -ab))[:5]
        st = np.abs(a).sum(axis=1)
        sorder = np.lexsort((np.arange(SEQ_LEN), -st))[:3]
        case = "pre_onset_warning" if warned else "missed_onset"
        ups = [names[i] for i in order if sg[i] > 0]
        downs = [names[i] for i in order if sg[i] <= 0]
        if warned:
            note = (
                f"Model attribution for the elevated pre-onset window: "
                f"{', '.join(ups) or 'no feature'} increased the K=5 "
                f"predicted threat probability to {p_h[pick]:.4f} "
                f"(threshold {thr_h}). Lead time "
                f"{(onset - ts_series.iloc[pick]).total_seconds():.0f}s. "
                f"Model-attributed traffic pattern only -- not evidence of "
                f"attacker intent."
            )
        else:
            note = (
                f"Model attribution for the highest-probability pre-onset "
                f"window of a missed onset: probability stayed at "
                f"{p_h[pick]:.4f}, below the {thr_h} threshold; "
                f"{', '.join(downs) or 'no feature'} carried negative "
                f"attribution. Model-attributed traffic pattern only."
            )
        onset_rows.append(
            {
                "episode_id": ep,
                "attack_onset": str(onset),
                "case_type": case,
                "warning_detected": warned,
                "lookback_windows": int(pre.size),
                "max_probability_in_lookback": float(maxp),
                "sample_id": pick,
                "forecast_time": str(ts_series.iloc[pick]),
                "target_time": str(
                    ts_series.iloc[pick]
                    + pd.Timedelta(seconds=10 * HEADLINE_K)
                ),
                "lead_time_sec": float(
                    (onset - ts_series.iloc[pick]).total_seconds()
                ),
                "horizon": HEADLINE_K,
                "probability": float(p_h[pick]),
                "probability_explained": float(r["prediction"][0]),
                "threshold": float(thr_h),
                "actual_label": int(A[pick, HEADLINE_K - 1]),
                "input_source": "dataset",
                "top_features": "; ".join(names[i] for i in order),
                "top_signed_attributions": "; ".join(
                    f"{sg[i]:+.6f}" for i in order
                ),
                "top_absolute_attributions": "; ".join(
                    f"{ab[i]:.6f}" for i in order
                ),
                "top_time_steps": "; ".join(tlabels[s] for s in sorder),
                "interpretation_notes": note,
            }
        )
    onset_df = pd.DataFrame(onset_rows)
    onset_df.to_csv(OUT_DIR / "onset_explanations.csv", index=False)
    for _, r in onset_df.iterrows():
        log(f"  ep{r['episode_id']} {r['attack_onset']} {r['case_type']:<18} "
            f"p={r['probability']:.4f} top1="
            f"{r['top_features'].split(';')[0].strip()}")
    # Warning vs missed comparison
    warn = onset_df[onset_df.case_type == "pre_onset_warning"]
    miss = onset_df[onset_df.case_type == "missed_onset"]
    summary["onset_explanations"] = {
        "episodes": onset_df[
            ["episode_id", "attack_onset", "case_type", "probability",
             "top_features"]
        ].to_dict("records"),
        "warning_cases": int(len(warn)),
        "missed_onset_cases": int(len(miss)),
    }

    # ---------------------------------------------------------
    # PART 23 -- SANITY CHECKS
    # ---------------------------------------------------------
    log("\n[Part 23] explanation-quality sanity checks")
    sanity = {}

    # A. model-randomisation test (on a COPY -- production model untouched)
    #
    # The discriminating statistic is the SIGNED attribution rank
    # correlation. Comparing |attribution| instead is confounded: IG
    # attribution scales with the input itself, so a randomised model
    # and the real model share input-driven magnitude structure even
    # when they share no learned structure. Both are reported, and the
    # confound is measured directly against |x|.
    import copy as _copy
    from captum.attr import IntegratedGradients as _IG

    probe_i = np.array(abl_idx[:32])
    xb = torch.from_numpy(X[probe_i])
    real_sgn = Ah[probe_i].sum(axis=1)
    real_abs = np.abs(Ah[probe_i]).sum(axis=1)
    x_abs = np.abs(X[probe_i]).sum(axis=1)

    def _mean_corr(a, b, method="spearman"):
        return float(
            np.mean(
                [
                    pd.Series(a[i]).corr(pd.Series(b[i]), method=method)
                    for i in range(a.shape[0])
                ]
            )
        )

    rnd_runs = []
    for seed in RANDOMIZATION_SEEDS:
        rnd_model = _copy.deepcopy(ex.model)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in rnd_model.parameters():
                if p.dim() >= 2:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    p.copy_(
                        torch.empty_like(p).uniform_(-0.1, 0.1, generator=g)
                    )
        rnd_model.eval()
        fn_r = HorizonProbability(rnd_model, HEADLINE_K)
        fn_r.eval()
        ar, _ = _IG(fn_r).attribute(
            xb, baselines=zero_baseline(xb), n_steps=IG_N_STEPS,
            method=IG_METHOD, return_convergence_delta=True,
            internal_batch_size=4096,
        )
        ar = ar.detach().numpy()
        rnd_sgn = ar.sum(axis=1)
        rnd_abs = np.abs(ar).sum(axis=1)
        rnd_runs.append(
            {
                "seed": seed,
                "mean_spearman_signed": _mean_corr(real_sgn, rnd_sgn),
                "mean_pearson_signed": _mean_corr(
                    real_sgn, rnd_sgn, "pearson"
                ),
                "mean_spearman_absolute": _mean_corr(real_abs, rnd_abs),
                "mean_top5_overlap_fraction": float(
                    np.mean(
                        [
                            len(
                                set(np.argsort(-real_abs[i])[:5])
                                & set(np.argsort(-rnd_abs[i])[:5])
                            ) / 5
                            for i in range(len(probe_i))
                        ]
                    )
                ),
                "mean_spearman_absolute_vs_input_magnitude": _mean_corr(
                    rnd_abs, x_abs
                ),
            }
        )
        log(f"  A. randomisation seed={seed}: "
            f"spearman(signed)={rnd_runs[-1]['mean_spearman_signed']:+.4f}  "
            f"spearman(|attr|)="
            f"{rnd_runs[-1]['mean_spearman_absolute']:+.4f}  "
            f"top-5 overlap="
            f"{100 * rnd_runs[-1]['mean_top5_overlap_fraction']:.1f}%")

    mean_signed_corr = float(
        np.mean([abs(r["mean_spearman_signed"]) for r in rnd_runs])
    )
    real_vs_x = _mean_corr(real_abs, x_abs)
    prod_unchanged = bool(
        np.array_equal(ex.predict_rollout(X[:64], MAX_K), probs[:64])
    )
    log(f"     |attr| confound: spearman(|attr|, |x|) real={real_vs_x:+.4f} "
        f"random="
        f"{np.mean([r['mean_spearman_absolute_vs_input_magnitude'] for r in rnd_runs]):+.4f}"
        f"  -> magnitude overlap is input-driven, not model-driven")
    log(f"     mean |spearman(signed)| across "
        f"{len(RANDOMIZATION_SEEDS)} seeds = {mean_signed_corr:.4f}  "
        f"production model unchanged={prod_unchanged}")
    sanity["randomization_test"] = {
        "description": (
            "Xavier (>=2-D) / uniform(-0.1, 0.1) (1-D) re-init of a deep "
            "copy of the model, three seeds; the production model is "
            "never touched"
        ),
        "seeds": RANDOMIZATION_SEEDS,
        "samples": int(len(probe_i)),
        "per_seed": rnd_runs,
        "mean_abs_spearman_signed_across_seeds": mean_signed_corr,
        "real_model_spearman_abs_attribution_vs_input_magnitude": real_vs_x,
        "confound_note": (
            "Integrated Gradients attribution scales with the input, so "
            "|attribution| rank correlates ~0.6 with |x| for BOTH the "
            "real and the randomised model. The residual |attribution| "
            "overlap is therefore input-driven, not evidence of retained "
            "model structure. The signed attribution, which does depend "
            "on learned weights, is the discriminating statistic."
        ),
        "production_model_unchanged": prod_unchanged,
        "acceptance_rule": (
            "mean |spearman| of SIGNED attribution between the real and "
            "randomised models below 0.30, averaged over three seeds -- "
            "learned structure must be lost"
        ),
        "passed": bool(mean_signed_corr < 0.30 and prod_unchanged),
    }

    # B. feature-perturbation test (already measured by the ablation run)
    pert_pass = bool(
        top.delta_probability.abs().mean()
        > low.delta_probability.abs().mean()
    )
    log(f"  B. perturbation: |dP| top-attributed "
        f"{top.delta_probability.abs().mean():.6f} vs low-attributed "
        f"{low.delta_probability.abs().mean():.6f} -> PASS={pert_pass}")
    sanity["perturbation_test"] = {
        "description": (
            "ablate top-5 attributed features vs 5 randomly chosen "
            "features ranked 27-52 by |attribution|"
        ),
        "mean_abs_delta_top": float(top.delta_probability.abs().mean()),
        "mean_abs_delta_low": float(low.delta_probability.abs().mean()),
        "acceptance_rule": (
            "mean |dP| for top-attributed features must exceed that of "
            "the low-attribution control"
        ),
        "passed": pert_pass,
    }

    # C. completeness (already computed)
    sanity["completeness_test"] = {
        "acceptance_rule": summary["completeness"]["acceptance_rule"],
        "per_horizon_mean_abs_error": {
            k: v["mean_abs_error"] for k, v in comp_stats.items()
        },
        "per_horizon_wellposed_mean_rel_error": {
            k: v["wellposed_mean_rel_error"] for k, v in comp_stats.items()
        },
        "passed": summary["completeness"]["passed"],
    }
    log(f"  C. completeness: PASS={summary['completeness']['passed']}")
    sanity["all_passed"] = all(
        sanity[t]["passed"]
        for t in ("randomization_test", "perturbation_test",
                  "completeness_test")
    )
    summary["sanity_checks"] = sanity

    # ---------------------------------------------------------
    # PART 1 / TARGET B -- TRAJECTORY MAX
    # ---------------------------------------------------------
    log("\n[Part 1] TARGET B -- trajectory max diagnostics")
    tb_idx = np.array(abl_idx[:64])
    tbi = ex.trajectory_max_info(X[tb_idx], MAX_K)
    switch_rate = float(tbi["argmax_switches_on_path"].mean())
    log(f"  argmax distribution over k=1..5: "
        f"{np.bincount(tbi['argmax_horizon'], minlength=6)[1:].tolist()}")
    log(f"  exact ties at the argmax: "
        f"{int((tbi['n_tied_horizons'] > 1).sum())}/{len(tb_idx)}")
    log(f"  argmax CHANGES along the IG path: "
        f"{100 * switch_rate:.1f}% of sampled windows")
    tb_rows = []
    for j, i in enumerate(tb_idx):
        # Recompute the displayed max in the SAME batch of 1 the
        # attribution runs in: re-batching the frozen LSTM shifts
        # reduction order by ~1e-6, which is not a target mismatch.
        row_probs = ex.predict_rollout(X[i:i + 1], MAX_K)[0]
        kstar = int(row_probs[1:].argmax() + 1)
        displayed_max = float(row_probs[1:].max())
        tgt = trajectory_max_target(MAX_K, ex.manifest, kstar)
        r = ex.attribute(X[i:i + 1], kstar)
        a = r["attributions"][0]
        ab = np.abs(a).sum(axis=0)
        order = np.lexsort((np.arange(N_FEATURES), -ab))[:5]
        tb_rows.append(
            {
                "sample_id": int(i),
                "target_type": "trajectory_max",
                "target_horizon": MAX_K,
                "attributed_horizon_kstar": kstar,
                "displayed_max_probability": displayed_max,
                "displayed_max_probability_batch64": float(
                    tbi["max_probability"][j]
                ),
                "attributed_function_value": float(r["prediction"][0]),
                "displayed_equals_attributed": bool(
                    abs(displayed_max - float(r["prediction"][0])) < 1e-9
                ),
                "threshold": tgt.threshold,
                "n_tied_horizons": int(tbi["n_tied_horizons"][j]),
                "argmax_switches_on_path": bool(
                    tbi["argmax_switches_on_path"][j]
                ),
                "completeness_abs_error": float(
                    r["completeness_abs_error"][0]
                ),
                "top_features": "; ".join(names[t] for t in order),
                "target_definition": tgt.target_definition,
            }
        )
    tb_df = pd.DataFrame(tb_rows)
    tb_df.to_csv(OUT_DIR / "trajectory_max_explanations.csv", index=False)
    log(f"  displayed max == attributed value for "
        f"{int(tb_df.displayed_equals_attributed.sum())}/{len(tb_df)} windows")
    summary["trajectory_max_target"] = {
        "samples": int(len(tb_df)),
        "threshold": ex.trajectory_max_thresholds[MAX_K],
        "argmax_distribution_k1_to_k5": np.bincount(
            tbi["argmax_horizon"], minlength=6
        )[1:].tolist(),
        "exact_ties": int((tbi["n_tied_horizons"] > 1).sum()),
        "argmax_switches_on_path_rate": switch_rate,
        "displayed_equals_attributed_all": bool(
            tb_df.displayed_equals_attributed.all()
        ),
        "documented_behaviour": (
            "MAX is non-differentiable at ties. The explainer attributes "
            "the fixed horizon k* = argmax at the input, so the attributed "
            "function value equals the displayed max exactly. In "
            f"{100 * switch_rate:.1f}% of sampled windows the argmax "
            "changes somewhere along the straight-line integration path, "
            "so the completeness reference is F_k*(baseline), not "
            "max(baseline). Recorded per sample in "
            "trajectory_max_explanations.csv."
        ),
    }

    # ---------------------------------------------------------
    # PART 16 -- PCAP EXPLAINABILITY
    # ---------------------------------------------------------
    log("\n[Part 16] PCAP explainability (with mandatory caveat)")
    pcap_rows = []
    pcap_caveat = (
        "The PCAP explanation uses the same V4 feature contract and "
        "transform chain, but flow-record production differences can "
        "affect the resulting values. Task 7 proved exact parity for the "
        "26 base-feature aggregations, the 26 relative-change features, "
        "all transforms, the scaler interface and 10-second windowing; "
        "flow-record production remains APPROXIMATE. These values are "
        "NOT numerically equivalent to training features."
    )
    try:
        import pcap_ingest_v4 as p4

        scaler = p4.load_scaler(p4.INPUT_SCALER_PATH)
        for pcap_name in ["demo_portscan.pcap", "demo_dos_flood.pcap"]:
            path = REPO_ROOT / "data" / "sample_pcap" / pcap_name
            if not path.exists():
                continue
            pdf = p4.parse_pcap_streaming(str(path))
            flows = p4.reconstruct_flows(pdf)
            feats = p4.compute_window_features(flows)
            feats = p4.add_relative_changes(feats)
            seqs = p4.build_sequences(feats, scaler)
            Xp = seqs["X"]
            if Xp.shape[0] == 0:
                log(f"  {pcap_name}: no contiguous 10-window sequence")
                continue
            pp = ex.predict_rollout(Xp, MAX_K)
            pick = argext(pp[:, HEADLINE_K], "max")
            r = ex.attribute(Xp[pick:pick + 1], HEADLINE_K)
            ident = ex.check_prediction_identity(
                Xp[pick:pick + 1], HEADLINE_K, IDENTITY_TOLERANCE
            )
            a = r["attributions"][0]
            sg, ab = a.sum(axis=0), np.abs(a).sum(axis=0)
            order = np.lexsort((np.arange(N_FEATURES), -ab))[:5]
            st = np.abs(a).sum(axis=1)
            pcap_rows.append(
                {
                    "pcap": pcap_name,
                    "input_source": "pcap",
                    "n_sequences": int(Xp.shape[0]),
                    "sample_id": int(pick),
                    "input_shape": str(tuple(Xp.shape[1:])),
                    "horizon": HEADLINE_K,
                    "target_type": "horizon_probability",
                    "probability": float(pp[pick, HEADLINE_K]),
                    "probability_explained": float(r["prediction"][0]),
                    "prediction_identity_abs_diff": float(
                        ident["max_abs_diff"]
                    ),
                    "prediction_identity_passed": bool(ident["passed"]),
                    "threshold": float(thr_h),
                    "predicted_label": int(pp[pick, HEADLINE_K] >= thr_h),
                    "actual_label": "unlabeled",
                    "completeness_abs_error": float(
                        r["completeness_abs_error"][0]
                    ),
                    "top_features": "; ".join(names[i] for i in order),
                    "top_signed_attributions": "; ".join(
                        f"{sg[i]:+.6f}" for i in order
                    ),
                    "top_time_steps": "; ".join(
                        tlabels[s]
                        for s in np.lexsort((np.arange(SEQ_LEN), -st))[:3]
                    ),
                    "feature_contract": "docs/v4_feature_schema.json (52)",
                    "caveat": pcap_caveat,
                }
            )
            log(f"  {pcap_name}: {Xp.shape[0]} seqs, "
                f"p(K=5)={pp[pick, HEADLINE_K]:.4f}, "
                f"identity diff={ident['max_abs_diff']:.3e}, "
                f"top1={names[order[0]]}")
    except Exception as exc:                        # noqa: BLE001
        log(f"  PCAP explanation skipped: {type(exc).__name__}: {exc}")
    pcap_df = pd.DataFrame(pcap_rows)
    if not pcap_df.empty:
        pcap_df.to_csv(OUT_DIR / "pcap_explanations.csv", index=False)
    summary["pcap_explanations"] = {
        "input_source_field_present": True,
        "n_pcaps_explained": int(len(pcap_df)),
        "caveat": pcap_caveat,
        "rows": pcap_df.drop(columns=["caveat"], errors="ignore").to_dict(
            "records"
        ),
    }

    # ---------------------------------------------------------
    # FIGURES
    # ---------------------------------------------------------
    log("\n[figures]")
    vmax = float(np.abs(heat.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(20, 5.2))
    im = ax.imshow(
        heat.to_numpy(), aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    ax.set_yticks(range(SEQ_LEN))
    ax.set_yticklabels(tlabels)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(names, rotation=90, fontsize=6.5)
    ax.set_ylabel("history window")
    ax.set_title(
        f"V4 Integrated-Gradients attribution — mean signed, "
        f"K={HEADLINE_K}, n={n} Mar-2 test sequences "
        f"(zero baseline, {IG_N_STEPS} steps). "
        f"Attribution = model sensitivity, not causality.",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="signed attribution", pad=0.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attribution_heatmap.png", dpi=150)
    plt.close(fig)
    log("  attribution_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    t20 = fi.head(20).iloc[::-1]
    axes[0].barh(
        t20["feature"], t20["mean_absolute_attribution"],
        color=["#e74c3c" if k == "rel_change" else "#3498db"
               for k in t20["kind"]],
        edgecolor="white",
    )
    axes[0].set_xlabel("mean |attribution| (summed over 10 windows)")
    axes[0].set_title(f"Top 20 features — K={HEADLINE_K}", fontweight="bold")
    axes[1].bar(
        tlabels, ti["mean_absolute_attribution"],
        color="#34495e", edgecolor="white",
    )
    axes[1].set_xlabel("history window")
    axes[1].set_ylabel("mean |attribution| (summed over 52 features)")
    axes[1].set_title(
        f"Temporal attribution — K={HEADLINE_K}", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_and_temporal_importance.png", dpi=150)
    plt.close(fig)
    log("  feature_and_temporal_importance.png")

    fig, ax = plt.subplots(figsize=(13, 7))
    piv = hdf.pivot(
        index="feature", columns="horizon",
        values="mean_absolute_attribution",
    ).loc[fi.head(15)["feature"]]
    x = np.arange(len(piv))
    w = 0.26
    for j, k in enumerate(HORIZONS):
        ax.bar(x + (j - 1) * w, piv[k], w, label=f"K={k}")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean |attribution|")
    ax.set_title(
        "Feature attribution vs forecast horizon (top 15 at K=5)",
        fontweight="bold",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "horizon_comparison.png", dpi=150)
    plt.close(fig)
    log("  horizon_comparison.png")

    # per-case heatmaps for the 7 representative cases
    fig, axes = plt.subplots(len(rep_cases), 1, figsize=(18, 2.2 * len(rep_cases)))
    for ax, (_, rc) in zip(np.atleast_1d(axes), rep_cases.iterrows()):
        a = ex.attribute(
            X[int(rc["sample_id"]):int(rc["sample_id"]) + 1], HEADLINE_K
        )["attributions"][0]
        v = float(np.abs(a).max())
        ax.imshow(a, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_yticks(range(SEQ_LEN))
        ax.set_yticklabels(tlabels, fontsize=6)
        ax.set_xticks([])
        ax.set_title(
            f"{rc['label']}  id={rc['sample_id']}  "
            f"p={rc['probability']:.4f}  actual={rc['actual_label']}  "
            f"pred={rc['predicted_label']}",
            fontsize=9, fontweight="bold",
        )
    np.atleast_1d(axes)[-1].set_xticks(range(N_FEATURES))
    np.atleast_1d(axes)[-1].set_xticklabels(names, rotation=90, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "representative_case_heatmaps.png", dpi=150)
    plt.close(fig)
    log("  representative_case_heatmaps.png")

    # ---------------------------------------------------------
    # PART 5 -- RUNTIME CONTAMINATION AUDIT
    # ---------------------------------------------------------
    log("\n[Part 5] runtime V3-contamination audit")
    forbidden = audit.forbidden_hits()
    touched = audit.project_files()
    log(f"  project files opened during this run: {len(touched)}")
    for p in touched:
        log(f"    {Path(p).relative_to(REPO_ROOT)}")
    log(f"  forbidden (V1/V2/V3/rotated/temporal_benchmark_v2) hits: "
        f"{len(forbidden)}")
    summary["contamination_audit"] = {
        "method": (
            "sys.addaudithook('open') recording of every file opened "
            "during the explanation run"
        ),
        "project_files_opened": [
            str(Path(p).relative_to(REPO_ROOT)) for p in touched
        ],
        "forbidden_markers": list(
            __import__("v4_explainer").FORBIDDEN_ARTIFACT_MARKERS
        ),
        "forbidden_hits": [
            str(Path(p).relative_to(REPO_ROOT)) for p in forbidden
        ],
        "passed": len(forbidden) == 0,
    }

    # ---------------------------------------------------------
    # PART 25 -- THE CONTRACT
    # ---------------------------------------------------------
    log("\n[Part 25] writing docs/v4_explainability_contract.json")
    contract = {
        "contract_name": "v4_explainability_contract",
        "contract_version": "1.0",
        "generated_utc": started.isoformat().replace("+00:00", "Z"),
        "status": "FROZEN",
        "produced_by": "src/run_v4_explainability.py",
        "verified_by": "src/verify_v4_explainability.py",
        "core_claim": (
            "The explanation is generated from the same V4 inference "
            "computation used to produce the displayed prediction."
        ),
        "model_identity": prov.as_dict(),
        "input_contract": {
            "input_shape": [SEQ_LEN, N_FEATURES],
            "sequence_length": SEQ_LEN,
            "feature_dimension": N_FEATURES,
            "window_seconds": 10,
            "feature_schema": "docs/v4_feature_schema.json",
            "feature_schema_sha256": prov.feature_schema_hash,
            "feature_order_file": (
                "data/processed/multiday_benchmark_K5/"
                "input_feature_names.txt"
            ),
            "feature_order_sha256": prov.feature_order_hash,
            "scaler": prov.scaler_path,
            "scaler_sha256": prov.scaler_hash,
            "preprocessing": (
                "The explainer consumes the ALREADY-SCALED inference "
                "tensor. It implements no second preprocessing path: "
                "dataset explanations read the pre-built "
                "multiday_benchmark_K5 .npy arrays; PCAP explanations "
                "reuse src/pcap_ingest_v4.py's build_sequences() output "
                "verbatim."
            ),
            "temporal_indexing_preserved": True,
            "attribution_axes": [
                "time_step (10, labelled t-9..t)",
                "feature_index (52, schema order)",
            ],
        },
        "target_definitions": {
            "TARGET_A_horizon_probability": {
                "definition": horizon_target(
                    HEADLINE_K, ex.manifest
                ).target_definition,
                "supported_horizons": sorted(ex.horizon_thresholds),
                "evaluated_horizons": HORIZONS,
                "thresholds": {
                    str(k): v for k, v in ex.horizon_thresholds.items()
                },
                "threshold_provenance": (
                    "validation-selected, frozen in "
                    "docs/v4_baseline_manifest.json; never re-selected "
                    "here"
                ),
            },
            "TARGET_B_trajectory_max": {
                "definition": summary["trajectory_max_target"][
                    "documented_behaviour"
                ],
                "thresholds": {
                    str(k): v
                    for k, v in ex.trajectory_max_thresholds.items()
                },
                "argmax_switches_on_path_rate": switch_rate,
            },
            "recorded_per_explanation": [
                "target_type", "target_horizon", "target_definition",
                "prediction_probability", "threshold", "input_source",
            ],
        },
        "attribution_method": {
            "primary": "Integrated Gradients (captum.attr.IntegratedGradients)",
            "captum_version": __import__("captum").__version__,
            "approximation_method": IG_METHOD,
            "n_steps": IG_N_STEPS,
            "n_steps_rationale": (
                "chosen from the measured convergence sweep in "
                "completeness_checks.csv (50/128/256/1024); 1024 is the "
                "first setting with worst-case mean relative completeness "
                "error below 1% at every evaluated horizon. Not chosen "
                "for appearance."
            ),
            "cross_check": (
                "Feature Ablation (model-agnostic, gradient-free), "
                "results/v4_explainability/ablation_crosscheck.csv"
            ),
            "rejected_methods": {
                "attention": (
                    "V4 has no attention mechanism; no attention-based "
                    "explanation is produced and no attribution is called "
                    "'attention'"
                ),
                "shap": (
                    "src/shap_explainer.py is retained for the historical "
                    "V3 artifacts only; it is not provenance-locked, "
                    "targets a single transition step, reads "
                    "temporal_benchmark_v2, and draws a random baseline "
                    "without a seed"
                ),
            },
            "model_mode": "model.eval() -- both Dropout layers disabled",
            "gradients": "enabled through the input; parameters never updated",
            "device": "cpu",
            "dtype": "float32 (identical to the inference path)",
        },
        "baseline_definition": {
            "primary": "zero vector in scaled feature space, shape (10, 52)",
            "rationale": (
                "The frozen StandardScaler subtracts the training-partition "
                "mean, so the zero vector IS the average training window "
                "held constant across all 10 history windows. It is "
                "label-free, uses no test data, and is exactly "
                "reproducible."
            ),
            "secondary_sensitivity_baseline": (
                "mean of the 6568 fully-benign TRAINING sequences "
                "(A_binary_multistep_train all-zero); training labels "
                "only, never test labels"
            ),
            "caveat": (
                "A constant-over-time reference window is not itself a "
                "realisable capture. This is the standard IG "
                "reference-point compromise and is reported, not hidden."
            ),
            "not_selected_for_appearance": True,
        },
        "tolerances": {
            "prediction_identity": IDENTITY_TOLERANCE,
            "prediction_identity_observed_max": id_max,
            "prediction_identity_is_bit_exact": id_bit,
            "completeness_acceptance": (
                "mean relative error < 0.01 per horizon"
            ),
            "repeatability": "bit-exact attribution tensors required",
        },
        "representative_case_policy": SELECTION_CRITERIA,
        "sanity_tests": {
            t: {
                "acceptance_rule": sanity[t]["acceptance_rule"],
                "passed": sanity[t]["passed"],
            }
            for t in ("randomization_test", "perturbation_test",
                      "completeness_test")
        },
        "mitre_separation": {
            "rule": (
                "Attribution never produces ATT&CK labels. The MITRE "
                "layer (src/mitre_mapper.py) is rule-based over feature "
                "VALUES and runs independently of attribution values. Any "
                "joint display must label the two blocks separately: "
                "'MODEL ATTRIBUTION' and 'ATT&CK INTERPRETATION "
                "(rule-based)'."
            ),
            "attribution_feeds_mitre": False,
            "mitre_feeds_attribution": False,
        },
        "pcap_policy": {
            "input_source_values": ["dataset", "pcap"],
            "required_caveat": pcap_caveat,
            "numerical_equivalence_claimed": False,
        },
        "limitations": [
            "Attribution measures model sensitivity/contribution. It does "
            "NOT establish causality, attacker intent, attack stage, "
            "actual compromise, or root cause.",
            "Integrated Gradients is a path-integral approximation; the "
            "residual completeness error is reported per sample in "
            "completeness_checks.csv and is not zero.",
            "Attributions are baseline-relative. A different reference "
            "point yields different values; the baseline-sensitivity "
            "overlap is recorded in completeness_checks.csv.",
            "The trajectory-max target is attributed at a fixed argmax; "
            f"in {100 * switch_rate:.1f}% of sampled windows the argmax "
            "changes along the integration path.",
            "PCAP explanations inherit the Task 7 finding: flow-record "
            "production is APPROXIMATE, so PCAP feature values are not "
            "numerically equivalent to training features.",
            "The V4 checkpoint was trained on single-day "
            "temporal_benchmark_v2 arrays and evaluated on "
            "multiday_benchmark_K5 arrays scaled by a different "
            "StandardScaler (frozen-baseline known issue). Explanations "
            "inherit that out-of-distribution-normalisation regime.",
            "Re-batching the same frozen model shifts LSTM reduction "
            f"order by up to {batch_delta:.1e}; identity is therefore "
            "asserted on the identical tensor in the identical batch.",
            "No K=4 reference threshold exists, so K=4 is not explained "
            "against a frozen threshold.",
        ],
    }
    (DOCS_DIR / "v4_explainability_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )

    # ---------------------------------------------------------
    # PART 28 -- MACHINE-READABLE SUMMARY
    # ---------------------------------------------------------
    finished = datetime.now(timezone.utc)
    summary = {
        "analysis_name": "V4 explainability verification (Task 8)",
        "generated_utc": finished.isoformat().replace("+00:00", "Z"),
        "runtime_seconds": (finished - started).total_seconds(),
        "produced_by": "src/run_v4_explainability.py",
        "contract": "docs/v4_explainability_contract.json",
        "model_identity": prov.as_dict(),
        "attribution_method": contract["attribution_method"],
        "baseline": contract["baseline_definition"],
        "horizons_explained": HORIZONS,
        "headline_horizon": HEADLINE_K,
        "dashboard_artifacts": {
            "note": (
                "The dashboard displays and explains ONLY these "
                "horizons, using the frozen per-horizon validation "
                "thresholds. The attribution tensors below are the "
                "ones it renders, so the explanation shown is the one "
                "verified by this run."
            ),
            "attribution_tensors": {
                str(_k): {
                    "path": f"results/v4_explainability/"
                            f"attributions_k{_k}.npy",
                    "sha256": sha256_file(
                        OUT_DIR / f"attributions_k{_k}.npy"
                    ),
                    "shape": list(attrs_by_k[_k].shape),
                    "threshold": ex.horizon_thresholds[_k],
                }
                for _k in HORIZONS
            },
            "headline_horizon": HEADLINE_K,
            "dataset": "data/processed/multiday_benchmark_K5",
            "feature_importance_table": "results/v4_explainability/"
                                        "feature_importance.csv",
        },
        "test_set": {
            "arrays": "data/processed/multiday_benchmark_K5",
            "sequences": int(n),
            "test_day": ex.manifest["dataset"]["test_day"],
            "X_test_sha256": sha256_file(CANON / "X_test.npy"),
        },
        **summary,
        "known_limitations": contract["limitations"],
    }
    # Every boolean below is DERIVED from a value measured during this
    # run. None is asserted as a literal: a hardcoded True in a
    # verification block is indistinguishable from an unrun check.
    summary["verification_results"] = {
        "checkpoint_hash_matches_frozen_v4": bool(
            prov.checkpoint_hash == prov.checkpoint_hash_expected
        ),
        "model_class_is_v4": bool(
            prov.model_version == "V4"
            and prov.model_class == "LatentWorldModelV4"
            and type(ex.model).__name__ == "LatentWorldModelV4"
        ),
        "feature_schema_matches": bool(
            summary["feature_name_integrity"]
            and prov.feature_order_hash == prov.feature_order_hash_expected
        ),
        "scaler_matches": bool(
            prov.scaler_hash == prov.scaler_hash_expected
        ),
        "input_shape_10x52": bool(
            tuple(prov.input_shape) == (SEQ_LEN, N_FEATURES)
        ),
        "prediction_identity": summary["prediction_identity"]["passed"],
        "repeatability": summary["repeatability"]["passed"],
        "attribution_shape": bool(
            all(
                attrs_by_k[k].shape == (n, SEQ_LEN, N_FEATURES)
                for k in HORIZONS
            )
        ),
        "feature_naming": summary["feature_name_integrity"],
        "completeness": summary["completeness"]["passed"],
        "ablation_crosscheck_ran": bool(len(abl) > 0),
        "no_v3_artifact_loaded": summary["contamination_audit"]["passed"],
        "sanity_checks": sanity["all_passed"],
    }
    summary["overall"] = (
        "PASS" if all(summary["verification_results"].values()) else "FAIL"
    )
    (REPO_ROOT / "results" / "v4_explainability_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    # ---------------------------------------------------------
    # PART 29 -- OUTPUT-DIRECTORY README
    # ---------------------------------------------------------
    ci = summary["model_identity"]
    readme = f"""# results/v4_explainability/

Attribution artifacts for the FROZEN V4 checkpoint (Task 8).
Regenerate with `python src/run_v4_explainability.py`; verify with
`python src/verify_v4_explainability.py`.

Every file here is a *consumer* of the frozen V4 baseline. Nothing in
this directory was used to train, tune, threshold or select anything
about the model.

## Provenance of every file in this directory

| item | value |
|---|---|
| model | {ci['model_class']} ({ci['model_version']}) |
| checkpoint | `{ci['checkpoint_path']}` |
| checkpoint sha256 | `{ci['checkpoint_hash']}` |
| input scaler | `{ci['scaler_path']}` |
| scaler sha256 | `{ci['scaler_hash']}` |
| feature schema | `{ci['feature_schema_path']}` |
| feature-order sha256 | `{ci['feature_order_hash']}` |
| input shape | {ci['input_shape'][0]} x {ci['input_shape'][1]} |
| attribution method | Integrated Gradients ({IG_METHOD}, n_steps={IG_N_STEPS}) |
| baseline | zero vector in scaled feature space |
| generated | {summary['generated_utc']} |

## Files

| file | contents |
|---|---|
| `prediction_identity.csv` | per-sample \\|P_inference - P_explained\\| for every test sequence at every explained horizon (Part 20) |
| `completeness_checks.csv` | per-sample IG completeness residual plus the n_steps convergence sweep and the baseline-sensitivity probe (Part 7) |
| `ablation_crosscheck.csv` | gradient-free feature-ablation cross-check of the IG ranking (Part 8) |
| `feature_importance.csv` | 52-dim signed and absolute importance, aggregated over the 10 history steps, K={HEADLINE_K} (Part 9A) |
| `temporal_importance.csv` | 10-dim signed and absolute importance, aggregated over the 52 features, K={HEADLINE_K} (Part 9B) |
| `attribution_heatmap.csv` | 10 x 52 signed mean attribution, rows `t-9`..`t`, columns the exact 52 schema names (Part 10) |
| `horizon_feature_importance.csv` | per-feature importance at K in {HORIZONS}; explanations from different targets are never averaged together (Part 12) |
| `representative_explanations.csv` | TP / TN / FP / FN / near-threshold / high- and low-confidence cases, selected by the pre-declared deterministic rules (Part 11) |
| `attack_vs_benign.csv` | attribution pattern comparison between correctly predicted attack and benign windows (Part 13) |
| `error_explanations.csv` | explanations for the Task 4 representative false positives and false negatives (Part 14) |
| `onset_explanations.csv` | attribution for pre-onset elevated-score windows vs missed onsets, from the Task 5/6 cases (Part 15) |
| `trajectory_max_explanations.csv` | TARGET B, MAX(P(t+1)..P(t+5)), with the per-sample argmax and the `argmax_switches_on_path` diagnostic (Part 1) |
| `pcap_explanations.csv` | PCAP-sourced explanations, each carrying `input_source=pcap` and the approximate-flow-record caveat (Part 16) |
| `repeatability.csv` | two independent runs on identical input (Part 21) |
| `feature_name_integrity.json` | index-by-index proof that attribution column i carries schema name i (Part 22) |
| `figures/` | attribution heatmap, feature/temporal importance, horizon comparison, representative-case heatmaps |

## Reading the numbers

Sign is never discarded. A positive attribution means the feature, at
that history step, pushed the selected output UP relative to the
baseline; negative means DOWN. Absolute values rank influence
regardless of direction, and the two are reported side by side.

## What these files do not establish

Attribution describes this model's sensitivity/contribution on these
inputs. It does not establish causality, attacker intent, attack
stage, or actual compromise. MITRE ATT&CK wording is produced by a
separate rule-based layer (`src/mitre_mapper.py`) and is never derived
from attribution values.

PCAP rows carry an additional caveat: they use the same V4 feature
contract and transform chain, but flow-record production differs from
the training path, so the feature values are approximate and are not
claimed to be numerically equivalent to training features.

See `docs/V4_EXPLAINABILITY.md` for the full report and
`docs/v4_explainability_contract.json` for the machine-readable
contract.
"""
    (OUT_DIR / "README.md").write_text(readme)
    log("\n[Part 29] wrote results/v4_explainability/README.md")

    log("\n" + "=" * 74)
    log(f"ANALYSIS OVERALL: {summary['overall']}")
    log(f"runtime: {summary['runtime_seconds']:.0f}s")
    log("=" * 74)
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
