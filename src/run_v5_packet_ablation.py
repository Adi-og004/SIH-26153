"""
=============================================================
V5 PACKET-BLOCK CONTRIBUTION TEST  (Task 9)
=============================================================

Measures how much the frozen V5 checkpoint's output depends on the 30-D
packet suffix versus the 52-D V4 flow prefix, on the identical 2,692
mixed-scope sequences built in Task 8.

Read-only attribution diagnostic. Nothing is trained, refitted or tuned;
the checkpoint, both scalers, the 0.5 threshold and every feature
definition are used exactly as frozen. No labels are constructed and no
detection performance is claimed.

-------------------------------------------------------------
ABLATION METHOD — STATED EXACTLY
-------------------------------------------------------------
A block is "neutralised" by overwriting its RAW feature values with that
feature's own value from the frozen V5 input scaler, `scaler.mean_`, and
then letting `V5Inference` apply its normal preprocessing exactly once.

Because the engine standardises with z = (x - mean_) / scale_, a raw value
of mean_ maps to exactly 0.0 in scaler space. Substituting raw means is
therefore mathematically identical to writing standardised zeros, but it is
done on the RAW input so the model's preprocessing path is exercised
unchanged and is applied only once. This is verified numerically at
runtime (the scaled block must be exactly 0).

A neutralised block carries no sequence-specific and no temporal
information: every sequence, and every one of its 10 timesteps, receives
the identical training-mean vector. What survives is the model's response
to the block that was left intact.

Conditions:
    A ORIGINAL        82-D mixed-scope input, untouched
    B PACKET_ABLATED  features 0:52 intact,  52:82 -> scaler mean
    C PREFIX_ABLATED  features 52:82 intact, 0:52  -> scaler mean
    D BOTH_NEUTRAL    all 82 -> scaler mean  (sanity reference; every
                      sequence becomes identical, so all outputs must be
                      constant — checked at runtime)

Usage:
    python src/run_v5_packet_ablation.py

=============================================================
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer_latent_world_model_v5 import V5Inference

REPO_ROOT = SRC_DIR.parent
MIXED_DIR = REPO_ROOT / "results" / "v5_mixed_scope_diagnostic"
OUT_DIR = REPO_ROOT / "results" / "v5_packet_ablation"

X_PATH = MIXED_DIR / "X.npy"
METADATA_PATH = MIXED_DIR / "metadata.csv"

K = 5
PREFIX = slice(0, 52)
SUFFIX = slice(52, 82)
THRESHOLDS = (0.1, 0.25, 0.5)
DELTA_BANDS = (0.01, 0.05, 0.10)

# Same contextual metadata as Task 8 — CIC's documented Bot sessions,
# Atlantic local 10:11-11:34 and 14:24-15:55, at UTC-4.
BOT_SESSIONS_UTC = [("14:11", "15:34"), ("18:24", "19:55")]

CAVEAT = (
    "Attribution diagnostic only. The substrate has no per-window labels, "
    "so no accuracy, precision, recall, F1 or ROC is computed and no "
    "detection claim is made. A neutralised block is set to the frozen "
    "scaler mean, which is exactly 0 in scaler space; it therefore carries "
    "no sequence-specific or temporal information."
)


def _rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def run_condition(engine: V5Inference, X: np.ndarray, label: str) -> dict:
    """Validate the input, then run the frozen model over it in batches."""
    n = X.shape[0]
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan or n_inf:
        raise RuntimeError(f"{label}: non-finite input ({n_nan} NaN, {n_inf} Inf)")

    probs = np.empty((n, K))
    latents = np.empty((n, K, engine.latent_dim))
    current = np.empty((n, engine.latent_dim))
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        res = engine.k_step_rollout(X[start:stop], steps=K)
        probs[start:stop] = res.threat_probabilities
        latents[start:stop] = res.future_latents
        current[start:stop] = res.current_latent

    print(f"  [{label:<15}] shape {X.shape} | NaN {n_nan} | Inf {n_inf} "
          f"| scaler dims {X.shape[2]}=={engine.input_dim} "
          f"| non-finite out {int((~np.isfinite(probs)).sum())}")
    return {"probs": probs, "latents": latents, "current": current,
            "n_nan": n_nan, "n_inf": n_inf}


def distribution(probs: np.ndarray) -> dict:
    out = {}
    for k in range(K):
        c = probs[:, k]
        e = {"min": float(c.min()), "median": float(np.median(c)),
             "mean": float(c.mean()), "p95": float(np.percentile(c, 95)),
             "p99": float(np.percentile(c, 99)), "max": float(c.max())}
        for t in THRESHOLDS:
            e[f"count_ge_{t}"] = int((c >= t).sum())
            e[f"fraction_ge_{t}"] = float((c >= t).mean())
        out[f"t+{k+1}"] = e
    return out


def delta_stats(original: np.ndarray, ablated: np.ndarray) -> dict:
    d = np.abs(ablated - original)
    e = {"mean_abs_delta": float(d.mean()),
         "median_abs_delta": float(np.median(d)),
         "p95_abs_delta": float(np.percentile(d, 95)),
         "max_abs_delta": float(d.max())}
    for band in DELTA_BANDS:
        e[f"fraction_changing_gt_{band}"] = float((d > band).mean())
    return e


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(X_PATH).astype(np.float64)
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded {X_PATH.relative_to(REPO_ROOT)}  shape={X.shape}")
    if X.shape[0] != len(metadata):
        raise RuntimeError("X and metadata disagree on sequence count")

    engine = V5Inference()
    engine.load_model()
    mean = np.asarray(engine.input_scaler.mean_, dtype=np.float64)
    print(f"checkpoint {engine.checkpoint_path.name} | threshold "
          f"{engine.threshold} (frozen, untouched) | scaler NOT refitted")

    # ---------- build the four inputs ----------
    _rule("STEP 1 — construct conditions and validate inputs")
    X_packet = X.copy(); X_packet[:, :, SUFFIX] = mean[SUFFIX]
    X_prefix = X.copy(); X_prefix[:, :, PREFIX] = mean[PREFIX]
    X_both = np.broadcast_to(mean, X.shape).copy()

    # verify the neutralisation lands exactly on 0 in scaler space
    scaled_check = engine._scale_input(X_both[:1])
    if not np.allclose(scaled_check, 0.0, atol=1e-9):
        raise RuntimeError("raw-mean substitution did not map to 0 in scaler space")
    print("  raw-mean substitution -> exactly 0.0 in scaler space: VERIFIED "
          f"(max |z| {np.abs(scaled_check).max():.2e})")

    conditions = {
        "ORIGINAL": X, "PACKET_ABLATED": X_packet,
        "PREFIX_ABLATED": X_prefix, "BOTH_NEUTRAL": X_both,
    }
    results = {name: run_condition(engine, arr, name)
               for name, arr in conditions.items()}

    # sanity: BOTH_NEUTRAL must give an identical output for every sequence
    bn = results["BOTH_NEUTRAL"]["probs"]
    spread = float(bn.std(axis=0).max())
    print(f"  BOTH_NEUTRAL across-sequence spread (must be ~0): {spread:.2e}"
          f"  -> {'CONSTANT as expected' if spread < 1e-9 else 'UNEXPECTED VARIATION'}")

    # ---------- 2. distributions ----------
    _rule("STEP 2 — threat-probability distribution per condition")
    dists = {name: distribution(r["probs"]) for name, r in results.items()}
    rows = []
    for name, dist in dists.items():
        for horizon, e in dist.items():
            rows.append({"condition": name, "horizon": horizon, **e})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    print(f"  {'condition':<16}{'horizon':<8}{'median':>10}{'mean':>10}"
          f"{'p95':>10}{'max':>10}{'n>=0.5':>9}")
    for name in conditions:
        for horizon in (f"t+{k+1}" for k in range(K)):
            e = dists[name][horizon]
            print(f"  {name:<16}{horizon:<8}{e['median']:>10.5f}{e['mean']:>10.5f}"
                  f"{e['p95']:>10.5f}{e['max']:>10.5f}{e['count_ge_0.5']:>9}")

    # ---------- 3 + 4. pairwise and horizon-specific deltas ----------
    _rule("STEP 3/4 — prediction change vs ORIGINAL (overall and per horizon)")
    original = results["ORIGINAL"]["probs"]
    pairwise, horizon_rows = {}, []
    for name in ("PACKET_ABLATED", "PREFIX_ABLATED", "BOTH_NEUTRAL"):
        ablated = results[name]["probs"]
        pairwise[name] = {"overall": delta_stats(original, ablated), "by_horizon": {}}
        for k in range(K):
            e = delta_stats(original[:, k], ablated[:, k])
            pairwise[name]["by_horizon"][f"t+{k+1}"] = e
            horizon_rows.append({"condition": name, "horizon": f"t+{k+1}", **e})
    pd.DataFrame(horizon_rows).to_csv(OUT_DIR / "horizon_effects.csv", index=False)

    print(f"  {'condition':<16}{'mean|d|':>10}{'median|d|':>11}{'p95|d|':>10}"
          f"{'max|d|':>10}{'>0.01':>9}{'>0.05':>9}{'>0.10':>9}")
    for name, e in pairwise.items():
        o = e["overall"]
        print(f"  {name:<16}{o['mean_abs_delta']:>10.5f}"
              f"{o['median_abs_delta']:>11.5f}{o['p95_abs_delta']:>10.5f}"
              f"{o['max_abs_delta']:>10.5f}"
              f"{o['fraction_changing_gt_0.01']:>9.4f}"
              f"{o['fraction_changing_gt_0.05']:>9.4f}"
              f"{o['fraction_changing_gt_0.1']:>9.4f}")
    print(f"\n  per-horizon mean |delta|:")
    print(f"  {'condition':<16}" + "".join(f"{f't+{k+1}':>11}" for k in range(K)))
    for name, e in pairwise.items():
        print(f"  {name:<16}" + "".join(
            f"{e['by_horizon'][f't+{k+1}']['mean_abs_delta']:>11.5f}" for k in range(K)))

    # ---------- 5. sequence-level ----------
    _rule("STEP 5 — sequence-level effect (max probability across K=5)")
    original_max = original.max(axis=1)
    sequence_frame = pd.DataFrame({
        "sequence_index": metadata["sequence_index"],
        "timestamp_start_utc": metadata["timestamp_start_utc"],
        "original_max_probability": original_max,
    })
    sequence_effects = {}
    for name in ("PACKET_ABLATED", "PREFIX_ABLATED", "BOTH_NEUTRAL"):
        ablated_max = results[name]["probs"].max(axis=1)
        change = np.abs(ablated_max - original_max)
        flipped = (original_max >= engine.threshold) != (ablated_max >= engine.threshold)
        sequence_frame[f"{name}_max_probability"] = ablated_max
        sequence_frame[f"{name}_abs_change"] = change
        sequence_frame[f"{name}_alert_flipped"] = flipped
        sequence_effects[name] = {
            "median_abs_change": float(np.median(change)),
            "mean_abs_change": float(change.mean()),
            "p95_abs_change": float(np.percentile(change, 95)),
            "max_abs_change": float(change.max()),
            "n_alert_status_changed": int(flipped.sum()),
            "fraction_alert_status_changed": float(flipped.mean()),
        }
    sequence_frame.to_csv(OUT_DIR / "sequence_effects.csv", index=False)
    print(f"  {'condition':<16}{'median':>10}{'mean':>10}{'p95':>10}{'max':>10}"
          f"{'alerts flipped':>16}")
    for name, e in sequence_effects.items():
        print(f"  {name:<16}{e['median_abs_change']:>10.5f}{e['mean_abs_change']:>10.5f}"
              f"{e['p95_abs_change']:>10.5f}{e['max_abs_change']:>10.5f}"
              f"{e['n_alert_status_changed']:>9} / {len(original_max)}")

    # ---------- 6. latent effects ----------
    _rule("STEP 6 — latent-state effect")
    original_latents = results["ORIGINAL"]["latents"]
    latent_rows, latent_effects = [], {}
    for name in ("ORIGINAL", "PACKET_ABLATED", "PREFIX_ABLATED", "BOTH_NEUTRAL"):
        lat = results[name]["latents"]
        norms = np.linalg.norm(lat, axis=2)
        entry = {"condition": name,
                 "mean_latent_norm": float(norms.mean()),
                 "p95_latent_norm": float(np.percentile(norms, 95)),
                 "max_latent_norm": float(norms.max())}
        if name != "ORIGINAL":
            d = np.abs(lat - original_latents)
            entry.update({
                "mean_abs_latent_diff": float(d.mean()),
                "p95_abs_latent_diff": float(np.percentile(d, 95)),
                "max_abs_latent_diff": float(d.max()),
                "mean_latent_l2_distance": float(
                    np.linalg.norm(lat - original_latents, axis=2).mean()),
            })
        latent_rows.append(entry)
        latent_effects[name] = entry
    pd.DataFrame(latent_rows).to_csv(OUT_DIR / "latent_effects.csv", index=False)
    print(f"  {'condition':<16}{'mean|z|norm':>13}{'mean|diff|':>12}"
          f"{'p95|diff|':>11}{'max|diff|':>11}{'mean L2':>10}")
    for name, e in latent_effects.items():
        print(f"  {name:<16}{e['mean_latent_norm']:>13.4f}"
              f"{e.get('mean_abs_latent_diff', 0.0):>12.4f}"
              f"{e.get('p95_abs_latent_diff', 0.0):>11.4f}"
              f"{e.get('max_abs_latent_diff', 0.0):>11.4f}"
              f"{e.get('mean_latent_l2_distance', 0.0):>10.4f}")

    # ---------- 7. temporal context ----------
    _rule("STEP 7 — temporal context (documented Bot sessions, metadata only)")
    t = pd.to_datetime(metadata["timestamp_start_utc"])
    inside = np.zeros(len(metadata), dtype=bool)
    for lo, hi in BOT_SESSIONS_UTC:
        inside |= ((t >= pd.Timestamp(f"2018-03-02 {lo}:00"))
                   & (t <= pd.Timestamp(f"2018-03-02 {hi}:00"))).to_numpy()
    top20_original = np.argsort(original_max)[::-1][:20]
    temporal = {"baseline_fraction_inside_sessions": float(inside.mean())}
    print(f"  baseline: {inside.mean():.1%} of sequences lie inside a documented session")
    print(f"  {'condition':<16}{'top-20 inside':>15}{'top-20 retained':>18}"
          f"{'median max-p (top-20 orig windows)':>38}")
    for name in ("ORIGINAL", "PACKET_ABLATED", "PREFIX_ABLATED", "BOTH_NEUTRAL"):
        mx = results[name]["probs"].max(axis=1)
        top20 = np.argsort(mx)[::-1][:20]
        retained = len(set(top20) & set(top20_original))
        temporal[name] = {
            "top20_inside_sessions": int(inside[top20].sum()),
            "top20_retained_from_original": int(retained),
            "median_max_prob_on_original_top20": float(np.median(mx[top20_original])),
        }
        print(f"  {name:<16}{inside[top20].sum():>12}/20{retained:>15}/20"
              f"{np.median(mx[top20_original]):>38.5f}")

    # ---------- compact table ----------
    _rule("COMPACT COMPARISON")
    compact_rows = []
    print(f"  {'condition':<17}{'median t+1':>12}{'median t+5':>12}"
          f"{'p95 max-prob':>14}{'global max':>12}{'frac >=0.5':>12}")
    for name in ("ORIGINAL", "PACKET_ABLATED", "PREFIX_ABLATED", "BOTH_NEUTRAL"):
        p = results[name]["probs"]
        row = {"condition": name,
               "median_t1": float(np.median(p[:, 0])),
               "median_t5": float(np.median(p[:, 4])),
               "p95_max_probability": float(np.percentile(p.max(axis=1), 95)),
               "global_max_probability": float(p.max()),
               "fraction_ge_0.5": float((p >= 0.5).mean())}
        compact_rows.append(row)
        print(f"  {name:<17}{row['median_t1']:>12.5f}{row['median_t5']:>12.5f}"
              f"{row['p95_max_probability']:>14.5f}"
              f"{row['global_max_probability']:>12.5f}{row['fraction_ge_0.5']:>12.5f}")

    comparison = {
        "compact_table": compact_rows,
        "distributions": dists,
        "pairwise_vs_original": pairwise,
        "sequence_effects": sequence_effects,
        "latent_effects": latent_effects,
        "temporal_context": temporal,
        "caveat": CAVEAT,
    }
    (OUT_DIR / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    manifest = {
        "experiment": "V5 packet-block contribution test (Task 9)",
        "input": str(X_PATH.relative_to(REPO_ROOT)),
        "input_shape": list(X.shape),
        "n_sequences": int(X.shape[0]),
        "rollout_steps": K,
        "checkpoint": str(engine.checkpoint_path.relative_to(REPO_ROOT)),
        "scaler": str(engine.input_scaler_path.relative_to(REPO_ROOT)),
        "threshold": float(engine.threshold),
        "checkpoint_modified": False, "scaler_refitted": False,
        "threshold_tuned": False, "retrained": False,
        "ablation_method": (
            "A block is neutralised by overwriting its RAW values with that "
            "feature's frozen scaler mean_, then applying the engine's normal "
            "preprocessing exactly once. Since z = (x - mean_)/scale_, raw "
            "mean_ maps to exactly 0.0 in scaler space; verified at runtime. "
            "A neutralised block carries no sequence-specific or temporal "
            "information."
        ),
        "conditions": {
            "ORIGINAL": "82-D mixed-scope input, untouched",
            "PACKET_ABLATED": "features 52:82 -> scaler mean; 0:52 intact",
            "PREFIX_ABLATED": "features 0:52 -> scaler mean; 52:82 intact",
            "BOTH_NEUTRAL": "all 82 -> scaler mean (constant-output reference)",
        },
        "both_neutral_across_sequence_spread": spread,
        "bot_sessions_utc_context_only": BOT_SESSIONS_UTC,
        "labels_used": False,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "caveat": CAVEAT,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  wrote ablation_summary.csv, horizon_effects.csv, "
          f"sequence_effects.csv,\n        latent_effects.csv, comparison.json, "
          f"manifest.json -> {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"\n{CAVEAT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
