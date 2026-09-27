"""
=============================================================
V5 MIXED-SCOPE DIAGNOSTIC  (Task 8)
=============================================================

Tests whether the frozen V5 checkpoint behaves sensibly when the inference
input is deliberately built to match the SCOPE PATTERN that Task 7 inferred
from its scaler:

    columns  0-51 : organization-wide Mar-2 V4 state (states_v2)
    columns 52-81 : single-host packet features (host 172.31.69.30)

This reproduces the suspected TRAINING scope. It is explicitly NOT an
attempt to build a physically best real-world representation, and it is not
a benchmark: the substrate has no per-window labels, so no accuracy,
precision, recall, F1 or ROC is computed anywhere.

Nothing is trained, refitted or tuned. The checkpoint, both scalers, the
0.5 threshold and every feature definition are used exactly as frozen.

-------------------------------------------------------------
TIME ALIGNMENT — RECONSTRUCTED, THEN VERIFIED
-------------------------------------------------------------
The two sources use different time conventions:

  * packet host .30 : window_idx on true UTC epoch bins,
                      abs_bin = window_idx + win0 (win0 from the Task-3
                      manifest, itself derived from the pcap).
  * states_v2       : `window_start` is CIC's Atlantic-local clock with the
                      AM/PM marker DROPPED, so it parses into a meaningless
                      01:00-12:59 range and sorts non-chronologically.

The states_v2 file contains exactly ONE discontinuity (11,780 s at row
1622), splitting it into two blocks. Reconstruction: rows 1622..3142 are
the AM block (real local = clock), rows 0..1621 are the PM block (real
local = clock + 12 h). UTC = local + 4 h (America/Halifax, UTC-4 on
2018-03-02 — the offset already established in this repository from two
independent pcap anchors).

That reconstruction is VERIFIED here against three independent anchors
before any data is built; if any check fails the script stops. No timezone
offset is invented or fitted, and no window is interpolated.

-------------------------------------------------------------
FEATURE DEFINITIONS — REUSED, NOT REDEFINED
-------------------------------------------------------------
states_v2 carries the 26 BASE features only. The 26 relative-change
features are produced with the transform copied verbatim from
`src/prepare_multiday_benchmark.py` (L88-L95), applied in the file's own
row order so the result is bit-for-bit the frozen V4 representation. The
30 packet features are taken unchanged from the existing host-.30 table.

Usage:
    python src/build_v5_mixed_scope_diagnostic.py

=============================================================
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pcap_ingest_v4 as v4
import packet_feature_builder as pfb
from infer_latent_world_model_v5 import V5Inference

REPO_ROOT = SRC_DIR.parent
OUT_DIR = REPO_ROOT / "results" / "v5_mixed_scope_diagnostic"

STATES_V2 = (
    REPO_ROOT / "data" / "processed" / "states_v2"
    / "Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv"
)
PACKET_CSV = (
    REPO_ROOT / "results" / "v5_features"
    / "packet_windows_march02_host_172.31.69.30.csv"
)
SINGLE_MANIFEST = REPO_ROOT / "results" / "v5_demo" / "demo_manifest.json"
V5_SCALER = (
    REPO_ROOT / "data" / "processed" / "v5_multistep_K5" / "v5_input_scaler.pkl"
)

WINDOW_SECONDS = 10
SEQ_LEN = 10
V5_DIM = 82
K = 5

# Contextual metadata only — CIC's documented Bot sessions for 2018-03-02,
# Atlantic local 10:11-11:34 and 14:24-15:55, converted at UTC-4.
BOT_SESSIONS_UTC = [("14:11", "15:34"), ("18:24", "19:55")]

SCOPE_NOTE = (
    "MIXED-SCOPE DIAGNOSTIC SUBSTRATE. Columns 0-51 are organization-wide "
    "Mar-2 flow state (CIC states_v2, whole 172.31.0.0/16 victim network); "
    "columns 52-81 are single-host packet features (172.31.69.30). The two "
    "halves describe the SAME real 10 s windows but DIFFERENT observation "
    "scopes. This deliberately reproduces the scope pattern Task 7 inferred "
    "from the V5 scaler, to test the frozen checkpoint against the "
    "distribution it appears to have been fitted on. It is NOT a physically "
    "coherent sensor view and NOT a proposed production representation. No "
    "labels exist for it; no detection or accuracy claim can be made from it."
)


class DiagnosticError(RuntimeError):
    pass


def _rule(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# =============================================================
# STEP 1 — ALIGNMENT
# =============================================================


def load_orgwide() -> tuple[pd.DataFrame, dict]:
    """Load org-wide Mar-2 states, add the canonical 26 rel-change features,
    and reconstruct a true UTC epoch for every window."""
    _rule("STEP 1 — alignment audit")

    df = pd.read_csv(STATES_V2)
    base = list(v4.BASE_FEATURES)
    missing = [c for c in base if c not in df.columns]
    if missing:
        raise DiagnosticError(f"states_v2 missing base features: {missing}")

    # ---- canonical relative-change transform, copied verbatim from
    # src/prepare_multiday_benchmark.py L88-L95. Applied in the FILE's row
    # order so the 52-D vector is exactly the frozen V4 representation.
    state = df[base].apply(pd.to_numeric, errors="coerce")
    state = state.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    delta = state.diff().fillna(0.0)
    previous_state = state.shift(1)
    relative = delta / (previous_state.abs() + 1e-8)
    relative.iloc[0] = 0.0
    relative = relative.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
    relative.columns = [f"{f}_rel_change" for f in base]
    df = pd.concat([df[["window_start", "attack_present", "attack_ratio"]],
                    state, relative], axis=1)

    # ---- reconstruct true UTC from the AM/PM-stripped Atlantic clock ----
    clock = pd.to_datetime(df["window_start"])
    steps = clock.diff().dt.total_seconds()
    jumps = int((steps.dropna() != WINDOW_SECONDS).sum())
    if jumps != 1:
        raise DiagnosticError(
            f"expected exactly 1 clock discontinuity (the AM/PM wrap), found {jumps}"
        )
    boundary = int(np.nanargmax(steps.to_numpy()))

    # rows >= boundary are the AM block (local = clock);
    # rows <  boundary are the PM block (local = clock + 12h).
    offset_hours = np.where(np.arange(len(df)) >= boundary, 4, 16)
    utc = clock + pd.to_timedelta(offset_hours, unit="h")
    df["utc"] = utc
    # NB: pandas 3 yields datetime64[us] here, so derive epoch seconds
    # explicitly rather than assuming a nanosecond int64 view.
    epoch_seconds = (utc - pd.Timestamp("1970-01-01")).dt.total_seconds()
    df["abs_bin"] = (epoch_seconds // WINDOW_SECONDS).astype("int64")

    # ---- VERIFY the reconstruction against three independent anchors ----
    attacks = df[df["attack_present"] == 1]
    first_bot_utc = attacks["utc"].min()
    last_utc = df["utc"].max()

    checks = []
    # anchor 1: first Bot window vs the pcap-verified first C2 beacon
    expected_onset = pd.Timestamp("2018-03-02 14:13:20")
    checks.append((
        "first Bot window == 14:13:20 UTC (pcap first C2 = 14:13:28)",
        abs((first_bot_utc - expected_onset).total_seconds()) <= WINDOW_SECONDS,
        f"{first_bot_utc}",
    ))
    # anchor 2: last window vs the pcap capture end (21:30:03-21:30:05 UTC)
    checks.append((
        "last window ~ pcap capture end 21:30 UTC",
        abs((last_utc - pd.Timestamp("2018-03-02 21:30:10")).total_seconds()) <= 60,
        f"{last_utc}",
    ))
    # anchor 3: the two blocks must join seamlessly once re-ordered
    ordered = df.sort_values("abs_bin")
    gaps = np.diff(ordered["abs_bin"].to_numpy())
    checks.append((
        "AM and PM blocks join contiguously after reordering",
        bool((gaps == 1).all()),
        f"{int((gaps != 1).sum())} gaps",
    ))
    for label, ok, note in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({note})")
    if not all(ok for _, ok, _ in checks):
        raise DiagnosticError("UTC reconstruction failed verification — stopping")

    info = {
        "rows": int(len(df)),
        "clock_discontinuity_row": boundary,
        "am_block_rows": int(len(df) - boundary),
        "pm_block_rows": int(boundary),
        "utc_first": _utc(ordered["utc"].iloc[0].timestamp()),
        "utc_last": _utc(ordered["utc"].iloc[-1].timestamp()),
        "rel_change_boundary_rows_affected": 2,
    }
    print(f"\n  org-wide states_v2: {info['rows']} windows, "
          f"{info['utc_first']} .. {info['utc_last']} UTC (contiguous)")
    return ordered.reset_index(drop=True), info


def load_packet(win0: int) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(PACKET_CSV)
    df["abs_bin"] = df["window_idx"] + win0
    info = {
        "rows": int(len(df)),
        "win0": int(win0),
        "utc_first": _utc(int(df["abs_bin"].min()) * WINDOW_SECONDS),
        "utc_last": _utc(int(df["abs_bin"].max()) * WINDOW_SECONDS),
    }
    print(f"  packet host .30   : {info['rows']} windows, "
          f"{info['utc_first']} .. {info['utc_last']} UTC (gaps present)")
    return df, info


# =============================================================
# MAIN
# =============================================================


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = pickle.load(open(V5_SCALER, "rb"))
    names = [str(n) for n in np.asarray(scaler.feature_names_in_).ravel()]
    packet_names = names[52:]
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    groups = (["V4_BASE"] * 26 + ["V4_RELATIVE"] * 26
              + ["PACKET_BASE"] * 21 + ["PACKET_RELATIVE"] * 9)
    if names[:52] != list(v4.INPUT_FEATURE_NAMES):
        raise DiagnosticError("V5 prefix order != frozen V4 feature order")
    if packet_names != list(pfb.PACKET_BASE_FEATURES) + list(pfb.PACKET_RELATIVE_FEATURES):
        raise DiagnosticError("V5 packet order != frozen packet contract order")

    win0 = int(json.loads(SINGLE_MANIFEST.read_text())["win0"])
    org, org_info = load_orgwide()
    packet, pkt_info = load_packet(win0)

    # ---------- STEP 2 ----------
    _rule("STEP 2 — mixed-scope 82-D construction")
    joined = org.merge(
        packet[["abs_bin"] + packet_names], on="abs_bin", how="inner"
    ).sort_values("abs_bin").reset_index(drop=True)
    overlap = {
        "aligned_windows": int(len(joined)),
        "utc_first": _utc(int(joined["abs_bin"].min()) * WINDOW_SECONDS),
        "utc_last": _utc(int(joined["abs_bin"].max()) * WINDOW_SECONDS),
    }
    print(f"  aligned windows (inner join on absolute 10s bin): "
          f"{overlap['aligned_windows']}")
    print(f"  overlap range: {overlap['utc_first']} .. {overlap['utc_last']} UTC")
    print(f"  org-wide only : {len(org) - len(joined)} windows dropped")
    print(f"  packet only   : {len(packet) - len(joined)} windows dropped")

    values = joined[names].to_numpy(dtype=np.float64)
    if values.shape[1] != V5_DIM:
        raise DiagnosticError(f"expected 82 columns, got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise DiagnosticError("non-finite values in the constructed matrix")
    print(f"  matrix: {values.shape}  (52 org-wide + 30 single-host = 82) "
          f"| finite: True | feature order == scaler: True")

    # ---------- STEP 3 ----------
    _rule("STEP 3 — contiguous 10-window sequences")
    bins = joined["abs_bin"].to_numpy(dtype=np.int64)
    breaks = np.flatnonzero(np.diff(bins) != 1) + 1
    runs = [r for r in np.split(np.arange(len(bins)), breaks) if len(r) >= SEQ_LEN]
    sequences, rows = [], []
    for run in runs:
        for start in range(len(run) - SEQ_LEN + 1):
            idx = run[start : start + SEQ_LEN]
            if not np.array_equal(np.diff(bins[idx]), np.ones(SEQ_LEN - 1)):
                raise DiagnosticError("non-contiguous sequence")
            sequences.append(values[idx])
            a, b = int(bins[idx[0]]), int(bins[idx[-1]])
            rows.append({
                "sequence_index": len(sequences) - 1,
                "abs_bin_start": a, "abs_bin_end": b,
                "timestamp_start_utc": _utc(a * WINDOW_SECONDS),
                "timestamp_end_utc": _utc(b * WINDOW_SECONDS),
                "scope": "orgwide_v4_prefix + singlehost_packet_suffix",
                "v4_source": "states_v2 Friday-02-03-2018 (org-wide)",
                "packet_source": "host 172.31.69.30 (single host)",
                "orgwide_attack_present_last_window":
                    int(joined["attack_present"].iloc[idx[-1]]),
            })
    if not sequences:
        raise DiagnosticError("no contiguous 10-window sequence exists")
    X = np.stack(sequences).astype(np.float64)
    metadata = pd.DataFrame(rows)
    print(f"  contiguous runs >= 10: {len(runs)} | sequences: {len(X)} | X {X.shape}")

    # ---------- STEP 4 ----------
    _rule("STEP 4 — scaler-domain shift (frozen V5 scaler, not refitted)")
    z = (values - mean) / scale
    absolute = np.abs(z)
    report_rows = []
    for j, name in enumerate(names):
        col = absolute[:, j]
        report_rows.append({
            "feature_index": j, "feature_name": name, "feature_group": groups[j],
            "training_mean": mean[j], "training_scale": scale[j],
            "demo_mean": values[:, j].mean(), "demo_std": values[:, j].std(ddof=1),
            "mean_abs_z": col.mean(), "median_abs_z": float(np.median(col)),
            "p95_abs_z": float(np.percentile(col, 95)),
            "fraction_abs_z_gt_3": float((col > 3).mean()),
            "signed_mean_z": z[:, j].mean(),
        })
    report = pd.DataFrame(report_rows)
    report.to_csv(OUT_DIR / "input_shift_report.csv", index=False)

    def block(mask):
        b = absolute[:, mask]
        return {
            "mean_abs_z": float(b.mean()),
            "median_abs_z": float(np.median(b)),
            "p95_abs_z": float(np.percentile(b, 95)),
            "fraction_abs_z_gt_3": float((b > 3).mean()),
        }

    prefix_mask = np.array([g.startswith("V4") for g in groups])
    suffix_mask = ~prefix_mask
    shift = {
        "A_v4_prefix": block(prefix_mask),
        "B_packet_suffix": block(suffix_mask),
        "C_all_82": block(np.ones(82, dtype=bool)),
    }
    print(f"  {'block':<20}{'mean|z|':>10}{'median':>10}{'p95':>10}{'frac>3':>10}")
    for label, entry in shift.items():
        print(f"  {label:<20}{entry['mean_abs_z']:>10.3f}"
              f"{entry['median_abs_z']:>10.3f}{entry['p95_abs_z']:>10.3f}"
              f"{entry['fraction_abs_z_gt_3']:>10.4f}")

    # ---------- STEP 5 ----------
    _rule("STEP 5 — frozen V5 inference (unchanged checkpoint/scaler/threshold)")
    engine = V5Inference()
    engine.load_model()
    print(f"  threshold (frozen): {engine.threshold}")
    probs = np.empty((X.shape[0], K))
    latents = np.empty((X.shape[0], K, engine.latent_dim))
    for start in range(0, X.shape[0], 256):
        stop = min(start + 256, X.shape[0])
        res = engine.k_step_rollout(X[start:stop], steps=K)
        probs[start:stop] = res.threat_probabilities
        latents[start:stop] = res.future_latents

    nonfinite = int((~np.isfinite(probs)).sum() + (~np.isfinite(latents)).sum())
    norms = np.linalg.norm(latents, axis=2)
    print(f"  sequences: {X.shape[0]} | non-finite outputs: {nonfinite}")
    print(f"  latent norm by horizon (mean): "
          + ", ".join(f"t+{k+1} {norms[:, k].mean():.3f}" for k in range(K)))
    print(f"  latent norm max {norms.max():.3f} | exploding(>1e3): {norms.max() > 1e3}")

    distribution = {}
    print(f"\n  {'horizon':<9}{'min':>10}{'median':>11}{'mean':>11}{'p95':>11}"
          f"{'p99':>11}{'max':>11}{'n>=0.1':>8}{'n>=0.25':>9}{'n>=0.5':>8}")
    for k in range(K):
        c = probs[:, k]
        e = {
            "min": float(c.min()), "median": float(np.median(c)),
            "mean": float(c.mean()), "p95": float(np.percentile(c, 95)),
            "p99": float(np.percentile(c, 99)), "max": float(c.max()),
            "n_ge_0.1": int((c >= 0.1).sum()), "fraction_ge_0.1": float((c >= 0.1).mean()),
            "n_ge_0.25": int((c >= 0.25).sum()), "fraction_ge_0.25": float((c >= 0.25).mean()),
            "n_ge_0.5": int((c >= 0.5).sum()), "fraction_ge_0.5": float((c >= 0.5).mean()),
        }
        distribution[f"t+{k+1}"] = e
        print(f"  t+{k+1:<8}{e['min']:>10.6f}{e['median']:>11.6f}{e['mean']:>11.6f}"
              f"{e['p95']:>11.6f}{e['p99']:>11.6f}{e['max']:>11.6f}"
              f"{e['n_ge_0.1']:>8}{e['n_ge_0.25']:>9}{e['n_ge_0.5']:>8}")

    any_ge = {f"any_horizon_ge_{t}": int((probs >= t).any(axis=1).sum())
              for t in (0.1, 0.25, 0.5)}
    print("\n  sequences with ANY horizon >= threshold: "
          + ", ".join(f"{k.split('_ge_')[1]}: {v}" for k, v in any_ge.items()))

    rollout = pd.DataFrame({
        "sequence_index": metadata["sequence_index"],
        "abs_bin_start": metadata["abs_bin_start"],
        "timestamp_start_utc": metadata["timestamp_start_utc"],
        **{f"p_t+{k+1}": probs[:, k] for k in range(K)},
        "max_probability": probs.max(axis=1),
        "mean_probability": probs.mean(axis=1),
    })
    rollout.to_csv(OUT_DIR / "rollout_summary.csv", index=False)

    # ---------- STEP 6 ----------
    _rule("STEP 6 — temporal co-occurrence (contextual metadata only)")
    t = pd.to_datetime(rollout["timestamp_start_utc"])
    in_session = np.zeros(len(rollout), dtype=bool)
    for lo, hi in BOT_SESSIONS_UTC:
        in_session |= ((t >= pd.Timestamp(f"2018-03-02 {lo}:00"))
                       & (t <= pd.Timestamp(f"2018-03-02 {hi}:00"))).to_numpy()
    inside = rollout.loc[in_session, "max_probability"]
    outside = rollout.loc[~in_session, "max_probability"]
    pct = float((rollout["max_probability"] < inside.median()).mean() * 100) if len(inside) else float("nan")
    print(f"  documented Bot sessions (UTC): {BOT_SESSIONS_UTC}")
    print(f"  sequences inside sessions : {int(in_session.sum())}")
    print(f"  sequences outside         : {int((~in_session).sum())}")
    print(f"  median max_probability inside  : {inside.median():.6f}")
    print(f"  median max_probability outside : {outside.median():.6f}")
    print(f"  percentile of inside-median within ALL outputs: {pct:.1f}%")
    top20 = rollout.nlargest(20, "max_probability")
    print(f"  of the top-20 highest-output sequences, "
          f"{int(in_session[top20.index].sum())}/20 fall inside a documented session")
    median_traj = [float(np.median(probs[:, k])) for k in range(K)]
    d = np.diff(median_traj)
    shape = ("rises" if (d > 0).all() else "falls" if (d < 0).all() else "oscillates")
    print(f"  median trajectory t+1..t+5: "
          + " -> ".join(f"{v:.5f}" for v in median_traj) + f"   ({shape})")
    temporal = {
        "bot_sessions_utc": BOT_SESSIONS_UTC,
        "n_inside": int(in_session.sum()), "n_outside": int((~in_session).sum()),
        "median_max_prob_inside": float(inside.median()),
        "median_max_prob_outside": float(outside.median()),
        "inside_median_percentile_of_all": pct,
        "top20_inside_sessions": int(in_session[top20.index].sum()),
        "median_trajectory": median_traj, "trajectory_shape": shape,
        "caveat": ("Temporal co-occurrence only. No per-window labels exist for "
                   "this mixed-scope substrate; no detection, accuracy or "
                   "early-warning claim is made or computable."),
    }

    # ---------- STEP 7 ----------
    _rule("STEP 7 — comparison to previous real substrates")
    def prior(summary_csv, groups_csv, n_expected=None):
        s = pd.read_csv(summary_csv)
        g = pd.read_csv(groups_csv)
        col = "feature_group" if "feature_group" in g.columns else "group"
        p = s[[f"p_t+{k+1}" for k in range(K)]].to_numpy()
        v4rows = g[g[col].isin(["V4_BASE", "V4_RELATIVE"])]
        pkrows = g[g[col].isin(["PACKET_BASE", "PACKET_RELATIVE"])]
        wv = v4rows["feature_count"] / v4rows["feature_count"].sum()
        wp = pkrows["feature_count"] / pkrows["feature_count"].sum()
        return {
            "sequences": int(len(s)),
            "all82_mean_abs_z": float(
                (g["mean_abs_z"] * (g["feature_count"] / g["feature_count"].sum())).sum()),
            "v4_mean_abs_z": float((v4rows["mean_abs_z"] * wv).sum()),
            "packet_mean_abs_z": float((pkrows["mean_abs_z"] * wp).sum()),
            "p95_probability": float(np.percentile(p, 95)),
            "global_max_probability": float(p.max()),
            "fraction_ge_0.5": float((p >= 0.5).mean()),
            "median_t1": float(np.median(p[:, 0])),
            "median_t5": float(np.median(p[:, 4])),
        }

    comparison = {
        "single_host_.30": prior(
            REPO_ROOT / "results/v5_demo/rollout_summary.csv",
            REPO_ROOT / "results/v5_demo/input_shift_groups.csv"),
        "ten_host_pooled": prior(
            REPO_ROOT / "results/v5_demo_multihost/rollout_summary.csv",
            REPO_ROOT / "results/v5_demo_multihost/input_shift_groups.csv"),
        "mixed_scope_orgwide_v4_plus_singlehost_packet": {
            "sequences": int(X.shape[0]),
            "all82_mean_abs_z": shift["C_all_82"]["mean_abs_z"],
            "v4_mean_abs_z": shift["A_v4_prefix"]["mean_abs_z"],
            "packet_mean_abs_z": shift["B_packet_suffix"]["mean_abs_z"],
            "p95_probability": float(np.percentile(probs, 95)),
            "global_max_probability": float(probs.max()),
            "fraction_ge_0.5": float((probs >= 0.5).mean()),
            "median_t1": float(np.median(probs[:, 0])),
            "median_t5": float(np.median(probs[:, 4])),
        },
        "note": ("Measured differences only. Sequence counts differ because the "
                 "substrates cover different windows; no configuration is "
                 "labelled better or worse."),
        "scope_note": SCOPE_NOTE,
    }
    cols = ["sequences", "all82_mean_abs_z", "v4_mean_abs_z", "packet_mean_abs_z",
            "p95_probability", "global_max_probability", "fraction_ge_0.5",
            "median_t1", "median_t5"]
    print(f"  {'substrate':<34}" + "".join(f"{c[:11]:>13}" for c in cols))
    for key in ("single_host_.30", "ten_host_pooled",
                "mixed_scope_orgwide_v4_plus_singlehost_packet"):
        e = comparison[key]
        print(f"  {key[:33]:<34}" + "".join(
            f"{e[c]:>13.5f}" if isinstance(e[c], float) else f"{e[c]:>13}" for c in cols))
    (OUT_DIR / "comparison_to_previous.json").write_text(
        json.dumps(comparison, indent=2) + "\n")

    # ---------- artifacts ----------
    np.save(OUT_DIR / "X.npy", X)
    metadata.to_csv(OUT_DIR / "metadata.csv", index=False)
    manifest = {
        "experiment": "mixed-scope diagnostic (Task 8)",
        "purpose": ("reproduce the scope pattern inferred from the V5 scaler in "
                    "Task 7 and test the frozen checkpoint against it"),
        "v4_prefix_source": str(STATES_V2.relative_to(REPO_ROOT)),
        "v4_prefix_scope": "organization-wide (whole 172.31.0.0/16 victim network)",
        "packet_suffix_source": str(PACKET_CSV.relative_to(REPO_ROOT)),
        "packet_suffix_scope": "single host 172.31.69.30",
        "rel_change_transform": ("verbatim from src/prepare_multiday_benchmark.py "
                                 "L88-L95, applied in states_v2 file row order"),
        "time_alignment": {
            "join_key": "absolute 10s epoch bin",
            "packet_win0": win0,
            "states_v2_reconstruction": ("AM/PM-stripped Atlantic clock; rows "
                                         ">= discontinuity are AM (local=clock), "
                                         "rows < are PM (local=clock+12h); "
                                         "UTC = local + 4h"),
            "verified_anchors": 3,
            "org_info": org_info, "packet_info": pkt_info, "overlap": overlap,
        },
        "sequences": int(X.shape[0]),
        "scaler_path": str(V5_SCALER.relative_to(REPO_ROOT)),
        "scaler_refitted": False,
        "checkpoint_modified": False,
        "threshold": float(engine.threshold),
        "threshold_tuned": False,
        "labels_used_for_input": False,
        "input_shift": shift,
        "probability_distribution": distribution,
        "sequences_any_horizon_ge": any_ge,
        "latent_stability": {
            "mean_norm_by_horizon": [float(norms[:, k].mean()) for k in range(K)],
            "max_norm": float(norms.max()), "non_finite": nonfinite,
            "exploding": bool(norms.max() > 1e3),
        },
        "temporal_co_occurrence": temporal,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "scope_note": SCOPE_NOTE,
    }
    (OUT_DIR / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  wrote X.npy, metadata.csv, input_shift_report.csv, "
          f"rollout_summary.csv,\n        comparison_to_previous.json, "
          f"diagnostic_manifest.json -> {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"\n{SCOPE_NOTE}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DiagnosticError as exc:
        print(f"\nDIAGNOSTIC STOPPED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
