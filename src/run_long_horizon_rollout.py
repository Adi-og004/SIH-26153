"""
=============================================================
TASK 10 — LONG-HORIZON ROLLOUT + LATENT STABILITY ANALYSIS
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

INFERENCE-ONLY DIAGNOSTIC. Nothing is trained, tuned or modified.

Question: how far can the FROZEN V4 learned transition be rolled forward
before its latent state or its threat predictions stop carrying useful
information?

    z_t = encode(X_t)
    z_{t+k+1} = T(z_{t+k})          for k = 0 .. 9

TRUE AUTOREGRESSIVE ROLLOUT: the only observation ever consumed is the
length-10 input window X_t. No future observation is injected at any step;
each transition consumes the PREDICTED latent from the step before. There
is no teacher forcing anywhere in this file.

HORIZON REGIONS — these are not the same kind of number and are never
mixed in any output:
  K = 1..5   in-distribution rollout depth relative to the V4 training
             curriculum (the checkpoint's curriculum is
             [(1,1),(21,2),(41,3),(61,4),(81,5)], i.e. trained to depth 5)
  K = 6..10  EXTRAPOLATIVE rollout beyond the training curriculum. These
             are recursive applications of a transition that was never
             optimised at this depth. K=10 is NOT a trained 100-second
             forecast.

THRESHOLD POLICY
  K = 1,2,3,5      frozen V4 reference thresholds (0.45/0.54/0.65/0.72),
                   selected on VALIDATION when V4 was frozen. Reproduced,
                   never re-derived.
  K = 4, 6..10     no frozen threshold exists. An "extended-horizon
                   exploratory threshold" is selected on VALIDATION ONLY,
                   with the identical grid search the frozen reference
                   used. Test labels are never consulted for any threshold.

PROVENANCE (Part 23): the frozen V4 checkpoint was trained on the
single-day temporal_benchmark_v2 arrays but the reference evaluation feeds
it multiday_benchmark_K5 arrays scaled by a different StandardScaler. That
mismatch is documented in docs/v4_baseline_manifest.json known_issues. It
is NOT repaired here, and every extended-horizon result inherits it.

Run:
    ./venv/bin/python src/run_long_horizon_rollout.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pcap_ingest_v4 import LatentWorldModelV4, load_v4_model  # noqa: E402

BENCH = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
STATES_DIR = REPO_ROOT / "data" / "processed" / "states_v2"
MANIFEST_PATH = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
CKPT_PATH = REPO_ROOT / "models" / "latent_world_model_v4" / "latent_world_model_v4.pt"

OUT = REPO_ROOT / "results" / "long_horizon_rollout"
FIG = OUT / "figures"
SUMMARY_PATH = REPO_ROOT / "results" / "long_horizon_rollout_summary.json"

MAX_K = 10
TRAINED_K = 5          # the checkpoint's curriculum depth
SEQ_LEN = 10
WINDOW_SECONDS = 10

TRAIN_DAY_FILES = {
    "Wednesday-14-02-2018": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Thursday-15-02-2018": "Thursday-15-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Wednesday-28-02-2018": "Wednesday-28-02-2018_TrafficForML_CICFlowMeter_states_v2.csv",
    "Friday-02-03-2018": "Friday-02-03-2018_TrafficForML_CICFlowMeter_states_v2.csv",
}

# Frozen V4 reference thresholds — READ, never re-derived.
FROZEN_POINT_THRESHOLDS = {1: 0.45, 2: 0.54, 3: 0.65, 5: 0.72}
FROZEN_TRAJ_MAX_THRESHOLDS = {2: 0.51, 3: 0.68, 5: 0.69}

# Frozen V4 reference results the K=1..5 region must reproduce exactly.
FROZEN_POINT_RESULTS = {
    1: dict(f1=0.591967600404995, precision=0.9409871244635193,
            recall=0.43180699162973907, roc_auc=0.8558010432393837,
            fpr=0.0507380073800738, tp=877, fp=55, tn=1029, fn=1154),
    2: dict(f1=0.673871582962492, precision=0.9506726457399103,
            recall=0.5219103889709503, roc_auc=0.8774389036357129,
            fpr=0.0507380073800738, tp=1060, fp=55, tn=1029, fn=971),
    3: dict(f1=0.6951602765556254, precision=0.9609035621198957,
            recall=0.5445593303791236, roc_auc=0.8893420433465782,
            fpr=0.04151291512915129, tp=1106, fp=45, tn=1039, fn=925),
    5: dict(f1=0.7522549609140108, precision=0.9660231660231661,
            recall=0.6159527326440177, roc_auc=0.8876532746125098,
            fpr=0.04059040590405904, tp=1251, fp=44, tn=1040, fn=780),
}
FROZEN_TRAJ_MAX_RESULTS = {
    2: dict(f1=0.7127003699136868, roc_auc=0.8779881696633638),
    3: dict(f1=0.6370516617308325, roc_auc=0.8904168977806681),
    5: dict(f1=0.7979274611398963, roc_auc=0.8928704970360237),
}

V4_FROZEN_HASHES = {
    "models/latent_world_model_v4/latent_world_model_v4.pt":
        "76af8d8a9e02215923b1c0ab6b627bfc0809e856748666a55ee4b16cec5e5898",
    "data/processed/multiday_benchmark_K5/input_scaler.pkl":
        "e0ab9bebd26869950dda51a67bdde3192b0edbd00e20b623115ed2b653f04f19",
    "data/processed/multiday_benchmark_K5/target_scaler.pkl":
        "7155dec48f89d246359741032b002ca0bb44e4739e99daa3e28e77e312c55944",
    "data/processed/multiday_benchmark_K5/input_feature_names.txt":
        "137b58333f1c751522f30478219ff86b8c115aa749defb57ae80ee83559ecbb5",
    "data/processed/multiday_benchmark_K5/state_feature_names.txt":
        "64a8297798779b3616beeda307fef4679e67cfbd33e395a1059fb502a48b5499",
    "docs/v4_feature_schema.json":
        "2b4f820ca1f193e47b3b9c1eae4163cd9b2643886665f5f5d83f876e3ee094c4",
    "docs/v4_baseline_manifest.json":
        "b2765c2489ccee3845e9dc43990b187d894237905087f3969ed2fae82613937e",
}

GENERATED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")

# Validated categorical palette (dataviz reference instance, fixed slot order).
C_REF = "#2a78d6"      # frozen / in-curriculum region
C_EXT = "#eb6834"      # extrapolative region
C_AUX = "#1baf7a"
C_AUX2 = "#eda100"
C_AUX3 = "#e87ba4"
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rule(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _region(k: int) -> str:
    return "reference_in_curriculum" if k <= TRAINED_K else "extrapolative"


# =============================================================
# STEP 1 — VERIFY THE FROZEN V4 (Part 1)
# =============================================================

def step_verify_v4() -> dict:
    _rule("STEP 1 — verify the frozen V4 reference")
    manifest = json.loads(MANIFEST_PATH.read_text())
    report = {"artefacts": [], "checks": [], "status": "PASS"}

    for relative, expected in V4_FROZEN_HASHES.items():
        path = REPO_ROOT / relative
        actual = _sha256(path)
        ok = actual == expected
        report["artefacts"].append(
            {"path": relative, "expected_sha256": expected,
             "actual_sha256": actual, "unchanged": ok}
        )
        if not ok:
            report["status"] = "FAIL"
        print(f"  {'OK  ' if ok else 'DIFF'}  {relative}")

    ck = manifest["checkpoint"]
    model, ckpt = load_v4_model(CKPT_PATH)
    n_params = sum(p.numel() for p in model.parameters())

    checks = [
        ("checkpoint_sha256", _sha256(CKPT_PATH) == ck["sha256"],
         ck["sha256"]),
        ("model_class", ckpt.get("model_class") == ck["model_class"],
         ck["model_class"]),
        ("parameter_count", n_params == ck["parameter_count"],
         f"{n_params} == {ck['parameter_count']}"),
        ("input_dimension_52", ckpt["input_size"] == 52, ckpt["input_size"]),
        ("sequence_length_10", ckpt["sequence_length"] == SEQ_LEN, ckpt["sequence_length"]),
        ("latent_dimension_32", ckpt["latent_size"] == 32, ckpt["latent_size"]),
        ("decoder_output_26", ckpt["output_size"] == 26, ckpt["output_size"]),
        ("checkpoint_loads", True, "loaded with strict=True, eval() mode"),
        ("state_dict_keys", len(ckpt["model_state_dict"]) == ck["state_dict_keys"],
         ck["state_dict_keys"]),
        ("trained_curriculum_depth", ckpt["max_k"] == TRAINED_K,
         f"max_k={ckpt['max_k']} -> K>{TRAINED_K} is extrapolative"),
    ]
    for name, ok, detail in checks:
        report["checks"].append({"check": name, "result": "PASS" if ok else "FAIL",
                                 "detail": str(detail)})
        if not ok:
            report["status"] = "FAIL"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    print(f"\n  FROZEN V4 VERIFICATION: {report['status']}")
    return {"report": report, "model": model, "ckpt": ckpt, "manifest": manifest}


# =============================================================
# STEP 2 — EXTENDED GROUND TRUTH (K=6..10)
# =============================================================

def _day_table(day: str, state_features: list[str], target_scaler) -> dict:
    path = STATES_DIR / TRAIN_DAY_FILES[day]
    frame = pd.read_csv(path, low_memory=False)
    frame["window_start"] = pd.to_datetime(frame["window_start"], errors="coerce")
    frame = frame.dropna(subset=["window_start"]).sort_values("window_start")
    frame = frame.reset_index(drop=True)
    for feature in state_features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame[state_features] = (
        frame[state_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    scaled = np.nan_to_num(
        target_scaler.transform(frame[state_features]),
        nan=0.0, posinf=10.0, neginf=-10.0,
    ).astype(np.float32)
    return {
        "times": frame["window_start"].to_numpy(),
        "attack": frame["attack_present"].to_numpy().astype(np.int64),
        "ratio": frame["attack_ratio"].to_numpy().astype(np.float64),
        "state": scaled,
        "n": len(frame),
    }


def build_extended_truth(split: str, state_features, target_scaler, frozen_A) -> dict:
    """Recover ground-truth labels and scaled state targets for K=1..MAX_K
    by looking each sequence's t+1 timestamp up in the SAME states_v2 tables
    the benchmark was built from.

    This READS the benchmark's source; it writes nothing and modifies no
    benchmark array. Strict 10 s contiguity from t+1 through t+k is required
    for a target to count as available — exactly the rule
    prepare_multiday_benchmark.py applied for K<=5. Where contiguity breaks,
    the target is marked UNAVAILABLE rather than imputed.
    """
    meta = pd.read_csv(BENCH / f"{split}_metadata.csv")
    meta["timestamp"] = pd.to_datetime(meta["timestamp"])
    n = len(meta)

    labels = np.full((n, MAX_K), -1, dtype=np.int64)
    ratios = np.full((n, MAX_K), np.nan)
    states = np.full((n, MAX_K, len(state_features)), np.nan, dtype=np.float64)
    avail = np.zeros((n, MAX_K), dtype=bool)
    current = np.full(n, -1, dtype=np.int64)          # A(t): the last OBSERVED window
    current_state = np.full((n, len(state_features)), np.nan, dtype=np.float64)

    for day, day_rows in meta.groupby("source_day"):
        table = _day_table(day, state_features, target_scaler)
        lookup = {t: i for i, t in enumerate(table["times"])}
        rows = day_rows.index.to_numpy()
        base = np.array([lookup.get(t, -1) for t in day_rows["timestamp"]], dtype=np.int64)
        if (base < 0).any():
            raise RuntimeError(f"{split}/{day}: a t+1 timestamp is absent from states_v2")

        # A(t) is the window BEFORE the first target; it always exists because
        # the sequence needed 10 observed windows ending there.
        current[rows] = table["attack"][base - 1]
        current_state[rows] = table["state"][base - 1]

        for k in range(1, MAX_K + 1):
            target = base + (k - 1)
            inside = target < table["n"]
            safe = np.where(inside, target, 0)
            gap = (table["times"][safe] - table["times"][base])
            contiguous = gap.astype("timedelta64[s]").astype(np.int64) == WINDOW_SECONDS * (k - 1)
            ok = inside & contiguous
            sel = rows[ok]
            labels[sel, k - 1] = table["attack"][target[ok]]
            ratios[sel, k - 1] = table["ratio"][target[ok]]
            states[sel, k - 1] = table["state"][target[ok]]
            avail[sel, k - 1] = True

    # The K<=5 block MUST reproduce the frozen benchmark arrays exactly,
    # otherwise the reconstruction is wrong and nothing downstream is valid.
    if not np.array_equal(labels[:, :TRAINED_K], frozen_A.astype(np.int64)):
        raise RuntimeError(
            f"{split}: reconstructed K=1..5 labels do not match the frozen "
            f"A_binary_multistep_{split}.npy — reconstruction is unsound"
        )

    print(f"  {split:<5} n={n}  K=1..5 labels reproduce the frozen arrays exactly")
    print(f"        available targets per K: "
          f"{ {k: int(avail[:, k - 1].sum()) for k in range(1, MAX_K + 1)} }")
    return {
        "labels": labels, "ratios": ratios, "states": states, "avail": avail,
        "current": current, "current_state": current_state, "meta": meta,
    }


# =============================================================
# STEP 3 — AUTOREGRESSIVE ROLLOUT
# =============================================================

def rollout(model: LatentWorldModelV4, X: np.ndarray, k_max: int = MAX_K) -> dict:
    """encode once, then apply the transition k_max times, feeding each
    PREDICTED latent back in. Index 0 is z_t (no transition); index k is
    after exactly k transitions.

    The only observation consumed is X. Nothing else enters the loop — this
    is what makes it a genuine extended rollout rather than teacher forcing.
    """
    model.eval()
    with torch.no_grad():
        traj_z, traj_p, traj_s = model.k_step_rollout(torch.tensor(X), k_max)
    Z = np.stack([z.numpy() for z in traj_z], axis=1).astype(np.float64)
    P = np.stack([p.numpy() for p in traj_p], axis=1).astype(np.float64)
    S = np.stack([s.numpy() for s in traj_s], axis=1).astype(np.float64)
    assert Z.shape[1] == P.shape[1] == S.shape[1] == k_max + 1
    return {"Z": Z, "P": P, "S": S}


# =============================================================
# THRESHOLDS — VALIDATION ONLY, EVER
# =============================================================

def find_best_threshold(probs, targets) -> float:
    """Verbatim reproduction of the frozen reference's selection rule
    (src/k_step_rollout.py @ 2be3073): grid np.arange(0.01, 0.99, 0.01),
    strict > comparison, so the FIRST maximising threshold wins."""
    best_f1, best_thr = -1.0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        f = f1_score(targets, (probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    return float(best_thr)


def eval_metrics(targets, probs, thr) -> dict:
    preds = (probs >= thr).astype(int)
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    tp = int(((preds == 1) & (targets == 1)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    n_pos, n_neg = int(targets.sum()), int((targets == 0).sum())
    return {
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probs)) if n_pos and n_neg else np.nan,
        "pr_auc": float(average_precision_score(targets, probs)) if n_pos and n_neg else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "pos_samples": n_pos, "n": int(len(targets)),
    }


# =============================================================
# STEP 4 — HORIZON PERFORMANCE (Parts 5, 6)
# =============================================================

def step_horizon_performance(test, val, roll_test, roll_val) -> pd.DataFrame:
    _rule("STEP 4 — horizon performance, K=1..10")
    rows = []
    for k in range(1, MAX_K + 1):
        mask = test["avail"][:, k - 1]
        y = test["labels"][mask, k - 1]
        p = roll_test["P"][mask, k]

        vmask = val["avail"][:, k - 1]
        vy = val["labels"][vmask, k - 1]
        vp = roll_val["P"][vmask, k]

        if k in FROZEN_POINT_THRESHOLDS:
            thr = FROZEN_POINT_THRESHOLDS[k]
            thr_kind = "frozen_v4_reference"
            val_f1 = float(f1_score(vy, (vp >= thr).astype(int), zero_division=0))
        else:
            thr = find_best_threshold(vp, vy)
            thr_kind = "extended_horizon_exploratory"
            val_f1 = float(f1_score(vy, (vp >= thr).astype(int), zero_division=0))

        m = eval_metrics(y, p, thr) if y.sum() and (y == 0).sum() else None
        if m is None:
            rows.append({"horizon": k, "region": _region(k), "status": "NA_single_class"})
            continue

        row = {
            "horizon": k,
            "region": _region(k),
            "threshold": thr,
            "threshold_kind": thr_kind,
            "threshold_source": "validation only — test labels never used",
            "validation_f1_at_threshold": val_f1,
            "n_evaluated": m["n"],
            "n_unavailable": int((~mask).sum()),
            "positive_prevalence": float(y.mean()),
            "mean_probability": float(p.mean()),
            **{key: m[key] for key in ("f1", "precision", "recall", "roc_auc",
                                       "pr_auc", "fpr", "tp", "fp", "tn", "fn",
                                       "pos_samples")},
            "brier": float(((p - y) ** 2).mean()),
        }
        # Frozen-reference reproduction gate.
        if k in FROZEN_POINT_RESULTS:
            ref = FROZEN_POINT_RESULTS[k]
            diffs = {key: abs(row[key] - ref[key]) for key in
                     ("f1", "precision", "recall", "roc_auc", "fpr")}
            counts_ok = all(row[key] == ref[key] for key in ("tp", "fp", "tn", "fn"))
            row["reproduces_frozen_reference"] = bool(max(diffs.values()) < 1e-12 and counts_ok)
            row["max_abs_diff_vs_frozen"] = float(max(diffs.values()))
            if not row["reproduces_frozen_reference"]:
                raise RuntimeError(
                    f"K={k} does not reproduce the frozen V4 reference "
                    f"(max diff {max(diffs.values()):.3e}, counts_ok={counts_ok})"
                )
        else:
            row["reproduces_frozen_reference"] = None
            row["max_abs_diff_vs_frozen"] = None
        rows.append(row)

        tag = "FROZEN REF" if k in FROZEN_POINT_RESULTS else "exploratory"
        print(f"  K={k:<2} [{_region(k):<23}] thr={thr:.2f} ({tag:<11}) "
              f"F1={row['f1']:.4f}  ROC-AUC={row['roc_auc']:.4f}  "
              f"PR-AUC={row['pr_auc']:.4f}  n={m['n']}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "horizon_performance.csv", index=False)
    reproduced = frame["reproduces_frozen_reference"].fillna(False).sum()
    print(f"\n  K=1,2,3,5 reproduce the frozen reference exactly: {int(reproduced)}/4")
    print(f"  wrote {(OUT / 'horizon_performance.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 5 — TRAJECTORY EVENT PERFORMANCE (Part 7)
# =============================================================

def step_trajectory(test, val, roll_test, roll_val) -> pd.DataFrame:
    _rule("STEP 5 — trajectory-event performance")
    rows = []
    for k in [2, 3, 5, 6, 7, 8, 9, 10]:
        tmask = test["avail"][:, :k].all(axis=1)
        y = test["labels"][tmask, :k].max(axis=1)
        p = roll_test["P"][tmask, 1:k + 1].max(axis=1)

        vmask = val["avail"][:, :k].all(axis=1)
        vy = val["labels"][vmask, :k].max(axis=1)
        vp = roll_val["P"][vmask, 1:k + 1].max(axis=1)

        if k in FROZEN_TRAJ_MAX_THRESHOLDS:
            thr, thr_kind = FROZEN_TRAJ_MAX_THRESHOLDS[k], "frozen_v4_reference"
        else:
            thr, thr_kind = find_best_threshold(vp, vy), "extended_horizon_exploratory"

        m = eval_metrics(y, p, thr)
        row = {
            "horizon": k, "region": _region(k), "aggregation": "max",
            "target": "max(A(t+1)..A(t+k))", "score": "max(P(t+1)..P(t+k))",
            "threshold": thr, "threshold_kind": thr_kind,
            "validation_f1_at_threshold": float(
                f1_score(vy, (vp >= thr).astype(int), zero_division=0)),
            "n_evaluated": m["n"], **{key: m[key] for key in
                                      ("f1", "precision", "recall", "roc_auc", "pr_auc",
                                       "fpr", "tp", "fp", "tn", "fn", "pos_samples")},
        }
        if k in FROZEN_TRAJ_MAX_RESULTS:
            ref = FROZEN_TRAJ_MAX_RESULTS[k]
            diff = max(abs(row["f1"] - ref["f1"]), abs(row["roc_auc"] - ref["roc_auc"]))
            row["reproduces_frozen_reference"] = bool(diff < 1e-12)
            row["max_abs_diff_vs_frozen"] = float(diff)
            if not row["reproduces_frozen_reference"]:
                raise RuntimeError(f"trajectory K={k} does not reproduce the frozen reference "
                                   f"(diff {diff:.3e})")
        else:
            row["reproduces_frozen_reference"] = None
            row["max_abs_diff_vs_frozen"] = None
        rows.append(row)
        tag = "FROZEN REF" if k in FROZEN_TRAJ_MAX_RESULTS else "exploratory"
        print(f"  K={k:<2} [{_region(k):<23}] thr={thr:.2f} ({tag:<11}) "
              f"F1={row['f1']:.4f}  ROC-AUC={row['roc_auc']:.4f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "trajectory_performance.csv", index=False)
    print(f"  wrote {(OUT / 'trajectory_performance.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 6 — LATENT STABILITY (Parts 8, 9)
# =============================================================

def step_latent_stability(test, roll_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    _rule("STEP 6 — latent-state stability")
    Z, S = roll_test["Z"], roll_test["S"]
    n, steps, latent_dim = Z.shape

    norm = np.linalg.norm(Z, axis=2)
    rms = np.sqrt((Z ** 2).mean(axis=2))
    mean = Z.mean(axis=2)
    var = Z.var(axis=2)
    delta = np.zeros_like(norm)
    delta[:, 1:] = np.linalg.norm(np.diff(Z, axis=1), axis=2)
    displacement = np.linalg.norm(Z - Z[:, :1, :], axis=2)

    def _cos(a, b):
        num = (a * b).sum(axis=-1)
        den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    cos_prev = np.full_like(norm, np.nan)
    cos_prev[:, 1:] = _cos(Z[:, 1:, :], Z[:, :-1, :])
    cos_init = _cos(Z, Z[:, :1, :])

    # decoder error against the true future state, where one exists
    dec_err = np.full((n, steps), np.nan)
    for k in range(1, MAX_K + 1):
        mask = test["avail"][:, k - 1]
        err = S[mask, k, :] - test["states"][mask, k - 1, :]
        dec_err[mask, k] = np.sqrt((err ** 2).mean(axis=1))

    sample_ids = np.repeat(np.arange(n), steps)
    step_ids = np.tile(np.arange(steps), n)
    detail = pd.DataFrame({
        "sample_id": sample_ids,
        "step": step_ids,
        "region": np.where(step_ids <= TRAINED_K, "reference_in_curriculum", "extrapolative"),
        "latent_norm": norm.ravel(),
        "latent_rms": rms.ravel(),
        "latent_mean": mean.ravel(),
        "latent_variance": var.ravel(),
        "delta_norm": delta.ravel(),
        "cumulative_displacement": displacement.ravel(),
        "cosine_prev": cos_prev.ravel(),
        "cosine_initial": cos_init.ravel(),
        "decoded_state_error": dec_err.ravel(),
    })
    detail.loc[detail["step"] == 0, ["delta_norm"]] = np.nan
    detail.to_csv(OUT / "latent_stability.csv", index=False)

    rows = []
    for k in range(steps):
        row = {"horizon": k,
               "region": "encoder_output" if k == 0 else _region(k),
               "n_samples": n,
               "max_abs_latent_component": float(np.abs(Z[:, k, :]).max()),
               "across_sample_latent_std": float(Z[:, k, :].std(axis=0).mean())}
        for label, values in (("latent_norm", norm[:, k]),
                              ("delta_norm", delta[:, k] if k > 0 else np.full(n, np.nan)),
                              ("cumulative_displacement", displacement[:, k]),
                              ("latent_rms", rms[:, k]),
                              ("latent_variance", var[:, k]),
                              ("cosine_prev", cos_prev[:, k]),
                              ("cosine_initial", cos_init[:, k])):
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                for stat in ("mean", "median", "p90", "p95", "p99", "max", "min"):
                    row[f"{label}_{stat}"] = np.nan
                continue
            row[f"{label}_mean"] = float(valid.mean())
            row[f"{label}_median"] = float(np.median(valid))
            row[f"{label}_p90"] = float(np.percentile(valid, 90))
            row[f"{label}_p95"] = float(np.percentile(valid, 95))
            row[f"{label}_p99"] = float(np.percentile(valid, 99))
            row[f"{label}_max"] = float(valid.max())
            row[f"{label}_min"] = float(valid.min())
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "latent_stability_summary.csv", index=False)

    print("   K  norm_mean  norm_p99  delta_mean  displacement_mean  cos_prev  across_std")
    for _, r in summary.iterrows():
        d = "        -" if not np.isfinite(r["delta_norm_mean"]) else f"{r['delta_norm_mean']:9.4f}"
        c = "       -" if not np.isfinite(r["cosine_prev_mean"]) else f"{r['cosine_prev_mean']:8.4f}"
        print(f"  {int(r['horizon']):2d}  {r['latent_norm_mean']:9.4f} {r['latent_norm_p99']:9.4f}"
              f" {d}  {r['cumulative_displacement_mean']:17.4f} {c}  {r['across_sample_latent_std']:10.4f}")
    print(f"\n  wrote {(OUT / 'latent_stability.csv').relative_to(REPO_ROOT)} ({len(detail)} rows)")
    print(f"  wrote {(OUT / 'latent_stability_summary.csv').relative_to(REPO_ROOT)}")
    return detail, summary


# =============================================================
# STEP 7 — LATENT CONTRACTION / FIXED-POINT CHECK (Part 10)
# =============================================================

def step_contraction(roll_test) -> pd.DataFrame:
    _rule("STEP 7 — latent contraction / fixed-point check")
    Z = roll_test["Z"]
    delta = np.linalg.norm(np.diff(Z, axis=1), axis=2)       # (n, MAX_K)
    ratio = delta[:, 1:] / np.maximum(delta[:, :-1], 1e-12)  # (n, MAX_K-1)

    rows = []
    for k in range(2, MAX_K + 1):
        r = ratio[:, k - 2]
        rows.append({
            "horizon": k,
            "region": _region(k),
            "definition": "contraction_ratio = ||z_{t+k} - z_{t+k-1}|| / ||z_{t+k-1} - z_{t+k-2}||",
            "contraction_ratio_mean": float(r.mean()),
            "contraction_ratio_median": float(np.median(r)),
            "contraction_ratio_p90": float(np.percentile(r, 90)),
            "contraction_ratio_max": float(r.max()),
            "frac_samples_contracting": float((r < 1.0).mean()),
            "delta_norm_mean": float(delta[:, k - 1].mean()),
            "delta_norm_vs_step1": float(delta[:, k - 1].mean() / delta[:, 0].mean()),
        })
    frame = pd.DataFrame(rows)

    # Per-sample verdict, using pre-declared numeric rules.
    #   observed_contraction : delta shrinks by >= 50% from step 1 to step 10
    #                          AND >= 80% of consecutive ratios are < 1
    #   near_fixed_point     : final delta < 5% of the step-1 delta
    #   non_contracting      : otherwise
    shrink = delta[:, -1] / np.maximum(delta[:, 0], 1e-12)
    frac_lt1 = (ratio < 1.0).mean(axis=1)
    verdict = np.where(
        shrink < 0.05, "near_fixed_point",
        np.where((shrink <= 0.5) & (frac_lt1 >= 0.8), "observed_contraction", "non_contracting"),
    )
    counts = pd.Series(verdict).value_counts().to_dict()
    frame["per_sample_verdict_counts"] = json.dumps(counts)
    frame["note"] = (
        "OBSERVED contraction, measured empirically. This is NOT a proof of mathematical "
        "convergence: no Lipschitz constant is established and no limit is computed. The "
        "transition ends in LayerNorm, which bounds the latent RMS by construction, so a "
        "bounded, shrinking step sequence is the expected architectural behaviour rather "
        "than evidence of a learned attractor."
    )
    frame.to_csv(OUT / "latent_contraction.csv", index=False)

    print(f"  step-1 mean delta : {delta[:, 0].mean():.4f}")
    print(f"  step-{MAX_K} mean delta: {delta[:, -1].mean():.4f} "
          f"({100 * delta[:, -1].mean() / delta[:, 0].mean():.1f}% of step 1)")
    print(f"  mean contraction ratio across steps: {ratio.mean():.4f}")
    print(f"  samples contracting at every step  : {(ratio < 1.0).all(axis=1).mean() * 100:.1f}%")
    print(f"  per-sample verdicts: {counts}")
    print(f"  wrote {(OUT / 'latent_contraction.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 8 — PROBABILITY STABILITY (Part 11)
# =============================================================

def step_probability_stability(test, roll_test, horizon_frame) -> pd.DataFrame:
    _rule("STEP 8 — probability stability")
    P = roll_test["P"]
    traj = P[:, 1:MAX_K + 1]                    # P(t+1) .. P(t+10)
    n = len(P)
    diffs = np.diff(traj, axis=1)

    thresholds = {int(r["horizon"]): float(r["threshold"])
                  for _, r in horizon_frame.iterrows() if np.isfinite(r.get("threshold", np.nan))}

    def _first_cross(row, limits):
        for j in range(MAX_K):
            if row[j] >= limits[j]:
                return j + 1
        return -1

    limit_vector = np.array([thresholds.get(k, np.nan) for k in range(1, MAX_K + 1)])
    half = np.full(MAX_K, 0.5)

    monotone_up = (diffs >= -1e-12).all(axis=1)
    monotone_down = (diffs <= 1e-12).all(axis=1)
    sign = np.sign(np.where(np.abs(diffs) < 1e-9, 0.0, diffs))
    oscillation = np.zeros(n, dtype=int)
    for i in range(n):
        s = sign[i][sign[i] != 0]
        oscillation[i] = int((np.diff(s) != 0).sum()) if s.size > 1 else 0

    frame = pd.DataFrame({
        "sample_id": np.arange(n),
        "mean_probability": traj.mean(axis=1),
        "min_probability": traj.min(axis=1),
        "max_probability": traj.max(axis=1),
        "probability_range": traj.max(axis=1) - traj.min(axis=1),
        "total_probability_delta": traj[:, -1] - traj[:, 0],
        "mean_abs_step_delta": np.abs(diffs).mean(axis=1),
        "max_abs_step_delta": np.abs(diffs).max(axis=1),
        "monotonic_increasing": monotone_up,
        "monotonic_decreasing": monotone_down,
        "monotonic": monotone_up | monotone_down,
        "oscillation_count": oscillation,
        "saturated_high": (traj >= 0.98).any(axis=1),
        "saturated_low": (traj <= 0.02).any(axis=1),
        "frac_steps_saturated": ((traj >= 0.98) | (traj <= 0.02)).mean(axis=1),
        "first_step_above_0.5": [_first_cross(traj[i], half) for i in range(n)],
        "first_step_above_threshold": [_first_cross(traj[i], limit_vector) for i in range(n)],
        "threshold_note": "K=1,2,3,5 frozen V4 thresholds; K=4,6..10 "
                          "extended-horizon exploratory (validation-selected)",
        "A_current": test["current"],
    })
    for k in range(1, MAX_K + 1):
        frame[f"P_t{k}"] = traj[:, k - 1]
    frame.to_csv(OUT / "probability_stability.csv", index=False)

    print(f"  monotonic increasing over K=1..10 : {monotone_up.mean() * 100:.1f}% of samples")
    print(f"  monotonic decreasing              : {monotone_down.mean() * 100:.1f}%")
    print(f"  mean oscillation count            : {oscillation.mean():.2f} sign changes")
    print(f"  any step saturated (<=0.02 or >=0.98): "
          f"{frame['frac_steps_saturated'].gt(0).mean() * 100:.1f}%")
    print(f"  mean probability drift P(t+10)-P(t+1): {frame['total_probability_delta'].mean():+.4f}")
    print(f"  wrote {(OUT / 'probability_stability.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 9 — FORECAST ERROR (Parts 12, 13)
# =============================================================

def step_forecast_error(test, roll_test) -> pd.DataFrame:
    _rule("STEP 9 — forecast error vs horizon")
    S, P = roll_test["S"], roll_test["P"]
    rows = []
    for k in range(1, MAX_K + 1):
        mask = test["avail"][:, k - 1]
        target = test["states"][mask, k - 1, :]
        pred = S[mask, k, :]
        y = test["labels"][mask, k - 1]
        p = P[mask, k]

        # Two reference points, so "error does not grow" can be read correctly.
        persistence = test["current_state"][mask]            # copy the last OBSERVED state
        train_mean = np.zeros_like(target)                   # target-scaler mean is 0 by construction

        def _err(a):
            d = a - target
            return float(np.abs(d).mean()), float(np.sqrt((d ** 2).mean()))

        mae, rmse = _err(pred)
        p_mae, p_rmse = _err(persistence)
        m_mae, m_rmse = _err(train_mean)

        rows.append({
            "horizon": k, "region": _region(k), "n_evaluated": int(mask.sum()),
            "state_mae": mae, "state_rmse": rmse,
            "persistence_state_mae": p_mae, "persistence_state_rmse": p_rmse,
            "train_mean_state_mae": m_mae, "train_mean_state_rmse": m_rmse,
            "beats_persistence_mae": bool(mae < p_mae),
            "beats_train_mean_mae": bool(mae < m_mae),
            "skill_vs_train_mean_rmse": float(1.0 - rmse / m_rmse),
            "decoded_across_sample_std": float(pred.std(axis=0).mean()),
            "target_across_sample_std": float(target.std(axis=0).mean()),
            "brier": float(((p - y) ** 2).mean()),
            "brier_base_rate": float(((y.mean() - y) ** 2).mean()),
            "skill_vs_base_rate_brier": float(
                1.0 - ((p - y) ** 2).mean() / ((y.mean() - y) ** 2).mean()),
            "mean_probability": float(p.mean()),
            "state_target_note": "26-dim state in target_scaler space "
                                 "(data/processed/multiday_benchmark_K5/target_scaler.pkl); "
                                 "the transition is a DETERMINISTIC point estimate in latent "
                                 "space, not a distribution over states",
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "forecast_error.csv", index=False)

    print("   K   state_RMSE  persist_RMSE  mean_RMSE  skill_vs_mean   Brier  skill_vs_base")
    for _, r in frame.iterrows():
        print(f"  {int(r['horizon']):2d}   {r['state_rmse']:9.4f}  {r['persistence_state_rmse']:11.4f}"
              f"  {r['train_mean_state_rmse']:9.4f}  {r['skill_vs_train_mean_rmse']:+13.4f}"
              f"  {r['brier']:.4f}  {r['skill_vs_base_rate_brier']:+.4f}")

    first, last = frame.iloc[0], frame.iloc[-1]
    growth = last["state_rmse"] / first["state_rmse"]
    print(f"\n  state RMSE K=1 -> K=10: {first['state_rmse']:.4f} -> {last['state_rmse']:.4f} "
          f"(x{growth:.3f})")
    print(f"  wrote {(OUT / 'forecast_error.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 10 — PERSISTENCE CONTEXT (Part 14)
# =============================================================

def step_persistence(test) -> pd.DataFrame:
    _rule("STEP 10 — persistence context")
    rows = []
    for k in range(1, MAX_K + 1):
        mask = test["avail"][:, k - 1]
        y = test["labels"][mask, k - 1]
        a0 = test["current"][mask]
        onset = int(((a0 == 0) & (y == 1)).sum())
        offset = int(((a0 == 1) & (y == 0)).sum())
        rows.append({
            "horizon": k, "region": _region(k), "n_evaluated": int(mask.sum()),
            "persistence_accuracy": float((y == a0).mean()),
            "transition_fraction": float((y != a0).mean()),
            "onset_count": onset, "offset_count": offset,
            "transition_count": onset + offset,
            "positive_prevalence": float(y.mean()),
            "current_positive_rate": float(a0.mean()),
            "definition": "persistence = predict A(t+k) = A(t), where A(t) is the label of the "
                          "LAST OBSERVED window; transition = A(t+k) != A(t)",
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "persistence_context.csv", index=False)
    print("   K   persistence_acc  transition_frac  onsets  offsets  prevalence")
    for _, r in frame.iterrows():
        print(f"  {int(r['horizon']):2d}   {r['persistence_accuracy']:15.4f}  "
              f"{r['transition_fraction']:15.4f}  {int(r['onset_count']):6d}  "
              f"{int(r['offset_count']):7d}  {r['positive_prevalence']:10.4f}")
    print(f"  wrote {(OUT / 'persistence_context.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 11 — TRANSITION-ONLY ANALYSIS (Part 15)
# =============================================================

MIN_TRANSITION_N = 20   # pre-declared: below this a transition statistic is LOW POWER


def step_transitions(test, roll_test, horizon_frame) -> pd.DataFrame:
    _rule("STEP 11 — transition-conditioned behaviour")
    P = roll_test["P"]
    thresholds = {int(r["horizon"]): float(r["threshold"])
                  for _, r in horizon_frame.iterrows()}
    rows = []
    for k in range(1, MAX_K + 1):
        mask = test["avail"][:, k - 1]
        y = test["labels"][mask, k - 1]
        a0 = test["current"][mask]
        p = P[mask, k]
        onset = (a0 == 0) & (y == 1)
        offset = (a0 == 1) & (y == 0)
        persist_pos = (a0 == 1) & (y == 1)
        persist_neg = (a0 == 0) & (y == 0)
        thr = thresholds.get(k, np.nan)

        def _stat(m, fn):
            return float(fn(p[m])) if m.sum() else np.nan

        rows.append({
            "horizon": k, "region": _region(k),
            "n_evaluated": int(mask.sum()),
            "transition_count": int(onset.sum() + offset.sum()),
            "onset_count": int(onset.sum()),
            "offset_count": int(offset.sum()),
            "mean_onset_probability": _stat(onset, np.mean),
            "median_onset_probability": _stat(onset, np.median),
            "mean_offset_probability": _stat(offset, np.mean),
            "median_offset_probability": _stat(offset, np.median),
            "mean_persistent_positive_probability": _stat(persist_pos, np.mean),
            "mean_persistent_negative_probability": _stat(persist_neg, np.mean),
            "onset_minus_persistent_negative": (
                _stat(onset, np.mean) - _stat(persist_neg, np.mean)),
            "threshold": thr,
            "threshold_kind": ("frozen_v4_reference" if k in FROZEN_POINT_THRESHOLDS
                               else "extended_horizon_exploratory"),
            "onset_threshold_crossing_rate": (
                float((p[onset] >= thr).mean()) if onset.sum() and np.isfinite(thr) else np.nan),
            "offset_threshold_crossing_rate": (
                float((p[offset] >= thr).mean()) if offset.sum() and np.isfinite(thr) else np.nan),
            # Power keys on the SMALLER class, not the combined count: onset and
            # offset statistics are reported separately, so a combined n of 24
            # built from 12 onsets says nothing about onset reliability.
            "statistical_power": ("LOW — the smaller transition class has n < "
                                  f"{MIN_TRANSITION_N}, treat as anecdotal"
                                  if min(onset.sum(), offset.sum()) < MIN_TRANSITION_N
                                  else "adequate"),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "transition_metrics.csv", index=False)
    print("   K  onsets  offsets   meanP|onset  meanP|persist-neg  meanP|offset  power")
    for _, r in frame.iterrows():
        print(f"  {int(r['horizon']):2d}  {int(r['onset_count']):6d}  {int(r['offset_count']):7d}"
              f"   {r['mean_onset_probability']:11.4f}  "
              f"{r['mean_persistent_negative_probability']:16.4f}  "
              f"{r['mean_offset_probability']:12.4f}  {r['statistical_power'].split(' ')[0]}")
    print(f"  wrote {(OUT / 'transition_metrics.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 11b — IS IT A FORECAST, OR A NOWCAST? (well-powered)
# =============================================================
# The transition-conditioned statistics in step 11 rest on 3-12 onsets and
# cannot carry a verdict on their own. This step asks the same question with
# the FULL test set behind it:
#
#   does P(t+k) predict the FUTURE label A(t+k) better than it predicts the
#   CURRENT label A(t)?
#
# If it does not, the "k-step forecast" is really a description of the
# present that scores well on the future only because the label persists.
# Both AUCs use all ~3,105 samples, so this comparison is fully powered —
# unlike anything conditioned on transitions.

INFORMATIVENESS_RULES = [
    {"indicator": "beats_nowcast",
     "rule": "roc_auc(P_k vs A(t+k)) >= roc_auc(P_k vs A(t))",
     "power": "FULL — both AUCs use every evaluated sample (n ~ 3,105)"},
    {"indicator": "incremental_not_worsening",
     "rule": "incremental_auc(K) >= incremental_auc(K=1) - 0.005",
     "power": "FULL"},
    {"indicator": "onset_ranking_above_chance",
     "rule": "roc_auc(P_k vs A(t+k)) restricted to the A(t)=0 stratum >= 0.5",
     "power": "LOW — 3 to 12 positives; advisory only"},
    {"indicator": "onset_separation_retained",
     "rule": "(meanP|onset - meanP|persistent-negative)(K) >= 0.5 * the same at K=1",
     "power": "LOW — advisory only"},
]
INFORMATIVENESS_VERDICT_RULE = (
    "FORECAST_INFORMATIVE = beats_nowcast passes; "
    "NOWCAST_EQUIVALENT = beats_nowcast fails but |incremental_auc| <= 0.01; "
    "ANTI_INFORMATIVE = incremental_auc < -0.01 (the k-step probability describes the "
    "present measurably better than the future it is supposed to predict)"
)


def step_nowcast_vs_forecast(test, roll_test, transitions) -> pd.DataFrame:
    _rule("STEP 11b — forecast or nowcast? (full statistical power)")
    print("  Rules (declared before evaluation):")
    for rule in INFORMATIVENESS_RULES:
        print(f"    [{rule['power'].split(' ')[0]:<5}] {rule['indicator']}: {rule['rule']}")
    print(f"  Verdict: {INFORMATIVENESS_VERDICT_RULE}")
    print()

    P = roll_test["P"]
    rows = []
    for k in range(1, MAX_K + 1):
        mask = test["avail"][:, k - 1]
        y = test["labels"][mask, k - 1]
        a0 = test["current"][mask]
        p = P[mask, k]

        auc_future = float(roc_auc_score(y, p))
        auc_current = float(roc_auc_score(a0, p))
        incremental = auc_future - auc_current

        stratum = a0 == 0
        ys = y[stratum]
        auc_onset = (float(roc_auc_score(ys, p[stratum]))
                     if ys.sum() and (ys == 0).sum() else np.nan)

        tr = transitions[transitions["horizon"] == k].iloc[0]
        separation = float(tr["onset_minus_persistent_negative"])

        rows.append({
            "horizon": k, "region": _region(k), "n_evaluated": int(mask.sum()),
            "roc_auc_vs_future_label": auc_future,
            "roc_auc_vs_current_label": auc_current,
            "incremental_auc": incremental,
            "onset_stratum_roc_auc": auc_onset,
            "onset_stratum_n_positive": int(ys.sum()),
            "onset_separation": separation,
            "statistical_power": "FULL for the AUC comparison; LOW for the onset stratum",
        })

    frame = pd.DataFrame(rows)
    base_incremental = float(frame["incremental_auc"].iloc[0])
    base_separation = float(frame["onset_separation"].iloc[0])

    verdicts, checks = [], []
    for _, r in frame.iterrows():
        beats = r["roc_auc_vs_future_label"] >= r["roc_auc_vs_current_label"]
        not_worse = r["incremental_auc"] >= base_incremental - 0.005
        above_chance = bool(r["onset_stratum_roc_auc"] >= 0.5) if np.isfinite(
            r["onset_stratum_roc_auc"]) else False
        retained = r["onset_separation"] >= 0.5 * base_separation
        if beats:
            verdict = "FORECAST_INFORMATIVE"
        elif r["incremental_auc"] < -0.01:
            verdict = "ANTI_INFORMATIVE"
        else:
            verdict = "NOWCAST_EQUIVALENT"
        verdicts.append(verdict)
        checks.append((beats, not_worse, above_chance, retained))

    frame["pass_beats_nowcast"] = [c[0] for c in checks]
    frame["pass_incremental_not_worsening"] = [c[1] for c in checks]
    frame["advisory_onset_ranking_above_chance"] = [c[2] for c in checks]
    frame["advisory_onset_separation_retained"] = [c[3] for c in checks]
    frame["informativeness_verdict"] = verdicts
    frame["verdict_rule"] = INFORMATIVENESS_VERDICT_RULE
    frame.to_csv(OUT / "nowcast_vs_forecast.csv", index=False)

    print("   K   AUC vs A(t+k)  AUC vs A(t)  incremental  onset-stratum AUC (n+)  verdict")
    for _, r in frame.iterrows():
        print(f"  {int(r['horizon']):2d}   {r['roc_auc_vs_future_label']:12.4f} "
              f"{r['roc_auc_vs_current_label']:12.4f} {r['incremental_auc']:+12.4f}  "
              f"{r['onset_stratum_roc_auc']:18.4f} ({int(r['onset_stratum_n_positive']):2d})  "
              f"{r['informativeness_verdict']}")
    print()
    print("  READ THIS ALONGSIDE horizon_status.csv: that file answers 'is the rollout")
    print("  numerically stable', this one answers 'does the rollout carry forecasting")
    print("  information'. They are different questions with different answers.")
    print(f"  wrote {(OUT / 'nowcast_vs_forecast.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 12 — EXTENDED EARLY-WARNING ANALYSIS (Part 16)
# =============================================================

def step_warning(test, val, roll_test, roll_val) -> pd.DataFrame:
    _rule("STEP 12 — extended early-warning analysis")
    # Episodes are defined exactly as the frozen reference defines them:
    # on the K=1 label sequence over consecutive test sequence indices.
    target_seq = test["labels"][:, 0]
    onsets, active_at_start = [], False
    for i in range(len(target_seq)):
        if target_seq[i] == 1:
            if i == 0:
                active_at_start = True
            elif target_seq[i - 1] == 0:
                onsets.append(i)
    print(f"  observable onsets in the test split: {len(onsets)} "
          f"(attack active at start: {active_at_start})")

    rows = []
    for k in [1, 3, 5, 6, 7, 8, 9, 10]:
        vmask = val["avail"][:, :k].all(axis=1)
        vy = val["labels"][vmask, :k].max(axis=1)
        vp = roll_val["P"][vmask, 1:k + 1].max(axis=1)
        thr = find_best_threshold(vp, vy)
        kind = ("frozen_v4_reference_early_warning" if k == TRAINED_K
                else "extended_horizon_exploratory")

        score = roll_test["P"][:, 1:k + 1].max(axis=1)
        alert = (score >= thr).astype(int)
        tmask = test["avail"][:, :k].all(axis=1)
        event = np.zeros(len(target_seq), dtype=int)
        event[tmask] = test["labels"][tmask, :k].max(axis=1)

        for onset in onsets:
            lead, warned, prob_at = 0, False, np.nan
            for t in range(max(0, onset - k), onset):
                if alert[t] == 1:
                    lead = (onset - t) * WINDOW_SECONDS
                    prob_at = float(score[t])
                    warned = True
                    break
            rows.append({
                "horizon": k, "region": _region(k), "episode_onset_index": int(onset),
                "onset_timestamp": str(test["meta"]["timestamp"].iloc[onset]),
                "warned": bool(warned), "lead_time_seconds": int(lead),
                "score_at_first_warning": prob_at,
                "threshold": thr, "threshold_kind": kind,
                "threshold_source": "validation only",
            })

        false_alerts = int(((alert == 1) & (event == 0)).sum())
        hours = len(target_seq) * WINDOW_SECONDS / 3600.0
        warned_n = sum(1 for r in rows if r["horizon"] == k and r["warned"])
        leads = [r["lead_time_seconds"] for r in rows
                 if r["horizon"] == k and r["lead_time_seconds"] > 0]
        rows.append({
            "horizon": k, "region": _region(k), "episode_onset_index": -1,
            "onset_timestamp": "AGGREGATE",
            "warned": None,
            "lead_time_seconds": None, "score_at_first_warning": None,
            "threshold": thr, "threshold_kind": kind,
            "threshold_source": "validation only",
            "n_onsets": len(onsets), "n_warned": warned_n,
            "coverage_pct": 100.0 * warned_n / len(onsets) if onsets else np.nan,
            "mean_lead_seconds": float(np.mean(leads)) if leads else 0.0,
            "median_lead_seconds": float(np.median(leads)) if leads else 0.0,
            "false_alarm_windows_per_hour": false_alerts / hours,
        })
        print(f"  K={k:<2} [{_region(k):<23}] thr={thr:.2f}  warned {warned_n}/{len(onsets)}"
              f"  mean lead {float(np.mean(leads)) if leads else 0.0:5.1f}s"
              f"  FP/hr {false_alerts / hours:6.2f}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "extended_warning_analysis.csv", index=False)
    print(f"\n  NOTE: {len(onsets)} observable onsets is far too few for a reliable "
          f"early-warning estimate. Reported, not relied upon.")
    print(f"  wrote {(OUT / 'extended_warning_analysis.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 13 — K=5 VS K=10 (Part 17)
# =============================================================

def step_k5_vs_k10(horizon, trajectory, latent_summary, forecast,
                   persistence, transitions, warning) -> pd.DataFrame:
    _rule("STEP 13 — what changes between K=5 and K=10?")

    def _pick(frame, k, column):
        sel = frame[frame["horizon"] == k]
        return float(sel[column].iloc[0]) if len(sel) and column in sel else np.nan

    def _warn(k, column):
        sel = warning[(warning["horizon"] == k) & (warning["onset_timestamp"] == "AGGREGATE")]
        return float(sel[column].iloc[0]) if len(sel) else np.nan

    items = [
        ("latent_norm_mean", latent_summary, "latent_norm_mean", "lower is not better — bounded by LayerNorm"),
        ("latent_delta_norm_mean", latent_summary, "delta_norm_mean", "step size; shrinking = contraction"),
        ("cumulative_displacement_mean", latent_summary, "cumulative_displacement_mean", "distance travelled from z_t"),
        ("across_sample_latent_std", latent_summary, "across_sample_latent_std", "sample discriminability; collapse would drive this to 0"),
        ("decoded_state_rmse", forecast, "state_rmse", "vs the true future state"),
        ("skill_vs_train_mean_rmse", forecast, "skill_vs_train_mean_rmse", "1 - rmse/mean-predictor rmse; <=0 means no skill"),
        ("mean_probability", forecast, "mean_probability", "mean P(t+k)"),
        ("brier", forecast, "brier", "lower is better"),
        ("roc_auc", horizon, "roc_auc", "threshold-free discrimination"),
        ("pr_auc", horizon, "pr_auc", "threshold-free, prevalence-sensitive"),
        ("f1", horizon, "f1", "K=5 frozen threshold vs K=10 exploratory — NOT like-for-like"),
        ("recall", horizon, "recall", "same caveat as f1"),
        ("precision", horizon, "precision", "same caveat as f1"),
        ("persistence_accuracy", persistence, "persistence_accuracy", "how often A(t+k)==A(t)"),
        ("transition_fraction", persistence, "transition_fraction", "the only part a forecaster can win"),
        ("onset_count", persistence, "onset_count", "sample size for onset statistics"),
        ("mean_onset_probability", transitions, "mean_onset_probability", "P on genuine onsets"),
        ("mean_persistent_negative_probability", transitions,
         "mean_persistent_negative_probability", "P on quiet-stays-quiet"),
        ("onset_minus_persistent_negative", transitions, "onset_minus_persistent_negative",
         "onset separation; <=0 means onsets are not distinguished"),
        ("trajectory_f1", trajectory, "f1", "max-aggregated event F1"),
        ("trajectory_roc_auc", trajectory, "roc_auc", "max-aggregated event ROC-AUC"),
    ]

    rows = []
    for name, frame, column, note in items:
        v5, v10 = _pick(frame, 5, column), _pick(frame, 10, column)
        rows.append({
            "quantity": name, "k5": v5, "k10": v10,
            "absolute_change": v10 - v5,
            "relative_change_pct": (100.0 * (v10 - v5) / abs(v5)) if v5 not in (0, np.nan) and np.isfinite(v5) and v5 != 0 else np.nan,
            "k5_region": "reference_in_curriculum", "k10_region": "extrapolative",
            "note": note,
        })
    for name, column in (("early_warning_coverage_pct", "coverage_pct"),
                         ("early_warning_mean_lead_s", "mean_lead_seconds"),
                         ("false_alarm_windows_per_hour", "false_alarm_windows_per_hour")):
        v5, v10 = _warn(5, column), _warn(10, column)
        rows.append({
            "quantity": name, "k5": v5, "k10": v10, "absolute_change": v10 - v5,
            "relative_change_pct": (100.0 * (v10 - v5) / abs(v5)) if v5 else np.nan,
            "k5_region": "reference_in_curriculum", "k10_region": "extrapolative",
            "note": "only 3 observable onsets — anecdotal",
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "k5_vs_k10.csv", index=False)
    for _, r in frame.iterrows():
        print(f"  {r['quantity']:<38} K5={r['k5']:9.4f}  K10={r['k10']:9.4f}  "
              f"delta={r['absolute_change']:+9.4f}")
    print(f"  wrote {(OUT / 'k5_vs_k10.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 14 — PRACTICAL STABILITY BOUNDARY (Part 18)
# =============================================================
# The rules below are declared BEFORE they are applied and are evaluated
# arithmetically. None of them is a subjective judgement, and none is test
# F1 alone. Every threshold is relative to the K=1 value of the SAME
# quantity, so the boundary is about DEGRADATION WITH HORIZON, not about
# whether V4 is good in absolute terms.

STABILITY_RULES = [
    {
        "indicator": "latent_divergence",
        "rule": "latent_norm_p99(K) <= 1.5 * latent_norm_p99(K=1)",
        "detects": "exploding latent magnitude",
        "counted": True,
    },
    {
        "indicator": "latent_collapse",
        "rule": "across_sample_latent_std(K) >= 0.5 * across_sample_latent_std(K=1)",
        "detects": "collapse of all samples toward one point, which would destroy "
                   "sample-to-sample discriminability",
        "counted": True,
    },
    {
        "indicator": "decoder_error_growth",
        "rule": "state_rmse(K) <= 1.25 * state_rmse(K=1)",
        "detects": "state forecast degrading with horizon",
        "counted": True,
    },
    {
        "indicator": "probability_discrimination",
        "rule": "roc_auc(K) - 0.5 >= 0.8 * (roc_auc(K=1) - 0.5)",
        "detects": "loss of threshold-free discriminative information",
        "counted": True,
    },
    {
        "indicator": "probability_saturation",
        "rule": "std(P(t+K)) >= 0.5 * std(P(t+1))  AND  "
                "frac(0.02 < P(t+K) < 0.98) >= 0.5",
        "detects": "probabilities collapsing to a constant or pinning at the rails",
        "counted": True,
    },
    {
        "indicator": "onset_separation",
        "rule": "mean P on genuine onsets >= mean P on persistent negatives",
        "detects": "loss of the ability to distinguish a real onset from quiet traffic",
        "counted": False,
        "counted_reason": "ADVISORY ONLY — onset counts are 3..12, far below the "
                          "pre-declared MIN_TRANSITION_N=20, so this indicator has too "
                          "little power to decide a status on its own. It is measured, "
                          "reported, and excluded from the tally by rule, not by "
                          "convenience.",
    },
]

STATUS_RULE = ("STABLE = 0 counted indicators failed; "
               "DEGRADING = 1-2 failed; UNRELIABLE = 3 or more failed")


def step_horizon_status(horizon, latent_summary, forecast, roll_test, transitions) -> pd.DataFrame:
    _rule("STEP 14 — practical stability boundary")
    print("  Rules (declared before evaluation):")
    for rule in STABILITY_RULES:
        flag = "counted" if rule["counted"] else "ADVISORY"
        print(f"    [{flag:<8}] {rule['indicator']}: {rule['rule']}")
    print(f"  Status: {STATUS_RULE}")
    print()

    P = roll_test["P"]
    ref_norm_p99 = float(latent_summary[latent_summary["horizon"] == 1]["latent_norm_p99"].iloc[0])
    ref_std = float(latent_summary[latent_summary["horizon"] == 1]["across_sample_latent_std"].iloc[0])
    ref_rmse = float(forecast[forecast["horizon"] == 1]["state_rmse"].iloc[0])
    ref_auc = float(horizon[horizon["horizon"] == 1]["roc_auc"].iloc[0])
    ref_pstd = float(P[:, 1].std())

    rows = []
    for k in range(1, MAX_K + 1):
        norm_p99 = float(latent_summary[latent_summary["horizon"] == k]["latent_norm_p99"].iloc[0])
        std_k = float(latent_summary[latent_summary["horizon"] == k]["across_sample_latent_std"].iloc[0])
        rmse = float(forecast[forecast["horizon"] == k]["state_rmse"].iloc[0])
        auc = float(horizon[horizon["horizon"] == k]["roc_auc"].iloc[0])
        pstd = float(P[:, k].std())
        unsat = float(((P[:, k] > 0.02) & (P[:, k] < 0.98)).mean())
        onset_p = float(transitions[transitions["horizon"] == k]["mean_onset_probability"].iloc[0])
        negp = float(transitions[transitions["horizon"] == k]
                     ["mean_persistent_negative_probability"].iloc[0])

        checks = {
            "latent_divergence": norm_p99 <= 1.5 * ref_norm_p99,
            "latent_collapse": std_k >= 0.5 * ref_std,
            "decoder_error_growth": rmse <= 1.25 * ref_rmse,
            "probability_discrimination": (auc - 0.5) >= 0.8 * (ref_auc - 0.5),
            "probability_saturation": (pstd >= 0.5 * ref_pstd) and (unsat >= 0.5),
        }
        advisory = {"onset_separation": onset_p >= negp}
        n_failed = sum(1 for ok in checks.values() if not ok)
        status = "STABLE" if n_failed == 0 else ("DEGRADING" if n_failed <= 2 else "UNRELIABLE")

        rows.append({
            "horizon": k, "region": _region(k), "status": status,
            "counted_indicators_failed": n_failed,
            **{f"pass_{name}": bool(ok) for name, ok in checks.items()},
            "advisory_onset_separation_pass": bool(advisory["onset_separation"]),
            "latent_norm_p99": norm_p99, "latent_norm_p99_limit": 1.5 * ref_norm_p99,
            "across_sample_latent_std": std_k, "across_sample_latent_std_floor": 0.5 * ref_std,
            "state_rmse": rmse, "state_rmse_limit": 1.25 * ref_rmse,
            "roc_auc": auc, "roc_auc_floor": 0.5 + 0.8 * (ref_auc - 0.5),
            "probability_std": pstd, "probability_std_floor": 0.5 * ref_pstd,
            "frac_probability_unsaturated": unsat,
            "mean_onset_probability": onset_p,
            "mean_persistent_negative_probability": negp,
            "status_rule": STATUS_RULE,
            "scope_note": "NUMERICAL STABILITY ONLY. Whether the rollout carries forecasting "
                          "information is a separate question answered in "
                          "nowcast_vs_forecast.csv — a horizon can be perfectly stable and "
                          "still uninformative.",
        })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "horizon_status.csv", index=False)
    print("   K  status      failed  divergence collapse decoder discrim saturation | advisory onset")
    for _, r in frame.iterrows():
        m = lambda b: "  ok " if b else " FAIL"
        print(f"  {int(r['horizon']):2d}  {r['status']:<10}  {int(r['counted_indicators_failed']):6d}"
              f"  {m(r['pass_latent_divergence'])}     {m(r['pass_latent_collapse'])}"
              f"  {m(r['pass_decoder_error_growth'])}  {m(r['pass_probability_discrimination'])}"
              f" {m(r['pass_probability_saturation'])}   | {m(r['advisory_onset_separation_pass'])}")
    print(f"  wrote {(OUT / 'horizon_status.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 15 — REPRESENTATIVE CASES (Part 20)
# =============================================================
# Criteria are declared first and resolved by argmin/argmax over the whole
# test set, with ties broken by the lowest sample_id. No case is hand-picked
# and failure cases are mandatory members of the list.

def step_representative(test, roll_test, prob_frame, horizon_frame) -> pd.DataFrame:
    _rule("STEP 15 — representative cases")
    Z, P = roll_test["Z"], roll_test["P"]
    delta = np.linalg.norm(np.diff(Z, axis=1), axis=2)
    displacement = np.linalg.norm(Z - Z[:, :1, :], axis=2)[:, MAX_K]
    shrink = delta[:, -1] / np.maximum(delta[:, 0], 1e-12)
    traj = P[:, 1:MAX_K + 1]
    prob_range = traj.max(axis=1) - traj.min(axis=1)
    osc = prob_frame["oscillation_count"].to_numpy()

    thr5 = FROZEN_POINT_THRESHOLDS[5]
    thr10 = float(horizon_frame[horizon_frame["horizon"] == 10]["threshold"].iloc[0])
    y5 = test["labels"][:, 4]
    y10 = test["labels"][:, 9]
    ok5 = ((P[:, 5] >= thr5).astype(int) == y5) & test["avail"][:, 4]
    ok10 = ((P[:, 10] >= thr10).astype(int) == y10) & test["avail"][:, 9]
    onset10 = (test["current"] == 0) & (y10 == 1) & test["avail"][:, 9]

    def _argmin(values, mask=None):
        v = values.astype(float).copy()
        if mask is not None:
            v[~mask] = np.inf
        return int(np.argmin(v))

    def _argmax(values, mask=None):
        v = values.astype(float).copy()
        if mask is not None:
            v[~mask] = -np.inf
        return int(np.argmax(v))

    quiet = prob_range < 0.10
    cases = [
        ("stable_rollout", "smallest cumulative latent displacement at K=10 among samples "
         "whose probability moves less than 0.10 across K=1..10",
         _argmin(displacement, quiet if quiet.any() else None)),
        ("gradual_drift", "sample whose cumulative displacement at K=10 is closest to the "
         "median, among monotonically increasing probability trajectories",
         _argmin(np.abs(displacement - np.median(displacement)),
                 prob_frame["monotonic_increasing"].to_numpy())),
        ("strong_latent_contraction", "smallest ratio of the last step size to the first "
         "(||d10||/||d1||)", _argmin(shrink)),
        ("strong_latent_divergence", "largest cumulative latent displacement at K=10",
         _argmax(displacement)),
        ("probability_saturation", "largest fraction of steps with P >= 0.98 or P <= 0.02, "
         "tie-broken by highest mean probability",
         _argmax(prob_frame["frac_steps_saturated"].to_numpy() * 10
                 + prob_frame["mean_probability"].to_numpy())),
        ("probability_oscillation", "most sign changes in the step-to-step probability delta",
         _argmax(osc)),
        ("correct_k5_degraded_k10", "correct at K=5 under the frozen threshold and WRONG at "
         "K=10 under the exploratory threshold; largest probability change between them",
         _argmax(np.abs(P[:, 10] - P[:, 5]), ok5 & (~ok10) & test["avail"][:, 9])),
        ("incorrect_k5_and_k10", "wrong at BOTH K=5 and K=10; largest K=10 probability error",
         _argmax(np.abs(P[:, 10] - y10.astype(float)),
                 (~ok5) & (~ok10) & test["avail"][:, 4] & test["avail"][:, 9])),
        ("transition_onset", "a genuine onset (A(t)=0, A(t+10)=1); lowest sample_id",
         _argmax(onset10.astype(float) * (len(onset10) - np.arange(len(onset10))))),
    ]

    rows = []
    for name, criterion, i in cases:
        rows.append({
            "case": name, "selection_criterion": criterion, "sample_id": i,
            "timestamp_t_plus_1": str(test["meta"]["timestamp"].iloc[i]),
            "A_current": int(test["current"][i]),
            "A_t_plus_5": int(y5[i]) if test["avail"][i, 4] else None,
            "A_t_plus_10": int(y10[i]) if test["avail"][i, 9] else None,
            "latent_norm_k0": float(np.linalg.norm(Z[i, 0])),
            "latent_norm_k10": float(np.linalg.norm(Z[i, MAX_K])),
            "cumulative_displacement_k10": float(displacement[i]),
            "delta_first": float(delta[i, 0]), "delta_last": float(delta[i, -1]),
            "delta_shrink_ratio": float(shrink[i]),
            "P_k1": float(P[i, 1]), "P_k5": float(P[i, 5]), "P_k10": float(P[i, 10]),
            "probability_range": float(prob_range[i]),
            "oscillation_count": int(osc[i]),
            "correct_at_k5_frozen_threshold": bool(ok5[i]),
            "correct_at_k10_exploratory_threshold": bool(ok10[i]),
            **{f"P_t{k}": float(P[i, k]) for k in range(1, MAX_K + 1)},
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "representative_cases.csv", index=False)
    for _, r in frame.iterrows():
        print(f"  {r['case']:<28} id={r['sample_id']:<5} "
              f"P1={r['P_k1']:.3f} P5={r['P_k5']:.3f} P10={r['P_k10']:.3f}  "
              f"disp={r['cumulative_displacement_k10']:.3f}  "
              f"A(t)={r['A_current']} A(t+10)={r['A_t_plus_10']}")
    print(f"  wrote {(OUT / 'representative_cases.csv').relative_to(REPO_ROOT)}")
    return frame


# =============================================================
# STEP 16 — FIGURES (Part 19)
# =============================================================

def _ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=8, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_MUTED)


def _split_plot(ax, x, y, label_ref="K=1-5 in-curriculum", label_ext="K=6-10 extrapolative",
                marker="o"):
    """Every horizon plot shows the two regions as visually distinct series,
    so a reference number is never mistaken for an extrapolated one."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ref = x <= TRAINED_K
    ext = x >= TRAINED_K
    ax.axvspan(TRAINED_K + 0.5, MAX_K + 0.4, color="#f2f1ec", zorder=0)
    ax.plot(x[ref], y[ref], marker=marker, color=C_REF, linewidth=2,
            markersize=6, label=label_ref, zorder=3)
    ax.plot(x[ext], y[ext], marker=marker, color=C_EXT, linewidth=2,
            markersize=6, linestyle="--", label=label_ext, zorder=3)
    ax.axvline(TRAINED_K + 0.5, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
    ax.set_xticks(range(1, MAX_K + 1))


def step_figures(horizon, trajectory, latent_summary, forecast, persistence,
                 transitions, contraction, representative, informativeness, roll_test, test):
    _rule("STEP 16 — figures")
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams["figure.facecolor"] = SURFACE

    def _single(name, title, xlabel, ylabel, frame, column, source, annotate=None):
        fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
        _split_plot(ax, frame["horizon"], frame[column])
        _ax(ax, title, xlabel, ylabel)
        if annotate:
            ax.text(0.015, 0.03, annotate, transform=ax.transAxes, fontsize=7.5,
                    color=INK_MUTED, va="bottom")
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED, loc="best")
        fig.text(0.015, 0.005, f"source: {source}", fontsize=6.5, color=INK_MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(FIG / name, dpi=160, facecolor=SURFACE)
        plt.close(fig)

    _single("01_f1_vs_horizon.png",
            "F1 vs rollout horizon",
            "K (rollout steps, 10 s each)", "F1",
            horizon, "f1", "horizon_performance.csv",
            "K=1,2,3,5 use FROZEN V4 thresholds; K=4,6-10 use validation-selected\n"
            "exploratory thresholds — the F1 series is therefore NOT like-for-like.")
    _single("02_roc_auc_vs_horizon.png",
            "ROC-AUC vs rollout horizon  (threshold-free)",
            "K (rollout steps)", "ROC-AUC",
            horizon, "roc_auc", "horizon_performance.csv",
            "Threshold-free, so this series IS comparable across the whole range.")
    _single("03_pr_auc_vs_horizon.png",
            "PR-AUC vs rollout horizon  (threshold-free)",
            "K (rollout steps)", "PR-AUC",
            horizon, "pr_auc", "horizon_performance.csv",
            "Positive prevalence is 0.652 at every horizon — read PR-AUC against that floor.")
    _single("04_persistence_accuracy_vs_horizon.png",
            "Persistence accuracy vs horizon  —  predicting A(t+k)=A(t)",
            "K (rollout steps)", "accuracy of the persistence rule",
            persistence, "persistence_accuracy", "persistence_context.csv",
            "The label barely changes over 100 s. Any aggregate metric at large K is\n"
            "mostly measuring this, not forecasting skill.")
    _single("05_transition_fraction_vs_horizon.png",
            "Transition fraction vs horizon  —  the only part a forecaster can win",
            "K (rollout steps)", "fraction with A(t+k) != A(t)",
            persistence, "transition_fraction", "persistence_context.csv")
    _single("06_latent_norm_vs_horizon.png",
            "Latent L2 norm vs rollout step",
            "rollout step k", "mean ||z|| over 3,115 samples",
            latent_summary[latent_summary["horizon"] >= 1], "latent_norm_mean",
            "latent_stability_summary.csv",
            "The transition ends in LayerNorm, which bounds the latent scale by construction.")
    _single("07_latent_delta_vs_horizon.png",
            "Latent step size vs rollout step",
            "rollout step k", "mean ||z_k - z_{k-1}||",
            latent_summary[latent_summary["horizon"] >= 1], "delta_norm_mean",
            "latent_stability_summary.csv",
            "Monotonically shrinking: each transition moves the state less than the last.")
    _single("08_cumulative_displacement_vs_horizon.png",
            "Cumulative latent displacement vs rollout step",
            "rollout step k", "mean ||z_k - z_0||",
            latent_summary[latent_summary["horizon"] >= 1], "cumulative_displacement_mean",
            "latent_stability_summary.csv",
            "Saturating, not diverging — consistent with the shrinking step size.")

    # 09 decoder error with baselines
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _split_plot(ax, forecast["horizon"], forecast["state_rmse"],
                "V4 decoded state, K=1-5", "V4 decoded state, K=6-10")
    ax.plot(forecast["horizon"], forecast["train_mean_state_rmse"], color=C_AUX2,
            linewidth=2, linestyle="-.", marker="s", markersize=4,
            label="predict the training mean", zorder=3)
    ax.plot(forecast["horizon"], forecast["persistence_state_rmse"], color=C_AUX,
            linewidth=2, linestyle=":", marker="^", markersize=4,
            label="persistence: copy the last observed state", zorder=3)
    _ax(ax, "Decoded-state forecast error vs horizon, against two trivial baselines",
        "K (rollout steps)", "RMSE in target-scaler units")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED, loc="lower right")
    ax.text(0.015, 0.05, "The decoded state never beats the mean predictor at any horizon.",
            transform=ax.transAxes, fontsize=7.5, color=INK)
    fig.text(0.015, 0.005, "source: forecast_error.csv", fontsize=6.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "09_decoder_error_vs_horizon.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 10 threat probability vs horizon, split by transition class
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    ks = transitions["horizon"].to_numpy()
    ax.axvspan(TRAINED_K + 0.5, MAX_K + 0.4, color="#f2f1ec", zorder=0)
    ax.axvline(TRAINED_K + 0.5, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
    for column, colour, label, style in (
        ("mean_persistent_positive_probability", C_REF, "persistent positive  A(t)=1 -> A(t+k)=1", "-"),
        ("mean_offset_probability", C_AUX2, "offset  A(t)=1 -> A(t+k)=0  (n<=12)", "--"),
        ("mean_onset_probability", C_EXT, "ONSET  A(t)=0 -> A(t+k)=1  (n<=12)", "-"),
        ("mean_persistent_negative_probability", C_AUX, "persistent negative  A(t)=0 -> A(t+k)=0", "-"),
    ):
        ax.plot(ks, transitions[column], marker="o", markersize=5, color=colour,
                linewidth=2, linestyle=style, label=label, zorder=3)
    _ax(ax, "Mean predicted probability by transition class\n"
            "Onsets get LESS likely as the horizon grows, while persistent positives get more likely",
        "K (rollout steps)", "mean P(t+k)")
    ax.set_xticks(range(1, MAX_K + 1))
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK_MUTED, loc="center right")
    fig.text(0.015, 0.005, "source: transition_metrics.csv", fontsize=6.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "10_threat_probability_vs_horizon.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 11 K=5 vs K=10 latent trajectory comparison
    Z = roll_test["Z"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor=SURFACE)
    z0 = Z[:, 0, :]
    centred = z0 - z0.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:2].T
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(Z), size=120, replace=False))
    proj = (Z[pick] - z0.mean(axis=0)) @ basis
    for row in proj:
        axes[0].plot(row[:TRAINED_K + 1, 0], row[:TRAINED_K + 1, 1],
                     color=C_REF, alpha=0.30, linewidth=1, zorder=2)
        axes[0].plot(row[TRAINED_K:, 0], row[TRAINED_K:, 1],
                     color=C_EXT, alpha=0.30, linewidth=1, linestyle="--", zorder=3)
    axes[0].scatter(proj[:, 0, 0], proj[:, 0, 1], s=10, color=INK, zorder=4, label="z_t (encoder)")
    axes[0].plot([], [], color=C_REF, linewidth=2, label="k=0-5 in-curriculum")
    axes[0].plot([], [], color=C_EXT, linewidth=2, linestyle="--", label="k=5-10 extrapolative")
    _ax(axes[0], "Latent trajectories, first 2 principal components of z_t\n"
                 "120 deterministically sampled test sequences",
        "PC1", "PC2")
    axes[0].grid(color=GRID, linewidth=0.8)
    axes[0].legend(fontsize=7.5, frameon=False, labelcolor=INK_MUTED, loc="best")

    disp = np.linalg.norm(Z - Z[:, :1, :], axis=2)
    steps = np.arange(MAX_K + 1)
    for q, alpha, label in ((50, 1.0, "median"), (90, 0.55, "p90"), (99, 0.3, "p99")):
        vals = np.percentile(disp, q, axis=0)
        axes[1].plot(steps[:TRAINED_K + 1], vals[:TRAINED_K + 1], color=C_REF,
                     linewidth=2, alpha=alpha, marker="o", markersize=4)
        axes[1].plot(steps[TRAINED_K:], vals[TRAINED_K:], color=C_EXT, linewidth=2,
                     alpha=alpha, linestyle="--", marker="o", markersize=4, label=label)
    axes[1].axvspan(TRAINED_K + 0.5, MAX_K + 0.4, color="#f2f1ec", zorder=0)
    axes[1].axvline(TRAINED_K + 0.5, color=INK_MUTED, linewidth=1, linestyle=":")
    _ax(axes[1], "Cumulative latent displacement ||z_k - z_0||\n"
                 "Saturating across the whole distribution, not just on average",
        "rollout step k", "displacement")
    axes[1].set_xticks(steps)
    axes[1].legend(fontsize=8, frameon=False, labelcolor=INK_MUTED, title="percentile",
                   title_fontsize=8, loc="lower right")
    fig.text(0.01, 0.005, "source: latent_stability.csv / latent_stability_summary.csv",
             fontsize=6.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "11_k5_vs_k10_latent_trajectories.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 12 representative rollout trajectories
    P = roll_test["P"]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 8.0), facecolor=SURFACE)
    for ax, (_, case) in zip(axes.ravel(), representative.iterrows()):
        i = int(case["sample_id"])
        y = P[i, 1:MAX_K + 1]
        x = np.arange(1, MAX_K + 1)
        ax.axvspan(TRAINED_K + 0.5, MAX_K + 0.4, color="#f2f1ec", zorder=0)
        ax.plot(x[:TRAINED_K], y[:TRAINED_K], color=C_REF, marker="o", markersize=4,
                linewidth=2, zorder=3)
        ax.plot(x[TRAINED_K - 1:], y[TRAINED_K - 1:], color=C_EXT, marker="o", markersize=4,
                linewidth=2, linestyle="--", zorder=3)
        ax.axhline(FROZEN_POINT_THRESHOLDS[5], color=INK_MUTED, linewidth=1, linestyle=":")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(x)
        label = (f"{case['case']}\nid={i}  A(t)={case['A_current']}  "
                 f"A(t+10)={case['A_t_plus_10']}")
        _ax(ax, label, "K", "P(t+K)")
        ax.title.set_fontsize(8.5)
    fig.suptitle("Representative rollout trajectories — selected by pre-declared criteria, "
                 "failures included\nblue = K=1-5 in-curriculum, orange dashed = K=6-10 "
                 "extrapolative, dotted line = frozen K=5 threshold 0.72",
                 fontsize=10.5, color=INK, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005, "source: representative_cases.csv", fontsize=6.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(FIG / "12_representative_trajectories.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # 13 forecast vs nowcast — the decisive, fully powered comparison
    fig, ax = plt.subplots(figsize=(7.8, 4.4), facecolor=SURFACE)
    ks = informativeness["horizon"].to_numpy()
    ax.axvspan(TRAINED_K + 0.5, MAX_K + 0.4, color="#f2f1ec", zorder=0)
    ax.axvline(TRAINED_K + 0.5, color=INK_MUTED, linewidth=1, linestyle=":", zorder=2)
    ax.plot(ks, informativeness["roc_auc_vs_current_label"], marker="s", markersize=5,
            color=C_AUX2, linewidth=2, linestyle="--",
            label="predicting the CURRENT label A(t)  — a nowcast", zorder=3)
    ax.plot(ks, informativeness["roc_auc_vs_future_label"], marker="o", markersize=5,
            color=C_REF, linewidth=2,
            label="predicting the FUTURE label A(t+K) — the intended forecast", zorder=4)
    _ax(ax, "The k-step rollout describes the present, not the future\n"
            "At every horizon the probability ranks A(t) at least as well as A(t+K), "
            "and the gap widens",
        "K (rollout steps)", "ROC-AUC  (n ~ 3,105 — fully powered)")
    ax.set_xticks(range(1, MAX_K + 1))
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_MUTED, loc="lower right")
    for k in (1, 5, 10):
        r = informativeness[informativeness["horizon"] == k].iloc[0]
        ax.annotate(f"{r['incremental_auc']:+.4f}",
                    xy=(k, r["roc_auc_vs_future_label"]), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    fig.text(0.015, 0.005, "source: nowcast_vs_forecast.csv  |  annotations = incremental AUC "
             "(future minus current)", fontsize=6.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "13_forecast_vs_nowcast.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    for path in sorted(FIG.glob("*.png")):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")


# =============================================================
# STEP 17 — SUMMARY JSON (Part 21)
# =============================================================

def step_summary(verify, horizon, trajectory, latent_summary, contraction, prob_frame,
                 forecast, persistence, transitions, informativeness, warning, k5k10, status,
                 representative, ckpt) -> dict:
    _rule("STEP 17 — machine-readable summary")

    agg_warn = warning[warning["onset_timestamp"] == "AGGREGATE"]
    stable = status[status["status"] == "STABLE"]["horizon"].tolist()
    degrading = status[status["status"] == "DEGRADING"]["horizon"].tolist()
    unreliable = status[status["status"] == "UNRELIABLE"]["horizon"].tolist()

    import sklearn
    summary = {
        "task": "Task 10 — long-horizon rollout and latent stability",
        "generated_at_utc": GENERATED_AT,
        "experiment_type": "inference-only diagnostic — nothing trained, tuned or modified",
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "device": "cpu (deterministic; reproduces the frozen MPS reference exactly)",
        },
        "frozen_v4": {
            "verification": verify["report"],
            "checkpoint_sha256": V4_FROZEN_HASHES[
                "models/latent_world_model_v4/latent_world_model_v4.pt"],
            "trained_curriculum": ckpt["curriculum"],
            "trained_max_k": ckpt["max_k"],
            "latent_size": ckpt["latent_size"],
            "input_size": ckpt["input_size"],
            "transition_type": ckpt["transition_type"],
        },
        "rollout_definition": {
            "formula": "z_t = encode(X_t); z_{t+k+1} = T(z_{t+k})",
            "teacher_forcing": False,
            "observations_consumed": "the single length-10 input window X_t and nothing else",
            "max_k": MAX_K,
            "index_convention": "P[:, k] = after exactly k transitions = prediction for t+k; "
                                "P[:, 0] is the encoder latent with no transition applied",
        },
        "horizon_regions": {
            "reference_in_curriculum": [1, 2, 3, 4, 5],
            "extrapolative": [6, 7, 8, 9, 10],
            "statement": "The model was trained with curriculum depth through K=5; K=6..10 are "
                         "recursive extrapolation diagnostics, NOT trained horizons. K=10 is a "
                         "10-step recursive rollout, not a trained 100-second forecast.",
            "k4_note": "K=4 is inside the training curriculum but has NO frozen reference "
                       "number — the reference script evaluated horizons [1,2,3,5] only. Its "
                       "threshold here is validation-selected and exploratory.",
        },
        "threshold_policy": {
            "frozen_point_thresholds": FROZEN_POINT_THRESHOLDS,
            "frozen_trajectory_max_thresholds": FROZEN_TRAJ_MAX_THRESHOLDS,
            "exploratory_selection": "np.arange(0.01, 0.99, 0.01), maximise VALIDATION F1, "
                                     "strict > so the first maximiser wins — identical to the "
                                     "frozen reference's own rule",
            "test_labels_used_for_threshold_selection": False,
        },
        "horizon_performance": horizon.to_dict("records"),
        "trajectory_performance": trajectory.to_dict("records"),
        "latent_stability": latent_summary.to_dict("records"),
        "latent_contraction": contraction.drop(columns=["note"]).to_dict("records"),
        "probability_stability": {
            "n_samples": int(len(prob_frame)),
            "monotonic_increasing_pct": float(prob_frame["monotonic_increasing"].mean() * 100),
            "monotonic_decreasing_pct": float(prob_frame["monotonic_decreasing"].mean() * 100),
            "mean_oscillation_count": float(prob_frame["oscillation_count"].mean()),
            "mean_probability_range": float(prob_frame["probability_range"].mean()),
            "mean_drift_p10_minus_p1": float(prob_frame["total_probability_delta"].mean()),
            "pct_any_step_saturated": float(prob_frame["frac_steps_saturated"].gt(0).mean() * 100),
            "per_horizon_mean_probability": {
                int(r["horizon"]): float(r["mean_probability"])
                for _, r in forecast.iterrows()},
        },
        "forecast_error": forecast.drop(columns=["state_target_note"]).to_dict("records"),
        "persistence_context": persistence.to_dict("records"),
        "transition_metrics": transitions.to_dict("records"),
        "warning_metrics": agg_warn.dropna(axis=1, how="all").to_dict("records"),
        "k5_vs_k10": k5k10.to_dict("records"),
        "informativeness": {
            "rules": INFORMATIVENESS_RULES,
            "verdict_rule": INFORMATIVENESS_VERDICT_RULE,
            "per_horizon": informativeness.drop(columns=["verdict_rule"]).to_dict("records"),
            "headline": "At every horizon K=1..10 the k-step probability predicts the CURRENT "
                        "label A(t) at least as well as the future label A(t+k), and the gap "
                        "widens monotonically with K. The rollout behaves as a nowcast that "
                        "scores well on the future because the label persists, not as a "
                        "forecast.",
        },
        "horizon_status": {
            "rules": STABILITY_RULES,
            "status_rule": STATUS_RULE,
            "per_horizon": status.to_dict("records"),
            "stable_horizons": stable,
            "degrading_horizons": degrading,
            "unreliable_horizons": unreliable,
            "practical_operating_horizon_numerical_stability": max(stable) if stable else None,
            "practical_operating_horizon_note":
                "Numerical stability alone does not make a horizon useful. See "
                "informativeness.per_horizon: the incremental AUC over a pure nowcast is "
                "negative at EVERY horizon and worsens monotonically, so no horizon in "
                "K=1..10 is demonstrated to carry forecasting information beyond a "
                "description of the present.",
        },
        "representative_cases": representative[
            ["case", "selection_criterion", "sample_id", "A_current", "A_t_plus_10",
             "P_k1", "P_k5", "P_k10", "cumulative_displacement_k10", "delta_shrink_ratio",
             "oscillation_count", "correct_at_k5_frozen_threshold",
             "correct_at_k10_exploratory_threshold"]].to_dict("records"),
        "limitations": [
            "PROVENANCE, INHERITED NOT RESOLVED: the frozen V4 checkpoint was trained on the "
            "single-day temporal_benchmark_v2 arrays but is evaluated here on "
            "multiday_benchmark_K5 arrays scaled by a different StandardScaler. Every number "
            "in this task inherits that mismatch (docs/v4_baseline_manifest.json known_issues). "
            "This experiment does not repair it and does not resolve it.",
            "The test split is a SINGLE day (Friday-02-03-2018) with only 3 observable attack "
            "onsets, so every early-warning figure is anecdotal rather than an estimate.",
            "Transition-conditioned statistics rest on 3 to 12 onsets and 3 to 12 offsets. They "
            "are reported because the direction of the effect is consistent, but no individual "
            "value is statistically reliable.",
            "Aggregate metrics at large K are dominated by label persistence (99.2% at K=10), "
            "not by forecasting skill. Reading ROC-AUC alone across horizons is misleading.",
            "The F1 series across K is NOT like-for-like: K=1,2,3,5 use frozen thresholds and "
            "K=4,6..10 use validation-selected exploratory thresholds.",
            "Ground truth for K=6..10 was reconstructed from the same states_v2 source the "
            "benchmark was built from; 10 of 3,115 sequences lack a contiguous K=10 target and "
            "are excluded at that horizon rather than imputed.",
            "The stability boundary is defined relative to each quantity's own K=1 value, so it "
            "measures DEGRADATION WITH HORIZON. It says nothing about whether V4 is good in "
            "absolute terms — and on the state-decoder arm it is not (see forecast_error.csv).",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"  wrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return summary


# =============================================================
# MAIN
# =============================================================

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 10 — LONG-HORIZON ROLLOUT + LATENT STABILITY")
    print("=" * 70)
    print(f"generated: {GENERATED_AT}")
    print("INFERENCE ONLY — no training, no tuning, no modification of V4")

    verify = step_verify_v4()
    if verify["report"]["status"] != "PASS":
        print("\nFROZEN V4 IS MODIFIED — aborting before any analysis.")
        return 2
    model, ckpt = verify["model"], verify["ckpt"]

    _rule("STEP 2 — load benchmark and reconstruct extended ground truth")
    state_features = [l.strip() for l in
                      (BENCH / "state_feature_names.txt").read_text().splitlines() if l.strip()]
    with open(BENCH / "target_scaler.pkl", "rb") as fh:
        target_scaler = pickle.load(fh)

    X_test = np.load(BENCH / "X_test.npy")
    X_val = np.load(BENCH / "X_val.npy")
    A_test = np.load(BENCH / "A_binary_multistep_test.npy")
    A_val = np.load(BENCH / "A_binary_multistep_val.npy")
    print(f"  X_test {X_test.shape}   X_val {X_val.shape}")

    test = build_extended_truth("test", state_features, target_scaler, A_test)
    val = build_extended_truth("val", state_features, target_scaler, A_val)

    _rule("STEP 3 — autoregressive rollout, K=1..10")
    roll_test = rollout(model, X_test, MAX_K)
    roll_val = rollout(model, X_val, MAX_K)
    print(f"  test: probs {roll_test['P'].shape}  latents {roll_test['Z'].shape}  "
          f"decoded {roll_test['S'].shape}")
    print("  teacher forcing: NONE — only X_t enters the loop")

    horizon = step_horizon_performance(test, val, roll_test, roll_val)
    trajectory = step_trajectory(test, val, roll_test, roll_val)
    _, latent_summary = step_latent_stability(test, roll_test)
    contraction = step_contraction(roll_test)
    prob_frame = step_probability_stability(test, roll_test, horizon)
    forecast = step_forecast_error(test, roll_test)
    persistence = step_persistence(test)
    transitions = step_transitions(test, roll_test, horizon)
    informativeness = step_nowcast_vs_forecast(test, roll_test, transitions)
    warning = step_warning(test, val, roll_test, roll_val)
    k5k10 = step_k5_vs_k10(horizon, trajectory, latent_summary, forecast,
                           persistence, transitions, warning)
    status = step_horizon_status(horizon, latent_summary, forecast, roll_test, transitions)
    representative = step_representative(test, roll_test, prob_frame, horizon)
    step_figures(horizon, trajectory, latent_summary, forecast, persistence,
                 transitions, contraction, representative, informativeness, roll_test, test)
    summary = step_summary(verify, horizon, trajectory, latent_summary, contraction,
                           prob_frame, forecast, persistence, transitions, informativeness,
                           warning, k5k10, status, representative, ckpt)

    # V4 must still be untouched after everything.
    after = step_verify_v4()
    if after["report"]["status"] != "PASS":
        print("\nV4 INTEGRITY VIOLATION detected AFTER the run.")
        return 2

    _rule("DONE")
    st = summary["horizon_status"]
    print(f"  STABLE horizons     : {st['stable_horizons']}")
    print(f"  DEGRADING horizons  : {st['degrading_horizons']}")
    print(f"  UNRELIABLE horizons : {st['unreliable_horizons']}")
    print(f"  numerically stable through  : K={st['practical_operating_horizon_numerical_stability']}")
    inf_rows = summary["informativeness"]["per_horizon"]
    nowcast = [r["horizon"] for r in inf_rows if r["informativeness_verdict"] != "FORECAST_INFORMATIVE"]
    anti = [r["horizon"] for r in inf_rows if r["informativeness_verdict"] == "ANTI_INFORMATIVE"]
    print(f"  horizons beating a nowcast  : "
          f"{[r['horizon'] for r in inf_rows if r['informativeness_verdict'] == 'FORECAST_INFORMATIVE'] or 'NONE'}")
    print(f"  nowcast-equivalent or worse : K={nowcast}")
    print(f"  ANTI_INFORMATIVE            : K={anti}")
    print(f"  V4 integrity        : {after['report']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
