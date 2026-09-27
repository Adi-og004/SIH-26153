"""
=============================================================
V4 RELIABILITY LAYER — runtime failure / model-health signals
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Task 15. A decision-support layer that sits ON TOP of the FROZEN V4
world model and reports when a forecast should not be trusted.

This module NEVER trains, fine-tunes, refits a scaler, rethresholds the
V4 alert rule, or introduces a learned model. It reads the frozen
artifacts, re-runs the frozen inference path, and computes arithmetic
indicators from the result.

WHAT THIS IS NOT
----------------
The status this module emits is NOT a calibrated probability, NOT a
confidence score, and NOT a probability of correctness. It is a
rule-based status derived from a fixed reference distribution. No
reliability estimator was fitted, and no held-out test label was used
to choose any threshold.

THRESHOLD PROVENANCE
--------------------
Every distributional threshold comes from ONE place: the quantiles of
X_train.npy in data/processed/multiday_benchmark_K5 — the train
partition the frozen input_scaler was fit on — pushed through the
frozen checkpoint. Labels are never read. The val split is used only
to report the rule's flag rate on a second label-free reference split.

The two latent-norm fail-safe bounds are not distributional at all:
they are derived analytically from the frozen LayerNorm and Linear
weights (see derive_latent_norm_bounds) and are numerical
impossibility conditions.

KNOWN INHERITED LIMITATION
--------------------------
docs/v4_baseline_manifest.json -> known_issues records that the V4
checkpoint was TRAINED on the single-day temporal_benchmark_v2 arrays
(scaler fit on Feb-28 only) while the reference EVALUATION feeds it
multiday_benchmark_K5 arrays scaled by a DIFFERENT StandardScaler.
The reference distribution used here is therefore the distribution of
the SCALER THE INFERENCE PATH ACTUALLY USES, not the distribution the
checkpoint's weights were fitted against — which no longer exists on
disk. This layer measures, and does not repair, that mismatch.

Usage:
    import v4_reliability as rel
    ref = rel.load_reference()                  # frozen config JSON
    a   = rel.assess(x, probs, latents, states, ref)
    print(a.status, a.headline)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
MODEL_DIR = REPO_ROOT / "models" / "latent_world_model_v4"
CHECKPOINT = MODEL_DIR / "latent_world_model_v4.pt"
MANIFEST_PATH = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
OUT_DIR = REPO_ROOT / "results" / "v4_reliability"
CONFIG_PATH = OUT_DIR / "reliability_config.json"

LAYER_VERSION = "v4-reliability-1.0"

SEQ_LEN = 10
N_FEATURES = 52
N_STATE_FEATURES = 26
LATENT_SIZE = 32
MAX_K = 5

# The reliability layer runs on CPU by design, so its outputs are
# bit-identical run to run and machine to machine. The frozen
# screening_predictions.csv was produced on MPS; the resulting
# device-level gap is ~2e-6 in probability and is recorded, not hidden.
DEVICE = "cpu"

# Statuses, ordered least to most severe.
RELIABLE, CAUTION, UNRELIABLE = "RELIABLE", "CAUTION", "UNRELIABLE"
STATUS_ORDER = [RELIABLE, CAUTION, UNRELIABLE]

# Compact wording for the Live Forecast page. Deliberately NOT
# "confidence" / "certainty" / "probability of correctness".
STATUS_SHORT = {RELIABLE: "HIGH", CAUTION: "CAUTION", UNRELIABLE: "LOW"}


# =============================================================
# INDICATOR REGISTRY
# =============================================================
#
# Every indicator is (a) computed with arithmetic only, (b) one-sided
# (larger = further from the reference), and (c) tiered against the
# SAME reference distribution. "larger = worse" is asserted per
# indicator below; where the retrospective evidence contradicts it,
# that is recorded in the evidence pack rather than silently fixed.

@dataclass(frozen=True)
class Indicator:
    name: str
    group: str
    description: str
    # Plain-language clause used to build the explanation sentence.
    clause: str


INDICATORS: tuple[Indicator, ...] = (
    # ---- A. INPUT OOD / DISTANCE (frozen V4 input scaler) ----
    Indicator("rms_z", "input_ood",
              "Overall standardized magnitude: RMS of all 10x52 scaled "
              "feature values. 1.0 = a window at the training scale.",
              "the window's overall standardized magnitude"),
    Indicator("max_abs_z", "input_ood",
              "Largest single |z| anywhere in the 10x52 input tensor.",
              "the largest single standardized feature value"),
    Indicator("frac_gt2", "input_ood",
              "Fraction of the 520 scaled input values with |z| > 2.",
              "the fraction of features beyond |z|>2"),
    Indicator("frac_gt3", "input_ood",
              "Fraction of the 520 scaled input values with |z| > 3.",
              "the fraction of features beyond |z|>3"),
    Indicator("feature_support_exceedance", "input_ood",
              "Distance, in training standard deviations, by which the "
              "most extreme input value falls OUTSIDE the per-feature "
              "min/max range observed in the reference train split. "
              "0.0 means every value lies inside observed support.",
              "an input value outside the reference range for its feature"),

    # ---- B. ROLLOUT STABILITY ----
    Indicator("latent_norm_max", "rollout_stability",
              "Largest latent norm ||z_k|| over k=0..5.",
              "the peak latent magnitude"),
    Indicator("latent_norm_change", "rollout_stability",
              "Peak latent norm minus the encoder-output norm "
              "(max_k ||z_k|| - ||z_0||): latent growth across the rollout.",
              "latent magnitude growth across the rollout"),
    Indicator("step_delta_max", "rollout_stability",
              "Largest step-to-step latent jump ||z_k - z_{k-1}||, k=1..5.",
              "the largest latent step taken by the transition"),
    Indicator("displacement", "rollout_stability",
              "Total latent displacement ||z_5 - z_0||.",
              "total latent displacement over the 50 s rollout"),

    # ---- C. FORECAST SELF-CONSISTENCY ----
    Indicator("state_delta_max", "forecast_consistency",
              "Largest step-to-step change in the decoded 26-D state, "
              "||s_k - s_{k-1}||, in target_scaler space.",
              "the sharpest change in the predicted state"),
    Indicator("prob_max_abs_step", "forecast_consistency",
              "Largest single-step change in threat probability "
              "|P_k - P_{k-1}| over the rollout.",
              "the sharpest jump in threat probability"),
    Indicator("prob_range", "forecast_consistency",
              "max_k P_k - min_k P_k over k=0..5.",
              "the spread of threat probability across horizons"),
)

INDICATOR_NAMES = tuple(i.name for i in INDICATORS)
BY_NAME = {i.name: i for i in INDICATORS}

# feature_support_exceedance is handled as a HARD condition (H8) and is
# NOT also tiered against train quantiles: in the reference split it is
# identically 0 by construction, so p99 and max are both 0 and every
# non-zero value would trivially land BEYOND_REFERENCE. Tiering it would
# double-count the same evidence.
TIERED_INDICATORS = tuple(n for n in INDICATOR_NAMES
                          if n != "feature_support_exceedance")

# Tier labels.
TIER_OK, TIER_ELEVATED, TIER_BEYOND = "OK", "ELEVATED", "BEYOND_REFERENCE"

# ---- STATUS COMPOSITION RULE (declared, then applied unchanged) ----
# UNRELIABLE : any hard fail-safe condition fires, OR >= 2 indicators
#              land strictly outside the reference distribution's
#              observed support.
# CAUTION    : >= 1 indicator outside reference support, OR >= 2
#              indicators above the reference 99th percentile.
# RELIABLE   : otherwise.
RULE_BEYOND_FOR_UNRELIABLE = 2
RULE_BEYOND_FOR_CAUTION = 1
RULE_ELEVATED_FOR_CAUTION = 2

# Margin, in training standard deviations, past the reference per-feature
# observed range at which an input is treated as a hard extreme shift.
#
# Justification (no test labels involved). The input tensor is already in
# frozen-scaler z units, so 1.0 = one training standard deviation. Two
# independent grounds, both label-free:
#
#   (a) It must not fire on data the reference distribution covers. A
#       margin of 0 does: 10 val sequences extend the train range slightly,
#       which is what any finite sample does. The smallest margin in
#       {0, 1, 3, 5} that flags NEITHER reference split is 1.0 SD, so 3.0
#       satisfies this requirement with room to spare.
#   (b) 3.0 is not a new free parameter. |z| > 3 is already one of this
#       layer's indicators (frac_gt3, per the task specification), so the
#       hard condition reuses the same conventional numerical-outlier
#       threshold rather than introducing another constant.
#
# 3.0 is therefore the more conservative of two defensible values. The full
# sensitivity across all four candidate margins — including that 1.0 SD
# would flag 21 held-out sequences where 3.0 SD flags 11 — is recorded in
# reliability_summary.json -> input_shift_margin_sensitivity.
INPUT_SHIFT_HARD_MARGIN_SD = 3.0

# Reference quantile used for the ELEVATED tier.
ELEVATED_QUANTILE = 99.0


# =============================================================
# DERIVED NUMERICAL BOUNDS (fail-safe conditions H6 / H7)
# =============================================================

def derive_latent_norm_bounds(state_dict) -> dict:
    """Analytic upper bounds on ||z|| implied by the frozen weights.

    These are NOT fitted and NOT distributional. They follow from the
    architecture:

    k >= 1 (post-transition):
        z_k = transition_norm(u) = w (*) n + b, where LayerNorm makes n
        zero-mean and unit-variance over its 32 dims, so
        ||n||_2 = sqrt(32 * var/(var+eps)) <= sqrt(32).
        ||z_k|| <= ||w||_inf * sqrt(32) + ||b||_2.

    k = 0 (encoder output):
        h  = encoder_norm(lstm_out) with ||h|| <= ||w_e||_inf*sqrt(128)
             + ||b_e||_2
        z_0 = W h + b, so ||z_0|| <= sigma_max(W) * ||h|| + ||b||_2.

    Exceeding either bound is numerically impossible for an intact
    checkpoint, so it can only mean corruption, a wrong state dict, or
    a broken tensor — never an unusual input.
    """
    def _t(key):
        v = state_dict[key]
        return v.detach().cpu().numpy().astype(np.float64)

    tw, tb = _t("transition_norm.weight"), _t("transition_norm.bias")
    ew, eb = _t("encoder_norm.weight"), _t("encoder_norm.bias")
    fw, fb = _t("encoder_fc.weight"), _t("encoder_fc.bias")

    latent_dim = tw.shape[0]
    hidden_dim = ew.shape[0]

    transition_bound = (float(np.abs(tw).max()) * math.sqrt(latent_dim)
                        + float(np.linalg.norm(tb)))
    encoder_norm_bound = (float(np.abs(ew).max()) * math.sqrt(hidden_dim)
                          + float(np.linalg.norm(eb)))
    spectral = float(np.linalg.svd(fw, compute_uv=False)[0])
    encoder_bound = spectral * encoder_norm_bound + float(np.linalg.norm(fb))

    return {
        "encoder_output_norm_bound": encoder_bound,
        "transition_output_norm_bound": transition_bound,
        "derivation": {
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "transition_norm_weight_inf": float(np.abs(tw).max()),
            "transition_norm_bias_l2": float(np.linalg.norm(tb)),
            "encoder_norm_weight_inf": float(np.abs(ew).max()),
            "encoder_norm_bias_l2": float(np.linalg.norm(eb)),
            "encoder_fc_spectral_norm": spectral,
            "encoder_fc_bias_l2": float(np.linalg.norm(fb)),
            "formula_k_ge_1": "||z_k|| <= ||w||_inf * sqrt(d) + ||b||_2",
            "formula_k_eq_0": ("||z_0|| <= sigma_max(W_fc) * "
                               "(||w_enc||_inf*sqrt(h) + ||b_enc||_2) + ||b_fc||_2"),
        },
    }


# =============================================================
# REFERENCE DISTRIBUTION
# =============================================================

@dataclass
class Reference:
    """The frozen, label-free reference distribution and derived bounds."""
    version: str
    source: str
    n_reference_sequences: int
    quantiles: dict                 # name -> {"p99": float, "max": float}
    feature_min: np.ndarray         # (52,) reference per-feature minimum
    feature_max: np.ndarray         # (52,) reference per-feature maximum
    feature_names: list[str]
    bounds: dict                    # derive_latent_norm_bounds output
    scaler_contract: dict           # expected scaler shape / hash / order
    provenance: dict = field(default_factory=dict)

    def tier(self, name: str, value: float) -> str:
        q = self.quantiles[name]
        if value > q["max"]:
            return TIER_BEYOND
        if value > q["p99"]:
            return TIER_ELEVATED
        return TIER_OK


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(path: Path | str = CONFIG_PATH) -> Reference:
    """Load the frozen reliability config written by build_v4_reliability.py."""
    path = Path(path)
    if not path.exists():
        raise ReliabilityConfigError(
            f"reliability config not found: {path}. Build it with "
            f"`python src/build_v4_reliability.py`. There is no fallback: "
            f"a reliability status without its reference distribution would "
            f"be meaningless."
        )
    cfg = json.loads(path.read_text())
    if cfg.get("layer_version") != LAYER_VERSION:
        raise ReliabilityConfigError(
            f"reliability config version {cfg.get('layer_version')!r} does "
            f"not match this module ({LAYER_VERSION!r})"
        )
    return Reference(
        version=cfg["layer_version"],
        source=cfg["reference_distribution"]["source"],
        n_reference_sequences=cfg["reference_distribution"]["n_sequences"],
        quantiles={k: dict(v) for k, v in cfg["indicator_reference"].items()},
        feature_min=np.asarray(cfg["feature_support"]["min"], dtype=np.float64),
        feature_max=np.asarray(cfg["feature_support"]["max"], dtype=np.float64),
        feature_names=list(cfg["feature_support"]["feature_names"]),
        bounds=cfg["numerical_bounds"],
        scaler_contract=cfg["scaler_contract"],
        provenance=cfg.get("provenance", {}),
    )


class ReliabilityConfigError(RuntimeError):
    """Raised when the reliability reference cannot be loaded or trusted.

    Callers must surface this. A reliability layer that silently degrades
    into "looks fine" is worse than no reliability layer at all."""


# =============================================================
# INDICATOR COMPUTATION
# =============================================================

def compute_indicators(x, probs, latents, states, ref: Reference) -> dict:
    """Arithmetic reliability indicators for ONE sequence.

    x        (10, 52) or (1, 10, 52) frozen scaled input — the only
             tensor the model saw. No future observation, no label.
    probs    (6,)      rollout threat probabilities, k=0..5
    latents  (6, 32)   latent trajectory
    states   (6, 26)   decoded state trajectory

    All inputs are model-side quantities. Nothing observed after the
    forecast window is accepted here.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[0]
    probs = np.asarray(probs, dtype=np.float64).ravel()
    latents = np.asarray(latents, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)

    a = np.abs(x)
    zn = np.linalg.norm(latents, axis=1)
    step = np.linalg.norm(np.diff(latents, axis=0), axis=1)
    sstep = np.linalg.norm(np.diff(states, axis=0), axis=1)
    pstep = np.abs(np.diff(probs))

    # Signed, one-sided distance outside the reference per-feature range.
    over = np.maximum(x - ref.feature_max[None, :], 0.0)
    under = np.maximum(ref.feature_min[None, :] - x, 0.0)
    exceed = np.maximum(over, under)

    out = {
        "rms_z": float(np.sqrt((x ** 2).mean())),
        "max_abs_z": float(a.max()),
        "frac_gt2": float((a > 2).mean()),
        "frac_gt3": float((a > 3).mean()),
        "feature_support_exceedance": float(exceed.max()),
        "latent_norm_max": float(zn.max()),
        "latent_norm_change": float(zn.max() - zn[0]),
        "step_delta_max": float(step.max()) if step.size else 0.0,
        "displacement": float(np.linalg.norm(latents[-1] - latents[0])),
        "state_delta_max": float(sstep.max()) if sstep.size else 0.0,
        "prob_max_abs_step": float(pstep.max()) if pstep.size else 0.0,
        "prob_range": float(probs.max() - probs.min()),
    }

    # Per-step detail the dashboard renders. Not part of the rule.
    out["_detail"] = {
        "latent_norm_per_step": [float(v) for v in zn],
        "latent_step_delta": [float(v) for v in step],
        "state_step_delta": [float(v) for v in sstep],
        "prob_step_delta": [float(v) for v in pstep],
        "probs": [float(v) for v in probs],
        "mean_abs_z": float(a.mean()),
        "n_values_gt2": int((a > 2).sum()),
        "n_values_gt3": int((a > 3).sum()),
        "n_values_outside_support": int((exceed > 0).sum()),
        "worst_feature": (
            ref.feature_names[int(np.unravel_index(exceed.argmax(), exceed.shape)[1])]
            if exceed.max() > 0 else None
        ),
        "worst_feature_window": (
            int(np.unravel_index(exceed.argmax(), exceed.shape)[0])
            if exceed.max() > 0 else None
        ),
        "worst_abs_z_feature": ref.feature_names[
            int(np.unravel_index(a.argmax(), a.shape)[1])],
    }
    return out


# =============================================================
# HARD FAIL-SAFE CONDITIONS
# =============================================================

@dataclass
class HardFail:
    code: str
    name: str
    detail: str


def check_scaler_contract(scaler, ref: Reference) -> list[HardFail]:
    """H3 — the frozen input scaler must be the one the reference assumes.

    A scaler swap silently rescales every z, which would make every
    other indicator in this module meaningless."""
    fails = []
    c = ref.scaler_contract
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in != c["n_features_in"]:
        fails.append(HardFail(
            "H3", "scaler_incompatibility",
            f"input scaler reports n_features_in_={n_in}, "
            f"reference expects {c['n_features_in']}"))
    for attr in ("mean_", "scale_"):
        v = getattr(scaler, attr, None)
        if v is None:
            fails.append(HardFail("H3", "scaler_incompatibility",
                                  f"input scaler has no {attr}"))
            continue
        if len(v) != c["n_features_in"]:
            fails.append(HardFail(
                "H3", "scaler_incompatibility",
                f"scaler.{attr} has length {len(v)}, expected "
                f"{c['n_features_in']}"))
            continue
        got = float(np.sum(np.asarray(v, dtype=np.float64)))
        exp = float(c[f"{attr.rstrip('_')}_sum"])
        if not math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-6):
            fails.append(HardFail(
                "H3", "scaler_incompatibility",
                f"scaler.{attr} sum {got:.6f} != frozen reference "
                f"{exp:.6f} — this is not the frozen V4 input scaler"))
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None and list(names) != c["feature_order"]:
        fails.append(HardFail(
            "H3", "scaler_incompatibility",
            "scaler feature_names_in_ does not match the frozen "
            "input_feature_names.txt order"))
    return fails


def check_hard_conditions(x, probs, latents, states, ind: dict,
                          ref: Reference) -> list[HardFail]:
    """H1..H8. Each is a numerical or contract condition, not a quantile.

    The order is fixed so the emitted reason list is deterministic."""
    fails: list[HardFail] = []
    xa = np.asarray(x, dtype=np.float64)
    pa = np.asarray(probs, dtype=np.float64).ravel()
    za = np.asarray(latents, dtype=np.float64)
    sa = np.asarray(states, dtype=np.float64)
    x2 = xa[0] if xa.ndim == 3 else xa

    # H1 — non-finite model input
    if not np.isfinite(xa).all():
        fails.append(HardFail(
            "H1", "non_finite_input",
            f"{int((~np.isfinite(xa)).sum())} of {xa.size} input values "
            f"are NaN or Inf"))

    # H2 — invalid tensor shape
    if x2.shape != (SEQ_LEN, N_FEATURES):
        fails.append(HardFail(
            "H2", "invalid_input_shape",
            f"model input shape {tuple(x2.shape)}, expected "
            f"({SEQ_LEN}, {N_FEATURES})"))
    if za.shape != (MAX_K + 1, LATENT_SIZE):
        fails.append(HardFail(
            "H2", "invalid_latent_shape",
            f"latent trajectory shape {tuple(za.shape)}, expected "
            f"({MAX_K + 1}, {LATENT_SIZE})"))
    if sa.shape != (MAX_K + 1, N_STATE_FEATURES):
        fails.append(HardFail(
            "H2", "invalid_state_shape",
            f"decoded state shape {tuple(sa.shape)}, expected "
            f"({MAX_K + 1}, {N_STATE_FEATURES})"))
    if pa.shape != (MAX_K + 1,):
        fails.append(HardFail(
            "H2", "invalid_probability_shape",
            f"probability trajectory shape {tuple(pa.shape)}, expected "
            f"({MAX_K + 1},)"))

    # H4 — non-finite latent or decoded state
    if not np.isfinite(za).all():
        fails.append(HardFail(
            "H4", "non_finite_latent",
            f"{int((~np.isfinite(za)).sum())} non-finite latent components"))
    if not np.isfinite(sa).all():
        fails.append(HardFail(
            "H4", "non_finite_decoded_state",
            f"{int((~np.isfinite(sa)).sum())} non-finite decoded state values"))

    # H5 — invalid probability
    if not np.isfinite(pa).all():
        fails.append(HardFail("H5", "non_finite_probability",
                              "threat probability is NaN or Inf"))
    elif pa.size and (pa.min() < 0.0 or pa.max() > 1.0):
        fails.append(HardFail(
            "H5", "probability_out_of_range",
            f"threat probability range [{pa.min():.6f}, {pa.max():.6f}] "
            f"leaves [0, 1]"))

    # H6 / H7 — analytic latent-norm bounds (numerical explosion)
    if np.isfinite(za).all() and za.ndim == 2 and za.shape[0] >= 1:
        zn = np.linalg.norm(za, axis=1)
        b0 = ref.bounds["encoder_output_norm_bound"]
        bk = ref.bounds["transition_output_norm_bound"]
        if zn[0] > b0:
            fails.append(HardFail(
                "H6", "encoder_latent_norm_exceeds_analytic_bound",
                f"||z_0|| = {zn[0]:.4f} exceeds the architectural bound "
                f"{b0:.4f} implied by the frozen encoder weights"))
        over = [(k, float(v)) for k, v in enumerate(zn) if k >= 1 and v > bk]
        if over:
            k, v = over[0]
            fails.append(HardFail(
                "H7", "rollout_latent_norm_exceeds_analytic_bound",
                f"||z_{k}|| = {v:.4f} exceeds the LayerNorm-implied bound "
                f"{bk:.4f}; the transition output cannot reach this value"))

    # H8 — extreme input shift past the reference per-feature support
    exc = ind["feature_support_exceedance"]
    if exc > INPUT_SHIFT_HARD_MARGIN_SD:
        det = ind.get("_detail", {})
        feat = det.get("worst_feature")
        win = det.get("worst_feature_window")
        where = f" (feature `{feat}`, window {win})" if feat is not None else ""
        fails.append(HardFail(
            "H8", "extreme_input_shift",
            f"an input value lies {exc:.2f} training SD OUTSIDE the range "
            f"ever observed for that feature in the reference "
            f"distribution{where}; margin = "
            f"{INPUT_SHIFT_HARD_MARGIN_SD:.1f} SD"))

    return fails


# =============================================================
# ASSESSMENT
# =============================================================

@dataclass
class Assessment:
    status: str
    short: str                      # HIGH / CAUTION / LOW
    headline: str                   # one explainable sentence
    reasons: list[str]              # every contributing reason, ordered
    hard_fails: list[dict]
    tiers: dict                     # indicator -> tier
    indicators: dict                # indicator -> value
    detail: dict                    # per-step arrays for display
    n_elevated: int
    n_beyond: int
    rule: dict
    layer_version: str = LAYER_VERSION

    def to_row(self) -> dict:
        """Flat dict for reliability_per_sequence.csv."""
        row = {
            "reliability_status": self.status,
            "reliability_short": self.short,
            "n_hard_fails": len(self.hard_fails),
            "hard_fail_codes": "|".join(sorted({h["code"] for h in self.hard_fails})),
            "hard_fail_names": "|".join(h["name"] for h in self.hard_fails),
            "n_elevated": self.n_elevated,
            "n_beyond_reference": self.n_beyond,
            "triggering_indicators": "|".join(
                n for n in TIERED_INDICATORS if self.tiers.get(n) != TIER_OK),
            "headline": self.headline,
        }
        for n in INDICATOR_NAMES:
            row[n] = self.indicators[n]
            if n in self.tiers:
                row[f"tier_{n}"] = self.tiers[n]
        return row


def _headline(status, hard_fails, elevated, beyond, ind) -> str:
    """One sentence, always traceable to the indicators that produced it.

    Never asserts anything about correctness, only about whether the
    forecast was produced under conditions the reference distribution
    covers."""
    if status == UNRELIABLE:
        if hard_fails:
            h = hard_fails[0]
            if h.code == "H8":
                return ("UNRELIABLE — input is far outside the training "
                        "scaler distribution: " + h.detail)
            if h.code in ("H6", "H7"):
                return ("UNRELIABLE — the latent trajectory breaches an "
                        "architectural numerical bound: " + h.detail)
            if h.code == "H3":
                return ("UNRELIABLE — scaler incompatibility, so every "
                        "standardized indicator is meaningless: " + h.detail)
            return f"UNRELIABLE — hard fail-safe {h.code} ({h.name}): {h.detail}"
        names = ", ".join(BY_NAME[n].clause for n in beyond)
        return (f"UNRELIABLE — {len(beyond)} independent indicators fall "
                f"outside anything in the reference distribution: {names}.")

    if status == CAUTION:
        if beyond:
            n = beyond[0]
            return (f"CAUTION — {BY_NAME[n].clause} "
                    f"({ind[n]:.4g}) exceeds every value seen in the "
                    f"reference distribution, but no other indicator and "
                    f"no hard condition fired.")
        groups = sorted({BY_NAME[n].group for n in elevated})
        names = ", ".join(BY_NAME[n].clause for n in elevated)
        if groups == ["input_ood"]:
            return ("CAUTION — the forecast trajectory is within reference "
                    f"bounds but the current input is unusual: {names}.")
        if "input_ood" not in groups:
            return ("CAUTION — the input is within reference bounds but the "
                    f"rollout itself is unusual: {names}.")
        return (f"CAUTION — {len(elevated)} indicators sit above the "
                f"reference 99th percentile: {names}.")

    return ("RELIABLE — every reliability indicator lies within the "
            "reference distribution and no fail-safe condition fired. "
            "This describes the operating conditions of the forecast, "
            "not its correctness.")


def assess(x, probs, latents, states, ref: Reference,
           scaler=None) -> Assessment:
    """Full reliability assessment for one forecast.

    Pass `scaler` to include the H3 scaler-contract check; callers that
    consume the pre-scaled frozen arrays can omit it, in which case the
    contract is verified once at config-build time instead.
    """
    ind = compute_indicators(x, probs, latents, states, ref)
    detail = ind.pop("_detail")

    hard = check_hard_conditions(x, probs, latents, states, ind, ref)
    if scaler is not None:
        hard = check_scaler_contract(scaler, ref) + hard

    tiers = {n: ref.tier(n, ind[n]) for n in TIERED_INDICATORS}
    elevated = [n for n in TIERED_INDICATORS if tiers[n] == TIER_ELEVATED]
    beyond = [n for n in TIERED_INDICATORS if tiers[n] == TIER_BEYOND]

    if hard or len(beyond) >= RULE_BEYOND_FOR_UNRELIABLE:
        status = UNRELIABLE
    elif len(beyond) >= RULE_BEYOND_FOR_CAUTION or \
            len(elevated) >= RULE_ELEVATED_FOR_CAUTION:
        status = CAUTION
    else:
        status = RELIABLE

    reasons = [f"[{h.code}] {h.name}: {h.detail}" for h in hard]
    reasons += [f"[{TIER_BEYOND}] {n} = {ind[n]:.6g} > reference max "
                f"{ref.quantiles[n]['max']:.6g}" for n in beyond]
    reasons += [f"[{TIER_ELEVATED}] {n} = {ind[n]:.6g} > reference p99 "
                f"{ref.quantiles[n]['p99']:.6g}" for n in elevated]
    if not reasons:
        reasons = ["all indicators within the reference distribution"]

    return Assessment(
        status=status,
        short=STATUS_SHORT[status],
        headline=_headline(status, hard, elevated, beyond, ind),
        reasons=reasons,
        hard_fails=[asdict(h) for h in hard],
        tiers=tiers,
        indicators=ind,
        detail=detail,
        n_elevated=len(elevated),
        n_beyond=len(beyond),
        rule={
            "beyond_for_unreliable": RULE_BEYOND_FOR_UNRELIABLE,
            "beyond_for_caution": RULE_BEYOND_FOR_CAUTION,
            "elevated_for_caution": RULE_ELEVATED_FOR_CAUTION,
            "elevated_quantile": ELEVATED_QUANTILE,
            "input_shift_hard_margin_sd": INPUT_SHIFT_HARD_MARGIN_SD,
        },
    )


# =============================================================
# ABSTENTION WORDING (display only — Task 16 owns the decision)
# =============================================================

ABSTENTION_HEADLINE = "MODEL ABSTENTION RECOMMENDED"
ABSTENTION_BODY = (
    "Forecast should not be treated as reliable for this window."
)

STATUS_NOTE = (
    "This is a rule-based status computed from the frozen model's own "
    "inputs and rollout against a fixed reference distribution. It is "
    "NOT a calibrated probability, NOT a confidence score and NOT a "
    "probability of correctness. No reliability estimator was fitted and "
    "no held-out test label was used to set any threshold."
)
