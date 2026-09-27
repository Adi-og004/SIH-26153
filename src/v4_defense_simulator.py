"""
=============================================================
V4 DEFEND-NOW SIMULATOR — model-based counterfactual projection
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Task 18 / S4. Answers ONE question:

    "What would the FROZEN V4 World Model forecast if a defender applied a
     defined intervention to the current network state right now?"

The answer is a MODEL-BASED COUNTERFACTUAL PROJECTION. It is produced by
perturbing the current observed state along explicitly-justified feature
semantics and rolling the frozen model forward. Nothing more.

WHAT THIS IS NOT
----------------
- NOT a causal estimate. V4 was trained to predict the next network state
  and a threat probability from observed telemetry. It was NEVER shown an
  intervention, never given an action variable, and never trained on
  (action, outcome) pairs. It therefore cannot have learned intervention
  effects, and this module does not claim it did.
- NOT a real firewall / network action. Nothing here touches a socket, a
  device, or a configuration. It is arithmetic on a feature vector.
- NOT a proven operational effect and NOT guaranteed mitigation.
- NOT an automated defensive command.

THREE DISTINCT QUANTITIES, NEVER CONFLATED
------------------------------------------
    OBSERVED                the real, frozen 10x52 input window
    NO-ACTION FORECAST      the frozen K=5 rollout of that window
    DEFEND-NOW COUNTERFACTUAL   the frozen K=5 rollout of the PERTURBED
                                window

The observed state is never overwritten: baseline_state and
counterfactual_state are separate arrays throughout.

HOW THE STATE IS MODIFIED — AND WHAT IS PRESERVED
-------------------------------------------------
Interventions act on the RAW physical base quantities, never directly on
scaled values or on relative-change features. The pipeline is:

  1. Inverse-transform the base half to raw units with the FROZEN scaler.
  2. Apply the intervention to the FINAL timestep only (see SCOPE).
  3. Recompute the relative-change row for the final timestep with the
     FROZEN contract formula.
  4. Reassemble 52-D in the frozen feature order and re-scale with the
     FROZEN scaler.
  5. Roll forward with the FROZEN checkpoint.

SCOPE — why only the final timestep
-----------------------------------
The defender acts NOW. Timesteps t-90s..t-10s are OBSERVED HISTORY; a
"defend now" action cannot have changed them, and rewriting them would
fabricate a past that did not happen. Only the current window (the final
timestep) is perturbed. This is a deliberate, documented restriction, and
it is why the projected changes are modest: the V4 encoder is an LSTM over
ten windows, so one perturbed window has limited leverage on the encoding.
That is a property of the model being probed, reported rather than tuned
away.

PRESERVED EXACTLY
-----------------
Relative-change row 0 of every window CANNOT be recomputed from inside the
window: the sequences are sliding windows, so row 0 was computed against
the window BEFORE the sequence begins. It is therefore PRESERVED from the
observed tensor, never regenerated. All base rows 0..8 and all
relative-change rows 0..8 are likewise copied verbatim, so the only cells
this module writes are the final base row and the final relative-change
row.

VERIFIED FEATURE IDENTITIES
---------------------------
Measured on the frozen substrate, and relied upon by the transformations:

    log_flow_count      == log1p(flow_count)                 (max|d| 1.2e-07)
    window_packet_rate  == expm1(log_total_packets) / 10      (max rel 2.4e-07)
    window_byte_rate    == expm1(log_total_bytes) / 10        (max rel 7.2e-07)
    fwd_packet_ratio + bwd_packet_ratio == 1.0        (to 1.9e-08, float32)
    dst_port_entropy    <= log2(unique_dst_ports)             (0 violations)

Every intervention below PRESERVES all five. An intervention that reduced
window_packet_rate without also reducing log_total_packets would produce
internally impossible telemetry, so coupled propagation is mandatory, not
optional.

NOTE: fwd_byte_ratio + bwd_byte_ratio is NOT 1 (measured mean 0.663), so no
sum constraint is imposed on the byte ratios, and they are left unchanged.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

import feature_contract as fc

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
INPUT_SCALER = BENCH_DIR / "input_scaler.pkl"

LAYER_VERSION = "v4-defense-sim-1.0"

SEQ_LEN = 10
N_FEATURES = 52
N_BASE = 26
MAX_K = 5
WINDOW_SECONDS = 10

# The frozen contract constants, asserted against feature_contract at load.
RELATIVE_EPSILON = 1e-8
RELATIVE_CLIP = 10.0

BASE_INDEX = {name: i for i, name in enumerate(fc.BASE_FEATURES)}

# Honest wording contract. The UI and every artifact use these.
WORDING_REQUIRED = (
    "MODEL-BASED WHAT-IF PROJECTION",
    "This simulator does not estimate causal real-world effectiveness. "
    "It applies a defined state perturbation and rolls the frozen World "
    "Model forward.",
)
WORDING_FORBIDDEN = (
    "Defense will reduce the attack by X%",
    "Attack prevented",
    "Firewall blocked attacker",
    "Guaranteed mitigation",
    "risk reduction caused by defense",
)
DELTA_NAME = "model-projected change under intervention"


class DefenseSimulatorError(RuntimeError):
    """Raised when a frozen precondition for this simulator does not hold."""


# =============================================================
# FROZEN SCALER ACCESS  (forward / inverse only — never refit)
# =============================================================

@dataclass
class FrozenScaler:
    mean: np.ndarray          # (52,)
    scale: np.ndarray         # (52,)
    path: str

    @property
    def mean_base(self):
        return self.mean[:N_BASE]

    @property
    def scale_base(self):
        return self.scale[:N_BASE]

    @property
    def mean_rel(self):
        return self.mean[N_BASE:]

    @property
    def scale_rel(self):
        return self.scale[N_BASE:]


def load_scaler(path: Path = INPUT_SCALER) -> FrozenScaler:
    if not path.exists():
        raise DefenseSimulatorError(f"frozen input scaler not found: {path}")
    with open(path, "rb") as fh:
        s = pickle.load(fh)
    if int(s.n_features_in_) != N_FEATURES:
        raise DefenseSimulatorError(
            f"input_scaler has {s.n_features_in_} features, expected {N_FEATURES}")
    return FrozenScaler(mean=np.asarray(s.mean_, dtype=np.float64),
                        scale=np.asarray(s.scale_, dtype=np.float64),
                        path=str(path.relative_to(REPO_ROOT)))


def verify_frozen_formula() -> dict:
    """Assert this module's relative-change constants ARE the contract's."""
    if (float(fc.RELATIVE_EPSILON) != RELATIVE_EPSILON
            or float(fc.RELATIVE_CLIP) != RELATIVE_CLIP):
        raise DefenseSimulatorError(
            f"relative-change constants diverged from the frozen contract: "
            f"eps {fc.RELATIVE_EPSILON} vs {RELATIVE_EPSILON}, "
            f"clip {fc.RELATIVE_CLIP} vs {RELATIVE_CLIP}")
    return {
        "source_of_truth": "src/feature_contract.py",
        "epsilon": RELATIVE_EPSILON,
        "clip": RELATIVE_CLIP,
        "formula": "(x_t - x_{t-1}) / (|x_{t-1}| + 1e-8) clipped to [-10, 10]",
        "new_transform_introduced": False,
    }


def _relative_change(prev_raw: np.ndarray, cur_raw: np.ndarray) -> np.ndarray:
    """The frozen contract formula, one step, in RAW physical units."""
    rel = (cur_raw - prev_raw) / (np.abs(prev_raw) + RELATIVE_EPSILON)
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(rel, -RELATIVE_CLIP, RELATIVE_CLIP)


# =============================================================
# INTERVENTION DEFINITIONS
# =============================================================
#
# Each intervention is a DETERMINISTIC PURE FUNCTION of (raw base row,
# strength) -> new raw base row. It never sees a label, a future window, or
# a threat probability. Strength is always "simulated intervention
# strength", never "real-world effectiveness".

# Features every intervention leaves untouched, with the reason. Listing
# them is part of the deliverable: an unmodified feature is a decision.
UNCHANGED_COMMON = {
    "syn_rate": "per-flow flag presence / packets, averaged over flows. "
                "Dropping or throttling whole flows leaves the per-flow "
                "statistic's distribution unchanged in expectation; no "
                "signed change is derivable from the feature definition.",
    "ack_rate": "as syn_rate",
    "rst_rate": "as syn_rate. (A real block would likely RAISE rst_rate, but "
                "neither the magnitude nor the flow-accounting consequence is "
                "derivable from the contract, so it is not invented.)",
    "fin_rate": "as syn_rate",
    "psh_rate": "as syn_rate",
    "urg_rate": "as syn_rate",
    "mean_pkt_len": "mean over flows of each flow's mean packet length. A "
                    "uniform volume reduction does not change the mean-of-"
                    "per-flow-means in expectation.",
    "std_pkt_len": "mean over flows of each flow's packet-length std; same "
                   "argument as mean_pkt_len.",
    "log_mean_flow_duration": "a rate limit or block plausibly changes flow "
                              "lifetimes, but neither the sign nor the "
                              "magnitude follows from the feature definition. "
                              "Left unchanged rather than guessed.",
    "log_mean_fwd_iat": "as log_mean_flow_duration (a rate limit would "
                        "plausibly INCREASE inter-arrival time, but by an "
                        "amount the contract does not determine).",
    "log_mean_bwd_iat": "as log_mean_fwd_iat",
    "fwd_packet_ratio": "a ratio, invariant to uniform volume scaling. Also "
                        "constrained by the measured identity "
                        "fwd_packet_ratio + bwd_packet_ratio == 1.0 (to "
                        "float32 precision), which leaving both unchanged "
                        "preserves.",
    "bwd_packet_ratio": "as fwd_packet_ratio",
    "fwd_byte_ratio": "a ratio, invariant to uniform volume scaling. NOTE the "
                      "byte ratios do NOT sum to 1 (measured mean 0.663), so "
                      "no coupled redistribution can be defined for them and "
                      "none is attempted.",
    "bwd_byte_ratio": "as fwd_byte_ratio",
    "mean_down_up_ratio": "an integer-valued per-flow ratio, averaged. Not "
                          "derivably changed by uniform volume scaling.",
    "unique_protocols": "a count of distinct protocols. Throttling volume "
                        "does not remove a protocol from the window.",
    "protocol_entropy": "Shannon entropy over protocol counts; unchanged for "
                        "the same reason as unique_protocols.",
}


def _scale_volume(row: np.ndarray, keep: float) -> None:
    """Scale the three underlying volume quantities by `keep`, IN PLACE,
    propagating to every feature derived from them.

    This is where the verified identities are honoured:
        flow_count          -> log_flow_count
        total_packets       -> log_total_packets, window_packet_rate
        total_bytes         -> log_total_bytes,  window_byte_rate
    Modifying a log feature without its rate twin (or vice versa) would
    emit internally impossible telemetry.
    """
    i = BASE_INDEX

    fcount = row[i["flow_count"]] * keep
    row[i["flow_count"]] = fcount
    row[i["log_flow_count"]] = np.log1p(max(fcount, 0.0))

    tot_pkts = np.expm1(row[i["log_total_packets"]]) * keep
    tot_pkts = max(tot_pkts, 0.0)
    row[i["log_total_packets"]] = np.log1p(tot_pkts)
    row[i["window_packet_rate"]] = tot_pkts / WINDOW_SECONDS

    tot_bytes = np.expm1(row[i["log_total_bytes"]]) * keep
    tot_bytes = max(tot_bytes, 0.0)
    row[i["log_total_bytes"]] = np.log1p(tot_bytes)
    row[i["window_byte_rate"]] = tot_bytes / WINDOW_SECONDS


def intervene_rate_limit(row: np.ndarray, strength: float) -> np.ndarray:
    """DEFEND_NOW — uniform traffic rate limit at the network edge.

    A broadly applicable immediate defensive action: throttle the volume of
    traffic traversing the monitored boundary by `strength`, without
    selecting particular destinations or ports.

    Modified (all coupled, see _scale_volume):
        flow_count, log_flow_count,
        log_total_packets, window_packet_rate,
        log_total_bytes,   window_byte_rate
    Everything else is unchanged — see UNCHANGED_COMMON, plus the port /
    destination-diversity features, because throttling volume does not
    remove a destination from the window.
    """
    out = row.copy()
    _scale_volume(out, 1.0 - float(strength))
    return out


def intervene_restrict_egress(row: np.ndarray, strength: float) -> np.ndarray:
    """BLOCK_C2_OR_RESTRICT_EGRESS — egress allow-listing.

    A targeted intervention: block a fraction `strength` of the distinct
    outbound destination ports observed in the current window, as an egress
    allow-list would. Two consequences ARE derivable from the contract:

      (a) DESTINATION DIVERSITY falls.
          unique_dst_ports    -> unique_dst_ports * (1 - strength)
          log_unique_dst_ports = log1p(that)

      (b) PORT ENTROPY falls, because Shannon entropy is bounded above by
          log2(support size). The entropy is rescaled by the ratio of the
          new and old CEILINGS:
              dst_port_entropy *= log2(max(n_new, 2)) / log2(max(n_old, 2))
          This shrinks entropy in proportion to the shrinking ceiling and
          PRESERVES the measured invariant entropy <= log2(nunique). It is a
          bound-preserving rescaling, NOT a claim about the true post-block
          port distribution, which the contract cannot determine.

      (c) VOLUME falls, first-order, because the blocked destinations carry
          a share of the window's flows. The simplest defensible assumption
          is a proportional share, so the same `strength` is applied to
          flow_count / total_packets / total_bytes with full coupled
          propagation.

    Direction ratios are deliberately NOT shifted: "forward" is defined as
    the direction of a flow's FIRST packet, so blocking egress removes whole
    outbound-initiated flows rather than re-balancing within surviving
    flows. The per-flow ratio distribution is therefore not derivably
    changed, and the exact fwd+bwd == 1 identity is preserved by leaving
    both alone.
    """
    out = row.copy()
    s = float(strength)
    i = BASE_INDEX

    n_old = np.expm1(out[i["log_unique_dst_ports"]])
    n_new = max(n_old * (1.0 - s), 0.0)
    out[i["log_unique_dst_ports"]] = np.log1p(n_new)

    ceil_old = np.log2(max(n_old, 2.0))
    ceil_new = np.log2(max(n_new, 2.0))
    ratio = min(ceil_new / ceil_old, 1.0) if ceil_old > 0 else 1.0
    out[i["dst_port_entropy"]] = out[i["dst_port_entropy"]] * ratio

    _scale_volume(out, 1.0 - s)
    return out


@dataclass(frozen=True)
class Intervention:
    key: str
    display: str
    concept: str
    parameter: str
    affected_features: tuple
    transformation: tuple
    justification: str
    unchanged: dict
    fn: object

    def apply(self, row: np.ndarray, strength: float) -> np.ndarray:
        if not (0.0 <= float(strength) <= 1.0):
            raise DefenseSimulatorError(
                f"simulated intervention strength must be in [0, 1], "
                f"got {strength}")
        return self.fn(row, strength)

    def to_dict(self):
        d = {k: v for k, v in asdict(self).items() if k != "fn"}
        d["affected_features"] = list(self.affected_features)
        d["transformation"] = list(self.transformation)
        return d


INTERVENTIONS: dict[str, Intervention] = {
    "RATE_LIMIT": Intervention(
        key="RATE_LIMIT",
        display="Defend now — uniform traffic rate limit",
        concept="Throttle the volume of traffic crossing the monitored "
                "boundary by the simulated strength, without selecting "
                "particular destinations or ports.",
        parameter="simulated intervention strength (fraction of volume "
                  "withheld), in [0, 1]",
        affected_features=(
            "flow_count", "log_flow_count",
            "log_total_packets", "window_packet_rate",
            "log_total_bytes", "window_byte_rate",
        ),
        transformation=(
            "keep = 1 - strength",
            "flow_count      <- flow_count * keep",
            "log_flow_count  <- log1p(flow_count_new)",
            "total_packets   <- expm1(log_total_packets) * keep",
            "log_total_packets   <- log1p(total_packets_new)",
            "window_packet_rate  <- total_packets_new / 10",
            "total_bytes     <- expm1(log_total_bytes) * keep",
            "log_total_bytes     <- log1p(total_bytes_new)",
            "window_byte_rate    <- total_bytes_new / 10",
        ),
        justification=(
            "flow_count, total_packets and total_bytes are the three "
            "underlying volume quantities in the frozen contract, and the "
            "log / rate features are exact derivations of them (identities "
            "verified to 1e-7 on the frozen substrate). A rate limit reduces "
            "the underlying quantities, so every derived feature must move "
            "with them. Nothing outside that derivation chain is touched."),
        unchanged={**UNCHANGED_COMMON,
                   "log_unique_dst_ports": "throttling volume does not remove "
                                           "a destination port from the window",
                   "dst_port_entropy": "as log_unique_dst_ports"},
        fn=intervene_rate_limit,
    ),
    "RESTRICT_EGRESS": Intervention(
        key="RESTRICT_EGRESS",
        display="Block C2 / restrict egress — destination allow-listing",
        concept="Block a fraction of the distinct outbound destination ports "
                "observed in the current window, as an egress allow-list "
                "would.",
        parameter="simulated intervention strength (fraction of distinct "
                  "destination ports blocked), in [0, 1]",
        affected_features=(
            "log_unique_dst_ports", "dst_port_entropy",
            "flow_count", "log_flow_count",
            "log_total_packets", "window_packet_rate",
            "log_total_bytes", "window_byte_rate",
        ),
        transformation=(
            "keep = 1 - strength",
            "n_old = expm1(log_unique_dst_ports)",
            "n_new = n_old * keep",
            "log_unique_dst_ports <- log1p(n_new)",
            "dst_port_entropy <- dst_port_entropy * "
            "min(log2(max(n_new,2)) / log2(max(n_old,2)), 1)",
            "then the same coupled volume reduction as RATE_LIMIT",
        ),
        justification=(
            "Two consequences follow from the contract's own definitions. "
            "log_unique_dst_ports is log1p(nunique of Dst Port), so blocking "
            "a fraction of destinations reduces the count directly. "
            "dst_port_entropy is Shannon entropy over the destination-port "
            "counts, which is bounded above by log2 of the support size; "
            "rescaling by the ratio of ceilings shrinks entropy with its "
            "bound and preserves the measured invariant "
            "entropy <= log2(nunique) (0 violations on the frozen "
            "substrate). Volume falls too, on the first-order assumption "
            "that blocked destinations carry a proportional share of the "
            "window's flows. Direction ratios are NOT shifted because "
            "'forward' is the direction of a flow's first packet, so egress "
            "blocking removes whole flows rather than re-balancing surviving "
            "ones."),
        unchanged={**UNCHANGED_COMMON},
        fn=intervene_restrict_egress,
    ),
}

DEFAULT_INTERVENTION = "RATE_LIMIT"
SWEEP_STRENGTHS = (0.10, 0.25, 0.50)


# =============================================================
# THE COUNTERFACTUAL STATE BUILDER
# =============================================================

@dataclass
class CounterfactualState:
    """Both states, kept strictly separate. The observed tensor is never
    written to."""
    baseline_state: np.ndarray        # (10, 52) scaled — the OBSERVED window
    counterfactual_state: np.ndarray  # (10, 52) scaled — the PERTURBED window
    intervention: str
    strength: float
    baseline_raw_last: np.ndarray     # (26,) raw, observed final window
    counterfactual_raw_last: np.ndarray
    changed_feature_indices: tuple
    cells_written: int

    def raw_feature_table(self) -> list[dict]:
        """Per-feature before/after in RAW physical units, for display."""
        rows = []
        for idx, name in enumerate(fc.BASE_FEATURES):
            b = float(self.baseline_raw_last[idx])
            c = float(self.counterfactual_raw_last[idx])
            rows.append({
                "feature": name,
                "observed_raw": b,
                "counterfactual_raw": c,
                "absolute_change": c - b,
                "relative_change_pct": (100.0 * (c - b) / b) if b != 0 else None,
                "modified": bool(idx in self.changed_feature_indices),
            })
        return rows


def build_counterfactual(x_window: np.ndarray,
                         intervention: str,
                         strength: float,
                         scaler: FrozenScaler) -> CounterfactualState:
    """Perturb the CURRENT state and reassemble a valid frozen 52-D window.

    x_window  (10, 52) or (1, 10, 52) — the frozen SCALED observed window.
              Treated as read-only.

    ONLY TWO ROWS ARE WRITTEN: the final base row and the final
    relative-change row. Everything else is copied verbatim from the
    observed tensor, which matters for two reasons:

      * relative-change row 0 references the window BEFORE this sequence
        begins (these are sliding windows) and is therefore not
        recomputable from inside the window. Copying it preserves it
        exactly.
      * copying the untouched rows means a zero-strength intervention
        perturbs nothing outside the final row, so the zero-strength
        equivalence check measures only the round-trip precision of the
        cells the simulator actually writes.
    """
    if intervention not in INTERVENTIONS:
        raise DefenseSimulatorError(
            f"unknown intervention {intervention!r}; "
            f"available: {sorted(INTERVENTIONS)}")

    x = np.asarray(x_window, dtype=np.float64)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise DefenseSimulatorError(f"expected one window, got {x.shape}")
        x = x[0]
    if x.shape != (SEQ_LEN, N_FEATURES):
        raise DefenseSimulatorError(
            f"x_window has shape {x.shape}, expected ({SEQ_LEN}, {N_FEATURES})")

    baseline = np.array(x, dtype=np.float64, copy=True)

    # --- raw units for the two rows the relative-change formula needs ---
    raw_prev = baseline[-2, :N_BASE] * scaler.scale_base + scaler.mean_base
    raw_last = baseline[-1, :N_BASE] * scaler.scale_base + scaler.mean_base

    iv = INTERVENTIONS[intervention]
    raw_last_cf = iv.apply(raw_last, strength)

    if not np.isfinite(raw_last_cf).all():
        raise DefenseSimulatorError(
            f"intervention {intervention} produced non-finite raw features")

    # --- reassemble: copy everything, overwrite only the final rows ---
    cf = np.array(baseline, dtype=np.float64, copy=True)
    cf[-1, :N_BASE] = (raw_last_cf - scaler.mean_base) / scaler.scale_base
    rel_last = _relative_change(raw_prev, raw_last_cf)
    cf[-1, N_BASE:] = (rel_last - scaler.mean_rel) / scaler.scale_rel

    if not np.isfinite(cf).all():
        raise DefenseSimulatorError("counterfactual state is non-finite")

    changed = tuple(int(i) for i in np.flatnonzero(
        ~np.isclose(raw_last_cf, raw_last, rtol=0.0, atol=0.0)))

    return CounterfactualState(
        baseline_state=baseline,
        counterfactual_state=cf,
        intervention=intervention,
        strength=float(strength),
        baseline_raw_last=raw_last,
        counterfactual_raw_last=raw_last_cf,
        changed_feature_indices=changed,
        cells_written=2 * N_BASE,
    )


# =============================================================
# SCENARIO RESULT
# =============================================================

@dataclass
class Scenario:
    """One no-action / defend-now comparison for one sequence."""
    sequence_id: int
    timestamp: str
    intervention: str
    strength: float
    no_action_probs: np.ndarray        # (6,) k=0..5
    defend_now_probs: np.ndarray       # (6,)
    alert_threshold: float
    no_action_reliability: str | None = None
    defend_now_reliability: str | None = None
    no_action_attack_path: str | None = None
    defend_now_attack_path: str | None = None
    no_action_next_stage: str | None = None
    defend_now_next_stage: str | None = None
    no_action_ttp_s: int | None = None
    defend_now_ttp_s: int | None = None
    changed_features: list = field(default_factory=list)

    # ---- projected change (NEVER called "risk reduction") ----

    @property
    def deltas(self) -> np.ndarray:
        """model-projected change under intervention, per horizon k=0..5."""
        return self.defend_now_probs - self.no_action_probs

    @property
    def relative_deltas(self) -> list:
        """Relative change, only where mathematically meaningful.

        None where the no-action probability is too small for a ratio to
        carry meaning — a 0.0001 -> 0.0002 change is not "+100%" in any
        useful sense."""
        out = []
        for b, d in zip(self.no_action_probs, self.deltas):
            out.append(float(d / b) if b >= 1e-3 else None)
        return out

    @property
    def no_action_max_future(self) -> float:
        return float(self.no_action_probs[1:].max())

    @property
    def defend_now_max_future(self) -> float:
        return float(self.defend_now_probs[1:].max())

    @property
    def max_future_delta(self) -> float:
        return self.defend_now_max_future - self.no_action_max_future

    @property
    def mean_delta_k5(self) -> float:
        return float(self.deltas[1:].mean())

    @property
    def direction(self) -> str:
        """Descriptive only. 'lower'/'higher' refer to the MODEL'S PROJECTED
        probability, not to real-world risk."""
        d = self.max_future_delta
        if abs(d) < 1e-6:
            return "unchanged"
        return "lower" if d < 0 else "higher"

    @property
    def no_action_alert(self) -> int:
        return int(self.no_action_max_future > self.alert_threshold)

    @property
    def defend_now_alert(self) -> int:
        return int(self.defend_now_max_future > self.alert_threshold)

    @property
    def crosses_threshold(self) -> bool:
        return self.no_action_alert != self.defend_now_alert

    def to_row(self) -> dict:
        r = {
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "intervention": self.intervention,
            "simulated_strength": self.strength,
            "alert_threshold": self.alert_threshold,
            "no_action_nowcast": float(self.no_action_probs[0]),
            "defend_now_nowcast": float(self.defend_now_probs[0]),
            "no_action_max_future": self.no_action_max_future,
            "defend_now_max_future": self.defend_now_max_future,
            "max_future_projected_change": self.max_future_delta,
            "mean_projected_change_k5": self.mean_delta_k5,
            "projected_direction": self.direction,
            "no_action_alert": self.no_action_alert,
            "defend_now_alert": self.defend_now_alert,
            "crosses_frozen_threshold": int(self.crosses_threshold),
            "no_action_reliability": self.no_action_reliability or "",
            "defend_now_reliability": self.defend_now_reliability or "",
            "reliability_changed": int(
                (self.no_action_reliability or "")
                != (self.defend_now_reliability or "")),
            "no_action_attack_path": self.no_action_attack_path or "",
            "defend_now_attack_path": self.defend_now_attack_path or "",
            "no_action_next_stage": self.no_action_next_stage or "",
            "defend_now_next_stage": self.defend_now_next_stage or "",
            "no_action_ttp_s": ("" if self.no_action_ttp_s is None
                                else self.no_action_ttp_s),
            "defend_now_ttp_s": ("" if self.defend_now_ttp_s is None
                                 else self.defend_now_ttp_s),
            "changed_features": "|".join(self.changed_features),
        }
        for k in range(MAX_K + 1):
            lbl = "t0" if k == 0 else f"t{k}"
            r[f"no_action_{lbl}"] = float(self.no_action_probs[k])
            r[f"defend_now_{lbl}"] = float(self.defend_now_probs[k])
            r[f"projected_change_{lbl}"] = float(self.deltas[k])
        return r


# =============================================================
# THE SIMULATOR
# =============================================================

class DefenseSimulator:
    """Frozen model + frozen scaler + a defined perturbation.

    Nothing is trained, fitted, or written. The model is used in eval mode
    under torch.no_grad() exactly as every other layer in this project uses
    it.
    """

    def __init__(self, model, alert_threshold: float,
                 scaler: FrozenScaler | None = None):
        self.model = model
        self.model.eval()
        self.alert_threshold = float(alert_threshold)
        self.scaler = scaler or load_scaler()
        verify_frozen_formula()

    def rollout(self, x_window: np.ndarray) -> tuple:
        """The frozen K=5 rollout. Returns (probs[6], latents, states)."""
        import torch
        x = np.asarray(x_window, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
        with torch.no_grad():
            tz, tp, ts = self.model.k_step_rollout(
                torch.tensor(x, dtype=torch.float32), MAX_K)
        probs = np.array([float(v[0]) for v in tp], dtype=np.float64)
        latents = np.stack([v[0].cpu().numpy() for v in tz])
        states = np.stack([v[0].cpu().numpy() for v in ts])
        return probs, latents, states

    def simulate(self, sequence_id: int, x_window: np.ndarray,
                 intervention: str = DEFAULT_INTERVENTION,
                 strength: float = 0.25,
                 timestamp: str = "",
                 no_action_probs: np.ndarray | None = None) -> tuple:
        """Run the no-action and defend-now scenarios from the SAME window.

        `no_action_probs` may be supplied to use the FROZEN stored
        probabilities for the baseline instead of recomputing them; the
        counterfactual is always computed fresh. Both scenarios start from
        exactly the same observed sequence — the ONLY difference is the
        intervention.

        Returns (Scenario, CounterfactualState, detail dict).
        """
        cfs = build_counterfactual(x_window, intervention, strength,
                                   self.scaler)

        p_base, z_base, s_base = self.rollout(cfs.baseline_state)
        p_cf, z_cf, s_cf = self.rollout(cfs.counterfactual_state)

        if no_action_probs is not None:
            p_base_used = np.asarray(no_action_probs, dtype=np.float64).ravel()
        else:
            p_base_used = p_base

        changed = [fc.BASE_FEATURES[i] for i in cfs.changed_feature_indices]

        sc = Scenario(
            sequence_id=int(sequence_id),
            timestamp=str(timestamp),
            intervention=intervention,
            strength=float(strength),
            no_action_probs=p_base_used,
            defend_now_probs=p_cf,
            alert_threshold=self.alert_threshold,
            changed_features=changed,
        )
        detail = {
            "baseline_recomputed_probs": p_base,
            "baseline_latents": z_base, "baseline_states": s_base,
            "counterfactual_latents": z_cf, "counterfactual_states": s_cf,
        }
        return sc, cfs, detail


# =============================================================
# METADATA
# =============================================================

def intervention_definitions() -> dict:
    return {
        "layer_version": LAYER_VERSION,
        "module": "src/v4_defense_simulator.py",
        "n_interventions": len(INTERVENTIONS),
        "default_intervention": DEFAULT_INTERVENTION,
        "sweep_strengths": list(SWEEP_STRENGTHS),
        "parameter_semantics": (
            "every strength is a SIMULATED INTERVENTION STRENGTH — a "
            "perturbation magnitude applied to the model's input state. It "
            "is NOT a real-world effectiveness, efficacy or success rate."),
        "scope": {
            "timesteps_modified": "the FINAL timestep only (the current state)",
            "why": ("the defender acts NOW; earlier timesteps are OBSERVED "
                    "HISTORY and rewriting them would fabricate a past that "
                    "did not happen"),
            "cells_written_per_sequence": 2 * N_BASE,
            "rows_preserved_verbatim": (
                "base rows 0..8 and relative-change rows 0..8, copied from "
                "the observed tensor"),
            "relative_change_row_0": (
                "PRESERVED, never regenerated — these are sliding windows, so "
                "row 0 was computed against the window before the sequence "
                "begins and is not recomputable from inside it"),
            "consequence": (
                "projected changes are modest by construction: the V4 encoder "
                "is an LSTM over ten windows, so one perturbed window has "
                "limited leverage. This is a property of the model being "
                "probed, reported rather than tuned away."),
        },
        "verified_identities": {
            "log_flow_count == log1p(flow_count)": "max|d| 1.2e-07",
            "window_packet_rate == expm1(log_total_packets)/10":
                "max relative 2.4e-07",
            "window_byte_rate == expm1(log_total_bytes)/10":
                "max relative 7.2e-07",
            "fwd_packet_ratio + bwd_packet_ratio == 1.0":
                "holds to 1.9e-08 — float32 storage precision on a "
                "quantity of order 1.0",
            "dst_port_entropy <= log2(unique_dst_ports)": "0 violations",
            "fwd_byte_ratio + bwd_byte_ratio": (
                "NOT 1 (measured mean 0.663) — so no sum constraint is "
                "imposed and the byte ratios are left unchanged"),
        },
        "frozen_formula": verify_frozen_formula(),
        "wording": {
            "required": list(WORDING_REQUIRED),
            "forbidden": list(WORDING_FORBIDDEN),
            "delta_is_called": DELTA_NAME,
            "delta_is_never_called": "risk reduction caused by defense",
        },
        "guarantees": {
            "retrains_v4": False,
            "modifies_checkpoint": False,
            "refits_scalers": False,
            "modifies_thresholds": False,
            "modifies_screening_predictions": False,
            "changes_feature_definitions": False,
            "executes_network_actions": False,
            "claims_causal_efficacy": False,
            "claims_model_learned_intervention_effects": False,
            "uses_labels_to_design_interventions": False,
            "observed_state_overwritten": False,
            "deterministic": True,
        },
        "interventions": {k: v.to_dict() for k, v in INTERVENTIONS.items()},
    }


if __name__ == "__main__":
    print(json.dumps(intervention_definitions(), indent=2, default=str))
