"""
=============================================================
ATT&CK-CONSISTENT PROGRESSION HYPOTHESIS LAYER  (Task 16 / S2)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

A forward-looking DECISION-SUPPORT layer on top of the FROZEN V4 world
model. It answers one question:

    "Given the current observed state and the V4 future rollout, what
     ATT&CK-consistent progression is currently suggested, and at what
     future horizon might the next stage appear?"

WHAT THIS IS
------------
An "ATT&CK-consistent progression hypothesis". Two frozen components are
composed, and nothing new is learned:

    observed 10x52 window                      (frozen screening_X.npy)
        -> V4 encoder/transition/decoder       (frozen checkpoint)
        -> 26-D decoded state at k = 0..5
        -> rule-space reconstruction           (frozen scalers + frozen
                                                relative-change formula)
        -> src/mitre_mapper.py, UNMODIFIED     (deterministic rule library)
        -> ordered progression hypothesis

WHAT THIS IS NOT
----------------
- NOT "learned attack path prediction". No ATT&CK classifier exists, was
  trained, or is loaded. Every tactic/technique string comes verbatim out
  of the existing rule library.
- NOT a claim of attacker intent or causality. The rules read telemetry;
  telemetry is not intent.
- NOT a universally correct kill chain. The progression graph in this
  module is a small, project-specific analytical ordering over the seven
  tactics the existing mapper can emit (section PROGRESSION GRAPH). Real
  intrusions are not required to follow it.
- NOT a calibrated probability of stage correctness. Progression
  confidence here is a three-level DETERMINISTIC label with published
  criteria, not a score.

FROZEN-ASSET DISCIPLINE
-----------------------
This module never trains, never fine-tunes, never refits or re-fits a
scaler, never re-derives a threshold, never writes to the V4 checkpoint,
never touches screening_X.npy / screening_predictions.csv, and never
reads a label column. The two scalers are used ONLY in their frozen
forward/inverse form. src/mitre_mapper.py is imported and called, never
edited.

LABEL BOUNDARY
--------------
event_timeline.csv, screening_metadata.csv label columns, and the
documented March-2 episode list are used for RETROSPECTIVE DESCRIPTION
ONLY, in a code path that runs strictly after every hypothesis has been
produced (see build_attack_path.py). No threshold, edge, or confidence
criterion in this file was chosen by looking at them.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

import mitre_mapper as mm

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
INPUT_SCALER = BENCH_DIR / "input_scaler.pkl"
TARGET_SCALER = BENCH_DIR / "target_scaler.pkl"

LAYER_VERSION = "attack-path-1.0"

SEQ_LEN = 10
N_FEATURES = 52
N_BASE = 26
MAX_K = 5
WINDOW_SECONDS = 10

# The frozen relative-change constants. Copied from NOWHERE: they are
# read off the single canonical definition in src/feature_contract.py ->
# reference_relative_change (RELATIVE_EPSILON, RELATIVE_CLIP) and are
# asserted equal to it by verify_frozen_formula() below, so a drift in
# the contract fails loudly instead of silently changing rule inputs.
RELATIVE_EPSILON = 1e-8
RELATIVE_CLIP = 10.0


# =============================================================
# 1. MACHINE-READABLE DESCRIPTION OF THE EXISTING RULE LIBRARY
# =============================================================
# Produced by INSPECTING src/mitre_mapper.py at runtime. Nothing here is
# hand-transcribed, so the description cannot drift from the mapper.

# Operators the mapper's TelemetryRule.evaluate actually implements.
SUPPORTED_OPERATORS = (">", "<", ">=", "<=", "spike")


def describe_rule_library() -> dict:
    """Machine-readable description of the EXISTING, UNMODIFIED mapper.

    Everything is derived by introspecting mitre_mapper at import time.
    """
    rules = mm.build_rule_library()

    described = []
    for r in rules:
        checks = []
        for c in r.checks:
            feat, op = c["feature"], c["operator"]
            # "spike" reads <feature>_rel_change from the rel-change half
            # of the 52-D vector; every other operator reads the base half.
            reads = f"{feat}_rel_change" if op == "spike" else feat
            checks.append({
                "feature": feat,
                "operator": op,
                "threshold": float(c["threshold"]),
                "reads_contract_feature": reads,
                "contract_index": (mm.INPUT_FEATURES.index(reads)
                                   if reads in mm.INPUT_FEATURES else None),
                "half": "rel_change" if op == "spike" else "base",
            })
        described.append({
            "rule_id": r.rule_id,
            "tactic_id": r.tactic_id,
            "tactic_name": r.tactic_name,
            "technique_id": r.technique_id,
            "technique_name": r.technique_name,
            "description": r.description,
            "min_confidence": r.min_confidence,
            "n_checks": len(r.checks),
            "checks": checks,
        })

    features_used = sorted({c["reads_contract_feature"]
                            for r in described for c in r["checks"]})
    ops_used = sorted({c["operator"] for r in described for c in r["checks"]})

    tactics = {}
    for r in described:
        tactics.setdefault(r["tactic_id"], {
            "tactic_id": r["tactic_id"],
            "tactic_name": r["tactic_name"],
            "rule_ids": [],
            "technique_ids": [],
        })
        tactics[r["tactic_id"]]["rule_ids"].append(r["rule_id"])
        if r["technique_id"]:
            tactics[r["tactic_id"]]["technique_ids"].append(r["technique_id"])

    return {
        "source_module": "src/mitre_mapper.py",
        "source_function": "build_rule_library()",
        "modified_by_this_layer": False,
        "n_rules": len(described),
        "feature_space": {
            "dimension": N_FEATURES,
            "base_features": N_BASE,
            "rel_change_features": N_FEATURES - N_BASE,
            "units": ("frozen input_scaler z-units; a threshold of 0.5 means "
                      "0.5 training standard deviations above the training mean"),
            "snapshot_taken_from": ("the LAST timestep of the 52-D window "
                                    "(MITREMapper._extract_feature_snapshot reads "
                                    "x_window[-1] and nothing else)"),
        },
        "evaluation_semantics": {
            "supported_operators": list(SUPPORTED_OPERATORS),
            "operators_actually_used": ops_used,
            "rule_score": "number of individual checks that matched (integer)",
            "multiple_rules_can_fire": True,
            "selection": ("ALL rules are evaluated; the winner is the rule with "
                          "the strictly-highest match count. Ties are resolved by "
                          "LIBRARY ORDER (the first rule to reach that count keeps "
                          "it), because the mapper's comparison is "
                          "`score > best_score`. This is best-match-with-"
                          "first-wins-ties, NOT pure first-match."),
            "partial_match_fires": ("yes — a rule fires on >=1 matched check, so a "
                                    "3-check rule can win with 1 match if no rule "
                                    "matches more. This is a LIMITATION, recorded "
                                    "not repaired."),
            "probability_gate": ("if attack_probability < mapper threshold the "
                                 "mapper returns tactic_id=NONE ('No Alert') and "
                                 "evaluates no rule at all"),
            "no_match_result": ("tactic_id=UNKNOWN "
                                "('Unknown / Insufficient Evidence')"),
        },
        "confidence_calculation": {
            "base": "rule.min_confidence (low|medium|high)",
            "adjustments": [
                "+1 if attack_probability > 0.5",
                "+1 if attack_probability > 0.7",
                "+1 if matched-evidence count >= 3",
                "+1 if temporal_phase in (mid, late)",
            ],
            "mapping": "score>=4 -> high; score>=2 -> medium; else low",
            "note": ("this is an ordinal heuristic inside the existing mapper, "
                     "not a probability. This layer reuses it verbatim."),
        },
        "temporal_phase_logic": {
            "source": "MITREMapper._determine_temporal_phase",
            "inputs": ["window_idx", "total_windows", "first_alert_idx"],
            "values": ["benign (below threshold)", "pre-alert", "onset",
                       "early", "mid", "late"],
            "definition": ("progress = (window_idx - first_alert_idx) / "
                           "max(total_windows - first_alert_idx, 1); "
                           "<0.25 early, <0.75 mid, else late"),
            "note": ("first_alert_idx is derived from MODEL OUTPUT only (the "
                     "first alerting sequence), never from a label"),
        },
        "features_read_by_rules": features_used,
        "tactics_available": list(tactics.values()),
        "non_tactic_outcomes": [
            {"tactic_id": "NONE", "tactic_name": "No Alert",
             "meaning": "forecast below the mapper threshold"},
            {"tactic_id": "UNKNOWN",
             "tactic_name": "Unknown / Insufficient Evidence",
             "meaning": "above threshold but no rule matched any check"},
        ],
        "known_limitations": [
            "Thresholds are hand-chosen z-score cut-points documented in the "
            "mapper docstring from test-set descriptive statistics; they were "
            "not fitted and are not re-tuned here.",
            "A rule fires on a single matched check, so evidence can be thin.",
            "Only one tactic is reported per window even though several rules "
            "may match; the runner-up rules are discarded by the mapper.",
            "No rule in the shipped library uses the 'spike' operator, so the "
            "26 rel-change features are currently never read by any rule. They "
            "are still reconstructed faithfully here so the mapper receives a "
            "complete, contract-valid 52-D vector.",
            "Tactic coverage is limited to the 7 network-observable tactics; "
            "Persistence, Privilege Escalation, Defense Evasion, Credential "
            "Access, Discovery and Collection have no rules.",
        ],
    }


# =============================================================
# 2. PROGRESSION GRAPH
# =============================================================
#
# ATT&CK tactics are NOT universally ordered. Real intrusions skip
# stages, repeat stages, and run stages concurrently. This graph is
# therefore NOT a claim about attacker behaviour. It is a small,
# project-specific ANALYTICAL ORDERING whose only job is to decide which
# forward transitions this dashboard is willing to VISUALISE, so that a
# numerically noisy relabelling between two adjacent horizons is not
# presented to an analyst as a progression.
#
# NODES: exactly the 7 tactics the existing mapper can emit. Nothing
# else. NONE ("No Alert") and UNKNOWN ("Insufficient Evidence") are
# deliberately NOT nodes — they carry no tactic semantics, so no edge
# can lead into or out of them and no progression is ever claimed
# through them.
#
# EDGE ADMISSION TEST — an edge A -> B is included only if BOTH hold:
#   (i)  ORDERING: B is ATT&CK-consistent as a stage that can follow A
#        within one intrusion, reading each node's meaning as the
#        SHIPPED RULE for that tactic defines it (not the tactic's full
#        ATT&CK scope).
#   (ii) DISTINGUISHABILITY: the rules for A and B key on materially
#        different telemetry, so "B after A" is a change in the observed
#        signature rather than the same signature relabelled.
#
# Every edge below carries its justification. Every edge that a generic
# kill chain would include but this graph REJECTS is recorded in
# REJECTED_EDGES with the reason, so the omissions are auditable too.

RECON = "Reconnaissance"
INITIAL_ACCESS = "Initial Access"
EXECUTION = "Execution"
LATERAL = "Lateral Movement"
C2 = "Command and Control"
EXFIL = "Exfiltration"
IMPACT = "Impact"

NO_ALERT = "No Alert"
UNKNOWN = "Unknown / Insufficient Evidence"

# Canonical display order (analytical, not causal).
STAGE_ORDER = [RECON, INITIAL_ACCESS, EXECUTION, LATERAL, C2, EXFIL, IMPACT]

TACTIC_IDS = {
    RECON: "TA0043",
    INITIAL_ACCESS: "TA0001",
    EXECUTION: "TA0002",
    LATERAL: "TA0008",
    C2: "TA0011",
    EXFIL: "TA0010",
    IMPACT: "TA0040",
    NO_ALERT: "NONE",
    UNKNOWN: "UNKNOWN",
}

PROGRESSION_EDGES: dict[tuple[str, str], str] = {
    (RECON, INITIAL_ACCESS):
        "RECON_PORT_SCAN / RECON_HOST_DISCOVERY key on port breadth and port "
        "entropy; INITIAL_ACCESS_INFILTRATION keys on flow count with SYN+FIN "
        "completion. Enumeration preceding connection completion against a "
        "discovered service is the canonical ATT&CK ordering, and the two "
        "signatures differ (breadth vs. handshake completion).",
    (RECON, LATERAL):
        "LATERAL_MOVEMENT_SPREAD requires log_flow_count>0.5 AND "
        "log_unique_dst_ports>0.3 AND log_total_packets>0.3 — strictly more "
        "than the recon rules, which need no packet volume. Internal "
        "enumeration hardening into volumetric remote-service use is a real "
        "observable escalation, not a relabelling.",
    (INITIAL_ACCESS, EXECUTION):
        "EXECUTION_PAYLOAD_DELIVERY keys on packet volume with PSH (data "
        "push), which is a different signature from the SYN/FIN handshake "
        "activity of the infiltration rule. Foothold then payload is the "
        "canonical ordering.",
    (INITIAL_ACCESS, C2):
        "A channel established directly after access. C2_BEACONING / "
        "C2_ENCRYPTED_CHANNEL key on LOW packet-size variance and low "
        "protocol entropy — the opposite direction from the initial-access "
        "rule's volume/flag criteria, so the transition is clearly observable.",
    (EXECUTION, C2):
        "Payload delivery followed by channel establishment. Execution keys "
        "on high volume + PSH, C2 on low variance + download ratio.",
    (EXECUTION, LATERAL):
        "Post-execution spread. Execution needs volume + PSH; lateral "
        "additionally needs port breadth, so lateral is a strict escalation.",
    (EXECUTION, IMPACT):
        "IMPACT_RESOURCE_EXHAUSTION requires window_packet_rate>1.0 AND "
        "syn_rate>0.5 — by far the most extreme thresholds in the library. "
        "Execution escalating to a volumetric denial-of-service effect is "
        "ATT&CK-consistent and the signature is unambiguous.",
    (LATERAL, C2):
        "Spread followed by channel consolidation; opposite telemetry "
        "directions (breadth/volume vs. uniform low-variance flows).",
    (LATERAL, EXFIL):
        "Collection across reached hosts leaving the network. Lateral keys on "
        "flow/port breadth, exfil on byte volume with LARGE mean packet "
        "length — volume-in-bytes rather than spread-in-flows.",
    (C2, EXFIL):
        "The canonical ATT&CK ordering: an established channel then carries "
        "data out. C2 keys on small uniform packets, exfil on large packets "
        "and high byte rate, so the shift is directly observable.",
    (C2, IMPACT):
        "Tasking over an established channel producing a destructive or "
        "volumetric effect.",
    (EXFIL, IMPACT):
        "Data theft followed by a destructive or disruptive effect.",
}

REJECTED_EDGES: dict[tuple[str, str], str] = {
    (RECON, EXECUTION):
        "REJECTED — skips access entirely. Scanning telemetry cannot support "
        "a jump to payload execution.",
    (RECON, C2):
        "REJECTED — a C2 channel implies a foothold this graph has not seen.",
    (RECON, EXFIL):
        "REJECTED — implies collection with no observed access or channel.",
    (RECON, IMPACT):
        "REJECTED — the two signatures are both SYN-heavy, so recon->impact "
        "would frequently be a threshold artefact rather than an escalation. "
        "Fails the DISTINGUISHABILITY test.",
    (INITIAL_ACCESS, LATERAL):
        "REJECTED — this project's semantics require an execution foothold "
        "before volumetric internal spread is a defensible reading.",
    (INITIAL_ACCESS, EXFIL):
        "REJECTED — implies collection before any execution or channel.",
    (INITIAL_ACCESS, IMPACT):
        "REJECTED — both rules read flow count and SYN rate, so the "
        "transition risks being a pure threshold crossing on the same "
        "signature. Fails DISTINGUISHABILITY.",
    (EXECUTION, EXFIL):
        "REJECTED — conservative: exfiltration is only visualised once a "
        "channel (C2) or a spread (Lateral Movement) has been observed.",
    (LATERAL, IMPACT):
        "REJECTED — conservative. Lateral and impact both key on volume; "
        "the escalation is only visualised via C2.",
    (EXFIL, C2):
        "REJECTED — backward relative to the analytical ordering.",
    "any -> Reconnaissance":
        "REJECTED for every source — Reconnaissance is a graph SOURCE only. "
        "Re-scanning happens in real intrusions, but visualising a backward "
        "edge would let two noisy horizons oscillate as a fake progression.",
    "any -> Initial Access except Reconnaissance":
        "REJECTED — Initial Access is reachable only from Reconnaissance.",
    "NONE / UNKNOWN as endpoints":
        "REJECTED — neither carries tactic semantics, so no progression is "
        "claimed into or out of them. A sequence whose current stage is "
        "NONE or UNKNOWN yields NO FORWARD PROGRESSION IDENTIFIED or "
        "PROGRESSION AMBIGUOUS, never an invented stage.",
}

TERMINAL_STAGES = (IMPACT,)


def allowed(src: str, dst: str) -> bool:
    """Is src -> dst an admitted forward-visualisation edge?"""
    return (src, dst) in PROGRESSION_EDGES


def successors(stage: str) -> list[str]:
    return [b for (a, b) in PROGRESSION_EDGES if a == stage]


def describe_progression_graph() -> dict:
    return {
        "nature": ("project-specific ANALYTICAL ORDERING for forward "
                   "visualisation. NOT a claim that intrusions follow this "
                   "order, and NOT a universal kill chain."),
        "nodes": list(STAGE_ORDER),
        "node_source": ("exactly the tactics src/mitre_mapper.py can emit; "
                        "no tactic was added"),
        "excluded_nodes": {
            "NONE / No Alert": "no tactic semantics — cannot be an endpoint",
            "UNKNOWN / Insufficient Evidence":
                "no tactic semantics — cannot be an endpoint",
        },
        "edge_admission_test": {
            "i_ordering": ("B is ATT&CK-consistent as a stage following A, read "
                           "through the SHIPPED RULE's meaning for each tactic"),
            "ii_distinguishability": ("the two rules key on materially different "
                                      "telemetry, so B-after-A is an observable "
                                      "signature change, not a relabelling"),
        },
        "n_edges": len(PROGRESSION_EDGES),
        "edges": [{"from": a, "to": b, "justification": j}
                  for (a, b), j in PROGRESSION_EDGES.items()],
        "rejected_edges": [{"edge": (k if isinstance(k, str) else f"{k[0]} -> {k[1]}"),
                            "reason": v}
                           for k, v in REJECTED_EDGES.items()],
        "terminal_stages": list(TERMINAL_STAGES),
        "acyclic": True,
    }


# =============================================================
# 3. RULE-SPACE RECONSTRUCTION  (the only numerical machinery here)
# =============================================================

class AttackPathError(RuntimeError):
    """Raised when a frozen precondition for this layer does not hold."""


@dataclass
class FrozenScalers:
    """The two frozen scalers, used ONLY in forward/inverse form."""
    mean_base: np.ndarray       # (26,) input_scaler.mean_[:26]
    scale_base: np.ndarray      # (26,)
    mean_rel: np.ndarray        # (26,) input_scaler.mean_[26:]
    scale_rel: np.ndarray       # (26,)
    target_matches_input_base: bool
    input_scaler_path: str
    target_scaler_path: str


def load_scalers(input_path: Path = INPUT_SCALER,
                 target_path: Path = TARGET_SCALER) -> FrozenScalers:
    """Load the frozen scalers and verify the decoder-space identity.

    CRITICAL, VERIFIED FACT: target_scaler (26-D, the space the V4 decoder
    writes into) is element-wise IDENTICAL to the first 26 columns of
    input_scaler (the space the mapper's base-feature thresholds live in).
    So a decoded 26-D state is ALREADY in mapper base-feature z-units and
    needs NO transform. This function asserts that rather than assuming it;
    if the two ever diverge, this layer refuses to run instead of silently
    comparing thresholds against the wrong units.
    """
    for p in (input_path, target_path):
        if not p.exists():
            raise AttackPathError(f"frozen scaler not found: {p}")

    with open(input_path, "rb") as fh:
        ins = pickle.load(fh)
    with open(target_path, "rb") as fh:
        tgt = pickle.load(fh)

    if int(ins.n_features_in_) != N_FEATURES:
        raise AttackPathError(
            f"input_scaler has {ins.n_features_in_} features, expected {N_FEATURES}")
    if int(tgt.n_features_in_) != N_BASE:
        raise AttackPathError(
            f"target_scaler has {tgt.n_features_in_} features, expected {N_BASE}")

    same = (np.array_equal(ins.mean_[:N_BASE], tgt.mean_)
            and np.array_equal(ins.scale_[:N_BASE], tgt.scale_))
    if not same:
        raise AttackPathError(
            "target_scaler is no longer identical to input_scaler[:26]. The "
            "decoded state would not be in the mapper's threshold units, and "
            "this layer will not guess a conversion.")

    return FrozenScalers(
        mean_base=np.asarray(ins.mean_[:N_BASE], dtype=np.float64),
        scale_base=np.asarray(ins.scale_[:N_BASE], dtype=np.float64),
        mean_rel=np.asarray(ins.mean_[N_BASE:], dtype=np.float64),
        scale_rel=np.asarray(ins.scale_[N_BASE:], dtype=np.float64),
        target_matches_input_base=True,
        input_scaler_path=str(input_path.relative_to(REPO_ROOT)),
        target_scaler_path=str(target_path.relative_to(REPO_ROOT)),
    )


def verify_frozen_formula() -> dict:
    """Assert this module's relative-change constants ARE the contract's.

    The formula is defined once, in src/feature_contract.py. This layer
    must not introduce a new transform, so the constants are checked
    against that single definition and the check is reported.
    """
    import feature_contract as fc
    eps_ok = float(fc.RELATIVE_EPSILON) == RELATIVE_EPSILON
    clip_ok = float(fc.RELATIVE_CLIP) == RELATIVE_CLIP
    if not (eps_ok and clip_ok):
        raise AttackPathError(
            f"relative-change constants diverged from the frozen contract: "
            f"epsilon {fc.RELATIVE_EPSILON} vs {RELATIVE_EPSILON}, "
            f"clip {fc.RELATIVE_CLIP} vs {RELATIVE_CLIP}")

    # Numerical agreement with the contract's own reference implementation,
    # on a deterministic synthetic state trajectory.
    import pandas as pd
    rng = np.random.default_rng(20260926)
    frame = pd.DataFrame(
        rng.normal(5.0, 2.0, size=(8, N_BASE)), columns=fc.BASE_FEATURES)
    expected = fc.reference_relative_change(frame).to_numpy()
    raw = frame.to_numpy()
    got = np.zeros_like(raw)
    for i in range(1, len(raw)):
        got[i] = _relative_change(raw[i - 1], raw[i])
    max_delta = float(np.abs(expected - got).max())
    if max_delta > 1e-12:
        raise AttackPathError(
            f"this layer's relative-change differs from "
            f"feature_contract.reference_relative_change by {max_delta:.3e}")

    return {
        "source_of_truth": "src/feature_contract.py::reference_relative_change",
        "epsilon": RELATIVE_EPSILON,
        "clip": RELATIVE_CLIP,
        "formula": "(x_t - x_{t-1}) / (|x_{t-1}| + 1e-8), clipped to [-10, 10]",
        "max_abs_delta_vs_contract": max_delta,
        "new_transform_introduced": False,
    }


def _relative_change(prev_raw: np.ndarray, cur_raw: np.ndarray) -> np.ndarray:
    """The frozen contract formula, one step, in RAW physical units."""
    delta = cur_raw - prev_raw
    rel = delta / (np.abs(prev_raw) + RELATIVE_EPSILON)
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(rel, -RELATIVE_CLIP, RELATIVE_CLIP)


def reconstruct_rule_space(x_window: np.ndarray,
                           decoded_states: np.ndarray,
                           sc: FrozenScalers) -> np.ndarray:
    """Build the (MAX_K+1, 52) rule-space trajectory the mapper will read.

    Args
    ----
    x_window        (10, 52) or (1, 10, 52) — the frozen, scaled observed
                    window. The ONLY tensor the model saw.
    decoded_states  (6, 26)  — V4 decoder output for k = 0..5, in
                    target_scaler space.
    sc              frozen scalers from load_scalers().

    Returns
    -------
    (6, 52) float64. Row 0 is the CURRENT state; rows 1..5 are t+1..t+5.

    HOW, EXACTLY
    ------------
    Row 0 (CURRENT) is NOT reconstructed. It is the actual observed 52-D
    vector x_window[-1], byte-for-byte what src/mitre_mapper.py already
    reads today on the MITRE ATT&CK Timeline page. Using the decoded k=0
    reconstruction here instead would substitute a lossy autoencoding of
    an observation we actually have.

    Rows 1..5 (FUTURE) are assembled from the ACTUAL decoded trajectory:

      base half  [0:26]
        = decoded_states[k] verbatim. No transform. Justified by
          load_scalers(): target_scaler == input_scaler[:26] element-wise,
          so the decoder already writes into the mapper's threshold units.

      rel-change half [26:52]
        The contract defines rel-change on RAW physical units, so:
          1. inverse-transform the base trajectory to raw units with the
             FROZEN scaler:  raw = z * scale_base + mean_base
             The chain's k=0 element is the OBSERVED last window
             (x_window[-1, :26] inverse-transformed), so t+1's change is
             measured against reality, not against a reconstruction.
          2. apply the FROZEN contract formula _relative_change(raw_{k-1},
             raw_k) — same epsilon, same clip, no new transform.
          3. forward-transform with the FROZEN scaler's rel-change columns:
             z = (rel - mean_rel) / scale_rel
        Every step uses only frozen parameters in their existing
        forward/inverse form. Nothing is fitted.

    No future observation enters this function. decoded_states is the
    model's own prediction; x_window is the past.
    """
    x = np.asarray(x_window, dtype=np.float64)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise AttackPathError(f"expected one window, got batch {x.shape}")
        x = x[0]
    if x.shape != (SEQ_LEN, N_FEATURES):
        raise AttackPathError(
            f"x_window has shape {x.shape}, expected ({SEQ_LEN}, {N_FEATURES})")

    dec = np.asarray(decoded_states, dtype=np.float64)
    if dec.shape != (MAX_K + 1, N_BASE):
        raise AttackPathError(
            f"decoded_states has shape {dec.shape}, "
            f"expected ({MAX_K + 1}, {N_BASE})")

    out = np.zeros((MAX_K + 1, N_FEATURES), dtype=np.float64)

    # --- row 0: the real observation, untouched ---
    out[0] = x[-1]

    # --- raw-unit chain: observed last window, then the decoded future ---
    obs_raw = x[-1, :N_BASE] * sc.scale_base + sc.mean_base
    raw_chain = [obs_raw] + [dec[k] * sc.scale_base + sc.mean_base
                             for k in range(1, MAX_K + 1)]

    for k in range(1, MAX_K + 1):
        out[k, :N_BASE] = dec[k]
        rel_raw = _relative_change(raw_chain[k - 1], raw_chain[k])
        out[k, N_BASE:] = (rel_raw - sc.mean_rel) / sc.scale_rel

    if not np.isfinite(out).all():
        raise AttackPathError("reconstructed rule-space contains non-finite values")
    return out


# =============================================================
# 4. STAGE EVALUATION  (delegated to the UNMODIFIED mapper)
# =============================================================

@dataclass
class StageAssessment:
    """One horizon's rule-based ATT&CK reading. Every field originates in
    src/mitre_mapper.py; none is computed here."""
    k: int
    horizon_label: str
    timestamp: str
    threat_probability: float
    tactic_id: str
    tactic_name: str
    technique_id: str | None
    technique_name: str | None
    rule_confidence: str
    temporal_phase: str
    supporting_features: list = field(default_factory=list)
    supporting_values: dict = field(default_factory=dict)
    explanation: str = ""

    @property
    def is_tactic(self) -> bool:
        """True only for the 7 real tactic nodes — not NONE, not UNKNOWN."""
        return self.tactic_name in STAGE_ORDER

    def to_dict(self):
        d = asdict(self)
        d["is_tactic"] = self.is_tactic
        return d


def evaluate_stage(mapper: "mm.MITREMapper",
                   rule_vector: np.ndarray,
                   k: int,
                   window_idx: int,
                   timestamp: str,
                   total_windows: int,
                   first_alert_idx: int | None,
                   threat_probability: float) -> StageAssessment:
    """Run the EXISTING mapper on one rule-space vector.

    The mapper reads only x_window[-1], so a (1, 52) array is passed: it
    receives exactly the intended snapshot and cannot silently average
    over fabricated history. Rule evaluation, best-match selection,
    confidence boosting, the probability gate and the UNKNOWN fallback
    are all the mapper's, unmodified.
    """
    mapping = mapper.map_single_window(
        x_window=rule_vector.reshape(1, N_FEATURES),
        window_idx=window_idx,
        timestamp=timestamp,
        total_windows=total_windows,
        first_alert_idx=first_alert_idx,
        attack_prob=float(threat_probability),
    )
    md = vars(mapping)
    return StageAssessment(
        k=k,
        horizon_label=("t (observed)" if k == 0 else f"t+{k * WINDOW_SECONDS}s"),
        timestamp=str(timestamp),
        threat_probability=float(threat_probability),
        tactic_id=str(md["tactic_id"]),
        tactic_name=str(md["tactic_name"]),
        technique_id=md.get("technique_id"),
        technique_name=md.get("technique_name"),
        rule_confidence=str(md.get("confidence", "n/a")),
        temporal_phase=str(md.get("temporal_phase", "")),
        supporting_features=list(md.get("triggering_features") or []),
        supporting_values={k2: float(v) for k2, v
                           in (md.get("feature_values") or {}).items()},
        explanation=str(md.get("analyst_explanation") or ""),
    )


# =============================================================
# 5. PROGRESSION CONFIDENCE  (deterministic, three levels)
# =============================================================
#
# This is NOT a statistical probability that the stage is correct. No
# such quantity is available and none is invented. It is a three-level
# DETERMINISTIC label whose criteria are published in full below and
# evaluated by rules_for_confidence() with no free parameters beyond the
# already-frozen alert threshold.

CONF_HIGH, CONF_MEDIUM, CONF_LOW = "HIGH", "MEDIUM", "LOW"

CONFIDENCE_CRITERIA = {
    "nature": ("deterministic three-level label. NOT a probability of "
               "stage correctness, NOT calibrated, NOT fitted."),
    "inputs": [
        "persistence: length of the CONSECUTIVE run of future horizons, "
        "starting at the horizon where the candidate stage first appears, "
        "that carry the SAME tactic AND the SAME technique",
        "rule_confidence: the existing MITREMapper confidence for the "
        "first horizon of that run (low|medium|high)",
        "threat_stability: whether the frozen V4 threat probability stays "
        "at or above the FROZEN alert threshold across the whole run",
    ],
    "HIGH": ("persistence >= 2 AND rule_confidence == 'high' AND "
             "threat_stability is True"),
    "MEDIUM": ("persistence >= 2 (any rule_confidence), OR "
               "(persistence == 1 AND rule_confidence == 'high' AND "
               "threat_stability is True)"),
    "LOW": "everything else — a single weakly-supported horizon",
    "no_tuning": ("the only numeric input is the frozen alert threshold "
                  "read from screening_predictions.csv. No threshold was "
                  "chosen, swept, or tuned on any label."),
}


def progression_confidence(persistence: int,
                           rule_confidence: str,
                           threat_stable: bool) -> str:
    rc = (rule_confidence or "").lower()
    if persistence >= 2 and rc == "high" and threat_stable:
        return CONF_HIGH
    if persistence >= 2:
        return CONF_MEDIUM
    if persistence == 1 and rc == "high" and threat_stable:
        return CONF_MEDIUM
    return CONF_LOW


# =============================================================
# 6. PATH CONSTRUCTION
# =============================================================

STATUS_PROGRESSION = "FORWARD_PROGRESSION_IDENTIFIED"
STATUS_NONE = "NO_FORWARD_PROGRESSION_IDENTIFIED"
STATUS_AMBIGUOUS = "PROGRESSION_AMBIGUOUS"

STATUS_DISPLAY = {
    STATUS_PROGRESSION: "FORWARD PROGRESSION HYPOTHESIS",
    STATUS_NONE: "NO FORWARD PROGRESSION IDENTIFIED",
    STATUS_AMBIGUOUS: "PROGRESSION AMBIGUOUS — ANALYST REVIEW",
}


@dataclass
class ProgressionStep:
    """One admitted forward transition."""
    from_stage: str
    to_stage: str
    to_tactic_id: str
    technique_id: str | None
    technique_name: str | None
    first_horizon_k: int
    time_to_progression_s: int
    timestamp: str
    threat_probability: float
    rule_confidence: str
    progression_confidence: str
    persistence_horizons: int
    threat_stable: bool
    supporting_features: list
    supporting_values: dict
    evidence_window: str
    edge_justification: str

    def to_dict(self):
        return asdict(self)


@dataclass
class AttackPathHypothesis:
    """The complete per-sequence output of this layer."""
    sequence_id: int
    status: str
    current_stage: str
    current_tactic_id: str
    current_technique_id: str | None
    current_technique_name: str | None
    current_rule_confidence: str
    current_threat_probability: float
    current_timestamp: str
    current_supporting_features: list
    current_supporting_values: dict
    max_forecast_probability: float
    alert: int
    alert_threshold: float
    steps: list                      # list[ProgressionStep]
    horizon_stages: list             # list[StageAssessment]
    reliability_status: str | None
    reliability_short: str | None
    reliability_headline: str | None
    ambiguity_reason: str | None
    # Where the chain's first node came from. "observed_current" when the
    # OBSERVED window already maps to a tactic; "projected_at_k<N>" when it
    # does not and the anchor was established at the first predicted horizon
    # that yields a tactic (permitted by the spec's "current/projected
    # stage"). None when no tactic appears anywhere in k=0..5.
    anchor_origin: str | None = None
    anchor_stage: str | None = None
    anchor_k: int | None = None
    # Machine-readable ambiguity taxonomy, so the distribution can be
    # reported without parsing prose.
    ambiguity_kind: str | None = None
    layer_version: str = LAYER_VERSION

    @property
    def next_step(self):
        return self.steps[0] if self.steps else None

    @property
    def time_to_first_progression_s(self):
        return self.steps[0].time_to_progression_s if self.steps else None

    @property
    def reliability_suppresses_timing(self) -> bool:
        """UNRELIABLE forecasts must not show a bare precise-looking time."""
        return self.reliability_status == "UNRELIABLE"

    def to_dict(self):
        return {
            "sequence_id": self.sequence_id,
            "status": self.status,
            "status_display": STATUS_DISPLAY[self.status],
            "current_stage": self.current_stage,
            "current_tactic_id": self.current_tactic_id,
            "current_technique_id": self.current_technique_id,
            "current_technique_name": self.current_technique_name,
            "current_rule_confidence": self.current_rule_confidence,
            "current_threat_probability": self.current_threat_probability,
            "current_timestamp": self.current_timestamp,
            "current_supporting_features": self.current_supporting_features,
            "current_supporting_values": self.current_supporting_values,
            "max_forecast_probability": self.max_forecast_probability,
            "alert": self.alert,
            "alert_threshold": self.alert_threshold,
            "n_steps": len(self.steps),
            "time_to_first_progression_s": self.time_to_first_progression_s,
            "steps": [s.to_dict() for s in self.steps],
            "horizon_stages": [h.to_dict() for h in self.horizon_stages],
            "reliability_status": self.reliability_status,
            "reliability_short": self.reliability_short,
            "reliability_headline": self.reliability_headline,
            "reliability_suppresses_timing": self.reliability_suppresses_timing,
            "ambiguity_reason": self.ambiguity_reason,
            "ambiguity_kind": self.ambiguity_kind,
            "anchor_origin": self.anchor_origin,
            "anchor_stage": self.anchor_stage,
            "anchor_k": self.anchor_k,
            "layer_version": self.layer_version,
        }


def _persistence(stages: list[StageAssessment], k: int) -> int:
    """Consecutive run length of (tactic, technique) starting at horizon k."""
    ref = stages[k]
    run = 0
    for j in range(k, MAX_K + 1):
        s = stages[j]
        if (s.tactic_name == ref.tactic_name
                and s.technique_id == ref.technique_id):
            run += 1
        else:
            break
    return run


AMB_CURRENT_NOT_TACTIC = "current_and_projected_stages_carry_no_tactic"
AMB_NO_ADMITTED_EDGE = "no_admitted_edge_from_anchor_stage"
AMB_SINGLE_STAGE_ONLY = "single_stage_persists_no_transition"

AMBIGUITY_KINDS = {
    AMB_CURRENT_NOT_TACTIC: (
        "The forecast is elevated but neither the observed window nor any "
        "predicted horizon maps to one of the 7 tactic nodes — every horizon "
        "returned No Alert or Unknown / Insufficient Evidence. The threat "
        "layer and the ATT&CK layer disagree; an analyst must review."),
    AMB_NO_ADMITTED_EDGE: (
        "An anchor stage exists and later horizons produce OTHER tactics, but "
        "none of those transitions is an admitted edge in the progression "
        "graph. The relabelling is treated as unexplained rather than "
        "presented as a progression."),
    AMB_SINGLE_STAGE_ONLY: (
        "One tactic is present and persists across the rollout with no second "
        "tactic appearing. There is a stage but no transition, so no "
        "time-to-progression is defined."),
}


def build_path(sequence_id: int,
               stages: list[StageAssessment],
               max_forecast_probability: float,
               alert: int,
               alert_threshold: float,
               reliability=None) -> AttackPathHypothesis:
    """Assemble the ordered progression hypothesis for one sequence.

    ALGORITHM (fully deterministic, no randomness, no learned component)

      1. ESTABLISH THE ANCHOR — the chain's first node.
         If the OBSERVED window (k=0) maps to one of the 7 tactic nodes,
         the anchor is that stage and anchor_origin = "observed_current".
         Otherwise (No Alert / Unknown at k=0) the anchor is the FIRST
         predicted horizon k in 1..5 that yields a tactic, and
         anchor_origin = "projected_at_k<N>". The spec admits an edge from
         "the current/projected stage", so anchoring on a projected stage
         is in scope. NONE and UNKNOWN are never anchors and never nodes,
         so no stage is ever invented for them.

      2. CHAIN FORWARD from the anchor's horizon:
         for k = anchor_k+1 .. 5, in time order:
             s := stage at horizon k
             skip if s is not one of the 7 tactic nodes
             skip if s == anchor            (no stage is repeated)
             skip if (anchor -> s) is not an admitted graph edge
             otherwise EMIT a step (anchor -> s at horizon k), then
                 anchor := s
         Stop at a terminal stage. The graph is acyclic and each step
         advances the anchor along an edge, so the path is a simple
         forward chain.

    Every emitted step is a tactic -> tactic transition across an admitted
    edge, produced entirely by the deterministic rule layer.

    A "one-window numerical spike" is not suppressed by an extra invented
    rule: it is reported with persistence = 1, which CONFIDENCE_CRITERIA
    maps to LOW unless the mapper's OWN rule confidence is already high and
    the threat probability is at or above the frozen threshold. The rule
    system's own evidence therefore decides, as required.

    STATUS
      FORWARD_PROGRESSION_IDENTIFIED      at least one step emitted
      PROGRESSION_AMBIGUOUS               no step emitted, but the forecast
                                          is elevated (the frozen alert
                                          fired) — see ambiguity_kind
      NO_FORWARD_PROGRESSION_IDENTIFIED   otherwise. This is a valid,
                                          expected output.
    """
    cur = stages[0]

    # ---- 1. establish the anchor ----
    if cur.is_tactic:
        anchor = cur.tactic_name
        anchor_k = 0
        anchor_origin = "observed_current"
    else:
        anchor, anchor_k, anchor_origin = None, None, None
        for k in range(1, MAX_K + 1):
            if stages[k].is_tactic:
                anchor = stages[k].tactic_name
                anchor_k = k
                anchor_origin = f"projected_at_k{k}"
                break

    # ---- 2. chain forward ----
    steps: list[ProgressionStep] = []
    anchor_stage_first = anchor
    if anchor is not None:
        for k in range(anchor_k + 1, MAX_K + 1):
            s = stages[k]
            if not s.is_tactic or s.tactic_name == anchor:
                continue
            if not allowed(anchor, s.tactic_name):
                continue
            run = _persistence(stages, k)
            probs_in_run = [stages[j].threat_probability
                            for j in range(k, k + run)]
            stable = all(p >= alert_threshold for p in probs_in_run)
            steps.append(ProgressionStep(
                from_stage=anchor,
                to_stage=s.tactic_name,
                to_tactic_id=s.tactic_id,
                technique_id=s.technique_id,
                technique_name=s.technique_name,
                first_horizon_k=k,
                time_to_progression_s=k * WINDOW_SECONDS,
                timestamp=s.timestamp,
                threat_probability=s.threat_probability,
                rule_confidence=s.rule_confidence,
                progression_confidence=progression_confidence(
                    run, s.rule_confidence, stable),
                persistence_horizons=run,
                threat_stable=stable,
                supporting_features=list(s.supporting_features),
                supporting_values=dict(s.supporting_values),
                evidence_window=(
                    f"predicted horizons t+{k * WINDOW_SECONDS}s"
                    f"\u2026t+{(k + run - 1) * WINDOW_SECONDS}s"
                    if run > 1 else f"predicted horizon t+{k * WINDOW_SECONDS}s"),
                edge_justification=PROGRESSION_EDGES[(anchor, s.tactic_name)],
            ))
            anchor = s.tactic_name
            if anchor in TERMINAL_STAGES:
                break

    # ---- 3. status + ambiguity taxonomy ----
    ambiguity_reason = None
    ambiguity_kind = None
    if steps:
        status = STATUS_PROGRESSION
    elif alert:
        status = STATUS_AMBIGUOUS
        elevated = (f"The forecast is elevated (max t+1\u2026t+5 = "
                    f"{max_forecast_probability:.3f} >= {alert_threshold:.2f})")
        if anchor_stage_first is None:
            ambiguity_kind = AMB_CURRENT_NOT_TACTIC
            ambiguity_reason = (
                f"{elevated} but no horizon k=0\u20265 mapped to a tactic: the "
                f"observed window reads '{cur.tactic_name}' and no predicted "
                f"horizon produced one either, so no forward edge can be "
                f"evaluated.")
        else:
            others = sorted({s.tactic_name for s in stages
                             if s.is_tactic and s.tactic_name != anchor_stage_first})
            if others:
                ambiguity_kind = AMB_NO_ADMITTED_EDGE
                ambiguity_reason = (
                    f"{elevated}. Anchor stage '{anchor_stage_first}' "
                    f"({anchor_origin}), and later horizons produced "
                    f"{', '.join(others)}, but no admitted graph edge connects "
                    f"them, so the relabelling is not presented as a "
                    f"progression.")
            else:
                ambiguity_kind = AMB_SINGLE_STAGE_ONLY
                ambiguity_reason = (
                    f"{elevated}. '{anchor_stage_first}' is the only tactic "
                    f"across k=0\u20265, so there is a stage but no transition "
                    f"and no time-to-progression is defined.")
    else:
        status = STATUS_NONE

    rs = rsh = rhl = None
    if reliability is not None:
        rs = getattr(reliability, "status", None)
        rsh = getattr(reliability, "short", None)
        rhl = getattr(reliability, "headline", None)

    return AttackPathHypothesis(
        sequence_id=int(sequence_id),
        status=status,
        current_stage=cur.tactic_name,
        current_tactic_id=cur.tactic_id,
        current_technique_id=cur.technique_id,
        current_technique_name=cur.technique_name,
        current_rule_confidence=cur.rule_confidence,
        current_threat_probability=cur.threat_probability,
        current_timestamp=cur.timestamp,
        current_supporting_features=list(cur.supporting_features),
        current_supporting_values=dict(cur.supporting_values),
        max_forecast_probability=float(max_forecast_probability),
        alert=int(alert),
        alert_threshold=float(alert_threshold),
        steps=steps,
        horizon_stages=list(stages),
        reliability_status=rs,
        reliability_short=rsh,
        reliability_headline=rhl,
        ambiguity_reason=ambiguity_reason,
        anchor_origin=anchor_origin,
        anchor_stage=anchor_stage_first,
        anchor_k=anchor_k,
        ambiguity_kind=ambiguity_kind,
    )


# =============================================================
# 7. TIME-TO-PROGRESSION DEFINITION
# =============================================================

TIME_TO_PROGRESSION = {
    "name": "estimated model-based time-to-progression",
    "formula": "time_to_progression = k * 10 seconds",
    "why_10s": ("each V4 rollout step advances the world model by exactly one "
                "10-second feature window, the frozen WINDOW_SECONDS of the "
                "V4 ingestion contract"),
    "k_definition": ("the FIRST predicted horizon k in 1..5 at which the "
                     "candidate next stage is produced by the existing rule "
                     "library and is reachable from the current stage by an "
                     "admitted graph edge"),
    "range": "10 s to 50 s, or undefined when no progression is identified",
    "is_not": [
        "NOT a guaranteed time until an attacker progresses",
        "NOT a survival-analysis estimate",
        "NOT calibrated against observed progression times",
        "NOT valid beyond 50 s — the V4 rollout is frozen at K=5",
    ],
    "suppressed_when": ("the Task-15 reliability status is UNRELIABLE; the "
                        "panel then shows the warning instead of a bare "
                        "precise-looking number"),
}


# =============================================================
# 8. PREDICTOR FACADE
# =============================================================

class AttackPathPredictor:
    """Stateless facade: frozen model + frozen mapper -> hypothesis.

    Usage
    -----
        p = AttackPathPredictor(model, alert_threshold=0.69)
        h = p.predict(seq_id, x_window, decoded_states, probs, timestamps,
                      total_windows, first_alert_idx, max_forecast, alert)
    """

    def __init__(self, model, alert_threshold: float,
                 scalers: FrozenScalers | None = None,
                 device: str = "cpu"):
        # The mapper is constructed, never modified. Its threshold is the
        # FROZEN alert threshold read from screening_predictions.csv — this
        # layer chooses no threshold of its own.
        self.mapper = mm.MITREMapper(model=model,
                                     threshold=float(alert_threshold),
                                     device=device)
        self.alert_threshold = float(alert_threshold)
        self.scalers = scalers or load_scalers()

    @property
    def n_rules(self) -> int:
        return len(self.mapper.rules)

    def predict(self, sequence_id: int,
                x_window: np.ndarray,
                decoded_states: np.ndarray,
                probs: np.ndarray,
                timestamps: list[str],
                total_windows: int,
                first_alert_idx: int | None,
                max_forecast_probability: float,
                alert: int,
                reliability=None) -> AttackPathHypothesis:
        """One sequence in, one hypothesis out.

        probs        (6,) frozen V4 threat probabilities k=0..5
        timestamps   6 display timestamps, k=0..5
        """
        probs = np.asarray(probs, dtype=np.float64).ravel()
        if probs.shape != (MAX_K + 1,):
            raise AttackPathError(
                f"probs has shape {probs.shape}, expected ({MAX_K + 1},)")
        if len(timestamps) != MAX_K + 1:
            raise AttackPathError(
                f"got {len(timestamps)} timestamps, expected {MAX_K + 1}")

        rule_space = reconstruct_rule_space(x_window, decoded_states, self.scalers)

        stages = [
            evaluate_stage(
                mapper=self.mapper,
                rule_vector=rule_space[k],
                k=k,
                # The mapper's temporal-phase logic is index-based; advancing
                # the index by k keeps a predicted horizon on the same clock
                # as the replay. first_alert_idx comes from MODEL OUTPUT only.
                window_idx=int(sequence_id) + k,
                timestamp=str(timestamps[k]),
                total_windows=int(total_windows),
                first_alert_idx=first_alert_idx,
                threat_probability=float(probs[k]),
            )
            for k in range(MAX_K + 1)
        ]

        return build_path(
            sequence_id=sequence_id,
            stages=stages,
            max_forecast_probability=max_forecast_probability,
            alert=alert,
            alert_threshold=self.alert_threshold,
            reliability=reliability,
        )


# =============================================================
# 9. LAYER METADATA
# =============================================================

def layer_metadata() -> dict:
    """Everything a reviewer needs to audit this layer, in one dict."""
    return {
        "layer_version": LAYER_VERSION,
        "module": "src/attack_path_predictor.py",
        "task": "Task 16 / S2 — ATT&CK-consistent attack-path + "
                "time-to-progression",
        "wording": {
            "correct": "ATT&CK-consistent progression hypothesis",
            "forbidden": [
                "learned attack path prediction",
                "guaranteed time until attacker progresses",
                "the model predicts ATT&CK labels",
                "attacker intent",
            ],
        },
        "composition": [
            "frozen V4 checkpoint (encoder / gated transition / decoder / "
            "attack head) — read-only",
            "frozen input_scaler + target_scaler — forward/inverse only",
            "frozen relative-change formula from src/feature_contract.py",
            "src/mitre_mapper.py rule library — imported UNMODIFIED",
            "project-specific progression graph defined in this module",
        ],
        "guarantees": {
            "retrains_v4": False,
            "modifies_v4_checkpoint": False,
            "refits_scalers": False,
            "modifies_feature_definitions": False,
            "modifies_thresholds": False,
            "modifies_mitre_mapper": False,
            "modifies_screening_artifacts": False,
            "learned_attack_stage_classifier": False,
            "uses_labels_as_model_input": False,
            "uses_future_observations_as_model_input": False,
            "deterministic": True,
        },
        "rule_library": describe_rule_library(),
        "progression_graph": describe_progression_graph(),
        "time_to_progression": TIME_TO_PROGRESSION,
        "confidence_criteria": CONFIDENCE_CRITERIA,
        "statuses": {k: v for k, v in STATUS_DISPLAY.items()},
        "ambiguity_kinds": dict(AMBIGUITY_KINDS),
        "anchor_rule": {
            "observed_current": ("the observed window k=0 already maps to one of "
                                 "the 7 tactic nodes"),
            "projected_at_kN": ("k=0 mapped to No Alert or Unknown, so the chain "
                                "is anchored at the FIRST predicted horizon that "
                                "yields a tactic. Admitted by the spec's "
                                "'allowed edge from the current/projected stage'."),
            "never": ("No Alert and Unknown are never anchors and never nodes, so "
                      "no stage is invented for them"),
        },
        "reliability_integration": {
            "source": "src/v4_reliability.py (Task 15)",
            "RELIABLE": "hypothesis presented normally",
            "CAUTION": "visible caution badge alongside the hypothesis",
            "UNRELIABLE": ("panel states PROGRESSION HYPOTHESIS UNRELIABLE and "
                           "the precise time-to-progression is withheld"),
            "bypassable": False,
        },
        "limitations": [
            "The progression graph is an analytical ordering for "
            "visualisation, not a model of attacker behaviour.",
            "Stages inherit every limitation of the rule library, including "
            "single-check firing and hand-chosen z-score thresholds.",
            "Future stages are read off a DECODED prediction; decoder "
            "reconstruction error at k=5 is inherited, not corrected.",
            "The rel-change half of a future rule-space vector is "
            "reconstructed from the decoded base trajectory. It is faithful "
            "to the frozen formula but is a prediction of a derivative, so "
            "it is noisier than the base half. No shipped rule currently "
            "reads it.",
            "Only 3 documented onset episodes exist in the replay substrate, "
            "so no general attack-stage accuracy can be claimed.",
            "No stage label exists for this dataset, so no precision / "
            "recall / F1 over stages is computed or computable.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(layer_metadata(), indent=2, default=str))
