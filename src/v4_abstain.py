"""
=============================================================
V4 ABSTAIN LAYER — human-in-the-loop three-way decision
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Task 17 / S3. A DECISION-SUPPORT layer on top of the FROZEN V4 forecast,
the Task-15 reliability layer and the Task-16 attack-path layer. It emits
one of three analyst-facing outcomes:

    ALERT    the frozen screening criterion is crossed under operating
             conditions the reference distribution covers
    BENIGN   the frozen screening criterion is not crossed, likewise
    ABSTAIN  the model DECLINES TO MAKE AN AUTOMATED DECISION and
             recommends analyst review

ABSTAIN DOES NOT MEAN "the model says nothing". It means "this output
should not be acted on automatically". The forecast probability is still
produced, still displayed, and still available to the analyst.

WHAT THIS IS NOT
----------------
- NOT a learned abstention head. No neural network was trained, extended
  or fine-tuned. The policy is arithmetic over already-computed signals.
- NOT a calibrated confidence. No calibration was fitted and none is
  demonstrated, so the word "calibrated" is not used for any output.
- NOT a replacement for the Task-15 reliability layer. It CONSUMES that
  layer's status and cannot override it.
- NOT a change to the frozen 0.69 decision boundary. ALERT vs BENIGN is
  decided by exactly the frozen rule, unchanged.

THRESHOLD PROVENANCE
--------------------
Two numbers govern this layer.

  0.69  the frozen trajectory-MAX K=5 alert threshold. NOT chosen here.
        It comes from docs/v4_baseline_manifest.json ->
        threshold.per_horizon_trajectory_max["5"], selected by grid
        search on the VALIDATION split. This layer reads it and never
        modifies it.

  0.05  the abstention margin half-width. Selected on the VALIDATION
        split ONLY, by the declared constraint rule in
        MARGIN_SELECTION_RULE. No held-out March-2 test label was read
        before this number was frozen, and it was not revised after
        test performance was seen.

Usage:
    import v4_abstain as ab
    policy = ab.load_policy()                  # frozen JSON
    d = policy.decide(max_forecast=0.81, reliability_status="RELIABLE")
    print(d.decision, d.reasons)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "abstain"
POLICY_PATH = OUT_DIR / "decision_policy.json"

LAYER_VERSION = "v4-abstain-1.0"

# The three outcomes.
ALERT, BENIGN, ABSTAIN = "ALERT", "BENIGN", "ABSTAIN"
DECISIONS = (ALERT, BENIGN, ABSTAIN)

DECISION_DISPLAY = {
    ALERT: "ALERT",
    BENIGN: "BENIGN",
    ABSTAIN: "ABSTAIN / ANALYST REVIEW",
}

# Analyst-facing semantics. These strings are the contract the UI renders;
# they are deliberately worded so ABSTAIN cannot read as "no signal".
DECISION_SEMANTICS = {
    ALERT: ("The analyst receives a forecast-based warning: the frozen "
            "screening criterion is crossed within the next 50 s."),
    BENIGN: ("Current evidence and forecast do not cross the frozen "
             "screening criterion."),
    ABSTAIN: ("The model output should NOT be acted upon automatically. "
              "Analyst review is recommended because reliability or "
              "decision-margin evidence is insufficient for an automated "
              "call. This is the model DECLINING TO DECIDE — not the model "
              "reporting an absence of signal. The forecast probability is "
              "still shown and still available to the analyst."),
}

ABSTAIN_HEADLINE = "AUTOMATIC DECISION WITHHELD"
ABSTAIN_SUBLINE = "Analyst review recommended."

# Reason codes. Every abstention names at least one.
R_UNRELIABLE = "RELIABILITY_UNRELIABLE"
R_CAUTION = "RELIABILITY_CAUTION"
R_MARGIN = "DECISION_MARGIN_NARROW"

REASON_TEXT = {
    R_UNRELIABLE: ("Forecast reliability is UNRELIABLE — the Task-15 layer "
                   "reports the forecast was produced outside the reference "
                   "operating conditions (out-of-distribution input, rollout "
                   "instability, or a hard numerical fail-safe)."),
    R_CAUTION: ("Forecast reliability is CAUTION — one or more Task-15 "
                "indicators sit above the reference 99th percentile."),
    R_MARGIN: ("The forecast sits within the validation-derived abstention "
               "margin of the frozen 0.69 decision boundary, so the "
               "ALERT/BENIGN call would rest on a difference the validation "
               "split does not support resolving automatically."),
}

# Human-readable grouping used by the dashboard's reason list.
REASON_GROUP = {
    R_UNRELIABLE: "out-of-distribution input / forecast instability",
    R_CAUTION: "unusual operating conditions",
    R_MARGIN: "narrow decision margin",
}

MARGIN_SELECTION_RULE = {
    "split_used": "VALIDATION ONLY (data/processed/multiday_benchmark_K5/"
                  "X_val.npy + A_binary_multistep_val.npy)",
    "grid": "numpy.arange(0.00, 0.31, 0.01), rounded to 2 decimals",
    "rule": ("among all grid deltas whose SELECTIVE F1 on the non-abstained "
             "validation windows is >= the no-abstain baseline validation "
             "F1, choose the LARGEST delta"),
    "why_this_rule": (
        "A plain 'minimise selective error' rule is actively misleading on "
        "this data. The model's positive predictions cluster just above "
        "0.69, so a wide margin band abstains on almost every ALERT and "
        "leaves a retained set that is ~82% negative. Error rate then falls "
        "simply because the retained classifier approaches 'always BENIGN' "
        "— an artefact of class imbalance, not a better decision. The F1 "
        "floor is therefore imposed as a CONSTRAINT: abstention must not be "
        "bought by discarding the operating point's recall. Error and FPR "
        "reduction are then a measured OUTCOME of the gate, never its "
        "optimisation target."),
    "declared_before_test": True,
    "test_labels_used_for_selection": False,
}

# Signals that are deliberately NOT gate inputs, with the reason. Recorded
# because "we investigated it and rejected it" is a result.
REJECTED_SIGNALS = {
    "attack_path_status": (
        "REJECTED as a gate input. Measured on validation, the Task-16 "
        "status partitions EXACTLY on the 0.69 alert boundary "
        "(NO_FORWARD_PROGRESSION <=> alert=0; AMBIGUOUS and "
        "FORWARD_PROGRESSION <=> alert=1, 1473/1473 windows). It is "
        "downstream of the same threshold, so it carries no information "
        "independent of the ALERT/BENIGN bit and cannot inform the gate. "
        "It IS shown in the UI as decision RATIONALE, which is a different "
        "role."),
    "early_warning_status": (
        "REJECTED as a gate input. Early-warning status is defined relative "
        "to a DOCUMENTED ATTACK ONSET, so it is label-derived and only "
        "knowable retrospectively. Using it at decision time would be "
        "leakage. It appears in the retrospective reporting only."),
    "forecast_trajectory_consistency": (
        "INVESTIGATED, not adopted. Trajectory spread across k=1..5 is "
        "already an input to the Task-15 rollout-stability indicators, so "
        "adding it here would double-count the same evidence under a "
        "second, independently-chosen threshold."),
    "learned_confidence_mapping": (
        "NOT ATTEMPTED. No calibration was fitted, so no output of this "
        "layer is described as calibrated."),
}


class AbstainPolicyError(RuntimeError):
    """Raised when the frozen policy cannot be loaded or trusted.

    Callers must surface this. A decision layer that silently degrades
    into 'decide anyway' defeats its purpose."""


@dataclass
class Decision:
    """One window's three-way analyst decision."""
    decision: str
    display: str
    semantics: str
    reasons: list = field(default_factory=list)          # reason codes
    reason_text: list = field(default_factory=list)      # prose
    max_forecast: float = 0.0
    nowcast: float | None = None
    margin: float = 0.0                                  # |score - threshold|
    side: str = ""                                       # above/below boundary
    frozen_alert_bit: int = 0                            # the BASELINE call
    reliability_status: str = ""
    attack_path_status: str | None = None                # rationale only
    layer_version: str = LAYER_VERSION

    @property
    def abstained(self) -> bool:
        return self.decision == ABSTAIN

    @property
    def agrees_with_baseline(self) -> bool:
        """Does the non-abstained decision match plain frozen thresholding?

        It always must: this layer never flips ALERT<->BENIGN, it only
        withholds. Validated as a hard gate."""
        if self.abstained:
            return True
        return (self.decision == ALERT) == bool(self.frozen_alert_bit)

    def to_dict(self):
        d = asdict(self)
        d["abstained"] = self.abstained
        return d


@dataclass
class AbstainPolicy:
    """The frozen, versioned three-way policy."""
    version: str
    alert_threshold: float               # frozen 0.69 — read, never chosen
    margin_delta: float                  # validation-selected
    caution_abstains: bool               # validation-informed
    threshold_source: str
    margin_evidence: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # ---------------------------------------------------------
    # THE POLICY  (deterministic, 3 branches, evaluated in order)
    # ---------------------------------------------------------

    def decide(self, max_forecast: float,
               reliability_status: str,
               nowcast: float | None = None,
               attack_path_status: str | None = None) -> Decision:
        """Apply the frozen three-way policy to one window.

        ORDER OF EVALUATION — the first two branches are ABSTAIN branches,
        so a hard reliability failure can never be overridden by a
        confident-looking probability.

          1. HARD SAFETY RULE. reliability == UNRELIABLE -> ABSTAIN,
             unconditionally. No bypass exists, and no probability value
             can reach past this branch.
          2. MARGIN / CAUTION GATE. ABSTAIN if the forecast lies within
             margin_delta of the frozen boundary, or (when
             caution_abstains) if reliability == CAUTION.
          3. OTHERWISE the FROZEN rule decides, unchanged:
             max_forecast > alert_threshold -> ALERT else BENIGN.

        `attack_path_status` is accepted and carried through for UI
        rationale ONLY. It is never read by any branch — see
        REJECTED_SIGNALS.
        """
        score = float(max_forecast)
        margin = abs(score - self.alert_threshold)
        # The BASELINE call, recorded so the comparison in section 5 is
        # exact. Strict '>' matches the frozen selection convention.
        frozen_bit = int(score > self.alert_threshold)
        side = "above" if score > self.alert_threshold else "below"

        reasons: list[str] = []

        # ---- 1. hard safety rule ----
        if reliability_status == "UNRELIABLE":
            reasons.append(R_UNRELIABLE)

        # ---- 2. validation-frozen gate ----
        if margin < self.margin_delta:
            reasons.append(R_MARGIN)
        if self.caution_abstains and reliability_status == "CAUTION":
            reasons.append(R_CAUTION)

        if reasons:
            decision = ABSTAIN
        else:
            decision = ALERT if frozen_bit else BENIGN

        return Decision(
            decision=decision,
            display=DECISION_DISPLAY[decision],
            semantics=DECISION_SEMANTICS[decision],
            reasons=reasons,
            reason_text=[REASON_TEXT[r] for r in reasons],
            max_forecast=score,
            nowcast=(None if nowcast is None else float(nowcast)),
            margin=margin,
            side=side,
            frozen_alert_bit=frozen_bit,
            reliability_status=str(reliability_status),
            attack_path_status=attack_path_status,
        )

    # ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "layer_version": self.version,
            "decisions": list(DECISIONS),
            "decision_display": dict(DECISION_DISPLAY),
            "decision_semantics": dict(DECISION_SEMANTICS),
            "policy": {
                "evaluation_order": [
                    "1. HARD SAFETY RULE: reliability == UNRELIABLE -> "
                    "ABSTAIN (unconditional, no bypass)",
                    "2. VALIDATION-FROZEN GATE: |max_forecast - "
                    f"{self.alert_threshold}| < {self.margin_delta} -> "
                    "ABSTAIN"
                    + ("; or reliability == CAUTION -> ABSTAIN"
                       if self.caution_abstains else ""),
                    "3. OTHERWISE the frozen rule decides, unchanged: "
                    f"max_forecast > {self.alert_threshold} -> ALERT "
                    "else BENIGN",
                ],
                "alert_threshold": self.alert_threshold,
                "alert_threshold_changed_by_this_layer": False,
                "alert_threshold_source": self.threshold_source,
                "margin_delta": self.margin_delta,
                "caution_abstains": self.caution_abstains,
                "deterministic": True,
                "learned_components": 0,
                "never_flips_alert_and_benign": True,
            },
            "margin_selection_rule": MARGIN_SELECTION_RULE,
            "margin_evidence": self.margin_evidence,
            "reason_codes": {k: {"text": v, "group": REASON_GROUP[k]}
                             for k, v in REASON_TEXT.items()},
            "rejected_signals": dict(REJECTED_SIGNALS),
            "abstain_wording": {
                "headline": ABSTAIN_HEADLINE,
                "subline": ABSTAIN_SUBLINE,
                "must_not_be_presented_as": "model says nothing",
                "means": "model declines to make an automated decision",
            },
            "provenance": self.provenance,
        }

    def policy_hash(self) -> str:
        """Hash of the DECIDING parameters only.

        Evidence and prose are excluded, so the hash proves the decision
        function itself did not move between the validation freeze and the
        test evaluation."""
        core = json.dumps({
            "version": self.version,
            "alert_threshold": self.alert_threshold,
            "margin_delta": self.margin_delta,
            "caution_abstains": self.caution_abstains,
        }, sort_keys=True)
        return hashlib.sha256(core.encode()).hexdigest()


def load_policy(path: Path | str = POLICY_PATH) -> AbstainPolicy:
    """Load the frozen policy written by src/build_abstain.py."""
    path = Path(path)
    if not path.exists():
        raise AbstainPolicyError(
            f"decision policy not found: {path}. Build it with "
            f"`python src/build_abstain.py`. There is deliberately no "
            f"fallback: an abstention layer that invents its own gate when "
            f"the frozen one is missing would be worse than none.")
    cfg = json.loads(path.read_text())
    if cfg.get("layer_version") != LAYER_VERSION:
        raise AbstainPolicyError(
            f"policy version {cfg.get('layer_version')!r} does not match "
            f"this module ({LAYER_VERSION!r})")
    p = cfg["policy"]
    return AbstainPolicy(
        version=cfg["layer_version"],
        alert_threshold=float(p["alert_threshold"]),
        margin_delta=float(p["margin_delta"]),
        caution_abstains=bool(p["caution_abstains"]),
        threshold_source=str(p["alert_threshold_source"]),
        margin_evidence=cfg.get("margin_evidence", {}),
        provenance=cfg.get("provenance", {}),
    )


# =============================================================
# SELECTIVE-RISK METRICS
# =============================================================

def selective_metrics(decisions, y_true) -> dict:
    """Coverage and selective risk for a set of decisions.

    decisions  sequence of Decision (or of decision strings)
    y_true     0/1 ground truth, same length and order

    Every rate among non-abstained windows is computed on the RETAINED
    subset only, which is what "selective" means. Counts are reported
    alongside every rate so a rate over a tiny subset cannot be mistaken
    for a stable one.
    """
    d = [x.decision if isinstance(x, Decision) else str(x) for x in decisions]
    y = [int(v) for v in y_true]
    if len(d) != len(y):
        raise AbstainPolicyError(
            f"{len(d)} decisions vs {len(y)} labels — cannot align")

    n = len(d)
    n_alert = sum(v == ALERT for v in d)
    n_benign = sum(v == BENIGN for v in d)
    n_abstain = sum(v == ABSTAIN for v in d)

    tp = sum(1 for i in range(n) if d[i] == ALERT and y[i] == 1)
    fp = sum(1 for i in range(n) if d[i] == ALERT and y[i] == 0)
    tn = sum(1 for i in range(n) if d[i] == BENIGN and y[i] == 0)
    fn = sum(1 for i in range(n) if d[i] == BENIGN and y[i] == 1)
    kept = tp + fp + tn + fn

    def rate(num, den):
        return (num / den) if den else None

    return {
        "total_samples": n,
        "alert_count": n_alert,
        "benign_count": n_benign,
        "abstain_count": n_abstain,
        "coverage": rate(kept, n),
        "abstention_rate": rate(n_abstain, n),
        "n_non_abstained": kept,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "selective_error_rate": rate(fp + fn, kept),
        "selective_fpr": rate(fp, fp + tn),
        "selective_fnr": rate(fn, fn + tp),
        "selective_precision": rate(tp, tp + fp),
        "selective_recall": rate(tp, tp + fn),
        "selective_f1": rate(2 * tp, 2 * tp + fp + fn),
        "n_negatives_retained": fp + tn,
        "n_positives_retained": tp + fn,
        "retained_negative_share": rate(fp + tn, kept),
        "positives_abstained": sum(1 for i in range(n)
                                   if d[i] == ABSTAIN and y[i] == 1),
        "negatives_abstained": sum(1 for i in range(n)
                                   if d[i] == ABSTAIN and y[i] == 0),
    }


def baseline_metrics(scores, y_true, threshold) -> dict:
    """Plain frozen thresholding — the no-abstain comparison point.

    Reported in the SAME schema as selective_metrics so the two tables can
    be read side by side without re-deriving anything.
    """
    d = [ALERT if float(s) > threshold else BENIGN for s in scores]
    m = selective_metrics(d, y_true)
    m["policy"] = "baseline_frozen_thresholding_no_abstain"
    return m


def coverage_risk_curve(scores, y_true, reliability, threshold,
                        deltas, caution_abstains) -> list[dict]:
    """Coverage vs selective risk at a range of margin half-widths.

    Used for the required "coverage vs selective risk at several fixed
    abstention levels" table. Every row carries the retained-negative
    share, because that is what exposes an apparent risk reduction that is
    really a class-imbalance artefact.
    """
    rows = []
    for delta in deltas:
        pol = AbstainPolicy(
            version=LAYER_VERSION, alert_threshold=float(threshold),
            margin_delta=float(delta), caution_abstains=caution_abstains,
            threshold_source="sweep")
        ds = [pol.decide(max_forecast=s, reliability_status=r)
              for s, r in zip(scores, reliability)]
        m = selective_metrics(ds, y_true)
        m["margin_delta"] = float(delta)
        m["caution_abstains"] = caution_abstains
        rows.append(m)
    return rows
