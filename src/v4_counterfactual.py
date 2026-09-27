"""
=============================================================
V4 MULTI-INTERVENTION COUNTERFACTUAL LAYER
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Task 19 / S5. Generalises the validated Task-18 / S4 Defend-Now simulator
from ONE counterfactual (no action vs defend now) into a MULTI-SCENARIO
decision-support layer:

    NO ACTION   vs   RATE LIMIT   vs   RESTRICT EGRESS
    at three predeclared simulated strengths

and reports, for each scenario, the frozen model's projected

    threat probability trajectory     (k = 0 .. 5)
    ATT&CK progression hypothesis     (Task 16, unchanged)
    estimated time-to-progression     (Task 16, unchanged)
    reliability status                (Task 15, unchanged)
    forecast / rollout stability      (Task 15 indicators, unchanged)

WHAT THIS IS
------------
A MODEL-BASED WHAT-IF SIMULATOR. Each scenario is a defined perturbation of
the CURRENT observed state, rolled forward through the FROZEN V4 checkpoint.
The analyst compares the resulting projections and decides. Nothing here
decides, and nothing here acts.

WHAT THIS IS NOT
----------------
- NOT a causal effect estimator. V4 was never shown an intervention, never
  given an action variable, and never trained on (action, outcome) pairs, so
  it cannot have learned intervention effects and this layer does not claim
  it did.
- NOT a real defense engine. Nothing here touches a socket, a device or a
  configuration.
- NOT an automated response system. The abstention layer (Task 17) is never
  overridden: where automated action is withheld, it stays withheld, and the
  counterfactual view is offered for analyst review only.
- NOT proof that an intervention would work in reality.

RELATIONSHIP TO S4 — EXTENSION, NOT DUPLICATION
-----------------------------------------------
Every state transformation, every intervention definition, every strength
constant and the whole counterfactual-state builder are IMPORTED from
src/v4_defense_simulator.py and used unchanged. This module adds only:

    * a declarative scenario set (NO_ACTION + the cross product of the two
      S4 interventions with the three S4 strengths),
    * the multi-scenario comparison, the categorical progression-change
      taxonomy, and the scenario-wide summary,
    * the counterfactual-consistency checks required by the task.

S4's own artifacts in results/defense_simulation/ are never read for
mutation and never rewritten.

THE NO_ACTION BASELINE IS THE UNPERTURBED WINDOW
------------------------------------------------
NO_ACTION rolls the OBSERVED tensor forward verbatim — it does NOT take the
zero-strength round trip through the inverse-scale / re-scale path. That is
deliberate: it keeps the baseline bit-identical to what every other page of
the console shows, and it turns "zero strength == NO_ACTION" into a real
test of the round trip rather than a tautology.

IDENTICAL INITIAL CONDITIONS
----------------------------
Inherited from S4 and re-verified here for every scenario: timesteps 0..8 of
both halves of the 52-D window are copied verbatim from the observed tensor,
so every scenario for a given sequence shares byte-identical observed
history. Only the final base row and the final relative-change row differ,
and only by the intervention's own transformation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

import attack_path_predictor as ap
import v4_defense_simulator as ds
import v4_reliability as rly

LAYER_VERSION = "v4-counterfactual-1.0"

MAX_K = ds.MAX_K
WINDOW_SECONDS = ds.WINDOW_SECONDS
SEQ_LEN = ds.SEQ_LEN
N_FEATURES = ds.N_FEATURES
N_BASE = ds.N_BASE

# Honest wording contract for this layer. The dashboard and every artifact
# use these verbatim.
HEADLINE = "COUNTERFACTUAL MODEL SIMULATION"
DISCLAIMER_CAUSAL = (
    "These trajectories are model-based what-if projections. They do not "
    "estimate causal real-world defense effectiveness because the World "
    "Model was not trained on intervention/outcome data."
)
DISCLAIMER_DIRECTION = (
    "Different scenarios may produce higher or lower projected threat "
    "probabilities. This reflects the frozen model's response to the "
    "simulated state perturbation and should not be interpreted as proof "
    "that a defense would help or harm in reality."
)
DELTA_NAME = "projected change under scenario"
DELTA_IS_NEVER_CALLED = "risk reduction caused by scenario"
PROGRESSION_CHANGE_NAME = "model-projected progression change"
PROGRESSION_IS_NEVER_CALLED = "attack delayed by N seconds"

ABSTAIN_COEXISTENCE = (
    "Counterfactual exploration is available for analyst review, "
    "but automated action is withheld."
)
UNRELIABLE_BANNER = "COUNTERFACTUAL FORECAST UNRELIABLE"

WORKFLOW = ("OBSERVE", "FORECAST", "EXPLORE COUNTERFACTUAL", "COMPARE",
            "ANALYST DECIDES")


class CounterfactualError(RuntimeError):
    """Raised when a frozen precondition for this layer does not hold."""


# =============================================================
# 1. THE SCENARIO MODEL
# =============================================================
#
# Strengths are PREDECLARED here, as module constants, before any
# evaluation. They are the same three constants S4 froze; this module
# asserts that identity at import time rather than restating them by hand,
# so the two layers cannot drift apart.

DECLARED_STRENGTHS = tuple(ds.SWEEP_STRENGTHS)          # (0.10, 0.25, 0.50)
if DECLARED_STRENGTHS != (0.10, 0.25, 0.50):
    raise CounterfactualError(
        f"declared strengths diverged from the frozen S4 set: "
        f"{DECLARED_STRENGTHS}")

NO_ACTION = "NO_ACTION"
INTERVENTION_NONE = "NONE"

# The intervention ORDER is fixed so scenario_id numbering is stable.
INTERVENTION_ORDER = ("RATE_LIMIT", "RESTRICT_EGRESS")
if tuple(sorted(ds.INTERVENTIONS)) != tuple(sorted(INTERVENTION_ORDER)):
    raise CounterfactualError(
        f"S4 exposes interventions {sorted(ds.INTERVENTIONS)}, this layer "
        f"declares {sorted(INTERVENTION_ORDER)}")


@dataclass(frozen=True)
class ScenarioSpec:
    """One counterfactual scenario, fully reproducible from its fields.

    A scenario applied to a sequence is reproducible from exactly
    (sequence_id, intervention_type, strength) — scenario_id and
    scenario_name are display keys derived from those two.
    """
    scenario_id: str
    scenario_name: str
    intervention_type: str
    strength: float
    description: str

    @property
    def is_no_action(self) -> bool:
        return self.intervention_type == INTERVENTION_NONE

    @property
    def strength_pct(self) -> str:
        return f"{int(round(self.strength * 100))}%"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_no_action"] = self.is_no_action
        return d


def scenario_key(intervention_type: str, strength: float) -> str:
    """The stable id for a (intervention_type, strength) pair."""
    if intervention_type == INTERVENTION_NONE:
        return NO_ACTION
    return f"{intervention_type}@{float(strength):.2f}"


def _build_scenarios() -> tuple[ScenarioSpec, ...]:
    out = [ScenarioSpec(
        scenario_id=NO_ACTION,
        scenario_name="No action",
        intervention_type=INTERVENTION_NONE,
        strength=0.0,
        description=(
            "The observed window is rolled forward through the frozen V4 "
            "model with no perturbation of any kind. This is the frozen "
            "forecast every other page of the console shows, and it is the "
            "reference every scenario delta is measured against."),
    )]
    for key in INTERVENTION_ORDER:
        iv = ds.INTERVENTIONS[key]
        for s in DECLARED_STRENGTHS:
            out.append(ScenarioSpec(
                scenario_id=scenario_key(key, s),
                scenario_name=f"{iv.display} — {int(round(s * 100))}% "
                              f"simulated strength",
                intervention_type=key,
                strength=float(s),
                description=(
                    f"{iv.concept} Applied to the FINAL observed timestep "
                    f"only, at a simulated intervention strength of "
                    f"{s:.2f}. The transformation is the one S4 validated, "
                    f"imported unchanged."),
            ))
    return tuple(out)


SCENARIOS: tuple[ScenarioSpec, ...] = _build_scenarios()
SCENARIO_BY_ID = {s.scenario_id: s for s in SCENARIOS}
BASELINE_SCENARIO = SCENARIO_BY_ID[NO_ACTION]

# Scenario ids grouped by intervention, for the matrix and the UI.
SCENARIO_IDS_BY_INTERVENTION = {
    key: tuple(scenario_key(key, s) for s in DECLARED_STRENGTHS)
    for key in INTERVENTION_ORDER
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    if scenario_id not in SCENARIO_BY_ID:
        raise CounterfactualError(
            f"unknown scenario {scenario_id!r}; available: "
            f"{list(SCENARIO_BY_ID)}")
    return SCENARIO_BY_ID[scenario_id]


# =============================================================
# 2. THE THIRD INTERVENTION — CONSIDERED, MEASURED, DECLINED
# =============================================================
#
# The task admits an INTERVENTION C only if the EXISTING feature contract
# supports a distinct, defensible mathematical mapping, and forbids
# inventing a perturbation merely to raise the scenario count. The audit is
# recorded here so the absence of a third intervention is a documented
# decision rather than an omission.
#
# The 26 base features fall into five groups:
#
#   volume          flow_count / log_flow_count / log_total_packets /
#                   window_packet_rate / log_total_bytes / window_byte_rate
#                   -> already the RATE_LIMIT axis.
#   destination     log_unique_dst_ports / dst_port_entropy
#   diversity       -> already the RESTRICT_EGRESS axis.
#   protocol        unique_protocols / protocol_entropy
#   diversity       -> the ONLY remaining diversity axis. See below.
#   flag rates      syn/ack/rst/fin/psh/urg — per-flow statistics averaged
#                   over flows. S4 already established that no signed change
#                   follows from the feature definition for any of them.
#   shape / timing  mean_pkt_len, std_pkt_len, the three log-duration / IAT
#   / ratios        features, the four direction ratios, mean_down_up_ratio
#                   — invariant to uniform scaling, or with a sign the
#                   contract does not determine.
#
# PROTOCOL_ALLOWLIST would mirror RESTRICT_EGRESS exactly, on the protocol
# axis: unique_protocols * (1 - s), protocol_entropy rescaled by the ratio
# of log2 support ceilings, plus the coupled volume reduction. The
# mathematics is defensible. The SUPPORT is not: measured on the frozen
# substrate, unique_protocols takes only the values {1, 2, 3}, with 3 in
# 86.9% of the 31 150 observed windows. At the declared strengths a
# "fraction of protocols blocked" is not representable as a whole protocol
# — 10% of 3 protocols is 0.3 of a protocol — and at 50% the entropy
# rescaling saturates against the log2(max(n, 2)) floor. The parameter would
# therefore not mean what its name says, and the scenario would be a
# near-degenerate copy of RATE_LIMIT with a misleading label.
#
# Declined. The substrate measurement that decides it is recomputed by the
# builder and written to scenario_definitions.json, so the decision is
# re-checked on every run rather than asserted once.
THIRD_INTERVENTION_AUDIT = {
    "candidate": "PROTOCOL_ALLOWLIST",
    "concept": "block a fraction of the distinct protocols observed in the "
               "current window, as a service allow-list would",
    "would_modify": ["unique_protocols", "protocol_entropy",
                     "+ the coupled volume chain"],
    "mathematically_definable": True,
    "admitted": False,
    "reason": (
        "unique_protocols has support {1, 2, 3} on the frozen substrate, so "
        "a fractional 'fraction of protocols blocked' does not correspond to "
        "a whole protocol at any declared strength, and the entropy "
        "rescaling saturates against the log2(max(n,2)) floor at strength "
        "0.50. The parameter would not mean what its name says and the "
        "scenario would be a near-degenerate relabelling of RATE_LIMIT."),
    "rejected_alternatives": {
        "flag-rate interventions (syn/ack/rst/fin/psh/urg)":
            "per-flow statistics averaged over flows; S4 established that no "
            "signed change follows from the feature definition",
        "timing interventions (log_mean_fwd_iat, log_mean_flow_duration)":
            "a rate limit plausibly raises inter-arrival time, but neither "
            "sign nor magnitude follows from the contract",
        "direction-ratio interventions":
            "ratios invariant to uniform volume scaling; fwd+bwd packet "
            "ratios are additionally pinned to 1.0 by a measured identity",
        "packet-shape interventions (mean_pkt_len, std_pkt_len)":
            "means of per-flow means; unchanged in expectation under uniform "
            "volume reduction",
    },
    "consequence": (
        "this layer ships two interventions and one no-action baseline, at "
        "three predeclared strengths — seven scenarios in total"),
}


# =============================================================
# 3. PROGRESSION-CHANGE TAXONOMY  (categorical, never numeric)
# =============================================================
#
# Task 16 is used UNCHANGED. This module only COMPARES two of its outputs.
# Stage identity is compared categorically, exactly as the task requires; no
# stage is ever subtracted from another stage.

PC_DISAPPEARS = "progression_disappears"
PC_APPEARS = "progression_appears"
PC_LATER = "progression_shifts_later"
PC_EARLIER = "progression_shifts_earlier"
PC_SAME = "same_progression"
PC_STAGE_CHANGED = "progression_target_stage_changed"
PC_AMBIGUOUS = "ambiguous"
PC_NONE = "no_progression_identified"

PROGRESSION_CHANGES = (
    PC_DISAPPEARS, PC_APPEARS, PC_LATER, PC_EARLIER, PC_SAME,
    PC_STAGE_CHANGED, PC_AMBIGUOUS, PC_NONE,
)

PROGRESSION_CHANGE_DISPLAY = {
    PC_DISAPPEARS: "Projected progression disappears under this scenario",
    PC_APPEARS: "Projected progression appears under this scenario",
    PC_LATER: "Projected progression shifts later under this scenario",
    PC_EARLIER: "Projected progression shifts earlier under this scenario",
    PC_SAME: "Same projected progression",
    PC_STAGE_CHANGED: "Same projected timing, different projected next stage",
    PC_AMBIGUOUS: "Ambiguous — analyst review",
    PC_NONE: "No progression identified in either scenario",
}

# PC_STAGE_CHANGED is an EXTENSION of the seven categories the task lists.
# It exists because the task also requires stage transitions to be compared
# categorically: two hypotheses that reach DIFFERENT stages at the same
# horizon are not "the same progression", and collapsing them into it would
# hide exactly the comparison the task asks for.
PROGRESSION_CHANGE_NOTES = {
    "categories_from_task": [PC_DISAPPEARS, PC_APPEARS, PC_LATER, PC_EARLIER,
                             PC_SAME, PC_AMBIGUOUS, PC_NONE],
    "extension": {
        PC_STAGE_CHANGED: (
            "added because stage transitions are compared categorically: a "
            "hypothesis reaching a DIFFERENT next stage at the same horizon "
            "is not the same progression")},
    "stage_comparison": "categorical string equality on the ATT&CK stage "
                        "name — never a numeric difference of stage indices",
    "precedence": (
        "when BOTH the horizon and the stage move, the TIMING category wins "
        "(shifts_later / shifts_earlier) and the stage move is disclosed "
        "separately in next_stage_comparison. No row hides a stage change: "
        "every row carries both fields."),
}

STAGE_SAME = "same_stage"
STAGE_CHANGED = "stage_changed"
STAGE_NA = "not_applicable"


def classify_progression_change(base_status: str,
                                base_stage: str | None,
                                base_ttp: int | None,
                                cf_status: str,
                                cf_stage: str | None,
                                cf_ttp: int | None) -> tuple[str, str]:
    """Categorical comparison of two Task-16 hypotheses.

    Returns (progression_change, next_stage_comparison). Neither output is
    a causal statement: it describes how the MODEL'S PROJECTION moved when
    the input state was perturbed.
    """
    amb = ap.STATUS_AMBIGUOUS
    fwd = ap.STATUS_PROGRESSION

    base_has = (base_status == fwd) and base_stage is not None
    cf_has = (cf_status == fwd) and cf_stage is not None

    if base_has and cf_has:
        stage_cmp = STAGE_SAME if base_stage == cf_stage else STAGE_CHANGED
    else:
        stage_cmp = STAGE_NA

    if base_status == amb or cf_status == amb:
        return PC_AMBIGUOUS, stage_cmp
    if base_has and not cf_has:
        return PC_DISAPPEARS, stage_cmp
    if cf_has and not base_has:
        return PC_APPEARS, stage_cmp
    if not base_has and not cf_has:
        return PC_NONE, stage_cmp

    if base_ttp is None or cf_ttp is None:
        return PC_AMBIGUOUS, stage_cmp
    if int(cf_ttp) > int(base_ttp):
        return PC_LATER, stage_cmp
    if int(cf_ttp) < int(base_ttp):
        return PC_EARLIER, stage_cmp
    return (PC_SAME if stage_cmp == STAGE_SAME else PC_STAGE_CHANGED,
            stage_cmp)


# =============================================================
# 4. STABILITY  (Task-15 indicators, reused unchanged)
# =============================================================
#
# No new stability metric is invented. Forecast stability and rollout
# stability are read straight off the Task-15 Assessment, which already
# computes them as its forecast_consistency and rollout_stability indicator
# groups, and already tiers them against the frozen reference distribution.

FORECAST_STABILITY_INDICATORS = ("prob_max_abs_step", "prob_range")
ROLLOUT_STABILITY_INDICATORS = ("latent_norm_max", "latent_norm_change",
                                "step_delta_max", "displacement")
TIER_RANK = {rly.TIER_OK: 0, rly.TIER_ELEVATED: 1, rly.TIER_BEYOND: 2}
RANK_TIER = {v: k for k, v in TIER_RANK.items()}


def stability_block(a: rly.Assessment | None, names: tuple) -> dict:
    """Indicator values plus the worst Task-15 tier among them."""
    if a is None:
        return {"tier": "", **{n: None for n in names}}
    worst = max((TIER_RANK.get(a.tiers.get(n, rly.TIER_OK), 0) for n in names),
                default=0)
    out = {"tier": RANK_TIER[worst]}
    out.update({n: float(a.indicators[n]) for n in names})
    return out


# =============================================================
# 5. SCENARIO OUTCOME
# =============================================================

@dataclass
class ScenarioOutcome:
    """One scenario evaluated on one sequence, plus its delta vs NO_ACTION."""
    sequence_id: int
    timestamp: str
    scenario: ScenarioSpec
    probs: np.ndarray                      # (6,) k = 0..5
    alert_threshold: float

    reliability_status: str = ""
    reliability_headline: str = ""
    reliability_indicators_changed: tuple = ()
    forecast_stability: dict = field(default_factory=dict)
    rollout_stability: dict = field(default_factory=dict)

    progression_status: str = ""
    current_stage: str = ""
    next_stage: str | None = None
    time_to_progression_s: int | None = None

    changed_features: tuple = ()
    changed_feature_table: list = field(default_factory=list)

    # ---- comparison against NO_ACTION (filled by the engine) ----
    deltas: np.ndarray | None = None       # (6,)
    delta_max_forecast: float = 0.0
    delta_time_to_progression_s: int | None = None
    progression_change: str = ""
    next_stage_comparison: str = ""
    reliability_changed: bool = False

    @property
    def nowcast(self) -> float:
        return float(self.probs[0])

    @property
    def max_forecast(self) -> float:
        return float(self.probs[1:].max())

    @property
    def alert(self) -> int:
        return int(self.max_forecast > self.alert_threshold)

    @property
    def mean_delta_k5(self) -> float:
        return 0.0 if self.deltas is None else float(self.deltas[1:].mean())

    @property
    def direction(self) -> str:
        """Descriptive only. 'lower' / 'higher' refer to the MODEL'S
        PROJECTED probability, never to real-world risk."""
        d = self.delta_max_forecast
        if abs(d) < NEGLIGIBLE:
            return "unchanged"
        return "lower" if d < 0 else "higher"

    def to_row(self) -> dict:
        s = self.scenario
        r = {
            "sequence_id": self.sequence_id,
            "timestamp": self.timestamp,
            "scenario_id": s.scenario_id,
            "scenario_name": s.scenario_name,
            "intervention_type": s.intervention_type,
            "strength": s.strength,
            "nowcast_probability": self.nowcast,
            "max_forecast": self.max_forecast,
            "alert_threshold": self.alert_threshold,
            "alert": self.alert,
            "reliability_status": self.reliability_status,
            "reliability_changed_vs_no_action": int(self.reliability_changed),
            "reliability_indicators_changed":
                "|".join(self.reliability_indicators_changed),
            "forecast_stability_tier": self.forecast_stability.get("tier", ""),
            "prob_max_abs_step":
                self.forecast_stability.get("prob_max_abs_step"),
            "prob_range": self.forecast_stability.get("prob_range"),
            "rollout_stability_tier": self.rollout_stability.get("tier", ""),
            "latent_step_delta_max":
                self.rollout_stability.get("step_delta_max"),
            "latent_displacement": self.rollout_stability.get("displacement"),
            "progression_status": self.progression_status,
            "current_stage": self.current_stage,
            "next_projected_stage": self.next_stage or "",
            "time_to_progression_s": ("" if self.time_to_progression_s is None
                                      else self.time_to_progression_s),
            "projected_progression_change": self.progression_change,
            "next_stage_comparison": self.next_stage_comparison,
            "projected_direction": self.direction,
            "delta_max_forecast": self.delta_max_forecast,
            "mean_projected_change_k5": self.mean_delta_k5,
            "delta_time_to_progression_s": (
                "" if self.delta_time_to_progression_s is None
                else self.delta_time_to_progression_s),
            "changed_features": "|".join(self.changed_features),
        }
        for k in range(MAX_K + 1):
            r[f"prob_t{k}"] = float(self.probs[k])
            r[f"delta_t{k}"] = (0.0 if self.deltas is None
                                else float(self.deltas[k]))
        return r


# Below this absolute probability change a scenario is reported as
# "unchanged / negligible" rather than as a direction. 1e-6 is the same
# threshold S4 used, kept so the two layers agree.
NEGLIGIBLE = 1e-6


# =============================================================
# 6. THE ENGINE
# =============================================================

class CounterfactualEngine:
    """Frozen V4 + frozen scaler + the S4 perturbations, over a scenario set.

    Nothing is trained, fitted, refit or written. The model runs in eval()
    under torch.no_grad(), exactly as every other layer in this project uses
    it. No optimiser and no gradient step exists anywhere in this class.
    """

    def __init__(self, model, alert_threshold: float,
                 reference: rly.Reference | None = None,
                 predictor: ap.AttackPathPredictor | None = None,
                 scaler: ds.FrozenScaler | None = None):
        # The S4 simulator IS the state builder and the rollout. Reused, not
        # reimplemented, so the two layers cannot diverge.
        self.sim = ds.DefenseSimulator(model, alert_threshold=alert_threshold,
                                       scaler=scaler)
        self.alert_threshold = float(alert_threshold)
        self.reference = reference
        self.predictor = predictor

    # ---------------------------------------------------------

    def state_for(self, x_window: np.ndarray, spec: ScenarioSpec):
        """The (10, 52) scaled state a scenario is rolled forward from.

        NO_ACTION returns the OBSERVED tensor verbatim. Every intervention
        returns the S4 counterfactual state, whose timesteps 0..8 are copied
        from that same observed tensor.

        Returns (state, CounterfactualState | None).
        """
        x = np.asarray(x_window, dtype=np.float64)
        if x.ndim == 3:
            if x.shape[0] != 1:
                raise CounterfactualError(f"expected one window, got {x.shape}")
            x = x[0]
        if x.shape != (SEQ_LEN, N_FEATURES):
            raise CounterfactualError(
                f"x_window has shape {x.shape}, expected "
                f"({SEQ_LEN}, {N_FEATURES})")
        if spec.is_no_action:
            return np.array(x, dtype=np.float64, copy=True), None
        cfs = ds.build_counterfactual(x, spec.intervention_type, spec.strength,
                                      self.sim.scaler)
        return cfs.counterfactual_state, cfs

    # ---------------------------------------------------------

    def evaluate(self, sequence_id: int, x_window: np.ndarray,
                 spec: ScenarioSpec, timestamps: list,
                 total_windows: int, first_alert_idx: int | None,
                 stored_probs: np.ndarray | None = None) -> ScenarioOutcome:
        """Roll ONE scenario forward and attach Tasks 15 and 16, unchanged.

        `stored_probs` may be supplied for NO_ACTION to display the FROZEN
        stored probabilities instead of the recomputed ones; the recomputed
        rollout is still what feeds the reliability and attack-path layers,
        so those two layers always see a self-consistent trajectory.
        """
        state, cfs = self.state_for(x_window, spec)
        probs, latents, states = self.sim.rollout(state)

        shown = probs
        if spec.is_no_action and stored_probs is not None:
            shown = np.asarray(stored_probs, dtype=np.float64).ravel()

        out = ScenarioOutcome(
            sequence_id=int(sequence_id),
            timestamp=str(timestamps[1] if len(timestamps) > 1 else ""),
            scenario=spec,
            probs=shown,
            alert_threshold=self.alert_threshold,
        )
        if cfs is not None:
            out.changed_features = tuple(
                ds.fc.BASE_FEATURES[i] for i in cfs.changed_feature_indices)
            out.changed_feature_table = [
                r for r in cfs.raw_feature_table() if r["modified"]]

        assessment = None
        if self.reference is not None:
            assessment = rly.assess(state[None], probs, latents, states,
                                    self.reference)
            out.reliability_status = assessment.status
            out.reliability_headline = assessment.headline
        out.forecast_stability = stability_block(
            assessment, FORECAST_STABILITY_INDICATORS)
        out.rollout_stability = stability_block(
            assessment, ROLLOUT_STABILITY_INDICATORS)

        if self.predictor is not None:
            hyp = self.predictor.predict(
                sequence_id=int(sequence_id),
                x_window=state,
                decoded_states=states,
                probs=probs,
                timestamps=[str(t) for t in timestamps],
                total_windows=int(total_windows),
                first_alert_idx=first_alert_idx,
                max_forecast_probability=float(probs[1:].max()),
                alert=int(float(probs[1:].max()) > self.alert_threshold),
                reliability=assessment)
            out.progression_status = hyp.status
            out.current_stage = hyp.current_stage
            if hyp.next_step is not None:
                out.next_stage = hyp.next_step.to_stage
                out.time_to_progression_s = int(
                    hyp.next_step.time_to_progression_s)

        return out, assessment

    # ---------------------------------------------------------

    def compare(self, sequence_id: int, x_window: np.ndarray,
                timestamps: list, total_windows: int,
                first_alert_idx: int | None,
                stored_probs: np.ndarray | None = None,
                scenarios: tuple | None = None) -> dict:
        """Evaluate EVERY scenario on ONE sequence from identical history.

        Returns {"outcomes": {scenario_id: ScenarioOutcome},
                 "assessments": {scenario_id: Assessment | None},
                 "states": {scenario_id: (10, 52) ndarray},
                 "baseline": ScenarioOutcome}
        """
        specs = scenarios or SCENARIOS
        if specs[0].scenario_id != NO_ACTION:
            raise CounterfactualError(
                "the scenario set must begin with NO_ACTION so every delta "
                "has its reference")

        outcomes, assessments, states = {}, {}, {}
        for spec in specs:
            state, _cfs = self.state_for(x_window, spec)
            oc, a = self.evaluate(sequence_id, x_window, spec, timestamps,
                                  total_windows, first_alert_idx, stored_probs)
            outcomes[spec.scenario_id] = oc
            assessments[spec.scenario_id] = a
            states[spec.scenario_id] = state

        base = outcomes[NO_ACTION]
        base_a = assessments[NO_ACTION]
        base.deltas = np.zeros(MAX_K + 1, dtype=np.float64)
        base.progression_change = PC_SAME if base.progression_status == \
            ap.STATUS_PROGRESSION else (
                PC_AMBIGUOUS if base.progression_status == ap.STATUS_AMBIGUOUS
                else PC_NONE)
        base.next_stage_comparison = STAGE_NA

        for sid, oc in outcomes.items():
            if sid == NO_ACTION:
                continue
            oc.deltas = oc.probs - base.probs
            oc.delta_max_forecast = oc.max_forecast - base.max_forecast
            if (oc.time_to_progression_s is not None
                    and base.time_to_progression_s is not None):
                oc.delta_time_to_progression_s = (
                    oc.time_to_progression_s - base.time_to_progression_s)
            oc.progression_change, oc.next_stage_comparison = \
                classify_progression_change(
                    base.progression_status, base.next_stage,
                    base.time_to_progression_s,
                    oc.progression_status, oc.next_stage,
                    oc.time_to_progression_s)
            oc.reliability_changed = (
                oc.reliability_status != base.reliability_status)
            oc.reliability_indicators_changed = changed_indicators(
                base_a, assessments[sid])

        return {"outcomes": outcomes, "assessments": assessments,
                "states": states, "baseline": base}


def changed_indicators(base_a, cf_a) -> tuple:
    """Task-15 indicator names whose TIER moved between two assessments."""
    if base_a is None or cf_a is None:
        return ()
    return tuple(n for n in rly.TIERED_INDICATORS
                 if base_a.tiers.get(n) != cf_a.tiers.get(n))


# =============================================================
# 7. COUNTERFACTUAL CONSISTENCY  (the task's tests A .. E)
# =============================================================

HISTORY_TIMESTEPS = SEQ_LEN - 1          # 0 .. 8 are observed history


def verify_identical_history(states: dict) -> dict:
    """C. Every scenario shares byte-identical observed history 0..8.

    Also confirms that the states DO differ at the final timestep for every
    non-zero-strength intervention, so the check cannot pass vacuously.
    """
    ids = list(states)
    ref = states[NO_ACTION][:HISTORY_TIMESTEPS]
    mismatched, final_identical = [], []
    for sid in ids:
        if not np.array_equal(states[sid][:HISTORY_TIMESTEPS], ref):
            mismatched.append(sid)
        if sid != NO_ACTION and np.array_equal(states[sid][-1],
                                               states[NO_ACTION][-1]):
            final_identical.append(sid)
    return {
        "n_scenarios": len(ids),
        "history_timesteps_compared": HISTORY_TIMESTEPS,
        "scenarios_with_mismatched_history": mismatched,
        "identical_history": not mismatched,
        "scenarios_whose_final_state_did_not_move": final_identical,
        "check_is_non_vacuous": not final_identical,
    }


def verify_zero_strength(engine: CounterfactualEngine, x_window: np.ndarray):
    """B. A zero-strength intervention reproduces NO_ACTION.

    Returns {intervention: max |P_zero - P_no_action|}. The residual is the
    float32 round trip through inverse-scale -> relative-change -> re-scale,
    which NO_ACTION does not take; it is a measurement of that round trip,
    not a tolerance that was chosen to pass.
    """
    base_state, _ = engine.state_for(x_window, BASELINE_SCENARIO)
    p_base, _z, _s = engine.sim.rollout(base_state)
    out = {}
    for key in INTERVENTION_ORDER:
        spec = ScenarioSpec(scenario_id=scenario_key(key, 0.0),
                            scenario_name=f"{key} zero strength",
                            intervention_type=key, strength=0.0,
                            description="zero-strength equivalence probe")
        state, _cfs = engine.state_for(x_window, spec)
        p, _z2, _s2 = engine.sim.rollout(state)
        out[key] = float(np.abs(p - p_base).max())
    return out


def scenario_fingerprint(rows: list) -> str:
    """A / D. Stable hash of a scenario result set, for the determinism and
    repeatability checks."""
    import hashlib
    payload = json.dumps(
        [{k: (round(v, 10) if isinstance(v, float) else v)
          for k, v in sorted(r.items())} for r in rows],
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


# =============================================================
# 8. SCENARIO-WIDE SUMMARY
# =============================================================

def summarize_scenario(rows) -> dict:
    """Descriptive summary for ONE scenario across sequences.

    `rows` is an iterable of ScenarioOutcome.to_row() dicts for a single
    scenario_id. NO DIRECTION IS CALLED SUCCESSFUL: the summary counts
    lower, higher and unchanged separately and ranks nothing.
    """
    rows = list(rows)
    if not rows:
        return {}
    mx = np.array([r["max_forecast"] for r in rows], dtype=np.float64)
    dmax = np.array([r["delta_max_forecast"] for r in rows], dtype=np.float64)
    dmean = np.array([r["mean_projected_change_k5"] for r in rows],
                     dtype=np.float64)
    dk = np.array([[r[f"delta_t{k}"] for k in range(1, MAX_K + 1)]
                   for r in rows], dtype=np.float64)
    alert = np.array([r["alert"] for r in rows], dtype=int)
    thr = float(rows[0]["alert_threshold"])

    return {
        "scenario_id": rows[0]["scenario_id"],
        "intervention_type": rows[0]["intervention_type"],
        "strength": rows[0]["strength"],
        "n_sequences": len(rows),
        "median_max_probability": float(np.median(mx)),
        "mean_max_probability": float(mx.mean()),
        "p95_max_probability": float(np.percentile(mx, 95)),
        "global_max_probability": float(mx.max()),
        "mean_probability_delta_k5": float(dk.mean()),
        "median_probability_delta_k5": float(np.median(dmean)),
        "p95_absolute_delta": float(np.percentile(np.abs(dk), 95)),
        "mean_delta_max_forecast": float(dmax.mean()),
        "median_delta_max_forecast": float(np.median(dmax)),
        "n_projected_lower": int((dmax < -NEGLIGIBLE).sum()),
        "n_projected_higher": int((dmax > NEGLIGIBLE).sum()),
        "n_unchanged_or_negligible": int((np.abs(dmax) <= NEGLIGIBLE).sum()),
        "alert_threshold": thr,
        "n_alerting": int(alert.sum()),
        "negligible_band": NEGLIGIBLE,
    }


def threshold_crossings(base_rows: dict, rows) -> dict:
    """Sequences crossing / leaving the frozen threshold vs NO_ACTION.

    CROSSING  = NO_ACTION below or at the frozen threshold, scenario above.
    LEAVING   = NO_ACTION above the frozen threshold, scenario at or below.
    """
    crossing, leaving = 0, 0
    for r in rows:
        b = base_rows.get(r["sequence_id"])
        if b is None:
            continue
        if not b["alert"] and r["alert"]:
            crossing += 1
        elif b["alert"] and not r["alert"]:
            leaving += 1
    return {"n_crossing_frozen_threshold": crossing,
            "n_leaving_frozen_threshold": leaving}


def progression_counts(rows) -> dict:
    """Count of each categorical progression-change outcome."""
    counts = {c: 0 for c in PROGRESSION_CHANGES}
    for r in rows:
        c = r.get("projected_progression_change", "")
        if c in counts:
            counts[c] += 1
    return counts


def reliability_counts(rows) -> dict:
    counts = {s: 0 for s in rly.STATUS_ORDER}
    for r in rows:
        s = r.get("reliability_status", "")
        if s in counts:
            counts[s] += 1
    return counts


# =============================================================
# 9. DEMONSTRATION-CASE SELECTION  (deterministic, label-free)
# =============================================================
#
# Every rule below reads MODEL OUTPUT and TASK-15 / TASK-16 STATUS only. No
# ground-truth label, no event annotation and no attack ratio is reachable
# from any of them. Case C is selected on the MAGNITUDE of disagreement
# between scenarios, which is direction-agnostic: it cannot prefer a
# scenario that lowers the projected probability over one that raises it.

DEMO_CASES = {
    "A_elevated_threat": (
        "highest NO_ACTION max-forecast probability among the swept "
        "sequences; ties broken by lowest sequence_id"),
    "B_forward_progression": (
        "lowest sequence_id among swept sequences whose NO_ACTION Task-16 "
        "status is FORWARD_PROGRESSION_IDENTIFIED"),
    "C_scenarios_diverge": (
        "largest spread across scenarios of the projected max-forecast "
        "probability, max(scenario) - min(scenario). Direction-agnostic: it "
        "measures how much the scenarios DISAGREE, not which way any of "
        "them moves"),
    "D_unreliable_or_caution": (
        "lowest sequence_id among swept sequences whose NO_ACTION Task-15 "
        "status is UNRELIABLE; if none, the lowest whose status is CAUTION"),
    "E_progression_change": (
        "lowest sequence_id among swept sequences where at least one "
        "scenario's categorical progression-change outcome is not "
        "'same_progression' and not 'no_progression_identified'"),
    "F_progression_timing_shift": (
        "lowest sequence_id among swept sequences where at least one "
        "scenario shifts the estimated time-to-progression. "
        "Direction-agnostic: 'shifts later' and 'shifts earlier' are both "
        "admitted, so the rule cannot prefer the flattering direction"),
}


def select_demo_cases(per_sequence: dict) -> dict:
    """Apply the deterministic rules above.

    `per_sequence` maps sequence_id -> {scenario_id: row dict}. Returns
    {case: (sequence_id, basis)}; a case absent from the substrate is simply
    not returned, and the caller must say so rather than substitute one.
    """
    ids = sorted(per_sequence)
    if not ids:
        return {}
    chosen = {}

    def base(i):
        return per_sequence[i][NO_ACTION]

    # A — elevated threat.
    a = sorted(ids, key=lambda i: (-base(i)["max_forecast"], i))[0]
    chosen["A_elevated_threat"] = (a, DEMO_CASES["A_elevated_threat"])

    # B — an existing forward progression hypothesis under NO ACTION.
    b = [i for i in ids
         if base(i)["progression_status"] == ap.STATUS_PROGRESSION]
    if b:
        chosen["B_forward_progression"] = (b[0],
                                           DEMO_CASES["B_forward_progression"])

    # C — scenarios diverge most (direction-agnostic spread).
    def spread(i):
        v = [r["max_forecast"] for r in per_sequence[i].values()]
        return max(v) - min(v)

    c = sorted(ids, key=lambda i: (-spread(i), i))[0]
    chosen["C_scenarios_diverge"] = (c, DEMO_CASES["C_scenarios_diverge"])

    # D — unreliable, else caution.
    d = [i for i in ids if base(i)["reliability_status"] == rly.UNRELIABLE]
    if not d:
        d = [i for i in ids if base(i)["reliability_status"] == rly.CAUTION]
    if d:
        chosen["D_unreliable_or_caution"] = (
            d[0], DEMO_CASES["D_unreliable_or_caution"])

    # E — a visible categorical progression change.
    interesting = {PC_DISAPPEARS, PC_APPEARS, PC_LATER, PC_EARLIER,
                   PC_STAGE_CHANGED}
    e = [i for i in ids
         if any(r["projected_progression_change"] in interesting
                for sid, r in per_sequence[i].items() if sid != NO_ACTION)]
    if e:
        chosen["E_progression_change"] = (e[0],
                                          DEMO_CASES["E_progression_change"])

    # F — a shift in the estimated time-to-progression, either direction.
    timing = {PC_LATER, PC_EARLIER}
    f = [i for i in ids
         if any(r["projected_progression_change"] in timing
                for sid, r in per_sequence[i].items() if sid != NO_ACTION)]
    if f:
        chosen["F_progression_timing_shift"] = (
            f[0], DEMO_CASES["F_progression_timing_shift"])
    return chosen


# =============================================================
# 10. LAYER METADATA
# =============================================================

def layer_metadata() -> dict:
    return {
        "layer_version": LAYER_VERSION,
        "module": "src/v4_counterfactual.py",
        "task": "Task 19 / S5 — multi-intervention counterfactual defense "
                "simulator",
        "built_on": {
            "src/v4_defense_simulator.py": ds.LAYER_VERSION,
            "src/v4_reliability.py": rly.LAYER_VERSION,
            "src/attack_path_predictor.py": ap.LAYER_VERSION,
        },
        "extends_not_duplicates": (
            "every state transformation, intervention definition, strength "
            "constant and the counterfactual-state builder are imported from "
            "src/v4_defense_simulator.py and used unchanged. This module adds "
            "the scenario set, the multi-scenario comparison, the categorical "
            "progression taxonomy and the consistency checks."),
        "headline": HEADLINE,
        "disclaimers": {
            "causal": DISCLAIMER_CAUSAL,
            "direction": DISCLAIMER_DIRECTION,
        },
        "wording": {
            "delta_is_called": DELTA_NAME,
            "delta_is_never_called": DELTA_IS_NEVER_CALLED,
            "progression_delta_is_called": PROGRESSION_CHANGE_NAME,
            "progression_delta_is_never_called": PROGRESSION_IS_NEVER_CALLED,
            "abstain_coexistence": ABSTAIN_COEXISTENCE,
            "unreliable_banner": UNRELIABLE_BANNER,
        },
        "analyst_workflow": list(WORKFLOW),
        "declared_strengths": list(DECLARED_STRENGTHS),
        "strengths_declared_before_evaluation": True,
        "strengths_source": "src/v4_defense_simulator.py SWEEP_STRENGTHS, "
                            "asserted identical at import",
        "n_scenarios": len(SCENARIOS),
        "scenarios": [s.to_dict() for s in SCENARIOS],
        "reproducibility_key": ["sequence_id", "intervention_type", "strength"],
        "baseline_definition": (
            "NO_ACTION rolls the OBSERVED tensor forward verbatim. It does "
            "NOT take the zero-strength round trip, so 'zero strength == "
            "NO_ACTION' is a real test of that round trip rather than a "
            "tautology."),
        "identical_initial_conditions": {
            "timesteps_shared": f"0..{HISTORY_TIMESTEPS - 1}",
            "verified_per_sequence": True,
            "only_final_timestep_perturbed": True,
            "relative_change_row_0": (
                "preserved from the observed tensor, never regenerated — "
                "these are sliding windows, so row 0 references the window "
                "before the sequence begins"),
        },
        "third_intervention_audit": THIRD_INTERVENTION_AUDIT,
        "progression_change_taxonomy": {
            "categories": list(PROGRESSION_CHANGES),
            "display": dict(PROGRESSION_CHANGE_DISPLAY),
            "notes": PROGRESSION_CHANGE_NOTES,
        },
        "stability_definition": {
            "source": "src/v4_reliability.py indicators, reused unchanged",
            "forecast_stability": list(FORECAST_STABILITY_INDICATORS),
            "rollout_stability": list(ROLLOUT_STABILITY_INDICATORS),
            "new_metric_invented": False,
        },
        "demonstration_case_rules": dict(DEMO_CASES),
        "guarantees": {
            "retrains_v4": False,
            "modifies_v4_checkpoint": False,
            "refits_scalers": False,
            "modifies_thresholds": False,
            "modifies_screening_predictions": False,
            "changes_feature_definitions": False,
            "modifies_task15_logic": False,
            "modifies_task16_logic": False,
            "modifies_task17_logic": False,
            "modifies_s4_intervention_mathematics": False,
            "overwrites_s4_artifacts": False,
            "executes_network_actions": False,
            "claims_causal_efficacy": False,
            "uses_labels_to_tune_interventions": False,
            "selects_strength_after_inspecting_outcomes": False,
            "overrides_abstention": False,
            "ranks_or_scores_scenarios": False,
            "deterministic": True,
        },
        "limitations": [
            "No output of this layer is a causal effect. The model was never "
            "trained on interventions, so its response to a state "
            "perturbation is a property of the model, not of a defense.",
            "Only the FINAL observed timestep is perturbed, because earlier "
            "timesteps are observed history. The V4 encoder is an LSTM over "
            "ten windows, so projected changes are small by construction.",
            "Projected changes are heterogeneous in sign across sequences. A "
            "lower projection is not mitigation and a higher projection is "
            "not evidence that defending would be harmful.",
            "Interventions modify only features whose transformation follows "
            "from the frozen contract; plausible real consequences the "
            "contract cannot determine are deliberately not modelled, so "
            "every perturbation is incomplete by design.",
            "RESTRICT_EGRESS assumes blocked destinations carry a "
            "proportional share of window volume — a first-order assumption, "
            "not a measured property.",
            "Only two interventions are defensible on the frozen 26-feature "
            "contract; see third_intervention_audit.",
            "The scenario matrix is descriptive. It carries no overall score "
            "and no ranking, because the model cannot support one.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(layer_metadata(), indent=2, default=str))
