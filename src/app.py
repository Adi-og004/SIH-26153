"""
=============================================================
CYBER WORLD MODEL — STREAMLIT DASHBOARD
=============================================================

Interactive demo for Smart India Hackathon (SIH) 2026.

Ties together all components:
- V4 Latent World Model predictions (gated transition)
- MITRE ATT&CK mapping
- Integrated Gradients feature attribution
- K-step forward rollout

Run: streamlit run src/app.py

=============================================================
"""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

# Canonical PCAP -> V4 inference pipeline (single source of truth for
# the 📥 Data Input page's PCAP source). streamlit runs this file from src/,
# so a bare module import resolves.
import pcap_ingest_v4 as pcap_v4

# Prepared feature data loader (CSV/NPY -> V4 input). Keeps file
# parsing and validation separate from dashboard rendering.
import prepared_input_loader as pil

# Reused so an uploaded PCAP's results can drive every page (MITRE
# mapping + feature attribution), not just the ingestion page itself.
from mitre_mapper import MITREMapper

# The provenance-locked V4 attribution path (Task 8). This is the ONLY
# sanctioned explainer for V4: it refuses to construct unless the
# checkpoint, input scaler and feature order all hash-match
# docs/v4_baseline_manifest.json. src/shap_explainer.py is deliberately
# NOT imported here -- it is unlocked, attributes a single transition
# step, and draws an unseeded random baseline, so the explanation it
# produces is neither repeatable nor guaranteed to match the displayed
# prediction. See docs/V4_EXPLAINABILITY.md.
from v4_explainer import V4Explainer

# The validated Task-10 screening replay substrate (results/v4_screening_mvp).
# This is the demo's authoritative data path: frozen V4 probabilities over the
# real held-out March-2 day. It is READ-ONLY and has no synthetic fallback.
import screening_data as sd

# The Task-15 reliability / model-health layer (results/v4_reliability). A
# decision-support layer ON TOP of the frozen V4 model: it never changes a
# forecast, a threshold or a scaler, and the status it reports is rule-based,
# NOT a calibrated probability. See src/v4_reliability.py.
import v4_reliability as rly

# The Task-16 ATT&CK-consistent progression hypothesis layer. It composes the
# FROZEN V4 rollout with the EXISTING, UNMODIFIED rule library in
# src/mitre_mapper.py. It trains nothing and predicts no ATT&CK label: every
# tactic and technique string it shows comes verbatim out of that rule
# library. See src/attack_path_predictor.py.
import attack_path_predictor as apx

# The Task-17 human-in-the-loop abstain layer. A deterministic three-way
# analyst decision (ALERT / BENIGN / ABSTAIN) over the frozen forecast, the
# frozen 0.69 boundary and the Task-15 reliability status. It never changes a
# forecast and never moves the boundary; ABSTAIN means "the model declines to
# decide automatically", NOT "the model has no signal".
import v4_abstain as abs_layer

# The Task-18 defend-now simulator. A MODEL-BASED WHAT-IF PROJECTION: it
# perturbs the current observed state along explicitly-justified feature
# semantics and rolls the FROZEN world model forward. It performs no network
# action, estimates no causal effect, and never overwrites the observed state.
import v4_defense_simulator as dsim

# The Task-19 multi-intervention counterfactual layer. It EXTENDS the Task-18
# simulator -- every state transformation and strength constant is imported
# from it unchanged -- into a multi-scenario comparison. It is a MODEL-BASED
# WHAT-IF SIMULATOR: not a causal effect estimator, not a real defense engine,
# not an automated response system.
import v4_counterfactual as cfl

# The unified data-input layer (one page, one validated representation, one
# inference call for the benchmark, an uploaded PCAP and uploaded prepared
# V4 data). It owns no model weights and creates no second inference path.
import data_input as dinp


# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="Cyber World Model — Predictive Screening Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# CUSTOM CSS
# =============================================================

st.markdown("""
<style>
    :root {
        --surface:   #131722;
        --surface-2: #1b2130;
        --border:    #2a3346;
        --text:      #e6e9ef;
        --muted:     #8c96ab;
        --now:       #38bdf8;   /* OBSERVED / NOWCAST */
        --forecast:  #f5a524;   /* FORECAST          */
        --alert:     #ef4444;
        --ok:        #22c55e;
    }

    html, body, [class*="css"] { font-feature-settings: "tnum" 1; }

    /* Compact analyst cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
    }
    .card-label {
        font-size: 0.72rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.15;
        color: var(--text);
    }
    .card-sub { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }

    .card-now      { border-left: 3px solid var(--now); }
    .card-forecast { border-left: 3px solid var(--forecast); }
    .card-alert    { border-left: 3px solid var(--alert); }
    .card-ok       { border-left: 3px solid var(--ok); }

    .v-alert  { color: var(--alert);    font-weight: 700; }
    .v-ok     { color: var(--ok);       font-weight: 700; }
    .v-fore   { color: var(--forecast); font-weight: 700; }
    .v-now    { color: var(--now);      font-weight: 700; }

    /* NOW vs FORECAST pills */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .pill-now      { background: rgba(56,189,248,.14); color: var(--now);
                     border: 1px solid rgba(56,189,248,.35); }
    .pill-forecast { background: rgba(245,165,36,.14); color: var(--forecast);
                     border: 1px solid rgba(245,165,36,.35); }
    .pill-context  { background: rgba(140,150,171,.12); color: var(--muted);
                     border: 1px solid rgba(140,150,171,.3); }

    .tactic-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }

    .banner {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0 14px 0;
        border: 1px solid var(--border);
        background: var(--surface);
    }
    .banner-alert { border-left: 4px solid var(--alert);
                    background: linear-gradient(90deg, rgba(239,68,68,.10), transparent 60%); }
    .banner-ok    { border-left: 4px solid var(--ok);
                    background: linear-gradient(90deg, rgba(34,197,94,.10), transparent 60%); }
    .banner-gap   { border-left: 4px solid var(--muted);
                    background: repeating-linear-gradient(45deg,
                        rgba(140,150,171,.07) 0 10px, transparent 10px 20px); }

    .note {
        font-size: 0.78rem;
        color: var(--muted);
        border-left: 2px solid var(--border);
        padding-left: 10px;
        margin-top: 6px;
    }

    .demo-footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.76rem;
        border-top: 1px solid var(--border);
        margin-top: 28px;
        padding-top: 10px;
    }

    section[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# Shared Plotly styling so every chart in the console reads as one system.
PLOT_BG = "#131722"
GRID = "#2a3346"
C_NOW = "#38bdf8"        # observed / nowcast
C_FORECAST = "#f5a524"   # world-model forecast
C_ALERT = "#ef4444"
C_MUTED = "#8c96ab"


def style_fig(fig, height=None, legend=True):
    """Apply the console's shared chart styling."""
    # The horizontal legend sits just above the plot area, and Plotly draws the
    # title into that same top margin -- at t=36 the two collide and both become
    # unreadable. A titled chart with a legend gets room for the title band as well.
    has_title = bool(getattr(fig.layout.title, "text", None))
    stacked = has_title and legend
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="#e6e9ef", size=12),
        margin=dict(l=8, r=8, t=68 if stacked else 36, b=8),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    if stacked:
        fig.update_layout(title=dict(yref="container", y=1.0, yanchor="top",
                                     pad=dict(t=12, l=4)))
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def card(label, value, sub="", kind=""):
    """Compact analyst metric card."""
    cls = f"card {kind}" if kind else "card"
    sub_html = f'<div class="card-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="{cls}"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


# =============================================================
# PATHS
# =============================================================

# The FROZEN V4 reference benchmark -- the same arrays the reference
# evaluation and the Task 8 explainability run consume. The dashboard
# previously read data/processed/temporal_benchmark_v2 (the older
# single-day split) while calling itself V4; that made the displayed
# prediction a different quantity from the verified one.
DATA_DIR = Path("data/processed/multiday_benchmark_K5")
MODEL_DIR = Path("models/latent_world_model_v4")
MITRE_DIR = MODEL_DIR / "mitre_mapping"
# Verified Task 8 attribution artifacts. Replaces MODEL_DIR /
# "shap_analysis", which was produced by the unlocked legacy explainer.
XAI_DIR = Path("results/v4_explainability")
ROLLOUT_DIR = MODEL_DIR / "k_step_rollout"

# The horizons the dashboard may display. Exactly the horizons Task 8
# verified end to end; each has a frozen validation threshold and a
# stored, hash-pinned attribution tensor.
VERIFIED_HORIZONS = [1, 3, 5]
DEFAULT_HORIZON = 5


# =============================================================
# V4 MODEL
# =============================================================
#
# The architecture is NOT redefined here. A second copy of the class
# in the dashboard is a contamination vector: it can drift from the
# trained architecture while still loading the state dict, and it
# makes "which class produced this number?" unanswerable. The model
# comes from the canonical loader in src/pcap_ingest_v4.py, reached
# through V4Explainer, which hash-verifies the checkpoint against
# docs/v4_baseline_manifest.json before returning it.

# =============================================================
# FEATURE NAMES
# =============================================================

# Feature names are READ FROM THE FROZEN SCHEMA, never retyped here.
# Attribution column i must carry schema name i (Part 22); a second
# hand-maintained copy of the list can drift silently and mislabel
# every attribution on the page.
def _load_canonical_feature_names() -> list:
    names = [
        ln.strip()
        for ln in (DATA_DIR / "input_feature_names.txt")
        .read_text().splitlines()
        if ln.strip()
    ]
    schema_path = Path("docs/v4_feature_schema.json")
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        schema_names = [f["name"] for f in schema["features"]]
        if schema_names != names:
            raise RuntimeError(
                "docs/v4_feature_schema.json and "
                f"{DATA_DIR}/input_feature_names.txt disagree -- "
                "refusing to label attributions with an ambiguous "
                "feature order"
            )
    if len(names) != 52:
        raise RuntimeError(f"expected 52 feature names, got {len(names)}")
    return names


INPUT_FEATURES = _load_canonical_feature_names()


# =============================================================
# SCREENING SUBSTRATE (the demo's authoritative data path)
# =============================================================

@st.cache_resource
def load_screening():
    """The validated Task-10 screening replay substrate.

    Returns (substrate, error). On failure the error string is surfaced in
    the UI -- there is deliberately NO synthetic fallback."""
    try:
        return sd.load(), None
    except sd.ScreeningDataError as exc:
        return None, str(exc)
    except Exception as exc:                     # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


# The HELD-OUT reference results, read from the frozen baseline manifest.
# NOT checkpoint["test_attack_f1"]/["test_roc_auc"] (0.297 / 0.671): those are
# training-time numbers from the single-day temporal_benchmark_v2 split and the
# manifest explicitly records that they are not the reference results.
@st.cache_data
def load_reference_results():
    path = Path("docs/v4_baseline_manifest.json")
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("results_verified")


@st.cache_resource
def load_target_scaler():
    """The frozen 26-D target scaler, used ONLY to invert the decoder's
    output into physical units for display. Never refit."""
    import pickle
    with open(DATA_DIR / "target_scaler.pkl", "rb") as fh:
        return pickle.load(fh)


@st.cache_data(show_spinner=False)
def rollout_detail(seq_id: int):
    """Frozen V4 K=5 rollout for one screening sequence.

    Returns (probs[6], latents[6,32], decoded_states[6,26]). The ONLY input
    is the 52-D feature tensor -- no label or event annotation is reachable."""
    _model, _ckpt, expl = load_model()
    x = SCREENING.model_input(seq_id)
    xb = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        tz, tp, ts = expl.model.k_step_rollout(xb, sd.MAX_K)
    probs = torch.stack(tp, dim=1).cpu().numpy()[0]
    latents = torch.stack(tz, dim=1).cpu().numpy()[0]
    states = torch.stack(ts, dim=1).cpu().numpy()[0]
    return probs, latents, states


def threat_color(p, thr):
    return C_ALERT if p >= thr else (C_FORECAST if p >= 0.5 * thr else C_NOW)


# =============================================================
# RELIABILITY LAYER  (Task 15)
# =============================================================
# The reliability status is computed from the frozen model's OWN input and
# rollout against a fixed reference distribution built from the train split.
# No label, no future observation and no fitted estimator is involved, and
# the word "confidence" is deliberately avoided throughout.

RLY_COLOR = {"RELIABLE": "#22c55e", "CAUTION": "#f5a524", "UNRELIABLE": "#ef4444"}
RLY_CARD = {"RELIABLE": "card-ok", "CAUTION": "card-forecast", "UNRELIABLE": "card-alert"}
RLY_BANNER = {"RELIABLE": "banner-ok", "CAUTION": "banner-gap", "UNRELIABLE": "banner-alert"}


@st.cache_resource
def load_reliability():
    """The frozen Task-15 reliability reference. Returns (reference, error).

    There is deliberately no fallback: a status without its reference
    distribution would be a colour with no meaning."""
    try:
        return rly.load_reference(), None
    except rly.ReliabilityConfigError as exc:
        return None, str(exc)
    except Exception as exc:                     # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def reliability_of(seq_id: int):
    """Reliability assessment for one screening sequence, or None if the
    reference is unavailable. Inputs are the 52-D feature tensor and the
    frozen rollout only."""
    ref, _err = load_reliability()
    if ref is None:
        return None
    probs, latents, states = rollout_detail(seq_id)
    return rly.assess(SCREENING.model_input(seq_id), probs, latents, states, ref)


def reliability_chip(a, note=""):
    """Compact inline status chip. Wording is HIGH / CAUTION / LOW -- never
    "confidence", "certainty" or "probability of correctness"."""
    c = RLY_COLOR[a.status]
    return (
        f'<span class="pill" style="background:{c}22;color:{c};'
        f'border:1px solid {c}59;">Forecast reliability: {a.short}</span>'
        + (f'&nbsp;<span class="note">{note}</span>' if note else "")
    )


# =============================================================
# ATTACK-PATH / PROGRESSION LAYER  (Task 16)
# =============================================================
# A rule-based INTERPRETATION of the frozen forecast. Nothing here is a
# learned prediction of attacker intent, and no ATT&CK classifier exists.

APX_BANNER = (
    "ATT&CK progression is a rule-based model interpretation. "
    "It is not a learned prediction of attacker intent."
)

APX_STAGE_COLOR = {
    apx.RECON: "#38bdf8",
    apx.INITIAL_ACCESS: "#a78bfa",
    apx.EXECUTION: "#f59e0b",
    apx.LATERAL: "#fb7185",
    apx.C2: "#f472b6",
    apx.EXFIL: "#ef4444",
    apx.IMPACT: "#dc2626",
    apx.NO_ALERT: "#64748b",
    apx.UNKNOWN: "#94a3b8",
}
APX_CONF_COLOR = {"HIGH": "#22c55e", "MEDIUM": "#f5a524", "LOW": "#94a3b8"}


@st.cache_resource
def load_attack_path_predictor():
    """The Task-16 predictor, sharing the SAME frozen model instance the
    displayed forecast came from. Returns (predictor, error)."""
    try:
        _m, _ck, expl = load_model()
        return apx.AttackPathPredictor(
            expl.model, alert_threshold=SCREENING.alert_threshold), None
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def attack_path_of(seq_id: int):
    """ATT&CK-consistent progression hypothesis for one screening sequence.

    Inputs are the frozen 52-D window, the frozen V4 rollout and the frozen
    stored probabilities. No label, no event annotation, no future
    observation is reachable from here."""
    predictor, err = load_attack_path_predictor()
    if predictor is None:
        return None, err
    try:
        sub = SCREENING
        probs, _latents, dec_states = rollout_detail(seq_id)
        row = sub.preds.iloc[seq_id]
        t1 = pd.to_datetime(row["target_t1_timestamp"])
        stamps = [str(pd.to_datetime(row["observed_window_end"]))] + [
            str(t1 + pd.Timedelta(seconds=sd.WINDOW_SECONDS * (k - 1)))
            for k in range(1, sd.MAX_K + 1)]
        # The stored frozen probabilities are authoritative for display.
        stored = sub.probs[seq_id]
        first_alert = int(np.argmax(sub.alerts)) if sub.alerts.any() else None
        return predictor.predict(
            sequence_id=seq_id,
            x_window=sub.model_input(seq_id),
            decoded_states=dec_states,
            probs=stored,
            timestamps=stamps,
            total_windows=sub.n_sequences,
            first_alert_idx=first_alert,
            max_forecast_probability=float(row["max_forecast_probability"]),
            alert=int(row["alert"]),
            reliability=reliability_of(seq_id),
        ), None
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


def apx_stage_chip(stage, extra=""):
    c = APX_STAGE_COLOR.get(stage, "#7f8c8d")
    return (f'<span class="pill" style="background:{c}22;color:{c};'
            f'border:1px solid {c}59;">{stage}</span>'
            + (f'&nbsp;<span class="note">{extra}</span>' if extra else ""))


def apx_timing_text(h, step):
    """Time-to-progression wording. Withheld when the Task-15 reliability
    status is UNRELIABLE, so a precise-looking number is never shown
    without its warning."""
    if h.reliability_suppresses_timing:
        return ("timing withheld — forecast reliability LOW", True)
    return (f"+{step.time_to_progression_s} s", False)


# =============================================================
# ANALYST DECISION LAYER  (Task 17)
# =============================================================

ABS_COLOR = {"ALERT": "#ef4444", "BENIGN": "#22c55e", "ABSTAIN": "#f5a524"}
ABS_CARD = {"ALERT": "card-alert", "BENIGN": "card-ok",
            "ABSTAIN": "card-forecast"}
ABS_BANNER = {"ALERT": "banner-alert", "BENIGN": "banner-ok",
              "ABSTAIN": "banner-gap"}
ABS_ICON = {"ALERT": "🚨", "BENIGN": "✅", "ABSTAIN": "⛔"}


@st.cache_resource
def load_abstain_policy():
    """The frozen Task-17 decision policy. Returns (policy, error).

    No fallback: a decision layer that invents its own gate when the frozen
    one is missing would be worse than none."""
    try:
        return abs_layer.load_policy(), None
    except abs_layer.AbstainPolicyError as exc:
        return None, str(exc)
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def analyst_decision_of(seq_id: int):
    """Three-way analyst decision for one screening sequence.

    Composes the frozen stored forecast, the Task-15 reliability status and
    the Task-16 attack-path status (rationale only). Returns (decision,
    error)."""
    policy, err = load_abstain_policy()
    if policy is None:
        return None, err
    try:
        rel = reliability_of(seq_id)
        if rel is None:
            return None, ("the Task-15 reliability status is unavailable, and "
                          "this layer must not decide without it")
        hyp, _aperr = attack_path_of(seq_id)
        return policy.decide(
            max_forecast=float(SCREENING.max_forecast[seq_id]),
            reliability_status=rel.status,
            nowcast=float(SCREENING.probs[seq_id, 0]),
            attack_path_status=(hyp.status if hyp is not None else None),
        ), None
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


def abs_chip(d):
    c = ABS_COLOR[d.decision]
    return (f'<span class="pill" style="background:{c}22;color:{c};'
            f'border:1px solid {c}59;">Analyst decision: {d.decision}</span>')


# =============================================================
# DEFEND-NOW SIMULATOR  (Task 18)
# =============================================================

DSIM_DISCLAIMER = dsim.WORDING_REQUIRED[0]
DSIM_BODY = dsim.WORDING_REQUIRED[1]


@st.cache_resource
def load_defense_simulator():
    """The Task-18 simulator, sharing the SAME frozen model instance the
    displayed forecast came from. Returns (simulator, error)."""
    try:
        _m, _ck, expl = load_model()
        return dsim.DefenseSimulator(
            expl.model, alert_threshold=SCREENING.alert_threshold), None
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def defense_scenario(seq_id: int, intervention: str, strength: float):
    """No-action vs defend-now for one sequence.

    The no-action baseline uses the FROZEN stored probabilities, so the left
    half of the comparison is exactly what every other page shows. Returns
    (scenario_row, raw_feature_table, extras, error)."""
    sim, err = load_defense_simulator()
    if sim is None:
        return None, None, None, err
    try:
        sub = SCREENING
        sc, cfs, detail = sim.simulate(
            sequence_id=seq_id, x_window=sub.model_input(seq_id),
            intervention=intervention, strength=strength,
            timestamp=str(sub.preds["target_t1_timestamp"].iloc[seq_id]),
            no_action_probs=sub.probs[seq_id])

        ref, _e = load_reliability()
        extras = {"no_action_reliability": None, "defend_now_reliability": None,
                  "no_action_path": None, "defend_now_path": None}
        if ref is not None:
            a_b = rly.assess(cfs.baseline_state[None],
                             detail["baseline_recomputed_probs"],
                             detail["baseline_latents"],
                             detail["baseline_states"], ref)
            a_c = rly.assess(cfs.counterfactual_state[None],
                             sc.defend_now_probs,
                             detail["counterfactual_latents"],
                             detail["counterfactual_states"], ref)
            sc.no_action_reliability = a_b.status
            sc.defend_now_reliability = a_c.status
            extras["no_action_reliability"] = a_b
            extras["defend_now_reliability"] = a_c

        # Task-16 attack path on BOTH scenarios, using the same frozen mapper.
        try:
            apred = load_attack_path_predictor()[0]
            if apred is not None:
                row = sub.preds.iloc[seq_id]
                t1 = pd.to_datetime(row["target_t1_timestamp"])
                stamps = [str(pd.to_datetime(row["observed_window_end"]))] + [
                    str(t1 + pd.Timedelta(seconds=sd.WINDOW_SECONDS * (k - 1)))
                    for k in range(1, sd.MAX_K + 1)]
                fa = int(np.argmax(sub.alerts)) if sub.alerts.any() else None
                h_b = apred.predict(
                    sequence_id=seq_id, x_window=cfs.baseline_state,
                    decoded_states=detail["baseline_states"],
                    probs=detail["baseline_recomputed_probs"],
                    timestamps=stamps, total_windows=sub.n_sequences,
                    first_alert_idx=fa,
                    max_forecast_probability=sc.no_action_max_future,
                    alert=sc.no_action_alert)
                h_c = apred.predict(
                    sequence_id=seq_id, x_window=cfs.counterfactual_state,
                    decoded_states=detail["counterfactual_states"],
                    probs=sc.defend_now_probs, timestamps=stamps,
                    total_windows=sub.n_sequences, first_alert_idx=fa,
                    max_forecast_probability=sc.defend_now_max_future,
                    alert=sc.defend_now_alert)
                sc.no_action_attack_path = h_b.status
                sc.defend_now_attack_path = h_c.status
                extras["no_action_path"] = h_b
                extras["defend_now_path"] = h_c
        except Exception:                            # noqa: BLE001
            pass

        return sc, cfs.raw_feature_table(), extras, None
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, None, None, f"{type(exc).__name__}: {exc}"


# =============================================================
# MULTI-INTERVENTION COUNTERFACTUAL LAYER  (Task 19 / S5)
# =============================================================
# Extends the Task-18 simulator into a multi-scenario what-if comparison.
# Every transformation is S4's, imported unchanged. Nothing on the page acts.

CFL_HEADLINE = cfl.HEADLINE
CFL_CAUSAL = cfl.DISCLAIMER_CAUSAL
CFL_DIRECTION = cfl.DISCLAIMER_DIRECTION

# One colour per scenario, shared by every chart and table on the page so a
# line, a row and a chip always mean the same scenario.
CFL_COLOR = {
    cfl.NO_ACTION: "#e2e8f0",
    "RATE_LIMIT@0.10": "#bae6fd",
    "RATE_LIMIT@0.25": "#38bdf8",
    "RATE_LIMIT@0.50": "#0284c7",
    "RESTRICT_EGRESS@0.10": "#ddd6fe",
    "RESTRICT_EGRESS@0.25": "#a78bfa",
    "RESTRICT_EGRESS@0.50": "#7c3aed",
}

CFL_PC_STYLE = {
    cfl.PC_DISAPPEARS: ("#22c55e", "▽"),
    cfl.PC_APPEARS: ("#ef4444", "△"),
    cfl.PC_LATER: ("#38bdf8", "→"),
    cfl.PC_EARLIER: ("#f5a524", "←"),
    cfl.PC_SAME: ("#94a3b8", "="),
    cfl.PC_STAGE_CHANGED: ("#a78bfa", "≠"),
    cfl.PC_AMBIGUOUS: ("#f5a524", "?"),
    cfl.PC_NONE: ("#64748b", "·"),
}


@st.cache_resource
def load_counterfactual_engine():
    """The Task-19 engine, sharing the SAME frozen model instance the
    displayed forecast came from. Returns (engine, error)."""
    try:
        _m, _ck, expl = load_model()
        ref, _re = load_reliability()
        pred, _pe = load_attack_path_predictor()
        return cfl.CounterfactualEngine(
            expl.model, alert_threshold=SCREENING.alert_threshold,
            reference=ref, predictor=pred), None
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def counterfactual_compare(seq_id: int):
    """Every declared scenario for one sequence, from identical history.

    The NO_ACTION column uses the FROZEN stored probabilities, so the
    baseline on this page is exactly what every other page shows. Returns
    (result dict, history-verification dict, error)."""
    engine, err = load_counterfactual_engine()
    if engine is None:
        return None, None, err
    try:
        sub = SCREENING
        row = sub.preds.iloc[seq_id]
        t1 = pd.to_datetime(row["target_t1_timestamp"])
        stamps = [str(pd.to_datetime(row["observed_window_end"]))] + [
            str(t1 + pd.Timedelta(seconds=sd.WINDOW_SECONDS * (k - 1)))
            for k in range(1, sd.MAX_K + 1)]
        fa = int(np.argmax(sub.alerts)) if sub.alerts.any() else None
        res = engine.compare(
            sequence_id=seq_id, x_window=sub.model_input(seq_id),
            timestamps=stamps, total_windows=sub.n_sequences,
            first_alert_idx=fa, stored_probs=sub.probs[seq_id])
        return res, cfl.verify_identical_history(res["states"]), None
    except Exception as exc:                        # noqa: BLE001 - shown to user
        return None, None, f"{type(exc).__name__}: {exc}"


@st.cache_data(show_spinner=False)
def counterfactual_summary():
    """The substrate-wide scenario matrix written by
    src/build_counterfactual_defense.py, or None if it has not been built."""
    path = Path("results/counterfactual_defense/summary.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:                                # noqa: BLE001
        return None


def cfl_pc_chip(change: str) -> str:
    """Chip for a categorical progression-change outcome."""
    colour, glyph = CFL_PC_STYLE.get(change, ("#94a3b8", "·"))
    return (f'<span class="pill" style="background:{colour}22;color:{colour};'
            f'border:1px solid {colour}59;">{glyph}&nbsp;'
            f'{cfl.PROGRESSION_CHANGE_DISPLAY.get(change, change)}</span>')


def cfl_workflow_strip(active: str = "") -> str:
    """The analyst workflow, rendered so the page can never read as
    OBSERVE -> MODEL -> AUTOMATICALLY DEFEND."""
    parts = []
    for step in cfl.WORKFLOW:
        on = (step == active)
        parts.append(
            f'<span class="pill" style="'
            f'background:{"#38bdf822" if on else "#ffffff0d"};'
            f'color:{"#38bdf8" if on else "#8c96ab"};'
            f'border:1px solid {"#38bdf859" if on else "#ffffff1a"};">'
            f'{step}</span>')
    return "&nbsp;<span style='color:#4b5565;'>→</span>&nbsp;".join(parts)


def gap_banner(sub, seq_id):
    """Render a DATA GAP notice when the selected sequence sits at a real
    capture discontinuity. Nothing is interpolated across it."""
    for g in sub.capture_gaps():
        if seq_id in (g["last_sequence_id"], g["break_before_sequence_id"]):
            side = ("This is the LAST sequence before"
                    if seq_id == g["last_sequence_id"]
                    else "This is the FIRST sequence after")
            st.markdown(
                f'<div class="banner banner-gap">'
                f'<b>⚠ DATA GAP — raw capture discontinuity</b><br>'
                f'{side} a <b>{g["gap_seconds"] / 3600:.2f} h</b> gap in the '
                f'original CSE-CIC-IDS2018 capture '
                f'({g["from_timestamp"]} → {g["to_timestamp"]}).<br>'
                f'<span class="note">No windows exist in this interval. Nothing is '
                f'interpolated and no future windows are synthesised; replay '
                f'continues from the next valid segment.</span></div>',
                unsafe_allow_html=True,
            )
            return True
    return False


# =============================================================
# DATA LOADING (cached)
# =============================================================

TACTIC_COLORS = {
    "Reconnaissance": "#e74c3c",
    "Initial Access": "#e67e22",
    "Execution": "#95a5a6",
    "Command and Control": "#9b59b6",
    "Lateral Movement": "#3498db",
    "Exfiltration": "#2ecc71",
    "Impact": "#1abc9c",
    "Unknown / Insufficient Evidence": "#7f8c8d",
    "No Alert": "#2c3e50",
}

CONFIDENCE_COLORS = {
    "high": "#e74c3c",
    "medium": "#e67e22",
    "low": "#f1c40f",
    "n/a": "#34495e",
}


@st.cache_data
def load_test_data():
    X = np.load(DATA_DIR / "X_test.npy")
    meta = pd.read_csv(DATA_DIR / "test_metadata.csv")
    meta["timestamp"] = pd.to_datetime(
        meta["timestamp"]
    )
    return X, meta


@st.cache_data
def load_mitre_mappings(expected_rows: int):
    """Rule-based ATT&CK mappings, one row per benchmark window.

    The table is addressed POSITIONALLY (row i describes window i),
    so it is only valid for the dataset it was generated against. If
    the row count does not match the active benchmark, it is refused:
    rendering a stale row next to window i would attach the wrong
    ATT&CK interpretation to a prediction, which is worse than
    showing none.
    """
    csv_path = MITRE_DIR / "mitre_summary.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if len(df) != expected_rows:
        return None
    return df


@st.cache_data
def load_xai_summary():
    """The Task 8 run summary. Carries the hashes the displayed
    attribution tensors are pinned to."""
    path = Path("results/v4_explainability_summary.json")
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data
def load_feature_importance(horizon: int):
    """Global feature importance from the VERIFIED run, for one
    horizon. Columns are renamed to the names the pages render."""
    csv_path = XAI_DIR / "horizon_feature_importance.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["horizon"] == horizon].copy()
    if df.empty:
        return None
    df = df.rename(
        columns={
            "mean_absolute_attribution": "mean_abs_attribution",
            "mean_signed_attribution": "mean_signed_attribution",
        }
    )
    return df.sort_values(
        "mean_abs_attribution", ascending=False
    ).reset_index(drop=True)


@st.cache_data
def load_attributions(horizon: int):
    """Per-window attribution tensor (N, 10, 52) for one horizon,
    produced by the verified Task 8 run.

    The file is hash-pinned in results/v4_explainability_summary.json.
    If the hash does not match, the tensor is REFUSED rather than
    displayed: an attribution from an unidentified run must never be
    shown next to a prediction.
    """
    npy_path = XAI_DIR / f"attributions_k{horizon}.npy"
    if not npy_path.exists():
        return None
    summary = load_xai_summary()
    if summary is not None:
        pinned = (
            summary.get("dashboard_artifacts", {})
            .get("attribution_tensors", {})
            .get(str(horizon), {})
        )
        expected = pinned.get("sha256")
        if expected:
            import hashlib
            h = hashlib.sha256()
            with open(npy_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                return None
    return np.load(npy_path)


@st.cache_resource
def load_model():
    """The frozen V4 model, reached through the provenance-locked
    explainer.

    V4Explainer hash-verifies the checkpoint, the input scaler and the
    feature order against docs/v4_baseline_manifest.json and raises
    before returning anything if any of them has drifted. The SAME
    instance serves the displayed prediction and the explanation, so
    the two cannot diverge.
    """
    explainer = V4Explainer(device="cpu")
    return explainer.model, explainer.checkpoint, explainer


# =============================================================
# SIDEBAR
# =============================================================

SCREENING, SCREENING_ERROR = load_screening()
REFERENCE = load_reference_results()

with st.sidebar:
    # No remote assets: the console must run fully offline.
    st.markdown(
        '<div style="font-size:2.1rem;line-height:1;margin-bottom:2px;">🛡️</div>'
        '<div style="font-size:1.15rem;font-weight:700;">Cyber World Model</div>'
        '<div style="font-size:0.76rem;color:#8c96ab;">Predictive Screening Console '
        '· SIH 2026</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Select View",
        [
            "📥 Data Input",
            "🎯 Live Forecast",
            "⏱️ Early Warning",
            "🧭 Event Timeline",
            "🔮 Trajectory Rollout",
            "📊 Window Inspector",
            "🔍 Feature Attribution",
            "🛡️ Reliability / Model Health",
            "🗺️ MITRE ATT&CK Timeline",
            "🧭 Attack Path / Progression",
            "🧑‍💻 Analyst Decision",
            "🛡️ Defend Now Simulator",
            "🛡️ Counterfactual Defense Lab",
            "📐 Methodology & Limits",
        ],
        # Data Input leads the list because it is the one place data enters
        # the console, but the console still LANDS on Live Forecast.
        index=1,
        label_visibility="collapsed",
    )

    st.divider()

    # -------- REPLAY CONTROLS (screening substrate) --------
    if SCREENING is not None:
        st.markdown("### Replay Controls")
        n_seq = SCREENING.n_sequences
        default_seq = SCREENING.default_sequence_id

        if "seq_id" not in st.session_state:
            st.session_state.seq_id = default_seq

        def _clamp(v):
            return int(max(0, min(n_seq - 1, v)))

        nav1, nav2, nav3 = st.columns(3)
        if nav1.button("◀ Prev", use_container_width=True):
            st.session_state.seq_id = _clamp(st.session_state.seq_id - 1)
        if nav2.button("Next ▶", use_container_width=True):
            st.session_state.seq_id = _clamp(st.session_state.seq_id + 1)
        if nav3.button("⟲ Reset", use_container_width=True,
                       help="Return to the default demo sequence"):
            st.session_state.seq_id = default_seq

        st.session_state.seq_id = _clamp(
            st.number_input(
                "Sequence ID",
                min_value=0, max_value=n_seq - 1,
                value=int(st.session_state.seq_id), step=1,
            )
        )

        # Timestamp jump, restricted to real observed windows.
        ts_all = SCREENING.timestamps
        jump = st.selectbox(
            "Jump to timestamp",
            options=[None] + list(range(0, n_seq, 60)),
            format_func=lambda i: ("— select —" if i is None
                                   else f"{str(ts_all.iloc[i])[11:19]}  (#{i})"),
            help="Every 10th minute of the held-out replay day",
        )
        if jump is not None and st.button("Go to timestamp", use_container_width=True):
            st.session_state.seq_id = _clamp(jump)

        autoplay = st.toggle(
            "▶ Autoplay", value=False,
            help="Advance one 10 s window per refresh, stopping at a capture gap",
        )
        step_size = st.select_slider("Autoplay step", options=[1, 2, 5, 10], value=1)

        SEQ_ID = int(st.session_state.seq_id)
        seg_lo, seg_hi = SCREENING.segment_of(SEQ_ID)
        st.caption(
            f"Sequence **{SEQ_ID}** / {n_seq - 1} · "
            f"segment {seg_lo}–{seg_hi} · default **#{default_seq}**"
        )
    else:
        SEQ_ID, autoplay, step_size = 0, False, 1

    st.divider()

    st.markdown("### Architecture")
    st.markdown("""
    - **Encoder**: LSTM (128 hidden, 2 layers)
    - **Latent space**: 32 dimensions
    - **Transition**: gated residual + LayerNorm (V4)
    - **Rollout**: K=5 → **50 s** horizon
    - **Input**: 52 features × 10 windows (10 s each)
    - **Heads**: 26-D state decoder + threat probability
    """)

    st.divider()

    st.markdown("### Held-out Performance")
    model, ckpt, explainer = load_model()
    if REFERENCE is not None:
        ref5 = REFERENCE["trajectory_event_max"]["k=5"]
        col1, col2 = st.columns(2)
        col1.metric("F1 (K=5 traj.)", f"{ref5['f1']:.3f}")
        col2.metric("ROC-AUC", f"{ref5['roc_auc']:.3f}")
        st.caption(
            "Held-out **March-2** day, frozen validation-selected threshold. "
            "Not the checkpoint's training-time figures."
        )
    st.caption(
        f"checkpoint `{explainer.provenance.checkpoint_hash[:12]}…` "
        f"hash-verified against the frozen baseline manifest"
    )

    st.divider()

    st.markdown("### Analysis Horizon")
    horizon = st.radio(
        "Forecast horizon",
        VERIFIED_HORIZONS,
        index=VERIFIED_HORIZONS.index(DEFAULT_HORIZON),
        format_func=lambda k: f"t+{k}  ({10 * k}s ahead)",
        label_visibility="collapsed",
        help="Horizons with a frozen point threshold. Used by the attribution, "
             "MITRE and inspector pages.",
    )
    threshold = explainer.horizon_thresholds[horizon]
    st.metric(f"Frozen point threshold (K={horizon})", f"{threshold:.2f}")
    st.caption(
        "Validation-selected and frozen; test labels were not used "
        "to choose it."
    )

    st.divider()

    st.markdown("### Active Data Source")
    # READ-ONLY here. The source is CHOSEN in one place only -- the
    # 📥 Data Input page -- so there is no competing selector to disagree
    # with it. This block reports what that page decided.
    pcap_data = st.session_state.get("pcap_data")
    prepared_data = st.session_state.get("prepared_data")
    active_source = st.session_state.get("input_source", dinp.SRC_BENCHMARK)
    if active_source == dinp.SRC_PCAP and pcap_data is None:
        active_source = dinp.SRC_BENCHMARK
    if active_source == "prepared" and prepared_data is None:
        active_source = dinp.SRC_BENCHMARK

    if active_source == dinp.SRC_PCAP:
        st.markdown(
            '<span class="pill pill-context">📡 Uploaded PCAP</span>',
            unsafe_allow_html=True)
        st.caption(
            f"`{pcap_data['filename']}` — "
            f"{pcap_data['n_windows']} windows analyzed"
        )
    elif active_source == "prepared":
        st.markdown(
            '<span class="pill pill-context">📂 Prepared V4 data</span>',
            unsafe_allow_html=True)
        st.caption(
            f"`{prepared_data['filename']}` — "
            f"{prepared_data['n_windows']} sequence(s) loaded"
        )
    else:
        st.markdown(
            '<span class="pill pill-context">📊 Historical Benchmark</span>',
            unsafe_allow_html=True)
        st.caption(
            "Choose a different source on **📥 Data Input** to analyse a "
            "PCAP or prepared V4 data on every page."
        )


# =============================================================
# SCREENING SUBSTRATE GUARD
# =============================================================
# If the validated substrate is absent or malformed, say so plainly and
# stop. The demo never substitutes synthetic or partial data.

if SCREENING is None:
    st.error("### Screening substrate unavailable")
    st.markdown(
        f"The dashboard could not load the validated V4 screening replay "
        f"substrate from `results/v4_screening_mvp/`.\n\n"
        f"**Reason:** `{SCREENING_ERROR}`"
    )
    st.info(
        "Rebuild it with:\n\n```bash\npython src/build_v4_screening_mvp.py\n```\n\n"
        "No synthetic or placeholder data is substituted, so the demo stops here."
    )
    st.stop()


# -------- AUTOPLAY --------
# Advances one step per rerun and halts at a capture gap or segment end, so
# replay never runs silently across missing data.
if autoplay:
    _lo, _hi = SCREENING.segment_of(SEQ_ID)
    _next = SEQ_ID + int(step_size)
    if _next > _hi:
        st.sidebar.warning(
            "⏸ Autoplay paused at the end of this capture segment. "
            "Use **Next ▶** to continue after the data gap."
        )
    else:
        st.session_state.seq_id = _next
        time.sleep(0.6)
        st.rerun()


# =============================================================
# LOAD DATA
# =============================================================

X_test, test_meta = load_test_data()
mitre_df = load_mitre_mappings(len(test_meta))
importance_df = load_feature_importance(horizon)
attributions = load_attributions(horizon)

# Attack probabilities for the SELECTED horizon, from the same
# explainer instance that produces the attributions below. The
# displayed number and the explained number are therefore the same
# computation on the same tensor -- the property Task 8 Part 20
# tests (max |P - P_explained| = 0.0, bit-exact).
rollout_probs = explainer.predict_rollout(X_test, K=max(VERIFIED_HORIZONS))
probs = rollout_probs[:, horizon]

n_windows = len(test_meta)
n_alerts = (probs >= threshold).sum()
n_attacks_gt = test_meta["attack_present"].sum()


# =============================================================
# ACTIVE DATASET  (benchmark / uploaded PCAP / uploaded prepared data)
# =============================================================
# Every page below reads *_active variables, so the source chosen on the
# 📥 Data Input page swaps in without any page duplicating its own
# rendering logic -- and, crucially, without any source owning its own
# inference. Whatever produced the tensor, the probabilities below came
# from the same frozen rollout.

active_upload = None
if active_source == dinp.SRC_PCAP:
    active_upload = pcap_data
elif active_source == "prepared":
    active_upload = prepared_data

# True whenever a NON-benchmark dataset is active. Pages that require the
# frozen screening replay substrate check this and say so.
using_upload = active_upload is not None
using_pcap = (active_source == dinp.SRC_PCAP)


def upload_source_banner():
    """One line naming the uploaded dataset a page is showing."""
    if active_upload is None:
        return
    icon = "📡" if using_pcap else "📂"
    kind = ("uploaded PCAP" if using_pcap
            else f"uploaded prepared V4 data "
                 f"({active_upload.get('format_label', '')})")
    st.info(f"{icon} Showing your {kind}: "
            f"**{active_upload['filename']}**")


def substrate_only_notice(what: str):
    """Standard notice for a page that needs the frozen replay substrate."""
    st.info(
        f"{what} runs on the **screening replay substrate**, where the "
        f"frozen scores and the Task-15 / Task-16 / Task-17 layers are all "
        f"defined. Switch the source back to **Historical Benchmark** on "
        f"the **📥 Data Input** page to use this page."
    )


if using_upload:
    X_active = active_upload["X"]
    meta_active = active_upload["meta"]
    probs_active = active_upload["probs"]
    mitre_active = active_upload["mitre_df"]
    importance_active = active_upload["importance_df"]
    attributions_active = active_upload["attributions"]
    n_windows_active = active_upload["n_windows"]
    n_alerts_active = int(
        (probs_active >= threshold).sum()
    )
    has_ground_truth = False
else:
    X_active = X_test
    meta_active = test_meta
    probs_active = probs
    mitre_active = mitre_df
    importance_active = importance_df
    attributions_active = attributions
    n_windows_active = n_windows
    n_alerts_active = n_alerts
    has_ground_truth = True


# =============================================================
# PAGE: LIVE FORECAST  (primary screening view)
# =============================================================

if page == "🎯 Live Forecast":

    sub = SCREENING
    thr = sub.alert_threshold
    row = sub.preds.iloc[SEQ_ID]
    probs, latents, dec_states = rollout_detail(SEQ_ID)

    nowcast = float(row["nowcast_probability_t0"])
    fut = np.array([float(row[f"prob_t{k}"]) for k in range(1, sd.MAX_K + 1)])
    max_fut = float(row["max_forecast_probability"])
    argmax_k = int(np.argmax(fut)) + 1
    is_alert = bool(int(row["alert"]) == 1)
    obs_end = row["observed_window_end"]

    st.markdown("# 🎯 Live Threat Forecast")
    st.markdown(
        "> We are not only classifying the **current** network state. The world "
        "model rolls its latent state forward to forecast how the network state "
        "and threat probability may evolve over the **next 50 seconds**."
    )

    gap_banner(sub, SEQ_ID)

    # ---------- ALERT BANNER ----------
    if is_alert:
        st.markdown(
            f'<div class="banner banner-alert">'
            f'<span class="pill pill-forecast">Forecast alert</span>&nbsp;&nbsp;'
            f'<b style="font-size:1.05rem;">Elevated threat forecast within the next 50 s</b>'
            f'<br><span class="note">Maximum forecast probability '
            f'<b>{max_fut:.3f}</b> reaches the frozen alert threshold '
            f'<b>{thr:.2f}</b>, first peaking at <b>t+{argmax_k} '
            f'({sd.HORIZON_LABELS[argmax_k]})</b>. This is a model forecast over a '
            f'frozen, evaluation-derived threshold — not a statement that an attack '
            f'will occur.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner banner-ok">'
            f'<span class="pill pill-context">No forecast alert</span>&nbsp;&nbsp;'
            f'<b style="font-size:1.05rem;">Forecast stays below the frozen threshold</b>'
            f'<br><span class="note">Maximum forecast probability '
            f'<b>{max_fut:.3f}</b> remains under the frozen alert threshold '
            f'<b>{thr:.2f}</b> across all five horizons.</span></div>',
            unsafe_allow_html=True,
        )

    # ---------- COMPACT RELIABILITY INDICATOR (Task 15) ----------
    # Deliberately NOT called confidence, certainty, or probability of
    # correctness. It describes the conditions the forecast was produced
    # under, not the chance it is right.
    _rly = reliability_of(SEQ_ID)
    if _rly is not None:
        _c = RLY_COLOR[_rly.status]
        st.markdown(
            f'<div class="banner {RLY_BANNER[_rly.status]}" '
            f'style="padding:10px 14px;">'
            f'{reliability_chip(_rly)}&nbsp;&nbsp;'
            f'<b style="color:{_c};">{_rly.status}</b>'
            f'<span class="note">&nbsp;· {_rly.headline}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander("What does this reliability indicator mean?"):
            st.markdown(
                f"**Not a confidence score.** {rly.STATUS_NOTE}\n\n"
                f"It is a transparent rule over "
                f"{len(rly.TIERED_INDICATORS)} arithmetic indicators — input "
                f"distance from the frozen scaler's reference distribution, "
                f"rollout stability, and forecast self-consistency — plus "
                f"eight hard numerical fail-safes. Full detail, including "
                f"every triggering indicator, is on the "
                f"**🛡️ Reliability / Model Health** page.\n\n"
                f"Reasons for this window:"
            )
            for r in _rly.reasons:
                st.markdown(f"- `{r}`")
        if _rly.status == "UNRELIABLE":
            st.error(
                f"**{rly.ABSTENTION_HEADLINE}** — {rly.ABSTENTION_BODY} "
                f"See **🛡️ Reliability / Model Health** for the triggering "
                f"indicators."
            )

    # ---------- COMPACT ATTACK-PATH HYPOTHESIS CARD (Task 16) ----------
    # A rule-based interpretation of the SAME frozen rollout shown above.
    # The time-to-progression is withheld whenever the Task-15 reliability
    # status is UNRELIABLE, so a precise-looking number never appears
    # without its warning.
    _apx, _apx_err = attack_path_of(SEQ_ID)
    if _apx_err is not None:
        st.warning(
            f"**Attack-path hypothesis unavailable** for this sequence — "
            f"`{_apx_err}`. No substitute or guessed progression is shown."
        )
    elif _apx is not None:
        _cur_c = APX_STAGE_COLOR.get(_apx.current_stage, "#7f8c8d")
        _nxt = _apx.next_step
        if _nxt is not None:
            _t, _withheld = apx_timing_text(_apx, _nxt)
            _line = (f'Next projected: <b style="color:'
                     f'{APX_STAGE_COLOR.get(_nxt.to_stage, "#7f8c8d")};">'
                     f'{_nxt.to_stage}</b>')
            _line += (f' — {_t}' if _withheld
                      else f' in <b>{_t}</b>')
            _conf = (f'&nbsp;<span class="pill" style="background:'
                     f'{APX_CONF_COLOR[_nxt.progression_confidence]}22;color:'
                     f'{APX_CONF_COLOR[_nxt.progression_confidence]};">'
                     f'{_nxt.progression_confidence}</span>')
            _kind = "banner-alert"
        elif _apx.status == apx.STATUS_AMBIGUOUS:
            _line = "<b>PROGRESSION AMBIGUOUS — ANALYST REVIEW</b>"
            _conf = ""
            _kind = "banner-gap"
        else:
            _line = ("<b>No forward progression identified within 50 s.</b>")
            _conf = ""
            _kind = "banner-ok"

        _rl_note = ""
        if _apx.reliability_status == "UNRELIABLE":
            _rl_note = ('<br><span class="note" style="color:#ef4444;">'
                        '<b>PROGRESSION HYPOTHESIS UNRELIABLE</b> — the '
                        'forecast this interpretation reads was produced '
                        'outside the reference operating conditions.</span>')
        elif _apx.reliability_status == "CAUTION":
            _rl_note = ('<br><span class="note" style="color:#f5a524;">'
                        '⚠ CAUTION — forecast produced under unusual '
                        'operating conditions.</span>')

        st.markdown(
            f'<div class="banner {_kind}" style="padding:10px 14px;">'
            f'<span class="pill pill-context">attack-path hypothesis</span>'
            f'&nbsp;&nbsp;'
            f'Current: <b style="color:{_cur_c};">{_apx.current_stage}</b>'
            f'&nbsp;·&nbsp;{_line}{_conf}'
            f'{_rl_note}'
            f'<br><span class="note">{APX_BANNER} '
            f'Full chain, supporting telemetry and the progression graph are '
            f'on the <b>🧭 Attack Path / Progression</b> page.</span></div>',
            unsafe_allow_html=True,
        )

    # ---------- COMPACT ANALYST DECISION CARD (Task 17) ----------
    # The raw forecast probability is NOT replaced. Model forecast and
    # analyst decision layer are conceptually different and are shown as
    # two separate quantities.
    _ad, _ad_err = analyst_decision_of(SEQ_ID)
    if _ad_err is not None:
        st.warning(
            f"**Analyst decision layer unavailable** — `{_ad_err}`. The raw "
            f"forecast above is unaffected; no automated decision is shown."
        )
    elif _ad is not None:
        _c = ABS_COLOR[_ad.decision]
        _reasons = ""
        if _ad.abstained:
            _reasons = "<br>" + "".join(
                f'<span class="pill" style="background:{_c}22;color:{_c};'
                f'margin-right:6px;">{abs_layer.REASON_GROUP[r]}</span>'
                for r in _ad.reasons)
        st.markdown(
            f'<div class="banner {ABS_BANNER[_ad.decision]}" '
            f'style="padding:10px 14px;">'
            f'<span class="pill pill-context">analyst decision layer</span>'
            f'&nbsp;&nbsp;{ABS_ICON[_ad.decision]} '
            f'<b style="font-size:1.05rem;color:{_c};">'
            f'{_ad.display}</b>'
            f'<br><span class="note">'
            f'<b>Model forecast = {_ad.max_forecast:.3f}</b> &nbsp;·&nbsp; '
            f'<b>Analyst decision layer = {_ad.decision}</b>'
            f'<br>{_ad.semantics}</span>'
            f'{_reasons}'
            f'<br><span class="note">Full rationale on the '
            f'<b>🧑‍💻 Analyst Decision</b> page.</span></div>',
            unsafe_allow_html=True,
        )

    # ---------- DEFEND-NOW ENTRY POINT (Task 18) ----------
    # Opens the SELECTED sequence in the simulator. Nothing is executed on
    # any real network -- the simulator perturbs a feature vector and rolls
    # the frozen model forward.
    _dn1, _dn2 = st.columns([1, 3])
    with _dn1:
        if st.button("🛡️ What happens if I defend NOW?",
                     use_container_width=True,
                     help="Open this sequence in the Defend Now Simulator. "
                          "This is a model-based what-if projection and "
                          "performs NO action on any real network."):
            st.session_state["dsim_seq"] = SEQ_ID
            st.session_state["goto_dsim"] = True
    with _dn2:
        if st.session_state.get("goto_dsim"):
            st.success(
                f"Sequence **#{SEQ_ID}** is queued for the simulator — open "
                f"**🛡️ Defend Now Simulator** in the sidebar. "
                f"*{DSIM_DISCLAIMER}* — no network action is performed."
            )
        else:
            st.markdown(
                f'<div class="note">Model-based what-if projection: applies a '
                f'defined state perturbation and rolls the frozen world model '
                f'forward. It does <b>not</b> execute anything on a real '
                f'network and does <b>not</b> estimate causal '
                f'effectiveness.</div>',
                unsafe_allow_html=True)

    # ---------- HEADLINE CARDS ----------
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        card("Observed window (t)", str(obs_end)[11:19],
             f"{str(obs_end)[:10]} · seq #{SEQ_ID}", "card-now")
    with c2:
        card("Current threat <span class='pill pill-now'>now</span>",
             f"{nowcast:.3f}", "nowcast at t (no transition)", "card-now")
    with c3:
        card("Forecast horizon", "50 s", "K=5 rollout · 5 × 10 s windows", "card-forecast")
    with c4:
        card("Max future threat <span class='pill pill-forecast'>forecast</span>",
             f"{max_fut:.3f}", f"peak at t+{argmax_k} ({sd.HORIZON_LABELS[argmax_k]})",
             "card-forecast")
    with c5:
        card("Frozen threshold", f"{thr:.2f}", "validation-selected · K=5 trajectory MAX")
    with c6:
        card("Alert status",
             "<span class='v-alert'>ALERT</span>" if is_alert
             else "<span class='v-ok'>CLEAR</span>",
             "frozen threshold, unchanged",
             "card-alert" if is_alert else "card-ok")

    st.markdown("")

    # ---------- MAIN FORECAST CHART ----------
    st.markdown("### Threat trajectory — observed history vs world-model forecast")

    HISTORY_WINDOWS = 30
    seg_lo, _seg_hi = sub.segment_of(SEQ_ID)
    hist_lo = max(seg_lo, SEQ_ID - HISTORY_WINDOWS)      # never cross a capture gap
    hist_ids = list(range(hist_lo, SEQ_ID + 1))
    hist_x = [-(SEQ_ID - i) * sd.WINDOW_SECONDS for i in hist_ids]
    hist_y = sub.preds["nowcast_probability_t0"].iloc[hist_lo:SEQ_ID + 1].to_numpy(dtype=float)

    fore_x = [k * sd.WINDOW_SECONDS for k in range(1, sd.MAX_K + 1)]
    fore_y = fut

    fig = go.Figure()

    # Forecast region shading
    fig.add_vrect(x0=0, x1=sd.HORIZON_SECONDS, fillcolor=C_FORECAST, opacity=0.06,
                  layer="below", line_width=0)
    fig.add_vline(x=0, line_dash="dot", line_color="#ffffff", opacity=0.45)

    # OBSERVED / NOWCAST
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines+markers", name="Observed · nowcast at each t",
        line=dict(color=C_NOW, width=2.4),
        marker=dict(size=5, color=C_NOW),
        hovertemplate="t%{x:+d}s<br>nowcast %{y:.3f}<extra></extra>",
    ))

    # FORECAST (starts from the current nowcast so the handover is visible)
    fig.add_trace(go.Scatter(
        x=[0] + fore_x, y=[nowcast] + list(fore_y),
        mode="lines+markers", name="World-model forecast (K=5 rollout)",
        line=dict(color=C_FORECAST, width=3.2, dash="dash"),
        marker=dict(size=11, color=C_FORECAST, symbol="diamond",
                    line=dict(width=1, color="#1b2130")),
        hovertemplate="%{x:+d}s ahead<br>forecast %{y:.3f}<extra></extra>",
    ))

    # Frozen threshold reference
    fig.add_hline(
        y=thr, line_dash="dash", line_color=C_ALERT, opacity=0.9,
        annotation_text=f"Frozen alert threshold {thr:.2f}",
        annotation_position="top left",
        annotation_font=dict(color=C_ALERT, size=11),
        # keep the threshold label legible where the forecast line runs behind it
        annotation_bgcolor="rgba(19,23,34,0.82)",
    )

    # Mark horizons that exceed the threshold
    crossing = [(x, y) for x, y in zip(fore_x, fore_y) if y >= thr]
    if crossing:
        fig.add_trace(go.Scatter(
            x=[c[0] for c in crossing], y=[c[1] for c in crossing],
            mode="markers", name="At/above threshold",
            marker=dict(size=17, color="rgba(0,0,0,0)",
                        line=dict(color=C_ALERT, width=2.5)),
            hoverinfo="skip", showlegend=True,
        ))

    fig.update_layout(
        title=dict(text=f"Sequence #{SEQ_ID} · observed window t = {obs_end}", font=dict(size=13)),
        xaxis=dict(
            title="seconds relative to the current observed window t",
            tickmode="array",
            tickvals=[-90, -60, -30, 0, 10, 20, 30, 40, 50],
            ticktext=["t-90", "t-60", "t-30", "<b>t</b>",
                      "+10s", "+20s", "+30s", "+40s", "+50s"],
            range=[min(hist_x + [-30]) - 4, sd.HORIZON_SECONDS + 6],
        ),
        yaxis=dict(title="threat probability", range=[-0.02, 1.02]),
    )
    style_fig(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)

    lg1, lg2 = st.columns([3, 2])
    with lg1:
        st.markdown(
            '<span class="pill pill-now">observed</span> nowcast from the encoder '
            'latent z<sub>t</sub> — no transition applied.&nbsp;&nbsp;'
            '<span class="pill pill-forecast">forecast</span> each point is the attack '
            'head applied after <i>k</i> gated latent transitions.',
            unsafe_allow_html=True,
        )
    with lg2:
        st.markdown(
            '<div class="note"><b>t+4 (+40s)</b> is shown for completeness but has '
            '<b>no frozen point threshold</b> — the reference evaluation covers '
            'horizons 1, 2, 3 and 5 only. The alert above uses the frozen K=5 '
            'trajectory-MAX threshold.</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ---------- PER-HORIZON READOUT ----------
    st.markdown("### Forecast readout")
    delta = max_fut - nowcast
    pt = sub.point_thresholds
    ew = sd.early_warning_summary(sub)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        card("Predicted future threat", f"{max_fut:.3f}",
             f"max over t+1 … t+5 (the frozen trajectory score)", "card-forecast")
    with r2:
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
        colour = "v-alert" if delta > 0 else ("v-ok" if delta < 0 else "")
        card("Change vs observed now",
             f"<span class='{colour}'>{arrow} {delta:+.3f}</span>",
             f"nowcast {nowcast:.3f} → forecast {max_fut:.3f}",
             "card-forecast")
    with r3:
        first_cross = next((k for k in range(1, sd.MAX_K + 1) if fut[k - 1] >= thr), None)
        card("Forecast first crossing",
             f"t+{first_cross} ({sd.HORIZON_LABELS[first_cross]})" if first_cross
             else "none in 50 s",
             "earliest horizon reaching the frozen threshold",
             "card-alert" if first_cross else "card-ok")
    with r4:
        card("Early-warning state",
             "<span class='v-alert'>WARNING</span>" if is_alert
             else "<span class='v-ok'>QUIET</span>",
             f"evaluation: {ew['strict_coverage_pct']:.1f}% onset coverage, "
             f"{ew['strict_mean_lead_s']:.0f} s mean strict lead",
             "card-alert" if is_alert else "card-ok")

    horizon_tbl = pd.DataFrame({
        "Horizon": [f"t+{k}" for k in range(1, sd.MAX_K + 1)],
        "Ahead": [sd.HORIZON_LABELS[k] for k in range(1, sd.MAX_K + 1)],
        "Forecast probability": [f"{fut[k - 1]:.4f}" for k in range(1, sd.MAX_K + 1)],
        "Frozen point threshold": [f"{pt[k]:.2f}" if k in pt else "— none —"
                                   for k in range(1, sd.MAX_K + 1)],
        "vs trajectory threshold": ["at/above" if fut[k - 1] >= thr else "below"
                                     for k in range(1, sd.MAX_K + 1)],
    })
    st.dataframe(horizon_tbl, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="note">Wording is deliberate: these are <b>model forecasts</b> '
        'evaluated against <b>frozen, evaluation-derived thresholds</b>. The console '
        'does not present calibrated confidence intervals, and a forecast above the '
        'threshold is a screening signal for analyst review — not a guarantee that an '
        'attack will occur.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---------- NETWORK STATE FORECAST ----------
    st.markdown("### Network-state forecast — observed vs predicted trajectory")
    st.caption(
        "The decoder reconstructs the 26-D network state at each rollout step. "
        "Values are shown in physical units by inverting the frozen target scaler "
        "(the observed and decoded series share an identical standardisation)."
    )

    tscaler = load_target_scaler()
    obs_phys = tscaler.inverse_transform(sub.X[SEQ_ID][:, :26])       # (10, 26)
    dec_phys = tscaler.inverse_transform(dec_states)                  # (6, 26)

    chosen = st.multiselect(
        "State features",
        options=sd.STATE_FORECAST_FEATURES,
        default=["flow_count", "window_packet_rate", "syn_rate", "dst_port_entropy"],
        help="All drawn from the frozen 52-D V4 contract.",
    )
    if not chosen:
        st.info("Select at least one state feature to plot.")
    else:
        ncol = 2
        nrow = int(np.ceil(len(chosen) / ncol))
        sfig = make_subplots(rows=nrow, cols=ncol, subplot_titles=chosen,
                             vertical_spacing=0.17, horizontal_spacing=0.09)
        hist_rel = [-(sd.SEQ_LEN - 1 - s) * sd.WINDOW_SECONDS for s in range(sd.SEQ_LEN)]
        fore_rel = [k * sd.WINDOW_SECONDS for k in range(1, sd.MAX_K + 1)]

        for n, feat in enumerate(chosen):
            fi = sub.feature_names.index(feat)
            r, c = n // ncol + 1, n % ncol + 1
            sfig.add_vrect(x0=0, x1=sd.HORIZON_SECONDS, fillcolor=C_FORECAST,
                           opacity=0.06, layer="below", line_width=0, row=r, col=c)
            sfig.add_trace(go.Scatter(
                x=hist_rel, y=obs_phys[:, fi], mode="lines+markers",
                line=dict(color=C_NOW, width=2), marker=dict(size=4),
                name="Observed", legendgroup="obs", showlegend=(n == 0),
                hovertemplate="t%{x:+d}s<br>%{y:.3f}<extra></extra>",
            ), row=r, col=c)
            sfig.add_trace(go.Scatter(
                x=[0] + fore_rel, y=[dec_phys[0, fi]] + list(dec_phys[1:, fi]),
                mode="lines+markers", line=dict(color=C_FORECAST, width=2.4, dash="dash"),
                marker=dict(size=7, symbol="diamond"),
                name="Predicted (decoder)", legendgroup="fore", showlegend=(n == 0),
                hovertemplate="%{x:+d}s ahead<br>%{y:.3f}<extra></extra>",
            ), row=r, col=c)

        sfig.update_xaxes(tickmode="array", tickvals=[-90, -50, 0, 25, 50],
                          ticktext=["t-90", "t-50", "t", "+25s", "+50s"])
        style_fig(sfig, height=230 * nrow)
        for ann in sfig.layout.annotations:
            ann.font.size = 12
        st.plotly_chart(sfig, use_container_width=True)

        st.markdown(
            '<div class="note">These are the decoder\'s own reconstructions at each '
            'rollout step. They are model output, not observed measurements, and they '
            'carry no attack label of their own — the threat probability comes solely '
            'from the attack head applied to the same rolled-forward latent.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="demo-footer">Offline replay of real held-out '
        'March-2 telemetry · CSE-CIC-IDS2018 · frozen V4 checkpoint</div>',
        unsafe_allow_html=True,
    )


# =============================================================
# PAGE: EARLY WARNING
# =============================================================

elif page == "⏱️ Early Warning":

    sub = SCREENING
    thr = sub.alert_threshold
    row = sub.preds.iloc[SEQ_ID]
    ew = sd.early_warning_summary(sub)

    fut = np.array([float(row[f"prob_t{k}"]) for k in range(1, sd.MAX_K + 1)])
    max_fut = float(row["max_forecast_probability"])
    is_alert = bool(int(row["alert"]) == 1)
    first_cross = next((k for k in range(1, sd.MAX_K + 1) if fut[k - 1] >= thr), None)
    obs_end = row["observed_window_end"]

    st.markdown("# ⏱️ Early Warning")
    st.markdown(
        "> Whether the frozen forecast crosses the alert threshold **before** an "
        "attack becomes observable — and how much lead time that buys."
    )

    gap_banner(sub, SEQ_ID)

    # ---------- THE CARD, for the selected sequence ----------
    st.markdown("### This sequence")

    e1, e2, e3 = st.columns(3)
    with e1:
        card("Warning state",
             "<span class='v-alert'>WARNING</span>" if is_alert
             else "<span class='v-ok'>QUIET</span>",
             f"seq #{SEQ_ID} · observed t = {str(obs_end)[11:19]}",
             "card-alert" if is_alert else "card-ok")
    with e2:
        card("Forecast first crossing",
             f"t+{first_cross} <span class='pill pill-forecast'>"
             f"{sd.HORIZON_LABELS[first_cross]}</span>" if first_cross
             else "none within 50 s",
             f"earliest horizon reaching {thr:.2f}",
             "card-alert" if first_cross else "card-ok")
    with e3:
        card("Max forecast probability", f"{max_fut:.3f}",
             f"frozen threshold {thr:.2f}", "card-forecast")

    # Per-sequence lead time is NOT invented. It only exists if this window sits
    # in a documented pre-attack region and the onset index is known.
    region = sub.region_of(SEQ_ID)
    onset_ids = sub.timeline.index[sub.timeline["is_documented_onset"] == 1].tolist()
    next_onset = next((o for o in onset_ids if o > SEQ_ID), None)
    same_segment = (next_onset is not None
                    and sub.segment_of(SEQ_ID) == sub.segment_of(next_onset))

    if region == "pre_attack" and next_onset is not None and same_segment:
        lead = (next_onset - SEQ_ID) * sd.WINDOW_SECONDS
        onset_ts = sub.timeline["target_t1_timestamp"].iloc[next_onset]
        banner_cls = "banner-alert" if is_alert else "banner-gap"
        if is_alert:
            contrib = (f"The forecast is alerting here, so this window contributes "
                       f"a <b>{lead} s</b> pre-attack warning.")
        else:
            contrib = ("The forecast is not alerting here, so this window "
                       "contributes no warning.")
        st.markdown(
            f'<div class="banner {banner_cls}">'
            f'<b>Pre-attack window &mdash; lead time measurable</b><br>'
            f'<span class="note">This window lies <b>{lead} s</b> before the '
            f'documented onset at <b>{onset_ts}</b> (sequence #{next_onset}). '
            f'{contrib}'
            f'<br>The onset location comes from the documented event annotation &mdash; '
            f'it is <b>context for measuring lead time</b>, never a model input.'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    else:
        reason = {
            "benign": "no documented onset within the next 50 s",
            "attack_active": "an attack is already active at this window",
            "attack_active_at_start_no_observable_onset":
                "an attack is already active at capture start — this episode has no "
                "observable onset",
            "documented_attack_onset": "this window *is* the documented onset",
        }.get(region, "this window is not in a measurable pre-attack region")
        st.markdown(
            f'<div class="banner banner-gap">'
            f'<b>No episode-level lead time for this sequence</b><br>'
            f'<span class="note">Region: <code>{sd.REGION_DISPLAY.get(region, region)}</code> — '
            f'{reason}. A per-sequence lead time is <b>not invented</b> where it cannot '
            f'be established from the documented onset metadata.</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ---------- EVALUATION SUMMARY (strict, headline) ----------
    st.markdown("### Evaluation summary — strict pre-attack early warning")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        card("Onset coverage", f"{ew['strict_coverage_pct']:.1f}%",
             f"{ew['strict_warned']} of {ew['evaluable_onsets']} evaluable onsets warned",
             "card-forecast")
    with s2:
        card("Mean strict lead", f"{ew['strict_mean_lead_s']:.0f} s",
             f"median {ew['strict_median_lead_s']:.0f} s", "card-forecast")
    with s3:
        card("Evaluable onsets", f"{ew['evaluable_onsets']}",
             "one held-out day · small sample")
    with s4:
        card("False-alert rate", f"{ew['false_positive_windows_per_hour']:.2f}/h",
             "alerting windows with no attack in t+1…t+5")

    st.markdown(
        f'<div class="banner banner-gap"><b>Read this number carefully.</b>'
        f'<br><span class="note">'
        f'• The headline figures above are the <b>strict pre-attack</b> results: '
        f'<b>{ew["strict_coverage_pct"]:.1f}% coverage, {ew["strict_mean_lead_s"]:.0f} s '
        f'mean lead</b>, counting only windows that are benign at both the observation '
        f'time <i>t</i> and the evaluated target <i>t+1</i>.<br>'
        f'• The originally published reference convention credits the first alerting '
        f'window in the 5 windows before onset, which for 2 of 3 episodes still lies '
        f'<i>inside</i> the preceding active attack. That convention yields '
        f'{ew["reference_coverage_pct"]:.1f}% / {ew["reference_mean_lead_s"]:.0f} s. It is '
        f'disclosed here for completeness and is <b>not</b> the headline number.<br>'
        f'• {ew["caveat"]}</span></div>',
        unsafe_allow_html=True,
    )

    # ---------- PER-EPISODE TABLE ----------
    if ew["strict_episodes"]:
        st.markdown("#### Per-episode strict results")
        ep = pd.DataFrame(ew["strict_episodes"])
        show = pd.DataFrame({
            "Onset sequence": ep["onset_index"],
            "Onset timestamp": ep["onset_timestamp"],
            "Eligible pre-attack windows": ep["eligible_pre_attack_windows"],
            "Warned": np.where(ep["warned"], "YES", "NO"),
            "Strict lead (s)": np.where(ep["lead_time_seconds"] > 0,
                                        ep["lead_time_seconds"].astype(str), "—"),
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

        jump = st.selectbox(
            "Jump to an onset episode",
            options=list(ep["onset_index"]),
            format_func=lambda i: f"#{i} · {sub.timestamps.iloc[int(i)]}",
        )
        if st.button("Go to onset →"):
            st.session_state.seq_id = int(jump)
            st.rerun()

    st.markdown(
        '<div class="demo-footer">Offline replay of real held-out '
        'March-2 telemetry · frozen threshold, validation-selected</div>',
        unsafe_allow_html=True,
    )


# =============================================================
# PAGE: EVENT TIMELINE
# =============================================================

elif page == "🧭 Event Timeline":

    sub = SCREENING
    thr = sub.alert_threshold

    st.markdown("# 🧭 Event Timeline")
    st.markdown(
        "> Forecast threat over the whole held-out day, overlaid on the "
        "**documented** event regions — so the two can be compared directly."
    )

    st.markdown(
        '<div class="banner banner-gap">'
        '<span class="pill pill-context">ground truth · context only</span>&nbsp;&nbsp;'
        '<b>The shaded event regions are documented ground truth, not model input.</b>'
        '<br><span class="note">They come from '
        '<code>results/v4_screening_mvp/event_timeline.csv</code> and are used purely to '
        'annotate this chart. They never enter the model\'s 52-D input tensor, never '
        'influence the frozen threshold, never influence a prediction, and never '
        'generate a warning. The forecast curve below was computed from network '
        'features alone.</span></div>',
        unsafe_allow_html=True,
    )

    # ---------- CONTROLS ----------
    tl1, tl2 = st.columns([3, 2])
    with tl1:
        segments = sub.segments()
        seg_choice = st.selectbox(
            "Capture segment",
            options=list(range(len(segments))),
            format_func=lambda i: (
                f"Segment {i + 1}: {str(sub.timestamps.iloc[segments[i][0]])[11:19]} – "
                f"{str(sub.timestamps.iloc[segments[i][1]])[11:19]}  "
                f"({segments[i][1] - segments[i][0] + 1} windows)"
            ),
            help="The day is split by a real capture gap; segments are never joined.",
        )
    with tl2:
        series_choice = st.radio(
            "Forecast series",
            ["Max forecast (t+1…t+5)", "Nowcast at t", "Both"],
            index=0, horizontal=False,
        )

    lo, hi = segments[seg_choice]
    ids = list(range(lo, hi + 1))
    ts = sub.timestamps.iloc[lo:hi + 1]
    mf = sub.max_forecast[lo:hi + 1]
    nc = sub.preds["nowcast_probability_t0"].to_numpy(dtype=float)[lo:hi + 1]
    regions = sub.timeline["region"].to_numpy()[lo:hi + 1]

    fig = go.Figure()

    # ---------- DOCUMENTED EVENT REGIONS (context bands) ----------
    # Contiguous runs of the same region become one shaded band.
    start = 0
    for i in range(1, len(regions) + 1):
        if i == len(regions) or regions[i] != regions[start]:
            reg = regions[start]
            if reg != "benign":
                fig.add_vrect(
                    x0=ts.iloc[start], x1=ts.iloc[min(i, len(regions) - 1)],
                    fillcolor=sd.REGION_COLORS.get(reg, "#333"),
                    opacity=0.55, layer="below", line_width=0,
                )
            start = i

    # ---------- FORECAST SERIES ----------
    if series_choice in ("Max forecast (t+1…t+5)", "Both"):
        fig.add_trace(go.Scatter(
            x=ts, y=mf, mode="lines", name="Max forecast (t+1…t+5)",
            line=dict(color=C_FORECAST, width=1.7),
            hovertemplate="%{x|%H:%M:%S}<br>max forecast %{y:.3f}<extra></extra>",
        ))
    if series_choice in ("Nowcast at t", "Both"):
        fig.add_trace(go.Scatter(
            x=ts, y=nc, mode="lines", name="Nowcast at t",
            line=dict(color=C_NOW, width=1.4),
            hovertemplate="%{x|%H:%M:%S}<br>nowcast %{y:.3f}<extra></extra>",
        ))

    fig.add_hline(
        y=thr, line_dash="dash", line_color=C_ALERT, opacity=0.9,
        annotation_text=f"Frozen threshold {thr:.2f}",
        annotation_position="top left",
        annotation_font=dict(color=C_ALERT, size=11),
        # keep the threshold label legible where the forecast line runs behind it
        annotation_bgcolor="rgba(19,23,34,0.82)",
    )

    # Documented onsets as vertical markers
    onset_rows = sub.timeline.iloc[lo:hi + 1]
    onset_rows = onset_rows[onset_rows["is_documented_onset"] == 1]
    for _, orow in onset_rows.iterrows():
        fig.add_vline(
            x=orow["target_t1_timestamp"], line_color="#ffffff",
            line_width=1.6, line_dash="dot", opacity=0.85,
        )

    # Current replay position
    if lo <= SEQ_ID <= hi:
        fig.add_vline(x=sub.timestamps.iloc[SEQ_ID], line_color="#22c55e",
                      line_width=2, opacity=0.9)

    fig.update_layout(
        title=dict(text="Forecast threat vs documented event regions", font=dict(size=13)),
        xaxis=dict(title="observed time (March 2, 2018)"),
        yaxis=dict(title="threat probability", range=[-0.02, 1.02]),
    )
    style_fig(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- LEGEND for the context bands ----------
    st.markdown("**Documented event regions** (context, not model input)")
    legend_cols = st.columns(len(sd.REGION_ORDER))
    counts_all = sub.timeline["region"].value_counts()
    for col, reg in zip(legend_cols, sd.REGION_ORDER):
        with col:
            st.markdown(
                f'<div class="card" style="border-left:3px solid '
                f'{sd.REGION_COLORS[reg]};padding:10px 12px;">'
                f'<div class="card-label" style="font-size:.66rem;">'
                f'{sd.REGION_DISPLAY[reg]}</div>'
                f'<div style="font-size:1.15rem;font-weight:700;">'
                f'{int(counts_all.get(reg, 0))}</div>'
                f'<div class="card-sub">windows</div></div>',
                unsafe_allow_html=True,
            )
    st.caption(
        "White dotted lines mark the documented attack onsets. "
        "The green line marks the current replay position. "
        "Benign windows are left unshaded."
    )

    # ---------- CAPTURE GAP DISCLOSURE ----------
    gaps = sub.capture_gaps()
    if gaps:
        st.divider()
        st.markdown("### Capture gaps")
        for g in gaps:
            st.markdown(
                f'<div class="banner banner-gap">'
                f'<b>⚠ DATA GAP — {g["gap_seconds"] / 3600:.2f} h with no windows</b><br>'
                f'<span class="note"><code>{g["from_timestamp"]}</code> → '
                f'<code>{g["to_timestamp"]}</code>. This is a real discontinuity in the '
                f'raw CSE-CIC-IDS2018 capture. The replay is split into separate '
                f'segments at this point: no line is drawn across it, nothing is '
                f'interpolated, and no synthetic windows are created.</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="demo-footer">Offline replay of real held-out '
        'March-2 telemetry · event regions are ground-truth context only</div>',
        unsafe_allow_html=True,
    )


# =============================================================
# PAGE: MITRE ATT&CK TIMELINE
# =============================================================

# =============================================================
# PAGE — RELIABILITY / MODEL HEALTH   (Task 15)
# =============================================================
# The underlying reliability MECHANISM only. It reports; it does not
# decide. The ALERT / BENIGN / ABSTAIN human-in-the-loop decision system
# is Task 16 and is deliberately not built here.

elif page == "🛡️ Reliability / Model Health":

    sub = SCREENING
    RLY_REF, RLY_ERR = load_reliability()

    st.markdown("# 🛡️ Reliability / Model Health")
    st.markdown(
        "> The V4 world model is **frozen**. This page is a separate "
        "decision-support layer that asks a different question: *were the "
        "conditions under which this forecast was produced conditions the "
        "model has any business forecasting in?*"
    )

    if RLY_REF is None:
        st.error("### Reliability reference unavailable")
        st.markdown(
            f"The reliability layer could not load its reference "
            f"distribution from `results/v4_reliability/`.\n\n"
            f"**Reason:** `{RLY_ERR}`"
        )
        st.info(
            "Build it with:\n\n```bash\n"
            "python src/build_v4_reliability.py\n```\n\n"
            "No status is shown without its reference distribution: a "
            "colour with no reference behind it would be worse than "
            "nothing."
        )
        st.stop()

    a = reliability_of(SEQ_ID)
    row = sub.preds.iloc[SEQ_ID]
    probs, latents, dec_states = rollout_detail(SEQ_ID)
    thr = sub.alert_threshold
    colour = RLY_COLOR[a.status]

    gap_banner(sub, SEQ_ID)

    # ---------- HEADLINE STATUS ----------
    st.markdown(
        f'<div class="banner {RLY_BANNER[a.status]}">'
        f'<span class="pill" style="background:{colour}22;color:{colour};'
        f'border:1px solid {colour}59;">Reliability status</span>&nbsp;&nbsp;'
        f'<b style="font-size:1.45rem;color:{colour};">{a.status}</b>'
        f'<br><span class="note">{a.headline}</span></div>',
        unsafe_allow_html=True,
    )

    # ---------- ABSTENTION CALL-OUT ----------
    if a.status == "UNRELIABLE":
        st.markdown(
            f'<div class="banner banner-alert" style="padding:18px 20px;">'
            f'<div style="font-size:1.5rem;font-weight:800;color:'
            f'{C_ALERT};letter-spacing:.02em;">'
            f'⛔ {rly.ABSTENTION_HEADLINE}</div>'
            f'<div style="font-size:1.05rem;margin-top:6px;">'
            f'{rly.ABSTENTION_BODY}</div>'
            f'<div class="note" style="margin-top:8px;">'
            f'The forecast below is still shown, unaltered, because hiding '
            f'it would hide the evidence. It must not be acted on for this '
            f'window. Converting this into an automatic ABSTAIN decision is '
            f'Task 16 and is not done here.</div></div>',
            unsafe_allow_html=True,
        )

    st.caption(f"⚠️ {rly.STATUS_NOTE}")

    # ---------- HEADLINE CARDS ----------
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card("Sequence", f"#{SEQ_ID}",
             f"{str(row['target_t1_timestamp'])[11:19]} · t+1 target",
             RLY_CARD[a.status])
    with c2:
        card("Status", f'<span style="color:{colour};">{a.status}</span>',
             f"compact label: {a.short}", RLY_CARD[a.status])
    with c3:
        card("Hard fail-safes fired", f"{len(a.hard_fails)}",
             (", ".join(sorted({h["code"] for h in a.hard_fails}))
              if a.hard_fails else "none of H1–H8"),
             "card-alert" if a.hard_fails else "card-ok")
    with c4:
        card("Beyond reference range", f"{a.n_beyond}",
             f"of {len(rly.TIERED_INDICATORS)} indicators",
             "card-alert" if a.n_beyond else "card-ok")
    with c5:
        card("Above reference p99", f"{a.n_elevated}",
             f"of {len(rly.TIERED_INDICATORS)} indicators",
             "card-forecast" if a.n_elevated else "card-ok")

    st.markdown("")

    # ---------- WHY ----------
    st.markdown("### Why this status")
    st.markdown(
        "Every status is reconstructable from the lines below. There is no "
        "learned component and no hidden weighting."
    )
    for r in a.reasons:
        if r.startswith("[H"):
            st.error(f"**HARD FAIL-SAFE** — `{r}`")
        elif r.startswith("[BEYOND"):
            st.warning(f"`{r}`")
        elif r.startswith("[ELEVATED"):
            st.info(f"`{r}`")
        else:
            st.success(f"`{r}`")

    with st.expander("The rule, in full"):
        st.markdown(f"""
| Status | Condition |
|---|---|
| **UNRELIABLE** | any hard fail-safe H1–H8 fires, **or** ≥ {rly.RULE_BEYOND_FOR_UNRELIABLE} indicators exceed the reference **maximum** |
| **CAUTION** | ≥ {rly.RULE_BEYOND_FOR_CAUTION} indicator exceeds the reference maximum, **or** ≥ {rly.RULE_ELEVATED_FOR_CAUTION} indicators exceed the reference **p{rly.ELEVATED_QUANTILE:.0f}** |
| **RELIABLE** | otherwise |

**Reference distribution:** `{RLY_REF.source}` — {RLY_REF.n_reference_sequences:,}
sequences, the train partition the frozen `input_scaler` was fit on, pushed
through the frozen checkpoint. **No label array is read**, and no held-out test
label was used to set any threshold.

**Hard fail-safe conditions** (numerical / contract, not quantiles):

| Code | Condition |
|---|---|
| H1 | non-finite model input (NaN / Inf) |
| H2 | invalid input, latent, state or probability tensor shape |
| H3 | input scaler incompatible with the frozen contract |
| H4 | non-finite latent state or decoded state |
| H5 | non-finite or out-of-[0,1] threat probability |
| H6 | ‖z₀‖ > {RLY_REF.bounds['encoder_output_norm_bound']:.4f} — the analytic encoder bound |
| H7 | ‖z_k‖ > {RLY_REF.bounds['transition_output_norm_bound']:.4f} — the analytic LayerNorm transition bound |
| H8 | an input value lies > {rly.INPUT_SHIFT_HARD_MARGIN_SD:.0f} training SD outside the reference range for its feature |

H6 and H7 are **derived from the frozen weights**, not fitted: the transition
ends in `LayerNorm`, so ‖z_k‖ ≤ ‖w‖∞·√32 + ‖b‖₂ for k ≥ 1. Exceeding either
bound is numerically impossible for an intact checkpoint, so H6/H7 are
corruption detectors rather than instability detectors.
        """)

    # ---------- INDICATOR TABLE, BY GROUP ----------
    st.markdown("### Indicators")

    GROUP_TITLE = {
        "input_ood": "A · Input OOD / distance from the frozen scaler",
        "rollout_stability": "B · Rollout stability",
        "forecast_consistency": "C · Forecast self-consistency",
    }
    TIER_ICON = {"OK": "🟢 OK", "ELEVATED": "🟠 ELEVATED",
                 "BEYOND_REFERENCE": "🔴 BEYOND REFERENCE"}

    for grp, title in GROUP_TITLE.items():
        st.markdown(f"**{title}**")
        rows = []
        for ind in rly.INDICATORS:
            if ind.group != grp:
                continue
            v = a.indicators[ind.name]
            if ind.name in a.tiers:
                q = RLY_REF.quantiles[ind.name]
                rows.append({
                    "Indicator": ind.name,
                    "Value": f"{v:.4g}",
                    "Reference p99": f"{q['p99']:.4g}",
                    "Reference max": f"{q['max']:.4g}",
                    "% of ref max": (f"{100 * v / q['max']:.1f}%"
                                     if q["max"] > 0 else "—"),
                    "Tier": TIER_ICON[a.tiers[ind.name]],
                })
            else:
                rows.append({
                    "Indicator": ind.name + "  (hard condition H8)",
                    "Value": f"{v:.4g}",
                    "Reference p99": "0 by construction",
                    "Reference max": "0 by construction",
                    "% of ref max": "—",
                    "Tier": ("🔴 HARD FAIL"
                             if v > rly.INPUT_SHIFT_HARD_MARGIN_SD
                             else ("🟠 outside support" if v > 0 else "🟢 OK")),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
        with st.expander(f"What these {grp.replace('_', ' ')} indicators mean"):
            for ind in rly.INDICATORS:
                if ind.group == grp:
                    st.markdown(f"- **`{ind.name}`** — {ind.description}")

    # ---------- INPUT OOD DETAIL ----------
    st.markdown("### Input shift detail")
    d = a.detail
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        card("Mean |z| over 520 values", f"{d['mean_abs_z']:.3f}",
             "1.0 = the training scale")
    with o2:
        card("Values with |z| > 2", f"{d['n_values_gt2']} / 520",
             f"{100 * a.indicators['frac_gt2']:.1f}% of the input tensor")
    with o3:
        card("Values with |z| > 3", f"{d['n_values_gt3']} / 520",
             f"{100 * a.indicators['frac_gt3']:.1f}% of the input tensor")
    with o4:
        card("Outside reference range",
             f"{d['n_values_outside_support']} / 520",
             (f"worst: `{d['worst_feature']}` at window "
              f"{d['worst_feature_window']}") if d["worst_feature"]
             else "every value inside observed support",
             "card-alert" if d["n_values_outside_support"] else "card-ok")

    # Per-window |z| profile: where in the 10-window lookback the shift sits.
    xin = sub.model_input(SEQ_ID)[0]
    absz = np.abs(xin.astype(float))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"t-{(9 - i) * 10}" if i < 9 else "t" for i in range(10)],
        y=absz.max(1), mode="lines+markers", name="max |z| in window",
        line=dict(color=C_ALERT, width=2.6),
        hovertemplate="%{x}<br>max |z| %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[f"t-{(9 - i) * 10}" if i < 9 else "t" for i in range(10)],
        y=absz.mean(1), mode="lines+markers", name="mean |z| in window",
        line=dict(color=C_NOW, width=2.2),
        hovertemplate="%{x}<br>mean |z| %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=2, line_dash="dot", line_color="#8c96ab", opacity=0.7,
                  annotation_text="|z| = 2", annotation_position="top left",
                  annotation_font=dict(color="#8c96ab", size=10))
    fig.add_hline(y=3, line_dash="dash", line_color=C_FORECAST, opacity=0.8,
                  annotation_text="|z| = 3", annotation_position="top left",
                  annotation_font=dict(color=C_FORECAST, size=10))
    fig.update_layout(title="Standardized input magnitude across the "
                            "10-window lookback",
                      yaxis_title="|z| (training standard deviations)")
    st.plotly_chart(style_fig(fig, height=330), use_container_width=True)
    st.caption(
        f"Most extreme single feature this window: "
        f"**`{d['worst_abs_z_feature']}`** at |z| = "
        f"{a.indicators['max_abs_z']:.2f}. Values are already in frozen "
        f"`input_scaler` units, so no rescaling happens here."
    )

    # ---------- ROLLOUT / LATENT STABILITY ----------
    st.markdown("### Rollout and latent stability")
    ks = list(range(len(d["latent_norm_per_step"])))
    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.075,
        subplot_titles=("Latent norm ‖z_k‖", "Latent step ‖z_k − z_{k−1}‖",
                        "Threat probability P_k"),
    )
    fig.add_trace(go.Scatter(
        x=ks, y=d["latent_norm_per_step"], mode="lines+markers",
        line=dict(color=C_NOW, width=2.6), name="‖z_k‖",
        hovertemplate="k=%{x}<br>‖z‖ %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=RLY_REF.bounds["transition_output_norm_bound"],
                  line_dash="dash", line_color=C_ALERT, opacity=0.85,
                  annotation_text="H7 analytic bound",
                  annotation_font=dict(color=C_ALERT, size=10),
                  row=1, col=1)
    fig.add_hline(y=RLY_REF.quantiles["latent_norm_max"]["max"],
                  line_dash="dot", line_color=C_FORECAST, opacity=0.8,
                  annotation_text="reference max",
                  annotation_font=dict(color=C_FORECAST, size=10),
                  row=1, col=1)
    fig.add_trace(go.Bar(
        x=list(range(1, len(d["latent_step_delta"]) + 1)),
        y=d["latent_step_delta"], marker_color=C_FORECAST,
        name="step ‖Δz‖",
        hovertemplate="step %{x}<br>‖Δz‖ %{y:.3f}<extra></extra>"),
        row=1, col=2)
    fig.add_hline(y=RLY_REF.quantiles["step_delta_max"]["max"],
                  line_dash="dot", line_color=C_ALERT, opacity=0.8,
                  annotation_text="reference max",
                  annotation_font=dict(color=C_ALERT, size=10),
                  row=1, col=2)
    fig.add_trace(go.Scatter(
        x=ks, y=d["probs"], mode="lines+markers",
        line=dict(color=C_ALERT, width=2.6), name="P_k",
        hovertemplate="k=%{x}<br>P %{y:.4f}<extra></extra>"), row=1, col=3)
    fig.add_hline(y=thr, line_dash="dash", line_color="#8c96ab", opacity=0.8,
                  annotation_text=f"frozen threshold {thr:.2f}",
                  annotation_font=dict(color="#8c96ab", size=10),
                  row=1, col=3)
    fig.update_yaxes(range=[0, 1.02], row=1, col=3)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="rollout step k", row=1, col=c)
    st.plotly_chart(style_fig(fig, height=360, legend=False),
                    use_container_width=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        card("Peak ‖z_k‖", f"{a.indicators['latent_norm_max']:.3f}",
             f"H7 bound {RLY_REF.bounds['transition_output_norm_bound']:.3f}",
             "card-ok" if a.tiers["latent_norm_max"] == "OK" else "card-forecast")
    with s2:
        card("Latent growth", f"{a.indicators['latent_norm_change']:+.3f}",
             "max‖z_k‖ − ‖z₀‖")
    with s3:
        card("Largest latent step", f"{a.indicators['step_delta_max']:.3f}",
             f"reference max "
             f"{RLY_REF.quantiles['step_delta_max']['max']:.3f}")
    with s4:
        card("Total displacement", f"{a.indicators['displacement']:.3f}",
             f"reference max "
             f"{RLY_REF.quantiles['displacement']['max']:.3f}")
    st.caption(
        "Latent magnitude is **bounded by construction**: the V4 transition "
        "ends in `LayerNorm`, which fixes the latent RMS. A stable ‖z‖ is "
        "therefore weak evidence about the learned dynamics and strong "
        "evidence about the architecture — the same caveat "
        "`docs/LONG_HORIZON_ROLLOUT.md` §8 records. H7 is a corruption "
        "detector, not an instability detector."
    )

    # ---------- FORECAST SELF-CONSISTENCY ----------
    st.markdown("### Forecast self-consistency")
    f1, f2, f3 = st.columns(3)
    with f1:
        card("Sharpest state change",
             f"{a.indicators['state_delta_max']:.3f}",
             f"‖Δs‖ · reference max "
             f"{RLY_REF.quantiles['state_delta_max']['max']:.3f}")
    with f2:
        card("Sharpest probability jump",
             f"{a.indicators['prob_max_abs_step']:.4f}",
             f"reference max "
             f"{RLY_REF.quantiles['prob_max_abs_step']['max']:.3f}")
    with f3:
        card("Probability spread", f"{a.indicators['prob_range']:.4f}",
             f"reference max "
             f"{RLY_REF.quantiles['prob_range']['max']:.3f}")

    cc = pd.DataFrame({
        "step": [f"k={i} → k={i + 1}" for i in range(len(d["prob_step_delta"]))],
        "Δ threat probability": [f"{v:+.4f}" for v in np.diff(d["probs"])],
        "|Δ| probability": [f"{v:.4f}" for v in d["prob_step_delta"]],
        "‖Δ decoded state‖": [f"{v:.4f}" for v in d["state_step_delta"]],
        "‖Δ latent‖": [f"{v:.4f}" for v in d["latent_step_delta"]],
    })
    st.dataframe(cc, hide_index=True, use_container_width=True)

    # ---------- COHORT CONTEXT ----------
    st.markdown("### Where this window sits on the replay day")
    per_seq_path = Path("results/v4_reliability/reliability_per_sequence.csv")
    if per_seq_path.exists():
        allr = pd.read_csv(per_seq_path)
        counts = allr["reliability_status"].value_counts()
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            card("Windows assessed", f"{len(allr):,}",
                 "held-out March-2 replay day")
        for col, stt in zip((k2, k3, k4), rly.STATUS_ORDER):
            n = int(counts.get(stt, 0))
            with col:
                card(stt.title(), f"{n:,}",
                     f"{100 * n / len(allr):.2f}% of the day", RLY_CARD[stt])

        y = allr["reliability_status"].map(
            {"RELIABLE": 0, "CAUTION": 1, "UNRELIABLE": 2}).to_numpy()
        xs, ys = sub.insert_gap_breaks(allr["sequence_id"].tolist(), y.tolist())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color=C_FORECAST, width=1.3),
            name="status", hovertemplate="seq %{x}<extra></extra>"))
        for stt, lvl in (("CAUTION", 1), ("UNRELIABLE", 2)):
            m = allr["reliability_status"] == stt
            fig.add_trace(go.Scatter(
                x=allr.loc[m, "sequence_id"], y=np.full(int(m.sum()), lvl),
                mode="markers", name=stt,
                marker=dict(color=RLY_COLOR[stt], size=6),
                hovertemplate=f"{stt}<br>seq %{{x}}<extra></extra>"))
        fig.add_vline(x=SEQ_ID, line_dash="dot", line_color="#22c55e",
                      opacity=0.9, annotation_text="now",
                      annotation_font=dict(color="#22c55e", size=10))
        fig.update_yaxes(tickvals=[0, 1, 2],
                         ticktext=["RELIABLE", "CAUTION", "UNRELIABLE"])
        fig.update_xaxes(title_text="screening sequence id")
        fig.update_layout(title="Reliability status across the held-out day")
        st.plotly_chart(style_fig(fig, height=300), use_container_width=True)
        st.caption(
            "Lines break at real capture gaps in CSE-CIC-IDS2018; nothing is "
            "interpolated across them."
        )
    else:
        st.info(
            "Per-sequence reliability output not found. Build it with "
            "`python src/build_v4_reliability.py` to see the day-level view."
        )

    # ---------- RETROSPECTIVE FORECAST ERROR ----------
    st.markdown("### Retrospective forecast error analysis")
    st.markdown(
        "> **Retrospective only.** This compares the decoded state at t+k "
        "with the state that was *actually observed* at t+k. The observed "
        "future is read **after** the forecast and the status already exist "
        "and never enters either. It is a diagnostic, not a model input."
    )
    summ_path = Path("results/v4_reliability/reliability_summary.json")
    if summ_path.exists():
        summ = json.loads(summ_path.read_text())
        rfe = summ["retrospective_forecast_error"]
        e1, e2 = st.columns([1, 1])
        with e1:
            st.markdown("**State reconstruction error by horizon**")
            st.dataframe(pd.DataFrame([
                {"k": k, "mean MAE": f"{v['state_mae_mean']:.4f}",
                 "median": f"{v['state_mae_p50']:.4f}",
                 "p90": f"{v['state_mae_p90']:.4f}",
                 "max": f"{v['state_mae_max']:.4f}"}
                for k, v in rfe["by_horizon"].items()
            ]), hide_index=True, use_container_width=True)
        with e2:
            st.markdown("**State error by reliability status**")
            st.dataframe(pd.DataFrame([
                {"status": s, "n": v["n"],
                 "mean MAE": f"{v['state_mae_mean']:.4f}",
                 "median": f"{v['state_mae_p50']:.4f}",
                 "p90": f"{v['state_mae_p90']:.4f}"}
                for s, v in rfe["by_reliability_status"].items()
            ]), hide_index=True, use_container_width=True)

        if rfe["error_increases_with_severity"]:
            st.success(
                "Mean retrospective state error rises monotonically "
                "RELIABLE → CAUTION → UNRELIABLE. The layer therefore "
                "**identifies conditions associated with worse forecasts**. "
                "It has *not* been shown to predict when a specific "
                "attack-label decision is wrong — that is a different and "
                "unestablished claim."
            )
        else:
            st.warning(
                "Mean retrospective state error does **not** rise "
                "monotonically with status severity on this day. Reported "
                "as measured."
            )

        with st.expander("Indicator vs retrospective error (Spearman ρ)"):
            cr = sorted(rfe["indicator_correlations"].items(),
                        key=lambda kv: -abs(
                            kv[1]["spearman_rho_vs_seq_mean_state_mae"]))
            st.dataframe(pd.DataFrame([
                {"indicator": k,
                 "ρ vs sequence-mean state MAE":
                     f"{v['spearman_rho_vs_seq_mean_state_mae']:+.3f}"}
                for k, v in cr
            ]), hide_index=True, use_container_width=True)
            st.caption(rfe["correlation_note"])

        st.markdown("**This window**")
        fvo_path = Path("results/v4_reliability/forecast_vs_observed.csv")
        if fvo_path.exists():
            fv = pd.read_csv(fvo_path)
            fv = fv[fv["sequence_id"] == SEQ_ID]
            if len(fv):
                st.dataframe(pd.DataFrame({
                    "k": fv["horizon_k"],
                    "forecast P": fv["forecast_probability"].map("{:.4f}".format),
                    "state MAE": fv["state_mae"].map("{:.4f}".format),
                    "state RMSE": fv["state_rmse"].map("{:.4f}".format),
                    "max |error|": fv["state_max_abs_error"].map("{:.4f}".format),
                }), hide_index=True, use_container_width=True)
                st.caption(
                    "26-D state in `target_scaler` space. The V4 decoder does "
                    "not beat a persistence baseline at these horizons "
                    "(`docs/LONG_HORIZON_ROLLOUT.md` §11); this table "
                    "measures the frozen model as it is."
                )
    else:
        st.info(
            "Retrospective study not found. Build it with "
            "`python src/build_v4_reliability.py`."
        )

    # ---------- LIMITS ----------
    st.markdown("### What this page does not claim")
    if summ_path.exists():
        for lim in json.loads(summ_path.read_text())["known_limitations"]:
            st.markdown(f"- {lim}")
    st.caption(
        f"Reliability layer `{a.layer_version}` · reference "
        f"`{RLY_REF.source}` ({RLY_REF.n_reference_sequences:,} sequences) · "
        f"V4 frozen, not retrained, not refit, not rethresholded."
    )


elif page == "🗺️ MITRE ATT&CK Timeline":

    st.markdown("# 🗺️ MITRE ATT&CK Context")
    st.markdown(
        "> *A rule-based ATT&CK interpretation of the predicted network "
        "behaviour — an analyst-facing evidence layer on top of the forecast.*"
    )

    st.markdown(
        '<div class="banner banner-gap">'
        '<span class="pill pill-context">rule-based interpretation</span>&nbsp;&nbsp;'
        '<b>The world model does not predict ATT&CK labels.</b>'
        '<br><span class="note">V4 outputs a 26-D network state and a threat '
        'probability — nothing else. The tactic and technique shown here are produced '
        'by <code>src/mitre_mapper.py</code>, a separate <b>deterministic telemetry '
        'rule library</b> that reads the feature values and matches them against '
        'documented ATT&CK-consistent patterns. It is an interpretation layer for '
        'analysts, not a model output, and it was never trained on ATT&CK '
        'labels.</span></div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # LIVE ATT&CK CONTEXT FOR THE SELECTED REPLAY SEQUENCE
    # -------------------------------------------------------------
    # The stored mitre_summary.csv is addressed positionally and was built
    # against a different window set, so it cannot be aligned to the screening
    # substrate. The mapper itself is run directly instead -- same rule library,
    # no redesign, fed the frozen V4 model and the frozen threshold.
    if not using_upload:
        st.divider()
        st.markdown(f"### 🎯 ATT&CK context — sequence #{SEQ_ID}")
        try:
            mrow = SCREENING.preds.iloc[SEQ_ID]
            m_maxfut = float(mrow["max_forecast_probability"])
            m_thr = SCREENING.alert_threshold
            first_alert = (int(np.argmax(SCREENING.alerts))
                           if SCREENING.alerts.any() else None)

            live_mapper = MITREMapper(model=model, threshold=m_thr, device="cpu")
            mapping = live_mapper.map_single_window(
                x_window=SCREENING.X[SEQ_ID],
                window_idx=SEQ_ID,
                timestamp=str(mrow["target_t1_timestamp"]),
                total_windows=SCREENING.n_sequences,
                first_alert_idx=first_alert,
                attack_prob=m_maxfut,
            )
            md = mapping.as_dict() if hasattr(mapping, "as_dict") else vars(mapping)

            tactic = md.get("tactic_name") or "N/A"
            tcolor = TACTIC_COLORS.get(tactic, "#7f8c8d")
            g1, g2, g3 = st.columns(3)
            with g1:
                card("Tactic",
                     f'<span style="color:{tcolor};">{tactic}</span>',
                     f"{md.get('tactic_id') or '—'}")
            with g2:
                card("Technique", md.get("technique_name") or "—",
                     f"{md.get('technique_id') or 'no technique matched'}")
            with g3:
                conf = str(md.get("confidence", "n/a")).upper()
                card("Rule confidence", conf,
                     f"phase: {md.get('temporal_phase', '—')}")

            g4, g5 = st.columns(2)
            with g4:
                card("Forecast driving this context", f"{m_maxfut:.3f}",
                     f"max over t+1…t+5 · frozen threshold {m_thr:.2f}",
                     "card-forecast")
            with g5:
                card("Rule library", f"{len(live_mapper.rules)} rules",
                     "deterministic telemetry rules, z-score space")

            expl_txt = md.get("analyst_explanation") or ""
            if expl_txt:
                st.markdown("#### Analyst explanation")
                st.info(expl_txt)

            evidence = md.get("evidence") or md.get("triggering_features") or ""
            if evidence:
                with st.expander("Matched rule evidence"):
                    st.write(evidence)

            st.markdown(
                '<div class="note">This ATT&CK context is derived from the feature '
                'values in the observed window, independently of the Integrated '
                'Gradients attribution. Neither is derived from the other, so they '
                'are two separate lines of evidence about the same forecast.</div>',
                unsafe_allow_html=True,
            )
        except Exception as exc:                          # noqa: BLE001 - shown to user
            st.error(
                f"**ATT&CK context unavailable** for this sequence.\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )
            st.caption(
                "The rule-based mapper could not be run. No substitute or guessed "
                "ATT&CK context is displayed."
            )

        st.divider()
        st.markdown("### 📚 Stored day-wide mapping")

    if mitre_active is None:
        if using_upload:
            st.error(
                "MITRE mapping for this uploaded dataset is not "
                "available (feature attribution failed during "
                "loading — see the 📥 Data Input page)."
            )
        else:
            st.error(
                "Rule-based ATT&CK mappings are unavailable for the "
                "active benchmark. The stored table "
                "(`models/latent_world_model_v4/mitre_mapping/"
                "mitre_summary.csv`) was generated against a "
                "different window set, and its rows are addressed "
                "positionally, so it cannot be aligned to this "
                "dataset. Regenerate it with "
                "`python src/mitre_mapper.py`."
            )
        st.stop()

    if using_upload:
        upload_source_banner()

    st.divider()

    # Tactic distribution
    st.markdown("### Tactic Distribution")

    alert_df = mitre_active[
        mitre_active["tactic_id"] != "NONE"
    ]

    c1, c2 = st.columns([1, 2])

    with c1:
        tactic_counts = (
            alert_df["tactic_name"]
            .value_counts()
            .reset_index()
        )
        tactic_counts.columns = [
            "Tactic", "Count"
        ]

        fig_pie = px.pie(
            tactic_counts,
            values="Count",
            names="Tactic",
            color="Tactic",
            color_discrete_map=TACTIC_COLORS,
            hole=0.4,
        )
        fig_pie.update_layout(
            template="plotly_dark",
            height=350,
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(
            fig_pie, use_container_width=True
        )

    with c2:
        # Confidence breakdown
        conf_counts = (
            alert_df["confidence"]
            .value_counts()
            .reindex(["high", "medium", "low"],
                     fill_value=0)
        )

        st.markdown("#### Confidence Breakdown")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(
            "🔴 High",
            conf_counts.get("high", 0),
        )
        cc2.metric(
            "🟠 Medium",
            conf_counts.get("medium", 0),
        )
        cc3.metric(
            "🟡 Low",
            conf_counts.get("low", 0),
        )

        # Technique table
        st.markdown("#### Technique Breakdown")
        tech_counts = (
            alert_df[alert_df["technique_name"] != ""]
            ["technique_name"]
            .value_counts()
            .head(8)
            .reset_index()
        )
        tech_counts.columns = [
            "Technique", "Windows"
        ]
        st.dataframe(
            tech_counts,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Timeline
    st.markdown("### MITRE ATT&CK Timeline")

    # Uploaded prepared data without a `timestamp` column carries window
    # indices ("0", "1", ...) as its timestamps; plot those on the window
    # index axis rather than failing to parse them as dates. Benchmark and
    # PCAP timestamps always parse, so their axis is unchanged.
    _ts_parsed = pd.to_datetime(mitre_active["timestamp"], errors="coerce")
    mitre_active["timestamp_dt"] = (
        _ts_parsed if _ts_parsed.notna().all()
        else mitre_active["window_index"]
    )

    if has_ground_truth:
        prob_row, conf_row = 2, 3
        fig_timeline = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.25, 0.5, 0.25],
            subplot_titles=[
                "Ground Truth",
                "Attack Probability + MITRE Tactics",
                "Confidence Level",
            ],
        )

        # Ground truth
        fig_timeline.add_trace(
            go.Scatter(
                x=mitre_active["timestamp_dt"],
                y=mitre_active["ground_truth"],
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.3)",
                line=dict(
                    color="rgba(231,76,60,0.6)"
                ),
                name="Attack (GT)",
                showlegend=True,
            ),
            row=1, col=1,
        )
    else:
        prob_row, conf_row = 1, 2
        fig_timeline = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=[
                "Attack Probability + MITRE Tactics",
                "Confidence Level",
            ],
        )

    # Attack probability
    fig_timeline.add_trace(
        go.Scatter(
            x=mitre_active["timestamp_dt"],
            y=mitre_active["attack_probability"],
            mode="lines",
            line=dict(color="#ecf0f1", width=1),
            name="Attack Prob",
            showlegend=True,
        ),
        row=prob_row, col=1,
    )

    # MITRE tactic markers
    for tactic, color in TACTIC_COLORS.items():
        if tactic == "No Alert":
            continue
        mask = alert_df["tactic_name"] == tactic
        if mask.sum() == 0:
            continue

        tactic_data = alert_df[mask]
        fig_timeline.add_trace(
            go.Scatter(
                x=mitre_active.loc[tactic_data.index, "timestamp_dt"],
                y=tactic_data["attack_probability"],
                mode="markers",
                marker=dict(
                    color=color,
                    size=8,
                    symbol="diamond",
                ),
                name=tactic,
                showlegend=True,
            ),
            row=prob_row, col=1,
        )

    # Threshold
    fig_timeline.add_hline(
        y=threshold, row=prob_row, col=1,
        line_dash="dash", line_color="red",
        annotation_text="Threshold",
    )

    # Confidence
    conf_map = {"n/a": 0, "low": 1,
                "medium": 2, "high": 3}
    conf_values = mitre_active["confidence"].map(
        conf_map
    ).fillna(0)
    conf_colors = [
        CONFIDENCE_COLORS.get(c, "#34495e")
        for c in mitre_active["confidence"]
    ]

    fig_timeline.add_trace(
        go.Bar(
            x=mitre_active["timestamp_dt"],
            y=conf_values,
            marker_color=conf_colors,
            name="Confidence",
            showlegend=False,
        ),
        row=conf_row, col=1,
    )

    fig_timeline.update_layout(
        height=700 if has_ground_truth else 550,
        template="plotly_dark",
        legend=dict(
            orientation="h", y=-0.08,
            x=0.5, xanchor="center",
        ),
    )
    if has_ground_truth:
        fig_timeline.update_yaxes(
            title_text="Label", row=1, col=1,
        )
    fig_timeline.update_yaxes(
        title_text="Prob", row=prob_row, col=1,
    )
    fig_timeline.update_yaxes(
        title_text="Conf",
        tickvals=[0, 1, 2, 3],
        ticktext=["None", "Low", "Med", "High"],
        row=conf_row, col=1,
    )

    st.plotly_chart(
        fig_timeline, use_container_width=True
    )

    # Sample alerts table
    st.divider()
    st.markdown("### 🚨 Recent Alerts")

    alert_display = alert_df[[
        "window_index", "timestamp",
        "attack_probability", "tactic_name",
        "technique_name", "confidence",
        "triggering_features", "explanation",
    ]].head(20)

    st.dataframe(
        alert_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "attack_probability": st.column_config.ProgressColumn(
                "Attack Prob",
                min_value=0, max_value=1,
                format="%.2f",
            ),
        },
    )


# =============================================================
# PAGE: FEATURE ATTRIBUTION
# =============================================================

elif page == "🔍 Feature Attribution":

    st.markdown("# 🔍 Feature Attribution")
    st.markdown(
        "> *Which parts of the observed 10 x 52 network history "
        "contributed to this future-threat prediction?*"
    )
    st.caption(
        f"Integrated Gradients attribution for **P(t+{horizon})** on "
        f"the frozen V4 checkpoint "
        f"`{explainer.provenance.checkpoint_hash[:12]}…`, computed "
        f"from the same inference tensor as the displayed "
        f"probability. Attribution describes model "
        f"sensitivity/contribution — it does not establish "
        f"causality, attacker intent, or attack stage."
    )

    if using_upload:
        upload_source_banner()
    if using_pcap:
        st.warning(
            "**PCAP caveat.** This explanation uses the same V4 "
            "feature contract and transform chain as the training "
            "path, but flow-record production differs, so the "
            "resulting feature values are approximate. They are NOT "
            "numerically equivalent to training features."
        )

    # -------------------------------------------------------------
    # LIVE ATTRIBUTION FOR THE SELECTED REPLAY SEQUENCE
    # -------------------------------------------------------------
    # Computed on demand by the SAME provenance-locked V4Explainer that
    # produces the displayed probability, so the explanation and the number
    # cannot diverge.
    if not using_upload:
        st.divider()
        st.markdown(f"### 🎯 Why this forecast — sequence #{SEQ_ID}")

        xai_n = st.slider("Top contributing features", 5, 25, 8, key="xai_live_n")
        run_live = st.button("Explain this sequence", type="primary")

        if run_live or st.session_state.get("xai_done_for") == SEQ_ID:
            try:
                with st.spinner("Running Integrated Gradients on the frozen V4 model…"):
                    att = explainer.attribute(
                        SCREENING.model_input(SEQ_ID), k=horizon
                    )
                st.session_state["xai_done_for"] = SEQ_ID

                attrs = att["attributions"]
                signed = V4Explainer.feature_aggregate(attrs)["signed"][0]
                order = np.argsort(-np.abs(signed))[:xai_n]

                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    card("Explained probability", f"{float(att['prediction'][0]):.4f}",
                         f"P(t+{horizon}) · {sd.HORIZON_LABELS.get(horizon, '')}",
                         "card-forecast")
                with ac2:
                    card("Baseline probability",
                         f"{float(att['baseline_prediction'][0]):.4f}",
                         "zero-input reference")
                with ac3:
                    card("Completeness error",
                         f"{float(att['completeness_rel_error'][0]):.2e}",
                         "IG sum vs actual prediction delta")

                st.markdown("#### Top contributors")
                rows = []
                for i in order:
                    fname = INPUT_FEATURES[int(i)]
                    val = float(signed[int(i)])
                    rows.append({
                        "Feature": fname,
                        "Attribution": round(val, 6),
                        "Direction": ("▲ toward higher threat" if val > 0
                                      else "▼ toward lower threat"),
                        "What it measures": sd.feature_plain(fname),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)

                bfig = go.Figure(go.Bar(
                    x=[float(signed[int(i)]) for i in order][::-1],
                    y=[INPUT_FEATURES[int(i)] for i in order][::-1],
                    orientation="h",
                    marker_color=[C_ALERT if signed[int(i)] > 0 else C_NOW
                                  for i in order][::-1],
                    hovertemplate="%{y}<br>attribution %{x:.6f}<extra></extra>",
                ))
                bfig.update_layout(
                    title=dict(text=f"Signed attribution toward P(t+{horizon})",
                               font=dict(size=13)),
                    xaxis_title="signed attribution (summed over the 10 timesteps)",
                )
                style_fig(bfig, height=max(300, 26 * len(order)), legend=False)
                st.plotly_chart(bfig, use_container_width=True)

                st.markdown("#### Plain-English reading")
                for r in rows[:5]:
                    st.markdown(
                        f"- **`{r['Feature']}`** "
                        f"({sd.attribution_sentence(r['Feature'], r['Attribution'])}), "
                        f"attribution `{r['Attribution']:+.6f}`"
                    )

                st.markdown(
                    '<div class="note"><b>Attribution provenance.</b> '
                    f'Method: Integrated Gradients (<code>{att["method"]}</code>), '
                    f'{att["n_steps"]} steps, attribution tensor '
                    f'{list(attrs.shape)} = (N, 10, 52). Checkpoint '
                    f'<code>{explainer.provenance.checkpoint_hash[:12]}…</code>, '
                    f'hash-verified against <code>docs/v4_baseline_manifest.json</code> '
                    f'before the explainer would construct. Attribution describes model '
                    f'<i>sensitivity and contribution</i> — it does not establish '
                    f'causality, attacker intent, or attack stage.</div>',
                    unsafe_allow_html=True,
                )
            except Exception as exc:                      # noqa: BLE001 - shown to user
                st.error(
                    f"**Attribution unavailable.** The provenance-locked explainer "
                    f"could not produce an attribution for this sequence.\n\n"
                    f"`{type(exc).__name__}: {exc}`"
                )
                st.caption(
                    "Integrated Gradients requires the `captum` package. Install "
                    "dependencies with `pip install -r requirements.txt`. No "
                    "substitute or approximate explanation is shown."
                )
        else:
            st.info(
                "Press **Explain this sequence** to run Integrated Gradients on the "
                "frozen V4 checkpoint for the current replay sequence. It is computed "
                "on demand because it is the most expensive operation in the console."
            )

    st.divider()

    # Global importance
    st.markdown("### 🌍 Global Feature Importance")

    if importance_active is None:
        if using_upload:
            st.error(
                "Feature attribution for this uploaded dataset is not "
                "available (it may have failed during loading — see "
                "the 📥 Data Input page)."
            )
        else:
            st.error(
                "Precomputed global importance not found. "
                "Run `python src/run_v4_explainability.py` to populate "
                "`results/v4_explainability/`. The per-sequence attribution above "
                "does not depend on it."
            )
        st.stop()

    top_n = st.slider(
        "Number of features to show",
        5, 52, 20,
    )

    top_df = importance_active.head(top_n)

    colors = [
        "#e74c3c" if "_rel_change" in f
        else "#3498db"
        for f in top_df["feature"]
    ]

    fig_imp = go.Figure()
    fig_imp.add_trace(
        go.Bar(
            y=top_df["feature"][::-1],
            x=top_df["mean_abs_attribution"][::-1],
            orientation="h",
            marker_color=colors[::-1],
        )
    )
    fig_imp.update_layout(
        template="plotly_dark",
        height=max(400, top_n * 28),
        xaxis_title="Mean |Attribution|",
        title="Feature Importance (Integrated Gradients)",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # Attribution heatmap
    if attributions_active is None:
        st.warning(
            "Per-window attribution tensor unavailable: the stored "
            "tensor is missing, or its SHA256 does not match the hash "
            "pinned in results/v4_explainability_summary.json. It is "
            "withheld rather than displayed — an attribution whose "
            "originating run cannot be identified must not be shown "
            "next to a prediction. Re-run "
            "`python src/run_v4_explainability.py`."
        )
    if attributions_active is not None:
        st.markdown(
            "### 🌡️ Attribution Heatmap Over Time"
        )

        top10_feats = importance_active.head(10)[
            "feature"
        ].values
        top10_idx = [
            INPUT_FEATURES.index(f)
            for f in top10_feats
        ]

        n = min(attributions_active.shape[0], n_windows_active)
        last_attr = attributions_active[:n, -1, :]
        heatmap_data = last_attr[:, top10_idx].T

        fig_heat = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.3, 0.7],
            subplot_titles=[
                "Attack Probability",
                "Feature Attribution",
            ],
        )

        fig_heat.add_trace(
            go.Scatter(
                x=meta_active["timestamp"][:n],
                y=probs_active[:n],
                fill="tozeroy",
                fillcolor="rgba(52,152,219,0.3)",
                line=dict(
                    color="#3498db", width=1
                ),
                name="Attack Prob",
            ),
            row=1, col=1,
        )

        vmax = float(np.abs(heatmap_data).max())

        fig_heat.add_trace(
            go.Heatmap(
                z=heatmap_data,
                x=meta_active["timestamp"][:n],
                y=list(top10_feats),
                colorscale="RdBu_r",
                zmin=-vmax, zmax=vmax,
                colorbar=dict(
                    title="Attribution",
                    x=1.02,
                ),
            ),
            row=2, col=1,
        )

        fig_heat.update_layout(
            template="plotly_dark",
            height=550,
        )
        st.plotly_chart(
            fig_heat, use_container_width=True
        )

    # Attack vs Benign comparison
    st.divider()
    st.markdown(
        "### ⚔️ Attack vs Benign Attribution"
    )

    if not has_ground_truth:
        st.info(
            "This comparison needs labeled ground truth, "
            "which an uploaded PCAP doesn't have. Switch "
            "Data Source to **📊 Benchmark Dataset** to "
            "see it."
        )
    elif attributions_active is not None:
        n = min(attributions_active.shape[0], n_windows_active)
        last_attr = attributions_active[:n, -1, :]
        attack_mask = (
            meta_active["attack_present"]
            .values[:n] == 1
        )

        if attack_mask.sum() > 0:
            atk_mean = last_attr[attack_mask].mean(
                axis=0
            )
            ben_mean = last_attr[~attack_mask].mean(
                axis=0
            )

            top15_idx = np.argsort(
                np.abs(atk_mean)
            )[::-1][:15]

            fig_comp = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    "Attack Windows",
                    "Benign Windows",
                ],
            )

            feat_names = [
                INPUT_FEATURES[i]
                for i in top15_idx
            ]

            fig_comp.add_trace(
                go.Bar(
                    y=feat_names[::-1],
                    x=[atk_mean[i]
                       for i in top15_idx][::-1],
                    orientation="h",
                    marker_color=[
                        "#e74c3c"
                        if atk_mean[i] > 0
                        else "#3498db"
                        for i in top15_idx
                    ][::-1],
                    name="Attack",
                ),
                row=1, col=1,
            )

            fig_comp.add_trace(
                go.Bar(
                    y=feat_names[::-1],
                    x=[ben_mean[i]
                       for i in top15_idx][::-1],
                    orientation="h",
                    marker_color=[
                        "#e74c3c"
                        if ben_mean[i] > 0
                        else "#3498db"
                        for i in top15_idx
                    ][::-1],
                    name="Benign",
                ),
                row=1, col=2,
            )

            fig_comp.update_layout(
                template="plotly_dark",
                height=500,
                showlegend=False,
            )
            st.plotly_chart(
                fig_comp, use_container_width=True
            )


# =============================================================
# PAGE: WINDOW INSPECTOR
# =============================================================

elif page == "📊 Window Inspector":

    st.markdown("# 📊 Window Inspector")
    st.markdown(
        "> *Deep dive: the observed state, the 10 windows that feed the model, "
        "the future trajectory, the top contributing features and the ATT&CK "
        "context — all for one selected window.*"
    )

    if using_upload:
        upload_source_banner()

    st.divider()

    # Window selector
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_idx = st.slider(
            "Select window index (defaults to the replay position)",
            0, n_windows_active - 1,
            min(SEQ_ID if not using_upload else 0, n_windows_active - 1),
        )

    with col2:
        # Quick jump to alerts
        alert_indices = np.where(
            probs_active >= threshold
        )[0]
        if len(alert_indices) > 0:
            jump_to = st.selectbox(
                "Jump to alert window",
                alert_indices,
                format_func=lambda x: (
                    f"Window {x} "
                    f"(p={probs_active[x]:.2f})"
                ),
            )
            if st.button("Go →"):
                selected_idx = jump_to

    st.divider()

    # Window details
    prob = probs_active[selected_idx]
    ts = meta_active["timestamp"].iloc[selected_idx]

    # Top metrics
    if has_ground_truth:
        mc1, mc2, mc3, mc4 = st.columns(4)
    else:
        mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Window", f"#{selected_idx}")
    mc2.metric("Timestamp", str(ts)[11:19])
    mc3.metric(
        "Attack Probability",
        f"{prob:.3f}",
    )
    if has_ground_truth:
        gt = meta_active["attack_present"].iloc[
            selected_idx
        ]
        gt_label = "🔴 ATTACK" if gt else "🟢 BENIGN"
        mc4.metric("Ground Truth", gt_label)
        st.caption(
            "Ground truth is shown for evaluation transparency only. It is not part "
            "of the model's input and plays no role in the prediction."
        )

    # -------------------------------------------------------------
    # 10-WINDOW OBSERVED HISTORY  (the model's actual input)
    # -------------------------------------------------------------
    st.divider()
    st.markdown("### 🕙 The 10 observed windows that feed the world model")
    st.caption(
        "100 s of history, t-90 → t. These 52 features per window are the ONLY input "
        "the model receives. No label, event region or ground-truth field is included."
    )

    if not using_upload:
        gap_banner(SCREENING, selected_idx)

    hist_feats = st.multiselect(
        "Features to inspect",
        options=sd.STATE_FORECAST_FEATURES,
        default=["flow_count", "window_packet_rate", "syn_rate"],
        key="wi_hist_feats",
    )

    if not hist_feats:
        st.info("Select at least one feature to display the history.")
    else:
        step_labels = ["t" if s == sd.SEQ_LEN - 1
                       else f"t-{(sd.SEQ_LEN - 1 - s) * sd.WINDOW_SECONDS}"
                       for s in range(sd.SEQ_LEN)]
        if not using_upload:
            win_end = SCREENING.preds["observed_window_end"].iloc[selected_idx]
            hist_ts = [win_end - pd.Timedelta(seconds=(sd.SEQ_LEN - 1 - s) * sd.WINDOW_SECONDS)
                       for s in range(sd.SEQ_LEN)]
        else:
            hist_ts = [ts] * sd.SEQ_LEN

        tscaler_wi = load_target_scaler()
        obs_win = tscaler_wi.inverse_transform(X_active[selected_idx][:, :26])

        hist_tbl = pd.DataFrame({
            "Window": step_labels,
            "Timestamp": [str(t)[11:19] for t in hist_ts],
        })
        for f in hist_feats:
            hist_tbl[f] = np.round(obs_win[:, INPUT_FEATURES.index(f)], 3)
        st.dataframe(hist_tbl, use_container_width=True, hide_index=True)

        hfig = go.Figure()
        for f in hist_feats:
            hfig.add_trace(go.Scatter(
                x=step_labels, y=obs_win[:, INPUT_FEATURES.index(f)],
                mode="lines+markers", name=f,
                hovertemplate=f"{f}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
            ))
        hfig.update_layout(
            title=dict(text="Observed feature history (physical units)", font=dict(size=13)),
            xaxis_title="observed window", yaxis_title="value",
        )
        style_fig(hfig, height=300)
        st.plotly_chart(hfig, use_container_width=True)

        # Per-window attack probability where the screening set provides it.
        if not using_upload:
            seg_lo2, _ = SCREENING.segment_of(selected_idx)
            h_lo = max(seg_lo2, selected_idx - (sd.SEQ_LEN - 1))
            h_ids = list(range(h_lo, selected_idx + 1))
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(
                x=[f"t-{(selected_idx - i) * sd.WINDOW_SECONDS}" if i != selected_idx else "t"
                   for i in h_ids],
                y=SCREENING.preds["nowcast_probability_t0"].iloc[h_lo:selected_idx + 1],
                mode="lines+markers", name="Nowcast at each observed window",
                line=dict(color=C_NOW, width=2.2), marker=dict(size=7),
            ))
            pfig.add_hline(y=SCREENING.alert_threshold, line_dash="dash",
                           line_color=C_ALERT, opacity=0.8,
                           annotation_text=f"Frozen threshold "
                                           f"{SCREENING.alert_threshold:.2f}")
            pfig.update_layout(
                title=dict(text="Threat probability across the observed history",
                           font=dict(size=13)),
                yaxis=dict(range=[-0.02, 1.02], title="probability"),
            )
            style_fig(pfig, height=260)
            st.plotly_chart(pfig, use_container_width=True)

    # -------------------------------------------------------------
    # FUTURE TRAJECTORY for this window
    # -------------------------------------------------------------
    if not using_upload:
        st.divider()
        st.markdown("### 🔮 Forecast trajectory for this window")
        wi_probs, _wi_lat, _wi_dec = rollout_detail(selected_idx)
        wi_thr = SCREENING.alert_threshold
        wi_fut = wi_probs[1:]

        tfig = go.Figure()
        tfig.add_vrect(x0=0, x1=sd.HORIZON_SECONDS, fillcolor=C_FORECAST,
                       opacity=0.06, layer="below", line_width=0)
        tfig.add_trace(go.Scatter(
            x=[0], y=[wi_probs[0]], mode="markers", name="Observed nowcast (t)",
            marker=dict(size=13, color=C_NOW),
        ))
        tfig.add_trace(go.Scatter(
            x=[0] + [k * sd.WINDOW_SECONDS for k in range(1, sd.MAX_K + 1)],
            y=[wi_probs[0]] + list(wi_fut),
            mode="lines+markers", name="World-model forecast",
            line=dict(color=C_FORECAST, width=3, dash="dash"),
            marker=dict(size=10, symbol="diamond", color=C_FORECAST),
        ))
        tfig.add_hline(y=wi_thr, line_dash="dash", line_color=C_ALERT,
                       annotation_text=f"Frozen threshold {wi_thr:.2f}",
                       annotation_font=dict(color=C_ALERT, size=11))
        tfig.update_layout(
            xaxis=dict(tickmode="array", tickvals=[0, 10, 20, 30, 40, 50],
                       ticktext=["<b>t</b>", "+10s", "+20s", "+30s", "+40s", "+50s"],
                       title="forecast horizon"),
            yaxis=dict(range=[-0.02, 1.02], title="threat probability"),
        )
        style_fig(tfig, height=300)
        st.plotly_chart(tfig, use_container_width=True)

        wi1, wi2, wi3 = st.columns(3)
        with wi1:
            card("Max forecast", f"{wi_fut.max():.3f}",
                 f"peak at t+{int(np.argmax(wi_fut)) + 1}", "card-forecast")
        with wi2:
            card("Change vs now", f"{wi_fut.max() - wi_probs[0]:+.3f}",
                 f"nowcast {wi_probs[0]:.3f}", "card-forecast")
        with wi3:
            crossed = wi_fut.max() >= wi_thr
            card("Alert status",
                 "<span class='v-alert'>ALERT</span>" if crossed
                 else "<span class='v-ok'>CLEAR</span>",
                 "frozen threshold, unchanged",
                 "card-alert" if crossed else "card-ok")

    # MITRE mapping for this window
    if mitre_active is not None:
        row = mitre_active.iloc[selected_idx]

        st.divider()
        st.markdown("### 🗺️ ATT&CK INTERPRETATION (rule-based)")
        st.caption(
            "Rule-based ATT&CK-consistent interpretation, produced by "
            "src/mitre_mapper.py from raw feature values. It is a "
            "SEPARATE evidence layer: it is not derived from the "
            "model attribution below, and the attribution is not "
            "derived from it."
        )

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "Tactic",
            row.get("tactic_name", "N/A"),
        )
        mc2.metric(
            "Technique",
            row.get("technique_name", "N/A")
            or "N/A",
        )
        mc3.metric(
            "Confidence",
            str(row.get("confidence", "N/A"))
            .upper(),
        )

        explanation = row.get("explanation", "")
        if explanation:
            st.info(f"**Analyst Note:** {explanation}")

        triggers = row.get(
            "triggering_features", ""
        )
        if triggers:
            st.markdown(
                f"**Evidence:** `{triggers}`"
            )

    # Feature attribution for this window
    if (attributions_active is not None
            and selected_idx < attributions_active.shape[0]):

        st.divider()
        st.markdown("### 🔍 MODEL ATTRIBUTION")
        st.caption(
            f"Integrated Gradients on the frozen V4 checkpoint for "
            f"the displayed P(t+{horizon}) — the same computation "
            f"that produced the probability above. Attribution "
            f"describes this model's sensitivity on this input; it "
            f"does not establish causality, attacker intent, attack "
            f"stage, or actual compromise."
        )

        attr = attributions_active[selected_idx, -1, :]
        vals = X_active[selected_idx, -1, :]

        top_k = 10
        abs_attr = np.abs(attr)
        top_idx = np.argsort(abs_attr)[::-1][:top_k]

        attr_data = []
        for idx in top_idx:
            attr_data.append({
                "Feature": INPUT_FEATURES[idx],
                "Value (σ)": float(vals[idx]),
                "Attribution": float(attr[idx]),
                # Descriptive, not diagnostic: an attribution's sign
                # says which way the feature moved THIS model's
                # output, not whether traffic is malicious.
                "Direction": (
                    "↑ increased P" if attr[idx] > 0
                    else "↓ decreased P"
                ),
            })

        attr_df = pd.DataFrame(attr_data)

        # Horizontal bar chart
        fig_attr = go.Figure()
        fig_attr.add_trace(
            go.Bar(
                y=attr_df["Feature"][::-1],
                x=attr_df["Attribution"][::-1],
                orientation="h",
                marker_color=[
                    "#e74c3c" if a > 0
                    else "#3498db"
                    for a in attr_df[
                        "Attribution"
                    ][::-1]
                ],
                text=[
                    f"{v:+.2f}σ"
                    for v in attr_df[
                        "Value (σ)"
                    ][::-1]
                ],
                textposition="outside",
            )
        )
        fig_attr.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Attribution",
            title=(
                f"Window {selected_idx} — "
                f"Top {top_k} Features"
            ),
        )
        st.plotly_chart(
            fig_attr, use_container_width=True
        )

        # Feature value table
        st.dataframe(
            attr_df,
            use_container_width=True,
            hide_index=True,
        )

    # Raw features for this window
    st.divider()
    with st.expander(
        "📋 Raw Feature Values (last timestep)"
    ):
        raw_vals = X_active[selected_idx, -1, :]
        raw_df = pd.DataFrame({
            "Feature": INPUT_FEATURES,
            "Value (z-scored)": raw_vals,
        })
        st.dataframe(
            raw_df,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================
# PAGE: TRAJECTORY ROLLOUT
# =============================================================

elif page == "🔮 Trajectory Rollout":

    st.markdown("# 🔮 K-Step Trajectory Rollout")
    st.markdown(
        "> *The core World Model capability: "
        "simulating future network states to "
        "anticipate attacker progression.*"
    )

    if using_upload:
        upload_source_banner()

    st.divider()

    st.markdown(
        '<div class="banner banner-gap">'
        '<b>This is what makes it a world model, not a classifier.</b>'
        '<br><span class="note">A classifier maps the observed window to one label. '
        'V4 instead encodes 100 s of history into a latent state '
        '<b>z<sub>t</sub></b>, then applies a learned <b>gated residual transition</b> '
        'repeatedly to roll that latent forward. At every step the same two heads read '
        'the rolled-forward latent: a 26-D state decoder and a threat-probability head. '
        'Five transitions = a 50 s forecast.</span></div>',
        unsafe_allow_html=True,
    )

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        default_start = SEQ_ID if not using_upload else 0
        rollout_window = st.slider(
            "Starting window (defaults to the replay position)",
            0, n_windows_active - 1,
            min(default_start, n_windows_active - 1),
        )
    with col2:
        k_steps = st.slider(
            "Rollout steps (K)", 1, 10, sd.MAX_K,
            help="K=5 is the frozen training and evaluation regime.",
        )
        if k_steps > sd.MAX_K:
            st.caption(
                f"⚠ K={k_steps} extends **beyond the frozen K=5 regime** the model was "
                f"trained and evaluated at. Shown as exploration only — no frozen "
                f"threshold or reference metric exists past t+5."
            )

    if not using_upload:
        gap_banner(SCREENING, rollout_window)

    st.divider()

    # Perform rollout
    st.markdown("### Simulated Trajectory")

    with torch.no_grad():
        x = torch.tensor(
            X_active[rollout_window:rollout_window+1],
            dtype=torch.float32,
        )
        z = model.encode(x)

        trajectory_z = [z.numpy()[0]]
        trajectory_prob = []

        z_curr = z
        for step in range(k_steps):
            z_curr = model.transition_step(z_curr)
            logit = model.predict_attack(z_curr)
            prob_step = torch.sigmoid(logit).item()

            trajectory_z.append(
                z_curr.numpy()[0]
            )
            trajectory_prob.append(prob_step)

        # The nowcast read off z_t itself (no transition applied), plus the
        # decoder's reconstruction at every step -- so the chain below can show
        # state AND threat side by side.
        nowcast_z = torch.sigmoid(model.predict_attack(z)).item()
        decoded_chain = np.stack([
            model.decode(torch.tensor(zz[None], dtype=torch.float32)).numpy()[0]
            for zz in trajectory_z
        ])

    # ---------- THE LATENT CHAIN  Z_t -> Z_t+5 ----------
    st.markdown("#### Latent rollout chain")
    st.caption(
        "Each row is one gated residual transition in latent space. The same decoder "
        "and threat head read every step — the model is rolling a state forward, not "
        "emitting one classification."
    )

    z_norms_chain = np.linalg.norm(np.array(trajectory_z), axis=1)
    tscaler_chain = load_target_scaler()
    chain_phys = tscaler_chain.inverse_transform(decoded_chain[:, :26])
    fc_idx = INPUT_FEATURES.index("flow_count")
    pr_idx = INPUT_FEATURES.index("window_packet_rate")

    chain_probs = [nowcast_z] + trajectory_prob
    for step in range(k_steps + 1):
        label = ("Z<sub>t</sub>" if step == 0
                 else f"Z<sub>t+{step}</sub>")
        ahead = ("observed now · no transition" if step == 0
                 else f"{step * sd.WINDOW_SECONDS} s ahead · {step} transition"
                      f"{'s' if step > 1 else ''}")
        p = chain_probs[step]
        beyond = step > sd.MAX_K
        over = (not beyond) and p >= threshold
        accent = (C_MUTED if beyond else (C_ALERT if over else
                  (C_NOW if step == 0 else C_FORECAST)))
        pill = ('<span class="pill pill-now">now</span>' if step == 0
                else ('<span class="pill pill-context">beyond frozen K=5</span>' if beyond
                      else '<span class="pill pill-forecast">forecast</span>'))

        cc1, cc2, cc3, cc4 = st.columns([1.1, 1.5, 1.5, 1.2])
        with cc1:
            st.markdown(
                f'<div class="card" style="border-left:3px solid {accent};'
                f'padding:10px 12px;"><div style="font-size:1.35rem;font-weight:700;">'
                f'{label}</div><div class="card-sub">{pill}</div></div>',
                unsafe_allow_html=True,
            )
        with cc2:
            st.markdown(
                f'<div class="card" style="padding:10px 12px;">'
                f'<div class="card-label">Threat probability</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:{accent};">'
                f'{p:.4f}</div><div class="card-sub">{ahead}</div></div>',
                unsafe_allow_html=True,
            )
        with cc3:
            st.markdown(
                f'<div class="card" style="padding:10px 12px;">'
                f'<div class="card-label">Decoded state (sample)</div>'
                f'<div style="font-size:0.92rem;">'
                f'flow_count <b>{chain_phys[step, fc_idx]:.1f}</b><br>'
                f'packet_rate <b>{chain_phys[step, pr_idx]:.1f}</b></div></div>',
                unsafe_allow_html=True,
            )
        with cc4:
            st.markdown(
                f'<div class="card" style="padding:10px 12px;">'
                f'<div class="card-label">‖Z‖</div>'
                f'<div style="font-size:1.3rem;font-weight:700;">'
                f'{z_norms_chain[step]:.3f}</div>'
                f'<div class="card-sub">32-D latent norm</div></div>',
                unsafe_allow_html=True,
            )
        if step < k_steps:
            st.markdown(
                f'<div style="text-align:center;color:{C_FORECAST};font-size:1.1rem;'
                f'line-height:1;margin:1px 0;">↓<span style="font-size:.68rem;'
                f'color:#8c96ab;"> &nbsp;gated residual transition</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Plot trajectory probabilities
    fig_traj = go.Figure()

    fig_traj.add_trace(
        go.Scatter(
            x=list(range(1, k_steps + 1)),
            y=trajectory_prob,
            mode="lines+markers",
            line=dict(
                color="#e74c3c", width=2
            ),
            marker=dict(size=10),
            name="Predicted Attack Prob",
        )
    )

    fig_traj.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Threshold",
    )

    fig_traj.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Rollout Step (future)",
        yaxis_title="Attack Probability",
        title=f"Trajectory from Window "
              f"#{rollout_window}",
        yaxis=dict(range=[0, 1]),
    )

    st.plotly_chart(
        fig_traj, use_container_width=True
    )

    # Latent state evolution
    st.markdown("### Latent State Evolution")

    z_array = np.array(trajectory_z)
    z_norms = np.linalg.norm(z_array, axis=1)

    col1, col2 = st.columns(2)

    with col1:
        fig_norm = go.Figure()
        fig_norm.add_trace(
            go.Scatter(
                x=list(range(k_steps + 1)),
                y=z_norms,
                mode="lines+markers",
                line=dict(
                    color="#3498db", width=2
                ),
                marker=dict(size=8),
                name="||Z|| (L2 norm)",
            )
        )
        fig_norm.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Step",
            yaxis_title="||Z||",
            title="Latent State Norm",
        )
        st.plotly_chart(
            fig_norm, use_container_width=True
        )

    with col2:
        # Heatmap of latent dims
        fig_z = go.Figure()
        fig_z.add_trace(
            go.Heatmap(
                z=z_array.T,
                x=[f"t+{i}" for i in range(
                    k_steps + 1
                )],
                y=[f"Z_{i}" for i in range(32)],
                colorscale="RdBu_r",
            )
        )
        fig_z.update_layout(
            template="plotly_dark",
            height=300,
            title="Latent Dimensions Over Steps",
        )
        st.plotly_chart(
            fig_z, use_container_width=True
        )

    # Stability assessment
    st.divider()

    if k_steps >= 3:
        growth = z_norms[-1] / max(z_norms[0], 1e-6)
        if growth > 5:
            st.warning(
                f"⚠️ Latent state grew {growth:.1f}× "
                f"over {k_steps} steps. This is "
                f"unusual for V4 — please verify "
                f"the model checkpoint."
            )
        elif growth > 2:
            st.info(
                f"ℹ️ Moderate latent growth "
                f"({growth:.1f}×). "
                f"Predictions remain reliable "
                f"with V4's gated transition."
            )
        else:
            st.success(
                f"✅ Stable trajectory "
                f"({growth:.1f}× growth). "
                f"V4 gated transition keeping "
                f"latent state stable."
            )

    # Explanation
    st.markdown("### 💡 What is Trajectory Rollout?")
    st.markdown("""
    The **World Model** doesn't just classify the
    current state — it **simulates the future**.

    1. **Encode** the current 10-step window into
       latent state Z_t
    2. **Transition** Z_t → Z_{t+1} → Z_{t+2} → ...
       using the **gated residual transition (V4)**
    3. **Classify** each future state: what's the
       attack probability at each step?

    This enables **predictive defence**: detecting
    threats *before* they fully materialise, by
    observing how the latent trajectory evolves.

    **V4 upgrade:** The gated transition keeps latent
    states stable up to K=10+ steps (norm ratio ~1.07×),
    compared to V3 which diverged at K>3 (32× at K=10).
    """)


# =============================================================
# PAGE: LIVE PCAP INGESTION
# =============================================================

elif page == "🧭 Attack Path / Progression":

    st.markdown("# 🧭 Attack Path / Progression")
    st.markdown(
        "> *A forward-looking, ATT&CK-consistent progression hypothesis built "
        "by running the EXISTING telemetry rule library over the frozen V4 "
        "future rollout.*"
    )

    # ---------- MANDATORY BANNER ----------
    st.markdown(
        f'<div class="banner banner-gap">'
        f'<span class="pill pill-context">rule-based interpretation</span>'
        f'&nbsp;&nbsp;<b style="font-size:1.03rem;">{APX_BANNER}</b>'
        f'<br><span class="note">V4 outputs a 26-D network state and a threat '
        f'probability — nothing else. Every tactic and technique below is '
        f'produced by <code>src/mitre_mapper.py</code>, an unmodified '
        f'deterministic rule library, applied to the model\'s own predicted '
        f'future states. No ATT&CK classifier was trained, and the ordering '
        f'shown is an analytical ordering for visualisation, not a claim that '
        f'intrusions follow it.</span></div>',
        unsafe_allow_html=True,
    )

    if using_upload:
        st.info(
            "The progression hypothesis runs on the **screening replay "
            "substrate**. Switch the source back to **Historical Benchmark** "
            "on the **📥 Data Input** page to use this page. It is not "
            "computed for an uploaded dataset, because the K=5 rollout "
            "timing that defines time-to-progression is only meaningful on "
            "the contiguous 10 s "
            "replay."
        )
        st.stop()

    if SCREENING is None:
        st.error("The screening substrate is unavailable, so no progression "
                 "hypothesis can be produced.")
        st.stop()

    _hyp, _err = attack_path_of(SEQ_ID)
    if _err is not None or _hyp is None:
        st.error(
            f"**Progression hypothesis unavailable** for sequence #{SEQ_ID}.\n\n"
            f"`{_err}`"
        )
        st.caption("No substitute or guessed progression is displayed.")
        st.stop()

    gap_banner(SCREENING, SEQ_ID)

    # ---------- RELIABILITY INTEGRATION (Task 15) — NEVER BYPASSED ----------
    if _hyp.reliability_status == "UNRELIABLE":
        st.markdown(
            f'<div class="banner banner-alert">'
            f'<b style="font-size:1.08rem;color:#ef4444;">⛔ PROGRESSION '
            f'HYPOTHESIS UNRELIABLE</b>'
            f'<br><span class="note">{_hyp.reliability_headline}'
            f'<br><b>The estimated time-to-progression is withheld below.</b> '
            f'A precise-looking horizon would imply the forecast it is read '
            f'from is trustworthy, and the Task-15 reliability layer reports '
            f'that it is not. See <b>🛡️ Reliability / Model Health</b>.'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    elif _hyp.reliability_status == "CAUTION":
        st.markdown(
            f'<div class="banner banner-gap">'
            f'<span class="pill" style="background:#f5a52422;color:#f5a524;'
            f'border:1px solid #f5a52459;">⚠ CAUTION</span>&nbsp;&nbsp;'
            f'<b>Forecast produced under unusual operating conditions.</b>'
            f'<br><span class="note">{_hyp.reliability_headline}</span></div>',
            unsafe_allow_html=True,
        )
    elif _hyp.reliability_status == "RELIABLE":
        st.markdown(
            f'<div class="banner banner-ok" style="padding:9px 14px;">'
            f'{reliability_chip(reliability_of(SEQ_ID))}&nbsp;&nbsp;'
            f'<span class="note">Every reliability indicator lies within the '
            f'reference distribution, so the hypothesis is presented '
            f'normally.</span></div>',
            unsafe_allow_html=True,
        )

    # ---------- STATUS HEADLINE ----------
    if _hyp.status == apx.STATUS_NONE:
        st.markdown(
            f'<div class="banner banner-ok">'
            f'<b style="font-size:1.1rem;">NO FORWARD PROGRESSION '
            f'IDENTIFIED</b>'
            f'<br><span class="note">No admitted ATT&CK-consistent transition '
            f'was produced by the rule layer across t+10 s … t+50 s, and the '
            f'frozen alert did not fire (max forecast '
            f'{_hyp.max_forecast_probability:.3f} &lt; '
            f'{_hyp.alert_threshold:.2f}). This is a valid, expected '
            f'output — not a failure.</span></div>',
            unsafe_allow_html=True,
        )
    elif _hyp.status == apx.STATUS_AMBIGUOUS:
        st.markdown(
            f'<div class="banner banner-gap">'
            f'<b style="font-size:1.1rem;">PROGRESSION AMBIGUOUS — ANALYST '
            f'REVIEW</b>'
            f'<br><span class="note">{_hyp.ambiguity_reason}</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "This is the point of keeping the two layers separate: the threat "
            "forecast and the ATT&CK interpretation are independent, and they "
            "are allowed to disagree. The panel reports the disagreement "
            "instead of inventing a stage to resolve it."
        )
    else:
        st.markdown(
            f'<div class="banner banner-alert">'
            f'<b style="font-size:1.1rem;">FORWARD PROGRESSION HYPOTHESIS — '
            f'{len(_hyp.steps)} admitted transition'
            f'{"s" if len(_hyp.steps) != 1 else ""}</b>'
            f'<br><span class="note">Produced entirely by the deterministic '
            f'rule layer over the model\'s predicted future states.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---------- ANALYST DECISION INTEGRATION (Task 17) ----------
    # A progression hypothesis and an automated decision are separate
    # things. When the decision layer abstains, the hypothesis is still
    # shown -- withholding the automated call does not withhold the
    # evidence -- but it is labelled as not actionable automatically.
    _ad2, _ad2_err = analyst_decision_of(SEQ_ID)
    if _ad2 is not None and _ad2.abstained:
        _has_prog = _hyp.status == apx.STATUS_PROGRESSION
        if _has_prog:
            _msg = ("<b>Progression hypothesis available, but automated "
                    "decision withheld.</b><br>The chain below is the rule "
                    "layer's reading of the forecast. The analyst decision "
                    "layer has declined to act on it automatically.")
        elif _hyp.status == apx.STATUS_AMBIGUOUS:
            _msg = ("<b>Automated decision withheld — and the progression "
                    "hypothesis is itself ambiguous.</b><br>That ambiguity "
                    "is reported here as part of the picture the analyst "
                    "must review. Note that Task-16 status is NOT an input "
                    "to the abstention gate (it is downstream of the same "
                    "0.69 boundary); the abstention reasons are listed "
                    "below on their own terms.")
        else:
            _msg = ("<b>Automated decision withheld.</b><br>No forward "
                    "progression was identified for this sequence, so there "
                    "is no progression hypothesis to act on either way.")
        _rsn = "".join(
            f'<li><b>{abs_layer.REASON_GROUP[r]}</b> — '
            f'{abs_layer.REASON_TEXT[r]}</li>' for r in _ad2.reasons)
        st.markdown(
            f'<div class="banner banner-gap">'
            f'<span class="pill pill-context">analyst decision layer</span>'
            f'&nbsp;&nbsp;⛔ <b>{abs_layer.ABSTAIN_HEADLINE}</b> — '
            f'{abs_layer.ABSTAIN_SUBLINE}'
            f'<br><span class="note">{_msg}</span>'
            f'<ul class="note">{_rsn}</ul></div>',
            unsafe_allow_html=True)
    elif _ad2 is not None:
        st.markdown(
            f'<div class="banner {ABS_BANNER[_ad2.decision]}" '
            f'style="padding:9px 14px;">{abs_chip(_ad2)}&nbsp;&nbsp;'
            f'<span class="note">The analyst decision layer returned '
            f'<b>{_ad2.decision}</b> for this window. The progression '
            f'hypothesis below is a separate, rule-based interpretation and '
            f'did not influence that decision.</span></div>',
            unsafe_allow_html=True)

    # ---------- THE CHAIN ----------
    st.divider()
    st.markdown(f"### Progression chain — sequence #{SEQ_ID}")

    def _stage_block(title, stage, tech_id, tech_name, conf, conf_label,
                     when, ttp_txt, withheld, indicators):
        c = APX_STAGE_COLOR.get(stage, "#7f8c8d")
        tech = (f"{tech_id}: {tech_name}" if tech_id
                else ("no technique matched by the winning rule"
                      if stage in apx.STAGE_ORDER else "—"))
        conf_c = APX_CONF_COLOR.get(str(conf_label).upper(), "#94a3b8")
        ttp_html = ""
        if ttp_txt:
            ttp_html = (
                f'<div style="margin-top:6px;">'
                f'<span class="note">Estimated model-based '
                f'time-to-progression</span><br>'
                f'<span style="font-size:1.5rem;font-weight:700;color:'
                f'{"#ef4444" if withheld else c};">{ttp_txt}</span></div>')
        ind = ("".join(f'<code style="margin-right:6px;">{i}</code>'
                       for i in indicators[:6])
               or '<span class="note">—</span>')
        return (
            f'<div style="border:1px solid {c}59;border-left:4px solid {c};'
            f'border-radius:8px;padding:12px 14px;background:{c}11;">'
            f'<div class="note" style="text-transform:uppercase;'
            f'letter-spacing:.06em;">{title}</div>'
            f'<div style="font-size:1.25rem;font-weight:700;color:{c};'
            f'margin:2px 0 4px;">{stage}</div>'
            f'<div class="note">Technique: <b>{tech}</b></div>'
            f'<div class="note">Rule confidence: <b>{str(conf).upper()}</b>'
            + (f' · Progression confidence: <b style="color:{conf_c};">'
               f'{str(conf_label).upper()}</b>' if conf_label else '')
            + f'</div>'
            f'<div class="note">At: <b>{when}</b></div>'
            f'{ttp_html}'
            f'<div style="margin-top:8px;"><span class="note">Supporting '
            f'telemetry</span><br>{ind}</div>'
            f'</div>'
        )

    st.markdown(
        _stage_block(
            "Current stage (observed window t)",
            _hyp.current_stage, _hyp.current_technique_id,
            _hyp.current_technique_name, _hyp.current_rule_confidence, None,
            _hyp.current_timestamp, None, False,
            _hyp.current_supporting_features),
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="note">Nowcast probability at t: '
        f'<b>{_hyp.current_threat_probability:.3f}</b> · frozen threshold '
        f'<b>{_hyp.alert_threshold:.2f}</b>. The current stage is gated on the '
        f'<b>nowcast</b>, because it is a statement about <i>now</i>; when the '
        f'nowcast is below threshold the rule layer reports <code>No Alert</code> '
        f'even though the 50 s forecast may be elevated.</div>',
        unsafe_allow_html=True)

    if _hyp.anchor_origin and _hyp.anchor_origin != "observed_current":
        st.markdown(
            f'<div class="note" style="margin-top:8px;">↓</div>'
            f'<div class="banner banner-gap" style="padding:8px 12px;">'
            f'<span class="note">The observed window carries no tactic, so the '
            f'chain is anchored at the first <b>projected</b> horizon that '
            f'does: </span>'
            f'{apx_stage_chip(_hyp.anchor_stage)}'
            f'<span class="note"> at t+{_hyp.anchor_k * 10} s.</span></div>',
            unsafe_allow_html=True)

    for _i, _st in enumerate(_hyp.steps):
        _t, _withheld = apx_timing_text(_hyp, _st)
        st.markdown('<div style="text-align:center;font-size:1.6rem;'
                    'color:#64748b;line-height:1;">↓</div>',
                    unsafe_allow_html=True)
        st.markdown(
            _stage_block(
                ("Next projected stage" if _i == 0
                 else "Following possible stage"),
                _st.to_stage, _st.technique_id, _st.technique_name,
                _st.rule_confidence, _st.progression_confidence,
                _st.timestamp, _t, _withheld, _st.supporting_features),
            unsafe_allow_html=True)
        st.markdown(
            f'<div class="note">Threat probability at this horizon '
            f'<b>{_st.threat_probability:.3f}</b> · persists across '
            f'<b>{_st.persistence_horizons}</b> consecutive horizon'
            f'{"s" if _st.persistence_horizons != 1 else ""} '
            f'({_st.evidence_window}) · threat at/above threshold throughout: '
            f'<b>{"yes" if _st.threat_stable else "no"}</b></div>',
            unsafe_allow_html=True)
        with st.expander(f"Why this transition is admitted "
                         f"({_st.from_stage} → {_st.to_stage})"):
            st.markdown(f"**Graph edge justification.** {_st.edge_justification}")
            if _st.supporting_values:
                st.markdown("**Matched rule evidence** (frozen input-scaler "
                            "z-units, read from the predicted future state):")
                st.dataframe(
                    pd.DataFrame([
                        {"feature": k, "z-value": round(v, 4),
                         "plain meaning": sd.feature_plain(k)}
                        for k, v in _st.supporting_values.items()]),
                    use_container_width=True, hide_index=True)

    # ---------- PER-HORIZON TABLE ----------
    st.divider()
    st.markdown("### Rule reading at every horizon")
    st.caption(
        "The same deterministic rule library, applied to the observed window "
        "(t) and to each of the model's five predicted future states. Only "
        "transitions admitted by the progression graph become chain steps; "
        "the rest are shown here for completeness."
    )
    _hz = pd.DataFrame([{
        "horizon": h.horizon_label,
        "timestamp": h.timestamp,
        "threat prob": round(h.threat_probability, 4),
        "tactic": h.tactic_name,
        "tactic id": h.tactic_id,
        "technique": (f"{h.technique_id}: {h.technique_name}"
                      if h.technique_id else "—"),
        "rule confidence": h.rule_confidence.upper(),
        "phase": h.temporal_phase,
        "supporting telemetry": ", ".join(h.supporting_features) or "—",
    } for h in _hyp.horizon_stages])
    st.dataframe(_hz, use_container_width=True, hide_index=True)

    # ---------- XAI LINK (Task 8 attributions) ----------
    st.divider()
    st.markdown("### 🔍 Telemetry contributing to the model's predicted "
                "future state")
    st.caption(
        "These are Integrated Gradients attributions for the frozen V4 "
        "forecast at the relevant horizon. They describe which observed input "
        "features moved the model's predicted future state — **not** what "
        "caused an attack. Attribution and the ATT&CK rule reading are two "
        "independent lines of evidence; neither is derived from the other."
    )
    _target_step = _hyp.next_step
    if _target_step is None:
        st.info(
            "No projected stage exists for this sequence, so there is no "
            "future state to attribute. Per-horizon attributions for the "
            "forecast itself are on the **🔍 Feature Attribution** page."
        )
    else:
        _want_k = _target_step.first_horizon_k
        _avail = [1, 3, 5]
        _use_k = _want_k if _want_k in _avail else min(
            _avail, key=lambda a: (abs(a - _want_k), a))
        if _use_k != _want_k:
            st.warning(
                f"The Task-8 explainability run produced attribution tensors "
                f"for horizons **k = 1, 3, 5 only**. The projected stage is at "
                f"**k = {_want_k} (t+{_want_k * 10} s)**, which has no tensor, "
                f"so the nearest available horizon **k = {_use_k} "
                f"(t+{_use_k * 10} s)** is shown instead and labelled as such. "
                f"Nothing was interpolated."
            )
        _attr = load_attributions(_use_k)
        if _attr is None:
            st.error(
                f"Attribution tensor for k={_use_k} is unavailable or failed "
                f"its hash pin, so it is refused rather than displayed. "
                f"Regenerate with `python src/run_v4_explainability.py`."
            )
        elif _attr.shape[0] != SCREENING.n_sequences:
            st.error(
                f"Attribution tensor has {_attr.shape[0]} rows but the "
                f"substrate has {SCREENING.n_sequences} sequences, so rows "
                f"cannot be aligned. Refused rather than displayed."
            )
        else:
            _names = SCREENING.feature_names
            _a = np.asarray(_attr[SEQ_ID], dtype=float)      # (10, 52)
            _signed = _a.sum(axis=0)
            _order = np.argsort(-np.abs(_signed))[:10]
            _top = pd.DataFrame([{
                "feature": _names[j],
                "signed attribution": round(float(_signed[j]), 6),
                "direction": ("toward higher predicted threat"
                              if _signed[j] > 0
                              else "toward lower predicted threat"),
                "plain meaning": sd.feature_plain(_names[j]),
            } for j in _order])
            _fig = go.Figure(go.Bar(
                x=_top["signed attribution"][::-1],
                y=_top["feature"][::-1],
                orientation="h",
                marker_color=[C_ALERT if v > 0 else C_NOW
                              for v in _top["signed attribution"][::-1]],
            ))
            _fig.update_layout(
                title=(f"Top 10 contributing input features — forecast horizon "
                       f"k={_use_k} (t+{_use_k * 10} s), sequence #{SEQ_ID}"),
                xaxis_title="signed Integrated Gradients attribution "
                            "(summed over the 10 observed timesteps)",
                yaxis_title="")
            st.plotly_chart(style_fig(_fig, height=420),
                            use_container_width=True)
            with st.expander("Attribution values"):
                st.dataframe(_top, use_container_width=True, hide_index=True)

    # ---------- THE PROGRESSION GRAPH ----------
    st.divider()
    st.markdown("### The progression graph used for forward visualisation")
    st.caption(
        "ATT&CK tactics are **not** universally ordered — real intrusions "
        "skip, repeat and interleave stages. This graph is a small, "
        "project-specific analytical ordering over exactly the tactics the "
        "existing rule library can emit. Its only job is to decide which "
        "forward transitions this panel is willing to draw, so that a noisy "
        "relabelling between two adjacent horizons is not presented as a "
        "progression. `No Alert` and `Unknown / Insufficient Evidence` carry "
        "no tactic semantics and are deliberately not nodes."
    )
    _g = apx.describe_progression_graph()
    st.dataframe(
        pd.DataFrame([{"from": e["from"], "to": e["to"],
                       "why this edge is admitted": e["justification"]}
                      for e in _g["edges"]]),
        use_container_width=True, hide_index=True)
    with st.expander(f"Edges deliberately REJECTED ({len(_g['rejected_edges'])})"):
        st.caption(
            "Recorded so the omissions are auditable too — a generic kill "
            "chain would include several of these."
        )
        st.dataframe(
            pd.DataFrame([{"edge": r["edge"], "reason": r["reason"]}
                          for r in _g["rejected_edges"]]),
            use_container_width=True, hide_index=True)

    # ---------- DEFINITIONS ----------
    st.divider()
    st.markdown("### How to read this page")
    _d1, _d2 = st.columns(2)
    with _d1:
        _ttp = apx.TIME_TO_PROGRESSION
        st.markdown(
            f"**Estimated model-based time-to-progression**\n\n"
            f"`{_ttp['formula']}`\n\n"
            f"{_ttp['why_10s']}. *k* is {_ttp['k_definition']}\n\n"
            f"**It is not:**\n"
            + "\n".join(f"- {x}" for x in _ttp["is_not"])
        )
    with _d2:
        _cc = apx.CONFIDENCE_CRITERIA
        st.markdown(
            f"**Progression confidence** — {_cc['nature']}\n\n"
            f"- **HIGH** — {_cc['HIGH']}\n"
            f"- **MEDIUM** — {_cc['MEDIUM']}\n"
            f"- **LOW** — {_cc['LOW']}\n\n"
            f"{_cc['no_tuning']}"
        )

    with st.expander("What this page does not claim"):
        st.markdown(
            "- The world model does **not** predict ATT&CK labels. It predicts "
            "a 26-D network state and a threat probability.\n"
            "- No ATT&CK classifier was trained. Every tactic and technique "
            "string comes verbatim from `src/mitre_mapper.py`.\n"
            "- The progression hypothesis is **not** a statement of attacker "
            "intent, and the rules cannot observe intent — only telemetry.\n"
            "- The time-to-progression is **not** a guaranteed time until an "
            "attacker progresses, and it is not calibrated against observed "
            "progression times.\n"
            "- Future stages are read from a **decoded prediction**; the "
            "decoder's reconstruction error at longer horizons is inherited, "
            "not corrected.\n"
            "- Only **3** documented onset episodes exist in this replay "
            "substrate, and no stage-level ground-truth label exists at all, "
            "so no attack-stage accuracy, precision, recall or F1 is claimed "
            "or computable."
        )
        st.markdown("**Inherited limitations of the rule library**")
        for _l in apx.layer_metadata()["rule_library"]["known_limitations"]:
            st.markdown(f"- {_l}")

elif page == "🧑‍💻 Analyst Decision":

    st.markdown("# 🧑‍💻 Analyst Decision")
    st.markdown(
        "> *A three-way human-in-the-loop decision layer: the model either "
        "makes an automated call, or explicitly declines to and hands the "
        "window to an analyst.*"
    )

    _policy, _perr = load_abstain_policy()
    if _policy is None:
        st.error(
            f"**The frozen decision policy is unavailable**, so no decision "
            f"is shown.\n\n`{_perr}`"
        )
        st.caption(
            "There is deliberately no fallback. A decision layer that "
            "invented its own gate when the frozen one is missing would be "
            "worse than no decision layer at all."
        )
        st.stop()

    st.markdown(
        f'<div class="banner banner-gap">'
        f'<span class="pill pill-context">decision-support layer</span>'
        f'&nbsp;&nbsp;<b>ABSTAIN means the model declines to make an '
        f'automated decision — not that the model has no signal.</b>'
        f'<br><span class="note">The forecast probability is always produced '
        f'and always shown. This layer decides only whether that forecast '
        f'should be <i>acted on automatically</i>. It is deterministic, it '
        f'contains no learned component, and it never moves the frozen '
        f'<b>{_policy.alert_threshold:.2f}</b> decision boundary. No '
        f'calibration was fitted, so nothing here is a calibrated '
        f'probability of correctness.</span></div>',
        unsafe_allow_html=True,
    )

    if using_upload:
        substrate_only_notice("The analyst decision layer")
        st.stop()

    if SCREENING is None:
        st.error("The screening substrate is unavailable, so no decision "
                 "can be produced.")
        st.stop()

    _d, _derr = analyst_decision_of(SEQ_ID)
    if _d is None:
        st.error(
            f"**No decision available** for sequence #{SEQ_ID}.\n\n`{_derr}`"
        )
        st.caption("No guessed or substitute decision is displayed.")
        st.stop()

    gap_banner(SCREENING, SEQ_ID)

    # ---------- THE DECISION ----------
    _c = ABS_COLOR[_d.decision]
    st.markdown(
        f'<div class="banner {ABS_BANNER[_d.decision]}" '
        f'style="padding:18px 20px;">'
        f'<div class="note" style="text-transform:uppercase;'
        f'letter-spacing:.08em;">Decision — sequence #{SEQ_ID}</div>'
        f'<div style="font-size:2.1rem;font-weight:800;color:{_c};'
        f'line-height:1.15;margin:2px 0 6px;">'
        f'{ABS_ICON[_d.decision]} {_d.display}</div>'
        f'<span class="note">{_d.semantics}</span></div>',
        unsafe_allow_html=True,
    )

    if _d.abstained:
        st.markdown(
            f'<div class="banner banner-alert">'
            f'<b style="font-size:1.12rem;color:#f5a524;">⛔ '
            f'{abs_layer.ABSTAIN_HEADLINE}</b>'
            f'<br><b>{abs_layer.ABSTAIN_SUBLINE}</b>'
            f'<br><span class="note">This is not a missing answer. The '
            f'forecast below was produced normally; the layer has judged that '
            f'acting on it <i>automatically</i> is not supported by the '
            f'evidence, and is routing the window to a human.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Why the automated decision was withheld")
        for r in _d.reasons:
            st.markdown(
                f"- **{abs_layer.REASON_GROUP[r].upper()}** "
                f"(`{r}`) — {abs_layer.REASON_TEXT[r]}")

    # ---------- WHY: THE FOUR SIGNAL FAMILIES ----------
    st.divider()
    st.markdown("### Why? — the evidence at decision time")
    st.caption(
        "These four families are deliberately kept separate. Observed-now, "
        "forecast, reliability and ATT&CK interpretation answer different "
        "questions, and only the first three participate in the decision."
    )

    _q1, _q2, _q3, _q4 = st.columns(4)
    with _q1:
        card("Observed now (t)", f"{_d.nowcast:.3f}" if _d.nowcast is not None
             else "—", "nowcast · context only, not a gate input", "card-now")
    with _q2:
        card("Forecast (max t+1…t+5)", f"{_d.max_forecast:.3f}",
             f"frozen boundary {_policy.alert_threshold:.2f} · "
             f"{_d.side} it", "card-forecast")
    with _q3:
        _rl = reliability_of(SEQ_ID)
        card("Reliability", _d.reliability_status,
             "Task-15 · hard gate" if _d.reliability_status == "UNRELIABLE"
             else "Task-15",
             RLY_CARD.get(_d.reliability_status, ""))
    with _q4:
        card("Decision margin", f"{_d.margin:.3f}",
             f"abstain band ±{_policy.margin_delta:.2f} "
             f"(validation-derived)",
             "card-alert" if _d.margin < _policy.margin_delta else "card-ok")

    # ---------- FORECAST TRAJECTORY ----------
    st.markdown("#### Forecast trajectory")
    _probs = SCREENING.probs[SEQ_ID]
    _fx = ["t (now)"] + [sd.HORIZON_LABELS[k] for k in range(1, sd.MAX_K + 1)]
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_fx[1:], y=_probs[1:], mode="lines+markers", name="forecast",
        line=dict(color=C_FORECAST, width=3), marker=dict(size=9)))
    _fig.add_trace(go.Scatter(
        x=[_fx[0]], y=[_probs[0]], mode="markers", name="nowcast",
        marker=dict(color=C_NOW, size=12, symbol="diamond")))
    _fig.add_hline(y=_policy.alert_threshold, line_dash="dash",
                   line_color=C_ALERT,
                   annotation_text=f"frozen boundary "
                                   f"{_policy.alert_threshold:.2f}")
    _fig.add_hrect(y0=max(0.0, _policy.alert_threshold - _policy.margin_delta),
                   y1=min(1.0, _policy.alert_threshold + _policy.margin_delta),
                   fillcolor="#f5a524", opacity=0.14, line_width=0,
                   annotation_text=f"abstain band ±{_policy.margin_delta:.2f}",
                   annotation_position="top left")
    _fig.update_layout(yaxis_title="threat probability",
                       yaxis_range=[0, 1])
    st.plotly_chart(style_fig(_fig, height=340), use_container_width=True)

    # ---------- ATT&CK INTERPRETATION (rationale only) ----------
    _hy, _hyerr = attack_path_of(SEQ_ID)
    st.markdown("#### ATT&CK interpretation")
    if _hy is None:
        st.caption(f"Attack-path hypothesis unavailable (`{_hyerr}`).")
    else:
        _nx = _hy.next_step
        st.markdown(
            f"- Task-16 status: **{apx.STATUS_DISPLAY[_hy.status]}**\n"
            f"- Current stage: **{_hy.current_stage}**"
            + (f"\n- Next projected: **{_nx.to_stage}** at "
               f"**+{_nx.time_to_progression_s} s**" if _nx else ""))
        st.info(
            "**Not a gate input.** Measured on the validation split, the "
            "Task-16 status partitions exactly on the same 0.69 boundary "
            "(NO_FORWARD_PROGRESSION ⟺ no alert), so it carries no "
            "information independent of the ALERT/BENIGN bit and cannot "
            "inform the abstention gate. It is shown here as analyst "
            "rationale, which is a different role."
        )

    # ---------- TRIGGERING CONDITIONS ----------
    st.divider()
    st.markdown("### Triggering conditions — the policy, evaluated in order")
    _rows = []
    _c1_fire = _d.reliability_status == "UNRELIABLE"
    _c2a_fire = _d.margin < _policy.margin_delta
    _c2b_fire = _policy.caution_abstains and _d.reliability_status == "CAUTION"
    _rows.append({
        "order": 1, "branch": "HARD SAFETY RULE",
        "condition": "reliability == UNRELIABLE",
        "observed": _d.reliability_status,
        "fired": "YES → ABSTAIN" if _c1_fire else "no"})
    _rows.append({
        "order": 2, "branch": "VALIDATION-FROZEN GATE",
        "condition": f"|forecast − {_policy.alert_threshold:.2f}| < "
                     f"{_policy.margin_delta:.2f}",
        "observed": f"{_d.margin:.4f}",
        "fired": "YES → ABSTAIN" if _c2a_fire else "no"})
    _rows.append({
        "order": 2, "branch": "VALIDATION-FROZEN GATE",
        "condition": ("reliability == CAUTION" if _policy.caution_abstains
                      else "reliability == CAUTION (disabled by policy)"),
        "observed": _d.reliability_status,
        "fired": "YES → ABSTAIN" if _c2b_fire else "no"})
    _rows.append({
        "order": 3, "branch": "FROZEN RULE (unchanged)",
        "condition": f"forecast > {_policy.alert_threshold:.2f} → ALERT "
                     f"else BENIGN",
        "observed": f"{_d.max_forecast:.4f} ({_d.side})",
        "fired": ("not reached — abstained" if _d.abstained
                  else f"YES → {_d.decision}")})
    st.dataframe(pd.DataFrame(_rows), use_container_width=True,
                 hide_index=True)

    st.markdown(
        f'<div class="note">The frozen rule\'s own call for this window '
        f'(ordinary thresholding, no abstention) was '
        f'<b>{"ALERT" if _d.frozen_alert_bit else "BENIGN"}</b>. This layer '
        f'never flips ALERT↔BENIGN — it only withholds, which is verified as '
        f'a hard gate across all {SCREENING.n_sequences} replay '
        f'sequences.</div>',
        unsafe_allow_html=True)

    # ---------- SUBSTRATE-WIDE PICTURE ----------
    st.divider()
    st.markdown("### Across the whole replay")
    _dec_path = Path("results/abstain/per_sequence_decisions.csv")
    _man_path = Path("results/abstain/abstain_manifest.json")
    if not _dec_path.exists() or not _man_path.exists():
        st.caption(
            "Substrate-wide decision counts and the coverage-vs-selective-risk "
            "table are unavailable. Build them with "
            "`python src/build_abstain.py`."
        )
    else:
        _mn = json.loads(_man_path.read_text())
        _res = _mn["results"]["test"]
        _dcnt = _res["decision_counts"]
        _sel = _res["selective"]
        _base = _mn["results"]["comparison_baseline_vs_abstain"]["test"]["baseline"]

        _k1, _k2, _k3, _k4 = st.columns(4)
        with _k1:
            card("ALERT", f"{_dcnt.get('ALERT', 0):,}",
                 "automated warnings issued", "card-alert")
        with _k2:
            card("BENIGN", f"{_dcnt.get('BENIGN', 0):,}",
                 "automated clears", "card-ok")
        with _k3:
            card("ABSTAIN", f"{_dcnt.get('ABSTAIN', 0):,}",
                 "routed to an analyst", "card-forecast")
        with _k4:
            card("Coverage", f"{100 * _sel['coverage']:.1f}%",
                 "share decided automatically")

        st.markdown("#### Baseline vs abstain policy — held-out March-2 test")
        st.caption(
            "Neither column is 'best'. They answer different questions: the "
            "baseline decides every window; the abstain policy decides fewer "
            "windows and routes the rest to a human."
        )
        _cmp = pd.DataFrame([
            {"metric": "windows decided",
             "baseline (no abstain)": _base["n_non_abstained"],
             "abstain policy": _sel["n_non_abstained"]},
            {"metric": "coverage",
             "baseline (no abstain)": f"{100 * _base['coverage']:.2f}%",
             "abstain policy": f"{100 * _sel['coverage']:.2f}%"},
            {"metric": "abstained",
             "baseline (no abstain)": 0,
             "abstain policy": _sel["abstain_count"]},
            {"metric": "selective error rate",
             "baseline (no abstain)": f"{_base['selective_error_rate']:.4f}",
             "abstain policy": f"{_sel['selective_error_rate']:.4f}"},
            {"metric": "selective FPR",
             "baseline (no abstain)": f"{_base['selective_fpr']:.4f}",
             "abstain policy": f"{_sel['selective_fpr']:.4f}"},
            {"metric": "selective FNR",
             "baseline (no abstain)": f"{_base['selective_fnr']:.4f}",
             "abstain policy": f"{_sel['selective_fnr']:.4f}"},
            {"metric": "selective F1",
             "baseline (no abstain)": f"{_base['selective_f1']:.4f}",
             "abstain policy": f"{_sel['selective_f1']:.4f}"},
        ])
        st.dataframe(_cmp, use_container_width=True, hide_index=True)

        _risk_path = Path("results/abstain/test_selective_risk.csv")
        if _risk_path.exists():
            st.markdown("#### Coverage vs selective risk")
            _rk = pd.read_csv(_risk_path)
            _show = _rk[_rk["margin_delta"].isin(
                [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])]
            _f2 = make_subplots(specs=[[{"secondary_y": True}]])
            _f2.add_trace(go.Scatter(
                x=_rk["coverage"], y=_rk["selective_error_rate"],
                mode="lines+markers", name="selective error",
                line=dict(color=C_ALERT)), secondary_y=False)
            _f2.add_trace(go.Scatter(
                x=_rk["coverage"], y=_rk["selective_f1"],
                mode="lines+markers", name="selective F1",
                line=dict(color=C_FORECAST)), secondary_y=True)
            _frozen = _rk[_rk["is_frozen_policy_delta"]]
            if len(_frozen):
                _f2.add_trace(go.Scatter(
                    x=_frozen["coverage"], y=_frozen["selective_error_rate"],
                    mode="markers", name="frozen policy",
                    marker=dict(color="#ffffff", size=15,
                                symbol="star",
                                line=dict(color=C_ALERT, width=2))),
                    secondary_y=False)
            _f2.update_xaxes(title_text="coverage (share decided automatically)")
            _f2.update_yaxes(title_text="selective error rate",
                             secondary_y=False)
            _f2.update_yaxes(title_text="selective F1", secondary_y=True)
            st.plotly_chart(style_fig(_f2, height=380),
                            use_container_width=True)
            st.warning(
                "**Read this curve carefully.** Selective error falls as "
                "coverage drops, but on this data that is largely a "
                "class-imbalance artefact: the model's positive predictions "
                "cluster just above 0.69, so a wide abstain band withholds "
                "nearly every ALERT and the retained set approaches 'always "
                "BENIGN'. That is why the policy's margin was selected under "
                "an **F1 floor** rather than by minimising error, and why "
                "the table below reports the retained-negative share on "
                "every row."
            )
            st.dataframe(
                _show[["margin_delta", "coverage", "abstention_rate",
                       "selective_error_rate", "selective_fpr",
                       "selective_fnr", "selective_f1",
                       "retained_negative_share", "is_frozen_policy_delta"]],
                use_container_width=True, hide_index=True)

    # ---------- THE POLICY ----------
    st.divider()
    st.markdown("### The frozen policy")
    _pd = _policy.to_dict()
    for _line in _pd["policy"]["evaluation_order"]:
        st.markdown(f"- {_line}")
    _g1, _g2 = st.columns(2)
    with _g1:
        st.markdown(
            f"**Frozen boundary** `{_policy.alert_threshold:.2f}`\n\n"
            f"Source: {_policy.threshold_source}\n\n"
            f"**Not chosen by this layer** and not modified by it."
        )
    with _g2:
        st.markdown(
            f"**Abstention margin** `±{_policy.margin_delta:.2f}`\n\n"
            f"{abs_layer.MARGIN_SELECTION_RULE['rule']}.\n\n"
            f"Split used: {abs_layer.MARGIN_SELECTION_RULE['split_used']}. "
            f"Held-out test labels were not read before this number was "
            f"frozen."
        )
    with st.expander("Why an F1 floor rather than minimising error?"):
        st.markdown(abs_layer.MARGIN_SELECTION_RULE["why_this_rule"])
    with st.expander("Signals investigated and REJECTED as gate inputs"):
        for _k, _v in abs_layer.REJECTED_SIGNALS.items():
            st.markdown(f"- **`{_k}`** — {_v}")
    with st.expander("What this layer does not claim"):
        st.markdown(
            "- It does **not** claim abstention improves the system. It "
            "reduces false-positive rate at a coverage cost and slightly "
            "worsens false-negative rate; whether that trade is worth making "
            "is an operational choice.\n"
            "- It contains **no learned component** and **no fitted "
            "calibration**, so no output is a calibrated probability of "
            "correctness.\n"
            "- The validation split contains **zero UNRELIABLE windows**, so "
            "the hard UNRELIABLE→ABSTAIN rule is a declared safety rule "
            "inherited from Task 15, not one measured on data.\n"
            "- The CAUTION branch rests on a **small validation sample**, so "
            "it is directional evidence rather than a stable effect.\n"
            "- It does **not** replace the Task-15 reliability layer and "
            "cannot override it."
        )

elif page == "🛡️ Defend Now Simulator":

    st.markdown("# 🛡️ Defend Now Simulator")
    st.markdown(
        "> *What would the frozen V4 World Model forecast if a defender "
        "applied a defined intervention to the current network state right "
        "now?*"
    )

    # ---------- MANDATORY DISCLAIMER ----------
    st.markdown(
        f'<div class="banner banner-gap">'
        f'<span class="pill pill-context">counterfactual</span>&nbsp;&nbsp;'
        f'<b style="font-size:1.06rem;">{DSIM_DISCLAIMER}</b>'
        f'<br><span class="note">{DSIM_BODY}'
        f'<br>V4 was never shown an intervention, never given an action '
        f'variable and never trained on (action, outcome) pairs, so it '
        f'<b>cannot</b> have learned intervention effects. Nothing on this '
        f'page executes any action on any real network.</span></div>',
        unsafe_allow_html=True,
    )

    if using_upload:
        substrate_only_notice("The defend-now simulator")
        st.stop()

    if SCREENING is None:
        st.error("The screening substrate is unavailable, so no counterfactual "
                 "can be produced.")
        st.stop()

    # Sequence queued from the Live Forecast button, else the sidebar choice.
    _seq = int(st.session_state.get("dsim_seq", SEQ_ID))
    if _seq != SEQ_ID:
        st.caption(f"Showing sequence **#{_seq}**, queued from the Live "
                   f"Forecast page. Change the sidebar Sequence ID to "
                   f"override.")
        _seq = SEQ_ID if st.session_state.get("dsim_seq") is None else _seq

    # ---------- CONTROLS ----------
    _k1, _k2 = st.columns([2, 2])
    with _k1:
        _iv_keys = list(dsim.INTERVENTIONS)
        _iv = st.selectbox(
            "Intervention",
            _iv_keys,
            format_func=lambda k: dsim.INTERVENTIONS[k].display,
            help="Each intervention maps only onto features whose "
                 "transformation follows from the frozen feature contract.")
    with _k2:
        _strength = st.select_slider(
            "Simulated intervention strength",
            options=[0.0, 0.10, 0.25, 0.50, 0.75],
            value=0.25,
            format_func=lambda v: f"{int(v * 100)}%",
            help="A perturbation magnitude applied to the model's INPUT "
                 "state. This is NOT a real-world effectiveness, efficacy or "
                 "success rate.")
    st.markdown(
        '<div class="note">Strength is a <b>simulated intervention '
        'strength</b> — the size of the perturbation applied to the model '
        'input. It is <b>not</b> a real-world effectiveness.</div>',
        unsafe_allow_html=True)

    _sc, _feat, _extras, _err = defense_scenario(_seq, _iv, _strength)
    if _sc is None:
        st.error(f"**Counterfactual unavailable** for sequence #{_seq}.\n\n"
                 f"`{_err}`")
        st.caption("No substitute or guessed projection is displayed.")
        st.stop()

    gap_banner(SCREENING, _seq)

    _iv_def = dsim.INTERVENTIONS[_iv]

    # ---------- SIDE-BY-SIDE TRAJECTORIES ----------
    st.divider()
    st.markdown(f"### No action vs defend now — sequence #{_seq}")

    _hz = ["t (now)"] + [f"+{k * dsim.WINDOW_SECONDS}s"
                         for k in range(1, dsim.MAX_K + 1)]
    _thr = SCREENING.alert_threshold

    _f = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.07,
        subplot_titles=("NO ACTION — frozen forecast",
                        f"DEFEND NOW — counterfactual "
                        f"({int(_strength * 100)}% simulated strength)"))
    _f.add_trace(go.Scatter(
        x=_hz, y=_sc.no_action_probs, mode="lines+markers", name="no action",
        line=dict(color=C_FORECAST, width=3), marker=dict(size=9)),
        row=1, col=1)
    _f.add_trace(go.Scatter(
        x=_hz, y=_sc.defend_now_probs, mode="lines+markers",
        name="defend now (counterfactual)",
        line=dict(color=C_NOW, width=3), marker=dict(size=9)),
        row=1, col=2)
    # The no-action line is repeated faintly on the right so the comparison
    # is visible without switching panels.
    _f.add_trace(go.Scatter(
        x=_hz, y=_sc.no_action_probs, mode="lines", name="no action (ref)",
        line=dict(color=C_FORECAST, width=2, dash="dot"),
        showlegend=False), row=1, col=2)
    for _c in (1, 2):
        _f.add_hline(y=_thr, line_dash="dash", line_color=C_ALERT,
                     row=1, col=_c)
    _f.update_yaxes(title_text="threat probability", range=[0, 1], row=1, col=1)
    _f.update_layout(legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(style_fig(_f, height=420), use_container_width=True)
    st.markdown(
        f'<div class="note">Dashed red line is the frozen alert threshold '
        f'<b>{_thr:.2f}</b>. The dotted line on the right repeats the '
        f'no-action trajectory for comparison. Both scenarios start from '
        f'<b>exactly the same observed sequence</b>; the only difference is '
        f'the simulated intervention.</div>',
        unsafe_allow_html=True)

    # ---------- HEADLINE CARDS ----------
    _m1, _m2, _m3, _m4 = st.columns(4)
    with _m1:
        card("No-action max future", f"{_sc.no_action_max_future:.4f}",
             "frozen forecast, max t+1…t+5", "card-forecast")
    with _m2:
        card("Defend-now max future", f"{_sc.defend_now_max_future:.4f}",
             "counterfactual projection", "card-now")
    with _m3:
        _d = _sc.max_future_delta
        card("Projected change", f"{_d:+.4f}",
             "model-projected change under intervention",
             "card-ok" if _d < 0 else ("card-alert" if _d > 0 else ""))
    with _m4:
        card("Frozen threshold crossing",
             "YES" if _sc.crosses_threshold else "no",
             f"no-action {'ALERT' if _sc.no_action_alert else 'CLEAR'} → "
             f"defend-now {'ALERT' if _sc.defend_now_alert else 'CLEAR'}",
             "card-alert" if _sc.crosses_threshold else "")

    # ---------- PROJECTED CHANGE TABLE ----------
    st.divider()
    st.markdown("### Projected change")
    st.caption(
        "This is the **model-projected change under intervention** — the "
        "difference between two forecasts of the same frozen model. It is "
        "not a risk reduction, not a causal effect, and not a mitigation "
        "estimate."
    )
    _rel = _sc.relative_deltas
    _pc = pd.DataFrame([{
        "horizon": _hz[k],
        "no action": round(float(_sc.no_action_probs[k]), 6),
        "defend now": round(float(_sc.defend_now_probs[k]), 6),
        "projected change Δ": round(float(_sc.deltas[k]), 6),
        "relative change": ("—" if _rel[k] is None
                            else f"{100 * _rel[k]:+.2f}%"),
    } for k in range(1, dsim.MAX_K + 1)])
    st.dataframe(_pc, use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="note">Relative change is shown only where the no-action '
        'probability is large enough for a ratio to carry meaning; a '
        '0.0001→0.0002 change is not usefully "+100%".</div>',
        unsafe_allow_html=True)

    # ---------- ATT&CK PROGRESSION ----------
    st.divider()
    st.markdown("### ATT&CK progression under each scenario")
    _p1, _p2 = st.columns(2)

    def _path_block(col, title, hyp, status):
        with col:
            st.markdown(f"**{title}**")
            if hyp is None:
                st.caption("Attack-path interpretation unavailable.")
                return
            if status == apx.STATUS_NONE:
                st.info("**NO FORWARD PROGRESSION IDENTIFIED UNDER THIS "
                        "MODEL SCENARIO**")
            elif status == apx.STATUS_AMBIGUOUS:
                st.warning("**PROGRESSION AMBIGUOUS — ANALYST REVIEW**")
            _nx = hyp.next_step
            st.markdown(
                f"- Current stage: **{hyp.current_stage}**\n"
                + (f"- Next projected: **{_nx.to_stage}** "
                   f"({_nx.technique_id or 'no technique'}) at "
                   f"**+{_nx.time_to_progression_s} s**\n"
                   f"- Progression confidence: "
                   f"**{_nx.progression_confidence}**"
                   if _nx else "- No projected next stage"))

    _path_block(_p1, "NO ACTION", _extras.get("no_action_path"),
                _sc.no_action_attack_path)
    _path_block(_p2, f"DEFEND NOW ({int(_strength * 100)}%)",
                _extras.get("defend_now_path"), _sc.defend_now_attack_path)
    if _sc.no_action_attack_path != _sc.defend_now_attack_path:
        st.caption(
            f"The rule-based ATT&CK reading differs between the two "
            f"scenarios (`{_sc.no_action_attack_path}` → "
            f"`{_sc.defend_now_attack_path}`). That is a change in the "
            f"INTERPRETATION of a perturbed model state, not evidence that "
            f"an attacker's behaviour would change."
        )
    else:
        st.caption("The rule-based ATT&CK reading is unchanged between the "
                   "two scenarios.")

    # ---------- RELIABILITY ----------
    st.divider()
    st.markdown("### Reliability of each scenario")
    st.caption(
        "The Task-15 reliability layer is run separately on the observed "
        "state and on the perturbed state. No new reliability model was "
        "trained. If the intervention pushes the input out of distribution, "
        "it is reported here rather than hidden."
    )
    _r1, _r2 = st.columns(2)
    for _col, _lbl, _a in ((_r1, "NO ACTION", _extras.get("no_action_reliability")),
                           (_r2, f"DEFEND NOW ({int(_strength * 100)}%)",
                            _extras.get("defend_now_reliability"))):
        with _col:
            if _a is None:
                st.caption(f"**{_lbl}** — reliability layer unavailable.")
                continue
            _c = RLY_COLOR[_a.status]
            st.markdown(
                f'<div class="banner {RLY_BANNER[_a.status]}" '
                f'style="padding:10px 14px;">'
                f'<div class="note">{_lbl}</div>'
                f'<b style="color:{_c};font-size:1.1rem;">{_a.status}</b>'
                f'<br><span class="note">{_a.headline}</span></div>',
                unsafe_allow_html=True)
            with st.expander(f"Indicators — {_lbl}"):
                st.markdown(f"- elevated indicators: **{_a.n_elevated}**")
                st.markdown(f"- beyond reference: **{_a.n_beyond}**")
                for _rr in _a.reasons[:6]:
                    st.markdown(f"- `{_rr}`")

    _a_c = _extras.get("defend_now_reliability")
    if _a_c is not None and _a_c.status == "UNRELIABLE":
        st.error(
            "**COUNTERFACTUAL FORECAST UNRELIABLE** — the simulated "
            "intervention moved the model input outside the reference "
            "distribution, so the counterfactual trajectory above describes a "
            "state the frozen model was never calibrated for. Both scenarios "
            "are still shown; the projection must not be acted upon."
        )

    # ---------- WHAT WAS ACTUALLY CHANGED ----------
    st.divider()
    st.markdown("### Exactly what the intervention changed")
    st.markdown(f"**{_iv_def.display}** — {_iv_def.concept}")
    _ft = pd.DataFrame(_feat)
    _mod = _ft[_ft["modified"]].copy()
    _unmod = _ft[~_ft["modified"]].copy()
    st.markdown(f"#### Modified features ({len(_mod)} of 26) — raw physical units")
    if len(_mod):
        _mod["relative_change_pct"] = _mod["relative_change_pct"].map(
            lambda v: "—" if v is None or not np.isfinite(v) else f"{v:+.2f}%")
        st.dataframe(
            _mod[["feature", "observed_raw", "counterfactual_raw",
                  "absolute_change", "relative_change_pct"]]
            .rename(columns={"observed_raw": "observed (raw)",
                             "counterfactual_raw": "counterfactual (raw)",
                             "absolute_change": "absolute change",
                             "relative_change_pct": "relative change"}),
            use_container_width=True, hide_index=True)
    else:
        st.caption("Zero strength — no feature was modified.")
    with st.expander("The exact transformation"):
        for _t in _iv_def.transformation:
            st.code(_t, language="text")
        st.markdown(f"**Justification.** {_iv_def.justification}")
    with st.expander(f"Unmodified features ({len(_unmod)} of 26) and why"):
        st.caption(
            "An unmodified feature is a decision, not an oversight. Where the "
            "frozen contract does not determine a transformation, the feature "
            "is left alone rather than guessed."
        )
        st.dataframe(
            pd.DataFrame([{"feature": k, "why unchanged": v}
                          for k, v in _iv_def.unchanged.items()]),
            use_container_width=True, hide_index=True)
    st.markdown(
        f'<div class="note">Only the <b>final timestep</b> (the current '
        f'state) is perturbed: timesteps t−90s…t−10s are <b>observed '
        f'history</b>, and a "defend now" action cannot have changed them. '
        f'Relative-change row 0 is preserved exactly, because these are '
        f'sliding windows and that row references the window before the '
        f'sequence begins. Because the V4 encoder is an LSTM over ten '
        f'windows, one perturbed window has limited leverage — which is why '
        f'projected changes are modest.</div>',
        unsafe_allow_html=True)

    # ---------- STRENGTH SWEEP ----------
    st.divider()
    st.markdown("### Strength sweep for this sequence")
    st.caption(
        "Fixed strengths, not tuned. Descriptive only: it shows how this "
        "sequence's projected trajectory responds to the size of the "
        "perturbation."
    )
    _rows, _fig2 = [], go.Figure()
    _fig2.add_trace(go.Scatter(
        x=_hz, y=_sc.no_action_probs, mode="lines+markers", name="no action",
        line=dict(color=C_FORECAST, width=3)))
    for _s2, _col2 in zip([0.10, 0.25, 0.50],
                          ["#7dd3fc", "#38bdf8", "#0ea5e9"]):
        _s2c, _f2t, _e2, _er2 = defense_scenario(_seq, _iv, _s2)
        if _s2c is None:
            continue
        _fig2.add_trace(go.Scatter(
            x=_hz, y=_s2c.defend_now_probs, mode="lines+markers",
            name=f"defend now {int(_s2 * 100)}%",
            line=dict(color=_col2, width=2)))
        _rows.append({
            "simulated strength": f"{int(_s2 * 100)}%",
            "defend-now max future": round(_s2c.defend_now_max_future, 6),
            "projected change Δ": round(_s2c.max_future_delta, 6),
            "mean Δ over k=1…5": round(_s2c.mean_delta_k5, 6),
            "crosses frozen threshold": "yes" if _s2c.crosses_threshold else "no",
            "counterfactual reliability": _s2c.defend_now_reliability or "—",
        })
    _fig2.add_hline(y=_thr, line_dash="dash", line_color=C_ALERT)
    _fig2.update_layout(yaxis_title="threat probability", yaxis_range=[0, 1],
                        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(style_fig(_fig2, height=360), use_container_width=True)
    if _rows:
        st.dataframe(pd.DataFrame(_rows), use_container_width=True,
                     hide_index=True)

    # ---------- SUBSTRATE-WIDE SWEEP ----------
    _sw = Path("results/defense_simulation/defense_strength_sweep.csv")
    _mf = Path("results/defense_simulation/simulator_manifest.json")
    if _sw.exists():
        st.divider()
        st.markdown("### Across the replay substrate")
        _swd = pd.read_csv(_sw)
        _swd = _swd[_swd["intervention"] == _iv]
        st.dataframe(
            _swd[["simulated_strength", "n_sequences",
                  "median_max_probability", "p95_max_probability",
                  "global_max_probability", "mean_projected_change_k5",
                  "n_crossing_threshold", "n_less_elevated",
                  "n_more_elevated", "n_unchanged"]],
            use_container_width=True, hide_index=True)
        if _mf.exists():
            _hf = json.loads(_mf.read_text()).get("headline_finding", {})
            if _hf:
                st.warning(
                    f"**Read this honestly.** {_hf.get('direction_is_mixed','')} "
                    f"{_hf.get('magnitude_is_small','')}"
                )
                st.caption(_hf.get("interpretation", ""))

    with st.expander("What this simulator does not claim"):
        st.markdown(
            "- It does **not** estimate causal real-world effectiveness. V4 "
            "was never trained on interventions, so its response to a state "
            "perturbation is not a causal quantity.\n"
            "- It does **not** execute a firewall, routing or host action. "
            "Nothing here touches a socket, device or configuration.\n"
            "- A **lower** projected probability is not mitigation, and a "
            "**higher** projected probability is not evidence that defending "
            "would be harmful.\n"
            "- Real consequences the frozen contract cannot determine — a "
            "block raising `rst_rate`, a rate limit raising inter-arrival "
            "time — are deliberately **not** modelled, so the perturbation is "
            "incomplete by design.\n"
            "- `RESTRICT_EGRESS` assumes blocked destinations carry a "
            "proportional share of window volume: a first-order assumption, "
            "not a measured property."
        )

elif page == "🛡️ Counterfactual Defense Lab":

    st.markdown("# 🛡️ Counterfactual Defense Lab")
    st.markdown(
        "> *If a defender applied one of several defined interventions to "
        "the current network state right now, what would the frozen V4 "
        "World Model project — and how do the options compare?*"
    )

    # ---------- MANDATORY SCIENTIFIC DISCLOSURE ----------
    st.markdown(
        f'<div class="banner banner-gap">'
        f'<span class="pill pill-context">what-if</span>&nbsp;&nbsp;'
        f'<b style="font-size:1.15rem;letter-spacing:.02em;">'
        f'{CFL_HEADLINE}</b>'
        f'<br><span class="note"><b>{CFL_CAUSAL}</b></span>'
        f'<br><span class="note">{CFL_DIRECTION}</span></div>',
        unsafe_allow_html=True,
    )

    # ---------- ANALYST WORKFLOW ----------
    st.markdown(
        f'<div class="note" style="margin:2px 0 10px 0;">'
        f'{cfl_workflow_strip("EXPLORE COUNTERFACTUAL")}</div>',
        unsafe_allow_html=True)
    st.caption(
        "The analyst decides. This page never decides and never acts: it "
        "does not execute any firewall, routing or host action, and it does "
        "not override the frozen Task-17 decision shown further down."
    )

    if using_upload:
        substrate_only_notice("The counterfactual defense lab")
        st.stop()

    if SCREENING is None:
        st.error("The screening substrate is unavailable, so no "
                 "counterfactual can be produced.")
        st.stop()

    _cfl_n = SCREENING.n_sequences
    _cfl_default = SCREENING.default_sequence_id

    # ---------- 1. SELECT OBSERVED WINDOW ----------
    st.divider()
    st.markdown("### 1 · SELECT OBSERVED WINDOW")

    # A sidebar change propagates to the page; an on-page change sticks.
    if st.session_state.get("cfl_sidebar_seen") != SEQ_ID:
        st.session_state.cfl_seq = int(SEQ_ID)
        st.session_state.cfl_sidebar_seen = int(SEQ_ID)
    st.session_state.setdefault("cfl_seq", int(SEQ_ID))
    st.session_state.setdefault("cfl_scenario", cfl.SCENARIOS[1].scenario_id)
    st.session_state.setdefault("cfl_compare_all", True)

    _w1, _w2, _w3 = st.columns([2, 2, 1])
    with _w1:
        _seq = int(st.number_input(
            "Observed window (sequence id)", min_value=0,
            max_value=_cfl_n - 1, step=1, key="cfl_seq",
            help="The 10-window observed history the scenarios all start "
                 "from. Changing the sidebar Sequence ID also moves this."))
    with _w3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("⟲ Reset", use_container_width=True, key="cfl_reset",
                     help="Return every control on this page to its default "
                          "and re-sync to the sidebar sequence."):
            st.session_state.cfl_seq = int(_cfl_default)
            st.session_state.cfl_sidebar_seen = None
            st.session_state.cfl_scenario = cfl.SCENARIOS[1].scenario_id
            st.session_state.cfl_compare_all = True
            st.rerun()

    _thr = SCREENING.alert_threshold
    _row = SCREENING.preds.iloc[_seq]
    with _w2:
        st.markdown(
            f'<div class="note" style="padding-top:6px;">'
            f'observed window <b>{_row["observed_window_start"]}</b> → '
            f'<b>{_row["observed_window_end"]}</b><br>'
            f'10 windows × {sd.WINDOW_SECONDS}s · forecast horizon '
            f'{sd.HORIZON_SECONDS}s · frozen threshold '
            f'<b>{_thr:.2f}</b></div>',
            unsafe_allow_html=True)

    gap_banner(SCREENING, _seq)

    _res, _hist, _cerr = counterfactual_compare(_seq)
    if _res is None:
        st.error(f"**Counterfactual comparison unavailable** for sequence "
                 f"#{_seq}.\n\n`{_cerr}`")
        st.caption("No substitute or guessed projection is displayed.")
        st.stop()

    _out = _res["outcomes"]
    _assess = _res["assessments"]
    _base = _out[cfl.NO_ACTION]

    # ---------- OBSERVE / FORECAST snapshot ----------
    _o1, _o2, _o3, _o4 = st.columns(4)
    with _o1:
        card("Observed nowcast", f"{_base.nowcast:.4f}",
             "frozen P(threat) at t", "card-now")
    with _o2:
        card("No-action max future", f"{_base.max_forecast:.4f}",
             "frozen forecast, max t+1…t+5",
             "card-alert" if _base.alert else "card-forecast")
    with _o3:
        card("No-action reliability", _base.reliability_status or "—",
             "Task 15, unchanged",
             RLY_CARD.get(_base.reliability_status, ""))
    with _o4:
        card("Scenarios compared", f"{len(_out)}",
             f"1 no-action + {len(_out) - 1} interventions", "")

    # ---------- IDENTICAL INITIAL CONDITIONS ----------
    if _hist is not None:
        _ok = _hist["identical_history"] and _hist["check_is_non_vacuous"]
        st.markdown(
            f'<div class="banner {"banner-ok" if _ok else "banner-alert"}" '
            f'style="padding:9px 14px;">'
            f'<b>{"✓" if _ok else "✗"} Identical initial conditions</b>'
            f'<span class="note">&nbsp;— all {_hist["n_scenarios"]} '
            f'scenarios start from byte-identical observed history at '
            f'timesteps 0…{_hist["history_timesteps_compared"] - 1}. Only the '
            f'final observed state is counterfactually perturbed; the '
            f'relative-change row 0 is preserved, never regenerated.</span>'
            f'</div>',
            unsafe_allow_html=True)

    # ---------- 2. CONTROLS ----------
    st.divider()
    st.markdown("### 2 · EXPLORE COUNTERFACTUAL")

    _c1, _c2, _c3 = st.columns([2, 2, 2])
    with _c1:
        _iv_options = [cfl.INTERVENTION_NONE] + list(cfl.INTERVENTION_ORDER)
        _iv = st.selectbox(
            "Intervention",
            _iv_options, index=1,
            format_func=lambda k: ("No action" if k == cfl.INTERVENTION_NONE
                                   else dsim.INTERVENTIONS[k].display),
            help="Each intervention maps only onto features whose "
                 "transformation follows from the frozen feature contract. "
                 "The mathematics is the validated Task-18 one, unchanged.")
    with _c2:
        _s = st.select_slider(
            "Simulated intervention strength",
            options=list(cfl.DECLARED_STRENGTHS),
            value=cfl.DECLARED_STRENGTHS[1],
            format_func=lambda v: f"{int(round(v * 100))}%",
            disabled=(_iv == cfl.INTERVENTION_NONE),
            help="Only the three predeclared strengths are offered. They "
                 "were fixed before any result was inspected and this page "
                 "does not optimise them.")
    with _c3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        _compare_all = st.checkbox(
            "Compare all scenarios", key="cfl_compare_all",
            help="Overlay every declared scenario on one shared axis.")

    _sel_id = cfl.scenario_key(_iv, _s)
    _sel = _out[_sel_id]
    _sel_spec = _sel.scenario

    st.markdown(
        '<div class="note">Strength is a <b>simulated intervention '
        'strength</b> — the size of the perturbation applied to the model '
        'input. It is <b>not</b> a real-world effectiveness, efficacy or '
        'success rate, and no control on this page performs any real '
        'network or host action.</div>',
        unsafe_allow_html=True)

    # ---------- 3. SHARED-AXIS TRAJECTORY COMPARISON ----------
    st.divider()
    st.markdown("### 3 · COMPARE — projected threat trajectories")
    st.caption(
        "One shared axis. Every line is the frozen V4 K=5 rollout of the "
        "same observed history, differing only in the simulated "
        "perturbation applied to the final observed state."
    )

    _hz = ["t (now)"] + [f"+{k * cfl.WINDOW_SECONDS}s"
                         for k in range(1, cfl.MAX_K + 1)]
    _shown = (list(_out) if _compare_all
              else [cfl.NO_ACTION] + ([_sel_id] if _sel_id != cfl.NO_ACTION
                                      else []))

    _fig = go.Figure()
    for _sid in _shown:
        _oc = _out[_sid]
        _is_base = (_sid == cfl.NO_ACTION)
        _fig.add_trace(go.Scatter(
            x=_hz, y=_oc.probs, mode="lines+markers",
            name=("NO ACTION" if _is_base else _sid),
            line=dict(color=CFL_COLOR.get(_sid, "#94a3b8"),
                      width=4 if _is_base else 2.5,
                      dash="solid" if _is_base else
                      ("dot" if _oc.scenario.intervention_type
                       == "RESTRICT_EGRESS" else "dash")),
            marker=dict(size=9 if _is_base else 7)))
    _fig.add_hline(y=_thr, line_dash="dash", line_color=C_ALERT,
                   annotation_text=f"frozen alert threshold {_thr:.2f}",
                   annotation_position="top left")
    _fig.update_layout(yaxis_title="projected threat probability",
                       yaxis_range=[0, 1],
                       legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(style_fig(_fig, height=440), use_container_width=True)
    st.markdown(
        '<div class="note">The solid pale line is <b>NO ACTION</b> — the '
        'frozen forecast every other page of this console shows. Dashed '
        'lines are rate limits, dotted lines are egress restrictions; '
        'colour depth tracks the simulated strength. Lines may sit above '
        '<i>or</i> below the baseline.</div>',
        unsafe_allow_html=True)

    # ---------- PROJECTED THREAT CHANGE ----------
    st.markdown("#### PROJECTED THREAT CHANGE")
    st.caption(
        f"Each cell is the **{cfl.DELTA_NAME}**: the difference between two "
        f"forecasts of the same frozen model. It is not a causal effect and "
        f"it is never a *{cfl.DELTA_IS_NEVER_CALLED}*."
    )
    _delta_tbl = []
    for _sid in list(_out):
        if _sid == cfl.NO_ACTION:
            continue
        _oc = _out[_sid]
        _r = {"scenario": _sid}
        for _k in range(1, cfl.MAX_K + 1):
            _r[f"+{_k * cfl.WINDOW_SECONDS}s"] = round(float(_oc.deltas[_k]), 6)
        _r["max forecast Δ"] = round(_oc.delta_max_forecast, 6)
        _r["direction"] = _oc.direction
        _delta_tbl.append(_r)
    st.dataframe(pd.DataFrame(_delta_tbl), use_container_width=True,
                 hide_index=True)

    # ---------- FULL SCENARIO COMPARISON ----------
    st.markdown("#### Scenario comparison for this window")
    _cmp = []
    for _sid, _oc in _out.items():
        _cmp.append({
            "scenario": ("NO ACTION" if _sid == cfl.NO_ACTION else _sid),
            "nowcast": round(_oc.nowcast, 6),
            "max forecast": round(_oc.max_forecast, 6),
            "Δ max forecast": ("—" if _sid == cfl.NO_ACTION
                               else f"{_oc.delta_max_forecast:+.6f}"),
            "crosses 0.69": ("ALERT" if _oc.alert else "clear"),
            "reliability": _oc.reliability_status or "—",
            "forecast stability": _oc.forecast_stability.get("tier", "—"),
            "rollout stability": _oc.rollout_stability.get("tier", "—"),
            "progression": _oc.progression_status,
            "next stage": _oc.next_stage or "—",
            "time to progression": ("—" if _oc.time_to_progression_s is None
                                    else f"+{_oc.time_to_progression_s}s"),
        })
    st.dataframe(pd.DataFrame(_cmp), use_container_width=True, hide_index=True)
    st.caption(
        "Forecast stability and rollout stability are the Task-15 "
        "`forecast_consistency` and `rollout_stability` indicator groups, "
        "tiered against the frozen reference distribution and reused "
        "unchanged. No new stability metric was invented here."
    )

    # ---------- 4. ATT&CK PROGRESSION ----------
    st.divider()
    st.markdown("### 4 · ATT&CK PROGRESSION — no action vs scenario")
    st.caption(
        f"Task 16 is used unchanged. Stages are compared **categorically**, "
        f"never by subtracting one stage from another. The outcome is a "
        f"**{cfl.PROGRESSION_CHANGE_NAME}** — never "
        f"*'{cfl.PROGRESSION_IS_NEVER_CALLED}'*."
    )

    def _cfl_path_block(col, title, oc, is_base):
        with col:
            st.markdown(f"**{title}**")
            if oc.progression_status == apx.STATUS_NONE:
                st.info("**NO FORWARD PROGRESSION IDENTIFIED UNDER THIS "
                        "MODEL SCENARIO**")
            elif oc.progression_status == apx.STATUS_AMBIGUOUS:
                st.warning("**PROGRESSION AMBIGUOUS — ANALYST REVIEW**")
            _lines = [f"- Current stage: **{oc.current_stage or '—'}**"]
            if oc.next_stage:
                _lines.append(f"- Next projected stage: **{oc.next_stage}**")
                if oc.reliability_status == "UNRELIABLE":
                    _lines.append("- Time to next projected stage: "
                                  "**withheld** — forecast reliability LOW")
                else:
                    _lines.append(f"- Time to next projected stage: "
                                  f"**+{oc.time_to_progression_s} s**")
            else:
                _lines.append("- No projected next stage")
            st.markdown("\n".join(_lines))
            if not is_base:
                st.markdown(cfl_pc_chip(oc.progression_change),
                            unsafe_allow_html=True)

    _pa, _pb = st.columns(2)
    _cfl_path_block(_pa, "NO ACTION", _base, True)
    _cfl_path_block(
        _pb, ("NO ACTION (selected)" if _sel_id == cfl.NO_ACTION
              else f"SCENARIO — {_sel_id}"), _sel, _sel_id == cfl.NO_ACTION)

    st.markdown("#### TIME TO NEXT PROJECTED STAGE")
    _ttp = []
    for _sid, _oc in _out.items():
        _ttp.append({
            "scenario": ("NO ACTION" if _sid == cfl.NO_ACTION else _sid),
            "current stage": _oc.current_stage or "—",
            "next projected stage": _oc.next_stage or "—",
            "time to progression": (
                "withheld (reliability LOW)"
                if (_oc.next_stage and _oc.reliability_status == "UNRELIABLE")
                else ("—" if _oc.time_to_progression_s is None
                      else f"+{_oc.time_to_progression_s}s")),
            "Δ vs no action": (
                "—" if _oc.delta_time_to_progression_s is None
                else f"{_oc.delta_time_to_progression_s:+d}s"),
            "projected progression change": cfl.PROGRESSION_CHANGE_DISPLAY.get(
                _oc.progression_change, "—"),
            "next-stage comparison": _oc.next_stage_comparison,
        })
    st.dataframe(pd.DataFrame(_ttp), use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="note">A change here is a change in the <b>rule-based '
        'interpretation of a perturbed model state</b>. It is not evidence '
        "that an attacker's behaviour would change, and the seconds are not "
        'a countdown: they are k × 10 s, the frozen rollout step.</div>',
        unsafe_allow_html=True)

    # ---------- 5. RELIABILITY ----------
    st.divider()
    st.markdown("### 5 · RELIABILITY of each scenario")
    st.caption(
        "The Task-15 layer is run separately on the observed state and on "
        "each perturbed state. No new reliability model was trained. If a "
        "scenario pushes the input out of the validated regime, that is "
        "reported here rather than hidden, and the scenario is still shown."
    )

    _rel_tbl = []
    for _sid, _oc in _out.items():
        _rel_tbl.append({
            "scenario": ("NO ACTION" if _sid == cfl.NO_ACTION else _sid),
            "reliability": _oc.reliability_status or "—",
            "changed vs no action": ("—" if _sid == cfl.NO_ACTION
                                     else ("yes" if _oc.reliability_changed
                                           else "no")),
            "indicators whose tier moved": (
                ", ".join(_oc.reliability_indicators_changed) or "none"),
            "prob range (k=0…5)": (
                None if _oc.forecast_stability.get("prob_range") is None
                else round(_oc.forecast_stability["prob_range"], 6)),
            "largest single-step prob jump": (
                None if _oc.forecast_stability.get("prob_max_abs_step") is None
                else round(_oc.forecast_stability["prob_max_abs_step"], 6)),
            "latent displacement": (
                None if _oc.rollout_stability.get("displacement") is None
                else round(_oc.rollout_stability["displacement"], 4)),
        })
    st.dataframe(pd.DataFrame(_rel_tbl), use_container_width=True,
                 hide_index=True)

    _ra, _rb = st.columns(2)
    for _col, _lbl, _a in ((_ra, "NO ACTION", _assess.get(cfl.NO_ACTION)),
                           (_rb, ("NO ACTION (selected)"
                                  if _sel_id == cfl.NO_ACTION
                                  else f"SCENARIO — {_sel_id}"),
                            _assess.get(_sel_id))):
        with _col:
            if _a is None:
                st.caption(f"**{_lbl}** — reliability layer unavailable.")
                continue
            _c = RLY_COLOR[_a.status]
            st.markdown(
                f'<div class="banner {RLY_BANNER[_a.status]}" '
                f'style="padding:10px 14px;">'
                f'<div class="note">{_lbl}</div>'
                f'<b style="color:{_c};font-size:1.1rem;">{_a.status}</b>'
                f'<br><span class="note">{_a.headline}</span></div>',
                unsafe_allow_html=True)
            with st.expander(f"Indicators — {_lbl}"):
                st.markdown(f"- elevated indicators: **{_a.n_elevated}**")
                st.markdown(f"- beyond reference: **{_a.n_beyond}**")
                for _rr in _a.reasons[:6]:
                    st.markdown(f"- `{_rr}`")

    _unrel = [s for s, o in _out.items()
              if s != cfl.NO_ACTION and o.reliability_status == "UNRELIABLE"]
    if _unrel:
        st.error(
            f"**{cfl.UNRELIABLE_BANNER}** — {len(_unrel)} scenario(s) "
            f"(`{'`, `'.join(_unrel)}`) moved the model input outside the "
            f"reference distribution, so those trajectories describe a state "
            f"the frozen model was never calibrated for. They are still "
            f"displayed above, not suppressed: the analyst must be able to "
            f"see that the scenario itself left the validated input regime."
        )

    # ---------- 6. RELIABILITY + ABSTAIN INTEGRATION ----------
    st.divider()
    st.markdown("### 6 · ANALYST DECIDES — the frozen decision is unchanged")

    _dec, _derr = analyst_decision_of(_seq)
    if _dec is None:
        st.caption(f"The Task-17 decision layer is unavailable (`{_derr}`), "
                   f"so no decision status is shown for this window.")
    else:
        _dc = ABS_COLOR[_dec.decision]
        st.markdown(
            f'<div class="banner {ABS_BANNER[_dec.decision]}">'
            f'{ABS_ICON[_dec.decision]}&nbsp;<b style="color:{_dc};'
            f'font-size:1.12rem;">{_dec.display}</b>'
            f'<br><span class="note">{_dec.semantics}</span></div>',
            unsafe_allow_html=True)
        if _dec.abstained:
            st.warning(f"**{cfl.ABSTAIN_COEXISTENCE}**")
            for _t in _dec.reason_text:
                st.caption(f"• {_t}")
        if _base.reliability_status == "UNRELIABLE":
            st.error(
                "The observed state is already assessed **UNRELIABLE** by "
                "Task 15, so the automated decision is **already withheld** "
                "for this window. Exploring counterfactuals does not "
                "reinstate it."
            )
        st.caption(
            "Nothing on this page can change the decision above. The "
            "counterfactual layer does not import, call or bypass the frozen "
            "Task-17 policy — it sits beside it, for analyst review."
        )

    # ---------- 7. XAI ----------
    st.divider()
    st.markdown("### 7 · Model-sensitive telemetry")
    st.caption(
        "Signals contributing to the forecast for the **observed** current "
        "state, from the provenance-locked Integrated Gradients run. An "
        "attribution describes this model's sensitivity on this input. It "
        "is not a causal explanation of an intervention, and it does not "
        "establish attacker intent, attack stage or actual compromise."
    )
    _x1, _x2 = st.columns(2)
    with _x1:
        st.markdown("**Signals contributing to the forecast** "
                    f"(observed state, P(t+{horizon}))")
        if attributions_active is None:
            st.caption("Attribution tensors are unavailable for this "
                       "horizon, so none are displayed.")
        else:
            _attr = attributions_active[_seq, -1, :]
            _vals = X_active[_seq, -1, :]
            _top = np.argsort(np.abs(_attr))[::-1][:8]
            st.dataframe(pd.DataFrame([{
                "feature": INPUT_FEATURES[_i],
                "value (σ)": round(float(_vals[_i]), 4),
                "attribution": round(float(_attr[_i]), 6),
                "pushes forecast": ("higher" if _attr[_i] > 0 else "lower"),
            } for _i in _top]), use_container_width=True, hide_index=True)
    with _x2:
        st.markdown(f"**Base-state features changed by the scenario** "
                    f"(`{_sel_id}`)")
        if _sel_id == cfl.NO_ACTION:
            st.caption("NO ACTION changes nothing: the observed state is "
                       "rolled forward verbatim.")
        elif not _sel.changed_feature_table:
            st.caption("No base feature moved under this scenario.")
        else:
            st.dataframe(pd.DataFrame([{
                "feature": _r["feature"],
                "observed (raw)": _r["observed_raw"],
                "counterfactual (raw)": _r["counterfactual_raw"],
                "change": ("—" if _r["relative_change_pct"] is None
                           else f"{_r['relative_change_pct']:+.2f}%"),
            } for _r in _sel.changed_feature_table]),
                use_container_width=True, hide_index=True)
            st.caption(
                "These are the features the DEFINED intervention moved, in "
                "raw physical units, on the final observed timestep only. "
                "The attribution on the left explains the observed forecast; "
                "it does not explain this perturbation."
            )

    if _sel_id != cfl.NO_ACTION:
        _ivd = dsim.INTERVENTIONS[_sel_spec.intervention_type]
        with st.expander(f"The exact transformation — {_ivd.display}"):
            st.markdown(f"**Concept.** {_ivd.concept}")
            for _t in _ivd.transformation:
                st.code(_t, language="text")
            st.markdown(f"**Justification.** {_ivd.justification}")
            st.caption(
                "Imported unchanged from the validated Task-18 simulator. "
                "This layer adds no new state transformation."
            )

    # ---------- 8. SUBSTRATE-WIDE SCENARIO MATRIX ----------
    _cfl_sum = counterfactual_summary()
    if _cfl_sum:
        st.divider()
        st.markdown("### 8 · Defense scenario matrix — across the substrate")
        st.caption(
            f"{_cfl_sum['substrate']['n_sequences_swept']} sequences, "
            f"{_cfl_sum['substrate']['sweep_rule']}. **Descriptive only:** "
            f"there is no overall score and no ranking, and no direction is "
            f"called successful."
        )
        _mx = pd.DataFrame(_cfl_sum["scenario_matrix"])
        st.dataframe(
            _mx.rename(columns={
                "median_max_p": "median max P", "p95_max_p": "p95 max P",
                "mean_max_p": "mean max P", "global_max_p": "global max P",
                "mean_delta": "mean Δ", "median_delta": "median Δ",
                "p95_abs_delta": "p95 |Δ|",
                "crossing_threshold": "crossing 0.69",
                "leaving_threshold": "leaving 0.69",
                "progression_change": "progression change"}),
            use_container_width=True, hide_index=True)

        _pt = _cfl_sum.get("progression_change_totals", {})
        if _pt:
            st.markdown("#### Model-projected progression change — totals")
            st.dataframe(pd.DataFrame([
                {"outcome": cfl.PROGRESSION_CHANGE_DISPLAY.get(_k, _k),
                 "sequences × scenarios": _v}
                for _k, _v in _pt.items()]),
                use_container_width=True, hide_index=True)

    # ---------- SCIENTIFIC DISCLOSURE ----------
    st.divider()
    with st.expander("What this simulator does not claim", expanded=False):
        st.markdown(
            f"- **{CFL_CAUSAL}**\n"
            f"- **{CFL_DIRECTION}**\n"
            "- It is **not** a causal effect estimator, **not** a real "
            "defense engine, **not** an automated response system, and "
            "**not** proof that an intervention would work in reality.\n"
            "- A projected reduction does not mean that mitigation was "
            "achieved, and a projected increase is **not** evidence that "
            "defending would be harmful. Both are the frozen model's "
            "response to a state "
            "perturbation it was never trained to interpret.\n"
            "- Progression differences are **not** described as an attack "
            "delayed by N seconds: the seconds are k × 10 s rollout steps of "
            "a rule-based interpretation.\n"
            "- Only the **final** observed timestep is perturbed, because "
            "timesteps t−90s…t−10s are observed history. The V4 encoder is "
            "an LSTM over ten windows, so a one-window perturbation has "
            "limited leverage and projected changes are modest by "
            "construction.\n"
            "- Real consequences the frozen contract cannot determine — a "
            "block raising `rst_rate`, a rate limit raising inter-arrival "
            "time — are deliberately **not** modelled, so every perturbation "
            "is incomplete by design.\n"
            "- `RESTRICT_EGRESS` assumes blocked destinations carry a "
            "proportional share of window volume: a first-order assumption, "
            "**not** a measured property.\n"
            "- Only two interventions are defensible on the frozen "
            "26-feature contract. A third (protocol allow-listing) was "
            "considered, measured and declined — `unique_protocols` has "
            "support {1, 2, 3}, so a fractional 'fraction of protocols "
            "blocked' would **not** mean what its name says."
        )
    with st.expander("Scenario definitions"):
        st.dataframe(pd.DataFrame([{
            "scenario_id": _s.scenario_id,
            "scenario_name": _s.scenario_name,
            "intervention_type": _s.intervention_type,
            "strength": _s.strength,
            "description": _s.description,
        } for _s in cfl.SCENARIOS]), use_container_width=True,
            hide_index=True)
        st.caption(
            f"Strengths {list(cfl.DECLARED_STRENGTHS)} were declared in "
            f"`src/v4_counterfactual.py` before any evaluation and are "
            f"asserted identical to the frozen Task-18 set at import. Every "
            f"scenario is reproducible from "
            f"`{cfl.layer_metadata()['reproducibility_key']}`."
        )
# =============================================================
# PAGE: DATA INPUT  (the single entry point for every source)
# =============================================================
# One page, one source selector, one validated representation. The three
# sources differ only in how the (N, 10, 52) tensor is OBTAINED; once it
# exists, every one of them goes through dinp.run_inference(), which is the
# frozen V4 rollout. No source owns a second inference implementation.

elif page == "📥 Data Input":

    st.markdown("# 📥 Data Input")
    st.markdown(
        "> One place to give the console data. Choose a source, validate it, "
        "and every downstream page — forecast, reliability, attribution, "
        "ATT&CK, attack path, defense simulation, counterfactual lab — reads "
        "the same validated tensor through the same frozen V4 inference."
    )

    st.markdown(
        '<div class="banner banner-gap">'
        '<b>Data Input → validated V4 tensor → frozen V4 inference → '
        'forecast → reliability / abstain → attribution → ATT&amp;CK → '
        'attack path → defense simulation → counterfactual</b>'
        '<br><span class="note">Whatever the source, the tensor handed to '
        'the model is <b>(N, 10, 52)</b> in frozen-scaler space and the '
        'rollout is the same <code>k_step_rollout</code> call. Nothing on '
        'this page retrains, rescales or re-thresholds anything.</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---------- SOURCE SELECTOR ----------
    st.divider()
    st.markdown("### Data Source")

    _DI_MODES = [dinp.SRC_BENCHMARK, dinp.SRC_PCAP, "prepared"]
    _DI_LABEL = {
        dinp.SRC_BENCHMARK: "📊 Historical Benchmark",
        dinp.SRC_PCAP: "📡 Upload PCAP",
        "prepared": "📂 Upload Prepared V4 Data",
    }
    _di_current = st.session_state.get("input_source", dinp.SRC_BENCHMARK)
    if _di_current not in _DI_MODES:
        _di_current = dinp.SRC_BENCHMARK

    _di_mode = st.radio(
        "Select source",
        _DI_MODES,
        index=_DI_MODES.index(_di_current),
        format_func=lambda k: _DI_LABEL[k],
        horizontal=True,
        key="data_input_mode_radio",
        help="Only the controls for the selected source are shown. There is "
             "deliberately no second uploader competing with this one.",
    )

    # Persist the choice, then force ONE rerun so the sidebar and the
    # *_active resolution -- both rendered earlier in this same script pass
    # -- pick it up immediately.
    if _di_mode != st.session_state.get("input_source"):
        st.session_state.input_source = _di_mode
        st.rerun()

    with st.expander("How this project's files are classified"):
        st.caption(
            "Uploaded files are IDENTIFIED by their contents before they are "
            "validated, so a rejection explains what the file actually is "
            "rather than reporting a shape mismatch. Renaming a file does "
            "not change the verdict: the content decides."
        )
        st.dataframe(
            pd.DataFrame([
                {"file": _k,
                 "shape": _v["shape"],
                 "what it is": dinp.ROLE_DISPLAY[_v["role"]],
                 "valid V4 input": "✅ yes" if _v["valid"] else "❌ no",
                 "note": _v["note"]}
                for _k, _v in dinp.KNOWN_ARTIFACTS.items()
            ]),
            use_container_width=True, hide_index=True,
        )


    def _di_validation_card(prepared):
        """The compact validation card every source renders."""
        st.markdown("#### Validation")
        _rows = prepared.card_rows()
        _cols = st.columns(3)
        for _i, (_label, _value) in enumerate(_rows):
            with _cols[_i % 3]:
                st.markdown(
                    f'<div class="card"><div class="card-label">{_label}'
                    f'</div><div class="card-value" '
                    f'style="font-size:1.0rem;">{_value}</div></div>',
                    unsafe_allow_html=True)
        _ident = prepared.identification
        if _ident is not None and _ident.headline:
            st.caption(f"**Identified as:** {_ident.headline}"
                       + (f"  \n`{_ident.detail}`" if _ident.detail else ""))
        for _w in prepared.warnings:
            st.warning(f"⚠️ {_w}")
        if prepared.rejected_detail:
            with st.expander(
                    f"🗑️ {prepared.n_rejected} rejected row(s) — details"):
                for _d in prepared.rejected_detail:
                    st.text(_d)
        if not prepared.is_valid:
            st.error(f"### {dinp.INVALID_BADGE}")
            for _e in prepared.errors:
                st.error(_e)
            return False
        st.success(
            f"{dinp.VALID_BADGE} — **{prepared.n_sequences}** "
            f"{dinp.SEQUENCE_LENGTH}-window sequence(s) ready for the frozen "
            f"V4 rollout.")
        return True


    def _di_result_summary(prepared, traj, thr, key_prefix):
        """The compact result view shared by every source.

        `traj` is the (N, K+1) output of the ONE inference call. This
        function only renders it -- it performs no inference of its own.
        """
        _n = traj.shape[0]
        _probs_h = traj[:, horizon]
        _alerts = _probs_h >= thr
        _n_alerts = int(_alerts.sum())

        if _n_alerts > 0:
            st.error(f"⚠️ **THREAT DETECTED** — {_n_alerts} alert(s) across "
                     f"{_n} sequence(s). Max P(t+{horizon}): "
                     f"{float(_probs_h.max()):.4f}")
        else:
            st.success(f"✅ **No threats detected** — all {_n} sequence(s) "
                       f"below the frozen threshold {thr:.2f}. "
                       f"Max P(t+{horizon}): {float(_probs_h.max()):.4f}")

        _sel = 0
        if _n > 1:
            _sel = st.slider("Select sequence", 0, _n - 1,
                             min(int(st.session_state.get("seq_id", 0)), _n - 1)
                             if prepared.source_type == dinp.SRC_BENCHMARK
                             else 0,
                             key=f"{key_prefix}_seq")
        _row = traj[_sel]
        _now = float(_row[0])
        _fut = _row[1:]
        _maxf = float(_fut.max())
        _argk = int(np.argmax(_fut)) + 1
        _alert = _maxf >= thr

        _c1, _c2, _c3, _c4 = st.columns(4)
        with _c1:
            card("Nowcast (t+0)", f"{_now:.4f}", "current window", "card-now")
        with _c2:
            card(f"t+{horizon} ({horizon * 10}s)", f"{float(_row[horizon]):.4f}",
                 "selected horizon", "card-forecast")
        with _c3:
            card("Max forecast", f"{_maxf:.4f}", f"peak at t+{_argk}",
                 "card-forecast")
        with _c4:
            card("Status", "🚨 ALERT" if _alert else "✅ CLEAR",
                 f"threshold {thr:.2f}",
                 "card-alert" if _alert else "card-ok")

        _f = go.Figure()
        _f.add_trace(go.Scatter(
            x=[f"t+{k}" for k in range(traj.shape[1])], y=_row,
            mode="lines+markers", name="threat probability",
            line=dict(color=C_FORECAST, width=3), marker=dict(size=8)))
        _f.add_hline(y=thr, line_dash="dash", line_color=C_ALERT,
                     annotation_text=f"threshold ({thr:.2f})")
        _f.update_yaxes(range=[0, 1], title="P(attack)")
        _f.update_xaxes(title="rollout step")
        st.plotly_chart(style_fig(_f, height=330), use_container_width=True)
        return _sel


    def _di_next_steps():
        st.divider()
        st.markdown("#### Next")
        st.markdown(
            "This source is now active across the console. Open "
            "**🎯 Live Forecast**, **🔍 Feature Attribution**, "
            "**🗺️ MITRE ATT&CK Timeline**, **📊 Window Inspector** or "
            "**🔮 Trajectory Rollout** — each reads the same validated "
            "tensor through the same frozen inference.")
        st.caption(
            "The reliability, abstain, attack-path, defend-now and "
            "counterfactual pages additionally need the frozen screening "
            "replay substrate, so they stay on **Historical Benchmark**.")


    # =========================================================
    # MODE 1 — HISTORICAL BENCHMARK
    # =========================================================
    if _di_mode == dinp.SRC_BENCHMARK:

        st.divider()
        st.markdown("### 📊 Historical Benchmark")
        st.caption(
            "The frozen held-out benchmark: `X_test.npy` as model input, "
            "`A_binary_multistep_test.npy` as ground-truth labels and "
            "`test_metadata.csv` as timestamps. The labels and metadata are "
            "used for scoring and display only — the model is fed the "
            "tensor alone."
        )

        _bench = dinp.from_benchmark(
            X_test,
            timestamps=test_meta["timestamp"].tolist()
            if "timestamp" in test_meta.columns else None,
            source_name="X_test.npy")
        _di_validation_card(_bench)

        st.markdown(
            '<div class="note">Replay controls (Prev / Next / Reset / '
            'autoplay / Sequence ID) stay in the sidebar and are unchanged. '
            'The selector below mirrors them for convenience.</div>',
            unsafe_allow_html=True)

        _bt = st.number_input(
            "Sequence ID", min_value=0, max_value=int(len(X_test)) - 1,
            value=int(min(SEQ_ID, len(X_test) - 1)), step=1,
            key="di_bench_seq")
        _bt = int(_bt)

        # THE SINGLE INFERENCE CALL -- same function the other two modes use.
        _bench_one = dinp.from_benchmark(
            X_test[_bt:_bt + 1], source_name="X_test.npy")
        _traj_b = dinp.run_inference(_bench_one, explainer, sd.MAX_K)
        _di_result_summary(_bench_one, _traj_b, float(threshold),
                           "di_bench")

        if "attack_present" in test_meta.columns:
            _gt = test_meta["attack_present"].iloc[_bt]
            st.caption(
                f"Ground truth for this window: "
                f"**{'attack present' if bool(_gt) else 'benign'}** "
                f"(from `A_binary_multistep_test.npy` / `test_metadata.csv`; "
                f"never shown to the model).")
        _di_next_steps()

    # =========================================================
    # MODE 3 — UPLOAD PREPARED V4 DATA
    # =========================================================
    elif _di_mode == "prepared":

        st.divider()
        st.markdown("### 📂 Upload Prepared V4 Data")
        st.caption(
            f"**CSV** — must carry all {dinp.FEATURE_DIMENSION} canonical V4 "
            f"feature columns, one row per {dinp.WINDOW_SECONDS}-second "
            f"window in chronological order. Rows are built into contiguous "
            f"{dinp.SEQUENCE_LENGTH}-row sequences. An optional `timestamp` "
            f"column is used for ordering and "
            f"{dinp.WINDOW_SECONDS}-second continuity checks.  \n"
            f"**NPY** — must have shape "
            f"`({dinp.SEQUENCE_LENGTH}, {dinp.FEATURE_DIMENSION})` or "
            f"`(N, {dinp.SEQUENCE_LENGTH}, {dinp.FEATURE_DIMENSION})`. "
            f"Treated as already-prepared V4 input; no further scaling is "
            f"applied. Everything else is rejected."
        )

        _up = st.file_uploader(
            "Upload prepared V4 data",
            type=list(dinp.PREPARED_EXTENSIONS),
            key="di_prepared_uploader",
            help="A CSV of 52 V4 feature columns, or an NPY of shape "
                 "(N, 10, 52).")

        with st.expander(
                f"The {dinp.FEATURE_DIMENSION} canonical V4 feature names"):
            st.code("\n".join(
                f"{_i + 1:2d}. {_nm}"
                for _i, _nm in enumerate(pil.canonical_feature_names())))

        if _up is None:
            _stored = st.session_state.get("prepared_data")
            if _stored is not None:
                st.info(
                    f"📂 Currently active: **{_stored['filename']}** — "
                    f"{_stored['n_windows']} sequence(s). Upload another "
                    f"file to replace it.")
            st.stop()

        _up_bytes = _up.read()
        with st.spinner("🔍 Identifying and validating…"):
            _prep = dinp.from_upload(_up.name, _up_bytes)

        if not _di_validation_card(_prep):
            st.caption(
                "Nothing was loaded and the active source is unchanged. No "
                "substitute or guessed tensor is ever used.")
            st.stop()

        # THE SINGLE INFERENCE CALL.
        with st.spinner(
                f"🧠 Frozen V4 recursive K={int(ckpt['max_k'])} rollout on "
                f"{_prep.n_sequences} sequence(s)…"):
            _traj_p = dinp.run_inference(_prep, explainer, int(ckpt["max_k"]))

        _probs_p_h = _traj_p[:, horizon]
        _alerts_p = _probs_p_h >= float(threshold)

        # Attribution + ATT&CK, reusing the same provenance-locked explainer
        # and the same rule library the benchmark path uses.
        _imp_p, _attr_p, _mitre_p = None, None, None
        try:
            with st.spinner("🔍 Computing feature attribution…"):
                _r = explainer.attribute(_prep.x, horizon)
                _attr_p = _r["attributions"]
                _abs_p = np.abs(_attr_p).sum(axis=1)
                _sgn_p = _attr_p.sum(axis=1)
                _imp_p = pd.DataFrame({
                    "feature_index": np.arange(len(INPUT_FEATURES)),
                    "feature": INPUT_FEATURES,
                    "horizon": horizon,
                    "mean_abs_attribution": _abs_p.mean(axis=0),
                    "mean_signed_attribution": _sgn_p.mean(axis=0),
                }).sort_values("mean_abs_attribution",
                               ascending=False).reset_index(drop=True)
                _id_p = explainer.check_prediction_identity(_prep.x, horizon)
                if not _id_p["passed"]:
                    _attr_p, _imp_p = None, None
                    st.error(
                        "⚠️ Explanation withheld: the attributed prediction "
                        "did not match the displayed prediction "
                        f"(max diff {_id_p['max_abs_diff']:.2e}).")
        except Exception as _exc:                       # noqa: BLE001
            st.warning(f"⚠️ Feature attribution failed: {_exc}")

        try:
            with st.spinner("🗺️ Mapping to MITRE ATT&CK…"):
                _mapper_p = MITREMapper(model=model, threshold=float(threshold),
                                        device="cpu")
                _fa_p = (int(np.argmax(_alerts_p)) if _alerts_p.any()
                         else None)
                _maps_p = [
                    _mapper_p.map_single_window(
                        x_window=_prep.x[_i],
                        window_idx=_i,
                        timestamp=str(_i),
                        total_windows=_prep.n_sequences,
                        first_alert_idx=_fa_p,
                        attack_prob=float(_probs_p_h[_i]))
                    for _i in range(_prep.n_sequences)]
                _mitre_p = pd.DataFrame([{
                    "window_index": _m.window_index,
                    "timestamp": _m.timestamp,
                    "attack_probability": _m.attack_probability,
                    "tactic_id": _m.tactic_id,
                    "tactic_name": _m.tactic_name,
                    "technique_id": _m.technique_id or "",
                    "technique_name": _m.technique_name or "",
                    "confidence": _m.confidence,
                    "temporal_phase": _m.temporal_phase,
                    "triggering_features": "; ".join(_m.triggering_features),
                    "explanation": _m.analyst_explanation,
                } for _m in _maps_p])
        except Exception as _exc:                       # noqa: BLE001
            st.warning(f"⚠️ MITRE mapping failed: {_exc}")

        _meta_p = pd.DataFrame({
            "timestamp": (pd.to_datetime(_prep.timestamps[0])
                          + pd.to_timedelta(
                              np.arange(_prep.n_sequences)
                              * dinp.WINDOW_SECONDS, unit="s")
                          if _prep.timestamps
                          else pd.RangeIndex(_prep.n_sequences).astype(str)),
            "attack_present": (_prep.labels if _prep.has_labels
                               and _prep.labels is not None
                               and len(_prep.labels) == _prep.n_sequences
                               else np.nan),
        })

        _file_id_p = f"{_up.name}:{len(_up_bytes)}"
        st.session_state.prepared_data = {
            "file_id": _file_id_p,
            "filename": _up.name,
            "format_label": _prep.format_label,
            "source_type": _prep.source_type,
            "X": _prep.x,
            "meta": _meta_p,
            "probs": _probs_p_h,
            "mitre_df": _mitre_p,
            "importance_df": _imp_p,
            "attributions": _attr_p,
            "n_windows": _prep.n_sequences,
            "labels": _prep.labels,
        }
        st.session_state.input_source = "prepared"
        st.success(
            "📌 Loaded — every page that reads the active source now shows "
            "**📂 Prepared V4 data**.")

        # The sidebar and the *_active block ran earlier in this pass and
        # cannot see what was just stashed. One guarded rerun fixes that.
        if st.session_state.get("_prep_rerun_done_for") != _file_id_p:
            st.session_state._prep_rerun_done_for = _file_id_p
            st.rerun()

        st.divider()
        st.markdown("### Result")
        _di_result_summary(_prep, _traj_p, float(threshold), "di_prep")

        if _prep.n_sequences > 1:
            _ftl = go.Figure()
            _ftl.add_trace(go.Scatter(
                x=list(range(_prep.n_sequences)), y=_probs_p_h,
                mode="lines+markers", name=f"P(attack) t+{horizon}",
                line=dict(color=C_FORECAST, width=2), marker=dict(size=4)))
            _ftl.add_trace(go.Scatter(
                x=list(range(_prep.n_sequences)), y=_traj_p[:, 0],
                mode="lines", name="nowcast t+0",
                line=dict(color=C_NOW, width=1, dash="dot")))
            _ftl.add_hline(y=float(threshold), line_dash="dash",
                           line_color=C_ALERT)
            _ftl.update_xaxes(title="sequence index")
            _ftl.update_yaxes(range=[0, 1], title="P(attack)")
            st.plotly_chart(style_fig(_ftl, height=340),
                            use_container_width=True)

        if _prep.has_labels and _prep.labels is not None:
            st.divider()
            st.markdown("### 📊 Evaluation (ground truth supplied in the file)")
            st.caption(
                "Scored against the labels in YOUR file, at the frozen "
                "validation-selected per-horizon thresholds. This is not the "
                "frozen benchmark result.")
            from sklearn.metrics import (f1_score, precision_score,
                                         recall_score, roc_auc_score)
            _lab = np.asarray(_prep.labels)[:_prep.n_sequences]
            _gt = (_lab > 0.5).astype(int)
            _erows = []
            for _kk in range(1, sd.MAX_K + 1):
                _pk = _traj_p[:len(_gt), _kk]
                _tk = float(explainer.horizon_thresholds.get(_kk, threshold))
                _pr = (_pk >= _tk).astype(int)
                _e = {"Horizon": f"t+{_kk} ({_kk * 10}s)",
                      "Threshold": f"{_tk:.2f}"}
                _e["F1"] = f"{f1_score(_gt, _pr, zero_division=0):.4f}"
                _e["Precision"] = f"{precision_score(_gt, _pr, zero_division=0):.4f}"
                _e["Recall"] = f"{recall_score(_gt, _pr, zero_division=0):.4f}"
                _e["ROC-AUC"] = (f"{roc_auc_score(_gt, _pk):.4f}"
                                 if len(np.unique(_gt)) > 1
                                 else "n/a (single class)")
                _erows.append(_e)
            st.dataframe(pd.DataFrame(_erows), use_container_width=True,
                         hide_index=True)
        else:
            st.caption(
                "💡 Include an `attack_present`, `attack_label` or `label` "
                "column in a CSV to score the forecast against your own "
                "ground truth.")

        _di_next_steps()

    # =========================================================
    # MODE 2 — UPLOAD PCAP
    # (the existing ingestion pipeline, unchanged)
    # =========================================================
    elif _di_mode == dinp.SRC_PCAP:

        st.divider()
        st.markdown("### 📡 Upload PCAP")
        st.caption(
            "Drag and drop a raw `.pcap` / `.pcapng` capture. The existing "
            "ingestion pipeline is used unchanged: "
            "**packets → flow reconstruction → 10-second aggregation → 52-D "
            "V4 state → 10 historical windows → frozen V4 → K=5 rollout**."
        )

        WINDOW_SECONDS = 10
        SEQ_LEN = 10
        BASE_FEATURES = INPUT_FEATURES[:26]

        # --- Helper functions (inlined) ---

        def pcap_entropy(values):
            if len(values) == 0:
                return 0.0
            counts = pd.Series(values).value_counts()
            probs = counts / counts.sum()
            return float(
                -np.sum(probs * np.log2(probs + 1e-10))
            )

        def parse_pcap_upload(pcap_bytes):
            # Streaming parse via the canonical pipeline (PcapReader
            # iterator -- constant memory, no rdpcap).
            with tempfile.NamedTemporaryFile(
                suffix=".pcap", delete=False
            ) as tmp:
                tmp.write(pcap_bytes)
                tmp_path = tmp.name
            try:
                return pcap_v4.parse_pcap_streaming(tmp_path)
            finally:
                # The parse returns a fully materialised DataFrame, so the
                # uploaded capture is not left behind in the temp directory.
                Path(tmp_path).unlink(missing_ok=True)

        def pcap_reconstruct_flows(packet_df):
            # Canonical 5-tuple flow reconstruction.
            return pcap_v4.reconstruct_flows(packet_df)

        def pcap_compute_features(flow_df, packet_df):
            # Canonical 26 base features per 10s window (per-packet TCP
            # flag fractions, matching pcap_ingest_v4).
            return pcap_v4.compute_window_features(flow_df, packet_df)

        def pcap_add_rel_changes(window_df):
            # Canonical rel-change semantics (first row zeroed, inf->0,
            # clip [-10, 10]) -- matches prepare_multiday_benchmark.py.
            return pcap_v4.add_relative_changes(window_df)

        # --- Upload widget ---
        uploaded = st.file_uploader(
            "Upload a .pcap file",
            type=["pcap", "pcapng", "cap"],
            help="Drag and drop a raw packet capture "
                 "file from Wireshark, tcpdump, etc."
        )

        if uploaded is not None:
            file_id = (uploaded.name, uploaded.size)
            pcap_bytes = uploaded.read()
            file_size_mb = len(pcap_bytes) / (1024 * 1024)

            st.markdown(
                f"**File:** `{uploaded.name}` "
                f"({file_size_mb:.2f} MB)"
            )

            with st.spinner(
                "🔍 Parsing packets from PCAP..."
            ):
                packet_df = parse_pcap_upload(pcap_bytes)

            if len(packet_df) == 0:
                st.error(
                    "No IP packets found in this file!"
                )
            else:
                # Show packet summary
                duration = packet_df["timestamp"].max()
                n_pkts = len(packet_df)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric(
                    "📦 IP Packets", f"{n_pkts:,}"
                )
                col2.metric(
                    "⏱️ Duration",
                    f"{duration:.1f}s"
                )
                col3.metric(
                    "🌐 Unique IPs",
                    f"{packet_df['src_ip'].nunique() + packet_df['dst_ip'].nunique()}"
                )
                col4.metric(
                    "📊 File Size",
                    f"{file_size_mb:.2f} MB"
                )

                # Process through pipeline
                with st.spinner(
                    "⚙️ Reconstructing flows "
                    "& extracting features..."
                ):
                    flow_df = pcap_reconstruct_flows(
                        packet_df
                    )
                    feature_df = pcap_compute_features(
                        flow_df, packet_df
                    )
                    feature_df = pcap_add_rel_changes(
                        feature_df
                    )

                n_windows = len(feature_df)
                n_flows = len(flow_df)

                st.success(
                    f"✅ Extracted **{n_flows}** flows "
                    f"across **{n_windows}** time windows"
                )

                # Canonical scaling + strictly-contiguous sequence
                # windowing (rejects sequences that span an empty-window
                # temporal gap). Uses the multiday_benchmark_K5 scaler.
                pcap_scaler = pcap_v4.load_scaler(
                    pcap_v4.INPUT_SCALER_PATH
                )
                seq_pack = pcap_v4.build_sequences(
                    feature_df, pcap_scaler
                )
                X_pcap = seq_pack["X"]
                seq_timestamps = seq_pack["seq_ts_rel"]
                n_gap_rejected = seq_pack["n_rejected_gap"]

                if len(X_pcap) == 0:
                    st.warning(
                        f"⚠️ Need at least {SEQ_LEN} strictly "
                        f"contiguous 10s windows "
                        f"({SEQ_LEN * WINDOW_SECONDS}s of gap-free "
                        f"traffic). Candidates: "
                        f"{seq_pack['n_candidate']}, all rejected "
                        f"for temporal gaps: {n_gap_rejected}."
                    )
                else:
                    # True recursive K=5 latent rollout (encode once,
                    # feed each predicted latent back through the gated
                    # transition). NOT five independent predictions.
                    k_pcap = int(ckpt["max_k"])
                    with st.spinner(
                        f"🧠 Running V4 recursive K={k_pcap} "
                        f"rollout..."
                    ):
                        # THE SINGLE INFERENCE CALL, shared with the
                        # benchmark and prepared-data modes. The PCAP
                        # feature pipeline above is untouched; only the
                        # rollout is routed through the common path so no
                        # source carries its own inference code.
                        _prep_pcap = dinp.from_pcap(
                            X_pcap,
                            timestamps=list(seq_timestamps),
                            source_name=uploaded.name,
                            n_rejected=int(n_gap_rejected),
                        )
                        traj_pcap = dinp.run_inference(
                            _prep_pcap, explainer, k_pcap
                        )                                   # (N, K+1)
                    _di_validation_card(_prep_pcap)
                    nowcast_pcap = traj_pcap[:, 0]
                    # The displayed probability is P(t+k) at the SELECTED
                    # horizon -- the same scalar the explainer attributes
                    # below. It was previously max(t+0..t+K), a different
                    # quantity from the one being explained, which is the
                    # mismatch Task 8 Part 1 forbids.
                    probs_pcap = traj_pcap[:, horizon]

                    # FROZEN validation threshold for this horizon, read
                    # from the baseline manifest -- never from the PCAP,
                    # and no longer the checkpoint-embedded 0.05, which
                    # was selected during training on the older
                    # single-day split and is not the reference threshold.
                    pcap_threshold = float(
                        explainer.horizon_thresholds[horizon]
                    )
                    alerts_pcap = (
                        probs_pcap >= pcap_threshold
                    )
                    n_alerts_pcap = int(
                        alerts_pcap.sum()
                    )
                    max_prob_pcap = float(
                        probs_pcap.max()
                    )

                    # --- Pseudo-datetimes so every other page's
                    # datetime-based plotting/slicing works unchanged
                    # (a PCAP has elapsed seconds since capture start,
                    # not calendar dates). ---
                    pcap_timestamps_dt = (
                        pd.Timestamp("2000-01-01")
                        + pd.to_timedelta(seq_timestamps, unit="s")
                    )
                    pcap_meta = pd.DataFrame({
                        "timestamp": pcap_timestamps_dt,
                        "attack_present": np.nan,
                    })

                    # --- MITRE ATT&CK mapping, reusing the same rule
                    # library the benchmark dataset uses. ---
                    with st.spinner(
                        "🗺️ Mapping to MITRE ATT&CK..."
                    ):
                        pcap_mapper = MITREMapper(
                            model=model,
                            threshold=pcap_threshold,
                            device="cpu",
                        )
                        first_alert_idx_pcap = (
                            int(np.argmax(alerts_pcap))
                            if alerts_pcap.any() else None
                        )
                        pcap_mappings = [
                            pcap_mapper.map_single_window(
                                x_window=X_pcap[i],
                                window_idx=i,
                                timestamp=str(pcap_timestamps_dt[i]),
                                total_windows=len(X_pcap),
                                first_alert_idx=first_alert_idx_pcap,
                                attack_prob=float(probs_pcap[i]),
                            )
                            for i in range(len(X_pcap))
                        ]
                        pcap_mitre_df = pd.DataFrame([
                            {
                                "window_index": m.window_index,
                                "timestamp": m.timestamp,
                                "attack_probability": (
                                    m.attack_probability
                                ),
                                "tactic_id": m.tactic_id,
                                "tactic_name": m.tactic_name,
                                "technique_id": m.technique_id or "",
                                "technique_name": (
                                    m.technique_name or ""
                                ),
                                "confidence": m.confidence,
                                "temporal_phase": m.temporal_phase,
                                "triggering_features": "; ".join(
                                    m.triggering_features
                                ),
                                "explanation": m.analyst_explanation,
                            }
                            for m in pcap_mappings
                        ])

                    # --- Feature attribution (Integrated Gradients,
                    # provenance-locked V4 explainer).
                    #
                    # Same explainer instance, same checkpoint and same
                    # horizon as the probability displayed above, so the
                    # explanation describes the number on screen. The
                    # baseline is the zero vector in scaled feature space
                    # (the training-partition mean under the frozen
                    # StandardScaler) -- deterministic, label-free, and
                    # independent of the uploaded PCAP.
                    #
                    # PCAP CAVEAT: the same V4 feature contract and
                    # transform chain are used, but flow-record
                    # production differs from the training path, so these
                    # feature values are approximate and are NOT
                    # numerically equivalent to training features. ---
                    pcap_importance_df = None
                    pcap_attributions = None
                    try:
                        with st.spinner(
                            "🔍 Computing feature attribution "
                            "(Integrated Gradients)..."
                        ):
                            r = explainer.attribute(X_pcap, horizon)
                            pcap_attributions = r["attributions"]
                            _abs = np.abs(pcap_attributions).sum(axis=1)
                            _sgn = pcap_attributions.sum(axis=1)
                            pcap_importance_df = pd.DataFrame(
                                {
                                    "feature_index": np.arange(
                                        len(INPUT_FEATURES)
                                    ),
                                    "feature": INPUT_FEATURES,
                                    "horizon": horizon,
                                    "mean_abs_attribution": _abs.mean(axis=0),
                                    "mean_signed_attribution": _sgn.mean(axis=0),
                                }
                            ).sort_values(
                                "mean_abs_attribution", ascending=False
                            ).reset_index(drop=True)

                            # Part 20 identity test, live: the attributed
                            # value must equal the displayed probability.
                            _ident = explainer.check_prediction_identity(
                                X_pcap, horizon
                            )
                            if not _ident["passed"]:
                                pcap_attributions = None
                                pcap_importance_df = None
                                st.error(
                                    "⚠️ Explanation withheld: the "
                                    "attributed prediction did not match "
                                    "the displayed prediction "
                                    f"(max diff {_ident['max_abs_diff']:.2e})."
                                )
                    except Exception as exc:
                        st.warning(
                            f"⚠️ Feature attribution failed: {exc}"
                        )

                    # --- Persist for every other page (Overview,
                    # MITRE Timeline, Feature Attribution, Window
                    # Inspector, Trajectory Rollout) via the sidebar
                    # "Data Source" toggle. ---
                    st.session_state.pcap_data = {
                        "file_id": file_id,
                        "filename": uploaded.name,
                        "format_label": _prep_pcap.format_label,
                        "source_type": dinp.SRC_PCAP,
                        "X": X_pcap,
                        "meta": pcap_meta,
                        "probs": probs_pcap,
                        "mitre_df": pcap_mitre_df,
                        "importance_df": pcap_importance_df,
                        "attributions": pcap_attributions,
                        "n_windows": len(X_pcap),
                    }
                    st.session_state.input_source = dinp.SRC_PCAP
                    st.success(
                        "📌 Results saved — every page that reads the "
                        "active source now shows **📡 Uploaded PCAP**."
                    )

                    # The sidebar (rendered earlier in this same script
                    # pass) can't see the pcap_data we just stashed
                    # above -- force one rerun so it picks it up
                    # immediately. Guarded so re-running the pipeline
                    # after the rerun doesn't loop forever.
                    if st.session_state.get(
                        "_pcap_rerun_done_for"
                    ) != file_id:
                        st.session_state._pcap_rerun_done_for = file_id
                        st.rerun()

                    if n_gap_rejected:
                        st.caption(
                            f"⏱️ {n_gap_rejected} candidate "
                            f"sequence(s) rejected: they spanned "
                            f"an empty-window temporal gap."
                        )

                    st.divider()

                    # Verdict
                    if n_alerts_pcap > 0:
                        st.error(
                            f"⚠️ **THREAT DETECTED** — "
                            f"{n_alerts_pcap} alert(s) "
                            f"triggered across "
                            f"{len(probs_pcap)} windows. "
                            f"Max probability: "
                            f"{max_prob_pcap:.4f}"
                        )
                    else:
                        st.success(
                            f"✅ **No threats detected** — "
                            f"All {len(probs_pcap)} windows "
                            f"below threshold. "
                            f"Max probability: "
                            f"{max_prob_pcap:.4f}"
                        )

                    # Results metrics
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric(
                        "🔢 Sequences Analyzed",
                        f"{len(probs_pcap)}"
                    )
                    rc2.metric(
                        "🚨 Alerts",
                        f"{n_alerts_pcap}"
                    )
                    rc3.metric(
                        "📈 Max Probability",
                        f"{max_prob_pcap:.4f}"
                    )

                    # Timeline chart
                    st.markdown("### Attack Probability "
                                "Timeline")

                    fig_pcap = go.Figure()
                    fig_pcap.add_trace(
                        go.Scatter(
                            x=seq_timestamps,
                            y=probs_pcap,
                            mode="lines+markers",
                            name="Peak prob (t+0..t+K)",
                            line=dict(
                                color="#8e44ad",
                                width=2,
                            ),
                            marker=dict(size=4),
                        )
                    )
                    fig_pcap.add_trace(
                        go.Scatter(
                            x=seq_timestamps,
                            y=nowcast_pcap,
                            mode="lines",
                            name="Nowcast prob (t+0)",
                            line=dict(
                                color="#3498db",
                                width=1,
                                dash="dot",
                            ),
                        )
                    )
                    fig_pcap.add_hline(
                        y=pcap_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=(
                            f"Threshold "
                            f"({pcap_threshold})"
                        ),
                    )

                    # Shade alert zones
                    for i, (t, a) in enumerate(
                        zip(seq_timestamps, alerts_pcap)
                    ):
                        if a:
                            fig_pcap.add_vrect(
                                x0=t - 5, x1=t + 5,
                                fillcolor="red",
                                opacity=0.15,
                                layer="below",
                                line_width=0,
                            )

                    fig_pcap.update_layout(
                        xaxis_title="Time (seconds)",
                        yaxis_title="Attack Probability",
                        yaxis_range=[0, 1],
                        template="plotly_dark",
                        height=400,
                        margin=dict(
                            l=40, r=20, t=30, b=40
                        ),
                    )
                    st.plotly_chart(
                        fig_pcap, use_container_width=True
                    )

                    # K-step recursive rollout trajectories
                    st.markdown(
                        f"### Recursive K={k_pcap} Rollout "
                        f"Trajectories"
                    )
                    st.caption(
                        "Each grey line is one sequence rolled "
                        "forward t+0 → t+%d in latent space; the "
                        "predicted latent is fed back into the "
                        "gated transition at every step. Red = "
                        "mean trajectory." % k_pcap
                    )
                    steps_pcap = list(range(k_pcap + 1))
                    fig_traj = go.Figure()
                    for row in traj_pcap:
                        fig_traj.add_trace(
                            go.Scatter(
                                x=steps_pcap, y=row,
                                mode="lines",
                                line=dict(
                                    color="rgba(150,150,150,0.25)",
                                    width=1,
                                ),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                    fig_traj.add_trace(
                        go.Scatter(
                            x=steps_pcap,
                            y=traj_pcap.mean(axis=0),
                            mode="lines+markers",
                            name="mean trajectory",
                            line=dict(color="#e74c3c", width=3),
                        )
                    )
                    fig_traj.add_hline(
                        y=pcap_threshold,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=(
                            f"Threshold ({pcap_threshold})"
                        ),
                    )
                    fig_traj.update_layout(
                        xaxis_title="Rollout step (t+k)",
                        yaxis_title="Attack Probability",
                        yaxis_range=[0, 1],
                        xaxis=dict(
                            tickmode="array",
                            tickvals=steps_pcap,
                        ),
                        template="plotly_dark",
                        height=380,
                        margin=dict(l=40, r=20, t=30, b=40),
                    )
                    st.plotly_chart(
                        fig_traj, use_container_width=True
                    )

                    # Protocol distribution
                    st.markdown(
                        "### Protocol Distribution"
                    )
                    proto_map = {
                        6: "TCP", 17: "UDP",
                        1: "ICMP", 2: "IGMP",
                    }
                    proto_counts = (
                        packet_df["protocol"]
                        .map(
                            lambda x: proto_map.get(
                                x, f"Other({x})"
                            )
                        )
                        .value_counts()
                    )
                    fig_proto = px.pie(
                        names=proto_counts.index,
                        values=proto_counts.values,
                        hole=0.4,
                        color_discrete_sequence=
                            px.colors.qualitative.Set2,
                    )
                    fig_proto.update_layout(
                        template="plotly_dark",
                        height=350,
                    )
                    st.plotly_chart(
                        fig_proto,
                        use_container_width=True,
                    )

                    # Raw results table
                    with st.expander(
                        "📋 Raw Results Table"
                    ):
                        res_df = pd.DataFrame({
                            "Window": range(
                                len(probs_pcap)
                            ),
                            "Time (s)": [
                                f"{t:.1f}"
                                for t in seq_timestamps
                            ],
                            "Nowcast (t+0)": [
                                f"{p:.4f}"
                                for p in nowcast_pcap
                            ],
                            "Peak (t+0..t+K)": [
                                f"{p:.4f}"
                                for p in probs_pcap
                            ],
                            "Alert": [
                                "🚨 YES" if a else "✅ No"
                                for a in alerts_pcap
                            ],
                        })
                        for _kk in range(1, k_pcap + 1):
                            res_df[f"t+{_kk}"] = [
                                f"{p:.4f}"
                                for p in traj_pcap[:, _kk]
                            ]
                        st.dataframe(
                            res_df,
                            use_container_width=True,
                            height=400,
                        )

        else:
            # Show instructions when no file uploaded
            st.info(
                "👆 Upload a `.pcap` file above to "
                "start the analysis pipeline.\n\n"
                "**Supported formats:** `.pcap`, "
                "`.pcapng`, `.cap`\n\n"
                "**How it works** (canonical `pcap_ingest_v4` "
                "pipeline):\n"
                "1. Packets are streamed with Scapy "
                "`PcapReader`\n"
                "2. Flows are reconstructed (5-tuple)\n"
                "3. 52 features are extracted per "
                "10-second window\n"
                "4. Features are z-score normalized using the "
                "`multiday_benchmark_K5` scaler\n"
                "5. V4 World Model runs a recursive K=5 latent "
                "rollout; the alert threshold is the **frozen, "
                "validation-selected** value read from "
                "`docs/v4_baseline_manifest.json` (not the "
                "checkpoint-embedded 0.05 training artefact)"
            )

            st.markdown("### Pipeline Architecture")
            st.code(
                ".pcap file\n"
                "  → Scapy PcapReader (streaming)\n"
                "  → Bidirectional flow reconstruction\n"
                "  → 10-second window aggregation\n"
                "  → 52-feature extraction\n"
                "  → Z-score normalization (K5 scaler)\n"
                "  → V4 recursive K=5 latent rollout\n"
                "  → Threat alerts + MITRE mapping",
                language=None,
            )




# =============================================================
# PAGE: METHODOLOGY & LIMITS  (scientific disclosure)
# =============================================================

elif page == "📐 Methodology & Limits":

    sub = SCREENING
    ew = sd.early_warning_summary(sub)
    man = sub.manifest

    st.markdown("# 📐 Methodology & Limits")
    st.markdown(
        "> Everything this console claims, and everything it cannot claim. "
        "Nothing here is hidden behind the demo."
    )

    # ---------- WHAT IS FROZEN ----------
    st.markdown("### What is frozen")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        card("Checkpoint", f"{explainer.provenance.checkpoint_hash[:10]}…",
             "hash-verified before use")
    with f2:
        card("Input representation", "52-D",
             "26 base + 26 relative-change")
    with f3:
        card("Observed history", "10 windows",
             "100 s at 10 s resolution")
    with f4:
        card("Rollout", "K = 5", "50 s forecast horizon")

    arch = man.get("v4_configuration", {})
    st.markdown(
        f"""
| Component | Frozen value |
|---|---|
| Checkpoint | `models/latent_world_model_v4/latent_world_model_v4.pt` |
| Parameters | {arch.get('parameter_count', 'n/a'):,} |
| Encoder | 2-layer LSTM (52 → 128) |
| Latent dimension | {arch.get('latent_dimension', 32)} |
| Transition | gated residual + LayerNorm (deterministic) |
| State decoder | 26-D network state |
| Threat head | scalar attack probability |
| Input scaler | `data/processed/multiday_benchmark_K5/input_scaler.pkl` |
| Feature order | `results/v4_screening_mvp/feature_names.txt` |
"""
    )

    st.divider()

    # ---------- EVALUATION ----------
    st.markdown("### Held-out evaluation")
    split = man.get("split", {})
    st.markdown(
        f"- **Train:** {', '.join(split.get('train_days', []))} "
        f"— {split.get('train_sequences', 0):,} sequences (85% chronological head of each day)\n"
        f"- **Validation:** 15% chronological tail of the same days "
        f"— {split.get('validation_sequences', 0):,} sequences\n"
        f"- **Held-out test:** **March 2, 2018** only "
        f"— {split.get('test_sequences', 0):,} sequences\n"
        f"- Mar-2 contributes **0 windows** to any scaler or model fit "
        f"(`mar2_in_any_fit = {split.get('mar2_in_any_fit')}`)\n"
        f"- Every threshold was selected on **validation only** and applied unchanged "
        f"to the test day."
    )

    if REFERENCE is not None:
        st.markdown("#### Reproduced reference metrics (held-out March-2)")
        pt_rows = []
        for k in [1, 2, 3, 5]:
            r = REFERENCE["future_prediction"][f"k={k}"]
            pt_rows.append({
                "Task": f"Point forecast t+{k}",
                "F1": f"{r['f1']:.4f}", "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}", "ROC-AUC": f"{r['roc_auc']:.4f}",
                "FPR": f"{r['fpr']:.4f}", "Threshold": f"{r['threshold']:.2f}",
            })
        r5 = REFERENCE["trajectory_event_max"]["k=5"]
        pt_rows.append({
            "Task": "Trajectory MAX K=5 (primary)",
            "F1": f"{r5['f1']:.4f}", "Precision": f"{r5['precision']:.4f}",
            "Recall": f"{r5['recall']:.4f}", "ROC-AUC": f"{r5['roc_auc']:.4f}",
            "FPR": f"{r5['fpr']:.4f}", "Threshold": f"{r5['threshold']:.2f}",
        })
        st.dataframe(pd.DataFrame(pt_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Reproduced exactly (to 1e-9) by `src/build_v4_screening_mvp.py` against "
            "`docs/v4_baseline_manifest.json`. The primary forecast alert in this "
            "console uses the trajectory-MAX K=5 threshold."
        )

    st.divider()

    # ---------- EARLY WARNING ----------
    st.markdown("### Early warning — the honest numbers")
    w1, w2, w3 = st.columns(3)
    with w1:
        card("Onset coverage", f"{ew['strict_coverage_pct']:.1f}%",
             f"{ew['strict_warned']} of {ew['evaluable_onsets']} onsets", "card-forecast")
    with w2:
        card("Mean strict lead", f"{ew['strict_mean_lead_s']:.0f} s",
             "strict pre-attack convention", "card-forecast")
    with w3:
        card("Evaluable onsets", f"{ew['evaluable_onsets']}",
             "single held-out day")

    st.warning(
        f"**This is a small sample.** {ew['caveat']} A 4th attack episode is already "
        f"active at the first evaluable window of the day and therefore has no "
        f"observable onset — it is excluded from early-warning analysis entirely, "
        f"though it is still fully counted in the classification metrics above."
    )
    st.markdown(
        f"**Two conventions exist, and they disagree.** The strict figures above count "
        f"only windows that are benign at both the observation time *t* and the "
        f"evaluated target *t+1*. The originally published convention credits the first "
        f"alerting window in the 5 windows before onset — which, for 2 of the 3 "
        f"episodes, still lies *inside* the preceding active attack. That convention "
        f"reports **{ew['reference_coverage_pct']:.1f}% / "
        f"{ew['reference_mean_lead_s']:.0f} s**. Coverage is identical under both; only "
        f"the lead time differs. This console headlines the strict number."
    )

    st.divider()

    # ---------- BASELINE ----------
    st.markdown("### The persistence baseline is near ceiling")
    st.markdown(
        "On ordinary *t+k* classification, simply predicting **\"the next window looks "
        "like the current one\"** achieves **F1 ≈ 0.997–0.999** on this day, because "
        "attack episodes are long relative to the 10 s window. "
        "That baseline is not beaten by the world model on that framing, and the "
        "console does not claim otherwise."
    )
    st.info(
        "**Why the forecasting framing matters.** Persistence, by construction, cannot "
        "warn *before* a state change — it only restates the present. The value of the "
        "rollout is in the pre-attack transition region, which is exactly where the "
        "early-warning evaluation above is measured, and exactly where the sample is "
        "small. That is the honest scope of the current evidence."
    )

    st.divider()

    # ---------- KNOWN CAVEATS ----------
    st.markdown("### Known caveats carried from the frozen V4 manifest")
    st.error(
        "**Train/evaluation normalisation mismatch (out-of-distribution).** The V4 "
        "checkpoint was *trained* on the single-day `temporal_benchmark_v2` arrays, "
        "whose scaler was fit on 2,384 Feb-28 windows. The reference evaluation — and "
        "this console — feed it `multiday_benchmark_K5` arrays standardised by a "
        "*different* scaler fit on 8,575 Feb-14/15/28 windows. The reported results are "
        "therefore produced under a normalisation the model never saw during training. "
        "This is recorded in the frozen manifest and deliberately left unchanged."
    )

    caveats = [
        "**Training is not bit-reproducible.** The exact arrays that produced the "
        "checkpoint were regenerated after it was trained and no longer match "
        "bit-for-bit. Evaluation *is* exactly reproducible; training is not.",
        "**No K=4 reference number.** The frozen reference evaluation covers horizons "
        "1, 2, 3 and 5. `prob_t4` (+40s) is displayed for continuity but has no frozen "
        "point threshold and no published metric.",
        "**The transition is deterministic.** V4 learns a point estimate in latent "
        "space, not a distribution P(S_t+1 | S_t). The console shows no predictive "
        "interval because the model does not produce one.",
        "**Probabilities are not calibrated.** Threat values are sigmoid outputs "
        "thresholded at a frozen, validation-selected operating point. They are "
        "screening scores, not calibrated likelihoods.",
        "**One held-out day.** All results here are from March 2, 2018 of "
        "CSE-CIC-IDS2018. Generalisation across days is analysed separately "
        "(`docs/ROTATED_GENERALIZATION.md`) and is not claimed by this console.",
        "**A real capture gap exists in the day.** See below; it is never interpolated.",
        "**ATT&CK context is rule-based.** It is a deterministic telemetry rule library, "
        "not a model output, and was never trained on ATT&CK labels.",
    ]
    for c in caveats:
        st.markdown(f"- {c}")

    st.divider()

    # ---------- DATA GAP ----------
    st.markdown("### Capture gap")
    cont = man.get("contiguity", {})
    for g in sub.capture_gaps():
        st.markdown(
            f'<div class="banner banner-gap">'
            f'<b>⚠ {g["gap_seconds"] / 3600:.2f} h with no windows</b><br>'
            f'<span class="note"><code>{g["from_timestamp"]}</code> → '
            f'<code>{g["to_timestamp"]}</code>. A real discontinuity in the raw '
            f'CSE-CIC-IDS2018 capture. Of {cont.get("candidate_blocks", "?")} candidate '
            f'sequence blocks, {cont.get("rejected_blocks", "?")} were rejected for '
            f'straddling it — matching the '
            f'{cont.get("rejected_blocks_expected_from_gaps", "?")} predicted from the '
            f'gap position. No sequence spans it, nothing is interpolated, and no '
            f'synthetic windows are created.</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ---------- PROVENANCE ----------
    st.markdown("### Substrate provenance")
    rd = sub.readiness
    gates = rd.get("checks", {})
    n_pass = sum(1 for v in gates.values() if v == "PASS")
    p1, p2, p3 = st.columns(3)
    with p1:
        card("Readiness gates", f"{n_pass}/{len(gates)}",
             f"overall: {rd.get('overall', 'n/a')}",
             "card-ok" if rd.get("overall") == "PASS" else "card-alert")
    with p2:
        card("Replay sequences", f"{sub.n_sequences:,}",
             "real held-out March-2 telemetry")
    with p3:
        card("Synthetic data", "none",
             "screening_X.npy is byte-identical to X_test.npy", "card-ok")

    with st.expander("All 14 readiness gates"):
        st.dataframe(
            pd.DataFrame({"Gate": list(gates.keys()), "Status": list(gates.values())}),
            use_container_width=True, hide_index=True,
        )
    with st.expander("Substrate notes recorded at build time"):
        for note in rd.get("notes", []):
            st.markdown(f"- {note}")

    st.markdown(
        '<div class="demo-footer">Offline replay of real held-out '
        'March-2 telemetry · all figures read from validated artifacts</div>',
        unsafe_allow_html=True,
    )


# =============================================================
# FOOTER
# =============================================================

st.divider()
_footer_seq = (f" · replay sequence #{SEQ_ID} of {SCREENING.n_sequences - 1}"
               if SCREENING is not None else "")
st.markdown(
    f"<div class='demo-footer'>"
    f"<b>Offline replay of real held-out March-2 telemetry</b> "
    f"(CSE-CIC-IDS2018){_footer_seq}<br>"
    f"🛡️ Cyber World Model — Predictive Cyber Defence Using World Models · "
    f"Smart India Hackathon 2026 · frozen V4 checkpoint, no network access required"
    f"</div>",
    unsafe_allow_html=True,
)
