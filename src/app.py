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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

# Canonical PCAP -> V4 inference pipeline (single source of truth for
# the "Live PCAP Ingestion" page). streamlit runs this file from src/,
# so a bare module import resolves.
import pcap_ingest_v4 as pcap_v4

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


# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="Cyber World Model — SIH 2026",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# CUSTOM CSS
# =============================================================

st.markdown("""
<style>
    /* Dark theme metrics */
    .metric-card {
        background: linear-gradient(135deg,
            #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #0f3460;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e94560;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0b0;
        margin-top: 4px;
    }

    /* Tactic badge */
    .tactic-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }

    /* Alert card */
    .alert-card {
        background: #1a1a2e;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Section headers */
    .section-header {
        border-bottom: 2px solid #e94560;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


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

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/"
        "security-shield-green.png",
        width=64,
    )
    st.title("🛡️ Cyber World Model")
    st.caption("SIH 2026 — Predictive Cyber Defence")

    st.divider()

    st.markdown("### Architecture")
    st.markdown("""
    - **Encoder**: LSTM (128 hidden, 2 layers)
    - **Latent Space**: 32 dimensions
    - **Transition**: Gated Residual + LayerNorm (V4)
    - **Classification**: Attack probability head
    - **Training**: Multi-step objective (K=5)
    - **Input**: 52 temporal network features
    - **Window**: 10 timesteps × 10s each
    """)

    st.divider()

    st.markdown("### Model Performance")
    model, ckpt, explainer = load_model()
    col1, col2 = st.columns(2)
    col1.metric("F1 Score",
                f"{ckpt['test_attack_f1']:.3f}")
    col2.metric("ROC-AUC",
                f"{ckpt['test_roc_auc']:.3f}")

    st.caption(
        f"checkpoint `{explainer.provenance.checkpoint_hash[:12]}…` "
        f"hash-verified against the frozen baseline manifest"
    )

    st.divider()

    st.markdown("### Forecast Horizon")
    horizon = st.radio(
        "Forecast horizon",
        VERIFIED_HORIZONS,
        index=VERIFIED_HORIZONS.index(DEFAULT_HORIZON),
        format_func=lambda k: f"t+{k}  ({10 * k}s ahead)",
        label_visibility="collapsed",
    )
    threshold = explainer.horizon_thresholds[horizon]
    st.metric(f"Frozen threshold (K={horizon})", f"{threshold:.2f}")
    st.caption(
        "Validation-selected and frozen; test labels were not used "
        "to choose it."
    )

    st.divider()

    st.markdown("### Navigation")
    page = st.radio(
        "Select View",
        [
            "🏠 Overview",
            "🗺️ MITRE ATT&CK Timeline",
            "🔍 Feature Attribution",
            "📊 Window Inspector",
            "🔮 Trajectory Rollout",
            "📡 Live PCAP Ingestion",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Data Source")
    pcap_data = st.session_state.get("pcap_data")
    if pcap_data is not None:
        data_source = st.radio(
            "Analyze which data?",
            ["📊 Benchmark Dataset", "📡 My Uploaded PCAP"],
            index=1,
            label_visibility="collapsed",
        )
        st.caption(
            f"📡 `{pcap_data['filename']}` — "
            f"{pcap_data['n_windows']} windows analyzed"
        )
    else:
        data_source = "📊 Benchmark Dataset"
        st.caption(
            "Upload a PCAP on **📡 Live PCAP Ingestion** to "
            "analyze your own traffic on every page."
        )


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
# ACTIVE DATASET (Benchmark vs. uploaded-PCAP toggle)
# =============================================================
# Every page below reads *_active variables so the "Data Source"
# sidebar toggle can swap in the just-uploaded PCAP's own results
# (stashed in st.session_state by the Live PCAP Ingestion page)
# without duplicating each page's rendering logic.

using_pcap = (
    data_source == "📡 My Uploaded PCAP" and pcap_data is not None
)

if using_pcap:
    X_active = pcap_data["X"]
    meta_active = pcap_data["meta"]
    probs_active = pcap_data["probs"]
    mitre_active = pcap_data["mitre_df"]
    importance_active = pcap_data["importance_df"]
    attributions_active = pcap_data["attributions"]
    n_windows_active = pcap_data["n_windows"]
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
# PAGE: OVERVIEW
# =============================================================

if page == "🏠 Overview":

    st.markdown(
        "# 🛡️ Predictive Cyber Defence "
        "Using World Models"
    )
    st.markdown(
        "> *Anticipating attacker progression before "
        "compromise — SIH 2026*"
    )

    if using_pcap:
        st.info(
            f"📡 Showing your uploaded PCAP: "
            f"**{pcap_data['filename']}**"
        )

    st.divider()

    # Key metrics
    if has_ground_truth:
        c1, c2, c3, c4, c5 = st.columns(5)
    else:
        c1, c2, c4, c5 = st.columns(4)

    c1.metric(
        "Total Windows", f"{n_windows_active}",
        help="10-second temporal windows analyzed"
    )
    c2.metric(
        "🚨 Alerts",
        f"{n_alerts_active}",
        f"{n_alerts_active/n_windows_active*100:.1f}% of windows",
    )
    if has_ground_truth:
        c3.metric(
            "Ground Truth Attacks",
            f"{n_attacks_gt}",
            f"{n_attacks_gt/n_windows*100:.1f}%",
        )
    c4.metric(
        "Features",
        "52",
        "26 base + 26 relative change",
    )
    c5.metric(
        "Latent Dims",
        "32",
        "Compressed state space",
    )

    st.divider()

    # Quick timeline
    st.markdown("### 📈 Attack Probability Timeline")

    if has_ground_truth:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.3, 0.7],
            subplot_titles=[
                "Ground Truth", "Model Predictions"
            ],
        )

        # Ground truth
        fig.add_trace(
            go.Scatter(
                x=meta_active["timestamp"],
                y=meta_active["attack_present"],
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.3)",
                line=dict(color="rgba(231,76,60,0.6)"),
                name="Attack (GT)",
            ),
            row=1, col=1,
        )
        prob_row = 2
    else:
        fig = make_subplots(rows=1, cols=1)
        prob_row = 1

    # Attack probability
    fig.add_trace(
        go.Scatter(
            x=meta_active["timestamp"],
            y=probs_active,
            mode="lines",
            line=dict(color="#3498db", width=1.5),
            name="Attack Probability",
        ),
        row=prob_row, col=1,
    )

    # Threshold line
    fig.add_hline(
        y=threshold, row=prob_row, col=1,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold={threshold}",
    )

    fig.update_layout(
        height=450,
        template="plotly_dark",
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.1,
            x=0.5, xanchor="center",
        ),
    )
    if has_ground_truth:
        fig.update_yaxes(
            title_text="Label", row=1, col=1,
        )
    fig.update_yaxes(
        title_text="Probability", row=prob_row, col=1,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Architecture diagram
    st.divider()
    st.markdown("### 🏗️ System Architecture")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### 1️⃣ Ingest")
        st.info(
            "52 network flow features aggregated "
            "into 10-second temporal windows"
        )
    with col2:
        st.markdown("#### 2️⃣ Encode")
        st.info(
            "LSTM encoder compresses 10-step "
            "sequences into 32-dim latent state Zₜ"
        )
    with col3:
        st.markdown("#### 3️⃣ Predict")
        st.info(
            "Residual transition forecasts Zₜ₊₁, "
            "attack head outputs threat probability"
        )
    with col4:
        st.markdown("#### 4️⃣ Explain")
        st.info(
            "MITRE ATT&CK mapping + Integrated "
            "Gradients attribution for analysts"
        )


# =============================================================
# PAGE: MITRE ATT&CK TIMELINE
# =============================================================

elif page == "🗺️ MITRE ATT&CK Timeline":

    st.markdown("# 🗺️ MITRE ATT&CK Mapping")
    st.markdown(
        "> *Mapping model predictions to the "
        "MITRE ATT&CK framework for actionable "
        "intelligence*"
    )

    if mitre_active is None:
        if using_pcap:
            st.error(
                "MITRE mapping for this PCAP is not "
                "available (feature attribution failed "
                "during upload — see the Live PCAP "
                "Ingestion page)."
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

    if using_pcap:
        st.info(
            f"📡 Showing your uploaded PCAP: "
            f"**{pcap_data['filename']}**"
        )

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

    mitre_active["timestamp_dt"] = pd.to_datetime(
        mitre_active["timestamp"]
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
                x=pd.to_datetime(
                    tactic_data["timestamp"]
                ),
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

    if importance_active is None:
        if using_pcap:
            st.error(
                "Feature attribution for this PCAP is not "
                "available (it may have failed during "
                "upload — see the Live PCAP Ingestion page)."
            )
        else:
            st.error(
                "Attribution data not found. "
                "Run `python src/run_v4_explainability.py` first."
            )
        st.stop()

    if using_pcap:
        st.info(
            f"📡 Showing your uploaded PCAP: "
            f"**{pcap_data['filename']}**"
        )
        st.warning(
            "**PCAP caveat.** This explanation uses the same V4 "
            "feature contract and transform chain as the training "
            "path, but flow-record production differs, so the "
            "resulting feature values are approximate. They are NOT "
            "numerically equivalent to training features."
        )

    st.divider()

    # Global importance
    st.markdown("### 🌍 Global Feature Importance")

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
        "> *Drill into any specific time window "
        "to see the model's prediction, MITRE "
        "mapping, and feature explanation.*"
    )

    if using_pcap:
        st.info(
            f"📡 Showing your uploaded PCAP: "
            f"**{pcap_data['filename']}**"
        )

    st.divider()

    # Window selector
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_idx = st.slider(
            "Select Window Index",
            0, n_windows_active - 1, 0,
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

    if using_pcap:
        st.info(
            f"📡 Showing your uploaded PCAP: "
            f"**{pcap_data['filename']}**"
        )

    st.divider()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        rollout_window = st.slider(
            "Select starting window",
            0, n_windows_active - 1,
            min(140, n_windows_active - 1),
        )
    with col2:
        k_steps = st.slider(
            "Rollout steps (K)",
            1, 10, 5,
        )

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

elif page == "📡 Live PCAP Ingestion":

    st.markdown(
        "## 📡 Live PCAP Ingestion"
    )
    st.markdown(
        "> **Drag & drop a raw `.pcap` file** and "
        "watch the World Model analyze it in "
        "real time."
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
        return pcap_v4.parse_pcap_streaming(tmp_path)

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
                    roll_pcap = pcap_v4.run_k_rollout(
                        model, X_pcap, k_pcap
                    )
                traj_pcap = roll_pcap["probs"]          # (N, K+1)
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
                    "X": X_pcap,
                    "meta": pcap_meta,
                    "probs": probs_pcap,
                    "mitre_df": pcap_mitre_df,
                    "importance_df": pcap_importance_df,
                    "attributions": pcap_attributions,
                    "n_windows": len(X_pcap),
                }
                st.success(
                    "📌 Results saved — every other page now "
                    "shows **📡 My Uploaded PCAP** via the "
                    "sidebar Data Source toggle."
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
            "rollout; threshold (0.05) is read from the "
            "checkpoint"
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
# FOOTER
# =============================================================

st.divider()
st.markdown(
    "<div style='text-align: center; "
    "color: #7f8c8d; font-size: 0.85rem;'>"
    "🛡️ Cyber World Model — Predictive Cyber "
    "Defence Using World Models<br>"
    "Smart India Hackathon (SIH) 2026 "
    "| Built with PyTorch + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
