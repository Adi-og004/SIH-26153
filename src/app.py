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
import pickle
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
import torch.nn as nn


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

DATA_DIR = Path("data/processed/temporal_benchmark_v2")
MODEL_DIR = Path("models/latent_world_model_v4")
MITRE_DIR = MODEL_DIR / "mitre_mapping"
SHAP_DIR = MODEL_DIR / "shap_analysis"
ROLLOUT_DIR = MODEL_DIR / "k_step_rollout"


# =============================================================
# V4 MODEL DEFINITION (Gated Residual + LayerNorm)
# =============================================================

class LatentWorldModelV4(nn.Module):

    def __init__(
        self, input_size, hidden_size,
        latent_size, output_size,
        num_layers, dropout
    ):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout if num_layers > 1
                else 0.0
            )
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(
            hidden_size, latent_size
        )

        # Gated residual transition (V4)
        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size)
        )
        self.gate_fc = nn.Linear(
            latent_size, latent_size
        )
        self.transition_norm = nn.LayerNorm(
            latent_size
        )

        self.decoder_fc1 = nn.Linear(latent_size, 64)
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(64, output_size)
        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(last_hidden)
        return self.encoder_fc(last_hidden)

    def transition_step(self, z):
        candidate = self.transition_mlp(z)
        gate = torch.sigmoid(self.gate_fc(z))
        z_next = gate * candidate + (1 - gate) * z
        return self.transition_norm(z_next)

    def decode(self, z):
        h = self.decoder_fc1(z)
        h = self.decoder_activation(h)
        return self.decoder_fc2(h)

    def predict_attack(self, z):
        return self.attack_head(z).squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        z_next = self.transition_step(z)
        return self.decode(z_next), z_next, \
            self.predict_attack(z_next), z


# =============================================================
# FEATURE NAMES
# =============================================================

INPUT_FEATURES = [
    "flow_count", "log_flow_count",
    "log_total_packets", "log_total_bytes",
    "window_packet_rate", "window_byte_rate",
    "syn_rate", "ack_rate", "rst_rate", "fin_rate",
    "psh_rate", "urg_rate",
    "mean_pkt_len", "std_pkt_len",
    "log_mean_flow_duration",
    "log_mean_fwd_iat", "log_mean_bwd_iat",
    "fwd_packet_ratio", "bwd_packet_ratio",
    "fwd_byte_ratio", "bwd_byte_ratio",
    "mean_down_up_ratio",
    "log_unique_dst_ports", "unique_protocols",
    "dst_port_entropy", "protocol_entropy",
    "flow_count_rel_change",
    "log_flow_count_rel_change",
    "log_total_packets_rel_change",
    "log_total_bytes_rel_change",
    "window_packet_rate_rel_change",
    "window_byte_rate_rel_change",
    "syn_rate_rel_change", "ack_rate_rel_change",
    "rst_rate_rel_change", "fin_rate_rel_change",
    "psh_rate_rel_change", "urg_rate_rel_change",
    "mean_pkt_len_rel_change",
    "std_pkt_len_rel_change",
    "log_mean_flow_duration_rel_change",
    "log_mean_fwd_iat_rel_change",
    "log_mean_bwd_iat_rel_change",
    "fwd_packet_ratio_rel_change",
    "bwd_packet_ratio_rel_change",
    "fwd_byte_ratio_rel_change",
    "bwd_byte_ratio_rel_change",
    "mean_down_up_ratio_rel_change",
    "log_unique_dst_ports_rel_change",
    "unique_protocols_rel_change",
    "dst_port_entropy_rel_change",
    "protocol_entropy_rel_change",
]


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
def load_mitre_mappings():
    csv_path = MITRE_DIR / "mitre_summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_feature_importance():
    csv_path = SHAP_DIR / "feature_importance.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_attributions():
    npy_path = SHAP_DIR / "attributions.npy"
    if npy_path.exists():
        return np.load(npy_path)
    return None


@st.cache_data
def load_window_explanations():
    csv_path = SHAP_DIR / "window_explanations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_DIR / "latent_world_model_v4.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = LatentWorldModelV4(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        latent_size=checkpoint["latent_size"],
        output_size=checkpoint["output_size"],
        num_layers=checkpoint["num_layers"],
        dropout=0.20,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()
    return model, checkpoint


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
    model, ckpt = load_model()
    col1, col2 = st.columns(2)
    col1.metric("F1 Score",
                f"{ckpt['test_attack_f1']:.3f}")
    col2.metric("ROC-AUC",
                f"{ckpt['test_roc_auc']:.3f}")

    st.metric("Threshold",
              f"{ckpt['selected_threshold']:.2f}")

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


# =============================================================
# LOAD DATA
# =============================================================

X_test, test_meta = load_test_data()
mitre_df = load_mitre_mappings()
importance_df = load_feature_importance()
attributions = load_attributions()
explanations_df = load_window_explanations()

# Compute attack probabilities
with torch.no_grad():
    X_t = torch.tensor(X_test, dtype=torch.float32)
    z = model.encode(X_t)
    z_next = model.transition_step(z)
    logits = model.predict_attack(z_next)
    probs = torch.sigmoid(logits).numpy()

threshold = ckpt["selected_threshold"]
n_windows = len(test_meta)
n_alerts = (probs >= threshold).sum()
n_attacks_gt = test_meta["attack_present"].sum()


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

    st.divider()

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Total Windows", f"{n_windows}",
        help="10-second temporal windows analyzed"
    )
    c2.metric(
        "🚨 Alerts",
        f"{n_alerts}",
        f"{n_alerts/n_windows*100:.1f}% of windows",
    )
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
            x=test_meta["timestamp"],
            y=test_meta["attack_present"],
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.3)",
            line=dict(color="rgba(231,76,60,0.6)"),
            name="Attack (GT)",
        ),
        row=1, col=1,
    )

    # Attack probability
    fig.add_trace(
        go.Scatter(
            x=test_meta["timestamp"],
            y=probs,
            mode="lines",
            line=dict(color="#3498db", width=1.5),
            name="Attack Probability",
        ),
        row=2, col=1,
    )

    # Threshold line
    fig.add_hline(
        y=threshold, row=2, col=1,
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
    fig.update_yaxes(
        title_text="Label", row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Probability", row=2, col=1,
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

    if mitre_df is None:
        st.error(
            "MITRE mapping data not found. "
            "Run `python src/mitre_mapper.py` first."
        )
        st.stop()

    st.divider()

    # Tactic distribution
    st.markdown("### Tactic Distribution")

    alert_df = mitre_df[
        mitre_df["tactic_id"] != "NONE"
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

    mitre_df["timestamp_dt"] = pd.to_datetime(
        mitre_df["timestamp"]
    )

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
            x=mitre_df["timestamp_dt"],
            y=mitre_df["ground_truth"],
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

    # Attack probability
    fig_timeline.add_trace(
        go.Scatter(
            x=mitre_df["timestamp_dt"],
            y=mitre_df["attack_probability"],
            mode="lines",
            line=dict(color="#ecf0f1", width=1),
            name="Attack Prob",
            showlegend=True,
        ),
        row=2, col=1,
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
            row=2, col=1,
        )

    # Threshold
    fig_timeline.add_hline(
        y=threshold, row=2, col=1,
        line_dash="dash", line_color="red",
        annotation_text="Threshold",
    )

    # Confidence
    conf_map = {"n/a": 0, "low": 1,
                "medium": 2, "high": 3}
    conf_values = mitre_df["confidence"].map(
        conf_map
    ).fillna(0)
    conf_colors = [
        CONFIDENCE_COLORS.get(c, "#34495e")
        for c in mitre_df["confidence"]
    ]

    fig_timeline.add_trace(
        go.Bar(
            x=mitre_df["timestamp_dt"],
            y=conf_values,
            marker_color=conf_colors,
            name="Confidence",
            showlegend=False,
        ),
        row=3, col=1,
    )

    fig_timeline.update_layout(
        height=700,
        template="plotly_dark",
        legend=dict(
            orientation="h", y=-0.08,
            x=0.5, xanchor="center",
        ),
    )
    fig_timeline.update_yaxes(
        title_text="Label", row=1, col=1,
    )
    fig_timeline.update_yaxes(
        title_text="Prob", row=2, col=1,
    )
    fig_timeline.update_yaxes(
        title_text="Conf",
        tickvals=[0, 1, 2, 3],
        ticktext=["None", "Low", "Med", "High"],
        row=3, col=1,
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
        "> *Why does the model predict an attack? "
        "Integrated Gradients reveals which network "
        "telemetry features drive each prediction.*"
    )

    if importance_df is None:
        st.error(
            "Attribution data not found. "
            "Run `python src/shap_explainer.py` first."
        )
        st.stop()

    st.divider()

    # Global importance
    st.markdown("### 🌍 Global Feature Importance")

    top_n = st.slider(
        "Number of features to show",
        5, 52, 20,
    )

    top_df = importance_df.head(top_n)

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
    if attributions is not None:
        st.markdown(
            "### 🌡️ Attribution Heatmap Over Time"
        )

        top10_feats = importance_df.head(10)[
            "feature"
        ].values
        top10_idx = [
            INPUT_FEATURES.index(f)
            for f in top10_feats
        ]

        n = min(attributions.shape[0], n_windows)
        last_attr = attributions[:n, -1, :]
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
                x=test_meta["timestamp"][:n],
                y=probs[:n],
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
                x=test_meta["timestamp"][:n],
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

    if attributions is not None:
        n = min(attributions.shape[0], n_windows)
        last_attr = attributions[:n, -1, :]
        attack_mask = (
            test_meta["attack_present"]
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

    st.divider()

    # Window selector
    col1, col2 = st.columns([2, 1])

    with col1:
        selected_idx = st.slider(
            "Select Window Index",
            0, n_windows - 1, 0,
        )

    with col2:
        # Quick jump to alerts
        alert_indices = np.where(
            probs >= threshold
        )[0]
        if len(alert_indices) > 0:
            jump_to = st.selectbox(
                "Jump to alert window",
                alert_indices,
                format_func=lambda x: (
                    f"Window {x} "
                    f"(p={probs[x]:.2f})"
                ),
            )
            if st.button("Go →"):
                selected_idx = jump_to

    st.divider()

    # Window details
    prob = probs[selected_idx]
    ts = test_meta["timestamp"].iloc[selected_idx]
    gt = test_meta["attack_present"].iloc[
        selected_idx
    ]
    gt_label = "🔴 ATTACK" if gt else "🟢 BENIGN"

    # Top metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Window", f"#{selected_idx}")
    mc2.metric("Timestamp", str(ts)[11:19])
    mc3.metric(
        "Attack Probability",
        f"{prob:.3f}",
    )
    mc4.metric("Ground Truth", gt_label)

    # MITRE mapping for this window
    if mitre_df is not None:
        row = mitre_df.iloc[selected_idx]

        st.divider()
        st.markdown("### 🗺️ MITRE ATT&CK Assessment")

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
    if (attributions is not None
            and selected_idx < attributions.shape[0]):

        st.divider()
        st.markdown("### 🔍 Feature Attribution")

        attr = attributions[selected_idx, -1, :]
        vals = X_test[selected_idx, -1, :]

        top_k = 10
        abs_attr = np.abs(attr)
        top_idx = np.argsort(abs_attr)[::-1][:top_k]

        attr_data = []
        for idx in top_idx:
            attr_data.append({
                "Feature": INPUT_FEATURES[idx],
                "Value (σ)": float(vals[idx]),
                "Attribution": float(attr[idx]),
                "Direction": (
                    "↑ Attack" if attr[idx] > 0
                    else "↓ Benign"
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
        raw_vals = X_test[selected_idx, -1, :]
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

    st.divider()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        rollout_window = st.slider(
            "Select starting window",
            0, n_windows - 1, 140,
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
            X_test[rollout_window:rollout_window+1],
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
        from scapy.all import rdpcap, IP, TCP, UDP
        with tempfile.NamedTemporaryFile(
            suffix=".pcap", delete=False
        ) as tmp:
            tmp.write(pcap_bytes)
            tmp_path = tmp.name

        packets = rdpcap(tmp_path)
        rows = []
        for pkt in packets:
            if not pkt.haslayer(IP):
                continue
            ip = pkt[IP]
            src_port, dst_port, flags = 0, 0, ""
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                flags = str(tcp.flags)
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                src_port = udp.sport
                dst_port = udp.dport
            rows.append({
                "timestamp": float(pkt.time),
                "src_ip": ip.src,
                "dst_ip": ip.dst,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": ip.proto,
                "pkt_len": len(pkt),
                "flags": flags,
            })
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df["timestamp"] = (
                df["timestamp"] - df["timestamp"].min()
            )
        return df

    def pcap_reconstruct_flows(packet_df):
        if len(packet_df) == 0:
            return pd.DataFrame()
        packet_df = packet_df.copy()
        packet_df["window_idx"] = (
            packet_df["timestamp"] // WINDOW_SECONDS
        ).astype(int)

        def make_flow_key(row):
            ips = sorted(
                [row["src_ip"], row["dst_ip"]]
            )
            ports = sorted(
                [row["src_port"], row["dst_port"]]
            )
            return (
                f"{ips[0]}:{ips[1]}:"
                f"{ports[0]}:{ports[1]}:"
                f"{row['protocol']}"
            )

        packet_df["flow_key"] = packet_df.apply(
            make_flow_key, axis=1
        )
        packet_df["is_forward"] = (
            packet_df["src_ip"] <= packet_df["dst_ip"]
        )

        flows = []
        for (win_idx, fk), grp in \
                packet_df.groupby(
                    ["window_idx", "flow_key"]
                ):
            fwd = grp[grp["is_forward"]]
            bwd = grp[~grp["is_forward"]]
            ts = grp["timestamp"].values
            dur = (
                ts.max() - ts.min()
                if len(ts) > 1 else 0.0
            )
            fwd_iats = (
                np.diff(sorted(
                    fwd["timestamp"].values
                ))
                if len(fwd) > 1
                else np.array([0.0])
            )
            bwd_iats = (
                np.diff(sorted(
                    bwd["timestamp"].values
                ))
                if len(bwd) > 1
                else np.array([0.0])
            )
            flows.append({
                "window_idx": win_idx,
                "flow_key": fk,
                "total_packets": len(grp),
                "total_bytes": grp["pkt_len"].sum(),
                "fwd_packets": len(fwd),
                "bwd_packets": len(bwd),
                "fwd_bytes": (
                    fwd["pkt_len"].sum()
                    if len(fwd) > 0 else 0
                ),
                "bwd_bytes": (
                    bwd["pkt_len"].sum()
                    if len(bwd) > 0 else 0
                ),
                "duration": dur,
                "mean_fwd_iat": fwd_iats.mean(),
                "mean_bwd_iat": bwd_iats.mean(),
                "pkt_lengths": grp[
                    "pkt_len"
                ].values.tolist(),
                "dst_port": grp[
                    "dst_port"
                ].iloc[0],
                "protocol": grp[
                    "protocol"
                ].iloc[0],
                "flags": "|".join(
                    grp["flags"].unique()
                ),
                "down_up_ratio": (
                    len(bwd) / max(len(fwd), 1)
                ),
            })
        return pd.DataFrame(flows)

    def pcap_compute_features(flow_df, packet_df):
        if len(flow_df) == 0:
            return pd.DataFrame()
        all_wins = sorted(
            flow_df["window_idx"].unique()
        )
        win_pkts = packet_df.groupby(
            (packet_df["timestamp"] // WINDOW_SECONDS)
            .astype(int)
        )
        rows = []
        for win_idx in all_wins:
            wf = flow_df[
                flow_df["window_idx"] == win_idx
            ]
            try:
                wp = win_pkts.get_group(win_idx)
            except KeyError:
                wp = pd.DataFrame()

            fc = len(wf)
            tp = wf["total_packets"].sum()
            tb = wf["total_bytes"].sum()
            wpr = tp / WINDOW_SECONDS
            wbr = tb / WINDOW_SECONDS

            af = "|".join(
                wf["flags"].fillna("")
            ).upper()
            nf = max(len(wp), 1)

            all_lens = []
            for lens in wf["pkt_lengths"]:
                all_lens.extend(lens)

            tfp = wf["fwd_packets"].sum()
            tbp = wf["bwd_packets"].sum()
            tfb = wf["fwd_bytes"].sum()
            tbb = wf["bwd_bytes"].sum()
            pt = max(tfp + tbp, 1)
            bt = max(tfb + tbb, 1)

            rows.append({
                "window_idx": win_idx,
                "timestamp": win_idx * WINDOW_SECONDS,
                "flow_count": fc,
                "log_flow_count": np.log1p(fc),
                "log_total_packets": np.log1p(tp),
                "log_total_bytes": np.log1p(tb),
                "window_packet_rate": wpr,
                "window_byte_rate": wbr,
                "syn_rate": af.count("S") / nf,
                "ack_rate": af.count("A") / nf,
                "rst_rate": af.count("R") / nf,
                "fin_rate": af.count("F") / nf,
                "psh_rate": af.count("P") / nf,
                "urg_rate": af.count("U") / nf,
                "mean_pkt_len": (
                    np.mean(all_lens)
                    if all_lens else 0.0
                ),
                "std_pkt_len": (
                    np.std(all_lens)
                    if len(all_lens) > 1 else 0.0
                ),
                "log_mean_flow_duration": np.log1p(
                    wf["duration"].mean()
                ),
                "log_mean_fwd_iat": np.log1p(
                    wf["mean_fwd_iat"].mean()
                ),
                "log_mean_bwd_iat": np.log1p(
                    wf["mean_bwd_iat"].mean()
                ),
                "fwd_packet_ratio": tfp / pt,
                "bwd_packet_ratio": tbp / pt,
                "fwd_byte_ratio": tfb / bt,
                "bwd_byte_ratio": tbb / bt,
                "mean_down_up_ratio": wf[
                    "down_up_ratio"
                ].mean(),
                "log_unique_dst_ports": np.log1p(
                    wf["dst_port"].nunique()
                ),
                "unique_protocols": wf[
                    "protocol"
                ].nunique(),
                "dst_port_entropy": pcap_entropy(
                    wf["dst_port"].values
                ),
                "protocol_entropy": pcap_entropy(
                    wf["protocol"].values
                ),
            })
        return pd.DataFrame(rows)

    def pcap_add_rel_changes(window_df):
        df = window_df.copy()
        eps = 1e-8
        for feat in BASE_FEATURES:
            prev = df[feat].shift(1).fillna(0)
            df[f"{feat}_rel_change"] = (
                (df[feat] - prev)
                / (prev.abs() + eps)
            ).clip(-10, 10)
        return df

    # --- Upload widget ---
    uploaded = st.file_uploader(
        "Upload a .pcap file",
        type=["pcap", "pcapng", "cap"],
        help="Drag and drop a raw packet capture "
             "file from Wireshark, tcpdump, etc."
    )

    if uploaded is not None:
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

            # Normalize
            scaler_path = (
                DATA_DIR / "input_scaler.pkl"
            )
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            feat_matrix = feature_df[
                INPUT_FEATURES
            ].values
            feat_scaled = scaler.transform(
                feat_matrix
            )

            # Create sequences
            sequences = []
            seq_timestamps = []
            for i in range(
                len(feat_scaled) - SEQ_LEN + 1
            ):
                sequences.append(
                    feat_scaled[i:i + SEQ_LEN]
                )
                seq_timestamps.append(
                    feature_df["timestamp"].iloc[
                        i + SEQ_LEN - 1
                    ]
                )

            if len(sequences) == 0:
                st.warning(
                    f"⚠️ Need at least "
                    f"{SEQ_LEN * WINDOW_SECONDS}s "
                    f"of traffic for analysis. "
                    f"This file has {duration:.0f}s."
                )
            else:
                X_pcap = np.array(
                    sequences, dtype=np.float32
                )

                # Run inference
                with st.spinner(
                    "🧠 Running V4 World Model "
                    "inference..."
                ):
                    with torch.no_grad():
                        X_pt = torch.tensor(
                            X_pcap,
                            dtype=torch.float32,
                        )
                        z_pcap = model.encode(X_pt)
                        z_next_pcap = (
                            model.transition_step(
                                z_pcap
                            )
                        )
                        logits_pcap = (
                            model.predict_attack(
                                z_next_pcap
                            )
                        )
                        probs_pcap = torch.sigmoid(
                            logits_pcap
                        ).numpy()

                pcap_threshold = ckpt[
                    "selected_threshold"
                ]
                alerts_pcap = (
                    probs_pcap >= pcap_threshold
                )
                n_alerts_pcap = int(
                    alerts_pcap.sum()
                )
                max_prob_pcap = float(
                    probs_pcap.max()
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
                        name="Attack Prob",
                        line=dict(
                            color="#3498db",
                            width=2,
                        ),
                        marker=dict(size=4),
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
                        "Attack Prob": [
                            f"{p:.4f}"
                            for p in probs_pcap
                        ],
                        "Alert": [
                            "🚨 YES" if a else "✅ No"
                            for a in alerts_pcap
                        ],
                    })
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
            "**How it works:**\n"
            "1. Packets are parsed using Scapy\n"
            "2. Flows are reconstructed (5-tuple)\n"
            "3. 52 features are extracted per "
            "10-second window\n"
            "4. Features are z-score normalized "
            "using the training scaler\n"
            "5. V4 World Model runs inference "
            "and flags threats"
        )

        st.markdown("### Pipeline Architecture")
        st.code(
            ".pcap file\n"
            "  → Scapy packet parser\n"
            "  → Bidirectional flow reconstruction\n"
            "  → 10-second window aggregation\n"
            "  → 52-feature extraction\n"
            "  → Z-score normalization\n"
            "  → V4 World Model inference\n"
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
