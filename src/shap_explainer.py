"""
=============================================================
FEATURE ATTRIBUTION — EXPLAINABILITY LAYER
=============================================================

This module explains WHY the V3 World Model predicts
an attack for a given window. It answers:

  "Which network telemetry features are driving
   the attack prediction?"

For example:
  "Attack predicted because log_unique_dst_ports
   spiked (+2.3σ) and log_flow_count increased
   (+1.1σ) above normal."

Uses Captum's Integrated Gradients (the gold standard
for neural network attribution, no TensorFlow needed).

Also computes SHAP-equivalent feature importance via
a manual permutation approach for robust results.

Output is structured for consumption by:
- The Streamlit dashboard
- The MITRE ATT&CK mapper
- Human analysts

=============================================================
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from captum.attr import (
    IntegratedGradients,
    FeatureAblation,
)


# =============================================================
# FEATURE NAMES (from input_feature_names.txt)
# =============================================================

INPUT_FEATURES = [
    "flow_count",
    "log_flow_count",
    "log_total_packets",
    "log_total_bytes",
    "window_packet_rate",
    "window_byte_rate",
    "syn_rate",
    "ack_rate",
    "rst_rate",
    "fin_rate",
    "psh_rate",
    "urg_rate",
    "mean_pkt_len",
    "std_pkt_len",
    "log_mean_flow_duration",
    "log_mean_fwd_iat",
    "log_mean_bwd_iat",
    "fwd_packet_ratio",
    "bwd_packet_ratio",
    "fwd_byte_ratio",
    "bwd_byte_ratio",
    "mean_down_up_ratio",
    "log_unique_dst_ports",
    "unique_protocols",
    "dst_port_entropy",
    "protocol_entropy",
    "flow_count_rel_change",
    "log_flow_count_rel_change",
    "log_total_packets_rel_change",
    "log_total_bytes_rel_change",
    "window_packet_rate_rel_change",
    "window_byte_rate_rel_change",
    "syn_rate_rel_change",
    "ack_rate_rel_change",
    "rst_rate_rel_change",
    "fin_rate_rel_change",
    "psh_rate_rel_change",
    "urg_rate_rel_change",
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
# V3 MODEL DEFINITION
# =============================================================

class LatentWorldModelV3(nn.Module):

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
        self.encoder_norm = nn.LayerNorm(
            hidden_size
        )
        self.encoder_fc = nn.Linear(
            hidden_size, latent_size
        )

        self.transition_delta = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size)
        )

        self.decoder_fc1 = nn.Linear(
            latent_size, 64
        )
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(
            64, output_size
        )

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(
            last_hidden
        )
        return self.encoder_fc(last_hidden)

    def transition_step(self, z):
        return z + self.transition_delta(z)

    def decode(self, z):
        h = self.decoder_fc1(z)
        h = self.decoder_activation(h)
        return self.decoder_fc2(h)

    def predict_attack(self, z):
        logits = self.attack_head(z)
        return logits.squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        z_next = self.transition_step(z)
        state = self.decode(z_next)
        attack = self.predict_attack(z_next)
        return state, z_next, attack, z

    def attack_probability(self, x):
        """
        End-to-end: input window → attack probability.
        This is the function the explainer attributes.
        """
        z = self.encode(x)
        z_next = self.transition_step(z)
        logit = self.predict_attack(z_next)
        return torch.sigmoid(logit).unsqueeze(-1)


# =============================================================
# WRAPPER FOR CAPTUM
# =============================================================

class AttackProbWrapper(nn.Module):
    """
    Thin wrapper that Captum can call.
    Input: (batch, seq_len, 52)
    Output: (batch, 1) attack probability

    Built from the encode -> transition_step -> predict_attack
    chain directly (rather than a model-specific convenience
    method) so it works unmodified against both the V3 and V4
    model classes.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z = self.model.encode(x)
        z_next = self.model.transition_step(z)
        logit = self.model.predict_attack(z_next)
        return torch.sigmoid(logit).unsqueeze(-1)


# =============================================================
# EXPLAINER CLASS
# =============================================================

class FeatureExplainer:
    """
    Explains V3 World Model predictions using
    Integrated Gradients (Captum).

    Provides:
    - Per-window feature attributions
    - Global feature importance ranking
    - Human-readable explanations

    Designed for consumption by:
    - Streamlit dashboard
    - MITRE ATT&CK mapper enrichment
    """

    def __init__(
        self,
        model,
        baseline_data: np.ndarray,
        device: str = "cpu",
        n_baseline: int = 50,
    ):
        self.model = model
        self.model.eval()
        self.device = torch.device(device)
        self.wrapper = AttackProbWrapper(model)
        self.wrapper.eval()

        # Baseline: mean of benign windows
        idx = np.random.choice(
            baseline_data.shape[0],
            size=min(
                n_baseline,
                baseline_data.shape[0]
            ),
            replace=False,
        )
        # Use mean as baseline (standard practice)
        self.baseline = torch.tensor(
            baseline_data[idx].mean(axis=0,
                                    keepdims=True),
            dtype=torch.float32,
        ).to(self.device)

        # Integrated Gradients — the gold standard
        self.ig = IntegratedGradients(self.wrapper)

    def explain_windows(
        self,
        X: np.ndarray,
        max_windows: int = 502,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Compute attributions for a batch of windows.

        Args:
            X: shape (N, seq_len, 52)

        Returns:
            attributions: shape (N, seq_len, 52)
        """
        n = min(X.shape[0], max_windows)
        all_attrs = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.tensor(
                X[start:end],
                dtype=torch.float32,
            ).to(self.device)

            # Expand baseline to match batch size
            baseline = self.baseline.expand_as(batch)

            attrs = self.ig.attribute(
                batch,
                baselines=baseline,
                n_steps=50,
                internal_batch_size=batch_size,
            )

            all_attrs.append(
                attrs.detach().cpu().numpy()
            )

            if (end % 100 == 0 or end == n):
                print(f"  Processed {end}/{n} windows")

        return np.concatenate(all_attrs, axis=0)

    def get_feature_importance(
        self, attributions: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute global feature importance.

        Takes the LAST timestep (matching MITRE mapper)
        and averages |attribution| across all windows.
        """
        # Use last timestep: shape (N, 52)
        last_step_attr = attributions[:, -1, :]

        mean_abs = np.abs(last_step_attr).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": INPUT_FEATURES,
            "mean_abs_attribution": mean_abs,
        }).sort_values(
            "mean_abs_attribution", ascending=False
        ).reset_index(drop=True)

        importance_df["rank"] = range(
            1, len(importance_df) + 1
        )

        return importance_df

    def explain_single_window(
        self,
        x_window: np.ndarray,
        attr_vals: np.ndarray,
        window_idx: int,
        top_n: int = 5,
    ) -> dict:
        """
        Generate a structured explanation for a
        single window.

        Args:
            x_window: shape (seq_len, 52)
            attr_vals: shape (seq_len, 52)
            window_idx: index
            top_n: number of top features

        Returns:
            dict with explanation details
        """
        # Use last timestep
        last_attr = attr_vals[-1]    # (52,)
        last_vals = x_window[-1]     # (52,)

        # Sort by absolute attribution
        abs_attr = np.abs(last_attr)
        top_idx = np.argsort(abs_attr)[::-1][:top_n]

        top_features = []
        for idx in top_idx:
            feat_name = INPUT_FEATURES[idx]
            attr_val = float(last_attr[idx])
            feat_val = float(last_vals[idx])

            direction = (
                "↑ increases" if attr_val > 0
                else "↓ decreases"
            )

            top_features.append({
                "feature": feat_name,
                "attribution": attr_val,
                "feature_value": feat_val,
                "direction": direction,
                "contribution": (
                    f"{feat_name} = {feat_val:.2f}σ "
                    f"({direction} attack prob "
                    f"by {abs(attr_val):.4f})"
                ),
            })

        # Generate natural language explanation
        drivers = []
        for f in top_features[:3]:
            if f["attribution"] > 0:
                drivers.append(
                    f"{f['feature']} "
                    f"({f['feature_value']:+.2f}σ)"
                )

        explanation = (
            f"Attack prediction driven by: "
            f"{', '.join(drivers)}"
            if drivers
            else "No strong individual feature drivers."
        )

        return {
            "window_index": window_idx,
            "top_features": top_features,
            "explanation": explanation,
            "total_attribution_magnitude": float(
                abs_attr.sum()
            ),
        }


# =============================================================
# MAIN — RUN EXPLAINABILITY ANALYSIS
# =============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute Integrated Gradients feature "
            "attribution for a World Model checkpoint."
        )
    )
    parser.add_argument(
        "--version",
        choices=["v3", "v4"],
        default="v4",
        help=(
            "Which World Model version to explain "
            "(default: v4, the canonical model)."
        ),
    )
    args = parser.parse_args()
    version = args.version

    DATA_DIR = Path(
        "data/processed/temporal_benchmark_v2"
    )
    MODEL_DIR = Path(
        f"models/latent_world_model_{version}"
    )
    OUTPUT_DIR = MODEL_DIR / "shap_analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use CPU for gradient computation (stable)
    DEVICE = torch.device("cpu")

    print("=" * 80)
    print("FEATURE ATTRIBUTION — INTEGRATED GRADIENTS")
    print("=" * 80)
    print(f"Model version: {version.upper()}")
    print(f"Device: {DEVICE}")

    # Load model
    print(f"\n--- Loading {version.upper()} model ---")
    if version == "v4":
        # Reuse the canonical V4 model/checkpoint loader instead of
        # duplicating the class definition -- see
        # src/pcap_ingest_v4.py.
        from pcap_ingest_v4 import load_v4_model, CHECKPOINT_PATH

        model, checkpoint = load_v4_model(CHECKPOINT_PATH)
        model = model.to(DEVICE)
    else:
        checkpoint = torch.load(
            MODEL_DIR / "latent_world_model_v3.pt",
            map_location=DEVICE,
            weights_only=False,
        )

        model = LatentWorldModelV3(
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
        model = model.to(DEVICE)
    model.eval()

    threshold = checkpoint["selected_threshold"]

    # Load data
    print("--- Loading data ---")
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    test_meta = pd.read_csv(
        DATA_DIR / "test_metadata.csv"
    )
    train_meta = pd.read_csv(
        DATA_DIR / "train_metadata.csv"
    )

    print(f"Train: {X_train.shape}")
    print(f"Test:  {X_test.shape}")

    # Use benign training windows as baseline
    benign_mask = (
        train_meta["attack_present"].values == 0
    )
    X_baseline = X_train[benign_mask]
    print(f"Baseline (benign): {X_baseline.shape}")

    # Get attack probabilities for test set
    print("\n--- Computing attack probabilities ---")
    with torch.no_grad():
        X_test_t = torch.tensor(
            X_test, dtype=torch.float32
        ).to(DEVICE)
        z_test = model.encode(X_test_t)
        z_test_next = model.transition_step(z_test)
        test_probs = (
            torch.sigmoid(model.predict_attack(z_test_next))
            .cpu().numpy()
        )

    # =========================================================
    # COMPUTE ATTRIBUTIONS
    # =========================================================

    print("\n--- Computing Integrated Gradients ---")
    print("(This takes ~1-2 minutes on CPU...)")

    explainer = FeatureExplainer(
        model=model,
        baseline_data=X_baseline,
        device=str(DEVICE),
        n_baseline=50,
    )

    attributions = explainer.explain_windows(
        X_test, max_windows=X_test.shape[0],
    )

    print(f"Attribution shape: {attributions.shape}")

    # =========================================================
    # ANALYSIS 1: Global Feature Importance
    # =========================================================

    print("\n" + "=" * 80)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 80)

    importance_df = explainer.get_feature_importance(
        attributions
    )

    print(f"\nTop 15 most important features:")
    print(
        f"{'Rank':<6} {'Feature':<45} "
        f"{'|Attr|':>10}"
    )
    print("-" * 63)
    for _, row in importance_df.head(15).iterrows():
        print(
            f"{int(row['rank']):<6} "
            f"{row['feature']:<45} "
            f"{row['mean_abs_attribution']:>10.6f}"
        )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    # =========================================================
    # ANALYSIS 2: Attack vs Benign Attribution
    # =========================================================

    print("\n" + "=" * 80)
    print("ATTACK vs BENIGN ATTRIBUTION")
    print("=" * 80)

    attack_mask = (
        test_meta["attack_present"].values == 1
    )
    benign_mask_test = ~attack_mask

    n_test = min(
        attributions.shape[0], len(attack_mask)
    )
    attack_mask = attack_mask[:n_test]
    benign_mask_test = benign_mask_test[:n_test]

    # Mean attribution for attack windows (last ts)
    last_attr = attributions[:n_test, -1, :]

    if attack_mask.sum() > 0:
        atk_mean = last_attr[attack_mask].mean(
            axis=0
        )
        ben_mean = last_attr[benign_mask_test].mean(
            axis=0
        )

        print(
            f"\nTop features DRIVING attack "
            f"predictions:"
        )
        atk_sort = np.argsort(atk_mean)[::-1]
        for i, idx in enumerate(atk_sort[:10]):
            feat = INPUT_FEATURES[idx]
            print(
                f"  {i+1}. {feat:<45} "
                f"attr={atk_mean[idx]:+.6f}"
            )

        print(
            f"\nTop features SUPPRESSING attack "
            f"predictions (benign windows):"
        )
        ben_sort = np.argsort(ben_mean)
        for i, idx in enumerate(ben_sort[:5]):
            feat = INPUT_FEATURES[idx]
            print(
                f"  {i+1}. {feat:<45} "
                f"attr={ben_mean[idx]:+.6f}"
            )

    # =========================================================
    # ANALYSIS 3: Per-Window Explanations
    # =========================================================

    print("\n" + "=" * 80)
    print("SAMPLE WINDOW EXPLANATIONS")
    print("=" * 80)

    top_prob_idx = np.argsort(test_probs)[::-1][:5]

    for i, widx in enumerate(top_prob_idx):
        if widx >= n_test:
            continue
        prob = test_probs[widx]
        ts = test_meta["timestamp"].iloc[widx]

        explanation = explainer.explain_single_window(
            x_window=X_test[widx],
            attr_vals=attributions[widx],
            window_idx=int(widx),
        )

        print(
            f"\n  Window {widx} | "
            f"Prob={prob:.3f} | {ts}"
        )
        print(f"  {explanation['explanation']}")
        for f in explanation["top_features"][:3]:
            print(f"    {f['contribution']}")

    # =========================================================
    # SAVE PER-WINDOW EXPLANATIONS
    # =========================================================

    all_explanations = []
    for widx in range(n_test):
        exp = explainer.explain_single_window(
            x_window=X_test[widx],
            attr_vals=attributions[widx],
            window_idx=widx,
        )
        top_feat_str = "; ".join(
            f['feature']
            for f in exp["top_features"][:3]
        )
        all_explanations.append({
            "window_index": widx,
            "timestamp": test_meta["timestamp"]
                .iloc[widx],
            "attack_probability": float(
                test_probs[widx]
            ),
            "ground_truth": int(
                test_meta["attack_present"]
                .iloc[widx]
            ),
            "top_3_features": top_feat_str,
            "explanation": exp["explanation"],
            "total_attribution": exp[
                "total_attribution_magnitude"
            ],
        })

    exp_df = pd.DataFrame(all_explanations)
    exp_df.to_csv(
        OUTPUT_DIR / "window_explanations.csv",
        index=False,
    )

    # =========================================================
    # VIZ 1: Global Feature Importance Bar Chart
    # =========================================================

    print("\n--- Generating visualizations ---")

    fig, ax = plt.subplots(figsize=(12, 8))

    top_n = 20
    top_df = importance_df.head(top_n).iloc[::-1]

    colors = []
    for feat in top_df["feature"]:
        if "_rel_change" in feat:
            colors.append("#e74c3c")
        else:
            colors.append("#3498db")

    ax.barh(
        top_df["feature"],
        top_df["mean_abs_attribution"],
        color=colors, edgecolor="white",
    )
    ax.set_xlabel("Mean |Attribution|")
    ax.set_title(
        "Top 20 Features by Integrated Gradients "
        "Attribution",
        fontsize=14, fontweight="bold",
    )

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="#3498db",
                  label="Base feature"),
            Patch(color="#e74c3c",
                  label="Relative change"),
        ],
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "01_feature_importance.png",
        dpi=150,
    )
    plt.close(fig)
    print("  Saved: 01_feature_importance.png")

    # =========================================================
    # VIZ 2: Attack vs Benign comparison
    # =========================================================

    if attack_mask.sum() > 0:
        fig, axes = plt.subplots(
            1, 2, figsize=(16, 8)
        )

        top_feat_idx = np.argsort(
            np.abs(atk_mean)
        )[::-1][:15]

        feat_names = [
            INPUT_FEATURES[i] for i in top_feat_idx
        ]

        # Attack windows
        atk_vals = [
            atk_mean[i] for i in top_feat_idx
        ]
        colors_atk = [
            "#e74c3c" if v > 0 else "#3498db"
            for v in atk_vals
        ]

        axes[0].barh(
            feat_names[::-1], atk_vals[::-1],
            color=colors_atk[::-1],
            edgecolor="white",
        )
        axes[0].set_title(
            "Attack Windows — Mean Attribution",
            fontsize=12, fontweight="bold",
        )
        axes[0].axvline(
            x=0, color="black", linewidth=0.5
        )

        # Benign windows
        ben_vals = [
            ben_mean[i] for i in top_feat_idx
        ]
        colors_ben = [
            "#e74c3c" if v > 0 else "#3498db"
            for v in ben_vals
        ]

        axes[1].barh(
            feat_names[::-1], ben_vals[::-1],
            color=colors_ben[::-1],
            edgecolor="white",
        )
        axes[1].set_title(
            "Benign Windows — Mean Attribution",
            fontsize=12, fontweight="bold",
        )
        axes[1].axvline(
            x=0, color="black", linewidth=0.5
        )

        fig.suptitle(
            "Feature Attribution: Attack vs Benign",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / "02_attack_vs_benign.png",
            dpi=150,
        )
        plt.close(fig)
        print("  Saved: 02_attack_vs_benign.png")

    # =========================================================
    # VIZ 3: Attribution over time for top features
    # =========================================================

    fig, axes = plt.subplots(
        4, 1, figsize=(18, 14),
        gridspec_kw={"height_ratios": [1, 2, 2, 2]},
    )

    gt = test_meta["attack_present"].values[:n_test]

    # Panel 1: Ground truth
    axes[0].fill_between(
        range(n_test), gt,
        alpha=0.4, color="crimson",
        label="Attack present",
    )
    axes[0].set_ylabel("Ground Truth")
    axes[0].legend(loc="upper right")
    axes[0].set_title(
        "Feature Attribution Over Time "
        "(Integrated Gradients)",
        fontsize=14, fontweight="bold",
    )

    # Top 3 features by importance
    top3_feats = importance_df.head(3)[
        "feature"
    ].values

    for panel_i, feat_name in enumerate(top3_feats):
        ax = axes[panel_i + 1]
        feat_idx = INPUT_FEATURES.index(feat_name)

        # Attribution for this feature over time
        feat_attr = last_attr[:, feat_idx]

        # Color positive (attack-driving) red,
        # negative blue
        pos_mask = feat_attr > 0
        neg_mask = feat_attr <= 0

        ax.bar(
            np.where(pos_mask)[0],
            feat_attr[pos_mask],
            color="#e74c3c", alpha=0.7, width=1,
        )
        ax.bar(
            np.where(neg_mask)[0],
            feat_attr[neg_mask],
            color="#3498db", alpha=0.7, width=1,
        )

        ax.axhline(
            y=0, color="black", linewidth=0.3
        )
        ax.set_ylabel(f"Attr\n{feat_name}")

        # Mark attack region
        for i in range(n_test):
            if gt[i] == 1:
                ax.axvspan(
                    i-0.5, i+0.5,
                    alpha=0.05, color="crimson",
                )

    ts = pd.to_datetime(test_meta["timestamp"])
    axes[-1].set_xlabel(
        f"Window index ({ts.iloc[0].strftime('%H:%M')}"
        f" to {ts.iloc[-1].strftime('%H:%M')})"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "03_attribution_over_time.png",
        dpi=150,
    )
    plt.close(fig)
    print("  Saved: 03_attribution_over_time.png")

    # =========================================================
    # VIZ 4: Heatmap of top features × windows
    # =========================================================

    print("  Generating heatmap...")

    top10_feats = importance_df.head(10)[
        "feature"
    ].values
    top10_idx = [
        INPUT_FEATURES.index(f) for f in top10_feats
    ]

    heatmap_data = last_attr[:, top10_idx].T

    fig, axes = plt.subplots(
        2, 1, figsize=(18, 8),
        gridspec_kw={"height_ratios": [1, 4]},
    )

    # Panel 1: Ground truth + probability
    axes[0].fill_between(
        range(n_test), gt,
        alpha=0.3, color="crimson",
        label="Ground truth",
    )
    axes[0].plot(
        range(n_test), test_probs[:n_test],
        color="steelblue", linewidth=0.8,
        label="Attack probability",
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Prob")
    axes[0].set_title(
        "Feature Attribution Heatmap "
        "(Top 10 Features)",
        fontsize=14, fontweight="bold",
    )

    # Panel 2: Heatmap
    vmax = np.abs(heatmap_data).max()
    im = axes[1].imshow(
        heatmap_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    axes[1].set_yticks(range(len(top10_feats)))
    axes[1].set_yticklabels(top10_feats, fontsize=9)
    axes[1].set_xlabel("Window index")

    fig.colorbar(
        im, ax=axes[1],
        label="Attribution (red=attack driver)",
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "04_attribution_heatmap.png",
        dpi=150,
    )
    plt.close(fig)
    print("  Saved: 04_attribution_heatmap.png")

    # =========================================================
    # SAVE ATTRIBUTIONS
    # =========================================================

    np.save(
        OUTPUT_DIR / "attributions.npy",
        attributions,
    )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print("\n" + "=" * 80)
    print("FEATURE ATTRIBUTION — SUMMARY")
    print("=" * 80)

    print(f"\nTop 5 attack-driving features:")
    for _, row in importance_df.head(5).iterrows():
        print(
            f"  {int(row['rank'])}. "
            f"{row['feature']}"
            f" (|attr|="
            f"{row['mean_abs_attribution']:.6f})"
        )

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)
