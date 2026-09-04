"""
=============================================================
RAW NETWORK TRAFFIC INGESTION — PCAP PROCESSING
=============================================================

This module reads raw .pcap (packet capture) files and
converts them into the exact 52-feature, 10-second window
format that our V3 World Model expects.

This is the bridge between real-world network traffic
and the AI model.

Pipeline:
  .pcap file
    → Scapy packet parser
    → Flow reconstruction (5-tuple grouping)
    → 10-second window aggregation
    → 52-feature extraction (matching training schema)
    → Z-score normalization (using saved scaler)
    → Ready for V3 model inference

Usage:
  python src/pcap_ingest.py data/sample_pcap/sample_http.pcap
  python src/pcap_ingest.py data/sample_pcap/  # entire folder

=============================================================
"""

import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scapy.all import (
    rdpcap, IP, TCP, UDP,
    conf as scapy_conf,
)

# Suppress Scapy warnings
scapy_conf.verb = 0


# =============================================================
# CONSTANTS
# =============================================================

WINDOW_SECONDS = 10   # Aggregate packets into 10s windows
SEQ_LEN = 10          # Model expects sequences of 10 windows
N_FEATURES = 52       # 26 base + 26 relative change

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
                dropout if num_layers > 1 else 0.0
            )
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(
            hidden_size, latent_size
        )
        self.transition_delta = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size)
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
        return z + self.transition_delta(z)

    def predict_attack(self, z):
        return self.attack_head(z).squeeze(-1)

    def forward(self, x):
        z = self.encode(x)
        z_next = self.transition_step(z)
        attack = self.predict_attack(z_next)
        return z_next, attack


# =============================================================
# STEP 1: PARSE PCAP → RAW PACKET TABLE
# =============================================================

def parse_pcap(pcap_path: str) -> pd.DataFrame:
    """
    Read a .pcap file and extract per-packet metadata.

    Returns a DataFrame with columns:
      timestamp, src_ip, dst_ip, src_port, dst_port,
      protocol, pkt_len, flags (TCP flags)
    """
    print(f"  Reading PCAP: {pcap_path}")
    packets = rdpcap(str(pcap_path))
    print(f"  Total packets: {len(packets)}")

    rows = []
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip = pkt[IP]
        proto = ip.proto  # 6=TCP, 17=UDP

        src_port = 0
        dst_port = 0
        flags = ""

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
            "protocol": proto,
            "pkt_len": len(pkt),
            "flags": flags,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  WARNING: No IP packets found!")
        return df

    # Normalize timestamps to start at 0
    df["timestamp"] = (
        df["timestamp"] - df["timestamp"].min()
    )

    print(f"  IP packets parsed: {len(df)}")
    print(
        f"  Duration: "
        f"{df['timestamp'].max():.1f} seconds"
    )

    return df


# =============================================================
# STEP 2: RECONSTRUCT FLOWS (5-tuple grouping)
# =============================================================

def reconstruct_flows(
    packet_df: pd.DataFrame,
    window_sec: float = WINDOW_SECONDS,
) -> pd.DataFrame:
    """
    Group packets into bidirectional flows within
    each time window.

    A flow = unique (src_ip, dst_ip, src_port,
    dst_port, protocol) within a window.

    Returns a DataFrame of per-flow statistics.
    """
    if len(packet_df) == 0:
        return pd.DataFrame()

    # Assign window index
    packet_df = packet_df.copy()
    packet_df["window_idx"] = (
        packet_df["timestamp"] // window_sec
    ).astype(int)

    # Create flow key (bidirectional)
    def make_flow_key(row):
        ips = sorted([row["src_ip"], row["dst_ip"]])
        ports = sorted([
            row["src_port"], row["dst_port"]
        ])
        return (
            f"{ips[0]}:{ips[1]}:"
            f"{ports[0]}:{ports[1]}:"
            f"{row['protocol']}"
        )

    packet_df["flow_key"] = packet_df.apply(
        make_flow_key, axis=1
    )

    # Determine forward/backward direction
    # (lower IP is considered "source")
    def is_forward(row):
        return row["src_ip"] <= row["dst_ip"]

    packet_df["is_forward"] = packet_df.apply(
        is_forward, axis=1
    )

    flows = []
    for (win_idx, flow_key), group in \
            packet_df.groupby(
                ["window_idx", "flow_key"]
            ):

        fwd = group[group["is_forward"]]
        bwd = group[~group["is_forward"]]

        timestamps = group["timestamp"].values
        duration = (
            timestamps.max() - timestamps.min()
            if len(timestamps) > 1 else 0.0
        )

        # Inter-arrival times
        fwd_iats = np.diff(
            sorted(fwd["timestamp"].values)
        ) if len(fwd) > 1 else np.array([0.0])
        bwd_iats = np.diff(
            sorted(bwd["timestamp"].values)
        ) if len(bwd) > 1 else np.array([0.0])

        flows.append({
            "window_idx": win_idx,
            "flow_key": flow_key,
            "total_packets": len(group),
            "total_bytes": group["pkt_len"].sum(),
            "fwd_packets": len(fwd),
            "bwd_packets": len(bwd),
            "fwd_bytes": fwd["pkt_len"].sum()
                if len(fwd) > 0 else 0,
            "bwd_bytes": bwd["pkt_len"].sum()
                if len(bwd) > 0 else 0,
            "duration": duration,
            "mean_fwd_iat": fwd_iats.mean()
                if len(fwd_iats) > 0 else 0.0,
            "mean_bwd_iat": bwd_iats.mean()
                if len(bwd_iats) > 0 else 0.0,
            "pkt_lengths": group[
                "pkt_len"
            ].values.tolist(),
            "dst_port": group[
                "dst_port"
            ].iloc[0],
            "protocol": group[
                "protocol"
            ].iloc[0],
            "flags": "|".join(
                group["flags"].unique()
            ),
            "down_up_ratio": (
                len(bwd) / max(len(fwd), 1)
            ),
        })

    return pd.DataFrame(flows)


# =============================================================
# STEP 3: COMPUTE 26 BASE FEATURES PER WINDOW
# =============================================================

def entropy(values):
    """Shannon entropy of a distribution."""
    if len(values) == 0:
        return 0.0
    counts = pd.Series(values).value_counts()
    probs = counts / counts.sum()
    return float(
        -np.sum(probs * np.log2(probs + 1e-10))
    )


def compute_window_features(
    flow_df: pd.DataFrame,
    packet_df: pd.DataFrame,
    window_sec: float = WINDOW_SECONDS,
) -> pd.DataFrame:
    """
    Compute the 26 base features for each window
    from the flow and packet data.

    Matches the exact feature schema used during
    training.
    """
    if len(flow_df) == 0:
        return pd.DataFrame()

    all_windows = sorted(
        flow_df["window_idx"].unique()
    )
    window_packets = packet_df.groupby(
        (packet_df["timestamp"] // window_sec)
        .astype(int)
    )

    rows = []
    for win_idx in all_windows:
        wf = flow_df[
            flow_df["window_idx"] == win_idx
        ]

        # Get packets for this window
        try:
            wp = window_packets.get_group(win_idx)
        except KeyError:
            wp = pd.DataFrame()

        # --- Core counts ---
        flow_count = len(wf)
        total_packets = wf["total_packets"].sum()
        total_bytes = wf["total_bytes"].sum()

        # --- Rates ---
        window_packet_rate = total_packets / window_sec
        window_byte_rate = total_bytes / window_sec

        # --- TCP flag rates ---
        all_flags = "|".join(
            wf["flags"].fillna("")
        ).upper()
        n_flags = max(len(wp), 1)

        syn_count = all_flags.count("S")
        ack_count = all_flags.count("A")
        rst_count = all_flags.count("R")
        fin_count = all_flags.count("F")
        psh_count = all_flags.count("P")
        urg_count = all_flags.count("U")

        syn_rate = syn_count / n_flags
        ack_rate = ack_count / n_flags
        rst_rate = rst_count / n_flags
        fin_rate = fin_count / n_flags
        psh_rate = psh_count / n_flags
        urg_rate = urg_count / n_flags

        # --- Packet lengths ---
        all_pkt_lens = []
        for lens in wf["pkt_lengths"]:
            all_pkt_lens.extend(lens)

        mean_pkt_len = (
            np.mean(all_pkt_lens)
            if all_pkt_lens else 0.0
        )
        std_pkt_len = (
            np.std(all_pkt_lens)
            if len(all_pkt_lens) > 1 else 0.0
        )

        # --- Duration & IAT ---
        mean_duration = wf["duration"].mean()
        mean_fwd_iat = wf["mean_fwd_iat"].mean()
        mean_bwd_iat = wf["mean_bwd_iat"].mean()

        # --- Directional ratios ---
        total_fwd_pkts = wf["fwd_packets"].sum()
        total_bwd_pkts = wf["bwd_packets"].sum()
        total_fwd_bytes = wf["fwd_bytes"].sum()
        total_bwd_bytes = wf["bwd_bytes"].sum()

        pkt_total = max(
            total_fwd_pkts + total_bwd_pkts, 1
        )
        byte_total = max(
            total_fwd_bytes + total_bwd_bytes, 1
        )

        fwd_packet_ratio = total_fwd_pkts / pkt_total
        bwd_packet_ratio = total_bwd_pkts / pkt_total
        fwd_byte_ratio = total_fwd_bytes / byte_total
        bwd_byte_ratio = total_bwd_bytes / byte_total
        mean_down_up_ratio = wf[
            "down_up_ratio"
        ].mean()

        # --- Diversity ---
        unique_dst_ports = wf[
            "dst_port"
        ].nunique()
        unique_protocols = wf[
            "protocol"
        ].nunique()

        dst_port_ent = entropy(
            wf["dst_port"].values
        )
        protocol_ent = entropy(
            wf["protocol"].values
        )

        # --- Log transforms ---
        log_flow_count = np.log1p(flow_count)
        log_total_packets = np.log1p(total_packets)
        log_total_bytes = np.log1p(total_bytes)
        log_mean_flow_duration = np.log1p(
            mean_duration
        )
        log_mean_fwd_iat = np.log1p(mean_fwd_iat)
        log_mean_bwd_iat = np.log1p(mean_bwd_iat)
        log_unique_dst_ports = np.log1p(
            unique_dst_ports
        )

        rows.append({
            "window_idx": win_idx,
            "timestamp": win_idx * window_sec,
            # 26 base features (exact order)
            "flow_count": flow_count,
            "log_flow_count": log_flow_count,
            "log_total_packets": log_total_packets,
            "log_total_bytes": log_total_bytes,
            "window_packet_rate": window_packet_rate,
            "window_byte_rate": window_byte_rate,
            "syn_rate": syn_rate,
            "ack_rate": ack_rate,
            "rst_rate": rst_rate,
            "fin_rate": fin_rate,
            "psh_rate": psh_rate,
            "urg_rate": urg_rate,
            "mean_pkt_len": mean_pkt_len,
            "std_pkt_len": std_pkt_len,
            "log_mean_flow_duration":
                log_mean_flow_duration,
            "log_mean_fwd_iat": log_mean_fwd_iat,
            "log_mean_bwd_iat": log_mean_bwd_iat,
            "fwd_packet_ratio": fwd_packet_ratio,
            "bwd_packet_ratio": bwd_packet_ratio,
            "fwd_byte_ratio": fwd_byte_ratio,
            "bwd_byte_ratio": bwd_byte_ratio,
            "mean_down_up_ratio":
                mean_down_up_ratio,
            "log_unique_dst_ports":
                log_unique_dst_ports,
            "unique_protocols": unique_protocols,
            "dst_port_entropy": dst_port_ent,
            "protocol_entropy": protocol_ent,
        })

    return pd.DataFrame(rows)


# =============================================================
# STEP 4: ADD RELATIVE CHANGE FEATURES
# =============================================================

BASE_FEATURES = INPUT_FEATURES[:26]

def add_relative_changes(
    window_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add 26 relative change features
    (matching training schema).

    rel_change = (current - previous) / (|previous| + ε)
    """
    df = window_df.copy()
    eps = 1e-8

    for feat in BASE_FEATURES:
        prev = df[feat].shift(1).fillna(0)
        df[f"{feat}_rel_change"] = (
            (df[feat] - prev)
            / (prev.abs() + eps)
        ).clip(-10, 10)  # Clip extremes

    return df


# =============================================================
# STEP 5: NORMALIZE & WINDOW INTO SEQUENCES
# =============================================================

def normalize_and_window(
    feature_df: pd.DataFrame,
    scaler_path: str,
    seq_len: int = SEQ_LEN,
) -> tuple:
    """
    Apply the saved z-score scaler from training,
    then create sliding windows of seq_len.

    Returns:
        X: np.ndarray of shape (N, seq_len, 52)
        timestamps: list of window timestamps
    """
    # Load the saved scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Extract the 52 features in order
    feature_matrix = feature_df[
        INPUT_FEATURES
    ].values

    # Apply z-score normalization
    feature_scaled = scaler.transform(feature_matrix)

    # Create sliding windows
    windows = []
    timestamps = []

    for i in range(len(feature_scaled) - seq_len + 1):
        windows.append(
            feature_scaled[i:i + seq_len]
        )
        timestamps.append(
            feature_df["timestamp"].iloc[
                i + seq_len - 1
            ]
        )

    if len(windows) == 0:
        return np.array([]), []

    X = np.array(windows, dtype=np.float32)
    return X, timestamps


# =============================================================
# STEP 6: RUN V3 MODEL INFERENCE
# =============================================================

def run_inference(
    X: np.ndarray,
    model_path: str,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Run the V3 World Model on the windowed data.

    Returns DataFrame with predictions.
    """
    # Load model
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
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
    model.eval()

    # Inference
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        z = model.encode(X_t)
        z_next = model.transition_step(z)
        logits = model.predict_attack(z_next)
        probs = torch.sigmoid(logits).numpy()

    # Classify
    alerts = probs >= threshold

    return probs, alerts, model


# =============================================================
# FULL PIPELINE
# =============================================================

def process_pcap(
    pcap_path: str,
    scaler_path: str,
    model_path: str,
    output_dir: str,
) -> dict:
    """
    Full end-to-end pipeline:
      .pcap → features → normalize → model → alerts

    Returns a summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcap_name = Path(pcap_path).stem

    print(f"\n{'='*60}")
    print(f"PCAP INGESTION: {pcap_name}")
    print(f"{'='*60}")

    # Step 1: Parse packets
    print("\n[1/6] Parsing PCAP packets...")
    packet_df = parse_pcap(pcap_path)
    if len(packet_df) == 0:
        print("  ERROR: No packets found!")
        return {"status": "error",
                "reason": "no packets"}

    # Step 2: Reconstruct flows
    print("[2/6] Reconstructing network flows...")
    flow_df = reconstruct_flows(packet_df)
    print(f"  Flows reconstructed: {len(flow_df)}")
    n_windows = flow_df["window_idx"].nunique()
    print(f"  Time windows: {n_windows}")

    # Step 3: Compute 26 base features
    print("[3/6] Computing base features...")
    feature_df = compute_window_features(
        flow_df, packet_df,
    )
    print(f"  Feature windows: {len(feature_df)}")

    # Step 4: Add relative changes → 52 features
    print("[4/6] Adding relative change features...")
    feature_df = add_relative_changes(feature_df)
    print(f"  Total features: {len(INPUT_FEATURES)}")

    # Save raw features
    feature_df.to_csv(
        output_dir / f"{pcap_name}_features.csv",
        index=False,
    )

    # Step 5: Normalize and window
    print("[5/6] Normalizing & creating sequences...")
    X, timestamps = normalize_and_window(
        feature_df, scaler_path,
    )

    if len(X) == 0:
        print(
            f"  WARNING: Not enough windows for "
            f"seq_len={SEQ_LEN}. Need at least "
            f"{SEQ_LEN} windows "
            f"({SEQ_LEN * WINDOW_SECONDS}s of "
            f"traffic)."
        )
        return {"status": "insufficient_data",
                "windows": n_windows}

    print(f"  Sequences created: {X.shape}")

    # Step 6: Model inference
    print("[6/6] Running V3 model inference...")
    probs, alerts, model = run_inference(
        X, model_path,
    )

    n_alerts = int(alerts.sum())
    max_prob = float(probs.max())

    print(f"\n{'='*60}")
    print(f"RESULTS: {pcap_name}")
    print(f"{'='*60}")
    print(f"  Total sequences analyzed: {len(probs)}")
    print(f"  Alerts triggered: {n_alerts}")
    print(f"  Max attack probability: {max_prob:.4f}")

    if n_alerts > 0:
        print(f"\n  ⚠️  THREAT DETECTED!")
        alert_idx = np.where(alerts)[0]
        for i in alert_idx[:5]:
            print(
                f"    Window {i}: "
                f"prob={probs[i]:.4f} "
                f"t={timestamps[i]:.1f}s"
            )
        if n_alerts > 5:
            print(f"    ... and {n_alerts - 5} more")
    else:
        print(f"\n  ✅ No threats detected.")

    # Save results
    results_df = pd.DataFrame({
        "window_index": range(len(probs)),
        "timestamp_sec": timestamps,
        "attack_probability": probs,
        "alert": alerts,
    })
    results_df.to_csv(
        output_dir / f"{pcap_name}_results.csv",
        index=False,
    )

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        timestamps, probs,
        color="#3498db", linewidth=1.5,
        label="Attack Probability",
    )
    ax.axhline(
        y=0.10, color="red",
        linestyle="--", label="Threshold (0.10)",
    )
    ax.fill_between(
        timestamps, probs,
        where=alerts,
        alpha=0.3, color="red",
        label="Alert Zone",
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Attack Probability")
    ax.set_title(
        f"PCAP Analysis: {pcap_name}",
        fontsize=14, fontweight="bold",
    )
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{pcap_name}_analysis.png",
        dpi=150,
    )
    plt.close(fig)
    print(
        f"\n  Saved: {pcap_name}_analysis.png"
    )

    return {
        "status": "success",
        "pcap": pcap_name,
        "total_packets": len(packet_df),
        "total_flows": len(flow_df),
        "total_windows": n_windows,
        "total_sequences": len(probs),
        "alerts": n_alerts,
        "max_probability": max_prob,
    }


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    DATA_DIR = Path(
        "data/processed/temporal_benchmark_v2"
    )
    MODEL_DIR = Path(
        "models/latent_world_model_v3"
    )
    OUTPUT_DIR = Path("models/pcap_analysis")

    SCALER_PATH = str(
        DATA_DIR / "input_scaler.pkl"
    )
    MODEL_PATH = str(
        MODEL_DIR / "latent_world_model_v3.pt"
    )

    # Determine input
    if len(sys.argv) > 1:
        pcap_input = Path(sys.argv[1])
    else:
        pcap_input = Path("data/sample_pcap")

    print("=" * 60)
    print("RAW NETWORK TRAFFIC INGESTION")
    print("PCAP → Features → World Model → Alerts")
    print("=" * 60)

    # Process single file or directory
    if pcap_input.is_file():
        pcap_files = [pcap_input]
    elif pcap_input.is_dir():
        pcap_files = sorted(
            pcap_input.glob("*.pcap")
        ) + sorted(
            pcap_input.glob("*.pcapng")
        )
    else:
        print(f"ERROR: {pcap_input} not found!")
        sys.exit(1)

    print(f"PCAP files found: {len(pcap_files)}")

    all_results = []
    for pcap_file in pcap_files:
        result = process_pcap(
            pcap_path=str(pcap_file),
            scaler_path=SCALER_PATH,
            model_path=MODEL_PATH,
            output_dir=str(OUTPUT_DIR),
        )
        all_results.append(result)

    # Final summary
    print(f"\n{'='*60}")
    print("INGESTION SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if r["status"] == "success":
            print(
                f"  {r['pcap']}: "
                f"{r['total_packets']} pkts → "
                f"{r['total_sequences']} sequences → "
                f"{r['alerts']} alerts "
                f"(max_prob={r['max_probability']:.4f})"
            )
        else:
            print(
                f"  {r.get('pcap', '?')}: "
                f"{r['status']}"
            )

    print(f"\nOutputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)
