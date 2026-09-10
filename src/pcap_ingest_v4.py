"""
=============================================================
PCAP -> V4 LATENT WORLD MODEL INFERENCE (CANONICAL)
=============================================================

This is the canonical bridge between raw packet captures
(.pcap / .pcapng) and the V4 Latent World Model.

It supersedes ``src/pcap_ingest.py`` (the legacy V3 single-step
script). Differences from the legacy script:

  1. V4 model class -- gated residual transition + LayerNorm,
     copied verbatim from ``train_latent_world_model_v4.py``.
  2. V4 checkpoint  -- ``models/latent_world_model_v4/
     latent_world_model_v4.pt``.
  3. Canonical scaler -- ``data/processed/multiday_benchmark_K5/
     input_scaler.pkl`` (fitted on the multiday train partition).
  4. Canonical feature ordering -- read from
     ``data/processed/multiday_benchmark_K5/
     input_feature_names.txt`` (single source of truth, 52 dims).
  5. True recursive K=5 latent rollout:
     ``encode -> [transition_step -> decode + attack_head] x 5``,
     feeding each predicted latent back into the transition.
     NOT five independent single-step predictions.
  6. Frozen threshold -- ``selected_threshold`` is read from the
     checkpoint (0.05). It is NEVER derived from the PCAP.
  7. Streaming PCAP read -- ``PcapReader`` iterator, not
     ``rdpcap`` (which loads the whole capture into memory).
  8. TRAINING-DEFINITION feature semantics -- the per-window
     26 base features are computed to mirror
     ``archive/create_network_states_v2.py`` (the code that
     built the ``states_v2`` CSVs the V4 model was trained on).
     See ``reconstruct_flows`` / ``compute_window_features``.
  9. Relative-change semantics matched to
     ``prepare_multiday_benchmark.py`` (first row zeroed,
     inf -> 0, clip [-10, 10]); post-scale ``nan_to_num``
     with posinf=10 / neginf=-10, matching training.
 10. Strict 10s temporal contiguity enforced when building
     sequences -- windows with no NEW flow create gaps, and a
     sequence that spans a gap is rejected (mirrors the
     ``np.all(diffs == 10)`` check in the benchmark builder).

------------------------------------------------------------
FEATURE PROVENANCE -- three distinct things, do not conflate
------------------------------------------------------------
  (a) OFFICIAL CICFlowMeter output -- the Java tool CIC ran to
      build CSE-CIC-IDS2018. NOT reproduced here (not installed,
      needs jnetpcap). The training ``states_v2`` CSVs are an
      aggregation of this.
  (b) TRAINING-DEFINITION EMULATION -- what THIS module now does:
      reconstruct whole-lifetime bidirectional 5-tuple flows
      (120 s idle timeout), initiator-based direction, assign
      each flow to the 10 s window of its FIRST packet, compute
      Flow Duration / Fwd IAT / Bwd IAT in MICROSECONDS over the
      whole flow, per-flow TCP-flag BOOLEANS, integer Down/Up
      ratio, L4-payload byte counts and packet-length stats,
      then aggregate per window exactly as
      ``create_network_states_v2.py`` (sum for totals, mean for
      per-flow quantities, nunique for diversity, safe_log1p,
      entropy_from_counts).
  (c) PCAP-derived implementation details that (b) still cannot
      match CICFlowMeter on: FIN/RST flow termination and
      subflow / active-idle / bulk logic are NOT emulated;
      long keepalive connections CICFlowMeter would split stay
      as one flow here. Direction is initiator-by-first-packet,
      which for mid-capture flows (no SYN observed) may differ
      from CICFlowMeter. These remain measured, not hidden.

Usage:
  python src/pcap_ingest_v4.py data/sample_pcap/smallFlows.pcap
  python src/pcap_ingest_v4.py path/to/dir_of_pcaps/
=============================================================
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scapy.all import IP, IPv6, TCP, UDP, PcapReader
from scapy.all import conf as scapy_conf

scapy_conf.verb = 0


# =============================================================
# CANONICAL PATHS & CONTRACT
# =============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

CANON_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
INPUT_SCALER_PATH = CANON_DIR / "input_scaler.pkl"
TARGET_SCALER_PATH = CANON_DIR / "target_scaler.pkl"
INPUT_FEATURE_NAMES_PATH = CANON_DIR / "input_feature_names.txt"
STATE_FEATURE_NAMES_PATH = CANON_DIR / "state_feature_names.txt"

MODEL_DIR = REPO_ROOT / "models" / "latent_world_model_v4"
CHECKPOINT_PATH = MODEL_DIR / "latent_world_model_v4.pt"

OUTPUT_DIR = REPO_ROOT / "models" / "pcap_analysis_v4"

WINDOW_SECONDS = 10   # aggregation bin width
SEQ_LEN = 10          # history windows fed to the encoder
CONTIGUITY_STEP = 10  # required spacing (seconds) between consecutive windows


def _read_feature_names(path: Path) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


INPUT_FEATURE_NAMES = _read_feature_names(INPUT_FEATURE_NAMES_PATH)   # 52
STATE_FEATURE_NAMES = _read_feature_names(STATE_FEATURE_NAMES_PATH)   # 26
BASE_FEATURES = INPUT_FEATURE_NAMES[:26]
REL_CHANGE_FEATURES = INPUT_FEATURE_NAMES[26:]

# Hard contract checks -- fail loudly if the canonical files drift.
assert len(INPUT_FEATURE_NAMES) == 52, (
    f"expected 52 input features, got {len(INPUT_FEATURE_NAMES)}"
)
assert len(STATE_FEATURE_NAMES) == 26, (
    f"expected 26 state features, got {len(STATE_FEATURE_NAMES)}"
)
assert BASE_FEATURES == STATE_FEATURE_NAMES, (
    "first 26 input features must equal the 26 state features"
)
assert REL_CHANGE_FEATURES == [f"{f}_rel_change" for f in BASE_FEATURES], (
    "rel-change features must be base features + '_rel_change', in order"
)


# =============================================================
# V4 MODEL DEFINITION
# (verbatim from src/train_latent_world_model_v4.py L387-569)
# =============================================================

class LatentWorldModelV4(nn.Module):
    """Encoder LSTM -> LayerNorm -> 32-D latent; gated residual
    deterministic transition -> LayerNorm; decoder + attack head."""

    def __init__(
        self,
        input_size,
        hidden_size,
        latent_size,
        output_size,
        num_layers,
        dropout,
    ):
        super().__init__()

        # ----- Encoder -----
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        # ----- Transition (gated residual + LayerNorm) -----
        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size),
        )
        self.gate_fc = nn.Linear(latent_size, latent_size)
        self.transition_norm = nn.LayerNorm(latent_size)

        # ----- Decoder -----
        self.decoder_fc1 = nn.Linear(latent_size, 64)
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(64, output_size)

        # ----- Attack head -----
        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        output, _ = self.encoder_lstm(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.encoder_norm(last_hidden)
        return self.encoder_fc(last_hidden)

    def transition_step(self, latent_state):
        candidate = self.transition_mlp(latent_state)
        gate = torch.sigmoid(self.gate_fc(latent_state))
        z_next = gate * candidate + (1 - gate) * latent_state
        return self.transition_norm(z_next)

    def decode(self, latent_state):
        hidden = self.decoder_fc1(latent_state)
        hidden = self.decoder_activation(hidden)
        return self.decoder_fc2(hidden)

    def predict_attack(self, latent_state):
        return self.attack_head(latent_state).squeeze(-1)

    def k_step_rollout(self, x, k):
        """Recursive K-step rollout. Returns (traj_z, traj_probs,
        traj_states), each of length k + 1. Index 0 is Z_0
        (nowcast); index i is the predicted state at t+i, with the
        predicted latent fed back into the transition each step."""
        z_current = self.encode(x)
        traj_z = [z_current]
        traj_probs = [torch.sigmoid(self.predict_attack(z_current))]
        traj_states = [self.decode(z_current)]
        for _ in range(k):
            z_current = self.transition_step(z_current)
            traj_probs.append(torch.sigmoid(self.predict_attack(z_current)))
            traj_z.append(z_current)
            traj_states.append(self.decode(z_current))
        return traj_z, traj_probs, traj_states


# =============================================================
# MODEL / SCALER LOADING
# =============================================================

def load_v4_model(checkpoint_path: Path = CHECKPOINT_PATH):
    """Load the V4 checkpoint and rebuild the model. Returns
    (model, checkpoint_dict). The model is in eval() mode."""
    ckpt = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    if ckpt.get("model_class") not in (None, "LatentWorldModelV4"):
        raise ValueError(
            f"checkpoint model_class = {ckpt.get('model_class')!r}, "
            f"expected 'LatentWorldModelV4'"
        )
    model = LatentWorldModelV4(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        latent_size=ckpt["latent_size"],
        output_size=ckpt["output_size"],
        num_layers=ckpt["num_layers"],
        dropout=0.20,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def load_scaler(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


# =============================================================
# STEP 1: STREAMING PCAP PARSE
# =============================================================

_FLAG_KEYS = ("f_fin", "f_syn", "f_rst", "f_psh", "f_ack", "f_urg")
_FLAG_LETTER = {
    "f_fin": "F", "f_syn": "S", "f_rst": "R",
    "f_psh": "P", "f_ack": "A", "f_urg": "U",
}

# CICFlowMeter FlowTimeout used by CIC for CSE-CIC-IDS2018 (and hence
# the states_v2 training features). A gap longer than this within a
# 5-tuple starts a new flow.
FLOW_IDLE_TIMEOUT_S = 120.0


def _safe_log1p(x):
    """Port of archive/create_network_states_v2.py:safe_log1p --
    coerce, drop inf, fill 0, clip negatives to 0, then log1p."""
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return np.log1p(s)


def _safe_divide(numerator, denominator):
    """Port of archive/create_network_states_v2.py:safe_divide."""
    r = pd.Series(numerator) / (pd.Series(denominator).abs() + 1e-8)
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _entropy_from_counts(series) -> float:
    """Exact port of archive/create_network_states_v2.py:
    entropy_from_counts -- Shannon entropy (log2) of the categorical
    value_counts; 0.0 when there is <= 1 distinct value."""
    counts = pd.Series(series).value_counts(dropna=True)
    if len(counts) <= 1:
        return 0.0
    p = counts.to_numpy(dtype=float) / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def parse_pcap_streaming(pcap_path: str) -> pd.DataFrame:
    """Iterate a capture packet-by-packet (constant memory) and build
    a per-packet table. Records, per IP packet: absolute epoch time,
    5-tuple, full frame length, L4 payload length (bytes above the
    transport header -- CICFlowMeter's "packet length" / "TotLen"
    semantics), and the six TCP flag booleans."""
    rows = []
    n_total = 0
    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            n_total += 1
            l3 = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if l3 is None:
                continue
            is_v6 = l3.name == "IPv6"
            proto = int(l3.nh) if is_v6 else int(l3.proto)

            src_port = 0
            dst_port = 0
            l4_payload = 0
            flag_bits = {k: False for k in _FLAG_KEYS}

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)
                flag_str = str(tcp.flags)
                for key in _FLAG_KEYS:
                    flag_bits[key] = _FLAG_LETTER[key] in flag_str
                l4_payload = len(tcp.payload)
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                src_port = int(udp.sport)
                dst_port = int(udp.dport)
                l4_payload = len(udp.payload)
            else:
                l4_payload = len(l3.payload)

            row = {
                "timestamp_abs": float(pkt.time),
                "src_ip": str(l3.src),
                "dst_ip": str(l3.dst),
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": proto,
                "frame_len": int(len(pkt)),
                "l4_payload_len": int(l4_payload),
            }
            row.update(flag_bits)
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  Packets read (streamed): {n_total}")
    if len(df) == 0:
        print("  WARNING: no IP packets found")
        return df

    df = df.sort_values("timestamp_abs", kind="stable").reset_index(drop=True)
    t0 = float(df["timestamp_abs"].min())
    df["timestamp"] = df["timestamp_abs"] - t0
    print(f"  IP packets: {len(df)}")
    print(
        f"  Capture span: {df['timestamp'].max():.1f} s "
        f"(t0 = {t0:.3f} epoch)"
    )
    df.attrs["t0_epoch"] = t0
    return df


# =============================================================
# STEP 2: WHOLE-LIFETIME FLOW RECONSTRUCTION
# (mirrors the CICFlowMeter flow unit that
#  archive/create_network_states_v2.py consumes)
# =============================================================

def reconstruct_flows(
    packet_df: pd.DataFrame,
    flow_idle_timeout_s: float = FLOW_IDLE_TIMEOUT_S,
) -> pd.DataFrame:
    """Reconstruct whole-lifetime bidirectional 5-tuple flows.

    * Flow key is the UNORDERED 5-tuple {ip_a, ip_b, port_a, port_b,
      proto}; forward direction = the (src_ip, src_port) of the flow's
      FIRST packet (connection-initiator semantics).
    * A flow is split ONLY on an idle gap longer than
      ``flow_idle_timeout_s`` (CICFlowMeter FlowTimeout = 120 s).
      It is NOT reset at 10 s window boundaries.
    * The flow is assigned to ``window_idx`` = the 10 s bin of its
      first packet; ALL of its statistics belong to that window.

    Per-flow columns produced (CICFlowMeter-column semantics):
      flow_duration_us / fwd_iat_mean_us / bwd_iat_mean_us -- MICROSECONDS
      tot_fwd_pkts / tot_bwd_pkts / tot_pkts
      totlen_fwd / totlen_bwd / total_bytes  -- L4 payload bytes
      pkt_len_mean / pkt_len_std             -- L4 payload bytes
      syn..urg                               -- per-flow BOOLEAN (0/1)
      down_up_ratio                          -- integer  bwd // fwd
    plus the per-flow ratios / rates that create_network_states_v2.py
    computes BEFORE the window groupby.
    """
    if packet_df is None or len(packet_df) == 0:
        return pd.DataFrame()

    pdf = packet_df
    src = pdf["src_ip"].to_numpy()
    dst = pdf["dst_ip"].to_numpy()
    sp = pdf["src_port"].to_numpy()
    dp = pdf["dst_port"].to_numpy()
    lo_ip = np.minimum(src, dst)
    hi_ip = np.maximum(src, dst)
    lo_pt = np.minimum(sp, dp)
    hi_pt = np.maximum(sp, dp)
    key = (
        pd.Series(lo_ip) + "|" + pd.Series(hi_ip) + "|"
        + pd.Series(lo_pt).astype(str) + "|" + pd.Series(hi_pt).astype(str)
        + "|" + pd.Series(pdf["protocol"].to_numpy()).astype(str)
    ).to_numpy()

    work = pdf.assign(_k=key)
    t0 = float(work["timestamp_abs"].min())
    win0 = int(t0 // WINDOW_SECONDS)

    flows = []
    for fkey, g in work.groupby("_k", sort=False):
        g = g.sort_values("timestamp_abs", kind="stable")
        ts = g["timestamp_abs"].to_numpy()
        if len(ts) > 1:
            brk = np.nonzero(np.diff(ts) > flow_idle_timeout_s)[0] + 1
            segments = np.split(np.arange(len(ts)), brk)
        else:
            segments = [np.arange(len(ts))]

        gsrc = g["src_ip"].to_numpy()
        gsp = g["src_port"].to_numpy()
        gdp = g["dst_port"].to_numpy()
        gproto = g["protocol"].to_numpy()
        gpay = g["l4_payload_len"].to_numpy().astype(np.int64)
        gflags = {fk: g[fk].to_numpy() for fk in _FLAG_KEYS}

        for seg in segments:
            if len(seg) == 0:
                continue
            i0 = seg[0]
            seg_ts = ts[seg]
            is_fwd = (gsrc[seg] == gsrc[i0]) & (gsp[seg] == gsp[i0])
            n_fwd = int(is_fwd.sum())
            n_bwd = int(len(seg) - n_fwd)
            n_pkts = n_fwd + n_bwd

            fwd_ts = np.sort(seg_ts[is_fwd])
            bwd_ts = np.sort(seg_ts[~is_fwd])
            seg_pay = gpay[seg]
            totlen_fwd = int(seg_pay[is_fwd].sum())
            totlen_bwd = int(seg_pay[~is_fwd].sum())

            dur_us = float((seg_ts[-1] - seg_ts[0]) * 1e6) if len(seg_ts) > 1 else 0.0
            fwd_iat_us = float(np.diff(fwd_ts).mean() * 1e6) if n_fwd > 1 else 0.0
            bwd_iat_us = float(np.diff(bwd_ts).mean() * 1e6) if n_bwd > 1 else 0.0

            pay_f = seg_pay.astype(float)
            pkt_len_mean = float(pay_f.mean()) if pay_f.size else 0.0
            # sample std (ddof=1), matching CICFlowMeter's commons-math
            # SummaryStatistics.getStandardDeviation()
            pkt_len_std = float(pay_f.std(ddof=1)) if pay_f.size > 1 else 0.0

            start = float(seg_ts[0])
            flows.append({
                "window_idx": int(start // WINDOW_SECONDS) - win0,
                "start_ts": start,
                "flow_key": fkey,   # canonical unordered 5-tuple (metadata;
                                    # used for cross-capture flow dedup)
                "dst_port": int(gdp[i0]),
                "protocol": int(gproto[i0]),
                "tot_pkts": n_pkts,
                "tot_fwd_pkts": n_fwd,
                "tot_bwd_pkts": n_bwd,
                "totlen_fwd": totlen_fwd,
                "totlen_bwd": totlen_bwd,
                "total_bytes": totlen_fwd + totlen_bwd,
                "flow_duration_us": dur_us,
                "fwd_iat_mean_us": fwd_iat_us,
                "bwd_iat_mean_us": bwd_iat_us,
                "pkt_len_mean": pkt_len_mean,
                "pkt_len_std": pkt_len_std,
                "syn": int(bool(gflags["f_syn"][seg].any())),
                "ack": int(bool(gflags["f_ack"][seg].any())),
                "rst": int(bool(gflags["f_rst"][seg].any())),
                "fin": int(bool(gflags["f_fin"][seg].any())),
                "psh": int(bool(gflags["f_psh"][seg].any())),
                "urg": int(bool(gflags["f_urg"][seg].any())),
                "down_up_ratio": int(n_bwd // n_fwd) if n_fwd > 0 else 0,
            })

    fl = pd.DataFrame(flows)
    if len(fl) == 0:
        return fl

    # per-flow ratios / rates -- computed BEFORE the window groupby,
    # exactly as create_network_states_v2.py does.
    fl["fwd_packet_ratio"] = _safe_divide(fl["tot_fwd_pkts"], fl["tot_pkts"]).to_numpy()
    fl["bwd_packet_ratio"] = _safe_divide(fl["tot_bwd_pkts"], fl["tot_pkts"]).to_numpy()
    fl["fwd_byte_ratio"] = _safe_divide(fl["totlen_fwd"], fl["total_bytes"]).to_numpy()
    fl["bwd_byte_ratio"] = _safe_divide(fl["totlen_bwd"], fl["total_bytes"]).to_numpy()
    denom = fl["tot_pkts"].to_numpy() + 1e-8
    for rate, flag in (("syn_rate", "syn"), ("ack_rate", "ack"),
                       ("rst_rate", "rst"), ("fin_rate", "fin"),
                       ("psh_rate", "psh"), ("urg_rate", "urg")):
        fl[rate] = fl[flag].to_numpy() / denom
    return fl


# =============================================================
# STEP 3: 26 BASE FEATURES PER WINDOW
# (window aggregation identical to
#  archive/create_network_states_v2.py -- flow assigned to the
#  window of its first packet; sum for totals, mean for per-flow
#  quantities, nunique for diversity, then safe_log1p + rates)
# =============================================================

def compute_window_features(
    flow_df: pd.DataFrame, packet_df: pd.DataFrame = None
) -> pd.DataFrame:
    if flow_df is None or len(flow_df) == 0:
        return pd.DataFrame()

    g = flow_df.groupby("window_idx", sort=True)
    agg = g.agg(
        flow_count=("window_idx", "size"),
        total_packets=("tot_pkts", "sum"),
        total_bytes=("total_bytes", "sum"),
        mean_flow_duration=("flow_duration_us", "mean"),
        mean_fwd_iat=("fwd_iat_mean_us", "mean"),
        mean_bwd_iat=("bwd_iat_mean_us", "mean"),
        mean_pkt_len=("pkt_len_mean", "mean"),
        std_pkt_len=("pkt_len_std", "mean"),
        syn_rate=("syn_rate", "mean"),
        ack_rate=("ack_rate", "mean"),
        rst_rate=("rst_rate", "mean"),
        fin_rate=("fin_rate", "mean"),
        psh_rate=("psh_rate", "mean"),
        urg_rate=("urg_rate", "mean"),
        fwd_packet_ratio=("fwd_packet_ratio", "mean"),
        bwd_packet_ratio=("bwd_packet_ratio", "mean"),
        fwd_byte_ratio=("fwd_byte_ratio", "mean"),
        bwd_byte_ratio=("bwd_byte_ratio", "mean"),
        mean_down_up_ratio=("down_up_ratio", "mean"),
        unique_dst_ports=("dst_port", "nunique"),
        unique_protocols=("protocol", "nunique"),
    ).reset_index()

    agg["log_flow_count"] = _safe_log1p(agg["flow_count"]).to_numpy()
    agg["log_total_packets"] = _safe_log1p(agg["total_packets"]).to_numpy()
    agg["log_total_bytes"] = _safe_log1p(agg["total_bytes"]).to_numpy()
    agg["log_mean_flow_duration"] = _safe_log1p(agg["mean_flow_duration"]).to_numpy()
    agg["log_mean_fwd_iat"] = _safe_log1p(agg["mean_fwd_iat"]).to_numpy()
    agg["log_mean_bwd_iat"] = _safe_log1p(agg["mean_bwd_iat"]).to_numpy()
    agg["log_unique_dst_ports"] = _safe_log1p(agg["unique_dst_ports"]).to_numpy()

    agg["window_packet_rate"] = agg["total_packets"] / WINDOW_SECONDS
    agg["window_byte_rate"] = agg["total_bytes"] / WINDOW_SECONDS

    ent_port = (
        g["dst_port"].apply(_entropy_from_counts)
        .reset_index(name="dst_port_entropy")
    )
    ent_proto = (
        g["protocol"].apply(_entropy_from_counts)
        .reset_index(name="protocol_entropy")
    )
    agg = agg.merge(ent_port, on="window_idx").merge(ent_proto, on="window_idx")

    agg["timestamp"] = agg["window_idx"] * WINDOW_SECONDS

    out = agg.reindex(columns=["window_idx", "timestamp"] + BASE_FEATURES)
    out[BASE_FEATURES] = (
        out[BASE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    return out.sort_values("window_idx").reset_index(drop=True)


# =============================================================
# STEP 4: RELATIVE-CHANGE FEATURES
# (semantics mirror prepare_multiday_benchmark.py L89-95)
# =============================================================

def add_relative_changes(window_df: pd.DataFrame) -> pd.DataFrame:
    df = window_df.copy().reset_index(drop=True)
    state = df[BASE_FEATURES].astype(float)
    delta = state.diff().fillna(0.0)
    previous_state = state.shift(1)
    rel = delta / (previous_state.abs() + 1e-8)
    rel.iloc[0] = 0.0
    rel = (
        rel.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-10.0, 10.0)
    )
    rel.columns = [f"{f}_rel_change" for f in BASE_FEATURES]
    return pd.concat([df, rel], axis=1)


# =============================================================
# STEP 5: SCALE + BUILD CONTIGUOUS SEQUENCES
# =============================================================

def build_sequences(
    feature_df: pd.DataFrame,
    input_scaler,
    seq_len: int = SEQ_LEN,
) -> dict:
    """Z-score with the canonical scaler, then slide a length-10
    window. A sequence is kept only if its 10 windows are strictly
    contiguous (consecutive ``window_idx``, i.e. 10s apart) --
    mirrors the ``np.all(diffs == 10)`` gate in the benchmark
    builder. Returns a dict with X and per-sequence bookkeeping."""
    # Pass a DataFrame with the canonical column names so the scaler
    # (fitted on named columns) validates the ordering itself.
    feats = feature_df.reindex(columns=INPUT_FEATURE_NAMES).astype(np.float64)
    scaled = input_scaler.transform(feats)
    scaled = np.nan_to_num(
        scaled, nan=0.0, posinf=10.0, neginf=-10.0
    ).astype(np.float32)

    win_idx = feature_df["window_idx"].to_numpy()
    ts_rel = feature_df["timestamp"].to_numpy()
    t0_epoch = feature_df.attrs.get("t0_epoch", 0.0)

    windows = []
    seq_end_window = []
    seq_ts_rel = []
    seq_ts_abs = []
    n_rejected_gap = 0

    for i in range(len(scaled) - seq_len + 1):
        block = win_idx[i:i + seq_len]
        if not np.all(np.diff(block) == 1):
            n_rejected_gap += 1
            continue
        windows.append(scaled[i:i + seq_len])
        seq_end_window.append(int(block[-1]))
        seq_ts_rel.append(float(ts_rel[i + seq_len - 1]))
        seq_ts_abs.append(float(t0_epoch + ts_rel[i + seq_len - 1]))

    X = (
        np.asarray(windows, dtype=np.float32)
        if windows else np.empty((0, seq_len, 52), dtype=np.float32)
    )
    return {
        "X": X,
        "seq_end_window": seq_end_window,
        "seq_ts_rel": seq_ts_rel,
        "seq_ts_abs": seq_ts_abs,
        "n_rejected_gap": n_rejected_gap,
        "n_candidate": max(len(scaled) - seq_len + 1, 0),
    }


# =============================================================
# STEP 6: RECURSIVE K=5 ROLLOUT
# =============================================================

def _warning_level(max_prob: float, threshold: float) -> str:
    if max_prob < threshold:
        return "CLEAR"
    if max_prob < 2 * threshold:
        return "ELEVATED"
    if max_prob < 0.25:
        return "WARNING"
    return "CRITICAL"


def run_k_rollout(
    model: LatentWorldModelV4,
    X: np.ndarray,
    k: int,
    target_scaler=None,
) -> dict:
    """True recursive rollout: encode once, then feed each
    predicted latent back through ``transition_step`` k times.
    Returns probability trajectories (k+1 columns: t+0..t+k) and,
    optionally, decoded next-state features in original units."""
    if len(X) == 0:
        return {"probs": np.empty((0, k + 1)), "decoded": None}

    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)
        traj_z, traj_probs, traj_states = model.k_step_rollout(x_t, k)

    probs = np.stack([p.numpy() for p in traj_probs], axis=1)  # (N, k+1)
    assert probs.shape[1] == k + 1, (
        f"rollout produced {probs.shape[1]} columns, expected {k + 1}"
    )

    decoded = None
    if target_scaler is not None:
        # traj_states index 0 = nowcast, 1..k = t+1..t+k
        decoded_steps = []
        for step_idx, state in enumerate(traj_states):
            arr = state.numpy()
            arr = target_scaler.inverse_transform(arr)
            for seq_i in range(arr.shape[0]):
                rec = {"seq_index": seq_i, "k": step_idx}
                rec.update(dict(zip(STATE_FEATURE_NAMES, arr[seq_i])))
                decoded_steps.append(rec)
        decoded = pd.DataFrame(decoded_steps)

    return {"probs": probs, "decoded": decoded}


# =============================================================
# FULL PIPELINE
# =============================================================

def process_pcap(
    pcap_path: str,
    output_dir: Path = OUTPUT_DIR,
    model=None,
    ckpt=None,
    input_scaler=None,
    target_scaler=None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = Path(pcap_path).stem

    if model is None or ckpt is None:
        model, ckpt = load_v4_model()
    if input_scaler is None:
        input_scaler = load_scaler(INPUT_SCALER_PATH)
    if target_scaler is None:
        target_scaler = load_scaler(TARGET_SCALER_PATH)

    k_rollout = int(ckpt["max_k"])
    threshold = float(ckpt["selected_threshold"])   # FROZEN, from checkpoint

    print(f"\n{'=' * 60}")
    print(f"PCAP -> V4 INFERENCE: {name}")
    print(f"{'=' * 60}")
    print(f"  checkpoint      : {CHECKPOINT_PATH}")
    print(f"  input scaler    : {INPUT_SCALER_PATH}")
    print(f"  threshold       : {threshold} (from checkpoint)")
    print(f"  rollout steps K : {k_rollout}")

    print("\n[1/6] Streaming packet parse...")
    packet_df = parse_pcap_streaming(pcap_path)
    if len(packet_df) == 0:
        return {"status": "error", "pcap": name, "reason": "no IP packets"}
    t0_epoch = packet_df.attrs.get("t0_epoch", 0.0)

    print("[2/6] Flow reconstruction...")
    flow_df = reconstruct_flows(packet_df)
    n_windows_with_traffic = int(flow_df["window_idx"].nunique())
    print(f"  whole-lifetime flows: {len(flow_df)}  |  windows with "
          f">=1 new flow: {n_windows_with_traffic}")

    print("[3/6] 26 base features per window...")
    feature_df = compute_window_features(flow_df, packet_df)
    print(f"  feature windows: {len(feature_df)}")

    print("[4/6] Relative-change features -> 52 dims...")
    feature_df = add_relative_changes(feature_df)
    feature_df.attrs["t0_epoch"] = t0_epoch

    # window-index gaps (empty 10s bins)
    wi = feature_df["window_idx"].to_numpy()
    gap_positions = int(np.sum(np.diff(wi) != 1)) if len(wi) > 1 else 0
    max_gap = int(np.max(np.diff(wi))) if len(wi) > 1 else 0
    print(f"  window-index gaps: {gap_positions}  |  max step: {max_gap}")

    feature_df.to_csv(output_dir / f"{name}_features.csv", index=False)

    print("[5/6] Scale + build strictly-contiguous sequences...")
    seq = build_sequences(feature_df, input_scaler)
    X = seq["X"]
    print(f"  candidate sequences : {seq['n_candidate']}")
    print(f"  rejected (temporal gap): {seq['n_rejected_gap']}")
    print(f"  usable sequences    : {len(X)}")
    if len(X) == 0:
        summary = {
            "status": "insufficient_data",
            "pcap": name,
            "windows_with_traffic": n_windows_with_traffic,
            "feature_windows": int(len(feature_df)),
            "window_index_gaps": gap_positions,
            "candidate_sequences": seq["n_candidate"],
            "rejected_temporal_gap": seq["n_rejected_gap"],
            "usable_sequences": 0,
            "threshold": threshold,
            "reason": (
                f"need >= {SEQ_LEN} strictly contiguous 10s windows "
                f"({SEQ_LEN * WINDOW_SECONDS}s of gap-free traffic)"
            ),
        }
        with open(output_dir / f"{name}_summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"  -> insufficient data; wrote {name}_summary.json")
        return summary

    print(f"[6/6] Recursive K={k_rollout} rollout...")
    roll = run_k_rollout(
        model, X, k_rollout, target_scaler=target_scaler
    )
    probs = roll["probs"]                       # (N, K+1)
    prob_current = probs[:, 0]
    prob_future = probs[:, 1:]
    max_future = prob_future.max(axis=1)
    max_all = probs.max(axis=1)

    rollout_df = pd.DataFrame({
        "seq_index": np.arange(len(probs)),
        "end_window_idx": seq["seq_end_window"],
        "timestamp_rel_sec": seq["seq_ts_rel"],
        "timestamp_abs_epoch": seq["seq_ts_abs"],
        "prob_t+0_nowcast": prob_current,
    })
    for kk in range(1, k_rollout + 1):
        rollout_df[f"prob_t+{kk}"] = probs[:, kk]
    rollout_df["max_future_prob"] = max_future
    rollout_df["max_all_prob"] = max_all
    rollout_df["alert_nowcast"] = prob_current >= threshold
    rollout_df["alert_any_future"] = max_future >= threshold
    rollout_df["warning_level"] = [
        _warning_level(float(m), threshold) for m in max_all
    ]
    rollout_df.to_csv(output_dir / f"{name}_rollout.csv", index=False)

    if roll["decoded"] is not None:
        roll["decoded"].to_csv(
            output_dir / f"{name}_decoded_states.csv", index=False
        )

    n_alert_nowcast = int((prob_current >= threshold).sum())
    n_alert_future = int((max_future >= threshold).sum())
    peak = float(max_all.max())
    peak_seq = int(np.argmax(max_all))
    overall_level = _warning_level(peak, threshold)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {name}")
    print(f"{'=' * 60}")
    print(f"  usable sequences        : {len(probs)}")
    print(f"  nowcast alerts (t+0)    : {n_alert_nowcast}")
    print(f"  future alerts (t+1..t+{k_rollout}): {n_alert_future}")
    print(f"  peak probability        : {peak:.4f} "
          f"(seq {peak_seq}, level {overall_level})")
    print(f"  peak trajectory         : "
          + " -> ".join(f"{v:.3f}" for v in probs[peak_seq]))

    _plot_trajectories(
        probs, seq["seq_ts_rel"], threshold, k_rollout, name, output_dir
    )

    summary = {
        "status": "success",
        "pcap": name,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
        "input_scaler": str(INPUT_SCALER_PATH.relative_to(REPO_ROOT)),
        "threshold": threshold,
        "threshold_source": "checkpoint.selected_threshold",
        "rollout_steps_K": k_rollout,
        "sequence_length": SEQ_LEN,
        "window_seconds": WINDOW_SECONDS,
        "input_dim": int(X.shape[2]),
        "latent_dim": int(ckpt["latent_size"]),
        "decoder_dim": int(ckpt["output_size"]),
        "packets": int(len(packet_df)),
        "flows": int(len(flow_df)),
        "windows_with_traffic": n_windows_with_traffic,
        "feature_windows": int(len(feature_df)),
        "window_index_gaps": gap_positions,
        "candidate_sequences": seq["n_candidate"],
        "rejected_temporal_gap": seq["n_rejected_gap"],
        "usable_sequences": int(len(probs)),
        "nowcast_alerts": n_alert_nowcast,
        "future_alerts": n_alert_future,
        "peak_probability": peak,
        "peak_sequence_index": peak_seq,
        "peak_trajectory": [float(v) for v in probs[peak_seq]],
        "overall_warning_level": overall_level,
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "feature_semantics": "training-definition emulation of "
        "archive/create_network_states_v2.py (whole-lifetime flows, "
        "120s idle timeout, initiator direction, microsecond "
        "duration/IAT, per-flow flag booleans, L4-payload bytes, "
        "flow assigned to first-packet window)",
        "residual_gaps_vs_official_cicflowmeter": [
            "FIN/RST flow termination not emulated (idle timeout only)",
            "no subflow / active-idle / bulk-transfer logic",
            "initiator direction from first observed packet; may differ "
            "from CICFlowMeter for flows whose SYN precedes the capture",
            "packet-length stat sample count may differ by ~1 from "
            "CICFlowMeter's internal accounting",
        ],
    }
    with open(output_dir / f"{name}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  outputs -> {output_dir}/")
    print(f"    {name}_features.csv")
    print(f"    {name}_rollout.csv")
    print(f"    {name}_decoded_states.csv")
    print(f"    {name}_trajectory.png")
    print(f"    {name}_summary.json")
    return summary


def _plot_trajectories(probs, ts_rel, threshold, k, name, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    steps = np.arange(k + 1)
    for i in range(probs.shape[0]):
        ax1.plot(steps, probs[i], color="#3498db", alpha=0.15, linewidth=0.8)
    ax1.plot(steps, probs.mean(axis=0), color="#e74c3c", linewidth=2.5,
             marker="o", label="mean trajectory")
    ax1.axhline(threshold, color="black", linestyle="--",
                label=f"threshold ({threshold})")
    ax1.set_xlabel("rollout step (t+k)")
    ax1.set_ylabel("attack probability")
    ax1.set_title(f"Recursive K={k} rollout trajectories -- {name}",
                  fontweight="bold")
    ax1.set_xticks(steps)
    ax1.set_ylim(0, 1)
    ax1.legend()

    max_all = probs.max(axis=1)
    ax2.plot(ts_rel, max_all, color="#8e44ad", linewidth=1.5,
             label="max prob over t+0..t+%d" % k)
    ax2.plot(ts_rel, probs[:, 0], color="#2980b9", linewidth=1.0,
             alpha=0.7, label="nowcast prob (t+0)")
    ax2.axhline(threshold, color="black", linestyle="--",
                label=f"threshold ({threshold})")
    ax2.fill_between(ts_rel, max_all, threshold,
                     where=(max_all >= threshold), color="red", alpha=0.25)
    ax2.set_xlabel("capture time (seconds, relative)")
    ax2.set_ylabel("attack probability")
    ax2.set_title("Per-sequence attack probability over capture time")
    ax2.set_ylim(0, 1)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_trajectory.png", dpi=140)
    plt.close(fig)


# =============================================================
# MAIN
# =============================================================

def _iter_pcaps(target: Path):
    if target.is_file():
        return [target]
    if target.is_dir():
        out = []
        for ext in ("*.pcap", "*.pcapng", "*.cap"):
            out.extend(sorted(target.glob(ext)))
        return out
    return []


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    target = Path(argv[0]) if argv else (REPO_ROOT / "data" / "sample_pcap")

    print("=" * 60)
    print("PCAP -> V4 LATENT WORLD MODEL (CANONICAL)")
    print("=" * 60)

    pcaps = _iter_pcaps(target)
    if not pcaps:
        print(f"ERROR: no pcap files at {target}")
        return 1

    model, ckpt = load_v4_model()
    input_scaler = load_scaler(INPUT_SCALER_PATH)
    target_scaler = load_scaler(TARGET_SCALER_PATH)

    results = []
    for pcap in pcaps:
        results.append(process_pcap(
            str(pcap),
            model=model,
            ckpt=ckpt,
            input_scaler=input_scaler,
            target_scaler=target_scaler,
        ))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        if r["status"] == "success":
            print(f"  {r['pcap']}: {r['packets']} pkts -> "
                  f"{r['usable_sequences']} seqs -> "
                  f"peak {r['peak_probability']:.4f} "
                  f"[{r['overall_warning_level']}]")
        else:
            print(f"  {r['pcap']}: {r['status']} ({r.get('reason', '')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
