"""
=============================================================
V5 CANONICAL PACKET FEATURE BUILDER
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

THE single implementation of the 21 packet-level features that the V5
experimental representation adds on top of the frozen V4 flow state.

Both the "training-side" path and the "PCAP-inference-side" path call the
functions in this module. There is no second copy of any formula, so
packet-feature parity between the two paths is EXACT BY CONSTRUCTION rather
than by coincidence (``src/run_v5_packet_features.py`` still verifies it
numerically across two independent invocation routes — see Part 15).

------------------------------------------------------------
LAYERING (Part 10) — each stage is a separate public function
------------------------------------------------------------
    pcap file
      -> parse_pcap_packets()        per-packet header table (no statistics)
      -> annotate_packets()          causal per-direction session state
                                     (retransmission, window-change)
      -> build_packet_windows()      10 s window aggregation -> 21 features
      -> add_packet_relative_changes()  the 9 selected relative changes

NOTHING in this module computes a flow-level feature. The 26 V4 flow
features are produced only by ``src/pcap_ingest_v4.py`` (PCAP path) and
``archive/create_network_states_v2.py`` (training path), both untouched.

------------------------------------------------------------
WINDOW SEMANTICS (Part 9)
------------------------------------------------------------
Identical bin width, origin and boundary rule to V4:
    window_idx = floor(packet_epoch_seconds / 10)      interval [t, t+10)
with the same ``- win0`` re-basing to the first window of the capture.

ONE DELIBERATE DIFFERENCE FROM THE V4 FLOW LAYER, documented because it is
load-bearing: V4 assigns a WHOLE FLOW (all of its statistics) to the window
of its FIRST packet. A packet feature instead assigns EACH PACKET to the
window containing THAT PACKET'S OWN timestamp. This is the only assignment
rule that is causally sound for packet statistics — the flow rule would let
packets arriving after t+10 contribute to the state at t, which is exactly
the leakage Part 19 forbids. See docs/V5_PACKET_FEATURES.md Section 15.

------------------------------------------------------------
DETERMINISTIC PACKET ORDER
------------------------------------------------------------
Capture timestamps are not unique — a busy microsecond holds several
packets — and the session-state annotations (retransmission, window change)
depend on the order in which packets are observed. Sorting on the timestamp
alone therefore leaves the result dependent on how the rows happened to be
arranged, which makes the builder non-deterministic across storage
round-trips.

Every packet consequently carries ``packet_index``, its CAPTURE-ORDER
ordinal, and every stage sorts on ``(timestamp_abs, packet_index)``. Within
one timestamp the capture order is the best available statement of what the
sensor actually saw first, so this is the correct tie-break, not merely a
convenient one. ``ensure_packet_order`` assigns the ordinal from the
caller's row order when a table arrives without one.

------------------------------------------------------------
CAUSALITY (Part 19)
------------------------------------------------------------
``annotate_packets`` is a SINGLE FORWARD PASS in non-decreasing timestamp
order. Every per-packet annotation is a function of that packet and of
packets with a strictly earlier (or equal) timestamp in the same directional
session. No annotation is ever revised once written, so no future packet can
retroactively change an earlier window's state.

------------------------------------------------------------
NUMERICAL POLICY (Part 13)
------------------------------------------------------------
* A window with no IP packets produces NO ROW (identical to V4's treatment
  of empty windows).
* A family with no eligible packets in a window (e.g. no TCP traffic)
  yields 0.0 for every feature of that family. 0.0 is the ABSENT encoding.
  This mirrors V4's ``replace([inf,-inf], nan).fillna(0.0)`` policy. The
  ambiguity it creates ("absent" vs "genuinely zero") is real and is stated
  in the contract's ``zero_value_behaviour`` field for every feature.
* Every division goes through ``_ratio`` which returns 0.0 on a zero
  denominator. No epsilon-inflated denominator is used for the packet
  family (unlike V4's flow rates) because the denominators here are exact
  integer packet counts, so the zero case is testable rather than smeared.
* Standard deviation uses ddof=1 and returns 0.0 for n <= 1, matching the
  V4 ``pkt_len_std`` convention.
* Percentiles use numpy's default linear interpolation, pinned explicitly.
* The 21 features are ``replace([inf,-inf], nan).fillna(0.0)`` before they
  leave ``build_packet_windows``, exactly as V4 does for its 26.
=============================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- must match src/pcap_ingest_v4.py / src/feature_contract.py ---
WINDOW_SECONDS = 10
SESSION_IDLE_TIMEOUT_S = 120.0   # CICFlowMeter FlowTimeout, reused for session scope
RELATIVE_EPSILON = 1e-8
RELATIVE_CLIP = 10.0
ENTROPY_EPSILON = 1e-12
PERCENTILE_METHOD = "linear"     # numpy default, pinned so both paths agree

SEQ_SPACE = 1 << 32              # TCP sequence numbers are 32-bit modular

BUILDER_VERSION = "v5.0.0"


# =============================================================
# FEATURE NAMES — the authoritative order of the 21 + 9
# =============================================================

TTL_FEATURES = [
    "ttl_mean",
    "ttl_std",
    "ttl_min",
    "ttl_max",
    "log_ttl_unique_count",
]

TCP_WINDOW_FEATURES = [
    "log_tcp_window_mean",
    "log_tcp_window_std",
    "log_tcp_window_max",
    "tcp_window_zero_ratio",
    "tcp_window_change_rate",
]

FRAGMENTATION_FEATURES = [
    "log_frag_packet_count",
    "frag_packet_ratio",
    "ip_df_ratio",
]

RETRANSMISSION_FEATURES = [
    "log_tcp_retrans_count",
    "tcp_retrans_ratio",
    "log_tcp_retrans_bytes",
]

PAYLOAD_FEATURES = [
    "payload_len_p50",
    "payload_len_p90",
    "payload_len_max",
    "payload_zero_ratio",
    "payload_len_entropy",
]

PACKET_BASE_FEATURES = (
    TTL_FEATURES
    + TCP_WINDOW_FEATURES
    + FRAGMENTATION_FEATURES
    + RETRANSMISSION_FEATURES
    + PAYLOAD_FEATURES
)

PACKET_FAMILY = {}
for _fam, _names in (
    ("ttl", TTL_FEATURES),
    ("tcp_window", TCP_WINDOW_FEATURES),
    ("fragmentation", FRAGMENTATION_FEATURES),
    ("retransmission", RETRANSMISSION_FEATURES),
    ("payload", PAYLOAD_FEATURES),
):
    for _n in _names:
        PACKET_FAMILY[_n] = _fam

# Part 12 — which NEW base features receive a relative-change companion.
# This is a DESIGN decision recorded here once; the rationale for every
# entry (included AND excluded) lives in the contract and in
# results/v5_features/feature_design_candidates.csv.
PACKET_RELATIVE_CHANGE_BASE = [
    "ttl_mean",
    "ttl_std",
    "log_ttl_unique_count",
    "log_tcp_window_mean",
    "log_tcp_window_std",
    "log_tcp_retrans_count",
    "payload_len_p50",
    "payload_len_p90",
    "payload_len_entropy",
]
PACKET_RELATIVE_FEATURES = [
    f"{name}_rel_change" for name in PACKET_RELATIVE_CHANGE_BASE
]

PACKET_BASE_COUNT = len(PACKET_BASE_FEATURES)            # 21
PACKET_RELATIVE_COUNT = len(PACKET_RELATIVE_FEATURES)    # 9

assert PACKET_BASE_COUNT == 21, PACKET_BASE_COUNT
assert PACKET_RELATIVE_COUNT == 9, PACKET_RELATIVE_COUNT
assert len(set(PACKET_BASE_FEATURES)) == PACKET_BASE_COUNT
assert all(name in PACKET_BASE_FEATURES for name in PACKET_RELATIVE_CHANGE_BASE)

# Per-packet table schema produced by parse_pcap_packets / accepted by
# annotate_packets. Anything downstream reads ONLY these columns.
PACKET_COLUMNS = [
    "timestamp_abs",     # float, absolute epoch seconds (UTC, from the capture)
    "packet_index",      # int, CAPTURE-ORDER ordinal; breaks timestamp ties deterministically
    "ip_version",        # 4 or 6
    "src_ip", "dst_ip",  # str
    "src_port", "dst_port",   # int, 0 when there is no L4 port
    "protocol",          # int, IPv4 proto / IPv6 next-header
    "ttl",               # int 0-255: IPv4 TTL or IPv6 Hop Limit (pooled, documented)
    "ip_df",             # bool, IPv4 Don't-Fragment; False for IPv6 (no DF bit exists)
    "ip_mf",             # bool, IPv4 More-Fragments / IPv6 fragment-header M flag
    "ip_frag_offset",    # int, fragment offset in 8-byte units (0 when not a fragment)
    "is_fragment",       # bool, ip_mf or ip_frag_offset > 0
    "is_tcp",            # bool, a parsable TCP header is present
    "tcp_window",        # int, RAW 16-bit header value (NOT window-scaled)
    "tcp_seq",           # int, 32-bit sequence number (0 when not TCP)
    "tcp_syn", "tcp_fin",  # bool
    "payload_len",       # int, L4 payload bytes from header arithmetic
]


# =============================================================
# SMALL NUMERICAL PRIMITIVES
# =============================================================

def _safe_log1p(value):
    """V5 log transform. Same shape as the V4 ``safe_log1p``: non-finite
    -> 0, negatives clipped to 0, then log1p. Applied to counts and to the
    heavy-tailed TCP-window statistics."""
    arr = np.asarray(value, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.log1p(np.clip(arr, 0.0, None))


def _ratio(numerator: float, denominator: float) -> float:
    """Exact-integer-denominator ratio. Returns 0.0 when the denominator is
    zero — NOT an epsilon-smeared value — so 'no eligible packets' is
    distinguishable in the tests from 'a genuinely tiny ratio'."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1); 0.0 for n <= 1. Matches the V4
    ``pkt_len_std`` convention so the two representations describe spread
    the same way."""
    if values.size <= 1:
        return 0.0
    return float(np.std(values.astype(float), ddof=1))


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(float), q, method=PERCENTILE_METHOD))


def _entropy_from_counts(values) -> float:
    """Shannon entropy (base 2) of the EMPIRICAL DISTRIBUTION OF VALUES.

    Used here on the integer payload LENGTHS, so this is a payload-LENGTH
    distribution entropy. It is NOT byte-value entropy: no payload byte is
    ever read by this module (Part 7). Returns 0.0 when there is at most one
    distinct length. Identical formula to the V4 ``entropy_from_counts``
    used for dst_port / protocol, including the 1e-12 inside the log.
    """
    counts = pd.Series(values).value_counts(dropna=True)
    if len(counts) <= 1:
        return 0.0
    p = counts.to_numpy(dtype=float) / counts.sum()
    return float(-np.sum(p * np.log2(p + ENTROPY_EPSILON)))


def ensure_packet_order(packet_df: pd.DataFrame) -> pd.DataFrame:
    """Return the packet table in canonical order: sorted by
    ``(timestamp_abs, packet_index)``.

    A table that has no ``packet_index`` gets one from its CURRENT row order,
    which is documented as meaning "the caller is presenting packets in
    capture order". A table that has one keeps it, so the ordering survives a
    CSV/parquet round-trip and a row shuffle — which is exactly what makes the
    training-side and PCAP-side routes produce identical output.
    """
    if packet_df is None or len(packet_df) == 0:
        return packet_df
    df = packet_df
    if "packet_index" not in df.columns or df["packet_index"].isna().any():
        df = df.assign(packet_index=np.arange(len(df), dtype=np.int64))
    out = df.sort_values(["timestamp_abs", "packet_index"], kind="stable").reset_index(drop=True)
    for key, value in packet_df.attrs.items():
        out.attrs[key] = value
    return out


def _seq_diff(a: int, b: int) -> int:
    """Signed 32-bit modular difference a - b, the standard TCP sequence
    comparison. Handles wraparound: a sequence that has wrapped past 2^32
    is correctly ordered relative to one that has not."""
    d = (int(a) - int(b)) % SEQ_SPACE
    return d - SEQ_SPACE if d >= (SEQ_SPACE >> 1) else d


# =============================================================
# STAGE 1 — PER-PACKET HEADER TABLE
# =============================================================

def parse_pcap_packets(pcap_path, limit: int | None = None) -> pd.DataFrame:
    """Stream a capture and emit one row per IP packet with the header
    fields the V5 packet families need.

    Extends — never replaces — what ``pcap_ingest_v4.parse_pcap_streaming``
    reads. The V4 parser is left exactly as frozen; this is a separate
    reader so that no V4 feature value can shift.

    Field notes that matter for correctness:

    ``ttl``      IPv4 ``ttl`` and IPv6 ``hlim`` are POOLED into one column.
                 Both are 8-bit remaining-hop-budget counters with identical
                 semantics and range, so pooling is sound; it is recorded
                 explicitly in the contract because the task requires the
                 choice to be visible. Non-IP packets are dropped before
                 this point and contribute to nothing.

    ``tcp_window`` is the RAW 16-bit header field. The TCP window-scale
                 option is NOT applied, because the scale factor is only
                 observable when the SYN handshake is inside the capture,
                 which is false for every mid-capture connection. Applying
                 it to some connections and not others would make the
                 feature inconsistent across windows; leaving it raw makes
                 it consistently a header-level measurement.

    ``payload_len`` comes from HEADER ARITHMETIC (IP total length minus IP
                 header minus L4 header), not from ``len(layer.payload)``.
                 The latter — which V4 uses — silently includes Ethernet
                 padding on sub-60-byte frames. V5 needs the true segment
                 length because it also drives TCP sequence accounting.
                 Falls back to ``len(layer.payload)`` when the IP total
                 length is unusable (0 under TSO/LRO offload, or negative
                 arithmetic from a truncated snaplen).

    Non-first IP fragments carry no L4 header, so ``is_tcp`` is False for
    them and they contribute only to the fragmentation and TTL families.
    """
    from scapy.all import IP, IPv6, TCP, UDP, PcapReader
    from scapy.all import conf as scapy_conf
    from scapy.layers.inet6 import IPv6ExtHdrFragment

    scapy_conf.verb = 0

    rows = []
    n_total = 0
    n_non_ip = 0

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            n_total += 1
            if limit is not None and n_total > limit:
                n_total -= 1
                break

            ip4 = pkt.getlayer(IP)
            ip6 = pkt.getlayer(IPv6)
            if ip4 is None and ip6 is None:
                n_non_ip += 1
                continue

            if ip4 is not None:
                l3 = ip4
                ip_version = 4
                ttl = int(ip4.ttl)
                proto = int(ip4.proto)
                flags = str(ip4.flags)
                ip_df = "DF" in flags
                ip_mf = "MF" in flags
                frag_offset = int(ip4.frag)
                ihl = int(ip4.ihl) * 4
                total_len = int(ip4.len)
                ip_payload_len = total_len - ihl if total_len > 0 else -1
            else:
                l3 = ip6
                ip_version = 6
                ttl = int(ip6.hlim)
                proto = int(ip6.nh)
                # IPv6 has no DF bit: the header is never fragmented in
                # transit, so DF is conceptually always set. It is recorded
                # as False and excluded from ip_df_ratio's denominator
                # (which is IPv4-only) rather than being invented.
                ip_df = False
                frag_hdr = pkt.getlayer(IPv6ExtHdrFragment)
                if frag_hdr is not None:
                    ip_mf = bool(int(frag_hdr.m))
                    frag_offset = int(frag_hdr.offset)
                    proto = int(frag_hdr.nh)
                else:
                    ip_mf = False
                    frag_offset = 0
                ip_payload_len = int(ip6.plen) if int(ip6.plen) > 0 else -1

            is_fragment = bool(ip_mf or frag_offset > 0)

            src_port = 0
            dst_port = 0
            is_tcp = False
            tcp_window = 0
            tcp_seq = 0
            tcp_syn = False
            tcp_fin = False

            tcp = pkt.getlayer(TCP)
            udp = pkt.getlayer(UDP)

            if tcp is not None:
                is_tcp = True
                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)
                tcp_window = int(tcp.window)
                tcp_seq = int(tcp.seq)
                flag_str = str(tcp.flags)
                tcp_syn = "S" in flag_str
                tcp_fin = "F" in flag_str
                hdr = int(tcp.dataofs) * 4
                payload_len = ip_payload_len - hdr if ip_payload_len >= 0 else -1
                if payload_len < 0:
                    payload_len = len(tcp.payload)
            elif udp is not None:
                src_port = int(udp.sport)
                dst_port = int(udp.dport)
                payload_len = int(udp.len) - 8 if int(udp.len) >= 8 else -1
                if payload_len < 0:
                    payload_len = len(udp.payload)
            else:
                payload_len = ip_payload_len if ip_payload_len >= 0 else len(l3.payload)

            rows.append(
                (
                    float(pkt.time), len(rows), ip_version, str(l3.src), str(l3.dst),
                    src_port, dst_port, proto, ttl,
                    ip_df, ip_mf, frag_offset, is_fragment,
                    is_tcp, tcp_window, tcp_seq, tcp_syn, tcp_fin,
                    max(int(payload_len), 0),
                )
            )

    df = pd.DataFrame(rows, columns=PACKET_COLUMNS)
    if len(df) == 0:
        df.attrs["n_frames"] = n_total
        df.attrs["n_non_ip"] = n_non_ip
        return df

    df = ensure_packet_order(df)
    df.attrs["n_frames"] = n_total
    df.attrs["n_non_ip"] = n_non_ip
    df.attrs["t0_epoch"] = float(df["timestamp_abs"].min())
    return df


# =============================================================
# STAGE 2 — CAUSAL PER-DIRECTION SESSION ANNOTATION
# =============================================================

def annotate_packets(packet_df: pd.DataFrame) -> pd.DataFrame:
    """Add the three annotations that need cross-packet session state.

    Emitted columns
    ---------------
    ``is_retransmission``   bool
    ``retrans_bytes``       int, ``payload_len`` when retransmitted else 0
    ``seq_bearing``         bool, the denominator population for the
                            retransmission ratio
    ``window_change_eligible`` bool, the denominator population for the
                            window change rate
    ``tcp_window_changed``  bool, the numerator population

    SESSION SCOPE
    -------------
    A "directional session" is the ORDERED 5-tuple
    ``(src_ip, src_port, dst_ip, dst_port, protocol)`` — A->B and B->A are
    two separate sessions, which is what TCP sequence space actually is
    (each endpoint has its own independent sequence space). A session is
    reset after an idle gap longer than ``SESSION_IDLE_TIMEOUT_S`` (120 s),
    matching the flow-segmentation timeout that
    ``pcap_ingest_v4.reconstruct_flows`` already uses, so a reused ephemeral
    port with a fresh ISN does not read as a retransmission.

    RETRANSMISSION RULE (Part 6)
    ----------------------------
    Let ``eff_len = payload_len + (1 if SYN) + (1 if FIN)`` — the number of
    sequence numbers the segment consumes, which is exactly TCP's own
    definition. A packet is a retransmission when ALL of:

      1. it is TCP,
      2. ``eff_len > 0``,
      3. it is not the first ``eff_len > 0`` packet of its directional
         session, and
      4. ``seq + eff_len`` is at or below the highest sequence number the
         session has already transmitted (32-bit modular comparison).

    Condition 2 is the important exclusion: a zero-length packet (pure ACK,
    window update, bare RST, keepalive) is NEVER counted. Duplicate pure
    ACKs are duplicate ACKs, not retransmissions, and conflating them is the
    most common way this metric is made misleading. SYN and FIN each consume
    one sequence number, so SYN retransmissions and FIN retransmissions ARE
    detected, which is the behaviour that matters for scan/flood traffic.

    Overlapping (partial) retransmissions are counted ONLY when the segment
    is fully re-covered by already-transmitted sequence space; a segment
    that extends past the high-water mark is treated as new data. This is
    the conservative direction — it under-counts rather than over-counts.

    KNOWN LIMITATION, stated rather than hidden: a genuinely out-of-order
    packet is indistinguishable from a retransmission under a high-water-
    mark rule, because both re-cover already-seen sequence space. Wireshark
    separates them using RTT and duplicate-ACK context that a per-window
    streaming builder cannot afford. The feature therefore measures
    "sequence-space re-coverage", which is retransmission plus reordering,
    and the contract names it that way.

    TCP WINDOW CHANGE (Part 4)
    --------------------------
    ``window_change_eligible`` is True for a TCP packet that HAS a
    predecessor in the same directional session (i.e. it is not the first).
    ``tcp_window_changed`` is True when that packet's raw window value
    differs from its predecessor's. Both are attributed to the window of
    the LATER packet, so a pair straddling a window boundary is counted in
    the later window and never in the earlier one — causal by construction.
    """
    empty = pd.DataFrame(
        columns=list(packet_df.columns) + [
            "is_retransmission", "retrans_bytes", "seq_bearing",
            "window_change_eligible", "tcp_window_changed",
        ]
    )
    if packet_df is None or len(packet_df) == 0:
        return empty

    df = ensure_packet_order(packet_df)
    n = len(df)

    ts = df["timestamp_abs"].to_numpy(dtype=float)
    is_tcp = df["is_tcp"].to_numpy(dtype=bool)
    seq = df["tcp_seq"].to_numpy(dtype=np.int64)
    syn = df["tcp_syn"].to_numpy(dtype=bool)
    fin = df["tcp_fin"].to_numpy(dtype=bool)
    win = df["tcp_window"].to_numpy(dtype=np.int64)
    pay = df["payload_len"].to_numpy(dtype=np.int64)

    # Directional key: ORDERED 5-tuple.
    dir_key = (
        df["src_ip"].astype(str) + ">" + df["dst_ip"].astype(str) + "|"
        + df["src_port"].astype(str) + ">" + df["dst_port"].astype(str) + "|"
        + df["protocol"].astype(str)
    ).to_numpy()

    is_retrans = np.zeros(n, dtype=bool)
    seq_bearing = np.zeros(n, dtype=bool)
    win_eligible = np.zeros(n, dtype=bool)
    win_changed = np.zeros(n, dtype=bool)

    # session key -> [max_seq_end, last_ts, last_window, has_seq_bearing]
    sessions: dict[str, list] = {}

    for i in range(n):
        if not is_tcp[i]:
            continue

        key = dir_key[i]
        state = sessions.get(key)
        if state is not None and (ts[i] - state[1]) > SESSION_IDLE_TIMEOUT_S:
            state = None          # idle timeout -> treat as a fresh session

        eff_len = int(pay[i]) + (1 if syn[i] else 0) + (1 if fin[i] else 0)
        bears_seq = eff_len > 0
        seq_bearing[i] = bears_seq

        if state is None:
            sessions[key] = [
                (int(seq[i]) + eff_len) % SEQ_SPACE if bears_seq else 0,
                float(ts[i]),
                int(win[i]),
                bool(bears_seq),
            ]
            continue

        # --- TCP window change (any TCP packet with a predecessor) ---
        win_eligible[i] = True
        if int(win[i]) != state[2]:
            win_changed[i] = True

        # --- retransmission (seq-bearing packets only) ---
        if bears_seq:
            seq_end = (int(seq[i]) + eff_len) % SEQ_SPACE
            if not state[3]:
                # first seq-bearing packet of the session: establishes the
                # high-water mark, never a retransmission
                state[0] = seq_end
                state[3] = True
            elif _seq_diff(seq_end, state[0]) <= 0:
                is_retrans[i] = True
            else:
                state[0] = seq_end

        state[1] = float(ts[i])
        state[2] = int(win[i])

    out = df.copy()      # NOTE: canonically ordered, not the caller's order
    out["is_retransmission"] = is_retrans
    out["retrans_bytes"] = np.where(is_retrans, pay, 0).astype(np.int64)
    out["seq_bearing"] = seq_bearing
    out["window_change_eligible"] = win_eligible
    out["tcp_window_changed"] = win_changed
    for key, value in packet_df.attrs.items():
        out.attrs[key] = value
    return out


# =============================================================
# STAGE 3 — 10 s WINDOW AGGREGATION -> THE 21 PACKET FEATURES
# =============================================================

def _window_feature_row(g: pd.DataFrame) -> dict:
    """The 21 packet features for ONE 10 s window. Every formula lives here
    exactly once; docs/v5_packet_feature_contract.json quotes this code."""
    n_ip = len(g)

    ttl = g["ttl"].to_numpy(dtype=np.int64)
    tcp_mask = g["is_tcp"].to_numpy(dtype=bool)
    n_tcp = int(tcp_mask.sum())
    pay = g["payload_len"].to_numpy(dtype=np.int64)

    # ---------- TTL (all IP packets, IPv4 TTL and IPv6 Hop Limit pooled) ----------
    row = {
        "ttl_mean": float(ttl.mean()) if n_ip else 0.0,
        "ttl_std": _std(ttl),
        "ttl_min": float(ttl.min()) if n_ip else 0.0,
        "ttl_max": float(ttl.max()) if n_ip else 0.0,
        "log_ttl_unique_count": float(_safe_log1p(len(np.unique(ttl)) if n_ip else 0)),
    }

    # ---------- TCP window (TCP packets only; 0.0 when there is no TCP) ----------
    if n_tcp:
        win = g["tcp_window"].to_numpy(dtype=np.int64)[tcp_mask]
        row["log_tcp_window_mean"] = float(_safe_log1p(win.mean()))
        row["log_tcp_window_std"] = float(_safe_log1p(_std(win)))
        row["log_tcp_window_max"] = float(_safe_log1p(win.max()))
        row["tcp_window_zero_ratio"] = _ratio(int((win == 0).sum()), n_tcp)
    else:
        row["log_tcp_window_mean"] = 0.0
        row["log_tcp_window_std"] = 0.0
        row["log_tcp_window_max"] = 0.0
        row["tcp_window_zero_ratio"] = 0.0

    n_win_eligible = int(g["window_change_eligible"].to_numpy(dtype=bool).sum())
    n_win_changed = int(g["tcp_window_changed"].to_numpy(dtype=bool).sum())
    row["tcp_window_change_rate"] = _ratio(n_win_changed, n_win_eligible)

    # ---------- IP fragmentation ----------
    n_frag = int(g["is_fragment"].to_numpy(dtype=bool).sum())
    row["log_frag_packet_count"] = float(_safe_log1p(n_frag))
    row["frag_packet_ratio"] = _ratio(n_frag, n_ip)
    ip4_mask = (g["ip_version"].to_numpy(dtype=np.int64) == 4)
    n_ip4 = int(ip4_mask.sum())
    n_df = int((g["ip_df"].to_numpy(dtype=bool) & ip4_mask).sum())
    row["ip_df_ratio"] = _ratio(n_df, n_ip4)

    # ---------- TCP retransmission ----------
    n_retrans = int(g["is_retransmission"].to_numpy(dtype=bool).sum())
    n_seq_bearing = int(g["seq_bearing"].to_numpy(dtype=bool).sum())
    row["log_tcp_retrans_count"] = float(_safe_log1p(n_retrans))
    row["tcp_retrans_ratio"] = _ratio(n_retrans, n_seq_bearing)
    row["log_tcp_retrans_bytes"] = float(
        _safe_log1p(int(g["retrans_bytes"].to_numpy(dtype=np.int64).sum()))
    )

    # ---------- payload-length distribution (metadata only, no bytes read) ----------
    row["payload_len_p50"] = _percentile(pay, 50)
    row["payload_len_p90"] = _percentile(pay, 90)
    row["payload_len_max"] = float(pay.max()) if n_ip else 0.0
    row["payload_zero_ratio"] = _ratio(int((pay == 0).sum()), n_ip)
    row["payload_len_entropy"] = _entropy_from_counts(pay)

    return row


def build_packet_windows(
    annotated_df: pd.DataFrame,
    window_seconds: int = WINDOW_SECONDS,
    win0: int | None = None,
) -> pd.DataFrame:
    """Aggregate annotated packets into the 21 packet features per 10 s
    window. Returns columns ``[window_idx, timestamp, packet_count] +
    PACKET_BASE_FEATURES``, one row per NON-EMPTY window, sorted.

    ``window_idx`` is ``floor(ts / window_seconds) - win0``. ``win0``
    defaults to the bin of the capture's first packet, which reproduces the
    re-basing ``pcap_ingest_v4.reconstruct_flows`` applies, so the two
    tables join on a common index. Pass ``win0`` explicitly when merging
    captures that must share an absolute origin.
    """
    if annotated_df is None or len(annotated_df) == 0:
        return pd.DataFrame(
            columns=["window_idx", "timestamp", "packet_count"] + PACKET_BASE_FEATURES
        )

    df = ensure_packet_order(annotated_df)
    ts = df["timestamp_abs"].to_numpy(dtype=float)
    abs_bin = np.floor(ts / window_seconds).astype(np.int64)
    if win0 is None:
        win0 = int(np.floor(float(ts.min()) / window_seconds))
    df = df.assign(window_idx=abs_bin - int(win0))

    rows = []
    for window_idx, g in df.groupby("window_idx", sort=True):
        row = {"window_idx": int(window_idx), "packet_count": int(len(g))}
        row.update(_window_feature_row(g))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["timestamp"] = out["window_idx"] * window_seconds
    out = out.reindex(
        columns=["window_idx", "timestamp", "packet_count"] + PACKET_BASE_FEATURES
    )
    out[PACKET_BASE_FEATURES] = (
        out[PACKET_BASE_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )
    out = out.sort_values("window_idx").reset_index(drop=True)
    out.attrs["win0"] = int(win0)
    for key, value in annotated_df.attrs.items():
        out.attrs.setdefault(key, value)
    return out


# =============================================================
# STAGE 4 — RELATIVE CHANGES FOR THE 9 SELECTED PACKET FEATURES
# =============================================================

def add_packet_relative_changes(window_df: pd.DataFrame) -> pd.DataFrame:
    """Append the 9 packet relative-change features.

    The transform is BYTE-FOR-BYTE the V4 definition
    (``prepare_multiday_benchmark.load_and_prep_day`` /
    ``pcap_ingest_v4.add_relative_changes`` /
    ``feature_contract.reference_relative_change``):

        r_t = (s_t - s_{t-1}) / (|s_{t-1}| + 1e-8)
        first row forced to exactly 0.0
        non-finite -> 0.0
        clip to [-10, +10]

    "previous window" means the previous EXISTING ROW of this table — not
    necessarily t-10 s, because empty windows produce no row. That is the
    same rule V4 uses, and the downstream 10-second contiguity gate is what
    stops a sequence from spanning such a gap. Reset scope is one capture.

    Only the 9 features in ``PACKET_RELATIVE_CHANGE_BASE`` get a companion;
    the other 12 are deliberately excluded (Part 12), each with a reason
    recorded in the contract.
    """
    df = window_df.copy().reset_index(drop=True)
    if len(df) == 0:
        for name in PACKET_RELATIVE_FEATURES:
            df[name] = pd.Series(dtype=float)
        return df

    state = df[PACKET_RELATIVE_CHANGE_BASE].astype(float)
    delta = state.diff().fillna(0.0)
    previous = state.shift(1)
    relative = delta / (previous.abs() + RELATIVE_EPSILON)
    relative.iloc[0] = 0.0
    relative = (
        relative.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-RELATIVE_CLIP, RELATIVE_CLIP)
    )
    relative.columns = PACKET_RELATIVE_FEATURES
    out = pd.concat([df, relative], axis=1)
    for key, value in window_df.attrs.items():
        out.attrs[key] = value
    return out


# =============================================================
# CONVENIENCE: the whole packet lane in one call
# =============================================================

def packet_features_from_pcap(
    pcap_path,
    limit: int | None = None,
    win0: int | None = None,
) -> pd.DataFrame:
    """parse -> annotate -> window -> relative change. The function both
    the training-side and the PCAP-side entry points call, so there is
    exactly one code path for the packet lane."""
    packets = parse_pcap_packets(pcap_path, limit=limit)
    annotated = annotate_packets(packets)
    windows = build_packet_windows(annotated, win0=win0)
    return add_packet_relative_changes(windows)


def packet_features_from_packet_table(
    packet_df: pd.DataFrame,
    win0: int | None = None,
) -> pd.DataFrame:
    """Same lane, entered from an already-materialised per-packet table
    (the route a training-side builder takes when packets arrive from
    storage rather than from a live capture)."""
    annotated = annotate_packets(packet_df)
    windows = build_packet_windows(annotated, win0=win0)
    return add_packet_relative_changes(windows)
