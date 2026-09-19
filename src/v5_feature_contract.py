"""
=============================================================
V5 PACKET-AWARE FEATURE CONTRACT — canonical, descriptive
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

The machine-readable definition of the 82-dimensional V5 experimental
network state.

    V5 input  =  V4 input (52, FROZEN, byte-identical prefix)
               + 21 NEW packet base features
               +  9 NEW packet relative-change features
              =  82

The V4 block is a PREFIX, not a reshuffle. ``v5_input[:52]`` is exactly the
vector the frozen V4 model consumes, in exactly the V4 order, produced by
exactly the V4 code. That makes "V4 is unchanged" a numerically checkable
claim rather than a promise, and it lets a V4-vs-V5 comparison isolate the
effect of the packet block.

This module is DESCRIPTIVE, like ``src/feature_contract.py`` for V4. It
rewires nothing:
  * the 26 flow features still come only from
    ``archive/create_network_states_v2.py`` (training) and
    ``src/pcap_ingest_v4.py`` (PCAP), both untouched;
  * the 21 packet features come only from
    ``src/packet_feature_builder.py``, which is the single shared
    implementation for both paths.

Nothing here reads, writes or hashes any V4 artefact.
=============================================================
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import feature_contract as v4c            # noqa: E402  (V4 contract, read-only)
import packet_feature_builder as pfb      # noqa: E402

SCHEMA_VERSION = "v5"
CONTRACT_VERSION = "1.0.0"

WINDOW_SECONDS = pfb.WINDOW_SECONDS
SEQUENCE_LENGTH = v4c.SEQUENCE_LENGTH

V4_FEATURE_NAMES = list(v4c.INPUT_FEATURE_NAMES)          # 52, frozen order
V4_BASE_FEATURES = list(v4c.BASE_FEATURES)                # 26
V4_RELATIVE_FEATURES = list(v4c.RELATIVE_FEATURES)        # 26

PACKET_BASE_FEATURES = list(pfb.PACKET_BASE_FEATURES)     # 21
PACKET_RELATIVE_FEATURES = list(pfb.PACKET_RELATIVE_FEATURES)   # 9

V5_FEATURE_NAMES = (
    V4_FEATURE_NAMES + PACKET_BASE_FEATURES + PACKET_RELATIVE_FEATURES
)
V5_DIMENSION = len(V5_FEATURE_NAMES)                      # 82

assert V5_DIMENSION == 82, V5_DIMENSION
assert V5_FEATURE_NAMES[:52] == V4_FEATURE_NAMES
assert len(set(V5_FEATURE_NAMES)) == V5_DIMENSION, "duplicate feature name"

CONTRACT_PATH = REPO_ROOT / "docs" / "v5_packet_feature_contract.json"
SCHEMA_PATH = REPO_ROOT / "docs" / "v5_feature_schema.json"
HASH_PATH = REPO_ROOT / "results" / "v5_features" / "v5_feature_contract_sha256.txt"


# =============================================================
# PACKET FEATURE DEFINITIONS
# =============================================================
# Every tuple below is:
#   (name, source_fields, aggregation, formula, unit, domain,
#    missing_rule, zero_behaviour, log_transformed, notes)
#
# `formula` quotes the actual implementation in
# src/packet_feature_builder.py, not a paraphrase of the name.

_NO_IP = "the window does not exist — a window with no IP packets produces no row at all"

_PACKET_DEFS = [
    # ---------------- TTL ----------------
    (
        "ttl_mean",
        ["IPv4.ttl", "IPv6.hlim"],
        "mean over all IP packets in the window",
        "float(ttl.mean()) over every IP packet whose timestamp falls in [t, t+10)",
        "hops (remaining hop budget, 8-bit)",
        "[0, 255]",
        _NO_IP,
        "0.0 is only reachable if every packet carried TTL 0, which a forwarded packet cannot; "
        "in practice a 0 here means the window was empty and no row was emitted",
        False,
        "IPv4 TTL and IPv6 Hop Limit are POOLED. Both are 8-bit remaining-hop counters with "
        "identical semantics, range and decrement rule, so a pooled statistic is well defined. "
        "The choice is recorded because it is a semantic merge, not an implementation detail; "
        "IPv6 is 0.05% of packets in the available captures.",
    ),
    (
        "ttl_std",
        ["IPv4.ttl", "IPv6.hlim"],
        "sample standard deviation (ddof=1) over all IP packets in the window",
        "np.std(ttl, ddof=1) for n > 1, else 0.0",
        "hops",
        "[0, ~127]",
        _NO_IP,
        "EXACTLY 0.0 for a constant-TTL window (all packets from one hop distance) and for a "
        "single-packet window. These two cases are not distinguishable from each other; "
        "packet_count is carried alongside so a consumer can separate them.",
        False,
        "ddof=1 matches the V4 std_pkt_len convention.",
    ),
    (
        "ttl_min",
        ["IPv4.ttl", "IPv6.hlim"],
        "min over all IP packets in the window",
        "float(ttl.min()) over every IP packet whose timestamp falls in [t, t+10)",
        "hops",
        "[0, 255]",
        _NO_IP,
        "0.0 only for an empty window (no row emitted)",
        False,
        "Order statistic: tracks the most distant observed source. Single-packet sensitive.",
    ),
    (
        "ttl_max",
        ["IPv4.ttl", "IPv6.hlim"],
        "max over all IP packets in the window",
        "float(ttl.max()) over every IP packet whose timestamp falls in [t, t+10)",
        "hops",
        "[0, 255]",
        _NO_IP,
        "0.0 only for an empty window (no row emitted)",
        False,
        "Order statistic: tracks the nearest / locally originated traffic.",
    ),
    (
        "log_ttl_unique_count",
        ["IPv4.ttl", "IPv6.hlim"],
        "log1p of the number of distinct TTL values in the window",
        "_safe_log1p(len(np.unique(ttl)))",
        "log1p(distinct TTL values)",
        "[0, log1p(256)] = [0, 5.549]",
        _NO_IP,
        "log1p(1) = 0.693 for a constant-TTL window; a literal 0.0 means zero packets",
        True,
        "Diversity counterpart to ttl_mean. A surge in distinct TTLs indicates traffic arriving "
        "from many different hop distances in one window (scan-like fan-out, or spoofed sources).",
    ),
    # ---------------- TCP window ----------------
    (
        "log_tcp_window_mean",
        ["TCP.window"],
        "mean over TCP packets in the window, then log1p",
        "_safe_log1p(win.mean()) where win = raw 16-bit TCP window of every TCP packet in [t, t+10)",
        "log1p(bytes, unscaled header units)",
        "[0, log1p(65535)] = [0, 11.09]",
        "no TCP packets in the window -> 0.0",
        "0.0 means EITHER no TCP traffic OR every advertised window was 0. Ambiguous by design, "
        "matching the V4 fillna(0.0) policy; tcp_window_zero_ratio separates the two cases.",
        True,
        "RAW header value: the TCP window-scale option is deliberately NOT applied, because the "
        "scale factor is only observable when the SYN handshake is inside the capture. Applying it "
        "to some connections and not others would make the feature inconsistent between windows.",
    ),
    (
        "log_tcp_window_std",
        ["TCP.window"],
        "sample standard deviation (ddof=1) over TCP packets, then log1p",
        "_safe_log1p(np.std(win, ddof=1)) for n > 1, else _safe_log1p(0.0) = 0.0",
        "log1p(bytes)",
        "[0, ~10.9]",
        "no TCP packets -> 0.0",
        "0.0 for a constant-window window, a single-TCP-packet window, and a no-TCP window alike",
        True,
        "Dispersion of advertised receive windows — wide dispersion means many peers with "
        "different buffer states inside one 10 s bin.",
    ),
    (
        "log_tcp_window_max",
        ["TCP.window"],
        "max over TCP packets, then log1p",
        "_safe_log1p(win.max()) over the raw window field of every TCP packet in [t, t+10)",
        "log1p(bytes)",
        "[0, 11.09]",
        "no TCP packets -> 0.0",
        "0.0 means no TCP packets or an all-zero-window window",
        True,
        "Upper extent of advertised receive capacity.",
    ),
    (
        "tcp_window_zero_ratio",
        ["TCP.window"],
        "count(window == 0) / count(TCP packets), per window",
        "_ratio(int((win == 0).sum()), n_tcp); returns exactly 0.0 when n_tcp == 0",
        "dimensionless fraction",
        "[0, 1]",
        "no TCP packets -> 0.0",
        "0.0 means no zero-window packets OR no TCP at all",
        False,
        "Zero-window advertisements are explicit receiver flow-control stalls. Chosen over the raw "
        "tcp_window_zero_count because the count is confounded with window volume, which "
        "log_total_packets already carries.",
    ),
    (
        "tcp_window_change_rate",
        ["TCP.window", "ordered 5-tuple session"],
        "count(window value differs from the same directional session's previous packet) / "
        "count(TCP packets that HAVE such a predecessor), per window",
        "_ratio(n_win_changed, n_win_eligible) where a packet is 'eligible' iff it is TCP and not "
        "the first packet of its directional session (ordered 5-tuple, 120 s idle reset), and "
        "'changed' iff its raw window value differs from that predecessor's",
        "dimensionless fraction",
        "[0, 1]",
        "no eligible TCP packets -> 0.0",
        "0.0 means every advertised window was steady, or there was no eligible TCP packet",
        False,
        "EXACT DEFINITION, stated because 'change rate' is otherwise ambiguous: the denominator is "
        "TCP packets WITH a predecessor, NOT all TCP packets, so the first packet of every session "
        "is excluded from both numerator and denominator. A pair that straddles a window boundary "
        "is attributed to the LATER packet's window only — never to the earlier one — which keeps "
        "the feature causal.",
    ),
    # ---------------- Fragmentation ----------------
    (
        "log_frag_packet_count",
        ["IPv4.flags.MF", "IPv4.frag", "IPv6ExtHdrFragment.m", "IPv6ExtHdrFragment.offset"],
        "log1p of the number of fragmented IP packets in the window",
        "_safe_log1p(sum(is_fragment)) where is_fragment = (MF set) OR (fragment offset > 0)",
        "log1p(packets)",
        "[0, inf)",
        _NO_IP,
        "0.0 for an unfragmented window — which is EVERY window in every capture available to this "
        "project (see the limitation note)",
        True,
        "Fragment recognition follows RFC 791 exactly, NOT 'any nonzero flag bit': the DF bit is "
        "nonzero on 96% of real packets and means the opposite of fragmentation. "
        "first fragment = MF set AND offset == 0; middle fragment = MF set AND offset > 0; "
        "last fragment = MF clear AND offset > 0; unfragmented = MF clear AND offset == 0. "
        "IPv6 routers never fragment, so IPv6 fragmentation is read only from the Fragment "
        "extension header (M flag and offset), and an IPv6 packet without that header is "
        "unfragmented by definition.",
    ),
    (
        "frag_packet_ratio",
        ["IPv4.flags.MF", "IPv4.frag", "IPv6ExtHdrFragment"],
        "count(fragmented IP packets) / count(IP packets), per window",
        "_ratio(n_frag, n_ip) — fragmented IP packets over all IP packets in [t, t+10)",
        "dimensionless fraction",
        "[0, 1]",
        _NO_IP,
        "0.0 for an unfragmented window; identically 0.0 in all available captures",
        False,
        "Scale-free companion to log_frag_packet_count: the count detects a fragmentation flood, "
        "the ratio detects a small but wholly fragmented conversation.",
    ),
    (
        "ip_df_ratio",
        ["IPv4.flags.DF"],
        "count(IPv4 packets with DF set) / count(IPv4 packets), per window",
        "_ratio(n_df, n_ip4) — the denominator is IPv4 packets ONLY",
        "dimensionless fraction",
        "[0, 1]",
        "no IPv4 packets in the window -> 0.0",
        "0.0 means no DF-marked packet, or no IPv4 packet at all",
        False,
        "IPv6 is EXCLUDED from both numerator and denominator because IPv6 has no DF bit — "
        "routers may never fragment IPv6, so 'DF' is not a property an IPv6 packet can carry, and "
        "counting IPv6 as DF=0 would fabricate a value. This is the only fragmentation-family "
        "feature with real variance in the available captures (0.96 +/- 0.09 on real traffic).",
    ),
    # ---------------- Retransmission ----------------
    (
        "log_tcp_retrans_count",
        ["TCP.seq", "TCP.flags.SYN", "TCP.flags.FIN", "payload_len", "ordered 5-tuple session"],
        "log1p of the number of sequence-space re-covering TCP packets in the window",
        "_safe_log1p(sum(is_retransmission)); is_retransmission is True iff the packet is TCP, "
        "eff_len = payload_len + (1 if SYN) + (1 if FIN) > 0, the packet is not the first "
        "seq-bearing packet of its directional session, and seq + eff_len is at or below that "
        "session's running high-water mark under 32-bit modular comparison",
        "log1p(packets)",
        "[0, inf)",
        "no TCP packets -> 0.0",
        "0.0 means a clean window with no re-covered sequence space, or no TCP at all",
        True,
        "eff_len > 0 is the load-bearing exclusion: zero-length packets (pure ACK, window update, "
        "bare RST, keepalive) are NEVER counted, so duplicate ACKs are not mistaken for "
        "retransmissions. SYN and FIN each consume one sequence number, so SYN and FIN "
        "retransmissions ARE detected. Partial/overlapping retransmissions count only when the "
        "segment is wholly re-covered — the conservative direction. "
        "MEASURED LIMITATION: a genuinely out-of-order packet is indistinguishable from a "
        "retransmission under any high-water-mark rule, so this feature measures 'sequence-space "
        "re-coverage' = retransmission + reordering, and is named and documented as such rather "
        "than being claimed as a pure retransmission count.",
    ),
    (
        "tcp_retrans_ratio",
        ["TCP.seq", "payload_len", "TCP.flags"],
        "count(retransmissions) / count(seq-bearing TCP packets), per window",
        "_ratio(n_retrans, n_seq_bearing) where seq-bearing means eff_len > 0",
        "dimensionless fraction",
        "[0, 1]",
        "no seq-bearing TCP packets -> 0.0",
        "0.0 for a clean window or a window with no seq-bearing TCP packet",
        False,
        "The denominator is seq-bearing packets, NOT all TCP packets: dividing by a population "
        "that cannot be retransmitted by construction would make the ratio depend on the "
        "ACK/data mix instead of on retransmission behaviour.",
    ),
    (
        "log_tcp_retrans_bytes",
        ["TCP.seq", "payload_len"],
        "log1p of the total retransmitted payload bytes in the window",
        "_safe_log1p(sum(payload_len where is_retransmission))",
        "log1p(bytes)",
        "[0, inf)",
        "no TCP packets -> 0.0",
        "0.0 when nothing was retransmitted AND when only zero-payload SYN/FIN were retransmitted "
        "— a real and useful distinction from log_tcp_retrans_count, which is nonzero in the "
        "second case (control-plane retransmission with no data cost)",
        True,
        "Separates cheap control retransmission from expensive bulk retransmission.",
    ),
    # ---------------- Payload-length distribution ----------------
    (
        "payload_len_p50",
        ["payload_len (header arithmetic)"],
        "50th percentile over all IP packets in the window",
        "np.percentile(payload_len, 50, method='linear')",
        "bytes",
        "[0, 65535]",
        _NO_IP,
        "0.0 is COMMON and meaningful — it means at least half the packets in the window carried "
        "no payload (ACK-dominated traffic)",
        False,
        "Robust central tendency. Deliberately NOT a mean: V4's mean_pkt_len already carries a "
        "mean-of-flow-means of the same quantity, and a second mean would be near-collinear.",
    ),
    (
        "payload_len_p90",
        ["payload_len"],
        "90th percentile over all IP packets in the window",
        "np.percentile(payload_len, 90, method='linear')",
        "bytes",
        "[0, 65535]",
        _NO_IP,
        "0.0 means over 90% of packets were payload-free",
        False,
        "Upper-tail shape: separates bulk transfer from control chatter independently of the "
        "median. Together with p50 this is the 'richer payload-size distribution' the problem "
        "statement asks for.",
    ),
    (
        "payload_len_max",
        ["payload_len"],
        "max over all IP packets in the window",
        "float(payload_len.max()) over every IP packet in [t, t+10)",
        "bytes",
        "[0, 65535]",
        _NO_IP,
        "0.0 means no packet in the window carried any payload",
        False,
        "Can exceed the link MTU when the capture point sees TSO/LRO-coalesced segments (values "
        "up to 64240 are present in the real March-2 captures). That is a property of where the "
        "capture was taken, not a parsing error, and is left unclipped.",
    ),
    (
        "payload_zero_ratio",
        ["payload_len"],
        "count(payload_len == 0) / count(IP packets), per window",
        "_ratio(int((payload_len == 0).sum()), n_ip)",
        "dimensionless fraction",
        "[0, 1]",
        _NO_IP,
        "0.0 means every packet carried payload",
        False,
        "payload_nonzero_ratio is EXCLUDED as exactly 1 - payload_zero_ratio: an affine duplicate "
        "carries no information a linear model or a network cannot already reach.",
    ),
    (
        "payload_len_entropy",
        ["payload_len"],
        "Shannon entropy (base 2) of the empirical distribution of payload LENGTH VALUES",
        "-sum(p * log2(p + 1e-12)) over value_counts(payload_len) / total; 0.0 when <= 1 distinct "
        "length. Identical formula to the V4 dst_port_entropy / protocol_entropy.",
        "bits",
        "[0, log2(n_packets)]",
        _NO_IP,
        "EXACTLY 0.0 when every packet in the window has the same payload length — the classic "
        "fixed-size beacon signature — and also for a single-packet window",
        False,
        "THIS IS A LENGTH-DISTRIBUTION ENTROPY, NOT BYTE-VALUE SHANNON ENTROPY. No payload byte is "
        "ever read by any part of this pipeline; the only input is the integer length from the "
        "IP/L4 headers. The two quantities are routinely confused and measure different things: "
        "byte-value entropy detects encryption/compression of content, length entropy detects "
        "regularity of message sizes.",
    ),
]

assert [d[0] for d in _PACKET_DEFS] == PACKET_BASE_FEATURES, "contract/builder order drift"


# =============================================================
# RELATIVE-CHANGE DECISIONS (Part 12)
# =============================================================
# Every one of the 21 new base features gets an explicit ruling.

_RELATIVE_DECISIONS = {
    "ttl_mean": (True,
        "Denominator is a hop count that is never near zero in a non-empty window, so the ratio is "
        "numerically stable. A shift in mean hop distance between adjacent windows is exactly the "
        "'new peer population' signal the world model should see as a transition."),
    "ttl_std": (True,
        "Can have a zero denominator (constant-TTL window), where the ratio saturates the +/-10 "
        "clip. That saturation is retained deliberately: constant-TTL -> varied-TTL is a genuine "
        "regime change and the clipped value is a correct 'large jump' indicator, not a numerical "
        "accident. Same behaviour the frozen V4 contract already accepts for std_pkt_len."),
    "ttl_min": (False,
        "Extreme order statistic of a bounded 8-bit field. A single distant packet moves it, so its "
        "window-to-window ratio is noise-dominated; the informative content is its LEVEL, which the "
        "base feature already carries."),
    "ttl_max": (False,
        "Extreme order statistic of a bounded 8-bit field, and in practice pinned at the initial "
        "TTL of the nearest traffic source (64/128/255), so its window-to-window ratio is almost "
        "always 0 punctuated by single-packet artefacts. Its LEVEL is the informative part."),
    "log_ttl_unique_count": (True,
        "Diversity/volume-like and already log-compressed, so the relative change is a stable "
        "bounded quantity. A fan-out burst appears here before it appears in the mean."),
    "log_tcp_window_mean": (True,
        "Flow-control state is inherently about change: the level alone cannot distinguish a "
        "steadily small window from a window that has just collapsed."),
    "log_tcp_window_std": (True,
        "Log-compressed dispersion; the change captures the onset of heterogeneous peer behaviour."),
    "log_tcp_window_max": (False,
        "Extreme order statistic, in practice pinned at the largest advertising peer's constant "
        "value, so its relative change is ~0 with occasional single-packet spikes."),
    "tcp_window_zero_ratio": (False,
        "Bounded proportion that is exactly 0.0 in the large majority of windows. Its relative "
        "change is then 0/0 -> epsilon-driven clip saturation, which encodes the ARRIVAL of any "
        "zero-window packet as a constant +10 regardless of magnitude. The base ratio already "
        "conveys that arrival, with magnitude, and without the saturation artefact."),
    "tcp_window_change_rate": (False,
        "Already a first-order change quantity. Its relative change is a second difference of the "
        "window-size signal: unstable, hard to interpret, and not something the 10-window LSTM "
        "context needs pre-computed."),
    "log_frag_packet_count": (False,
        "Structurally zero in every window of every capture available to this project. A relative "
        "change over an all-zero series is identically 0.0 and would add a dead dimension."),
    "frag_packet_ratio": (False,
        "Structurally zero in every window of every capture available to this project, exactly as "
        "log_frag_packet_count is. A relative change over an all-zero series is identically 0.0, "
        "so including it would add a second dead dimension rather than a second signal."),
    "ip_df_ratio": (False,
        "Bounded proportion that sits near a near-constant level (0.96 +/- 0.09 on real traffic). "
        "Its relative change is a small-amplitude, near-zero-mean quantity dominated by the traffic "
        "mix rather than by any security-relevant transition."),
    "log_tcp_retrans_count": (True,
        "The one retransmission feature that earns a relative change: retransmission is bursty, "
        "and the onset of a burst (a congestion or scan event) is a transition the model should "
        "see directly. Well populated in the real captures (1.4% of seq-bearing packets, present "
        "in 24% of windows), so this is not an all-zero series."),
    "tcp_retrans_ratio": (False,
        "Bounded proportion, zero in most windows -> the same 0/0 clip-saturation problem as "
        "tcp_window_zero_ratio, and near-collinear with the count's relative change when nonzero."),
    "log_tcp_retrans_bytes": (False,
        "Near-collinear with log_tcp_retrans_count's relative change, because both move together "
        "whenever a data retransmission burst starts. Only the count's version is kept."),
    "payload_len_p50": (True,
        "Multiplicative shifts in typical message size are meaningful and the log-free byte scale "
        "makes the ratio directly interpretable."),
    "payload_len_p90": (True,
        "Upper-tail shifts mark the start and end of bulk transfer, which is precisely a "
        "transition event."),
    "payload_len_max": (False,
        "Extreme order statistic, usually pinned at the MTU or at a TSO segment size, so its "
        "relative change is ~0 punctuated by single-packet spikes."),
    "payload_zero_ratio": (False,
        "Bounded proportion whose level is the informative quantity; its relative change is a "
        "small bounded number that duplicates the level."),
    "payload_len_entropy": (True,
        "Consistent with the frozen V4 convention, which gives dst_port_entropy and "
        "protocol_entropy relative changes. An entropy collapse (traffic becoming uniformly sized, "
        "i.e. beacon-like) is a transition, not a level."),
}

assert set(_RELATIVE_DECISIONS) == set(PACKET_BASE_FEATURES)
assert [n for n, (inc, _) in _RELATIVE_DECISIONS.items() if inc] == \
    list(pfb.PACKET_RELATIVE_CHANGE_BASE), "relative-change decision/builder drift"


# =============================================================
# SHARED SEMANTICS BLOCKS
# =============================================================

PACKET_WINDOW_SEMANTICS = {
    "window_seconds": WINDOW_SECONDS,
    "interval": "[t, t+10) — left-closed, right-open",
    "origin": "epoch-aligned multiples of 10 s: window_idx = floor(packet_epoch_seconds / 10) - win0",
    "win0": "the bin of the capture's first packet, identical to the re-basing "
            "src/pcap_ingest_v4.py:reconstruct_flows applies, so the packet-window table and the "
            "flow-window table share an index",
    "rounding": "floor, never round-half; no tolerance band",
    "packet_ordering": "packets are processed in (timestamp_abs, packet_index) order, where "
                       "packet_index is the CAPTURE-ORDER ordinal. Capture timestamps are not "
                       "unique, and the session-state annotations (retransmission, TCP window "
                       "change) depend on observation order, so a timestamp-only sort would "
                       "leave the result dependent on how rows happened to be arranged. Within "
                       "one timestamp the capture order is the best available statement of what "
                       "the sensor saw first. This tie-break is what makes the training-side and "
                       "PCAP-side routes produce bit-identical output across a storage "
                       "round-trip.",
    "row_assignment": "EACH PACKET is assigned to the window containing ITS OWN timestamp",
    "row_assignment_difference_from_v4_flow_layer":
        "The V4 flow layer assigns a WHOLE FLOW — all of its statistics — to the window of its "
        "FIRST packet. The packet layer cannot do that: it would let packets that arrive after "
        "t+10 contribute to the state at t, which is the future-information leakage Part 19 "
        "forbids. The bin WIDTH, ORIGIN and BOUNDARY RULE are identical between the two layers; "
        "only the assignment unit differs (packet vs flow), and that difference is what makes the "
        "packet layer causal.",
    "empty_windows": "produce NO row, identical to the V4 flow layer. The downstream 10-second "
                     "contiguity gate (np.all(diffs == 1) on window_idx) rejects any sequence "
                     "spanning such a gap.",
    "first_window": "kept as-is; its relative-change row is forced to exactly 0.0",
    "last_window": "kept as-is; no packet feature ever reads forward",
    "timestamp_source": "the pcap per-packet capture timestamp (scapy pkt.time), i.e. the clock of "
                        "the capture host at the capture point",
    "timezone": {
        "pcap_packet_clock": "true UTC epoch seconds",
        "training_csv_clock": "CSE-CIC-IDS2018 CICFlowMeter CSV Timestamp is Atlantic local time "
                              "(America/Halifax, UTC-4 on 2018-03-02) with the AM/PM marker "
                              "dropped by the dataset authors",
        "policy": "UNCHANGED from V4. The packet layer floors UTC epoch seconds, exactly as the V4 "
                  "PCAP path already does; the training path's local-clock handling is not touched. "
                  "No timezone conversion is introduced anywhere by V5.",
        "consequence": "bin width and boundary rule are identical across paths, but the two clocks "
                       "are not co-anchored. This matters only when aligning a PCAP window to a "
                       "dataset window; it changes no feature VALUE.",
    },
}

PACKET_TRANSFORM_SEMANTICS = {
    "safe_log1p": {
        "definition": "non-finite -> 0.0, clip(lower=0.0), then log1p",
        "implementation": "src/packet_feature_builder.py:_safe_log1p",
        "applied_to": [d[0] for d in _PACKET_DEFS if d[8]],
        "applied": "AT window aggregation, BEFORE relative change",
    },
    "ratio": {
        "definition": "numerator / denominator, returning EXACTLY 0.0 when denominator == 0",
        "implementation": "src/packet_feature_builder.py:_ratio",
        "difference_from_v4": "V4's flow rates use an epsilon-inflated denominator "
                              "(x / (|y| + 1e-8)). The packet layer does not, because its "
                              "denominators are exact integer packet counts: the zero case is a "
                              "real, enumerable condition ('no TCP packets in this window') that "
                              "should be handled explicitly and be testable, not smeared by an "
                              "epsilon. This is a deliberate V5 policy choice and changes nothing "
                              "about the 26 V4 features.",
        "applied": "AT window aggregation",
    },
    "std": {
        "definition": "sample standard deviation, ddof=1; exactly 0.0 for n <= 1",
        "implementation": "src/packet_feature_builder.py:_std",
        "matches": "the V4 pkt_len_std convention (ddof=1)",
    },
    "percentile": {
        "definition": "numpy linear interpolation, method='linear', pinned explicitly",
        "implementation": "src/packet_feature_builder.py:_percentile",
        "note": "pinned so the method cannot drift with a numpy default change and silently move "
                "every p50/p90 value",
    },
    "entropy_from_counts": {
        "definition": "Shannon entropy base 2 of value_counts(dropna=True), p + 1e-12 inside the "
                      "log, 0.0 when <= 1 distinct value",
        "implementation": "src/packet_feature_builder.py:_entropy_from_counts",
        "matches": "the V4 entropy_from_counts formula exactly",
        "applied_to": ["payload_len_entropy"],
    },
    "nan_inf_handling": {
        "definition": "the 21 packet features are replace([inf,-inf], NaN).fillna(0.0) before "
                      "leaving build_packet_windows",
        "matches": "the identical V4 policy in compute_window_features",
        "applied": "AFTER window aggregation, BEFORE relative change",
    },
}

PACKET_RELATIVE_CHANGE_SEMANTICS = {
    "formula": "r_t = (s_t - s_{t-1}) / (|s_{t-1}| + 1e-8)",
    "epsilon": v4c.RELATIVE_EPSILON,
    "clipping": f"clip(-{v4c.RELATIVE_CLIP}, +{v4c.RELATIVE_CLIP}) AFTER the division",
    "previous_window": "the immediately preceding EXISTING ROW of the packet-window table — not "
                       "necessarily t-10 s, because empty windows produce no row",
    "first_row": "forced to exactly 0.0",
    "non_finite": "replace([inf, -inf], NaN).fillna(0.0)",
    "reset_scope": "one capture (PCAP path) / one source day (training path) — identical to V4",
    "log_order": "computed on the ALREADY log-transformed base feature, identical to V4",
    "identical_to_v4": True,
    "v4_semantics_unchanged": "The 26 V4 relative-change features are computed by the untouched V4 "
                              "code and are not re-derived, re-ordered or re-defined here.",
    "selection_rule": "NOT every new base feature receives a relative change. 9 of 21 do. The "
                      "ruling and reason for each of the 21 is in features[].relative_change_"
                      "decision / relative_change_reason.",
    "implementation": "src/packet_feature_builder.py:add_packet_relative_changes",
}

NORMALISATION_SEMANTICS = {
    "class": "sklearn.preprocessing.StandardScaler",
    "expected_input_dimension": V5_DIMENSION,
    "status": "NOT FITTED — a V5 scaler can only be fitted on a V5 training partition, and no V5 "
              "training partition exists (see docs/v5_packet_dataset_manifest.json). Every one of "
              "the 82 features is therefore declared 'normalised: true, scaler: pending' rather "
              "than being given a fabricated scaler.",
    "v4_scaler_reuse": "FORBIDDEN. data/processed/multiday_benchmark_K5/input_scaler.pkl is a "
                       "52-dimensional frozen V4 artefact; it is read by V5 code only to verify "
                       "its hash is unchanged, never to transform an 82-dimensional vector.",
    "other_transforms": "none — no PCA, no whitening, no feature selection, identical to V4",
}


# =============================================================
# CONTRACT ASSEMBLY
# =============================================================

def _v4_entry(index: int, name: str) -> dict:
    """Re-expose the frozen V4 contract entry under the V5 envelope.
    The V4 contract is the source; nothing here redefines a V4 formula."""
    v4_contract = _v4_contract_cached()
    source = v4_contract["features"][index]
    return {
        "index": index,
        "name": name,
        "schema_origin": "v4",
        "category": source["category"],
        "feature_level": "flow",
        "source_field": source.get("depends_on", []),
        "aggregation": source["aggregation"],
        "aggregation_window": source["aggregation_window"],
        "formula_training": source["training_formula"],
        "formula_pcap": source["pcap_formula"],
        "unit": source["expected_unit"],
        "domain": "see docs/feature_contract_v4.json (frozen)",
        "missing_value_rule": source["missing_value_behaviour"],
        "zero_value_behaviour": "unchanged from V4",
        "log_transformed": source["log_transformed"],
        "normalised": True,
        "relative_change_decision": "included" if not source["relative_change_feature"] else "n/a",
        "relative_change_reason": "frozen V4 convention — all 26 base features have a relative "
                                  "change; NOT re-litigated by V5",
        "depends_on": source["depends_on"],
        "parity": source["flow_layer_parity"],
        "parity_note": source["parity_note"],
        "implementation": {
            "training": v4c.TRAINING_IMPL,
            "pcap": v4c.PCAP_IMPL,
        },
        "v5_modified": False,
    }


_V4_CONTRACT_CACHE = {}


def _v4_contract_cached() -> dict:
    if "c" not in _V4_CONTRACT_CACHE:
        _V4_CONTRACT_CACHE["c"] = v4c.build_contract()
    return _V4_CONTRACT_CACHE["c"]


def _packet_entry(index: int, definition: tuple) -> dict:
    (name, source_fields, aggregation, formula, unit, domain,
     missing_rule, zero_behaviour, log_transformed, notes) = definition
    included, reason = _RELATIVE_DECISIONS[name]
    return {
        "index": index,
        "name": name,
        "schema_origin": "v5",
        "category": "base",
        "feature_level": "packet",
        "family": pfb.PACKET_FAMILY[name],
        "source_field": source_fields,
        "aggregation": aggregation,
        "aggregation_window": f"{WINDOW_SECONDS}s, [t, t+10), per-packet assignment",
        "formula": formula,
        "unit": unit,
        "domain": domain,
        "missing_value_rule": missing_rule,
        "zero_value_behaviour": zero_behaviour,
        "log_transformed": log_transformed,
        "normalised": True,
        "relative_change_decision": "included" if included else "excluded",
        "relative_change_reason": reason,
        "depends_on": source_fields,
        "parity": "EXACT",
        "parity_note": "single shared implementation (src/packet_feature_builder.py) used by both "
                       "the training-side and the PCAP-side entry point; verified numerically "
                       "across two independent invocation routes",
        "implementation": {
            "training": "src/packet_feature_builder.py",
            "pcap": "src/packet_feature_builder.py",
        },
        "notes": notes,
        "v5_modified": False,
    }


def _packet_relative_entry(index: int, name: str) -> dict:
    base_name = name[: -len("_rel_change")]
    return {
        "index": index,
        "name": name,
        "schema_origin": "v5",
        "category": "relative_change",
        "feature_level": "packet",
        "family": pfb.PACKET_FAMILY[base_name],
        "source_field": [base_name],
        "aggregation": "none (cross-window difference)",
        "aggregation_window": f"{WINDOW_SECONDS}s vs the preceding EXISTING packet-window row",
        "formula": PACKET_RELATIVE_CHANGE_SEMANTICS["formula"] + f"  on {base_name}",
        "unit": "dimensionless signed ratio, clipped to [-10, 10]",
        "domain": "[-10, 10]",
        "missing_value_rule": "first row exactly 0.0; non-finite -> 0.0",
        "zero_value_behaviour": "0.0 when the feature is unchanged between adjacent windows",
        "log_transformed": False,
        "normalised": True,
        "relative_change_decision": "n/a (is itself a relative change)",
        "relative_change_reason": _RELATIVE_DECISIONS[base_name][1],
        "depends_on": [base_name],
        "parity": "EXACT",
        "parity_note": "same shared implementation as the base feature",
        "implementation": {
            "training": "src/packet_feature_builder.py:add_packet_relative_changes",
            "pcap": "src/packet_feature_builder.py:add_packet_relative_changes",
        },
        "notes": "identical transform semantics to the 26 frozen V4 relative-change features",
        "v5_modified": False,
    }


def build_packet_contract() -> dict:
    """docs/v5_packet_feature_contract.json — the NEW packet features only,
    described in full. The V4 block is referenced, not restated."""
    features = [
        _packet_entry(52 + offset, definition)
        for offset, definition in enumerate(_PACKET_DEFS)
    ]
    features += [
        _packet_relative_entry(52 + len(_PACKET_DEFS) + offset, name)
        for offset, name in enumerate(PACKET_RELATIVE_FEATURES)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "scope": "the NEW packet-level features V5 adds on top of the frozen V4 52-dimensional "
                 "input. The 26 flow base features and their 26 relative changes are NOT redefined "
                 "here — see docs/feature_contract_v4.json, which V5 does not modify.",
        "window_seconds": WINDOW_SECONDS,
        "sequence_length": SEQUENCE_LENGTH,
        "packet_base_feature_count": len(PACKET_BASE_FEATURES),
        "packet_relative_feature_count": len(PACKET_RELATIVE_FEATURES),
        "packet_feature_count": len(PACKET_BASE_FEATURES) + len(PACKET_RELATIVE_FEATURES),
        "v5_total_dimension": V5_DIMENSION,
        "index_offset": 52,
        "index_offset_note": "packet features occupy indices 52..81; indices 0..51 are the frozen "
                             "V4 vector in the frozen V4 order",
        "families": {
            "ttl": pfb.TTL_FEATURES,
            "tcp_window": pfb.TCP_WINDOW_FEATURES,
            "fragmentation": pfb.FRAGMENTATION_FEATURES,
            "retransmission": pfb.RETRANSMISSION_FEATURES,
            "payload": pfb.PAYLOAD_FEATURES,
        },
        "builder": {
            "module": "src/packet_feature_builder.py",
            "version": pfb.BUILDER_VERSION,
            "session_idle_timeout_s": pfb.SESSION_IDLE_TIMEOUT_S,
            "session_scope": "ORDERED 5-tuple (src_ip, src_port, dst_ip, dst_port, protocol); "
                             "A->B and B->A are separate sessions, matching TCP's independent "
                             "per-direction sequence spaces",
            "parser_fields": pfb.PACKET_COLUMNS,
            "shared_by_both_paths": True,
        },
        "window_semantics": PACKET_WINDOW_SEMANTICS,
        "transform_semantics": PACKET_TRANSFORM_SEMANTICS,
        "relative_change_semantics": PACKET_RELATIVE_CHANGE_SEMANTICS,
        "normalisation_semantics": NORMALISATION_SEMANTICS,
        "content_inspection": "NONE. No payload byte is read anywhere in this pipeline. Every "
                              "payload feature is derived from the integer LENGTH carried in the "
                              "IP/L4 headers.",
        "features": features,
    }


def build_schema() -> dict:
    """docs/v5_feature_schema.json — the full 82-dimensional input,
    categorised by level (flow vs packet) and by category."""
    flow_base = [_v4_entry(i, name) for i, name in enumerate(V4_BASE_FEATURES)]
    flow_rel = [
        _v4_entry(26 + i, name) for i, name in enumerate(V4_RELATIVE_FEATURES)
    ]
    packet_contract = build_packet_contract()
    packet_base = [f for f in packet_contract["features"] if f["category"] == "base"]
    packet_rel = [f for f in packet_contract["features"] if f["category"] == "relative_change"]

    return {
        "version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "description": "V5 experimental packet-aware network state. V4 remains the frozen "
                       "reference baseline and is NOT modified by this schema.",
        "total_dimension": V5_DIMENSION,
        "sequence_length": SEQUENCE_LENGTH,
        "window_seconds": WINDOW_SECONDS,
        "model_input_shape": [SEQUENCE_LENGTH, V5_DIMENSION],
        "feature_order": V5_FEATURE_NAMES,
        "layout": {
            "0-25": "flow base (frozen V4)",
            "26-51": "flow relative change (frozen V4)",
            "52-72": "packet base (new in V5)",
            "73-81": "packet relative change (new in V5)",
        },
        "v4_prefix_invariant": {
            "claim": "v5_input[:52] is byte-identical to the V4 input vector, in the V4 order, "
                     "produced by the unmodified V4 code",
            "verified_by": "src/verify_v5_packet_features.py check 14",
        },
        "base_features": {
            "flow": [f["name"] for f in flow_base],
            "packet": [f["name"] for f in packet_base],
        },
        "relative_change_features": {
            "flow": [f["name"] for f in flow_rel],
            "packet": [f["name"] for f in packet_rel],
        },
        "counts": {
            "flow_base": len(flow_base),
            "flow_relative_change": len(flow_rel),
            "packet_base": len(packet_base),
            "packet_relative_change": len(packet_rel),
            "total": V5_DIMENSION,
        },
        "packet_families": packet_contract["families"],
        "window_semantics": PACKET_WINDOW_SEMANTICS,
        "relative_change_semantics": PACKET_RELATIVE_CHANGE_SEMANTICS,
        "normalisation_semantics": NORMALISATION_SEMANTICS,
        "features": flow_base + flow_rel + packet_base + packet_rel,
        "v4_reference": {
            "schema": "docs/v4_feature_schema.json",
            "contract": "docs/feature_contract_v4.json",
            "dimension": len(V4_FEATURE_NAMES),
            "modified_by_v5": False,
        },
    }


# =============================================================
# DETERMINISTIC SCHEMA HASH (Part 21)
# =============================================================

def canonical_payload(schema: dict | None = None) -> str:
    """Deterministic canonical form of everything that DEFINES the V5
    schema: feature names, order, formulas, units, windowing, transforms,
    missing-value policy and relative-change definitions. Descriptive-only
    fields (prose notes, parity commentary, file paths) are excluded so the
    hash changes if and only if the SEMANTICS change."""
    schema = schema or build_schema()
    payload = {
        "schema_version": schema["version"],
        "total_dimension": schema["total_dimension"],
        "sequence_length": schema["sequence_length"],
        "window_seconds": schema["window_seconds"],
        "feature_order": schema["feature_order"],
        "features": [
            {
                "index": f["index"],
                "name": f["name"],
                "category": f["category"],
                "feature_level": f["feature_level"],
                "aggregation": f["aggregation"],
                "aggregation_window": f["aggregation_window"],
                "formula": f.get("formula") or f.get("formula_pcap"),
                "formula_training": f.get("formula_training"),
                "unit": f["unit"],
                "domain": f["domain"],
                "missing_value_rule": f["missing_value_rule"],
                "zero_value_behaviour": f["zero_value_behaviour"],
                "log_transformed": f["log_transformed"],
                "normalised": f["normalised"],
                "relative_change_decision": f["relative_change_decision"],
                "depends_on": f["depends_on"],
            }
            for f in schema["features"]
        ],
        "window_semantics": {
            key: PACKET_WINDOW_SEMANTICS[key]
            for key in ("window_seconds", "interval", "origin", "rounding", "packet_ordering",
                        "row_assignment", "empty_windows", "first_window", "last_window",
                        "timestamp_source")
        },
        "transform_semantics": {
            key: value["definition"]
            for key, value in PACKET_TRANSFORM_SEMANTICS.items()
        },
        "relative_change_semantics": {
            key: PACKET_RELATIVE_CHANGE_SEMANTICS[key]
            for key in ("formula", "epsilon", "clipping", "previous_window",
                        "first_row", "non_finite", "reset_scope", "log_order")
        },
        "normalisation": {
            "class": NORMALISATION_SEMANTICS["class"],
            "expected_input_dimension": NORMALISATION_SEMANTICS["expected_input_dimension"],
            "other_transforms": NORMALISATION_SEMANTICS["other_transforms"],
        },
        "v4_contract_hash": v4c.contract_hash(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def schema_hash(schema: dict | None = None) -> str:
    return hashlib.sha256(canonical_payload(schema).encode("utf-8")).hexdigest()


def load_schema() -> dict:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"V5 schema not found: {SCHEMA_PATH}")
    return json.loads(SCHEMA_PATH.read_text())


def load_packet_contract() -> dict:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"V5 packet contract not found: {CONTRACT_PATH}")
    return json.loads(CONTRACT_PATH.read_text())


if __name__ == "__main__":
    print(f"V5 dimension     : {V5_DIMENSION}")
    print(f"  flow base      : {len(V4_BASE_FEATURES)}  (frozen V4)")
    print(f"  flow rel change: {len(V4_RELATIVE_FEATURES)}  (frozen V4)")
    print(f"  packet base    : {len(PACKET_BASE_FEATURES)}  (new)")
    print(f"  packet rel chg : {len(PACKET_RELATIVE_FEATURES)}  (new)")
    print(f"V5 schema sha256 : {schema_hash()}")
    print(f"V4 contract hash : {v4c.contract_hash()}  (read-only)")
