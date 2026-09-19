"""
Generate the deterministic V5 packet-feature fixtures (Part 14).

CRITICAL PROPERTY: this generator NEVER imports src/packet_feature_builder.py.
Every expected value below is produced by the small independent reference
implementation in this file (plain ``math`` / ``collections``), written
directly from the contract's formulas, or is stated as a literal that
follows from the definition (an entropy of one distinct value is exactly
0.0, a ratio of 2 in 5 is exactly 0.4).

That is the whole point: the tests compare the builder against an
INDEPENDENT computation, not against its own output.

Regenerate:
    python tests/fixtures/v5_packet_features/_generate_fixtures.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Defaults applied to every fixture packet; a fixture overrides only what it
# is actually testing, which keeps each file readable.
PACKET_DEFAULTS = {
    "timestamp_abs": 0.0,
    "ip_version": 4,
    "src_ip": "10.0.0.1",
    "dst_ip": "10.0.0.2",
    "src_port": 1000,
    "dst_port": 80,
    "protocol": 6,
    "ttl": 64,
    "ip_df": False,
    "ip_mf": False,
    "ip_frag_offset": 0,
    "is_tcp": True,
    "tcp_window": 1000,
    "tcp_seq": 0,
    "tcp_syn": False,
    "tcp_fin": False,
    "payload_len": 0,
}


# =============================================================
# INDEPENDENT REFERENCE ARITHMETIC (no builder import)
# =============================================================

def r_mean(xs):
    return sum(xs) / len(xs)


def r_std(xs):
    """Sample standard deviation, ddof=1; 0.0 for n <= 1."""
    n = len(xs)
    if n <= 1:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def r_pct(xs, q):
    """numpy 'linear' percentile, written out longhand."""
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = (q / 100.0) * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo]) + (float(s[hi]) - float(s[lo])) * (pos - lo)


def r_entropy(xs):
    """Shannon entropy base 2 of the VALUE distribution; 0.0 if <= 1 distinct."""
    counts = Counter(xs)
    if len(counts) <= 1:
        return 0.0
    n = len(xs)
    return -sum((c / n) * math.log2(c / n + 1e-12) for c in counts.values())


def r_log1p(x):
    return math.log(1.0 + max(float(x), 0.0))


def r_ratio(a, b):
    return 0.0 if b == 0 else float(a) / float(b)


def r_rel(current, previous):
    """The relative-change transform, longhand."""
    value = (current - previous) / (abs(previous) + 1e-8)
    return max(-10.0, min(10.0, value))


# =============================================================
# FIXTURE DEFINITIONS
# =============================================================

def p(**overrides):
    row = dict(PACKET_DEFAULTS)
    row.update(overrides)
    return row


FIXTURES = []


def add(name, description, derivation, packets, expected_windows):
    FIXTURES.append(
        {
            "name": name,
            "description": description,
            "derivation": derivation,
            "packet_defaults": PACKET_DEFAULTS,
            "packets": packets,
            "expected_windows": expected_windows,
        }
    )


# ---------- 01: different TTL values ----------
_ttls = [32, 64, 64, 128]
add(
    "01_ttl_varied",
    "four non-TCP packets with four different TTL values in one window",
    "ttl_mean = (32+64+64+128)/4; ttl_std = sqrt(sum sq dev / 3); unique = 3 distinct values",
    [p(timestamp_abs=0.0 + i * 0.1, ttl=t, is_tcp=False, protocol=17, payload_len=10)
     for i, t in enumerate(_ttls)],
    [{
        "window_idx": 0,
        "packet_count": 4,
        "features": {
            "ttl_mean": r_mean(_ttls),
            "ttl_std": r_std(_ttls),
            "ttl_min": 32.0,
            "ttl_max": 128.0,
            "log_ttl_unique_count": r_log1p(3),
            # no TCP at all -> the whole TCP-window and retransmission
            # families collapse to the documented 0.0 absent-encoding
            "log_tcp_window_mean": 0.0,
            "log_tcp_window_std": 0.0,
            "log_tcp_window_max": 0.0,
            "tcp_window_zero_ratio": 0.0,
            "tcp_window_change_rate": 0.0,
            "log_tcp_retrans_count": 0.0,
            "tcp_retrans_ratio": 0.0,
            "log_tcp_retrans_bytes": 0.0,
        },
    }],
)

# ---------- 02: constant TTL ----------
add(
    "02_ttl_constant",
    "four packets all with TTL 64 — std must be exactly 0.0, unique count exactly 1",
    "constant series: std (ddof=1) = 0.0 exactly; log_ttl_unique_count = log1p(1) = ln(2)",
    [p(timestamp_abs=i * 0.1, ttl=64, is_tcp=False, protocol=17, payload_len=10)
     for i in range(4)],
    [{
        "window_idx": 0,
        "packet_count": 4,
        "features": {
            "ttl_mean": 64.0,
            "ttl_std": 0.0,
            "ttl_min": 64.0,
            "ttl_max": 64.0,
            "log_ttl_unique_count": r_log1p(1),
        },
    }],
)

# ---------- 03: TCP packets with changing window size ----------
_wins3 = [1000, 2000, 2000, 500, 500, 500]
add(
    "03_tcp_window_changing",
    "one directional session whose advertised window changes twice",
    "eligible = 5 packets (all but the session's first). changed = p2 (1000->2000) and "
    "p4 (2000->500) = 2. rate = 2/5 = 0.4 exactly.",
    [p(timestamp_abs=i * 0.1, tcp_window=w, tcp_seq=1000 + i * 100, payload_len=100)
     for i, w in enumerate(_wins3)],
    [{
        "window_idx": 0,
        "packet_count": 6,
        "features": {
            "log_tcp_window_mean": r_log1p(r_mean(_wins3)),
            "log_tcp_window_std": r_log1p(r_std(_wins3)),
            "log_tcp_window_max": r_log1p(2000),
            "tcp_window_zero_ratio": 0.0,
            "tcp_window_change_rate": 0.4,
        },
    }],
)

# ---------- 04: zero TCP window ----------
_wins4 = [0, 0, 1000, 0]
add(
    "04_tcp_window_zero",
    "three zero-window advertisements out of four TCP packets",
    "zero_ratio = 3/4 = 0.75 exactly. std of [0,0,1000,0] with ddof=1 is exactly 500.0. "
    "change rate: eligible = 3, changed = p3 (0->1000) and p4 (1000->0) = 2, so 2/3.",
    [p(timestamp_abs=i * 0.1, tcp_window=w, tcp_seq=1000 + i * 100, payload_len=100)
     for i, w in enumerate(_wins4)],
    [{
        "window_idx": 0,
        "packet_count": 4,
        "features": {
            "log_tcp_window_mean": r_log1p(250.0),
            "log_tcp_window_std": r_log1p(500.0),
            "log_tcp_window_max": r_log1p(1000),
            "tcp_window_zero_ratio": 0.75,
            "tcp_window_change_rate": r_ratio(2, 3),
        },
    }],
)

# ---------- 05: no TCP traffic ----------
_ttls5 = [50, 60]
add(
    "05_no_tcp",
    "UDP-only window — every TCP-dependent feature must be the 0.0 absent-encoding, "
    "while the TTL and payload families are still computed normally",
    "no TCP packet exists, so every TCP denominator is 0 and _ratio returns exactly 0.0; "
    "log1p(0) = 0.0 for the log-transformed TCP features",
    [p(timestamp_abs=0.0, ttl=50, is_tcp=False, protocol=17, src_port=5353, dst_port=5353,
       payload_len=200),
     p(timestamp_abs=1.0, ttl=60, is_tcp=False, protocol=17, src_port=5353, dst_port=5353,
       payload_len=300)],
    [{
        "window_idx": 0,
        "packet_count": 2,
        "features": {
            "ttl_mean": 55.0,
            "ttl_std": r_std(_ttls5),
            "log_tcp_window_mean": 0.0,
            "log_tcp_window_std": 0.0,
            "log_tcp_window_max": 0.0,
            "tcp_window_zero_ratio": 0.0,
            "tcp_window_change_rate": 0.0,
            "log_tcp_retrans_count": 0.0,
            "tcp_retrans_ratio": 0.0,
            "log_tcp_retrans_bytes": 0.0,
            "payload_len_p50": r_pct([200, 300], 50),
            "payload_len_p90": r_pct([200, 300], 90),
            "payload_len_max": 300.0,
            "payload_zero_ratio": 0.0,
            "payload_len_entropy": r_entropy([200, 300]),
        },
    }],
)

# ---------- 06: fragmented packets ----------
add(
    "06_fragmented",
    "IPv4 first/middle/last fragments, unfragmented packets, DF-marked packets, an "
    "unfragmented IPv6 packet and an IPv6 fragment-header packet",
    "fragmented = MF set OR offset > 0 -> p1 (MF,off=0 first), p2 (MF,off=185 middle), "
    "p3 (MF clear, off=370 last), p8 (IPv6 fragment header) = 4 of 8. "
    "DF is NOT fragmentation: p5 and p6 carry DF and are unfragmented. "
    "ip_df_ratio denominator is IPv4 ONLY -> 2 DF of 6 IPv4 packets.",
    [
        p(timestamp_abs=0.0, ip_mf=True, ip_frag_offset=0, payload_len=1480),
        p(timestamp_abs=0.1, ip_mf=True, ip_frag_offset=185, is_tcp=False, payload_len=1480),
        p(timestamp_abs=0.2, ip_mf=False, ip_frag_offset=370, is_tcp=False, payload_len=800),
        p(timestamp_abs=0.3, payload_len=100, tcp_seq=1),
        p(timestamp_abs=0.4, ip_df=True, payload_len=100, tcp_seq=2000),
        p(timestamp_abs=0.5, ip_df=True, payload_len=100, tcp_seq=3000),
        p(timestamp_abs=0.6, ip_version=6, src_ip="fe80::1", dst_ip="fe80::2",
          is_tcp=False, protocol=17, payload_len=100),
        p(timestamp_abs=0.7, ip_version=6, src_ip="fe80::1", dst_ip="fe80::2",
          ip_mf=True, ip_frag_offset=0, is_tcp=False, protocol=17, payload_len=1400),
    ],
    [{
        "window_idx": 0,
        "packet_count": 8,
        "features": {
            "log_frag_packet_count": r_log1p(4),
            "frag_packet_ratio": 0.5,
            "ip_df_ratio": r_ratio(2, 6),
        },
    }],
)

# ---------- 07: no fragmentation ----------
add(
    "07_no_fragmentation",
    "four IPv4 packets, all DF-marked, none fragmented",
    "frag count = 0 -> log1p(0) = 0.0 exactly; frag ratio = 0/4 = 0.0; df ratio = 4/4 = 1.0",
    [p(timestamp_abs=i * 0.1, ip_df=True, tcp_seq=1000 + i * 100, payload_len=100)
     for i in range(4)],
    [{
        "window_idx": 0,
        "packet_count": 4,
        "features": {
            "log_frag_packet_count": 0.0,
            "frag_packet_ratio": 0.0,
            "ip_df_ratio": 1.0,
        },
    }],
)

# ---------- 08: duplicate TCP sequence ranges ----------
_pay8 = [100, 100, 100, 100, 0, 0, 50]
add(
    "08_tcp_retransmission",
    "re-sent sequence ranges plus duplicate PURE ACKs that must NOT be counted",
    "p1 seq=1000 len=100 establishes the high-water mark at 1100 (never a retransmission). "
    "p2 seq=1100 len=100 advances it to 1200. "
    "p3 seq=1100 len=100 ends at 1200 <= 1200 -> RETRANSMISSION (100 bytes). "
    "p4 seq=1000 len=100 ends at 1100 <= 1200 -> RETRANSMISSION (100 bytes). "
    "p5 and p6 are zero-length pure ACKs with an identical sequence number: eff_len = 0, so "
    "they are NOT seq-bearing and NOT retransmissions — this is the duplicate-ACK exclusion. "
    "p7 seq=1200 len=50 ends at 1250 > 1200 -> new data. "
    "retrans = 2, retrans bytes = 200, seq-bearing = 5 (p1,p2,p3,p4,p7), ratio = 2/5.",
    [
        p(timestamp_abs=0.0, tcp_seq=1000, payload_len=100),
        p(timestamp_abs=0.1, tcp_seq=1100, payload_len=100),
        p(timestamp_abs=0.2, tcp_seq=1100, payload_len=100),
        p(timestamp_abs=0.3, tcp_seq=1000, payload_len=100),
        p(timestamp_abs=0.4, tcp_seq=1200, payload_len=0),
        p(timestamp_abs=0.5, tcp_seq=1200, payload_len=0),
        p(timestamp_abs=0.6, tcp_seq=1200, payload_len=50),
    ],
    [{
        "window_idx": 0,
        "packet_count": 7,
        "features": {
            "log_tcp_retrans_count": r_log1p(2),
            "tcp_retrans_ratio": 0.4,
            "log_tcp_retrans_bytes": r_log1p(200),
            "tcp_window_change_rate": 0.0,
            "payload_len_p50": r_pct(_pay8, 50),
            "payload_len_p90": r_pct(_pay8, 90),
            "payload_len_max": 100.0,
            "payload_zero_ratio": r_ratio(2, 7),
            "payload_len_entropy": r_entropy(_pay8),
        },
    }],
)

# ---------- 09: clean non-retransmitted TCP ----------
add(
    "09_tcp_clean",
    "five strictly advancing TCP segments — nothing may be flagged",
    "every segment starts exactly where the previous one ended, so seq+len is always strictly "
    "above the high-water mark: retrans = 0, ratio = 0/5 = 0.0, bytes = log1p(0) = 0.0",
    [p(timestamp_abs=i * 0.1, tcp_seq=1000 + i * 100, payload_len=100) for i in range(5)],
    [{
        "window_idx": 0,
        "packet_count": 5,
        "features": {
            "log_tcp_retrans_count": 0.0,
            "tcp_retrans_ratio": 0.0,
            "log_tcp_retrans_bytes": 0.0,
        },
    }],
)

# ---------- 10: zero payload ----------
add(
    "10_zero_payload",
    "four packets carrying no payload at all",
    "every length is 0 -> p50 = p90 = max = 0.0, zero ratio = 4/4 = 1.0, and the length "
    "distribution has ONE distinct value so its entropy is exactly 0.0",
    [p(timestamp_abs=i * 0.1, tcp_seq=1000, payload_len=0) for i in range(4)],
    [{
        "window_idx": 0,
        "packet_count": 4,
        "features": {
            "payload_len_p50": 0.0,
            "payload_len_p90": 0.0,
            "payload_len_max": 0.0,
            "payload_zero_ratio": 1.0,
            "payload_len_entropy": 0.0,
            # zero-length packets are not seq-bearing, so nothing can be
            # flagged as a retransmission however often the seq repeats
            "log_tcp_retrans_count": 0.0,
            "tcp_retrans_ratio": 0.0,
        },
    }],
)

# ---------- 11: mixed payload lengths ----------
_pay11 = [0, 100, 500, 1460, 1460]
add(
    "11_mixed_payload",
    "a realistic mix: one ACK, a small write, a medium write and two MTU-sized segments",
    "sorted lengths [0,100,500,1460,1460]. p50 position = 0.5*4 = 2 -> exactly 500. "
    "p90 position = 0.9*4 = 3.6 -> 1460 + (1460-1460)*0.6 = exactly 1460. "
    "zero ratio = 1/5 = 0.2. Entropy over counts {0:1,100:1,500:1,1460:2}.",
    [p(timestamp_abs=i * 0.1, tcp_seq=1000 + sum(_pay11[:i]), payload_len=n)
     for i, n in enumerate(_pay11)],
    [{
        "window_idx": 0,
        "packet_count": 5,
        "features": {
            "payload_len_p50": 500.0,
            "payload_len_p90": 1460.0,
            "payload_len_max": 1460.0,
            "payload_zero_ratio": 0.2,
            "payload_len_entropy": r_entropy(_pay11),
        },
    }],
)

# ---------- 12: identical payload lengths ----------
add(
    "12_identical_payload",
    "five packets all carrying exactly 512 bytes — the fixed-size beacon signature",
    "one distinct length -> entropy exactly 0.0; every percentile and the max are 512; "
    "zero ratio = 0.0",
    [p(timestamp_abs=i * 0.1, tcp_seq=1000 + i * 512, payload_len=512) for i in range(5)],
    [{
        "window_idx": 0,
        "packet_count": 5,
        "features": {
            "payload_len_p50": 512.0,
            "payload_len_p90": 512.0,
            "payload_len_max": 512.0,
            "payload_zero_ratio": 0.0,
            "payload_len_entropy": 0.0,
        },
    }],
)

# ---------- 13: multi-window relative change ----------
add(
    "13_multi_window_relative_change",
    "three consecutive windows with a rising then falling mean TTL — exercises the "
    "relative-change transform including the 0/0 case",
    "row 0 is forced to exactly 0.0. row 1 = (150-100)/(100+1e-8). "
    "row 2 = (50-150)/(150+1e-8). ttl_std is 0.0 in every window, so its relative change "
    "is (0-0)/(0+1e-8) = 0.0 exactly — NOT NaN.",
    [
        p(timestamp_abs=1.0, ttl=100, is_tcp=False, protocol=17, payload_len=10),
        p(timestamp_abs=2.0, ttl=100, is_tcp=False, protocol=17, payload_len=10),
        p(timestamp_abs=11.0, ttl=150, is_tcp=False, protocol=17, payload_len=10),
        p(timestamp_abs=12.0, ttl=150, is_tcp=False, protocol=17, payload_len=10),
        p(timestamp_abs=21.0, ttl=50, is_tcp=False, protocol=17, payload_len=10),
        p(timestamp_abs=22.0, ttl=50, is_tcp=False, protocol=17, payload_len=10),
    ],
    [
        {"window_idx": 0, "packet_count": 2,
         "features": {"ttl_mean": 100.0, "ttl_std": 0.0},
         "relative": {"ttl_mean_rel_change": 0.0, "ttl_std_rel_change": 0.0}},
        {"window_idx": 1, "packet_count": 2,
         "features": {"ttl_mean": 150.0, "ttl_std": 0.0},
         "relative": {"ttl_mean_rel_change": r_rel(150.0, 100.0),
                      "ttl_std_rel_change": 0.0}},
        {"window_idx": 2, "packet_count": 2,
         "features": {"ttl_mean": 50.0, "ttl_std": 0.0},
         "relative": {"ttl_mean_rel_change": r_rel(50.0, 150.0),
                      "ttl_std_rel_change": 0.0}},
    ],
)

# ---------- 14: TCP sequence wraparound ----------
add(
    "14_seq_wraparound",
    "a retransmission whose sequence number has wrapped past 2^32",
    "p1 seq=100 len=100 sets the high-water mark to 200. p2 seq=4294967200 len=50 ends at "
    "4294967250. Under 32-bit MODULAR comparison that is 246 sequence numbers BEFORE 200, so "
    "it is already-transmitted space -> RETRANSMISSION. A naive unsigned comparison "
    "(4294967250 <= 200?) would be False and would miss it, which is exactly what this "
    "fixture guards against. retrans = 1 of 2 seq-bearing, 50 retransmitted bytes.",
    [
        p(timestamp_abs=0.0, tcp_seq=100, payload_len=100),
        p(timestamp_abs=0.1, tcp_seq=4294967200, payload_len=50),
    ],
    [{
        "window_idx": 0,
        "packet_count": 2,
        "features": {
            "log_tcp_retrans_count": r_log1p(1),
            "tcp_retrans_ratio": 0.5,
            "log_tcp_retrans_bytes": r_log1p(50),
        },
    }],
)

# ---------- 15: window-boundary causality ----------
add(
    "15_window_boundary_causality",
    "a window-size change that straddles a 10 s boundary — it must be attributed to the "
    "LATER window and must leave the earlier window untouched",
    "window 0 holds p1 (t=1, win=1000) and p2 (t=2, win=1000): eligible = 1 (p2), changed = 0, "
    "rate = 0.0. window 1 holds p3 (t=11, win=2000) and p4 (t=12, win=2000): eligible = 2, "
    "changed = 1 (p3 differs from its predecessor p2), rate = 0.5. If the builder leaked "
    "backwards, window 0's mean would not be exactly log1p(1000).",
    [
        p(timestamp_abs=1.0, tcp_window=1000, tcp_seq=1000, payload_len=100),
        p(timestamp_abs=2.0, tcp_window=1000, tcp_seq=1100, payload_len=100),
        p(timestamp_abs=11.0, tcp_window=2000, tcp_seq=1200, payload_len=100),
        p(timestamp_abs=12.0, tcp_window=2000, tcp_seq=1300, payload_len=100),
    ],
    [
        {"window_idx": 0, "packet_count": 2,
         "features": {"log_tcp_window_mean": r_log1p(1000),
                      "log_tcp_window_max": r_log1p(1000),
                      "tcp_window_change_rate": 0.0}},
        {"window_idx": 1, "packet_count": 2,
         "features": {"log_tcp_window_mean": r_log1p(2000),
                      "log_tcp_window_max": r_log1p(2000),
                      "tcp_window_change_rate": 0.5}},
    ],
)

# ---------- 16: SYN retransmission with no payload ----------
add(
    "16_syn_retransmission",
    "a SYN re-sent twice with no payload — the count must fire while the byte counter stays 0",
    "a SYN consumes one sequence number, so eff_len = 1 and the packet IS seq-bearing. "
    "p1 sets the high-water mark to 501. p2 and p3 reuse seq=500 and end at 501 <= 501, so "
    "both are retransmissions. Their payload is 0, so retrans bytes = 0 and "
    "log_tcp_retrans_bytes = log1p(0) = 0.0 while log_tcp_retrans_count = log1p(2). "
    "This is the documented count-vs-bytes distinction: control-plane retransmission at no "
    "data cost.",
    [
        p(timestamp_abs=0.0, tcp_seq=500, tcp_syn=True, payload_len=0, tcp_window=8192),
        p(timestamp_abs=1.0, tcp_seq=500, tcp_syn=True, payload_len=0, tcp_window=8192),
        p(timestamp_abs=2.0, tcp_seq=500, tcp_syn=True, payload_len=0, tcp_window=8192),
    ],
    [{
        "window_idx": 0,
        "packet_count": 3,
        "features": {
            "log_tcp_retrans_count": r_log1p(2),
            "tcp_retrans_ratio": r_ratio(2, 3),
            "log_tcp_retrans_bytes": 0.0,
        },
    }],
)


def main():
    for fixture in FIXTURES:
        path = HERE / f"{fixture['name']}.json"
        path.write_text(json.dumps(fixture, indent=2, sort_keys=False) + "\n")
        print(f"wrote {path.relative_to(HERE.parent.parent.parent)}")
    print(f"\n{len(FIXTURES)} fixtures generated.")


if __name__ == "__main__":
    main()
