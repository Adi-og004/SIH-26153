"""
=============================================================
DEMO PCAP GENERATOR
=============================================================

Synthesizes 5 self-contained demo captures for the Streamlit
"📥 Data Input" page (Upload PCAP source): ~220s each, first half benign
baseline traffic, second half a distinct MITRE-mappable attack
pattern. Well past the 100s (10 x 10s windows) the pipeline
needs for one sequence, so every file yields pure-benign,
transition, and pure-attack sequences to click through live.

The ingestion pipeline (pcap_ingest_v4.py) aggregates features
per FLOW, attributed entirely to the 10s window of the flow's
FIRST packet -- so every window that should show elevated
activity needs at least one NEW 5-tuple flow starting inside
it. Every flow below uses a fresh source port so 5-tuples never
repeat (no accidental cross-window merging via the 120s idle
timeout).

Usage: python src/generate_demo_pcaps.py
Output: data/sample_pcap/demo_<scenario>.pcap
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap

WINDOW_SECONDS = 10
N_BENIGN_WINDOWS = 11
N_ATTACK_WINDOWS = 11
T0_EPOCH = 1_759_000_000.0  # arbitrary but realistic-looking epoch

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_pcap"

random.seed(42)

_sport_counter = [20000]


def next_sport() -> int:
    _sport_counter[0] += 1
    return _sport_counter[0]


def pkt(src, dst, sport, dport, flags=None, payload=b"", proto="tcp"):
    ip = IP(src=src, dst=dst)
    if proto == "tcp":
        l4 = TCP(sport=sport, dport=dport, flags=flags or "")
    else:
        l4 = UDP(sport=sport, dport=dport)
    p = Ether() / ip / l4
    if payload:
        p = p / Raw(load=payload)
    return p


# =============================================================
# BENIGN TRAFFIC (shared baseline for every demo file)
# =============================================================

CLIENTS = ["10.0.0.5", "10.0.0.7", "10.0.0.9", "10.0.0.11"]
WEB_SERVER = "10.0.0.1"
DNS_SERVER = "10.0.0.2"


def tcp_flow(t0, src, dst, dport, payload_c2s=0, payload_s2c=0):
    sport = next_sport()
    t = t0
    out = []
    out.append((t, pkt(src, dst, sport, dport, "S"))); t += 0.003
    out.append((t, pkt(dst, src, dport, sport, "SA"))); t += 0.003
    out.append((t, pkt(src, dst, sport, dport, "A"))); t += 0.004
    if payload_c2s:
        out.append((t, pkt(src, dst, sport, dport, "PA", os.urandom(payload_c2s))))
        t += 0.015
    if payload_s2c:
        out.append((t, pkt(dst, src, dport, sport, "PA", os.urandom(payload_s2c))))
        t += 0.015
    out.append((t, pkt(src, dst, sport, dport, "FA"))); t += 0.003
    out.append((t, pkt(dst, src, dport, sport, "FA"))); t += 0.003
    out.append((t, pkt(src, dst, sport, dport, "A")))
    return out


def udp_flow(t0, src, dst, dport=53, q_size=40, r_size=90):
    sport = next_sport()
    out = [(t0, pkt(src, dst, sport, dport, payload=os.urandom(q_size), proto="udp"))]
    out.append((t0 + 0.02, pkt(dst, src, dport, sport, payload=os.urandom(r_size), proto="udp")))
    return out


def benign_window(t0):
    # A handful of concurrent, varied flows per window -- closer to
    # real multi-connection browsing than one lone flow, which reads
    # as unusually sparse (and hence anomalous) to the model.
    pkts = []
    n_flows = random.randint(4, 7)
    for _ in range(n_flows):
        client = random.choice(CLIENTS)
        dport = random.choice([80, 443, 443, 22])
        pkts += tcp_flow(
            t0 + random.uniform(0, 9), client, WEB_SERVER, dport,
            payload_c2s=random.randint(80, 400),
            payload_s2c=random.randint(200, 1200),
        )
        if random.random() < 0.5:
            pkts += udp_flow(t0 + random.uniform(0, 9), client, DNS_SERVER)
    return pkts


def build_benign_phase(start_window=0):
    pkts = []
    for w in range(start_window, start_window + N_BENIGN_WINDOWS):
        t0 = T0_EPOCH + w * WINDOW_SECONDS
        pkts += benign_window(t0)
    return pkts


# =============================================================
# ATTACK SCENARIOS
# =============================================================

def portscan_window(t0, intensity=1.0, attacker="192.168.50.60", target=WEB_SERVER):
    # Fast sweep (hundreds of probes/window, like an aggressive
    # Nmap -T4 SYN scan) -- the model's own attack signal tracks
    # raw flow/packet volume, so the scan needs to be genuinely
    # loud, not just "many ports" in the abstract.
    n_ports = int(320 + intensity * 130)
    pkts = []
    t = t0
    ports = random.sample(range(1, 65000), n_ports)
    for i, port in enumerate(ports):
        sport = next_sport()
        pkts.append((t, pkt(attacker, target, sport, port, "S")))
        t += 0.006
        if i % 12 == 0:
            pkts.append((t, pkt(target, attacker, port, sport, "SA")))
        else:
            pkts.append((t, pkt(target, attacker, port, sport, "R")))
        t += 0.006
    return pkts


def beacon_window(t0, intensity=1.0, host="10.0.0.15", c2="203.0.113.50", dport=8443):
    # Several fixed-cadence check-ins per window (the beaconing
    # fingerprint) with a growing payload, as if staging recon data
    # out in small increments each callback.
    size = int(500 + intensity * 800)
    n_checkins = 10
    t = t0
    out = []
    for _ in range(n_checkins):
        sport = next_sport()
        out.append((t, pkt(host, c2, sport, dport, "S"))); t += 0.01
        out.append((t, pkt(c2, host, dport, sport, "SA"))); t += 0.01
        out.append((t, pkt(host, c2, sport, dport, "A"))); t += 0.01
        out.append((t, pkt(host, c2, sport, dport, "PA", os.urandom(size)))); t += 0.02
        out.append((t, pkt(c2, host, dport, sport, "PA", os.urandom(max(32, size // 4))))); t += 0.02
        out.append((t, pkt(host, c2, sport, dport, "FA"))); t += 0.01
        out.append((t, pkt(c2, host, dport, sport, "FA"))); t += 0.01
        out.append((t, pkt(host, c2, sport, dport, "A"))); t += 0.82
    return out


LATERAL_TARGETS = ["10.0.0.20", "10.0.0.21", "10.0.0.22", "10.0.0.23", "10.0.0.24"]
LATERAL_PORTS = [22, 3389, 445]


def lateral_window(t0, intensity=1.0, attacker="10.0.0.15"):
    # Rapid-fire connection attempts across every internal host/port
    # combo, repeatedly -- a noisy, fast pivot (matches the volume
    # scale the model actually reacts to) rather than a slow crawl.
    n_hops = int(320 + intensity * 120)
    pkts = []
    t = t0
    for i in range(n_hops):
        target = random.choice(LATERAL_TARGETS)
        port = random.choice(LATERAL_PORTS)
        pkts += tcp_flow(
            t, attacker, target, port,
            payload_c2s=random.randint(40, 100),
            payload_s2c=random.randint(50, 120),
        )
        t += 0.016
        if t - t0 > 9.0:
            break
    return pkts


def exfil_window(t0, intensity=1.0, host="10.0.0.15", dst="198.51.100.77", dport=443):
    # Several parallel high-rate streams (a multi-threaded exfil
    # tool) -- both flow_count and raw byte/packet volume spike
    # together, matching the scale the model reacts to.
    n_streams = 60
    n_pkts_per_stream = int(6 + intensity * 4)
    size = int(1300 + intensity * 200)
    out = []
    for _ in range(n_streams):
        sport = next_sport()
        t = t0 + random.uniform(0, 0.5)
        out.append((t, pkt(host, dst, sport, dport, "S"))); t += 0.005
        out.append((t, pkt(dst, host, dport, sport, "SA"))); t += 0.005
        out.append((t, pkt(host, dst, sport, dport, "A"))); t += 0.005
        for _ in range(n_pkts_per_stream):
            out.append((t, pkt(host, dst, sport, dport, "PA", os.urandom(size))))
            t += 0.03
            if t - t0 > 9.0:
                break
        out.append((t, pkt(host, dst, sport, dport, "FA"))); t += 0.005
        out.append((t, pkt(dst, host, dport, sport, "FA"))); t += 0.005
        out.append((t, pkt(host, dst, sport, dport, "A")))
    return out


def dos_window(t0, intensity=1.0, target=WEB_SERVER, port=80):
    # Full flood from the first attack window.
    n_pkts = int(320 + intensity * 180)
    pkts = []
    t = t0
    for _ in range(n_pkts):
        src = f"185.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}"
        sport = random.randint(1024, 65535)
        pkts.append((t, pkt(src, target, sport, port, "S")))
        t += 9.0 / n_pkts
    return pkts


SCENARIOS = {
    "demo_portscan": ("Reconnaissance -- Network Service Scanning", portscan_window),
    "demo_c2_beacon": ("Command and Control -- Beaconing", beacon_window),
    "demo_lateral_movement": ("Lateral Movement -- Remote Services", lateral_window),
    "demo_exfiltration": ("Exfiltration -- Sustained Outbound Transfer", exfil_window),
    "demo_dos_flood": ("Impact -- Resource Exhaustion (SYN flood)", dos_window),
}


def build_attack_phase(window_fn, start_window=N_BENIGN_WINDOWS):
    pkts = []
    for i in range(N_ATTACK_WINDOWS):
        w = start_window + i
        t0 = T0_EPOCH + w * WINDOW_SECONDS
        # Full strength from the first attack window (a weak onset
        # gives the model nothing sharp to key off), then a mild
        # further climb -- 0.85 -> 1.0 -- so features keep moving.
        intensity = 0.85 + 0.15 * (i / (N_ATTACK_WINDOWS - 1))
        pkts += window_fn(t0, intensity=intensity)
    return pkts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (label, window_fn) in SCENARIOS.items():
        _sport_counter[0] = 20000
        random.seed(42)
        pkts = build_benign_phase() + build_attack_phase(window_fn)
        pkts.sort(key=lambda x: x[0])
        frames = []
        for t, p in pkts:
            p.time = t
            frames.append(p)
        out_path = OUT_DIR / f"{name}.pcap"
        wrpcap(str(out_path), frames)
        duration = frames[-1].time - frames[0].time
        print(
            f"{name}.pcap -- {label} | {len(frames)} packets | "
            f"{duration:.1f}s"
        )


if __name__ == "__main__":
    main()
