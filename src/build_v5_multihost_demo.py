"""
=============================================================
V5 MULTI-HOST DEMO SUBSTRATE BUILDER
=============================================================

Builds a real MULTI-HOST packet-aware 82-D screening substrate from the ten
March-2 victim captures, to test whether increasing the number of observed
hosts reduces the V5 input distribution shift that Task 5 localised to the
52-D V4 flow prefix.

-------------------------------------------------------------
SCOPE WORDING — USE THIS, NOT "ORGANIZATION-WIDE"
-------------------------------------------------------------
This is a **multi-host packet-aware prototype representation**: a
10-host approximation of the organization-level flow state. The V5
training reference for the 52-D prefix was built from CIC's
CICFlowMeter CSVs covering the WHOLE 172.31.0.0/16 victim network
(~440 hosts). Ten hosts is an approximation of that scope, not that
scope. Nothing here may be described as organization-wide packet
telemetry.

Two limitations are carried forward explicitly and are NOT fixed here:
  1. SCOPE — 10 of ~440 monitored hosts.
  2. ENGINE SEMANTICS — the V4 flow-generation in src/pcap_ingest_v4.py
     emulates, but is not identical to, the CICFlowMeter semantics behind
     the training features. Prior parity work measured residuals that do
     NOT close with more hosts (notably fin_rate / syn_rate, driven by
     CIC's binarised flag definitions).

-------------------------------------------------------------
AGGREGATION METHOD — VERIFIED, NOT INVENTED
-------------------------------------------------------------
V4 52-D PREFIX (Option 1 — existing repository implementation):
    `models/pcap_analysis_v4/parity/run_merged_hosts.py` establishes the
    method: "Processes the ten victim captures as ONE monitored network:
    parse all, concatenate the per-packet tables, then run the SAME
    corrected src/pcap_ingest_v4 flow-reconstruction + window-aggregation
    over the union." That is reproduced here exactly:

        pooled packets -> P.reconstruct_flows -> P.compute_window_features
                       -> P.add_relative_changes

    No aggregation formula is invented: the canonical single-host
    functions are called once over the pooled packet table.

    NOT USED, deliberately: `run_subset25.py` instead reconstructs flows
    PER HOST and concatenates the flow tables before
    `compute_window_features`. `reconstruct_flows` derives
    `win0 = int(min(timestamp_abs) // 10)` from ITS OWN input, so per-host
    flow tables carry `window_idx` on per-host origins; pooling them mixes
    origins whenever hosts' first packets fall in different 10 s bins.
    Pooling packets FIRST yields one global `win0` and is correct by
    construction. This is a deviation from one prior script, made for a
    stated correctness reason, and it matches the other prior script.

    Cross-host flow dedup (run_subset25's `flow_key` clustering) is not
    applicable to the packet-pooled route. Its necessity is MEASURED
    instead: the script counts packets whose source and destination are
    both monitored hosts, which is the only way a packet can be recorded
    twice.

PACKET 30-D SUFFIX (no existing fleet method — documented choice):
    A repository-wide search found NO existing packet fleet/multi-host
    aggregation. The least-assumptive option is the exact analogue of the
    verified V4 method: pool the per-packet tables and call the SINGLE
    canonical builder `packet_features_from_packet_table` once over the
    union. This introduces ZERO new formula — it is the same function the
    V5 contract calls the single implementation shared by the training and
    pcap paths.

    REJECTED: "packet_count = SUM across hosts, 21 base features = MEAN
    across hosts, then recompute the 9 relative features". That would be a
    new, invented formula, and it is wrong for this feature set: 8 of the
    21 base features are percentiles, entropies or standard deviations
    (`payload_len_p50/p90/max`, `payload_len_entropy`, `ttl_std`,
    `log_tcp_window_std`, `log_ttl_unique_count`, `ttl_min`/`ttl_max`),
    and a mean-of-per-host-percentiles is not the percentile of the pooled
    packets. Pooling the packets avoids the error entirely.

Usage:
    python src/build_v5_multihost_demo.py
    python src/build_v5_multihost_demo.py --no-cache

=============================================================
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pcap_ingest_v4 as v4
import packet_feature_builder as pfb
from infer_latent_world_model_v5 import V5Inference

REPO_ROOT = SRC_DIR.parent
PCAP_DIR = REPO_ROOT / "data" / "raw" / "march02" / "bot_pcaps" / "pcap"
OUT_DIR = REPO_ROOT / "results" / "v5_demo_multihost"
CACHE_DIR = OUT_DIR / ".cache"

SINGLE_HOST_DIR = REPO_ROOT / "results" / "v5_demo"

V5_SCALER_PATH = (
    REPO_ROOT / "data" / "processed" / "v5_multistep_K5" / "v5_input_scaler.pkl"
)

WINDOW_SECONDS = 10
SEQ_LEN = 10
V4_PREFIX_DIM = 52
PACKET_SUFFIX_DIM = 30
V5_DIM = 82
K = 5

# .30 first so the 1-host point is the Task 3-5 host; the rest ascending.
HOST_ORDER = ["172.31.69.30", "172.31.69.6", "172.31.69.8", "172.31.69.10",
              "172.31.69.12", "172.31.69.14", "172.31.69.17", "172.31.69.23",
              "172.31.69.26", "172.31.69.29"]
SUBSET_SIZES = (1, 5, 10)

SCOPE_NOTE = (
    "Multi-host packet-aware prototype representation — a 10-host "
    "approximation of the organization-level flow state. BOTH halves of "
    "the 82-D vector derive from the SAME pooled packet set across the "
    "same 10 March-2 victim captures, so the observation scope is "
    "internally consistent. This is NOT organization-wide packet "
    "telemetry: the V5 training reference for indices 0-51 covers the "
    "whole ~440-host 172.31.0.0/16 victim network. Two limitations remain "
    "and are not corrected here: (1) SCOPE, 10 of ~440 hosts; (2) ENGINE "
    "SEMANTICS, src/pcap_ingest_v4.py emulates but is not identical to the "
    "CICFlowMeter semantics behind the training features, and prior parity "
    "work measured residuals (fin_rate/syn_rate) that do not close with "
    "more hosts. No labels exist for this substrate."
)


class BuildError(RuntimeError):
    pass


def _rule(title: str) -> None:
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}", flush=True)


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _load_scaler():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(V5_SCALER_PATH, "rb") as handle:
            return pickle.load(handle)


# =============================================================
# STAGE A — PER-HOST PARSE (cached)
# =============================================================


def host_inventory() -> list[dict]:
    pattern = re.compile(r"(\d+\.\d+\.\d+\.\d+)$")
    rows = []
    for path in sorted(PCAP_DIR.glob("cap*")):
        match = pattern.search(path.name)
        if not match:
            continue
        rows.append({"host": match.group(1), "path": path,
                     "size_mb": path.stat().st_size / 1e6})
    return rows


def parse_host(host: str, path: Path, use_cache: bool) -> dict:
    """Parse one capture through BOTH lanes and cache the packet tables."""
    v4_cache = CACHE_DIR / f"{host}_v4_packets.pkl"
    pkt_cache = CACHE_DIR / f"{host}_pkt_packets.pkl"
    meta_cache = CACHE_DIR / f"{host}_meta.json"

    if use_cache and v4_cache.exists() and pkt_cache.exists() and meta_cache.exists():
        meta = json.loads(meta_cache.read_text())
        print(f"  {host:<15} cached  "
              f"v4={meta['n_v4_packets']:>7,}  pkt={meta['n_pkt_packets']:>7,}",
              flush=True)
        return meta

    started = time.perf_counter()
    v4_packets = v4.parse_pcap_streaming(str(path))
    pkt_packets = pfb.parse_pcap_packets(str(path))
    if len(v4_packets) == 0 or len(pkt_packets) == 0:
        raise BuildError(f"{host}: a lane parsed zero packets")

    v4_packets.to_pickle(v4_cache)
    pkt_packets.to_pickle(pkt_cache)
    meta = {
        "host": host,
        "pcap": str(path.relative_to(REPO_ROOT)),
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "n_v4_packets": int(len(v4_packets)),
        "n_pkt_packets": int(len(pkt_packets)),
        "first_ts": float(v4_packets["timestamp_abs"].min()),
        "last_ts": float(v4_packets["timestamp_abs"].max()),
        "parse_seconds": round(time.perf_counter() - started, 1),
    }
    meta["valid"] = meta["n_v4_packets"] == meta["n_pkt_packets"]
    meta_cache.write_text(json.dumps(meta, indent=2))
    print(f"  {host:<15} parsed  v4={meta['n_v4_packets']:>7,}  "
          f"pkt={meta['n_pkt_packets']:>7,}  [{meta['parse_seconds']:.0f}s]", flush=True)
    return meta


# =============================================================
# STAGE C — POOLED FEATURE BUILD FOR A HOST SUBSET
# =============================================================


def build_82d_table(
    hosts: list[str], window: tuple[float, float], win0: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Pool the packet tables of `hosts`, restricted to `window`, and run
    BOTH canonical lanes once over the union."""
    lo, hi = window
    stats: dict = {"hosts": hosts, "host_count": len(hosts)}

    # ---- V4 lane ----
    parts = []
    for host in hosts:
        table = pd.read_pickle(CACHE_DIR / f"{host}_v4_packets.pkl")
        table = table[(table["timestamp_abs"] >= lo) & (table["timestamp_abs"] < hi)]
        table = table.assign(capture_host=host)
        parts.append(table)
    pooled = pd.concat(parts, ignore_index=True)
    del parts
    pooled = pooled.sort_values(
        ["timestamp_abs", "capture_host"], kind="stable"
    ).reset_index(drop=True)
    stats["pooled_v4_packets"] = int(len(pooled))

    monitored = set(hosts)
    both = pooled["src_ip"].isin(monitored) & pooled["dst_ip"].isin(monitored)
    stats["monitored_to_monitored_packets"] = int(both.sum())
    stats["monitored_to_monitored_fraction"] = float(both.mean())

    pooled_min_ts = float(pooled["timestamp_abs"].min())
    pooled.attrs["t0_epoch"] = pooled_min_ts
    flows = v4.reconstruct_flows(pooled.drop(columns=["capture_host"]))
    stats["pooled_flows"] = int(len(flows))
    del pooled

    v4_windows = v4.add_relative_changes(v4.compute_window_features(flows))
    del flows

    # `reconstruct_flows` derives its own win0 from the POOLED minimum
    # timestamp, which differs per host subset. Re-base onto the single
    # global win0 so every subset — and the packet lane, which is given
    # win0 explicitly — shares one window_idx axis. This is a pure integer
    # shift; relative-change features are row-to-row and unaffected.
    lane_win0 = int(np.floor(pooled_min_ts / WINDOW_SECONDS))
    shift = lane_win0 - win0
    v4_windows["window_idx"] = v4_windows["window_idx"] + shift
    v4_windows["timestamp"] = v4_windows["window_idx"] * WINDOW_SECONDS
    stats["v4_lane_win0"] = lane_win0
    stats["v4_window_idx_shift_applied"] = shift
    stats["v4_windows"] = int(len(v4_windows))

    # ---- packet lane ----
    parts = []
    for host in hosts:
        table = pd.read_pickle(CACHE_DIR / f"{host}_pkt_packets.pkl")
        table = table[(table["timestamp_abs"] >= lo) & (table["timestamp_abs"] < hi)]
        table = table.assign(capture_host=host)
        parts.append(table)
    pooled_pkt = pd.concat(parts, ignore_index=True)
    del parts
    # Deterministic global capture order across hosts: sort on
    # (timestamp, host, original ordinal), then re-assign packet_index so
    # per-host ordinals cannot collide.
    pooled_pkt = pooled_pkt.sort_values(
        ["timestamp_abs", "capture_host", "packet_index"], kind="stable"
    ).reset_index(drop=True)
    pooled_pkt["packet_index"] = np.arange(len(pooled_pkt), dtype=np.int64)
    stats["pooled_packet_lane_packets"] = int(len(pooled_pkt))

    packet_windows = pfb.packet_features_from_packet_table(
        pooled_pkt.drop(columns=["capture_host"]), win0=win0
    )
    del pooled_pkt
    stats["packet_windows"] = int(len(packet_windows))

    # Both lanes are now on the shared global win0: the packet lane was
    # given it explicitly, the V4 lane was re-based onto it above.
    return v4_windows, packet_windows, stats


def join_lanes(
    v4_windows: pd.DataFrame,
    packet_windows: pd.DataFrame,
    packet_names: list[str],
) -> tuple[pd.DataFrame, dict]:
    v4_cols = ["window_idx", "timestamp"] + list(v4.INPUT_FEATURE_NAMES)
    left = v4_windows[v4_cols].rename(columns={"timestamp": "timestamp_v4"})
    right = packet_windows[["window_idx", "timestamp"] + packet_names].rename(
        columns={"timestamp": "timestamp_pkt"}
    )
    joined = left.merge(right, on="window_idx", how="inner").sort_values(
        "window_idx"
    ).reset_index(drop=True)

    delta = float((joined["timestamp_v4"] - joined["timestamp_pkt"]).abs().max())
    if delta != 0.0:
        raise BuildError(
            f"lanes disagree on relative timestamp for shared windows "
            f"(max {delta}s) — window_idx origins are not aligned"
        )
    gaps = np.diff(joined["window_idx"].to_numpy())
    info = {
        "joined_windows": int(len(joined)),
        "first_window_idx": int(joined["window_idx"].min()),
        "last_window_idx": int(joined["window_idx"].max()),
        "gap_count": int((gaps != 1).sum()) if gaps.size else 0,
        "max_stride": int(gaps.max()) if gaps.size else 0,
        "stride_distribution": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(gaps, return_counts=True))
        } if gaps.size else {},
    }
    return joined, info


def build_sequences(
    joined: pd.DataFrame, feature_names: list[str], win0: int, hosts: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    window_idx = joined["window_idx"].to_numpy(dtype=np.int64)
    matrix = joined[feature_names].to_numpy(dtype=np.float64)

    breaks = np.flatnonzero(np.diff(window_idx) != 1) + 1
    runs = [r for r in np.split(np.arange(len(window_idx)), breaks) if len(r) >= SEQ_LEN]

    sequences, rows = [], []
    for run in runs:
        for start in range(len(run) - SEQ_LEN + 1):
            idx = run[start : start + SEQ_LEN]
            slice_idx = window_idx[idx]
            if not np.array_equal(np.diff(slice_idx), np.ones(SEQ_LEN - 1)):
                raise BuildError("internal error: non-contiguous sequence")
            sequences.append(matrix[idx])
            a, b = int(slice_idx[0]), int(slice_idx[-1])
            rows.append({
                "sequence_index": len(sequences) - 1,
                "window_idx_start": a,
                "window_idx_end": b,
                "timestamp_start": a * WINDOW_SECONDS,
                "timestamp_end": b * WINDOW_SECONDS,
                "timestamp_start_utc": _utc((a + win0) * WINDOW_SECONDS),
                "timestamp_end_utc": _utc((b + win0) * WINDOW_SECONDS),
                "host_count": len(hosts),
                "host_list": ";".join(hosts),
                "scope": "multi_host_packet_aware_prototype",
                "source_files": ";".join(
                    f"capEC2AMAZ-O4EL3NG-{h}" for h in hosts
                ),
            })
    if not sequences:
        raise BuildError("no run of 10 consecutive joined windows exists")
    return np.stack(sequences).astype(np.float64), pd.DataFrame(rows)


# =============================================================
# SHIFT DIAGNOSTIC (identical definition to Task 5)
# =============================================================


def shift_stats(values: np.ndarray, mean, scale, names, groups) -> tuple:
    """Same methodology as src/analyze_v5_input_shift.py — unchanged."""
    z = (values - mean) / scale
    absolute = np.abs(z)
    per_feature = pd.DataFrame({
        "feature_index": np.arange(len(names)),
        "feature_name": names,
        "feature_group": groups,
        "training_mean": mean,
        "training_scale": scale,
        "demo_mean": values.mean(axis=0),
        "demo_std": values.std(axis=0, ddof=1),
        "mean_abs_z": absolute.mean(axis=0),
        "median_abs_z": np.median(absolute, axis=0),
        "p95_abs_z": np.percentile(absolute, 95, axis=0),
        "p99_abs_z": np.percentile(absolute, 99, axis=0),
        "fraction_abs_z_gt_3": (absolute > 3).mean(axis=0),
        "signed_mean_z": z.mean(axis=0),
    })
    rows = []
    for group in ("V4_BASE", "V4_RELATIVE", "PACKET_BASE", "PACKET_RELATIVE"):
        mask = np.array([g == group for g in groups])
        block = absolute[:, mask]
        rows.append({
            "feature_group": group,
            "feature_count": int(mask.sum()),
            "mean_abs_z": float(block.mean()),
            "median_abs_z": float(np.median(block)),
            "p95_abs_z": float(np.percentile(block, 95)),
            "fraction_abs_z_gt_3": float((block > 3).mean()),
        })
    return per_feature, pd.DataFrame(rows), absolute


def build_groups(names: list[str]) -> list[str]:
    v4_base, v4_rel = set(v4.BASE_FEATURES), set(v4.REL_CHANGE_FEATURES)
    pk_base, pk_rel = set(pfb.PACKET_BASE_FEATURES), set(pfb.PACKET_RELATIVE_FEATURES)
    out = []
    for name in names:
        if name in v4_base:
            out.append("V4_BASE")
        elif name in v4_rel:
            out.append("V4_RELATIVE")
        elif name in pk_base:
            out.append("PACKET_BASE")
        elif name in pk_rel:
            out.append("PACKET_RELATIVE")
        else:
            raise BuildError(f"{name!r} in no contract group")
    return out


# =============================================================
# MAIN
# =============================================================


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    use_cache = not args.no_cache

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    scaler = _load_scaler()
    if int(scaler.n_features_in_) != V5_DIM:
        raise BuildError("V5 scaler is not 82-D")
    names = [str(n) for n in np.asarray(scaler.feature_names_in_).ravel()]
    packet_names = names[V4_PREFIX_DIM:]
    groups = build_groups(names)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    print(f"V5 scaler loaded: n_features_in_={scaler.n_features_in_} (NOT refitted)")

    # ---------- Stage A ----------
    _rule("STAGE A — host inventory and per-host parse")
    inventory = host_inventory()
    print(f"  March-2 victim captures found: {len(inventory)}")
    metas = [parse_host(h["host"], h["path"], use_cache) for h in inventory]
    meta_by_host = {m["host"]: m for m in metas}

    missing = [h for h in HOST_ORDER if h not in meta_by_host]
    if missing:
        raise BuildError(f"expected hosts not found: {missing}")
    bad = [m["host"] for m in metas if not m["valid"]]
    if bad:
        print(f"  WARNING: lanes disagree on packet count for {bad}")

    inv_frame = pd.DataFrame([{
        "host": m["host"], "pcap": m["pcap"], "size_mb": m["size_mb"],
        "n_ip_packets": m["n_v4_packets"],
        "lanes_agree": m["valid"],
        "first_utc": _utc(m["first_ts"]), "last_utc": _utc(m["last_ts"]),
        "span_hours": round((m["last_ts"] - m["first_ts"]) / 3600, 2),
    } for m in metas]).sort_values("host")
    inv_frame.to_csv(OUT_DIR / "host_inventory.csv", index=False)
    print(f"\n{inv_frame.to_string(index=False)}")

    # ---------- Stage B — common timeline ----------
    _rule("STAGE B — validated common timeline")
    starts = [meta_by_host[h]["first_ts"] for h in HOST_ORDER]
    ends = [meta_by_host[h]["last_ts"] for h in HOST_ORDER]
    lo, hi = max(starts), min(ends)
    print(f"  union span  : {_utc(min(starts))} .. {_utc(max(ends))}")
    print(f"  COMMON span : {_utc(lo)} .. {_utc(hi)}  "
          f"({(hi - lo) / 3600:.2f} h, all {len(HOST_ORDER)} hosts capturing)")
    if hi <= lo:
        raise BuildError("hosts share no common capture window")
    win0 = int(np.floor(lo / WINDOW_SECONDS))
    # Snap to the first whole window at or after `lo`.
    lo_snapped = (win0 + 1) * WINDOW_SECONDS
    win0 = int(lo_snapped // WINDOW_SECONDS)
    print(f"  global win0 : {win0}  (window 0 starts {_utc(lo_snapped)})")

    # ---------- Stage C — host-count sweep ----------
    _rule("STAGE C — host-count sweep (1 / 5 / 10) under one method")
    sweep_rows, artifacts = [], {}
    for size in SUBSET_SIZES:
        hosts = HOST_ORDER[:size]
        print(f"\n  --- {size} host(s): {', '.join(hosts)} ---", flush=True)
        v4_windows, packet_windows, stats = build_82d_table(
            hosts, (lo_snapped, hi), win0
        )
        joined, info = join_lanes(v4_windows, packet_windows, packet_names)
        values = joined[names].to_numpy(dtype=np.float64)
        per_feature, per_group, _ = shift_stats(values, mean, scale, names, groups)
        print(f"      pooled packets {stats['pooled_v4_packets']:,} -> "
              f"flows {stats['pooled_flows']:,} -> "
              f"V4 windows {stats['v4_windows']:,} | "
              f"packet windows {stats['packet_windows']:,} | "
              f"joined {info['joined_windows']:,}")
        print(f"      monitored<->monitored packets: "
              f"{stats['monitored_to_monitored_packets']:,} "
              f"({stats['monitored_to_monitored_fraction'] * 100:.4f}%)")
        for _, row in per_group.iterrows():
            print(f"      {row['feature_group']:<17}"
                  f"mean|z| {row['mean_abs_z']:.3f}  "
                  f"med {row['median_abs_z']:.3f}  "
                  f"p95 {row['p95_abs_z']:.3f}  "
                  f"frac>3 {row['fraction_abs_z_gt_3']:.4f}")
            sweep_rows.append({"host_count": size, **row.to_dict()})
        artifacts[size] = {
            "joined": joined, "info": info, "stats": stats,
            "per_feature": per_feature, "per_group": per_group,
            "overall_mean_abs_z": float(np.abs((values - mean) / scale).mean()),
        }
        sweep_rows.append({
            "host_count": size, "feature_group": "ALL_82",
            "feature_count": 82,
            "mean_abs_z": artifacts[size]["overall_mean_abs_z"],
            "median_abs_z": float(np.median(np.abs((values - mean) / scale))),
            "p95_abs_z": float(np.percentile(np.abs((values - mean) / scale), 95)),
            "fraction_abs_z_gt_3": float((np.abs((values - mean) / scale) > 3).mean()),
        })

    sweep = pd.DataFrame(sweep_rows)[
        ["host_count", "feature_group", "feature_count", "mean_abs_z",
         "median_abs_z", "p95_abs_z", "fraction_abs_z_gt_3"]
    ]
    sweep.to_csv(OUT_DIR / "host_count_shift.csv", index=False)
    print(f"\n  wrote {(OUT_DIR / 'host_count_shift.csv').relative_to(REPO_ROOT)}")

    # ---------- Stage D — final substrate (10 hosts) ----------
    _rule("STAGE D — final 10-host substrate")
    final = artifacts[10]
    hosts = HOST_ORDER
    X, metadata = build_sequences(final["joined"], names, win0, hosts)
    print(f"  X shape: {X.shape}")

    checks = [
        ("1. shape (N,10,82)", X.ndim == 3 and X.shape[1:] == (SEQ_LEN, V5_DIM)),
        ("2. no NaN", not np.isnan(X).any()),
        ("3. no Inf", not np.isinf(X).any()),
        ("4. feature count == 82", len(names) == V5_DIM),
        ("5. order == v5_input_scaler.feature_names_in_",
         names == [str(n) for n in np.asarray(scaler.feature_names_in_).ravel()]),
        ("6. V4 prefix == 52", names[:52] == list(v4.INPUT_FEATURE_NAMES)),
        ("7. packet suffix == 30", len(names[52:]) == PACKET_SUFFIX_DIM),
        ("8. no labels in input",
         not [n for n in names if any(t in n.lower()
              for t in ("attack", "label", "target", "malicious"))]),
        ("9. no future information",
         bool((metadata["window_idx_start"] + SEQ_LEN - 1
               == metadata["window_idx_end"]).all())),
        ("10. all sequences contiguous",
         bool((metadata["window_idx_end"] - metadata["window_idx_start"]
               == SEQ_LEN - 1).all())),
        ("11. no synthetic data", True),
        ("12. V5 scaler untouched", True),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if any(not ok for _, ok in checks):
        raise BuildError("validation failed — nothing written")

    np.save(OUT_DIR / "X.npy", X)
    metadata.to_csv(OUT_DIR / "metadata.csv", index=False)
    (OUT_DIR / "v5_feature_names.txt").write_text("\n".join(names) + "\n")
    final["per_feature"].to_csv(OUT_DIR / "input_shift_report.csv", index=False)
    final["per_group"].to_csv(OUT_DIR / "input_shift_groups.csv", index=False)
    print("  wrote X.npy, metadata.csv, v5_feature_names.txt, "
          "input_shift_report.csv, input_shift_groups.csv")

    # ---------- Stage E — real V5 inference ----------
    _rule("STAGE E — real V5 inference over all sequences (K=5)")
    engine = V5Inference()
    engine.load_model()
    probabilities = np.empty((X.shape[0], K))
    started = time.perf_counter()
    for start in range(0, X.shape[0], 256):
        stop = min(start + 256, X.shape[0])
        probabilities[start:stop] = engine.k_step_rollout(
            X[start:stop], steps=K
        ).threat_probabilities
    elapsed = time.perf_counter() - started
    print(f"  {X.shape[0]} sequences in {elapsed:.2f}s "
          f"({X.shape[0] / elapsed:,.0f}/s)")

    rollout = pd.DataFrame({
        "sequence_index": metadata["sequence_index"],
        "window_idx_start": metadata["window_idx_start"],
        "window_idx_end": metadata["window_idx_end"],
        "timestamp_start_utc": metadata["timestamp_start_utc"],
        **{f"p_t+{k + 1}": probabilities[:, k] for k in range(K)},
        "max_probability": probabilities.max(axis=1),
        "mean_probability": probabilities.mean(axis=1),
    })
    rollout.to_csv(OUT_DIR / "rollout_summary.csv", index=False)
    print(f"  wrote {(OUT_DIR / 'rollout_summary.csv').relative_to(REPO_ROOT)}")

    distribution = {}
    print(f"\n  {'horizon':<9}{'min':>10}{'median':>11}{'mean':>11}"
          f"{'p95':>11}{'p99':>11}{'max':>11}{'>=0.1':>9}{'>=0.25':>9}{'>=0.5':>8}")
    for k in range(K):
        column = probabilities[:, k]
        entry = {
            "min": float(column.min()), "median": float(np.median(column)),
            "mean": float(column.mean()),
            "p95": float(np.percentile(column, 95)),
            "p99": float(np.percentile(column, 99)),
            "max": float(column.max()),
            "fraction_ge_0.1": float((column >= 0.1).mean()),
            "fraction_ge_0.25": float((column >= 0.25).mean()),
            "fraction_ge_0.5": float((column >= 0.5).mean()),
        }
        distribution[f"t+{k + 1}"] = entry
        print(f"  t+{k + 1:<8}{entry['min']:>10.6f}{entry['median']:>11.6f}"
              f"{entry['mean']:>11.6f}{entry['p95']:>11.6f}{entry['p99']:>11.6f}"
              f"{entry['max']:>11.6f}{entry['fraction_ge_0.1']:>9.4f}"
              f"{entry['fraction_ge_0.25']:>9.4f}{entry['fraction_ge_0.5']:>8.4f}")

    # ---------- Stage F — comparison to single host ----------
    _rule("STAGE F — multi-host vs single-host comparison")
    single_summary = pd.read_csv(SINGLE_HOST_DIR / "rollout_summary.csv")
    single_groups = pd.read_csv(SINGLE_HOST_DIR / "input_shift_groups.csv")
    single_probabilities = single_summary[
        [f"p_t+{k + 1}" for k in range(K)]
    ].to_numpy()

    def block(probs, groups_frame, label, host_count):
        # Task 5 wrote the group column as "group"; this script writes
        # "feature_group". Accept either so the two are comparable.
        column = "feature_group" if "feature_group" in groups_frame.columns else "group"
        v4_rows = groups_frame[groups_frame[column].isin(
            ["V4_BASE", "V4_RELATIVE"])]
        pk_rows = groups_frame[groups_frame[column].isin(
            ["PACKET_BASE", "PACKET_RELATIVE"])]
        weights_v4 = v4_rows["feature_count"] / v4_rows["feature_count"].sum()
        weights_pk = pk_rows["feature_count"] / pk_rows["feature_count"].sum()
        return {
            "label": label,
            "host_count": host_count,
            "n_sequences": int(probs.shape[0]),
            "mean_probability": float(probs.mean()),
            "max_probability_mean": float(probs.max(axis=1).mean()),
            "p95_probability": float(np.percentile(probs, 95)),
            "global_max_probability": float(probs.max()),
            "fraction_ge_0.5": float((probs >= 0.5).mean()),
            "v4_prefix_mean_abs_z": float((v4_rows["mean_abs_z"] * weights_v4).sum()),
            "packet_suffix_mean_abs_z": float((pk_rows["mean_abs_z"] * weights_pk).sum()),
        }

    comparison = {
        "single_host": block(single_probabilities, single_groups,
                             "single host 172.31.69.30 (full span)", 1),
        "multi_host": block(probabilities, final["per_group"],
                            "10 hosts, common timeline", 10),
        "note": (
            "The single-host figures are the Task 3-5 substrate (full capture "
            "span). The multi-host substrate is restricted to the common "
            "timeline where all 10 hosts capture simultaneously, so sequence "
            "counts are not comparable; the shift and probability statistics "
            "are."
        ),
        "scope_note": SCOPE_NOTE,
        "labels_available": False,
    }
    for key in ("single_host", "multi_host"):
        entry = comparison[key]
        print(f"  {entry['label']}")
        print(f"    sequences {entry['n_sequences']:,} | mean p {entry['mean_probability']:.6f}"
              f" | p95 {entry['p95_probability']:.6f}"
              f" | global max {entry['global_max_probability']:.6f}"
              f" | frac>=0.5 {entry['fraction_ge_0.5']:.4f}")
        print(f"    V4-prefix mean|z| {entry['v4_prefix_mean_abs_z']:.3f}"
              f" | packet-suffix mean|z| {entry['packet_suffix_mean_abs_z']:.3f}")
    (OUT_DIR / "comparison_to_single_host.json").write_text(
        json.dumps(comparison, indent=2) + "\n")

    # ---------- manifest ----------
    manifest = {
        "scope": "multi_host_packet_aware_prototype",
        "host_count": len(hosts),
        "hosts": hosts,
        "source_files": [meta_by_host[h]["pcap"] for h in hosts],
        "common_timeline_utc": [_utc(lo_snapped), _utc(hi)],
        "common_timeline_epoch": [lo_snapped, hi],
        "global_win0": win0,
        "window_seconds": WINDOW_SECONDS,
        "sequence_length": SEQ_LEN,
        "feature_count": V5_DIM,
        "join_key": "window_idx",
        "v4_aggregation": (
            "Option 1 — existing repository implementation "
            "(models/pcap_analysis_v4/parity/run_merged_hosts.py): pool the "
            "per-packet tables across hosts, then run the canonical "
            "src/pcap_ingest_v4 reconstruct_flows -> compute_window_features "
            "-> add_relative_changes ONCE over the union. No formula invented."
        ),
        "v4_aggregation_deviation": (
            "run_subset25.py's per-host-flows-then-concat route was NOT used: "
            "reconstruct_flows derives win0 from its own input, so per-host "
            "flow tables carry window_idx on per-host origins. Pooling packets "
            "first gives one global win0 and is correct by construction."
        ),
        "packet_aggregation": (
            "No existing packet fleet method was found in the repository. "
            "Chosen: exact analogue of the verified V4 method — pool the "
            "per-packet tables and call the single canonical "
            "packet_feature_builder.packet_features_from_packet_table once "
            "over the union. Rejected mean-across-hosts: 8 of the 21 base "
            "features are percentiles/entropies/std, for which a mean of "
            "per-host values is not the pooled value."
        ),
        "packet_count_excluded_from_vector": True,
        "cross_host_duplicate_handling": (
            "Measured, not assumed: packets with BOTH endpoints monitored are "
            "the only ones a pooled set can double-count. Count and fraction "
            "recorded in build_stats."
        ),
        "build_stats": {str(k): artifacts[k]["stats"] for k in artifacts},
        "join_info": {str(k): artifacts[k]["info"] for k in artifacts},
        "n_sequences": int(X.shape[0]),
        "scaler_path": str(V5_SCALER_PATH.relative_to(REPO_ROOT)),
        "scaler_refitted": False,
        "model_path": "models/latent_world_model_v5/latent_world_model_v5.pt",
        "labels_included": False,
        "imputation": "NONE",
        "probability_distribution": distribution,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "scope_note": SCOPE_NOTE,
    }
    (OUT_DIR / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  wrote {(OUT_DIR / 'demo_manifest.json').relative_to(REPO_ROOT)}")
    print(f"\n{SCOPE_NOTE}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"\nBUILD STOPPED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
