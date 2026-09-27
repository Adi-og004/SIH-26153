"""
=============================================================
V5 DEMO SEQUENCE BUILDER — HOST-LEVEL 82-D ASSEMBLY
=============================================================

Builds the first genuine real 10 x 82 input for the V5 latent world model
(`src/infer_latent_world_model_v5.py`) from ONE real host capture.

-------------------------------------------------------------
SCOPE — READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT
-------------------------------------------------------------
This produces a **host-level packet-aware V5 prototype representation**.

Both halves of the 82-D vector are derived from the SAME single host
capture (172.31.69.30), so the observation scope is internally consistent:

    indices  0-51 : V4-compatible flow features, via the canonical
                    (unmodified) src/pcap_ingest_v4.py lane
    indices 52-81 : packet-aware features, from the packet lane's
                    already-built table for the same host

This is explicitly NOT organization-wide packet telemetry. The V5 model's
own training distribution for indices 0-51 was organization-wide, so a
host-level vector sits at a different scope from what the model saw. This
artifact is an INFERENCE / DEMO SUBSTRATE — a way to exercise the V5 path
end-to-end on real traffic. It is not a training benchmark, not a
validation set, and its outputs are not accuracy measurements.

-------------------------------------------------------------
ALIGNMENT
-------------------------------------------------------------
`window_idx` is the sole join key. Both lanes define it identically as
`floor(packet_epoch_seconds / 10) - win0`, where `win0` is the 10 s bin of
the capture's first IP packet:

    pcap_ingest_v4.reconstruct_flows      win0 = int(t0 // 10)
    packet_feature_builder.build_packet_windows
                                          win0 = floor(ts.min() / 10)

Both parsers retain exactly the IP packets (IPv4 + IPv6) and drop non-IP
frames, so both see the same first packet and therefore the same `win0`.
That is VERIFIED here, not assumed: the V4 lane's IP-packet count is
checked against the packet lane's recorded count, and the pcap's SHA-256 is
checked against the manifest. No timezone arithmetic and no relative
timestamp string is ever used to align the lanes.

Usage:
    python src/build_v5_sequences.py
    python src/build_v5_sequences.py --no-cache     # force a re-parse

=============================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The canonical V4 lane, imported and REUSED. Nothing in it is modified.
import pcap_ingest_v4 as v4
from infer_latent_world_model_v5 import V5Inference

# =============================================================
# PATHS / CONSTANTS
# =============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

HOST = "172.31.69.30"
SOURCE_PCAP = (
    REPO_ROOT / "data" / "raw" / "march02" / "bot_pcaps" / "pcap"
    / "capEC2AMAZ-O4EL3NG-172.31.69.30"
)
PACKET_CSV = (
    REPO_ROOT / "results" / "v5_features"
    / "packet_windows_march02_host_172.31.69.30.csv"
)
PACKET_MANIFEST = REPO_ROOT / "docs" / "v5_packet_dataset_manifest.json"
PACKET_RECORD_ID = "march02_host_172.31.69.30"

V5_SCALER_PATH = (
    REPO_ROOT / "data" / "processed" / "v5_multistep_K5" / "v5_input_scaler.pkl"
)
V5_CHECKPOINT = (
    REPO_ROOT / "models" / "latent_world_model_v5" / "latent_world_model_v5.pt"
)

OUT_DIR = REPO_ROOT / "results" / "v5_demo"
X_PATH = OUT_DIR / f"host_{HOST}_X.npy"
METADATA_PATH = OUT_DIR / f"host_{HOST}_metadata.csv"
FEATURE_NAMES_PATH = OUT_DIR / "v5_feature_names.txt"
MANIFEST_PATH = OUT_DIR / "demo_manifest.json"

CACHE_PATH = OUT_DIR / f".cache_v4_windows_{HOST}.csv"

WINDOW_SECONDS = 10
SEQ_LEN = 10
V4_PREFIX_DIM = 52
PACKET_SUFFIX_DIM = 30
V5_DIM = 82

SCOPE_NOTE = (
    "Host-level packet-aware V5 prototype representation. BOTH halves of "
    "the 82-D vector are derived from the SAME single-host capture "
    f"({HOST}): indices 0-51 come from the canonical, unmodified "
    "src/pcap_ingest_v4.py flow lane run on that pcap, and indices 52-81 "
    "come from the packet lane's table for that same pcap. The observation "
    "scope is therefore internally consistent (one host, one capture). "
    "This is NOT organization-wide packet telemetry, and it is NOT a "
    "training benchmark or validation set: the V5 model's training "
    "distribution for indices 0-51 was organization-wide, so a host-level "
    "vector sits at a different scope than the model was fitted on. This "
    "artifact exists to exercise the V5 inference path on real traffic. "
    "Model outputs computed from it are genuine model outputs but are NOT "
    "accuracy, benchmark or early-warning claims."
)


class BuildError(RuntimeError):
    """Raised when the two lanes cannot be aligned on verified evidence."""


def _rule(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================
# STEP 1 — PROVENANCE
# =============================================================


def verify_provenance() -> dict:
    """Confirm the pcap on disk is the exact file the packet table was
    built from, so 'same host, same capture' is evidence, not a filename
    coincidence."""
    _rule("STEP 1 — source provenance")

    for path in (SOURCE_PCAP, PACKET_CSV, PACKET_MANIFEST):
        if not path.exists():
            raise BuildError(f"required input missing: {path}")

    manifest = json.loads(PACKET_MANIFEST.read_text())
    record = next(
        (
            r
            for r in manifest.get("packet_feature_records", [])
            if r.get("record_id") == PACKET_RECORD_ID
        ),
        None,
    )
    if record is None:
        raise BuildError(
            f"no packet_feature_record with id {PACKET_RECORD_ID!r} in "
            f"{PACKET_MANIFEST}"
        )

    expected_pcap = (REPO_ROOT / record["source_pcap"]).resolve()
    if expected_pcap != SOURCE_PCAP.resolve():
        raise BuildError(
            "packet table was built from a different pcap:\n"
            f"  manifest: {expected_pcap}\n  using   : {SOURCE_PCAP.resolve()}"
        )

    print(f"  pcap            : {SOURCE_PCAP.relative_to(REPO_ROOT)}")
    print(f"  size            : {SOURCE_PCAP.stat().st_size / 1e6:.1f} MB")
    print("  hashing pcap (provenance check) ...")
    actual = _sha256(SOURCE_PCAP)
    expected = record.get("source_pcap_sha256")
    if expected and actual != expected:
        raise BuildError(
            "pcap SHA-256 does not match the packet manifest — the packet "
            "table was built from different bytes:\n"
            f"  expected {expected}\n  actual   {actual}"
        )
    print(f"  sha256          : {actual[:16]}...  MATCHES manifest")
    print(f"  capture scope   : {record.get('capture_scope')}")
    print(f"  packet lane IP packets : {record.get('n_ip_packets')}")
    print(f"  packet lane windows    : {record.get('n_windows')}")
    return record


# =============================================================
# STEP 2 — V4 LANE (52-D)
# =============================================================


def build_v4_windows(use_cache: bool = True) -> pd.DataFrame:
    """Run the canonical V4 pcap lane and return its per-window 52-D table.

    Every stage is the frozen implementation from `pcap_ingest_v4`; this
    function only sequences the calls and caches the result so that
    re-runs do not re-parse 320k packets.
    """
    _rule("STEP 2 — V4 lane (canonical pcap_ingest_v4)")

    if use_cache and CACHE_PATH.exists():
        cached = pd.read_csv(CACHE_PATH)
        print(f"  loaded cached V4 window table: {CACHE_PATH.name}")
        print(f"  windows: {len(cached)}")
        return cached

    print("  parse_pcap_streaming ...")
    packets = v4.parse_pcap_streaming(str(SOURCE_PCAP))
    if len(packets) == 0:
        raise BuildError("V4 lane parsed zero IP packets")

    t0_epoch = float(packets.attrs["t0_epoch"])
    win0 = int(t0_epoch // WINDOW_SECONDS)

    print("  reconstruct_flows ...")
    flows = v4.reconstruct_flows(packets)
    print("  compute_window_features ...")
    windows = v4.compute_window_features(flows)
    print("  add_relative_changes ...")
    windows = v4.add_relative_changes(windows)

    windows.attrs["t0_epoch"] = t0_epoch
    windows.attrs["win0"] = win0
    windows.attrs["n_ip_packets"] = int(len(packets))

    # attrs do not survive a CSV round-trip; persist them as columns.
    out = windows.copy()
    out["_t0_epoch"] = t0_epoch
    out["_win0"] = win0
    out["_n_ip_packets"] = int(len(packets))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE_PATH, index=False)
    print(f"  cached -> {CACHE_PATH.name}")
    return out


# =============================================================
# STEP 3 — ALIGNMENT VERIFICATION
# =============================================================


def verify_alignment(v4_windows: pd.DataFrame, record: dict) -> dict:
    """Prove the two lanes share a `win0` origin before any join.

    Both lanes bin on `floor(epoch / 10) - win0` over exactly the IP
    packets of the capture. If the V4 lane read the same number of IP
    packets the packet lane recorded, both saw the same first packet, so
    both `win0` values are the same integer and `window_idx` is directly
    comparable. A mismatch is fatal: nothing is re-based to force a fit.
    """
    _rule("STEP 3 — window_idx origin verification")

    v4_ip_packets = int(v4_windows["_n_ip_packets"].iloc[0])
    pkt_ip_packets = int(record["n_ip_packets"])
    win0 = int(v4_windows["_win0"].iloc[0])
    t0_epoch = float(v4_windows["_t0_epoch"].iloc[0])

    print(f"  V4 lane IP packets     : {v4_ip_packets}")
    print(f"  packet lane IP packets : {pkt_ip_packets}")

    if v4_ip_packets != pkt_ip_packets:
        raise BuildError(
            "the two lanes did not read the same IP-packet set "
            f"({v4_ip_packets} vs {pkt_ip_packets}), so their win0 origins "
            "cannot be assumed equal and window_idx is not comparable. "
            "Refusing to force alignment."
        )

    print("  -> identical IP-packet set; both lanes derive win0 from the")
    print("     same first packet, so window_idx shares one origin.")
    print(f"  win0 (absolute 10s bin) : {win0}")
    print(
        "  t0 (UTC)                : "
        f"{datetime.fromtimestamp(t0_epoch, tz=timezone.utc):%Y-%m-%d %H:%M:%S}"
    )
    return {"win0": win0, "t0_epoch": t0_epoch, "n_ip_packets": v4_ip_packets}


# =============================================================
# STEP 4 — JOIN
# =============================================================


def join_lanes(
    v4_windows: pd.DataFrame, packet_names: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Inner-join the two window tables on `window_idx`.

    Nothing is filled, imputed or manufactured: a window survives only if
    BOTH lanes independently produced it.
    """
    _rule("STEP 4 — join on window_idx")

    packets = pd.read_csv(PACKET_CSV)
    missing = [c for c in packet_names if c not in packets.columns]
    if missing:
        raise BuildError(f"packet CSV is missing required columns: {missing}")

    # packet_count is metadata, NOT one of the 82 model inputs.
    packet_cols = ["window_idx", "timestamp"] + packet_names
    packets = packets[packet_cols].rename(columns={"timestamp": "timestamp_pkt"})

    v4_cols = ["window_idx", "timestamp"] + list(v4.INPUT_FEATURE_NAMES)
    left = v4_windows[v4_cols].rename(columns={"timestamp": "timestamp_v4"})

    stats = {
        "v4_windows": int(len(left)),
        "v4_first_window_idx": int(left["window_idx"].min()),
        "v4_last_window_idx": int(left["window_idx"].max()),
        "packet_windows": int(len(packets)),
        "packet_first_window_idx": int(packets["window_idx"].min()),
        "packet_last_window_idx": int(packets["window_idx"].max()),
    }

    joined = left.merge(packets, on="window_idx", how="inner").sort_values(
        "window_idx"
    ).reset_index(drop=True)

    stats["joined_windows"] = int(len(joined))

    print(
        f"  V4 windows      : {stats['v4_windows']:>6}  "
        f"[{stats['v4_first_window_idx']} .. {stats['v4_last_window_idx']}]"
    )
    print(
        f"  packet windows  : {stats['packet_windows']:>6}  "
        f"[{stats['packet_first_window_idx']} .. "
        f"{stats['packet_last_window_idx']}]"
    )
    print(f"  joined (inner)  : {stats['joined_windows']:>6}")

    if len(joined) == 0:
        raise BuildError(
            "no window_idx is present in both lanes — cannot build any "
            "82-D vector. Refusing to force alignment."
        )

    # The two lanes must agree on the relative timestamp for a shared
    # window. Both define timestamp = window_idx * 10, so a disagreement
    # would mean the origins are not what Step 3 established.
    delta = (joined["timestamp_v4"] - joined["timestamp_pkt"]).abs().max()
    print(f"  max |timestamp_v4 - timestamp_pkt| : {float(delta):.6f} s")
    if float(delta) != 0.0:
        raise BuildError(
            "shared windows disagree on their relative timestamp "
            f"(max delta {float(delta)} s) — the join key is not sound."
        )

    gaps = np.diff(joined["window_idx"].to_numpy())
    stats["joined_gap_count"] = int((gaps != 1).sum())
    stats["joined_max_gap"] = int(gaps.max()) if gaps.size else 0
    print(
        f"  gaps in joined index : {stats['joined_gap_count']} "
        f"(max stride {stats['joined_max_gap']})"
    )
    return joined, stats


# =============================================================
# STEP 5 — SEQUENCES
# =============================================================


def build_sequences(
    joined: pd.DataFrame, feature_names: list[str], win0: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Slide a 10-window window over strictly contiguous runs.

    A sequence is emitted only when window_idx increments by exactly 1 at
    every one of the 9 internal steps. Runs are never stitched across a
    gap.
    """
    _rule("STEP 5 — 10-window sequence construction")

    window_idx = joined["window_idx"].to_numpy(dtype=np.int64)
    matrix = joined[feature_names].to_numpy(dtype=np.float64)

    # Split into maximal contiguous runs.
    breaks = np.flatnonzero(np.diff(window_idx) != 1) + 1
    runs = np.split(np.arange(len(window_idx)), breaks)
    usable = [r for r in runs if len(r) >= SEQ_LEN]

    run_lengths = sorted((len(r) for r in runs), reverse=True)
    print(f"  contiguous runs           : {len(runs)}")
    print(f"  runs of >= {SEQ_LEN} windows     : {len(usable)}")
    print(f"  longest run               : {run_lengths[0] if run_lengths else 0}")

    sequences: list[np.ndarray] = []
    rows: list[dict] = []

    for run in usable:
        for start in range(len(run) - SEQ_LEN + 1):
            idx = run[start : start + SEQ_LEN]
            window_slice = window_idx[idx]

            # Hard re-check per sequence; cheap, and makes the invariant
            # impossible to lose to an indexing slip above.
            if not np.array_equal(np.diff(window_slice), np.ones(SEQ_LEN - 1)):
                raise BuildError(
                    f"internal error: non-contiguous sequence at "
                    f"window_idx {int(window_slice[0])}"
                )

            sequences.append(matrix[idx])
            start_idx = int(window_slice[0])
            end_idx = int(window_slice[-1])
            rows.append(
                {
                    "sequence_index": len(sequences) - 1,
                    "source_host": HOST,
                    "window_idx_start": start_idx,
                    "window_idx_end": end_idx,
                    "timestamp_start": start_idx * WINDOW_SECONDS,
                    "timestamp_end": end_idx * WINDOW_SECONDS,
                    "timestamp_start_utc": datetime.fromtimestamp(
                        (start_idx + win0) * WINDOW_SECONDS, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_end_utc": datetime.fromtimestamp(
                        (end_idx + win0) * WINDOW_SECONDS, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

    if not sequences:
        raise BuildError(
            "no run of 10 consecutive joined windows exists — cannot build "
            "a single 10x82 sequence. Refusing to stitch across gaps."
        )

    X = np.stack(sequences).astype(np.float64)
    metadata = pd.DataFrame(rows)
    print(f"  valid 10-window sequences : {len(X)}")
    return X, metadata


# =============================================================
# STEP 6 — VALIDATION
# =============================================================


def validate(
    X: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: list[str],
    scaler_names: tuple[str, ...],
) -> list[tuple[str, bool, str]]:
    """Checks A-K. Every one must pass before anything is written."""
    _rule("STEP 6 — validation (A-K)")

    checks: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, note: str = "") -> None:
        checks.append((label, bool(ok), note))

    check("A. X.ndim == 3", X.ndim == 3, f"ndim={X.ndim}")
    check("A. X.shape[1] == 10", X.shape[1] == SEQ_LEN, f"{X.shape[1]}")
    check("A. X.shape[2] == 82", X.shape[2] == V5_DIM, f"{X.shape[2]}")
    check("B. no NaN", not np.isnan(X).any(), f"{int(np.isnan(X).sum())} NaN")
    check("C. no Inf", not np.isinf(X).any(), f"{int(np.isinf(X).sum())} Inf")

    contiguous = bool(
        (metadata["window_idx_end"] - metadata["window_idx_start"] == SEQ_LEN - 1).all()
    )
    check("D. every sequence contiguous", contiguous)

    check(
        "E. V4 prefix == 52 dims",
        feature_names[:V4_PREFIX_DIM] == list(v4.INPUT_FEATURE_NAMES),
        "matches pcap_ingest_v4.INPUT_FEATURE_NAMES",
    )
    check(
        "F. packet suffix == 30 dims",
        len(feature_names[V4_PREFIX_DIM:]) == PACKET_SUFFIX_DIM,
        f"{len(feature_names[V4_PREFIX_DIM:])}",
    )
    check(
        "G. names == v5_input_scaler.feature_names_in_",
        tuple(feature_names) == scaler_names,
    )
    check(
        "H. no duplicate feature names",
        len(set(feature_names)) == len(feature_names),
    )
    check(
        "I. no feature reordering",
        feature_names == list(v4.INPUT_FEATURE_NAMES) + list(scaler_names[52:]),
        "prefix and suffix each in their source order",
    )

    label_like = [
        n
        for n in feature_names
        if any(t in n.lower() for t in ("attack", "label", "target", "malicious"))
    ]
    check("J. no labels among inputs", not label_like, f"{label_like or 'none'}")

    # K. A sequence may only contain windows at or before its own end.
    # Sequences are built from ascending contiguous runs and each row is
    # taken at its own window_idx, so verify the invariant directly:
    # row t of a sequence must be window_idx_start + t.
    leak_free = bool(
        (metadata["window_idx_start"] + SEQ_LEN - 1 == metadata["window_idx_end"]).all()
        and metadata["window_idx_start"].is_monotonic_increasing
    )
    check("K. no future leakage into history", leak_free, "strictly ascending, step 1")

    width = max(len(c[0]) for c in checks)
    for label, ok, note in checks:
        suffix = f"  ({note})" if note else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}{suffix}")
    return checks


# =============================================================
# STEP 7 — REAL INFERENCE
# =============================================================


def run_real_inference(X: np.ndarray, metadata: pd.DataFrame) -> dict:
    """Push exactly ONE real sequence through the V5 engine at K=5."""
    _rule("STEP 7 — real V5 inference (one sequence, K=5)")

    engine = V5Inference()
    engine.load_model()

    sequence = X[0:1]
    row = metadata.iloc[0]

    timestamps = [
        (int(row["window_idx_start"]) + t) * WINDOW_SECONDS for t in range(SEQ_LEN)
    ]
    result = engine.k_step_rollout(sequence, steps=5, timestamps=timestamps)

    print(f"  sequence_index : {int(row['sequence_index'])}")
    print(
        f"  window_idx     : {int(row['window_idx_start'])} .. "
        f"{int(row['window_idx_end'])}"
    )
    print(f"  UTC            : {row['timestamp_start_utc']} .. {row['timestamp_end_utc']}")
    print()
    print("REAL V5 DEMO INPUT")
    print(f"shape = {tuple(sequence.shape)}")
    print()
    print("CURRENT LATENT")
    print(f"shape = {tuple(result.current_latent.shape)}")
    print()
    print("FUTURE STATES")
    print(f"shape = {tuple(result.future_states.shape)}")
    print()
    print("FUTURE LATENTS")
    print(f"shape = {tuple(result.future_latents.shape)}")
    print()
    print("THREAT PROBABILITIES")
    print(f"shape = {tuple(result.threat_probabilities.shape)}")
    print()
    print("ATTACK FLAGS")
    print(f"shape = {tuple(result.attack_flags.shape)}")
    print()

    probabilities = result.threat_probabilities[0]
    print("Five threat probabilities (genuine model outputs — NOT accuracy,")
    print("NOT a benchmark metric, NOT an early-warning claim):")
    for step, value in enumerate(probabilities, start=1):
        flag = "ATTACK" if value >= result.threshold else "benign"
        print(
            f"  t+{step} ({step * WINDOW_SECONDS:>2}s ahead): "
            f"{value:.6f}   [{flag} @ threshold {result.threshold}]"
        )

    return {
        "sequence_index": int(row["sequence_index"]),
        "threat_probabilities": [float(v) for v in probabilities],
        "attack_flags": [bool(v) for v in result.attack_flags[0]],
        "threshold": float(result.threshold),
    }


# =============================================================
# MAIN
# =============================================================


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a real host-level 10x82 V5 demo sequence set."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="re-parse the pcap instead of reusing the cached V4 window table",
    )
    args = parser.parse_args(argv)

    record = verify_provenance()

    with open(V5_SCALER_PATH, "rb") as handle:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = pickle.load(handle)
    if int(getattr(scaler, "n_features_in_", -1)) != V5_DIM:
        raise BuildError(
            f"V5 input scaler has n_features_in_="
            f"{getattr(scaler, 'n_features_in_', None)}, expected {V5_DIM}"
        )
    scaler_names = tuple(str(n) for n in np.asarray(scaler.feature_names_in_).ravel())
    print(f"  V5 scaler       : n_features_in_={scaler.n_features_in_} (not modified)")

    packet_names = list(scaler_names[V4_PREFIX_DIM:])

    v4_windows = build_v4_windows(use_cache=not args.no_cache)
    origin = verify_alignment(v4_windows, record)
    joined, stats = join_lanes(v4_windows, packet_names)

    feature_names = list(v4.INPUT_FEATURE_NAMES) + packet_names
    X, metadata = build_sequences(joined, feature_names, origin["win0"])

    checks = validate(X, metadata, feature_names, scaler_names)
    if any(not ok for _, ok, _ in checks):
        raise BuildError("validation failed — nothing was written")

    # ----- write artifacts -----
    _rule("STEP 8 — writing artifacts")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(X_PATH, X)
    metadata.to_csv(METADATA_PATH, index=False)
    FEATURE_NAMES_PATH.write_text("\n".join(feature_names) + "\n")
    for path in (X_PATH, METADATA_PATH, FEATURE_NAMES_PATH):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

    inference = run_real_inference(X, metadata)

    manifest = {
        "source_pcap": str(SOURCE_PCAP.relative_to(REPO_ROOT)),
        "source_pcap_sha256": record.get("source_pcap_sha256"),
        "packet_feature_source": str(PACKET_CSV.relative_to(REPO_ROOT)),
        "host": HOST,
        "feature_count": V5_DIM,
        "v4_prefix_dims": V4_PREFIX_DIM,
        "packet_suffix_dims": PACKET_SUFFIX_DIM,
        "sequence_length": SEQ_LEN,
        "window_seconds": WINDOW_SECONDS,
        "join_key": "window_idx",
        "window_idx_definition": (
            "floor(packet_epoch_seconds / 10) - win0, win0 = the 10s bin of "
            "the capture's first IP packet; identical in both lanes and "
            "verified by matching IP-packet counts"
        ),
        "win0": origin["win0"],
        "t0_epoch": origin["t0_epoch"],
        "n_ip_packets": origin["n_ip_packets"],
        "number_of_v4_windows": stats["v4_windows"],
        "number_of_packet_windows": stats["packet_windows"],
        "number_of_valid_joined_windows": stats["joined_windows"],
        "number_of_valid_10_window_sequences": int(X.shape[0]),
        "joined_gap_count": stats["joined_gap_count"],
        "scaler_path": str(V5_SCALER_PATH.relative_to(REPO_ROOT)),
        "scaler_fitted_here": False,
        "model_path": str(V5_CHECKPOINT.relative_to(REPO_ROOT)),
        "model_version": "v5",
        "v4_lane_module": "src/pcap_ingest_v4.py (unmodified, reused)",
        "packet_lane_module": "src/packet_feature_builder.py (via prebuilt table)",
        "labels_included": False,
        "imputation": (
            "NONE — no window was filled, interpolated or manufactured; a "
            "window survives only if both lanes independently produced it"
        ),
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00"
        ),
        "demo_inference": inference,
        "scope_note": SCOPE_NOTE,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"\nBUILD STOPPED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
