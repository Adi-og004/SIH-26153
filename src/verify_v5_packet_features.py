"""
=============================================================
V5 PACKET FEATURE VERIFIER (Task 9, Part 29)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Fourteen independent checks over the artefacts that
``src/run_v5_packet_features.py`` produced. Run the pipeline first.

A feature family is reported PASS only when it is IMPLEMENTED and
INDEPENDENTLY RE-DERIVED here from the contract's written formula — not
from the builder's own output. A family that could not be implemented
reliably is reported FAIL and drags OVERALL to FAIL; it is never quietly
downgraded to a pass.

A family that IS reliably implemented but happens to be degenerate in the
captures available to this project (IP fragmentation, which is absent from
every capture on disk) passes the implementation check and is flagged in
the NOTES block underneath. That distinction — "the code is wrong" versus
"the data has none" — is the whole point of separating the two.

Run:
    ./venv/bin/python src/verify_v5_packet_features.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import feature_contract as v4c            # noqa: E402
import packet_feature_builder as pfb      # noqa: E402
import v5_feature_contract as v5c         # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "v5_features"
DOCS_DIR = REPO_ROOT / "docs"
SUMMARY_PATH = REPO_ROOT / "results" / "v5_packet_feature_summary.json"

# Small, fast, real-format capture used for the independent re-derivation.
PROBE_PCAP = REPO_ROOT / "data" / "sample_pcap" / "smallFlows.pcap"

V4_EXPECTED_SHA256 = {
    "models/latent_world_model_v4/latent_world_model_v4.pt":
        "76af8d8a9e02215923b1c0ab6b627bfc0809e856748666a55ee4b16cec5e5898",
    "data/processed/multiday_benchmark_K5/input_scaler.pkl":
        "e0ab9bebd26869950dda51a67bdde3192b0edbd00e20b623115ed2b653f04f19",
    "docs/v4_feature_schema.json":
        "2b4f820ca1f193e47b3b9c1eae4163cd9b2643886665f5f5d83f876e3ee094c4",
    "docs/v4_baseline_manifest.json":
        "b2765c2489ccee3845e9dc43990b187d894237905087f3969ed2fae82613937e",
}

TOL = 1e-9
FAILURES: list[str] = []
NOTES: list[str] = []


def _fail(check: str, detail: str) -> bool:
    FAILURES.append(f"{check}: {detail}")
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# =============================================================
# INDEPENDENT RE-DERIVATION OF THE 21 FORMULAS
# =============================================================
# Written from docs/v5_packet_feature_contract.json, in plain python, with
# no call into packet_feature_builder's aggregation code. This is what
# turns "the family is implemented" into a checkable claim.

def _ref_std(values):
    n = len(values)
    if n <= 1:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))


def _ref_pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    pos = (q / 100.0) * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo]) + (float(s[hi]) - float(s[lo])) * (pos - lo)


def _ref_entropy(values):
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    n = len(values)
    return -sum((c / n) * math.log2(c / n + 1e-12) for c in counts.values())


def _ref_log1p(x):
    return math.log(1.0 + max(float(x), 0.0))


def _ref_ratio(a, b):
    return 0.0 if b == 0 else float(a) / float(b)


def _reference_window(g: pd.DataFrame) -> dict:
    """The 21 features for one window, re-derived from the contract text."""
    ttl = list(g["ttl"])
    n_ip = len(g)
    tcp = g[g["is_tcp"]]
    win = list(tcp["tcp_window"])
    n_tcp = len(win)
    pay = list(g["payload_len"])

    out = {
        "ttl_mean": sum(ttl) / n_ip,
        "ttl_std": _ref_std(ttl),
        "ttl_min": float(min(ttl)),
        "ttl_max": float(max(ttl)),
        "log_ttl_unique_count": _ref_log1p(len(set(ttl))),

        "log_tcp_window_mean": _ref_log1p(sum(win) / n_tcp) if n_tcp else 0.0,
        "log_tcp_window_std": _ref_log1p(_ref_std(win)) if n_tcp else 0.0,
        "log_tcp_window_max": _ref_log1p(max(win)) if n_tcp else 0.0,
        "tcp_window_zero_ratio": _ref_ratio(sum(1 for w in win if w == 0), n_tcp),
        "tcp_window_change_rate": _ref_ratio(
            int(g["tcp_window_changed"].sum()), int(g["window_change_eligible"].sum())
        ),

        "log_frag_packet_count": _ref_log1p(int(g["is_fragment"].sum())),
        "frag_packet_ratio": _ref_ratio(int(g["is_fragment"].sum()), n_ip),
        "ip_df_ratio": _ref_ratio(
            int((g["ip_df"] & (g["ip_version"] == 4)).sum()),
            int((g["ip_version"] == 4).sum()),
        ),

        "log_tcp_retrans_count": _ref_log1p(int(g["is_retransmission"].sum())),
        "tcp_retrans_ratio": _ref_ratio(
            int(g["is_retransmission"].sum()), int(g["seq_bearing"].sum())
        ),
        "log_tcp_retrans_bytes": _ref_log1p(int(g["retrans_bytes"].sum())),

        "payload_len_p50": _ref_pct(pay, 50),
        "payload_len_p90": _ref_pct(pay, 90),
        "payload_len_max": float(max(pay)),
        "payload_zero_ratio": _ref_ratio(sum(1 for x in pay if x == 0), n_ip),
        "payload_len_entropy": _ref_entropy(pay),
    }
    return out


_PROBE_CACHE: dict = {}


def _probe():
    """Parse the probe capture once and cache the annotated table + built
    features. Real headers, real timestamp ties, small enough to be fast."""
    if "built" not in _PROBE_CACHE:
        packets = pfb.parse_pcap_packets(PROBE_PCAP)
        annotated = pfb.annotate_packets(packets)
        win0 = int(np.floor(float(packets["timestamp_abs"].min()) / pfb.WINDOW_SECONDS))
        built = pfb.add_packet_relative_changes(
            pfb.build_packet_windows(annotated, win0=win0)
        )
        abs_bin = np.floor(
            annotated["timestamp_abs"].to_numpy(float) / pfb.WINDOW_SECONDS
        ).astype(np.int64)
        annotated = annotated.assign(window_idx=abs_bin - win0)
        _PROBE_CACHE.update(
            {"packets": packets, "annotated": annotated, "built": built, "win0": win0}
        )
    return _PROBE_CACHE


def _family_check(family: str) -> bool:
    """Re-derive every feature of one family, on every window of the probe
    capture, and require agreement with the builder."""
    probe = _probe()
    built, annotated = probe["built"], probe["annotated"]
    names = {
        "ttl": pfb.TTL_FEATURES,
        "tcp_window": pfb.TCP_WINDOW_FEATURES,
        "fragmentation": pfb.FRAGMENTATION_FEATURES,
        "retransmission": pfb.RETRANSMISSION_FEATURES,
        "payload": pfb.PAYLOAD_FEATURES,
    }[family]

    worst = 0.0
    n_compared = 0
    for window_idx, g in annotated.groupby("window_idx", sort=True):
        row = built[built["window_idx"] == window_idx]
        if len(row) != 1:
            return _fail(family, f"window {window_idx} missing from the built table")
        row = row.iloc[0]
        reference = _reference_window(g)
        for name in names:
            diff = abs(float(row[name]) - float(reference[name]))
            worst = max(worst, diff)
            n_compared += 1

    if worst > TOL:
        return _fail(
            family,
            f"independent re-derivation of the contract formula disagrees with the "
            f"builder by {worst:.3e} (tolerance {TOL:.0e})",
        )

    # Liveness is informational, never a pass/fail on its own.
    dead = [
        name for name in names
        if float(built[name].max()) == float(built[name].min())
    ]
    if dead:
        NOTES.append(
            f"{family}: {', '.join(dead)} is constant on the probe capture "
            f"({PROBE_PCAP.name}). The formula is verified; the DATA carries no variance."
        )
    return True


# =============================================================
# THE FOURTEEN CHECKS
# =============================================================

def check_01_schema_exists() -> bool:
    for path in (v5c.SCHEMA_PATH, v5c.CONTRACT_PATH,
                 DOCS_DIR / "v5_packet_dataset_manifest.json"):
        if not path.exists():
            return _fail("schema", f"missing artefact {path.relative_to(REPO_ROOT)}")
    schema = v5c.load_schema()
    if schema["version"] != "v5":
        return _fail("schema", f"version is {schema['version']!r}, expected 'v5'")
    return True


def check_02_feature_count() -> bool:
    schema = v5c.load_schema()
    counts = schema["counts"]
    expected = {"flow_base": 26, "flow_relative_change": 26,
                "packet_base": 21, "packet_relative_change": 9, "total": 82}
    if counts != expected:
        return _fail("schema", f"counts {counts} != {expected}")
    if schema["total_dimension"] != 82 or len(schema["feature_order"]) != 82:
        return _fail("schema", "total_dimension / feature_order disagree with 82")
    if len(set(schema["feature_order"])) != 82:
        return _fail("schema", "duplicate feature name in the order")
    return True


def check_03_feature_order_deterministic() -> bool:
    on_disk = v5c.load_schema()["feature_order"]
    rebuilt = [v5c.build_schema()["feature_order"] for _ in range(3)]
    if any(order != on_disk for order in rebuilt):
        return _fail("schema", "rebuilding the schema does not reproduce the stored order")
    indices = [f["index"] for f in v5c.load_schema()["features"]]
    if indices != list(range(82)):
        return _fail("schema", "feature indices are not 0..81 in order")
    return True


def check_04_packet_features_populated() -> bool:
    path = OUT_DIR / "pcap_conformance.csv"
    if not path.exists():
        return _fail("packet builder", "pcap_conformance.csv missing — run the pipeline first")
    frame = pd.read_csv(path)
    base = frame[frame["kind"] == "base"]
    if base.empty:
        return _fail("packet builder", "no base-feature conformance rows")
    # every packet feature must appear for every capture
    for capture, group in base.groupby("capture"):
        missing = set(pfb.PACKET_BASE_FEATURES) - set(group["feature"])
        if missing:
            return _fail("packet builder", f"{capture} missing features {sorted(missing)}")
        if int(group["n_windows"].min()) == 0:
            return _fail("packet builder", f"{capture} produced zero windows")
    # at least one real capture must show most families actually carrying values
    real = base[base["capture"].str.startswith("march02")]
    if len(real):
        alive = int((real["frac_nonzero"] > 0).sum())
        if alive < 15:
            return _fail(
                "packet builder",
                f"only {alive}/21 packet features carry a non-zero value on real traffic",
            )
        NOTES.append(
            f"packet builder: {alive}/21 base features carry non-zero values across "
            f"{int(real['n_windows'].iloc[0])} windows of real March-2 traffic."
        )
    return True


def check_05_no_nan_or_inf() -> bool:
    frame = pd.read_csv(OUT_DIR / "pcap_conformance.csv")
    if int(frame["n_missing"].sum()) != 0:
        return _fail("transforms", f"{int(frame['n_missing'].sum())} NaN values in built features")
    if int(frame["n_non_finite"].sum()) != 0:
        return _fail("transforms",
                     f"{int(frame['n_non_finite'].sum())} non-finite values in built features")
    probe = _probe()["built"]
    values = probe[pfb.PACKET_BASE_FEATURES + pfb.PACKET_RELATIVE_FEATURES].to_numpy(float)
    if not np.isfinite(values).all():
        return _fail("transforms", "non-finite value produced on the probe capture")
    return True


def check_06_formulas_match_contract() -> bool:
    """The contract's formula text must name real implementation symbols, and
    the numbers must match an independent re-derivation of that text."""
    contract = v5c.load_packet_contract()
    source = (REPO_ROOT / "src" / "packet_feature_builder.py").read_text()
    for feature in contract["features"]:
        if feature["category"] != "base":
            continue
        if feature["name"] not in source:
            return _fail("transforms", f"{feature['name']} is not referenced in the builder")
        if len(feature["formula"]) < 20:
            return _fail("transforms", f"{feature['name']} has no substantive formula")
    # the numeric half is covered per family by check_07..check_11
    for helper in ("_safe_log1p", "_ratio", "_std", "_percentile", "_entropy_from_counts"):
        if helper not in source:
            return _fail("transforms", f"documented helper {helper} missing from the builder")
    if f'PERCENTILE_METHOD = "{pfb.PERCENTILE_METHOD}"' not in source:
        return _fail("transforms", "the percentile method is not pinned in the builder")
    return True


def check_07_packet_builder_works() -> bool:
    probe = _probe()
    built = probe["built"]
    if len(built) == 0:
        return _fail("packet builder", "the probe capture produced no windows")
    expected_columns = ["window_idx", "timestamp", "packet_count"] + \
        pfb.PACKET_BASE_FEATURES + pfb.PACKET_RELATIVE_FEATURES
    if list(built.columns) != expected_columns:
        return _fail("packet builder", "built table columns drifted from the contract order")
    if int(built["packet_count"].sum()) != len(probe["packets"]):
        return _fail("packet builder", "windowed packet counts do not sum to the parsed total")
    return True


def check_08_parity() -> bool:
    path = OUT_DIR / "packet_parity_results.csv"
    if not path.exists():
        return _fail("parity", "packet_parity_results.csv missing — run the pipeline first")
    frame = pd.read_csv(path)
    if frame.empty:
        return _fail("parity", "no parity comparisons recorded")
    mismatches = frame[~frame["bitwise_identical"].astype(bool)]
    if len(mismatches):
        return _fail(
            "parity",
            f"{len(mismatches)} of {len(frame)} comparisons are not bit-identical "
            f"(worst {mismatches['max_abs_diff'].max():.3e})",
        )
    NOTES.append(
        f"parity: {len(frame)}/{len(frame)} (capture, feature) comparisons bit-identical. "
        "Both routes call one shared implementation, so this verifies the plumbing, not a "
        "second independent implementation — none exists, because the training data is flow "
        "CSVs with no packet fields."
    )
    return True


def check_09_windowing() -> bool:
    schema = v5c.load_schema()
    semantics = schema["window_semantics"]
    if semantics["window_seconds"] != 10:
        return _fail("windowing", f"window_seconds is {semantics['window_seconds']}, expected 10")
    if "[t, t+10)" not in semantics["interval"]:
        return _fail("windowing", "the interval is not documented as [t, t+10)")
    if pfb.WINDOW_SECONDS != v4c.WINDOW_SECONDS:
        return _fail("windowing",
                     "the packet window width differs from the frozen V4 window width")

    probe = _probe()
    annotated, win0 = probe["annotated"], probe["win0"]
    ts = annotated["timestamp_abs"].to_numpy(float)
    idx = annotated["window_idx"].to_numpy(np.int64)
    lower = (idx + win0) * pfb.WINDOW_SECONDS
    if not np.all((ts >= lower) & (ts < lower + pfb.WINDOW_SECONDS)):
        return _fail("windowing", "a packet was assigned outside its own [t, t+10) bin")

    built = probe["built"]
    if not np.array_equal(
        built["timestamp"].to_numpy(), built["window_idx"].to_numpy() * pfb.WINDOW_SECONDS
    ):
        return _fail("windowing", "timestamp != window_idx * 10")
    if built["window_idx"].is_monotonic_increasing is False:
        return _fail("windowing", "windows are not emitted in ascending order")
    return True


def check_10_relative_change() -> bool:
    schema = v5c.load_schema()
    semantics = schema["relative_change_semantics"]
    if semantics["epsilon"] != v4c.RELATIVE_EPSILON:
        return _fail("relative change", "epsilon differs from the frozen V4 epsilon")
    if not semantics["identical_to_v4"]:
        return _fail("relative change", "the contract no longer claims V4-identical semantics")

    contract = v5c.load_packet_contract()
    base = [f for f in contract["features"] if f["category"] == "base"]
    rulings = {f["name"]: f["relative_change_decision"] for f in base}
    if set(rulings) != set(pfb.PACKET_BASE_FEATURES):
        return _fail("relative change", "a base feature has no relative-change ruling")
    included = [n for n, r in rulings.items() if r == "included"]
    if included != list(pfb.PACKET_RELATIVE_CHANGE_BASE):
        return _fail("relative change", "the contract's inclusion list disagrees with the builder")
    if len(included) != 9:
        return _fail("relative change", f"{len(included)} relative changes, expected 9")

    # numeric: reproduce the transform against the frozen V4 reference impl
    built = _probe()["built"]
    frame = built[pfb.PACKET_RELATIVE_CHANGE_BASE].astype(float).reset_index(drop=True)
    delta = frame.diff().fillna(0.0)
    previous = frame.shift(1)
    reference = delta / (previous.abs() + v4c.RELATIVE_EPSILON)
    reference.iloc[0] = 0.0
    reference = (
        reference.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        .clip(-v4c.RELATIVE_CLIP, v4c.RELATIVE_CLIP)
    )
    actual = built[pfb.PACKET_RELATIVE_FEATURES].to_numpy(float)
    if not np.allclose(actual, reference.to_numpy(float), rtol=0, atol=1e-12):
        return _fail("relative change",
                     "the packet relative-change transform differs from the V4 definition")
    if np.abs(actual[0]).max() != 0.0:
        return _fail("relative change", "the first row is not exactly 0.0")
    if np.abs(actual).max() > v4c.RELATIVE_CLIP + TOL:
        return _fail("relative change", "a value escaped the +/-10 clip")
    return True


def check_11_leakage() -> bool:
    path = OUT_DIR / "leakage_check.json"
    if not path.exists():
        return _fail("leakage", "leakage_check.json missing — run the pipeline first")
    report = json.loads(path.read_text())
    if report["overall"] != "PASS":
        return _fail("leakage", f"recorded leakage status is {report['overall']}")
    for capture, result in report["captures"].items():
        if not result["prefix_truncation_invariant"]:
            return _fail("leakage", f"{capture}: a past window changed when the future arrived")
        if not result["packets_inside_their_own_window"]:
            return _fail("leakage", f"{capture}: a packet contributed outside its own window")
        if not result["relative_change_is_backward_only"]:
            return _fail("leakage", f"{capture}: relative change is not backward-only")

    # re-run the decisive test live rather than trusting the stored report
    probe = _probe()
    packets, win0 = probe["packets"], probe["win0"]
    full = probe["built"]
    cutoff = (int(full["window_idx"].max()) + win0) * pfb.WINDOW_SECONDS
    truncated = pfb.packet_features_from_packet_table(
        packets[packets["timestamp_abs"] < cutoff], win0=win0
    )
    columns = pfb.PACKET_BASE_FEATURES + pfb.PACKET_RELATIVE_FEATURES
    for window_idx in truncated["window_idx"]:
        a = truncated[truncated["window_idx"] == window_idx][columns].to_numpy(float)
        b = full[full["window_idx"] == window_idx][columns].to_numpy(float)
        if not np.array_equal(a, b):
            return _fail("leakage",
                         f"live re-check: window {window_idx} changed when later packets arrived")
    return True


def check_12_schema_hash() -> bool:
    path = OUT_DIR / "v5_feature_contract_sha256.txt"
    if not path.exists():
        return _fail("schema", "v5_feature_contract_sha256.txt missing")
    recorded = None
    for line in path.read_text().splitlines():
        if line.startswith("v5_schema_sha256"):
            recorded = line.split()[-1]
    if recorded is None:
        return _fail("schema", "no v5_schema_sha256 line in the hash file")

    live = v5c.schema_hash()
    if live != recorded:
        return _fail("schema", f"schema hash drifted: live {live[:16]}... != recorded "
                               f"{recorded[:16]}...")
    stored = v5c.load_schema().get("schema_sha256")
    if stored != live:
        return _fail("schema", "docs/v5_feature_schema.json carries a stale schema_sha256")
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    if summary.get("schema_sha256") != live:
        return _fail("schema", "results/v5_packet_feature_summary.json carries a stale hash")
    # the hash must actually respond to a semantic change
    mutated = v5c.build_schema()
    mutated["features"][52]["unit"] = "MUTATED FOR THE SELF-TEST"
    if v5c.schema_hash(mutated) == live:
        return _fail("schema", "the hash does not respond to a change in a feature's unit")
    return True


def check_13_dataset_provenance() -> bool:
    path = DOCS_DIR / "v5_packet_dataset_manifest.json"
    if not path.exists():
        return _fail("provenance", "v5_packet_dataset_manifest.json missing")
    manifest = json.loads(path.read_text())
    records = manifest.get("packet_feature_records", [])
    if not records:
        return _fail("provenance", "the manifest holds no packet records")
    required = (
        "source_pcap", "source_pcap_sha256", "capture_scope", "timestamp_basis",
        "parser_version", "feature_contract_version", "feature_schema_sha256",
        "generation_timestamp_utc", "n_ip_packets", "n_windows", "n_missing_values",
    )
    for record in records:
        for key in required:
            if record.get(key) in (None, ""):
                return _fail("provenance", f"{record['record_id']} missing {key}")
        source = REPO_ROOT / record["source_pcap"]
        if not source.exists():
            return _fail("provenance", f"{record['record_id']}: source pcap no longer on disk")
    if manifest["training_dataset"]["exists"]:
        return _fail("provenance",
                     "the manifest claims a V5 training dataset exists — none was built")
    if "NONE" not in manifest["training_dataset"]["imputation_used"]:
        return _fail("provenance", "the manifest does not assert that no imputation was used")
    if not (OUT_DIR / "dataset_availability.csv").exists():
        return _fail("provenance", "dataset_availability.csv missing")
    NOTES.append(
        f"provenance: {len(records)} packet records, each with a source-pcap SHA-256. "
        "No V5 training dataset was built and no value was imputed."
    )
    return True


def check_14_v4_integrity() -> bool:
    for relative, expected in V4_EXPECTED_SHA256.items():
        path = REPO_ROOT / relative
        if not path.exists():
            return _fail("V4 integrity", f"{relative} is missing")
        actual = _sha256(path)
        if actual != expected:
            return _fail("V4 integrity",
                         f"{relative} CHANGED\n     expected {expected}\n     actual   {actual}")

    # the V4 contract itself must still describe 52 features in the frozen order
    frozen_order = (
        REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_feature_names.txt"
    ).read_text().split()
    if len(frozen_order) != 52:
        return _fail("V4 integrity", "the frozen V4 feature-name file no longer holds 52 names")
    if v4c.INPUT_FEATURE_NAMES != frozen_order:
        return _fail("V4 integrity", "the V4 contract's order drifted from the frozen file")
    if v5c.V5_FEATURE_NAMES[:52] != frozen_order:
        return _fail("V4 integrity",
                     "the V5 schema's first 52 features are NOT the frozen V4 vector")
    if v4c.build_contract()["feature_dimension"] != 52:
        return _fail("V4 integrity", "the V4 contract no longer reports 52 dimensions")
    return True


# =============================================================
# REPORT
# =============================================================

def main() -> int:
    results = {}

    results["Schema"] = all([
        check_01_schema_exists(),
        check_02_feature_count(),
        check_03_feature_order_deterministic(),
    ])
    results["Packet builder"] = all([
        check_07_packet_builder_works(),
        check_04_packet_features_populated(),
    ])
    results["TTL"] = _family_check("ttl")
    results["TCP window"] = _family_check("tcp_window")
    results["Fragmentation"] = _family_check("fragmentation")
    results["Retransmission"] = _family_check("retransmission")
    results["Payload distribution"] = _family_check("payload")
    results["Windowing"] = check_09_windowing()
    results["Transforms"] = all([check_05_no_nan_or_inf(), check_06_formulas_match_contract()])
    results["Relative change"] = check_10_relative_change()
    results["Training/PCAP parity"] = check_08_parity()
    results["Leakage check"] = check_11_leakage()
    results["Dataset provenance"] = check_13_dataset_provenance()
    results["V4 integrity"] = all([check_12_schema_hash(), check_14_v4_integrity()])

    print()
    print("V5 PACKET FEATURE VERIFICATION")
    print("------------------------------")
    for label, ok in results.items():
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
    overall = all(results.values())
    print()
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    if FAILURES:
        print()
        print("FAILURES")
        print("--------")
        for failure in FAILURES:
            print(f"  - {failure}")

    if NOTES:
        print()
        print("NOTES (informational — these do not change any verdict above)")
        print("------------------------------------------------------------")
        for note in NOTES:
            print(f"  - {note}")

    print()
    print("SCOPE OF THIS VERIFICATION")
    print("--------------------------")
    print("  A family PASSes when it is implemented AND its contract formula is")
    print("  independently re-derived here and matches the builder to 1e-9. It does")
    print("  NOT assert that the family is informative, nor that the available data")
    print("  exercises it — see the NOTES block for that.")
    print()
    print("  This verifier does NOT check a V5 model: none exists. Training was not")
    print("  attempted because the three TRAIN/VAL benchmark days have no packet")
    print("  captures at all (results/v5_features/dataset_availability.csv).")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
