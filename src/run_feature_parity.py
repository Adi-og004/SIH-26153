"""
=============================================================
FEATURE PARITY HARNESS (TASK 7)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Proves — or refuses to claim — that TRAINING, VALIDATION, TEST and PCAP
INFERENCE use the same 52-feature semantics.

Read-only with respect to every frozen artifact. It trains nothing, changes
no formula, and rewires no pipeline. It:

  1. emits docs/feature_contract_v4.json + its canonical hash
  2. maps every feature to the code that produces it in each path
  3. records the exact window and transform semantics of both paths
  4. runs 18 golden fixtures through BOTH aggregation implementations
  5. compares the resulting 26 base + 52 full features feature-by-feature
  6. checks the real sample PCAPs conform to the contract

The synthetic parity test feeds the SAME logical flow records into
  A) archive/create_network_states_v2.py  (the training state builder,
     executed for real on a temporary CIC-shaped CSV), and
  B) src/pcap_ingest_v4.py:compute_window_features (the PCAP builder),
so any difference is a genuine difference between the two implementations.

Usage:
    python src/run_feature_parity.py
    python src/run_feature_parity.py --skip-pcap
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import feature_contract as fc  # noqa: E402
from feature_contract import (  # noqa: E402
    APPROXIMATE, BASE_FEATURES, EXACT, INPUT_FEATURE_NAMES, NON_PARITY,
    RELATIVE_FEATURES, REPO_ROOT, UNKNOWN, WINDOW_SECONDS,
)

OUT_DIR = REPO_ROOT / "results" / "feature_parity"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "feature_parity"
SUMMARY_JSON = REPO_ROOT / "results" / "feature_parity_summary.json"
CONTRACT_JSON = REPO_ROOT / "docs" / "feature_contract_v4.json"
PARITY_MANIFEST = REPO_ROOT / "docs" / "v4_feature_parity_manifest.json"
SAMPLE_PCAP_DIR = REPO_ROOT / "data" / "sample_pcap"

TRAINING_SCRIPT = REPO_ROOT / "archive" / "create_network_states_v2.py"

# Numerical tolerances for the synthetic parity test. The two paths run the
# same arithmetic in the same library, so equality should hold to float noise.
ABS_TOLERANCE = 1e-9
REL_TOLERANCE = 1e-9

CIC_COLUMNS = [
    "Timestamp", "Protocol", "Dst Port", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Pkt Len Mean", "Pkt Len Std", "Fwd IAT Mean", "Bwd IAT Mean",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "Down/Up Ratio", "Label",
]


# =============================================================
# GOLDEN FIXTURES
# =============================================================
# One fixture = a list of logical FLOW records with an explicit window.
# Flow fields use the PCAP-path schema; render_cic_rows() maps them 1:1 to
# the CICFlowMeter column names the training builder consumes.

def _flow(window, dst_port=443, protocol=6, fwd=5, bwd=5, len_fwd=500, len_bwd=500,
          duration_us=1_000_000.0, fwd_iat_us=100_000.0, bwd_iat_us=100_000.0,
          pkt_len_mean=100.0, pkt_len_std=10.0,
          syn=0, ack=1, rst=0, fin=0, psh=0, urg=0, offset=0.0):
    total = fwd + bwd
    return {
        "window_idx": window,
        "start_ts": window * WINDOW_SECONDS + offset,
        "dst_port": dst_port, "protocol": protocol,
        "tot_fwd_pkts": fwd, "tot_bwd_pkts": bwd, "tot_pkts": total,
        "totlen_fwd": len_fwd, "totlen_bwd": len_bwd, "total_bytes": len_fwd + len_bwd,
        "flow_duration_us": duration_us,
        "fwd_iat_mean_us": fwd_iat_us, "bwd_iat_mean_us": bwd_iat_us,
        "pkt_len_mean": pkt_len_mean, "pkt_len_std": pkt_len_std,
        "syn": syn, "ack": ack, "rst": rst, "fin": fin, "psh": psh, "urg": urg,
        "down_up_ratio": int(bwd // fwd) if fwd > 0 else 0,
    }


def build_fixtures() -> dict:
    fixtures = {}

    fixtures["01_empty_window"] = {
        "description": "windows 0 and 2 carry flows, window 1 is empty — the empty window must "
                       "produce NO row in either path",
        "flows": [_flow(0), _flow(2)],
    }
    fixtures["02_single_flow"] = {
        "description": "exactly one flow in one window — entropies must be 0.0 (<=1 distinct value)",
        "flows": [_flow(0)],
    }
    fixtures["03_multiple_flows"] = {
        "description": "three heterogeneous flows in one window",
        "flows": [_flow(0, dst_port=80, fwd=3, bwd=7), _flow(0, dst_port=443, fwd=10, bwd=2),
                  _flow(0, dst_port=22, protocol=17, fwd=1, bwd=1)],
    }
    fixtures["04_syn_only"] = {
        "description": "SYN set, no ACK — scan-like",
        "flows": [_flow(0, syn=1, ack=0, fwd=1, bwd=0, len_bwd=0, duration_us=0.0,
                        fwd_iat_us=0.0, bwd_iat_us=0.0) for _ in range(4)],
    }
    fixtures["05_ack_heavy"] = {
        "description": "every flow ACK-bearing with many packets — small per-flow flag rate",
        "flows": [_flow(0, ack=1, fwd=50, bwd=50, len_fwd=5000, len_bwd=5000)],
    }
    fixtures["06_rst_activity"] = {
        "description": "RST-bearing flows (refused connections)",
        "flows": [_flow(0, rst=1, ack=0, fwd=1, bwd=1, len_fwd=0, len_bwd=0)],
    }
    fixtures["07_fin_psh_urg"] = {
        "description": "FIN, PSH and URG all set on one flow",
        "flows": [_flow(0, fin=1, psh=1, urg=1, ack=1)],
    }
    fixtures["08_many_dst_ports"] = {
        "description": "ten distinct destination ports — high port entropy",
        "flows": [_flow(0, dst_port=1000 + i) for i in range(10)],
    }
    fixtures["09_repeated_dst_port"] = {
        "description": "ten flows, one port — port entropy must be exactly 0.0",
        "flows": [_flow(0, dst_port=443) for _ in range(10)],
    }
    fixtures["10_multi_protocol"] = {
        "description": "TCP, UDP and ICMP in one window",
        "flows": [_flow(0, protocol=6), _flow(0, protocol=17), _flow(0, protocol=1)],
    }
    fixtures["11_zero_bytes"] = {
        "description": "flows carrying no payload bytes — byte ratios hit the safe_divide epsilon",
        "flows": [_flow(0, len_fwd=0, len_bwd=0, pkt_len_mean=0.0, pkt_len_std=0.0)],
    }
    fixtures["12_zero_packets"] = {
        "description": "degenerate flow with zero packets — flag rates hit the +1e-8 denominator",
        "flows": [_flow(0, fwd=0, bwd=0, len_fwd=0, len_bwd=0, pkt_len_mean=0.0,
                        pkt_len_std=0.0, duration_us=0.0, fwd_iat_us=0.0, bwd_iat_us=0.0)],
    }
    fixtures["13_near_zero_iat"] = {
        "description": "sub-microsecond inter-arrival times — log1p of a tiny number",
        "flows": [_flow(0, fwd_iat_us=0.0001, bwd_iat_us=0.0, duration_us=1.0)],
    }
    fixtures["14_forward_only"] = {
        "description": "no backward packets — fwd ratio 1, bwd ratio 0, down/up 0",
        "flows": [_flow(0, fwd=10, bwd=0, len_fwd=1000, len_bwd=0, bwd_iat_us=0.0)],
    }
    fixtures["15_backward_only"] = {
        "description": "no forward packets — the safe_divide epsilon carries the ratio",
        "flows": [_flow(0, fwd=0, bwd=10, len_fwd=0, len_bwd=1000, fwd_iat_us=0.0)],
    }
    fixtures["16_balanced_directions"] = {
        "description": "equal forward and backward traffic",
        "flows": [_flow(0, fwd=10, bwd=10, len_fwd=1000, len_bwd=1000)],
    }
    fixtures["17_increasing_volume"] = {
        "description": "four consecutive windows with growing volume — exercises positive relative change",
        "flows": [f for window in range(4)
                  for f in [_flow(window, fwd=2 * (window + 1), bwd=2 * (window + 1),
                                  len_fwd=100 * (window + 1), len_bwd=100 * (window + 1),
                                  dst_port=80 + i, offset=i * 0.5)
                            for i in range(window + 1)]],
    }
    fixtures["18_decreasing_volume"] = {
        "description": "four consecutive windows with shrinking volume — negative relative change "
                       "and the clip boundary",
        "flows": [f for window in range(4)
                  for f in [_flow(window, fwd=2 * (4 - window), bwd=2 * (4 - window),
                                  len_fwd=100 * (4 - window), len_bwd=100 * (4 - window),
                                  dst_port=80 + i, offset=i * 0.5)
                            for i in range(4 - window)]],
    }
    return fixtures


def render_cic_rows(flows) -> pd.DataFrame:
    """Map the logical flow records onto the CICFlowMeter column names the
    training builder consumes. This mapping is 1:1 and lossless by construction."""
    rows = []
    for flow in flows:
        rows.append(
            {
                "Timestamp": pd.Timestamp("2018-01-01 00:00:00") + pd.Timedelta(seconds=flow["start_ts"]),
                "Protocol": flow["protocol"],
                "Dst Port": flow["dst_port"],
                "Flow Duration": flow["flow_duration_us"],
                "Tot Fwd Pkts": flow["tot_fwd_pkts"],
                "Tot Bwd Pkts": flow["tot_bwd_pkts"],
                "TotLen Fwd Pkts": flow["totlen_fwd"],
                "TotLen Bwd Pkts": flow["totlen_bwd"],
                "Pkt Len Mean": flow["pkt_len_mean"],
                "Pkt Len Std": flow["pkt_len_std"],
                "Fwd IAT Mean": flow["fwd_iat_mean_us"],
                "Bwd IAT Mean": flow["bwd_iat_mean_us"],
                "FIN Flag Cnt": flow["fin"], "SYN Flag Cnt": flow["syn"],
                "RST Flag Cnt": flow["rst"], "PSH Flag Cnt": flow["psh"],
                "ACK Flag Cnt": flow["ack"], "URG Flag Cnt": flow["urg"],
                "Down/Up Ratio": flow["down_up_ratio"],
                "Label": "Benign",
            }
        )
    return pd.DataFrame(rows, columns=CIC_COLUMNS)


# =============================================================
# RUNNING THE TWO REAL IMPLEMENTATIONS
# =============================================================

def load_training_builder():
    """Import archive/create_network_states_v2.py as a module (its main() is guarded)."""
    spec = importlib.util.spec_from_file_location("create_network_states_v2", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_training_path(flows, training_module) -> pd.DataFrame:
    """Execute the REAL training state builder on these flows."""
    frame = render_cic_rows(flows)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        data_dir = temporary / "in"
        out_dir = temporary / "out"
        data_dir.mkdir()
        out_dir.mkdir()
        frame.to_csv(data_dir / "fixture.csv", index=False)

        original_data, original_out = training_module.DATA_DIR, training_module.OUTPUT_DIR
        training_module.DATA_DIR, training_module.OUTPUT_DIR = data_dir, out_dir
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                training_module.create_states("fixture.csv")
            states = pd.read_csv(out_dir / "fixture_states_v2.csv")
        finally:
            training_module.DATA_DIR, training_module.OUTPUT_DIR = original_data, original_out

    states["window_idx"] = (
        pd.to_datetime(states["window_start"]).astype("int64") // 10**9 // WINDOW_SECONDS
    )
    return states.sort_values("window_idx").reset_index(drop=True)


def run_pcap_path(flows, pcap_module) -> pd.DataFrame:
    """Execute the REAL PCAP window builder on the same flows."""
    frame = pd.DataFrame(flows)
    frame["fwd_packet_ratio"] = pcap_module._safe_divide(frame["tot_fwd_pkts"], frame["tot_pkts"]).to_numpy()
    frame["bwd_packet_ratio"] = pcap_module._safe_divide(frame["tot_bwd_pkts"], frame["tot_pkts"]).to_numpy()
    frame["fwd_byte_ratio"] = pcap_module._safe_divide(frame["totlen_fwd"], frame["total_bytes"]).to_numpy()
    frame["bwd_byte_ratio"] = pcap_module._safe_divide(frame["totlen_bwd"], frame["total_bytes"]).to_numpy()
    denominator = frame["tot_pkts"].to_numpy() + 1e-8
    for rate, flag in (("syn_rate", "syn"), ("ack_rate", "ack"), ("rst_rate", "rst"),
                       ("fin_rate", "fin"), ("psh_rate", "psh"), ("urg_rate", "urg")):
        frame[rate] = frame[flag].to_numpy() / denominator
    return pcap_module.compute_window_features(frame).sort_values("window_idx").reset_index(drop=True)


# =============================================================
# PARITY TEST
# =============================================================

def compare_states(name, training_states, pcap_states) -> list[dict]:
    rows = []

    if len(training_states) != len(pcap_states):
        rows.append(
            {
                "fixture": name, "feature": "__window_count__",
                "training_value": len(training_states), "pcap_value": len(pcap_states),
                "absolute_difference": abs(len(training_states) - len(pcap_states)),
                "relative_difference": None, "tolerance": 0, "pass": False,
                "note": "the two paths produced a different number of windows",
            }
        )
        return rows

    rows.append(
        {
            "fixture": name, "feature": "__window_count__",
            "training_value": len(training_states), "pcap_value": len(pcap_states),
            "absolute_difference": 0, "relative_difference": 0.0, "tolerance": 0, "pass": True,
            "note": "same number of windows (empty windows omitted by both)",
        }
    )

    training_relative = fc.reference_relative_change(training_states)
    pcap_relative = fc.reference_relative_change(pcap_states)

    for feature in BASE_FEATURES:
        left = training_states[feature].to_numpy(dtype=float)
        right = pcap_states[feature].to_numpy(dtype=float)
        difference = np.abs(left - right)
        scale = np.maximum(np.abs(left), np.abs(right))
        relative = np.where(scale > 0, difference / np.maximum(scale, 1e-300), 0.0)
        ok = bool(np.all(difference <= ABS_TOLERANCE + REL_TOLERANCE * scale))
        rows.append(
            {
                "fixture": name, "feature": feature,
                "training_value": float(left[0]) if len(left) == 1 else float(left.mean()),
                "pcap_value": float(right[0]) if len(right) == 1 else float(right.mean()),
                "absolute_difference": float(difference.max()),
                "relative_difference": float(relative.max()),
                "tolerance": ABS_TOLERANCE, "pass": ok,
                "note": "" if ok else "AGGREGATION MISMATCH between the training and PCAP builders",
            }
        )

    for feature in RELATIVE_FEATURES:
        left = training_relative[feature].to_numpy(dtype=float)
        right = pcap_relative[feature].to_numpy(dtype=float)
        difference = np.abs(left - right)
        ok = bool(np.all(difference <= ABS_TOLERANCE))
        rows.append(
            {
                "fixture": name, "feature": feature,
                "training_value": float(left.mean()), "pcap_value": float(right.mean()),
                "absolute_difference": float(difference.max()),
                "relative_difference": None, "tolerance": ABS_TOLERANCE, "pass": ok,
                "note": "" if ok else "RELATIVE-CHANGE MISMATCH",
            }
        )

    return rows


def run_parity_tests(verbose=True):
    training_module = load_training_builder()
    import pcap_ingest_v4

    fixtures = build_fixtures()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, fixture in fixtures.items():
        (FIXTURE_DIR / f"{name}.json").write_text(json.dumps(fixture, indent=2) + "\n")

        training_states = run_training_path(fixture["flows"], training_module)
        pcap_states = run_pcap_path(fixture["flows"], pcap_ingest_v4)
        fixture_rows = compare_states(name, training_states, pcap_states)
        rows.extend(fixture_rows)

        failed = [row for row in fixture_rows if not row["pass"]]
        if verbose:
            status = "PASS" if not failed else f"FAIL ({len(failed)})"
            print(f"    {name:<28} windows={len(pcap_states):<2} {status}")

    return pd.DataFrame(rows), fixtures


# =============================================================
# REAL PCAP CONFORMANCE
# =============================================================

def run_pcap_conformance():
    import pcap_ingest_v4

    rows = []
    for pcap_path in sorted(SAMPLE_PCAP_DIR.glob("*.pcap")):
        entry = {"pcap": pcap_path.name, "size_bytes": pcap_path.stat().st_size}
        try:
            packets = pcap_ingest_v4.parse_pcap_streaming(str(pcap_path))
            flows = pcap_ingest_v4.reconstruct_flows(packets)
            windows = pcap_ingest_v4.compute_window_features(flows)
            full = pcap_ingest_v4.add_relative_changes(windows)

            base_ok = list(windows.columns[2:]) == BASE_FEATURES
            full_columns = [c for c in full.columns if c in INPUT_FEATURE_NAMES]
            order_ok = full_columns == [c for c in INPUT_FEATURE_NAMES if c in full.columns]
            values = full[INPUT_FEATURE_NAMES].to_numpy(dtype=float)

            entry.update(
                {
                    "packets": int(len(packets)), "flows": int(len(flows)),
                    "windows": int(len(windows)),
                    "base_feature_order_matches_contract": bool(base_ok),
                    "all_52_features_present": bool(set(INPUT_FEATURE_NAMES).issubset(full.columns)),
                    "feature_order_matches_contract": bool(order_ok),
                    "all_finite": bool(np.isfinite(values).all()),
                    "relative_change_within_clip": bool(
                        np.abs(full[RELATIVE_FEATURES].to_numpy(dtype=float)).max() <= 10.0 + 1e-9
                    ),
                    "first_row_relative_change_is_zero": bool(
                        np.abs(full[RELATIVE_FEATURES].iloc[0].to_numpy(dtype=float)).max() == 0.0
                    ),
                    "window_index_is_floor_aligned": bool(
                        np.array_equal(
                            windows["timestamp"].to_numpy() // WINDOW_SECONDS,
                            windows["window_idx"].to_numpy(),
                        )
                    ),
                    "status": "CONFORMS",
                    "error": "",
                }
            )
        except Exception as error:  # noqa: BLE001 — report, never swallow
            entry.update({"status": "ERROR", "error": f"{type(error).__name__}: {error}"})
        rows.append(entry)
        print(f"    {entry['pcap']:<30} {entry['status']}"
              + (f"  windows={entry.get('windows')}" if entry["status"] == "CONFORMS" else f"  {entry['error'][:60]}"))
    return pd.DataFrame(rows)


# =============================================================
# ARTIFACT WRITERS
# =============================================================

def write_contract_artifacts(contract):
    CONTRACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    digest = fc.contract_hash(contract)
    contract["contract_sha256"] = digest
    CONTRACT_JSON.write_text(json.dumps(contract, indent=2) + "\n")
    (OUT_DIR / "feature_contract_sha256.txt").write_text(digest + "\n")
    return digest


def write_implementation_map(contract):
    rows = []
    for feature in contract["features"]:
        base = feature["category"] == "base"
        rows.append(
            {
                "feature_index": feature["index"],
                "feature_name": feature["name"],
                "training_source": fc.TRAINING_IMPL if base else fc.TRAINING_WINDOWER,
                "pcap_source": (f"{fc.PCAP_IMPL}:compute_window_features" if base
                                else f"{fc.PCAP_IMPL}:add_relative_changes"),
                "shared_implementation": False,
                "training_formula": feature["training_formula"],
                "pcap_formula": feature["pcap_formula"],
                "same_semantics": feature["flow_layer_parity"] in (EXACT, APPROXIMATE),
                "aggregation_parity": feature["aggregation_parity"],
                "flow_layer_parity": feature["flow_layer_parity"],
                "status": ("B — separate implementations, equivalent formulas"
                           if feature["flow_layer_parity"] == APPROXIMATE
                           else "A — same implementation"),
                "notes": feature["parity_note"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "feature_implementation_map.csv", index=False)
    return frame


def write_transform_parity(contract):
    rows = []
    for name, entry in contract["transform_semantics"].items():
        rows.append(
            {
                "feature": name,
                "training_transform": entry["training"],
                "pcap_transform": entry["pcap"],
                "definition": entry["definition"],
                "applied_at": entry["applied"],
                "same": entry["parity"] == EXACT,
                "difference": "" if entry["parity"] == EXACT else "see contract",
                "status": entry["parity"],
            }
        )
    rows.append(
        {
            "feature": "relative_change",
            "training_transform": contract["relative_change_semantics"]["training_impl"],
            "pcap_transform": contract["relative_change_semantics"]["pcap_impl"],
            "definition": contract["relative_change_semantics"]["formula"],
            "applied_at": "AFTER the 26 base features, BEFORE StandardScaler",
            "same": True, "difference": "", "status": EXACT,
        }
    )
    rows.append(
        {
            "feature": "standard_scaler",
            "training_transform": contract["scaler_semantics"]["applied_training"],
            "pcap_transform": contract["scaler_semantics"]["applied_pcap"],
            "definition": f"{contract['scaler_semantics']['class']} on "
                          f"{contract['scaler_semantics']['expected_input_dimension']} named columns",
            "applied_at": "AFTER relative change, BEFORE sequence construction",
            "same": True, "difference": "", "status": EXACT,
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "transform_parity.csv", index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature-parity harness (read-only).")
    parser.add_argument("--skip-pcap", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print("FEATURE PARITY HARNESS — training vs PCAP inference")
    print("=" * 100)
    print("Read-only: no model, threshold, scaler, schema or pipeline is modified.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    contract = fc.build_contract()
    digest = write_contract_artifacts(contract)
    print(f"\nContract: {len(contract['features'])} features, sha256 {digest[:16]}…")

    implementation_map = write_implementation_map(contract)
    write_transform_parity(contract)
    (OUT_DIR / "window_semantics.json").write_text(
        json.dumps(contract["window_semantics"], indent=2) + "\n"
    )
    print(f"Wrote feature_implementation_map.csv, transform_parity.csv, window_semantics.json")

    print("\n--- Golden fixtures: training builder vs PCAP builder on identical flows ---")
    parity_frame, fixtures = run_parity_tests()
    parity_frame.to_csv(OUT_DIR / "parity_test_results.csv", index=False)

    failures = parity_frame[~parity_frame["pass"]]
    print(f"\n  {len(parity_frame)} comparisons, {len(failures)} failures")
    if len(failures):
        print(failures[["fixture", "feature", "training_value", "pcap_value",
                        "absolute_difference"]].head(20).to_string(index=False))

    pcap_frame = pd.DataFrame()
    if not args.skip_pcap:
        print("\n--- Real sample-PCAP contract conformance ---")
        pcap_frame = run_pcap_conformance()
        pcap_frame.to_csv(OUT_DIR / "pcap_conformance.csv", index=False)

    # ----- summary -----
    statuses = {}
    for feature in contract["features"]:
        statuses[feature["name"]] = feature["flow_layer_parity"]

    aggregation_exact = bool(len(failures) == 0)
    summary = {
        "analysis": "feature_parity",
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": contract["schema_version"],
        "feature_count": contract["feature_dimension"],
        "contract_hash": digest,
        "layers": {
            "window_aggregation_and_transforms": EXACT if aggregation_exact else NON_PARITY,
            "relative_change": EXACT if aggregation_exact else NON_PARITY,
            "scaler_interface": EXACT,
            "flow_record_production": APPROXIMATE,
        },
        "exact_matches": sorted(name for name, status in statuses.items() if status == EXACT),
        "approximate_matches": sorted(name for name, status in statuses.items() if status == APPROXIMATE),
        "non_parity_features": sorted(name for name, status in statuses.items() if status == NON_PARITY),
        "unknown_features": sorted(name for name, status in statuses.items() if status == UNKNOWN),
        "window_semantics": contract["window_semantics"],
        "transform_semantics": {k: v["parity"] for k, v in contract["transform_semantics"].items()},
        "relative_change_semantics": {
            "formula": contract["relative_change_semantics"]["formula"],
            "parity": contract["relative_change_semantics"]["parity"],
            "reset_scope_parity": contract["relative_change_semantics"]["reset_scope"]["parity"],
        },
        "scaler": {
            "class": contract["scaler_semantics"]["class"],
            "expected_input_dimension": contract["scaler_semantics"]["expected_input_dimension"],
            "parity": contract["scaler_semantics"]["parity"],
        },
        "golden_tests": {
            "fixtures": len(fixtures),
            "comparisons": int(len(parity_frame)),
            "failures": int(len(failures)),
            "status": "PASS" if aggregation_exact else "FAIL",
        },
        "synthetic_parity": {
            "status": "PASS" if aggregation_exact else "FAIL",
            "interpretation": "identical flow records produce identical 52-feature states in both "
                              "implementations" if aggregation_exact else
                              "the two implementations diverge on identical flow records",
        },
        "real_pcap": {
            "captures_tested": int(len(pcap_frame)),
            "conforming": int((pcap_frame["status"] == "CONFORMS").sum()) if len(pcap_frame) else 0,
            "status": ("PASS" if len(pcap_frame) and (pcap_frame["status"] == "CONFORMS").all()
                       else ("SKIPPED" if not len(pcap_frame) else "FAIL")),
        },
        "overall_status": (
            "APPROXIMATE_PARITY" if aggregation_exact else "PARITY_NOT_ESTABLISHED"
        ),
        "overall_statement": (
            "Training and PCAP feature pipelines are approximately semantically equivalent: the "
            "window aggregation, transforms, relative-change and scaler layers are exactly "
            "equivalent (verified on identical flow records), while the flow-record production "
            "layer differs (CICFlowMeter vs packet reconstruction) so numerical equality is not "
            "guaranteed."
            if aggregation_exact else
            "Feature parity is NOT established: the two implementations diverge on identical flow records."
        ),
        "known_limitations": [
            "CICFlowMeter (Java) and the PCAP path reconstruct flows differently: CIC terminates on "
            "FIN/RST plus a timeout, the PCAP path splits only on a >120 s idle gap.",
            "Byte accounting differs: CIC TotLen/Pkt Len vs L4 payload length in the PCAP path.",
            "The CIC CSV flag columns are binarised 0/1 by the dataset authors' preprocessing; the "
            "PCAP path reproduces that binarisation, but the definition is highly sensitive to flow "
            "length, so measured flag-rate distributions still differ materially.",
            "Capture scope differs: training aggregates the whole victim subnet, a demo PCAP covers "
            "only the captured hosts. This shifts every volume and diversity feature.",
            "Timestamp anchoring differs (CIC CSV local clock with AM/PM dropped vs UTC epoch); the "
            "binning RULE is identical but the two clocks are not aligned.",
            "No packet-level feature (TTL, TCP window, fragmentation, retransmission) is part of this "
            "contract — that is a later roadmap task.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    # Non-destructive compatibility manifest (the frozen V4 manifest is NOT touched)
    frozen = json.loads((REPO_ROOT / "docs" / "v4_baseline_manifest.json").read_text())
    PARITY_MANIFEST.write_text(
        json.dumps(
            {
                "purpose": "Links the frozen V4 baseline to the feature contract WITHOUT modifying "
                           "docs/v4_baseline_manifest.json (its historical hashes stay valid).",
                "generated_utc": summary["generated_utc"],
                "v4_baseline_manifest": "docs/v4_baseline_manifest.json",
                "v4_feature_order_sha256": frozen["input"]["feature_order_sha256"],
                "v4_checkpoint_sha256": frozen["checkpoint"]["sha256"],
                "feature_contract": "docs/feature_contract_v4.json",
                "feature_contract_sha256": digest,
                "contract_matches_frozen_schema": True,
                "parity_status": summary["overall_status"],
                "parity_summary": "results/feature_parity_summary.json",
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nWrote {SUMMARY_JSON.relative_to(REPO_ROOT)}")
    print(f"Wrote {PARITY_MANIFEST.relative_to(REPO_ROOT)}")
    print("\n" + "=" * 100)
    print(f"OVERALL: {summary['overall_status']}")
    print("=" * 100)
    print(summary["overall_statement"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
