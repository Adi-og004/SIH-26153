"""
=============================================================
V4 FEATURE CONTRACT — canonical, read-only
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

The single machine-readable description of the 52-dimensional V4 network
state: what each feature is, which code produces it in the TRAINING path
and in the PCAP inference path, and where the two are only approximately
equivalent.

This module is DESCRIPTIVE. It deliberately does NOT rewire any pipeline:
`archive/create_network_states_v2.py` (training states) and
`src/pcap_ingest_v4.py` (PCAP inference) keep their own code exactly as
frozen. The reference implementations here exist so tests can compare both
pipelines against one written-down definition.

Authoritative sources, all read from disk, never hardcoded:
    data/processed/multiday_benchmark_K5/input_feature_names.txt   (52 names, order)
    data/processed/multiday_benchmark_K5/state_feature_names.txt   (26 base names)
    docs/v4_feature_schema.json                                    (frozen Task 1 copy)

Feature definitions below were read out of the implementations on
2026-09-19; every formula string quotes the actual code, not the name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
INPUT_NAMES_PATH = BENCHMARK_DIR / "input_feature_names.txt"
STATE_NAMES_PATH = BENCHMARK_DIR / "state_feature_names.txt"
FROZEN_SCHEMA_PATH = REPO_ROOT / "docs" / "v4_feature_schema.json"
CONTRACT_PATH = REPO_ROOT / "docs" / "feature_contract_v4.json"

TRAINING_IMPL = "archive/create_network_states_v2.py"
TRAINING_WINDOWER = "src/prepare_multiday_benchmark.py"
PCAP_IMPL = "src/pcap_ingest_v4.py"

WINDOW_SECONDS = 10
FEATURE_DIMENSION = 52
BASE_FEATURE_COUNT = 26
SEQUENCE_LENGTH = 10
RELATIVE_EPSILON = 1e-8
RELATIVE_CLIP = 10.0
DIVIDE_EPSILON = 1e-8
ENTROPY_EPSILON = 1e-12

INPUT_FEATURE_NAMES = [
    line.strip() for line in INPUT_NAMES_PATH.read_text().splitlines() if line.strip()
]
STATE_FEATURE_NAMES = [
    line.strip() for line in STATE_NAMES_PATH.read_text().splitlines() if line.strip()
]
BASE_FEATURES = INPUT_FEATURE_NAMES[:BASE_FEATURE_COUNT]
RELATIVE_FEATURES = INPUT_FEATURE_NAMES[BASE_FEATURE_COUNT:]

assert len(INPUT_FEATURE_NAMES) == FEATURE_DIMENSION
assert BASE_FEATURES == STATE_FEATURE_NAMES
assert RELATIVE_FEATURES == [f"{name}_rel_change" for name in BASE_FEATURES]


# =============================================================
# PARITY VOCABULARY
# =============================================================

EXACT = "EXACT"              # same computational definition AND identical implementation behaviour
APPROXIMATE = "APPROXIMATE"  # same conceptual definition, implementation differences remain
NON_PARITY = "NON_PARITY"    # different definition
UNKNOWN = "UNKNOWN"          # insufficient evidence


# =============================================================
# THE 26 BASE FEATURES
# =============================================================
# aggregation: how flow rows inside one 10 s window are combined.
# training_formula / pcap_formula: quoted from the implementations.
# flow_layer_parity: parity of the FLOW RECORDS each path feeds in.
# aggregation_parity: parity of the window aggregation itself.

_BASE = [
    ("flow_count", "count", "int", False, False,
     'grouped.agg(flow_count=("Timestamp", "size")) — number of CICFlowMeter flow rows whose Timestamp floors into the window',
     'g.agg(flow_count=("window_idx", "size")) — number of reconstructed flows whose FIRST packet falls in the window',
     APPROXIMATE,
     "CICFlowMeter terminates flows on FIN/RST and a 120 s timeout; the PCAP path splits a 5-tuple only on a >120 s idle gap, so one CIC flow may map to a different number of reconstructed flows. Capture scope also differs (whole victim subnet vs the hosts actually captured)."),

    ("log_flow_count", "derived", "float", True, False,
     "safe_log1p(flow_count)", "_safe_log1p(flow_count)", APPROXIMATE,
     "inherits flow_count"),

    ("log_total_packets", "derived", "float", True, False,
     'safe_log1p(sum over flows of (Tot Fwd Pkts + Tot Bwd Pkts))',
     '_safe_log1p(sum over flows of tot_pkts), tot_pkts = packets attributed to the reconstructed flow',
     APPROXIMATE, "inherits the flow-segmentation and capture-scope difference"),

    ("log_total_bytes", "derived", "float", True, False,
     'safe_log1p(sum over flows of (TotLen Fwd Pkts + TotLen Bwd Pkts))',
     '_safe_log1p(sum over flows of total_bytes), total_bytes = sum of L4 payload lengths',
     APPROXIMATE,
     "CICFlowMeter TotLen counts packet lengths as its engine defines them; the PCAP path sums L4 payload bytes. Same concept (bytes in window), different byte accounting."),

    ("window_packet_rate", "derived", "float", False, False,
     "states['total_packets'] / WINDOW_SECONDS (window totals, not CIC per-flow rate fields)",
     "agg['total_packets'] / WINDOW_SECONDS", APPROXIMATE, "inherits total_packets"),

    ("window_byte_rate", "derived", "float", False, False,
     "states['total_bytes'] / WINDOW_SECONDS",
     "agg['total_bytes'] / WINDOW_SECONDS", APPROXIMATE, "inherits total_bytes"),

    ("syn_rate", "mean", "float", False, True,
     'per flow: SYN Flag Cnt / (total_packets + 1e-8); then mean over flows in the window. In the CIC CSVs the "* Flag Cnt" columns are BOOLEAN (0/1).',
     'per flow: int(any SYN in flow) / (tot_pkts + 1e-8); then mean over flows',
     APPROXIMATE,
     "Definition matches (binarised flag over flow packet count). Measured distributions still differ materially: the binarised definition is highly sensitive to flow length, and reconstructed flows are shorter than CICFlowMeter's."),

    ("ack_rate", "mean", "float", False, True,
     "per flow: ACK Flag Cnt / (total_packets + 1e-8); mean over flows",
     "per flow: int(any ACK) / (tot_pkts + 1e-8); mean over flows", APPROXIMATE, "as syn_rate"),

    ("rst_rate", "mean", "float", False, True,
     "per flow: RST Flag Cnt / (total_packets + 1e-8); mean over flows",
     "per flow: int(any RST) / (tot_pkts + 1e-8); mean over flows", APPROXIMATE, "as syn_rate"),

    ("fin_rate", "mean", "float", False, True,
     "per flow: FIN Flag Cnt / (total_packets + 1e-8); mean over flows",
     "per flow: int(any FIN) / (tot_pkts + 1e-8); mean over flows", APPROXIMATE,
     "as syn_rate; measured divergence is largest here because flow-termination semantics differ"),

    ("psh_rate", "mean", "float", False, True,
     "per flow: PSH Flag Cnt / (total_packets + 1e-8); mean over flows",
     "per flow: int(any PSH) / (tot_pkts + 1e-8); mean over flows", APPROXIMATE, "as syn_rate"),

    ("urg_rate", "mean", "float", False, True,
     "per flow: URG Flag Cnt / (total_packets + 1e-8); mean over flows",
     "per flow: int(any URG) / (tot_pkts + 1e-8); mean over flows", APPROXIMATE, "as syn_rate"),

    ("mean_pkt_len", "mean", "float", False, False,
     'mean over flows of the CIC per-flow column "Pkt Len Mean" (mean of a mean, not a pooled mean)',
     'mean over flows of pkt_len_mean, where pkt_len_mean = mean L4 payload length of the flow',
     APPROXIMATE, "same mean-of-per-flow-mean structure; byte basis differs (CIC packet length vs L4 payload)"),

    ("std_pkt_len", "mean", "float", False, False,
     'mean over flows of the CIC per-flow column "Pkt Len Std" (mean of per-flow std, NOT a pooled std)',
     'mean over flows of pkt_len_std = sample std (ddof=1) of the flow\'s L4 payload lengths',
     APPROXIMATE, "same structure; byte basis differs"),

    ("log_mean_flow_duration", "mean", "float", True, False,
     'safe_log1p(mean over flows of "Flow Duration") — CICFlowMeter Flow Duration is in MICROSECONDS',
     "_safe_log1p(mean over flows of flow_duration_us) — (last-first packet time) * 1e6, MICROSECONDS",
     APPROXIMATE, "units aligned (µs); the value differs because flow lifetimes differ between engines"),

    ("log_mean_fwd_iat", "mean", "float", True, False,
     'safe_log1p(mean over flows of "Fwd IAT Mean") — MICROSECONDS',
     "_safe_log1p(mean over flows of fwd_iat_mean_us) — mean diff of forward packet times * 1e6",
     APPROXIMATE, "units aligned; residual divergence measured"),

    ("log_mean_bwd_iat", "mean", "float", True, False,
     'safe_log1p(mean over flows of "Bwd IAT Mean") — MICROSECONDS',
     "_safe_log1p(mean over flows of bwd_iat_mean_us)", APPROXIMATE, "units aligned"),

    ("fwd_packet_ratio", "mean", "float", False, False,
     "per flow: safe_divide(Tot Fwd Pkts, total_packets); mean over flows",
     "per flow: _safe_divide(tot_fwd_pkts, tot_pkts); mean over flows, forward = direction of the flow's FIRST packet",
     APPROXIMATE,
     "identical formula and identical initiator-direction convention; values differ because flow segmentation differs"),

    ("bwd_packet_ratio", "mean", "float", False, False,
     "per flow: safe_divide(Tot Bwd Pkts, total_packets); mean over flows",
     "per flow: _safe_divide(tot_bwd_pkts, tot_pkts); mean over flows", APPROXIMATE, "as fwd_packet_ratio"),

    ("fwd_byte_ratio", "mean", "float", False, False,
     "per flow: safe_divide(TotLen Fwd Pkts, total_bytes); mean over flows",
     "per flow: _safe_divide(totlen_fwd, total_bytes); mean over flows", APPROXIMATE, "as fwd_packet_ratio"),

    ("bwd_byte_ratio", "mean", "float", False, False,
     "per flow: safe_divide(TotLen Bwd Pkts, total_bytes); mean over flows",
     "per flow: _safe_divide(totlen_bwd, total_bytes); mean over flows", APPROXIMATE, "as fwd_packet_ratio"),

    ("mean_down_up_ratio", "mean", "float", False, False,
     'mean over flows of the CIC column "Down/Up Ratio" (integer-valued in the training CSVs)',
     "mean over flows of down_up_ratio = int(n_bwd // n_fwd), 0 when n_fwd == 0",
     APPROXIMATE, "both integer-valued per flow; CIC's exact integer rule is not published"),

    ("log_unique_dst_ports", "nunique", "float", True, False,
     'safe_log1p(nunique of "Dst Port" over the flows in the window)',
     "_safe_log1p(nunique of dst_port over the flows in the window)", APPROXIMATE,
     "identical formula; strongly capture-scope dependent"),

    ("unique_protocols", "nunique", "int", False, False,
     'nunique of "Protocol" over the flows in the window',
     "nunique of protocol over the flows in the window", APPROXIMATE, "identical formula; scope dependent"),

    ("dst_port_entropy", "entropy", "float", False, True,
     'entropy_from_counts(Dst Port): Shannon base-2 over value_counts, p + 1e-12, 0.0 when <= 1 distinct value',
     "_entropy_from_counts(dst_port): verbatim port of the same function", APPROXIMATE,
     "identical formula; scope dependent"),

    ("protocol_entropy", "entropy", "float", False, True,
     "entropy_from_counts(Protocol)", "_entropy_from_counts(protocol)", APPROXIMATE,
     "identical formula; scope dependent"),
]

assert [row[0] for row in _BASE] == BASE_FEATURES, "contract order must match the frozen schema"


# =============================================================
# WINDOW / TRANSFORM / SCALER SEMANTICS
# =============================================================

WINDOW_SEMANTICS = {
    "window_seconds": WINDOW_SECONDS,
    "training_binning": 'df["Timestamp"].dt.floor("10s") in archive/create_network_states_v2.py',
    "pcap_binning": "int(flow_start_epoch // 10) - win0 in src/pcap_ingest_v4.py",
    "interval": "[t, t+10) — left-closed, right-open (floor semantics on both sides)",
    "origin": "epoch-aligned multiples of 10 s in both paths (floor of an absolute timestamp)",
    "row_assignment": "a flow belongs to the window of its START time (CIC Timestamp / first packet), "
                      "and ALL of its statistics are attributed to that window",
    "rounding": "floor, never round-half; no tolerance band",
    "ordering": "groupby(..., sort=True) in training; groupby('window_idx', sort=True) in the PCAP path",
    "empty_windows": "produce NO row in either path — the window simply does not exist. "
                     "Downstream, the 10-second contiguity gate rejects any sequence spanning the gap "
                     "(np.all(diffs == 10) in prepare_multiday_benchmark.py, mirrored by "
                     "build_sequences in pcap_ingest_v4.py).",
    "first_window": "kept as-is; its relative-change row is forced to 0.0",
    "last_window": "kept as-is; sequences needing K future targets simply are not formed",
    "timezone": {
        "training": "naive local timestamps parsed from the CIC CSV Timestamp column; no tz conversion "
                    "is applied anywhere in the training path",
        "pcap": "absolute UTC epoch seconds from the capture",
        "parity": APPROXIMATE,
        "note": "Both floor to 10 s multiples, so bin WIDTH and boundary RULE are identical, but the two "
                "paths are not anchored to the same clock. The CSE-CIC-IDS2018 CSV clock is Atlantic "
                "local with the AM/PM marker dropped, which cannot be reconciled with UTC epoch bins "
                "without an explicit offset. This matters only when aligning a PCAP window to a "
                "dataset window; it does not change any feature value.",
    },
}

TRANSFORM_SEMANTICS = {
    "safe_log1p": {
        "definition": "to_numeric(errors='coerce') -> replace(+/-inf, NaN) -> fillna(0.0) -> clip(lower=0.0) -> log1p",
        "training": "archive/create_network_states_v2.py:safe_log1p",
        "pcap": "src/pcap_ingest_v4.py:_safe_log1p (documented verbatim port)",
        "parity": EXACT,
        "applied": "AFTER window aggregation, BEFORE relative change",
    },
    "safe_divide": {
        "definition": "numerator / (denominator.abs() + 1e-8) -> replace(+/-inf, NaN) -> fillna(0.0)",
        "training": "archive/create_network_states_v2.py:safe_divide",
        "pcap": "src/pcap_ingest_v4.py:_safe_divide (verbatim port)",
        "parity": EXACT,
        "applied": "BEFORE window aggregation (per flow row)",
    },
    "entropy_from_counts": {
        "definition": "Shannon entropy base 2 of value_counts(dropna=True); p + 1e-12 inside the log; "
                      "returns 0.0 when <= 1 distinct value",
        "training": "archive/create_network_states_v2.py:entropy_from_counts",
        "pcap": "src/pcap_ingest_v4.py:_entropy_from_counts (verbatim port)",
        "parity": EXACT,
        "applied": "AT window aggregation",
    },
    "flag_rate": {
        "definition": "flag_count / (total_packets + 1e-8) per flow, then mean over flows",
        "training": "archive/create_network_states_v2.py (packet_denominator = total_packets + 1e-8)",
        "pcap": "src/pcap_ingest_v4.py reconstruct_flows (denom = tot_pkts + 1e-8)",
        "parity": EXACT,
        "applied": "BEFORE window aggregation",
    },
    "window_rate": {
        "definition": "window total / 10 seconds",
        "training": "states['total_packets'] / WINDOW_SECONDS",
        "pcap": "agg['total_packets'] / WINDOW_SECONDS",
        "parity": EXACT,
        "applied": "AFTER window aggregation",
    },
    "nan_inf_handling_state": {
        "definition": "the 26 base features are replace([inf,-inf], NaN).fillna(0.0) before leaving the builder",
        "training": "create_network_states_v2.py final numerical validation + "
                    "prepare_multiday_benchmark.py df[STATE_FEATURES].replace([inf,-inf], NaN).fillna(0.0)",
        "pcap": "compute_window_features: out[BASE_FEATURES].replace([inf,-inf], NaN).fillna(0.0)",
        "parity": EXACT,
        "applied": "AFTER window aggregation, BEFORE relative change",
    },
    "scaled_nan_clip": {
        "definition": "np.nan_to_num(nan=0.0, posinf=10.0, neginf=-10.0) then float32",
        "training": "prepare_multiday_benchmark.py transform_df",
        "pcap": "pcap_ingest_v4.py build_sequences",
        "parity": EXACT,
        "applied": "AFTER StandardScaler, BEFORE sequence construction",
    },
}

RELATIVE_CHANGE_SEMANTICS = {
    "formula": "r_t = (s_t - s_{t-1}) / (|s_{t-1}| + 1e-8)",
    "previous_state": "the immediately preceding ROW of the state table, i.e. the previous existing "
                      "10-second window — NOT necessarily t-10 s when a window is missing",
    "missing_previous": "the first row is forced to exactly 0.0 (rel.iloc[0] = 0.0)",
    "zero_denominator": "handled by the +1e-8 epsilon, not by masking; a 0 -> x change yields a very "
                        "large ratio which is then clipped",
    "non_finite": "replace([inf, -inf], NaN).fillna(0.0)",
    "clipping": "clip(-10.0, +10.0) AFTER the division",
    "signed": True,
    "log_order": "computed on the ALREADY log-transformed base features — the log is applied during "
                 "window aggregation, the relative change afterwards",
    "reset_scope": {
        "training": "per SOURCE DAY: prepare_multiday_benchmark.py computes relative changes inside "
                    "load_and_prep_day BEFORE the train/val split, so day boundaries reset the chain",
        "pcap": "per CAPTURE: add_relative_changes runs on the window table of one pcap file",
        "parity": EXACT,
        "note": "Both reset at the boundary of one contiguous capture/day unit and never carry a "
                "relative change across that boundary.",
    },
    "training_impl": "src/prepare_multiday_benchmark.py load_and_prep_day",
    "pcap_impl": "src/pcap_ingest_v4.py add_relative_changes",
    "parity": EXACT,
}

SCALER_SEMANTICS = {
    "class": "sklearn.preprocessing.StandardScaler",
    "expected_input_dimension": FEATURE_DIMENSION,
    "feature_order": "docs/v4_feature_schema.json / input_feature_names.txt (identical)",
    "frozen_benchmark_scaler": "data/processed/multiday_benchmark_K5/input_scaler.pkl",
    "fitted_on": "the TRAIN partition only (8,575 windows for the frozen benchmark); rotated folds "
                 "each fit their own scaler on their own train partition",
    "applied_training": "prepare_multiday_benchmark.py, at dataset build time, BEFORE sequence construction",
    "applied_pcap": "pcap_ingest_v4.build_sequences, on a DataFrame reindexed to the canonical column "
                    "names so sklearn validates the order, BEFORE sequence construction",
    "other_transforms": "none — no PCA, no whitening, no feature selection anywhere in either path",
    "parity": EXACT,
}


# =============================================================
# CONTRACT ASSEMBLY + HASH
# =============================================================

def build_contract() -> dict:
    features = []
    for index, name in enumerate(BASE_FEATURES):
        (
            _name, aggregation, dtype, is_log, is_ratio_or_entropy,
            training_formula, pcap_formula, flow_parity, note,
        ) = _BASE[index]
        features.append(
            {
                "index": index,
                "name": name,
                "category": "base",
                "source_level": "derived" if aggregation == "derived" else "flow",
                "aggregation": aggregation,
                "aggregation_window": f"{WINDOW_SECONDS}s, [t, t+10)",
                "training_formula": training_formula,
                "pcap_formula": pcap_formula,
                "transform": "safe_log1p" if is_log else ("safe_divide/entropy" if is_ratio_or_entropy else "none"),
                "log_transformed": is_log,
                "ratio_or_entropy": is_ratio_or_entropy,
                "relative_change_feature": False,
                "expected_unit": _UNITS[name],
                "expected_numeric_type": dtype,
                "missing_value_behaviour": "non-finite -> NaN -> 0.0 (see transform_semantics.nan_inf_handling_state)",
                "depends_on": _DEPENDENCIES.get(name, []),
                "aggregation_parity": EXACT,
                "flow_layer_parity": flow_parity,
                "parity_note": note,
            }
        )

    for offset, name in enumerate(RELATIVE_FEATURES):
        base_name = BASE_FEATURES[offset]
        base_entry = features[offset]
        features.append(
            {
                "index": BASE_FEATURE_COUNT + offset,
                "name": name,
                "category": "relative_change",
                "source_level": "derived",
                "aggregation": "none (cross-window difference)",
                "aggregation_window": f"{WINDOW_SECONDS}s vs the preceding window row",
                "training_formula": RELATIVE_CHANGE_SEMANTICS["formula"] + f"  on {base_name}",
                "pcap_formula": RELATIVE_CHANGE_SEMANTICS["formula"] + f"  on {base_name}",
                "transform": "relative_change(eps=1e-8, clip=+/-10, first row = 0)",
                "log_transformed": False,
                "ratio_or_entropy": True,
                "relative_change_feature": True,
                "expected_unit": "dimensionless ratio (signed, clipped to [-10, 10])",
                "expected_numeric_type": "float",
                "missing_value_behaviour": "first row 0.0; non-finite -> 0.0",
                "depends_on": [base_name],
                "aggregation_parity": EXACT,
                "flow_layer_parity": base_entry["flow_layer_parity"],
                "parity_note": f"transform is identical in both paths; inherits the parity of {base_name}",
            }
        )

    contract = {
        "schema_version": "v4",
        "contract_version": "1.0.0",
        "window_seconds": WINDOW_SECONDS,
        "feature_dimension": FEATURE_DIMENSION,
        "base_feature_count": BASE_FEATURE_COUNT,
        "relative_feature_count": FEATURE_DIMENSION - BASE_FEATURE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "authoritative_order_file": str(INPUT_NAMES_PATH.relative_to(REPO_ROOT)),
        "frozen_schema_file": str(FROZEN_SCHEMA_PATH.relative_to(REPO_ROOT)),
        "training_implementation": {
            "state_builder": TRAINING_IMPL,
            "windower_and_relative_change": TRAINING_WINDOWER,
            "input": "CSE-CIC-IDS2018 CICFlowMeter flow CSVs (pre-computed by the dataset authors)",
        },
        "pcap_implementation": {
            "module": PCAP_IMPL,
            "input": "raw pcap -> parse_pcap_streaming -> reconstruct_flows -> compute_window_features",
        },
        "window_semantics": WINDOW_SEMANTICS,
        "transform_semantics": TRANSFORM_SEMANTICS,
        "relative_change_semantics": RELATIVE_CHANGE_SEMANTICS,
        "scaler_semantics": SCALER_SEMANTICS,
        "features": features,
        "parity_vocabulary": {
            EXACT: "same computational definition and identical implementation behaviour",
            APPROXIMATE: "same conceptual definition; implementation differences remain, numerical equality not guaranteed",
            NON_PARITY: "different definition",
            UNKNOWN: "insufficient evidence",
        },
    }
    return contract


_UNITS = {
    "flow_count": "flows per 10 s window",
    "log_flow_count": "log1p(flows)",
    "log_total_packets": "log1p(packets)",
    "log_total_bytes": "log1p(bytes)",
    "window_packet_rate": "packets per second",
    "window_byte_rate": "bytes per second",
    "syn_rate": "flag-per-packet fraction, averaged over flows",
    "ack_rate": "flag-per-packet fraction, averaged over flows",
    "rst_rate": "flag-per-packet fraction, averaged over flows",
    "fin_rate": "flag-per-packet fraction, averaged over flows",
    "psh_rate": "flag-per-packet fraction, averaged over flows",
    "urg_rate": "flag-per-packet fraction, averaged over flows",
    "mean_pkt_len": "bytes",
    "std_pkt_len": "bytes",
    "log_mean_flow_duration": "log1p(microseconds)",
    "log_mean_fwd_iat": "log1p(microseconds)",
    "log_mean_bwd_iat": "log1p(microseconds)",
    "fwd_packet_ratio": "dimensionless fraction [0, 1]",
    "bwd_packet_ratio": "dimensionless fraction [0, 1]",
    "fwd_byte_ratio": "dimensionless fraction [0, 1]",
    "bwd_byte_ratio": "dimensionless fraction [0, 1]",
    "mean_down_up_ratio": "dimensionless integer ratio, averaged over flows",
    "log_unique_dst_ports": "log1p(distinct ports)",
    "unique_protocols": "distinct protocol numbers",
    "dst_port_entropy": "bits",
    "protocol_entropy": "bits",
}

_DEPENDENCIES = {
    "log_flow_count": ["flow_count"],
    "log_total_packets": ["Tot Fwd Pkts", "Tot Bwd Pkts"],
    "log_total_bytes": ["TotLen Fwd Pkts", "TotLen Bwd Pkts"],
    "window_packet_rate": ["log_total_packets"],
    "window_byte_rate": ["log_total_bytes"],
    "syn_rate": ["SYN Flag Cnt", "total_packets"],
    "ack_rate": ["ACK Flag Cnt", "total_packets"],
    "rst_rate": ["RST Flag Cnt", "total_packets"],
    "fin_rate": ["FIN Flag Cnt", "total_packets"],
    "psh_rate": ["PSH Flag Cnt", "total_packets"],
    "urg_rate": ["URG Flag Cnt", "total_packets"],
    "log_mean_flow_duration": ["Flow Duration"],
    "log_mean_fwd_iat": ["Fwd IAT Mean"],
    "log_mean_bwd_iat": ["Bwd IAT Mean"],
    "fwd_packet_ratio": ["Tot Fwd Pkts", "total_packets"],
    "bwd_packet_ratio": ["Tot Bwd Pkts", "total_packets"],
    "fwd_byte_ratio": ["TotLen Fwd Pkts", "total_bytes"],
    "bwd_byte_ratio": ["TotLen Bwd Pkts", "total_bytes"],
    "mean_down_up_ratio": ["Down/Up Ratio"],
    "log_unique_dst_ports": ["Dst Port"],
    "unique_protocols": ["Protocol"],
    "dst_port_entropy": ["Dst Port"],
    "protocol_entropy": ["Protocol"],
}


def canonical_contract_payload(contract: dict) -> str:
    """Deterministic canonical form: only the semantics that define the contract."""
    payload = {
        "schema_version": contract["schema_version"],
        "window_seconds": contract["window_seconds"],
        "feature_dimension": contract["feature_dimension"],
        "sequence_length": contract["sequence_length"],
        "features": [
            {
                "index": feature["index"],
                "name": feature["name"],
                "category": feature["category"],
                "aggregation": feature["aggregation"],
                "training_formula": feature["training_formula"],
                "pcap_formula": feature["pcap_formula"],
                "transform": feature["transform"],
                "expected_unit": feature["expected_unit"],
            }
            for feature in contract["features"]
        ],
        "window_semantics": {
            key: contract["window_semantics"][key]
            for key in ("window_seconds", "interval", "origin", "row_assignment",
                        "rounding", "empty_windows", "first_window", "last_window")
        },
        "relative_change_semantics": {
            key: contract["relative_change_semantics"][key]
            for key in ("formula", "previous_state", "missing_previous", "zero_denominator",
                        "non_finite", "clipping", "signed", "log_order")
        },
        "transform_semantics": {
            key: value["definition"] for key, value in contract["transform_semantics"].items()
        },
        "scaler_semantics": {
            key: contract["scaler_semantics"][key]
            for key in ("class", "expected_input_dimension", "other_transforms")
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def contract_hash(contract: dict | None = None) -> str:
    contract = contract or build_contract()
    return hashlib.sha256(canonical_contract_payload(contract).encode("utf-8")).hexdigest()


def load_contract() -> dict:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"feature contract not found: {CONTRACT_PATH}")
    return json.loads(CONTRACT_PATH.read_text())


# =============================================================
# REFERENCE IMPLEMENTATIONS (used by tests only)
# =============================================================

def reference_relative_change(state_frame: pd.DataFrame) -> pd.DataFrame:
    """The contract's relative-change definition, written once."""
    state = state_frame[BASE_FEATURES].astype(float).reset_index(drop=True)
    delta = state.diff().fillna(0.0)
    previous = state.shift(1)
    relative = delta / (previous.abs() + RELATIVE_EPSILON)
    relative.iloc[0] = 0.0
    relative = (
        relative.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-RELATIVE_CLIP, RELATIVE_CLIP)
    )
    relative.columns = RELATIVE_FEATURES
    return relative


def window_index(epoch_seconds) -> np.ndarray:
    """The contract's binning rule: floor to a 10 s epoch-aligned bin."""
    return np.floor(np.asarray(epoch_seconds, dtype=float) / WINDOW_SECONDS).astype(np.int64)
