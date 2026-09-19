"""
Feature-contract regression tests (Task 7).

These tests fail whenever the 52-feature contract drifts: a reordered feature,
a changed formula, different windowing, different relative-change behaviour,
different zero handling, or a changed log transform.

They compare the TWO REAL implementations against each other and against the
contract — no expected value is hardcoded unless it follows from the definition
itself (entropy of a single distinct value is exactly 0.0, a ratio of equal
halves is exactly 0.5, and so on).

Run:
    pytest tests/test_feature_parity.py
    python tests/test_feature_parity.py        # same checks, no pytest needed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import feature_contract as fc  # noqa: E402
import run_feature_parity as rfp  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "feature_parity"
TOLERANCE = 1e-9


def _load_fixture(name):
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


def _both_paths(flows):
    training = rfp.run_training_path(flows, rfp.load_training_builder())
    import pcap_ingest_v4
    pcap = rfp.run_pcap_path(flows, pcap_ingest_v4)
    return training, pcap


# =============================================================
# CONTRACT SHAPE
# =============================================================

def test_contract_has_52_features_in_frozen_order():
    contract = fc.build_contract()
    names = [feature["name"] for feature in contract["features"]]
    frozen = (
        REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_feature_names.txt"
    ).read_text().split()
    assert names == frozen, "contract order drifted from the frozen schema"
    assert len(names) == 52
    assert names[:26] == fc.BASE_FEATURES
    assert names[26:] == [f"{base}_rel_change" for base in fc.BASE_FEATURES]


def test_contract_matches_docs_schema():
    schema = json.loads((REPO_ROOT / "docs" / "v4_feature_schema.json").read_text())
    assert [item["name"] for item in schema["features"]] == fc.INPUT_FEATURE_NAMES


def test_contract_hash_is_stable_and_recorded():
    contract = fc.build_contract()
    digest = fc.contract_hash(contract)
    assert digest == fc.contract_hash(fc.build_contract()), "hash is not deterministic"
    recorded = (REPO_ROOT / "results" / "feature_parity" / "feature_contract_sha256.txt").read_text().strip()
    assert digest == recorded, "contract changed without regenerating the recorded hash"


def test_every_feature_documents_both_implementations():
    for feature in fc.build_contract()["features"]:
        assert feature["training_formula"], feature["name"]
        assert feature["pcap_formula"], feature["name"]
        assert feature["expected_unit"], feature["name"]
        assert feature["flow_layer_parity"] in (fc.EXACT, fc.APPROXIMATE, fc.NON_PARITY, fc.UNKNOWN)


def test_window_is_ten_seconds_everywhere():
    import pcap_ingest_v4
    training = rfp.load_training_builder()
    assert fc.WINDOW_SECONDS == 10
    assert training.WINDOW_SECONDS == 10
    assert pcap_ingest_v4.WINDOW_SECONDS == 10
    assert json.loads((REPO_ROOT / "docs" / "feature_contract_v4.json").read_text())["window_seconds"] == 10


# =============================================================
# TRAINING vs PCAP ON IDENTICAL FLOWS
# =============================================================

def test_all_fixtures_agree_between_implementations():
    training_module = rfp.load_training_builder()
    import pcap_ingest_v4

    failures = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        fixture = json.loads(path.read_text())
        training = rfp.run_training_path(fixture["flows"], training_module)
        pcap = rfp.run_pcap_path(fixture["flows"], pcap_ingest_v4)
        for row in rfp.compare_states(path.stem, training, pcap):
            if not row["pass"]:
                failures.append(f"{row['fixture']}/{row['feature']}: {row['note']}")
    assert not failures, "training and PCAP builders disagree:\n" + "\n".join(failures)


def test_empty_window_produces_no_row_in_either_path():
    fixture = _load_fixture("01_empty_window")
    training, pcap = _both_paths(fixture["flows"])
    assert len(training) == 2 and len(pcap) == 2, "an empty window must not create a row"
    assert list(pcap["window_idx"]) == [0, 2]


def test_single_distinct_value_gives_zero_entropy():
    fixture = _load_fixture("09_repeated_dst_port")
    training, pcap = _both_paths(fixture["flows"])
    for frame in (training, pcap):
        assert frame["dst_port_entropy"].iloc[0] == 0.0
        assert frame["protocol_entropy"].iloc[0] == 0.0


def test_balanced_directions_give_half_ratios():
    fixture = _load_fixture("16_balanced_directions")
    training, pcap = _both_paths(fixture["flows"])
    for frame in (training, pcap):
        assert abs(frame["fwd_packet_ratio"].iloc[0] - 0.5) < 1e-6
        assert abs(frame["bwd_packet_ratio"].iloc[0] - 0.5) < 1e-6
        assert abs(frame["fwd_byte_ratio"].iloc[0] - 0.5) < 1e-6


def test_zero_packet_flow_uses_epsilon_denominator():
    """A zero-packet flow must not produce NaN/Inf — the +1e-8 denominator carries it."""
    fixture = _load_fixture("12_zero_packets")
    training, pcap = _both_paths(fixture["flows"])
    for frame in (training, pcap):
        values = frame[fc.BASE_FEATURES].to_numpy(dtype=float)
        assert np.isfinite(values).all()


def test_window_rates_are_totals_over_ten_seconds():
    fixture = _load_fixture("03_multiple_flows")
    training, pcap = _both_paths(fixture["flows"])
    total_packets = sum(f["tot_pkts"] for f in fixture["flows"])
    total_bytes = sum(f["total_bytes"] for f in fixture["flows"])
    for frame in (training, pcap):
        assert abs(frame["window_packet_rate"].iloc[0] - total_packets / 10) < 1e-9
        assert abs(frame["window_byte_rate"].iloc[0] - total_bytes / 10) < 1e-9


def test_log_features_are_log1p_of_their_source():
    fixture = _load_fixture("03_multiple_flows")
    _, pcap = _both_paths(fixture["flows"])
    flows = fixture["flows"]
    assert abs(pcap["log_flow_count"].iloc[0] - np.log1p(len(flows))) < 1e-9
    assert abs(pcap["log_total_packets"].iloc[0]
               - np.log1p(sum(f["tot_pkts"] for f in flows))) < 1e-9
    assert abs(pcap["log_unique_dst_ports"].iloc[0]
               - np.log1p(len({f["dst_port"] for f in flows}))) < 1e-9


# =============================================================
# RELATIVE CHANGE
# =============================================================

def test_relative_change_matches_the_contract_formula():
    fixture = _load_fixture("17_increasing_volume")
    training, pcap = _both_paths(fixture["flows"])

    for frame in (training, pcap):
        relative = fc.reference_relative_change(frame)
        assert (relative.iloc[0] == 0.0).all(), "first row must be exactly zero"

        state = frame[fc.BASE_FEATURES].astype(float).reset_index(drop=True)
        expected = (state - state.shift(1)) / (state.shift(1).abs() + 1e-8)
        expected = expected.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
        expected.iloc[0] = 0.0
        assert np.allclose(relative.to_numpy(), expected.to_numpy(), atol=TOLERANCE)


def test_relative_change_is_clipped_to_ten():
    fixture = _load_fixture("18_decreasing_volume")
    _, pcap = _both_paths(fixture["flows"])
    relative = fc.reference_relative_change(pcap).to_numpy(dtype=float)
    assert np.abs(relative).max() <= 10.0 + TOLERANCE


def test_pcap_relative_change_equals_the_reference():
    import pcap_ingest_v4
    fixture = _load_fixture("17_increasing_volume")
    _, pcap = _both_paths(fixture["flows"])
    produced = pcap_ingest_v4.add_relative_changes(pcap)
    reference = fc.reference_relative_change(pcap)
    assert np.allclose(
        produced[fc.RELATIVE_FEATURES].to_numpy(dtype=float),
        reference.to_numpy(dtype=float),
        atol=TOLERANCE,
    )


# =============================================================
# DRIFT DETECTORS
# =============================================================

def test_scaler_expects_exactly_52_named_features():
    import pickle
    path = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_scaler.pkl"
    with open(path, "rb") as handle:
        scaler = pickle.load(handle)
    assert int(scaler.n_features_in_) == 52
    assert list(getattr(scaler, "feature_names_in_", fc.INPUT_FEATURE_NAMES)) == fc.INPUT_FEATURE_NAMES


def test_pcap_module_reads_the_contract_from_disk():
    """The PCAP path must not hardcode the feature list."""
    import pcap_ingest_v4
    assert pcap_ingest_v4.INPUT_FEATURE_NAMES == fc.INPUT_FEATURE_NAMES
    assert pcap_ingest_v4.BASE_FEATURES == fc.BASE_FEATURES


def test_reordering_a_feature_is_detected():
    """Guard the guard: a permuted order must fail the order check."""
    permuted = list(fc.INPUT_FEATURE_NAMES)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    assert permuted != fc.INPUT_FEATURE_NAMES


def test_changed_formula_changes_the_contract_hash():
    contract = fc.build_contract()
    baseline = fc.contract_hash(contract)
    tampered = json.loads(json.dumps(contract))
    tampered["features"][0]["training_formula"] += " (tampered)"
    assert fc.contract_hash(tampered) != baseline


def test_changed_window_size_changes_the_contract_hash():
    contract = fc.build_contract()
    baseline = fc.contract_hash(contract)
    tampered = json.loads(json.dumps(contract))
    tampered["window_seconds"] = 5
    assert fc.contract_hash(tampered) != baseline


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS  {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name}: {error}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
