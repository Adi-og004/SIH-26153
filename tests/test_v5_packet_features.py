"""
V5 packet-feature tests (Task 9, Part 14).

Two kinds of check:

  1. FIXTURE CHECKS — 16 deterministic packet scenarios whose expected values
     were computed by the INDEPENDENT reference arithmetic in
     tests/fixtures/v5_packet_features/_generate_fixtures.py, which never
     imports the module under test. Nothing here is self-referential.

  2. CONTRACT / INVARIANT CHECKS — schema shape, the V4 prefix invariant,
     causality, numerical policy, and the packet-feature parity between the
     two invocation routes.

Run:
    pytest tests/test_v5_packet_features.py
    python tests/test_v5_packet_features.py      # same checks, no pytest needed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import packet_feature_builder as pfb      # noqa: E402
import v5_feature_contract as v5c         # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "v5_packet_features"
TOL = 1e-9


# =============================================================
# FIXTURE LOADING
# =============================================================

def _fixture_names():
    return sorted(
        path.stem for path in FIXTURE_DIR.glob("*.json") if not path.name.startswith("_")
    )


def _load(name):
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


def _packet_frame(fixture) -> pd.DataFrame:
    """Materialise a fixture's packets as the canonical per-packet table.

    ``is_fragment`` is derived here from MF / offset because that derivation
    belongs to the PARSER (which real captures exercise), not to the feature
    builder under test.
    """
    rows = []
    for ordinal, packet in enumerate(fixture["packets"]):
        row = dict(fixture["packet_defaults"])
        row.update(packet)
        row["is_fragment"] = bool(row["ip_mf"]) or int(row["ip_frag_offset"]) > 0
        # the fixture lists packets in capture order, so the list position IS
        # the capture ordinal that breaks timestamp ties
        row["packet_index"] = ordinal
        rows.append(row)
    frame = pd.DataFrame(rows).reindex(columns=pfb.PACKET_COLUMNS)
    return pfb.ensure_packet_order(frame)


def _build(fixture) -> pd.DataFrame:
    return pfb.packet_features_from_packet_table(_packet_frame(fixture))


def _assert_close(actual, expected, label):
    assert abs(float(actual) - float(expected)) <= TOL, (
        f"{label}: got {actual!r}, expected {expected!r} "
        f"(difference {abs(float(actual) - float(expected)):.3e})"
    )


# =============================================================
# 1. FIXTURE CHECKS
# =============================================================

def test_every_fixture_matches_independent_expected_values():
    names = _fixture_names()
    assert len(names) >= 12, f"expected at least the 12 required scenarios, found {len(names)}"

    checked = 0
    for name in names:
        fixture = _load(name)
        built = _build(fixture)
        expected_windows = fixture["expected_windows"]

        assert len(built) == len(expected_windows), (
            f"{name}: builder produced {len(built)} windows, fixture expects "
            f"{len(expected_windows)}"
        )

        for expected in expected_windows:
            row = built[built["window_idx"] == expected["window_idx"]]
            assert len(row) == 1, f"{name}: window {expected['window_idx']} missing"
            row = row.iloc[0]

            assert int(row["packet_count"]) == expected["packet_count"], (
                f"{name} w{expected['window_idx']}: packet_count "
                f"{row['packet_count']} != {expected['packet_count']}"
            )

            for feature, value in expected["features"].items():
                _assert_close(row[feature], value, f"{name} w{expected['window_idx']} {feature}")
                checked += 1

            for feature, value in expected.get("relative", {}).items():
                _assert_close(row[feature], value, f"{name} w{expected['window_idx']} {feature}")
                checked += 1

    assert checked > 100, f"only {checked} values compared — fixtures look too thin"


def test_required_scenarios_are_all_present():
    """Part 14 enumerates twelve mandatory scenarios; none may quietly vanish."""
    names = set(_fixture_names())
    required = {
        "01_ttl_varied", "02_ttl_constant",
        "03_tcp_window_changing", "04_tcp_window_zero", "05_no_tcp",
        "06_fragmented", "07_no_fragmentation",
        "08_tcp_retransmission", "09_tcp_clean",
        "10_zero_payload", "11_mixed_payload", "12_identical_payload",
    }
    assert required <= names, f"missing required fixtures: {sorted(required - names)}"


def test_duplicate_pure_acks_are_not_retransmissions():
    """The single easiest way to make this metric misleading — guarded directly."""
    packets = pd.DataFrame(
        [
            dict(timestamp_abs=t, packet_index=i, ip_version=4, src_ip="10.0.0.1", dst_ip="10.0.0.2",
                 src_port=1000, dst_port=80, protocol=6, ttl=64, ip_df=False, ip_mf=False,
                 ip_frag_offset=0, is_fragment=False, is_tcp=True, tcp_window=500,
                 tcp_seq=42, tcp_syn=False, tcp_fin=False, payload_len=0)
            for i, t in enumerate((0.0, 0.1, 0.2, 0.3, 0.4))
        ]
    ).reindex(columns=pfb.PACKET_COLUMNS)
    annotated = pfb.annotate_packets(packets)
    assert int(annotated["is_retransmission"].sum()) == 0
    assert int(annotated["seq_bearing"].sum()) == 0


def test_non_tcp_packets_never_enter_a_tcp_denominator():
    packets = _packet_frame(_load("05_no_tcp"))
    annotated = pfb.annotate_packets(packets)
    for column in ("is_retransmission", "seq_bearing",
                   "window_change_eligible", "tcp_window_changed"):
        assert not annotated[column].any(), f"{column} fired on a UDP-only capture"


def test_df_flag_is_not_counted_as_fragmentation():
    fixture = _load("07_no_fragmentation")
    built = _build(fixture)
    assert float(built["ip_df_ratio"].iloc[0]) == 1.0
    assert float(built["frag_packet_ratio"].iloc[0]) == 0.0
    assert float(built["log_frag_packet_count"].iloc[0]) == 0.0


# =============================================================
# 2. SCHEMA / CONTRACT SHAPE
# =============================================================

def test_v5_dimension_is_82_and_decomposes_correctly():
    schema = v5c.build_schema()
    assert schema["total_dimension"] == 82
    assert schema["counts"] == {
        "flow_base": 26,
        "flow_relative_change": 26,
        "packet_base": 21,
        "packet_relative_change": 9,
        "total": 82,
    }
    assert len(schema["feature_order"]) == 82
    assert len(set(schema["feature_order"])) == 82


def test_v4_block_is_an_unmodified_prefix():
    """The load-bearing version-boundary invariant."""
    schema = v5c.build_schema()
    frozen = (
        REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_feature_names.txt"
    ).read_text().split()
    assert len(frozen) == 52
    assert schema["feature_order"][:52] == frozen, "the V4 prefix was reordered"
    for feature in schema["features"][:52]:
        assert feature["schema_origin"] == "v4"
        assert feature["v5_modified"] is False
        assert feature["feature_level"] == "flow"


def test_packet_block_occupies_indices_52_to_81():
    schema = v5c.build_schema()
    for feature in schema["features"][52:]:
        assert feature["schema_origin"] == "v5"
        assert feature["feature_level"] == "packet"
    assert [f["name"] for f in schema["features"][52:73]] == pfb.PACKET_BASE_FEATURES
    assert [f["name"] for f in schema["features"][73:]] == pfb.PACKET_RELATIVE_FEATURES
    assert [f["index"] for f in schema["features"]] == list(range(82))


def test_feature_order_is_deterministic_across_rebuilds():
    assert v5c.build_schema()["feature_order"] == v5c.build_schema()["feature_order"]
    assert v5c.schema_hash() == v5c.schema_hash()


def test_every_packet_feature_has_a_complete_contract_entry():
    contract = v5c.build_packet_contract()
    required = (
        "index", "name", "source_field", "aggregation", "unit", "domain",
        "missing_value_rule", "zero_value_behaviour", "log_transformed",
        "normalised", "relative_change_decision", "relative_change_reason",
        "depends_on",
    )
    assert len(contract["features"]) == 30
    for feature in contract["features"]:
        for key in required:
            assert key in feature, f"{feature['name']} missing contract field {key}"
            assert feature[key] not in (None, "", []), (
                f"{feature['name']}.{key} is empty"
            )
        assert "formula" in feature and len(feature["formula"]) > 20


def test_every_new_base_feature_has_an_explicit_relative_change_ruling():
    contract = v5c.build_packet_contract()
    base = [f for f in contract["features"] if f["category"] == "base"]
    assert len(base) == 21
    included = [f["name"] for f in base if f["relative_change_decision"] == "included"]
    excluded = [f["name"] for f in base if f["relative_change_decision"] == "excluded"]
    assert len(included) + len(excluded) == 21, "a feature has no ruling"
    assert included == list(pfb.PACKET_RELATIVE_CHANGE_BASE)
    assert len(included) == 9 and len(excluded) == 12
    for feature in base:
        assert len(feature["relative_change_reason"]) > 40, (
            f"{feature['name']}: relative-change ruling has no substantive reason"
        )


def test_families_partition_the_21_packet_features():
    contract = v5c.build_packet_contract()
    families = contract["families"]
    assert sorted(families) == [
        "fragmentation", "payload", "retransmission", "tcp_window", "ttl"
    ]
    flat = [name for names in families.values() for name in names]
    assert sorted(flat) == sorted(pfb.PACKET_BASE_FEATURES)
    assert len(flat) == len(set(flat)) == 21


# =============================================================
# 3. NUMERICAL POLICY
# =============================================================

def test_no_nan_or_inf_in_any_fixture_output():
    for name in _fixture_names():
        built = _build(_load(name))
        values = built[pfb.PACKET_BASE_FEATURES + pfb.PACKET_RELATIVE_FEATURES].to_numpy(float)
        assert np.isfinite(values).all(), f"{name}: non-finite packet feature emitted"


def test_relative_changes_are_clipped_and_first_row_is_zero():
    for name in _fixture_names():
        built = _build(_load(name))
        rel = built[pfb.PACKET_RELATIVE_FEATURES].to_numpy(float)
        assert (rel >= -pfb.RELATIVE_CLIP - TOL).all()
        assert (rel <= pfb.RELATIVE_CLIP + TOL).all()
        assert np.abs(rel[0]).max() == 0.0, f"{name}: first relative-change row is not exactly 0"


def test_ratio_features_stay_within_zero_and_one():
    bounded = [
        "tcp_window_zero_ratio", "tcp_window_change_rate",
        "frag_packet_ratio", "ip_df_ratio",
        "tcp_retrans_ratio", "payload_zero_ratio",
    ]
    for name in _fixture_names():
        built = _build(_load(name))
        for feature in bounded:
            values = built[feature].to_numpy(float)
            assert (values >= -TOL).all() and (values <= 1.0 + TOL).all(), (
                f"{name}: {feature} left [0, 1]"
            )


def test_empty_capture_produces_an_empty_table_not_a_crash():
    empty = pd.DataFrame(columns=pfb.PACKET_COLUMNS)
    built = pfb.packet_features_from_packet_table(empty)
    assert len(built) == 0
    for feature in pfb.PACKET_BASE_FEATURES + pfb.PACKET_RELATIVE_FEATURES:
        assert feature in built.columns


def test_single_packet_window_has_zero_dispersion_not_nan():
    single = _packet_frame(
        {"packet_defaults": _load("09_tcp_clean")["packet_defaults"],
         "packets": [{"timestamp_abs": 0.0, "payload_len": 100, "tcp_seq": 5}]}
    )
    built = pfb.packet_features_from_packet_table(single)
    assert len(built) == 1
    row = built.iloc[0]
    assert float(row["ttl_std"]) == 0.0
    assert float(row["log_tcp_window_std"]) == 0.0
    # one TCP packet = the first of its session, so it is ineligible for both
    # the change rate and the retransmission test
    assert float(row["tcp_window_change_rate"]) == 0.0
    assert float(row["log_tcp_retrans_count"]) == 0.0


# =============================================================
# 4. CAUSALITY / LEAKAGE
# =============================================================

def test_truncating_the_future_never_changes_an_earlier_window():
    """The decisive leakage test: rebuild from a prefix of the capture and
    require every already-completed window to be bit-identical."""
    for name in ("13_multi_window_relative_change", "15_window_boundary_causality"):
        fixture = _load(name)
        packets = _packet_frame(fixture)
        full = pfb.packet_features_from_packet_table(packets)

        win0 = int(np.floor(float(packets["timestamp_abs"].min()) / pfb.WINDOW_SECONDS))
        cutoff = float(full["window_idx"].max()) * pfb.WINDOW_SECONDS + win0 * pfb.WINDOW_SECONDS
        prefix = packets[packets["timestamp_abs"] < cutoff]
        truncated = pfb.packet_features_from_packet_table(prefix, win0=win0)

        columns = pfb.PACKET_BASE_FEATURES + pfb.PACKET_RELATIVE_FEATURES
        for window_idx in truncated["window_idx"]:
            a = truncated[truncated["window_idx"] == window_idx][columns].to_numpy(float)
            b = full[full["window_idx"] == window_idx][columns].to_numpy(float)
            assert np.allclose(a, b, rtol=0, atol=0), (
                f"{name}: window {window_idx} changed when later packets were added — "
                f"future information is leaking backwards"
            )


def test_annotations_depend_only_on_the_past():
    """Every annotated packet must be reproducible from the prefix of the
    capture up to and including that packet."""
    packets = _packet_frame(_load("08_tcp_retransmission"))
    full = pfb.annotate_packets(packets)
    for i in range(1, len(packets) + 1):
        partial = pfb.annotate_packets(packets.iloc[:i].reset_index(drop=True))
        for column in ("is_retransmission", "retrans_bytes", "seq_bearing",
                       "window_change_eligible", "tcp_window_changed"):
            assert list(partial[column]) == list(full[column].iloc[:i]), (
                f"annotation {column} for packet {i - 1} depends on a later packet"
            )


def test_window_assignment_is_left_closed_right_open():
    base = _load("09_tcp_clean")["packet_defaults"]
    packets = _packet_frame({
        "packet_defaults": base,
        "packets": [
            {"timestamp_abs": 9.999999, "payload_len": 10, "tcp_seq": 1},
            {"timestamp_abs": 10.0, "payload_len": 10, "tcp_seq": 100},
        ],
    })
    built = pfb.build_packet_windows(pfb.annotate_packets(packets))
    assert list(built["window_idx"]) == [0, 1], "the [t, t+10) boundary rule drifted"
    assert list(built["packet_count"]) == [1, 1]


# =============================================================
# 5. PACKET-FEATURE PARITY BETWEEN THE TWO INVOCATION ROUTES
# =============================================================

def test_training_route_and_pcap_route_agree_exactly():
    """Part 15. The two paths share one implementation, so this test's job is
    to catch DRIFT introduced by the plumbing around it — serialisation,
    dtypes, row order — not to re-prove the formulas."""
    columns = pfb.PACKET_BASE_FEATURES + pfb.PACKET_RELATIVE_FEATURES
    for name in _fixture_names():
        packets = _packet_frame(_load(name))

        # route A ("pcap side"): in-memory table straight from the parser
        route_a = pfb.packet_features_from_packet_table(packets)

        # route B ("training side"): the same packets after a CSV round-trip
        # and a row shuffle, as they would arrive from storage
        buffer = packets.sample(frac=1.0, random_state=0).to_csv(index=False)
        from io import StringIO
        reloaded = pd.read_csv(StringIO(buffer))
        for flag in ("ip_df", "ip_mf", "is_fragment", "is_tcp", "tcp_syn", "tcp_fin"):
            reloaded[flag] = reloaded[flag].astype(bool)
        route_b = pfb.packet_features_from_packet_table(reloaded)

        assert list(route_a["window_idx"]) == list(route_b["window_idx"]), name
        a = route_a[columns].to_numpy(float)
        b = route_b[columns].to_numpy(float)
        assert np.array_equal(a, b), f"{name}: the two invocation routes diverged"


# =============================================================
# 6. V4 MUST NOT BE TOUCHED
# =============================================================

def test_v4_contract_and_feature_order_are_untouched():
    import feature_contract as v4c
    assert len(v4c.INPUT_FEATURE_NAMES) == 52
    assert v4c.BASE_FEATURES == v4c.STATE_FEATURE_NAMES
    assert v4c.build_contract()["feature_dimension"] == 52
    assert v5c.V5_FEATURE_NAMES[:52] == v4c.INPUT_FEATURE_NAMES


def test_no_packet_feature_name_collides_with_a_v4_name():
    import feature_contract as v4c
    v4_names = set(v4c.INPUT_FEATURE_NAMES)
    new_names = set(pfb.PACKET_BASE_FEATURES) | set(pfb.PACKET_RELATIVE_FEATURES)
    assert not (v4_names & new_names), f"name collision: {sorted(v4_names & new_names)}"


# =============================================================
# STANDALONE RUNNER
# =============================================================

def _main():
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"  FAIL  {name}")
            print(f"        {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}")
            print(f"        {type(exc).__name__}: {exc}")
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
