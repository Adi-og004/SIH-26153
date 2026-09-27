"""
=============================================================
INPUT-MODE IDENTITY TESTS — all sources feed one V4 pipeline
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

The point of the unified data-input layer is that a sequence produces the
SAME forecast no matter how it entered the console. These tests prove it
end to end, on real sequences from the frozen benchmark:

    Historical Benchmark  ==  NPY upload  ==  CSV upload

They also lock the artifact-identification rules, so a target-state array,
a ground-truth label array, a metadata table or a stored-predictions table
can never be mistaken for model input.

Nothing here retrains, rescales or re-thresholds anything: every path runs
the frozen checkpoint through dinp.run_inference().
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import data_input as dinp                      # noqa: E402
import prepared_input_loader as pil            # noqa: E402

BENCH = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
SCREEN = REPO_ROOT / "results" / "v4_screening_mvp"

# The tolerance the rest of the project already uses for a forecast that
# must be "the same" across devices / code paths (Tasks 12 / 15 / 18 / 19).
TOLERANCE = 1e-5

# Deterministic, label-free choice of probe sequences: a fixed stride over
# the frozen test tensor. No sequence is chosen by its outcome.
PROBE_IDS = [0, 7, 100, 1000, 2184, 3114]


# =============================================================
# FIXTURES
# =============================================================

@pytest.fixture(scope="module")
def X_test():
    path = BENCH / "X_test.npy"
    if not path.exists():
        pytest.skip(f"frozen benchmark tensor not available: {path}")
    return np.load(path)


@pytest.fixture(scope="module")
def explainer():
    torch = pytest.importorskip("torch")          # noqa: F841
    try:
        from v4_explainer import V4Explainer
        return V4Explainer(device="cpu")
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"frozen V4 explainer unavailable: {exc}")


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _csv_bytes(seq: np.ndarray, with_timestamp: bool = False) -> bytes:
    """A CSV representation of one (10, 52) sequence.

    The benchmark tensor is ALREADY in frozen-scaler space, and the CSV
    loader applies no scaling, so writing the values under their canonical
    column names is a faithful round trip. '%.17g' is written so float32
    values survive the text round trip exactly.
    """
    assert seq.shape == (dinp.SEQUENCE_LENGTH, dinp.FEATURE_DIMENSION)
    df = pd.DataFrame(seq, columns=pil.canonical_feature_names())
    if with_timestamp:
        df.insert(0, "timestamp",
                  pd.date_range("2018-03-02 00:00:00",
                                periods=len(df), freq="10s"))
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.17g")
    return buf.getvalue().encode()


# =============================================================
# 1. THE IDENTITY TEST
# =============================================================

@pytest.mark.parametrize("seq_id", PROBE_IDS)
def test_benchmark_npy_csv_predictions_identical(X_test, explainer, seq_id):
    """Benchmark == NPY upload == CSV upload, for the same sequence."""
    seq = X_test[seq_id]

    # --- 1. Historical Benchmark ---
    p_bench = dinp.from_benchmark(X_test[seq_id:seq_id + 1],
                                  source_name="X_test.npy")
    assert p_bench.is_valid
    traj_bench = dinp.run_inference(p_bench, explainer, 5)

    # --- 2. NPY upload (a (10, 52) single sequence) ---
    p_npy = dinp.from_upload("probe.npy", _npy_bytes(seq))
    assert p_npy.is_valid, p_npy.errors
    assert p_npy.n_sequences == 1
    traj_npy = dinp.run_inference(p_npy, explainer, 5)

    # --- 3. CSV upload (the same 10x52 sequence as 10 rows) ---
    p_csv = dinp.from_upload("probe.csv", _csv_bytes(seq))
    assert p_csv.is_valid, p_csv.errors
    assert p_csv.n_sequences == 1
    traj_csv = dinp.run_inference(p_csv, explainer, 5)

    assert traj_bench.shape == traj_npy.shape == traj_csv.shape == (1, 6)

    d_npy = float(np.abs(traj_bench - traj_npy).max())
    d_csv = float(np.abs(traj_bench - traj_csv).max())
    assert d_npy <= TOLERANCE, f"benchmark vs NPY differ by {d_npy:.3e}"
    assert d_csv <= TOLERANCE, f"benchmark vs CSV differ by {d_csv:.3e}"


def test_csv_with_timestamp_matches_benchmark(X_test, explainer):
    """A timestamped CSV takes the contiguity path and still matches."""
    seq_id = PROBE_IDS[0]
    seq = X_test[seq_id]
    p_bench = dinp.from_benchmark(X_test[seq_id:seq_id + 1])
    p_csv = dinp.from_upload("probe_ts.csv", _csv_bytes(seq, True))
    assert p_csv.is_valid, p_csv.errors
    d = float(np.abs(dinp.run_inference(p_bench, explainer, 5)
                     - dinp.run_inference(p_csv, explainer, 5)).max())
    assert d <= TOLERANCE, f"timestamped CSV differs by {d:.3e}"


def test_multi_sequence_npy_matches_benchmark(X_test, explainer):
    """A batched (N, 10, 52) upload matches the benchmark batch."""
    block = X_test[100:140]
    p_bench = dinp.from_benchmark(block)
    p_npy = dinp.from_upload("block.npy", _npy_bytes(block))
    assert p_npy.is_valid, p_npy.errors
    assert p_npy.n_sequences == len(block)
    d = float(np.abs(dinp.run_inference(p_bench, explainer, 5)
                     - dinp.run_inference(p_npy, explainer, 5)).max())
    assert d <= TOLERANCE, f"batched NPY differs by {d:.3e}"


def test_full_x_test_upload_is_accepted(X_test, explainer):
    """Uploading X_test.npy itself reproduces the benchmark exactly."""
    p = dinp.from_upload("X_test.npy", _npy_bytes(X_test[:64]))
    assert p.is_valid, p.errors
    assert p.identification.role == dinp.ROLE_MODEL_INPUT
    d = float(np.abs(
        dinp.run_inference(dinp.from_benchmark(X_test[:64]), explainer, 5)
        - dinp.run_inference(p, explainer, 5)).max())
    assert d <= TOLERANCE


# =============================================================
# 2. REJECTION OF PROJECT ARTIFACTS THAT ARE NOT MODEL INPUT
# =============================================================

def _require(path: Path):
    if not path.exists():
        pytest.skip(f"artifact not available: {path}")
    return path


def test_y_multistep_is_rejected_as_target_states():
    path = _require(BENCH / "Y_multistep_test.npy")
    p = dinp.from_upload("Y_multistep_test.npy", path.read_bytes())
    assert not p.is_valid
    assert p.identification.role == dinp.ROLE_TARGET_STATES
    assert dinp.MSG_TARGET_STATES in p.errors
    # The required wording, verbatim.
    assert "future target states, not V4 model input" in dinp.MSG_TARGET_STATES
    assert "(N,10,52)" in dinp.MSG_TARGET_STATES


def test_a_binary_is_rejected_as_ground_truth_labels():
    path = _require(BENCH / "A_binary_multistep_test.npy")
    p = dinp.from_upload("A_binary_multistep_test.npy", path.read_bytes())
    assert not p.is_valid
    assert p.identification.role == dinp.ROLE_BINARY_LABELS
    assert any("ground-truth attack labels" in e for e in p.errors)


def test_a_soft_is_rejected_as_soft_labels():
    path = _require(BENCH / "A_soft_multistep_test.npy")
    p = dinp.from_upload("A_soft_multistep_test.npy", path.read_bytes())
    assert not p.is_valid
    assert p.identification.role == dinp.ROLE_SOFT_LABELS


def test_test_metadata_is_rejected_as_metadata():
    path = _require(BENCH / "test_metadata.csv")
    p = dinp.from_upload("test_metadata.csv", path.read_bytes())
    assert not p.is_valid
    assert p.identification.role == dinp.ROLE_METADATA
    assert any("metadata" in e.lower() for e in p.errors)


def test_screening_predictions_is_rejected_as_predictions():
    path = _require(SCREEN / "screening_predictions.csv")
    p = dinp.from_upload("screening_predictions.csv", path.read_bytes())
    assert not p.is_valid
    assert p.identification.role == dinp.ROLE_PREDICTIONS
    assert any("predictions" in e.lower() for e in p.errors)


def test_event_timeline_is_rejected_as_annotations():
    path = _require(SCREEN / "event_timeline.csv")
    p = dinp.from_upload("event_timeline.csv", path.read_bytes())
    assert not p.is_valid
    assert p.identification.role == dinp.ROLE_ANNOTATIONS


def test_screening_x_is_accepted_as_model_input():
    path = _require(SCREEN / "screening_X.npy")
    arr = np.load(path)[:8]
    p = dinp.from_upload("screening_X.npy", _npy_bytes(arr))
    assert p.is_valid, p.errors
    assert p.identification.role == dinp.ROLE_MODEL_INPUT


# =============================================================
# 3. IDENTIFICATION IS BY CONTENT, NOT BY NAME
# =============================================================

def test_renamed_target_states_are_still_rejected():
    """Calling a target-state array X_test.npy does not make it input."""
    ident = dinp.identify_npy("X_test.npy", np.zeros((7, 5, 26)))
    assert ident.role == dinp.ROLE_TARGET_STATES
    assert not ident.is_model_input
    assert "the content was used, not the name" in ident.detail.lower()


def test_renamed_model_input_is_still_accepted():
    """A valid tensor under any name is accepted."""
    ident = dinp.identify_npy("Y_multistep_test.npy",
                              np.zeros((3, 10, 52)))
    assert ident.role == dinp.ROLE_MODEL_INPUT
    assert ident.is_model_input


def test_unknown_shape_explains_the_expected_shape():
    ident = dinp.identify_npy("mystery.npy", np.zeros((3, 9, 52)))
    assert ident.role == dinp.ROLE_UNKNOWN
    assert "(N,10,52)" in ident.explanation


# =============================================================
# 4. THE COMMON REPRESENTATION AND THE SINGLE INFERENCE CALL
# =============================================================

def test_every_source_produces_the_same_container(X_test):
    a = dinp.from_benchmark(X_test[:2])
    b = dinp.from_upload("x.npy", _npy_bytes(X_test[:2]))
    c = dinp.from_upload("x.csv", _csv_bytes(X_test[0]))
    d = dinp.from_pcap(X_test[:2], source_name="capture.pcap")
    for p in (a, b, c, d):
        assert isinstance(p, dinp.PreparedInput)
        assert p.is_valid, p.errors
        assert p.x.ndim == 3
        assert p.x.shape[1:] == (dinp.SEQUENCE_LENGTH, dinp.FEATURE_DIMENSION)
        assert p.x.dtype == np.float32
        assert p.source_type in dinp.SOURCE_TYPES
        assert p.source_name
        assert p.badge == dinp.VALID_BADGE
    assert {a.source_type, b.source_type, c.source_type, d.source_type} == {
        dinp.SRC_BENCHMARK, dinp.SRC_NPY, dinp.SRC_CSV, dinp.SRC_PCAP}


def test_run_inference_refuses_invalid_input(explainer):
    p = dinp.from_upload("Y.npy", _npy_bytes(np.zeros((4, 5, 26))))
    assert not p.is_valid
    with pytest.raises(ValueError):
        dinp.run_inference(p, explainer, 5)


def test_run_inference_delegates_to_the_frozen_rollout(X_test, explainer):
    """No second implementation: the shared call equals the explainer's."""
    block = X_test[:16]
    direct = explainer.predict_rollout(block.astype(np.float32), K=5)
    shared = dinp.run_inference(dinp.from_benchmark(block), explainer, 5)
    assert np.array_equal(direct, shared)


def test_validation_card_shape():
    p = dinp.from_benchmark(np.zeros((3, 10, 52), dtype=np.float32))
    rows = dict(p.card_rows())
    assert rows["Valid sequences"] == "3"
    assert rows["Rejected sequences"] == "0"
    assert rows["Status"] == dinp.VALID_BADGE


def test_layer_declares_no_model_mutation():
    g = dinp.layer_metadata()["guarantees"]
    assert g["retrains_v4"] is False
    assert g["modifies_v4_checkpoint"] is False
    assert g["modifies_scaler"] is False
    assert g["modifies_thresholds"] is False
    assert g["modifies_feature_order"] is False
    assert g["duplicate_inference_implementation"] is False
