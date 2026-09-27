"""
=============================================================
SUBMISSION REGRESSION TESTS — examples, PCAP path, frozen artifacts
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

These tests lock the behaviour a judge relies on from a fresh clone:

  1. The shipped example inputs (examples/prepared_v4/) validate, and the
     frozen V4 model scores them exactly as the stored, frozen screening
     predictions (results/v4_screening_mvp/screening_predictions.csv) say.
  2. The canonical PCAP pipeline (src/pcap_ingest_v4.py) still turns the
     shipped synthetic demo captures (examples/pcap/) into the same
     (N, 10, 52) tensors and the same probabilities as when the reference
     fixtures in tests/fixtures/pcap_regression/ were recorded.
  3. A PCAP-derived tensor, saved as .npy and re-uploaded as prepared data,
     yields bit-identical probabilities -- PCAP input and prepared input
     reach the same frozen inference call (data_input.run_inference).
  4. The frozen V4 artifacts still hash-match docs/v4_baseline_manifest.json.

The PCAP reference fixtures were generated from the unmodified pipeline and
checkpoint with the same calls this file makes (see _run_pcap_pipeline).
Nothing here trains, rescales or re-thresholds anything.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import data_input as dinp                      # noqa: E402
import prepared_input_loader as pil            # noqa: E402

EXAMPLES = REPO_ROOT / "examples"
SAMPLE_NPY = EXAMPLES / "prepared_v4" / "sample_input.npy"
SAMPLE_CSV = EXAMPLES / "prepared_v4" / "sample_input.csv"
PCAP_DIR = EXAMPLES / "pcap"
PCAP_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "pcap_regression"
SCREENING = REPO_ROOT / "results" / "v4_screening_mvp"
MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"

# Same tolerance as tests/test_input_mode_identity.py.
TOLERANCE = 1e-5

# The screening sequence ids the example NPY was extracted from, in order.
SAMPLE_SEQUENCE_IDS = [0, 100, 500, 1000, 2000]

# The frozen K=5 trajectory-MAX alert boundary the dashboard uses
# (docs/v4_baseline_manifest.json threshold.per_horizon_trajectory_max['5']).
ALERT_BOUNDARY = 0.69

PROB_COLUMNS = ["nowcast_probability_t0"] + [f"prob_t{k}" for k in range(1, 6)]


@pytest.fixture(scope="module")
def explainer():
    from v4_explainer import V4Explainer
    return V4Explainer(device="cpu")


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================
# 1. FROZEN ARTIFACTS ARE PRESENT AND UNMODIFIED
# =============================================================

def test_frozen_v4_artifacts_match_manifest():
    m = json.loads(MANIFEST.read_text())
    pinned = {
        m["checkpoint"]["path"]: m["checkpoint"]["sha256"],
        m["scalers"]["input_scaler"]["path"]: m["scalers"]["input_scaler"]["sha256"],
        m["scalers"]["target_scaler"]["path"]: m["scalers"]["target_scaler"]["sha256"],
        m["input"]["feature_order_file"]: m["input"]["feature_order_sha256"],
        m["input"]["state_feature_file"]: m["input"]["state_feature_sha256"],
    }
    for rel, expected in pinned.items():
        path = REPO_ROOT / rel
        assert path.exists(), f"frozen artifact missing: {rel}"
        assert _sha256(path) == expected, f"frozen artifact modified: {rel}"


def test_benchmark_test_tensor_is_the_screening_tensor():
    """X_test.npy (dashboard start-up) and screening_X.npy (replay substrate)
    must be the same bytes, as src/screening_data.py documents."""
    a = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "X_test.npy"
    b = SCREENING / "screening_X.npy"
    assert _sha256(a) == _sha256(b)


# =============================================================
# 2. SHIPPED EXAMPLE INPUTS
# =============================================================

def test_example_npy_is_valid_v4_input():
    p = dinp.from_upload(SAMPLE_NPY.name, SAMPLE_NPY.read_bytes())
    assert p.is_valid, p.errors
    assert p.x.shape == (5, 10, 52)
    assert p.x.dtype == np.float32


def test_example_npy_is_unmodified_benchmark_data():
    x = np.load(SAMPLE_NPY)
    screening_x = np.load(SCREENING / "screening_X.npy", mmap_mode="r")
    assert np.array_equal(x, screening_x[SAMPLE_SEQUENCE_IDS])


def test_example_csv_is_valid_and_equals_first_npy_sequence():
    p = dinp.from_upload(SAMPLE_CSV.name, SAMPLE_CSV.read_bytes())
    assert p.is_valid, p.errors
    assert p.x.shape == (1, 10, 52)
    assert np.array_equal(p.x[0], np.load(SAMPLE_NPY)[0])


def test_example_csv_header_is_the_canonical_feature_order():
    cols = list(pd.read_csv(SAMPLE_CSV, nrows=0).columns)
    assert cols == pil.canonical_feature_names()


def test_example_npy_scores_match_frozen_screening_predictions(explainer):
    """Upload path on the example == stored frozen V4 screening output."""
    p = dinp.from_upload(SAMPLE_NPY.name, SAMPLE_NPY.read_bytes())
    probs = dinp.run_inference(p, explainer, 5)                 # (5, 6)

    stored = pd.read_csv(SCREENING / "screening_predictions.csv")
    stored = stored.set_index("sequence_id").loc[SAMPLE_SEQUENCE_IDS]
    ref = stored[PROB_COLUMNS].to_numpy(dtype=np.float64)

    assert np.max(np.abs(probs - ref)) < TOLERANCE
    alerts = (probs[:, 1:].max(axis=1) >= ALERT_BOUNDARY).astype(int)
    assert alerts.tolist() == stored["alert"].astype(int).tolist()


# =============================================================
# 3. PCAP PATH REGRESSION + PCAP/PREPARED IDENTITY
# =============================================================

def _run_pcap_pipeline(pcap_path: Path):
    """The exact call sequence src/app.py uses for an uploaded PCAP."""
    import pcap_ingest_v4 as pv
    packets = pv.parse_pcap_streaming(str(pcap_path))
    flows = pv.reconstruct_flows(packets)
    windows = pv.add_relative_changes(pv.compute_window_features(flows, packets))
    seq = pv.build_sequences(windows, pv.load_scaler(pv.INPUT_SCALER_PATH))
    return packets, flows, windows, seq


PCAP_CASES = ["demo_portscan", "demo_c2_beacon"]


@pytest.mark.parametrize("name", PCAP_CASES)
def test_pcap_pipeline_regression(name, explainer):
    expected = json.loads((PCAP_FIXTURES / "expected_counts.json").read_text())[name]
    packets, flows, windows, seq = _run_pcap_pipeline(PCAP_DIR / f"{name}.pcap")

    assert len(packets) == expected["n_packets"]
    assert len(flows) == expected["n_flows"]
    assert len(windows) == expected["n_windows"]
    assert int(seq["n_rejected_gap"]) == expected["n_rejected_gap"]
    assert list(seq["X"].shape) == expected["X_shape"]

    ref_x = np.load(PCAP_FIXTURES / f"{name}_X.npy")
    np.testing.assert_allclose(seq["X"], ref_x, rtol=0, atol=TOLERANCE)

    prepared = dinp.from_pcap(seq["X"], timestamps=list(seq["seq_ts_rel"]),
                              source_name=f"{name}.pcap")
    assert prepared.is_valid
    probs = dinp.run_inference(prepared, explainer, 5)
    ref_p = np.load(PCAP_FIXTURES / f"{name}_probs.npy")
    np.testing.assert_allclose(probs, ref_p, rtol=0, atol=TOLERANCE)


@pytest.mark.parametrize("name", PCAP_CASES)
def test_pcap_and_prepared_upload_reach_the_same_inference(name, explainer):
    """PCAP-derived tensor re-uploaded as prepared NPY gives identical output."""
    _, _, _, seq = _run_pcap_pipeline(PCAP_DIR / f"{name}.pcap")
    via_pcap = dinp.run_inference(
        dinp.from_pcap(seq["X"], source_name=f"{name}.pcap"), explainer, 5)
    upload = dinp.from_upload(f"{name}.npy", _npy_bytes(seq["X"]))
    assert upload.is_valid, upload.errors
    via_upload = dinp.run_inference(upload, explainer, 5)
    assert np.array_equal(via_pcap, via_upload)
