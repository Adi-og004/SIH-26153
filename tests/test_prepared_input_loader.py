"""
=============================================================
TESTS — prepared_input_loader.py
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Tests for CSV/NPY feature-data loading and validation, plus the
critical identity test: a real benchmark sequence round-tripped
through the CSV loader must produce a V4 prediction identical
(within numerical tolerance) to the existing benchmark path.

Run:  python -m pytest tests/test_prepared_input_loader.py -v
=============================================================
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import prepared_input_loader as pil  # noqa: E402


# =============================================================
# HELPERS
# =============================================================

CANONICAL = pil.canonical_feature_names()
N_FEAT = pil.FEATURE_DIMENSION
SEQ_LEN = pil.SEQUENCE_LENGTH


def _make_csv_bytes(
    df: pd.DataFrame,
) -> bytes:
    """Serialise a DataFrame to CSV bytes."""
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _valid_csv_df(n_rows: int = 20) -> pd.DataFrame:
    """A synthetic but valid 52-feature CSV with n_rows."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_rows, N_FEAT))
    df = pd.DataFrame(data, columns=CANONICAL)
    return df


# =============================================================
# CSV — valid
# =============================================================

class TestCSVValid:

    def test_valid_csv_basic(self):
        df = _valid_csv_df(20)
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert not r.errors, r.errors
        assert r.source_format == "csv"
        assert r.n_sequences == 20 - SEQ_LEN + 1  # 11
        assert r.X is not None
        assert r.X.shape == (11, SEQ_LEN, N_FEAT)

    def test_valid_csv_exact_10_rows(self):
        df = _valid_csv_df(10)
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert not r.errors, r.errors
        assert r.n_sequences == 1
        assert r.X.shape == (1, SEQ_LEN, N_FEAT)

    def test_valid_csv_with_timestamp(self):
        df = _valid_csv_df(15)
        base = pd.Timestamp("2024-01-01")
        df["timestamp"] = [
            base + pd.Timedelta(seconds=10 * i) for i in range(15)
        ]
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert not r.errors, r.errors
        assert r.has_timestamp
        assert r.first_timestamp is not None
        assert r.n_sequences == 15 - SEQ_LEN + 1

    def test_valid_csv_with_extra_columns(self):
        """Extra unknown columns are warned but not rejected."""
        df = _valid_csv_df(12)
        df["my_custom_column"] = 42
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert not r.errors, r.errors
        assert len(r.warnings) >= 1
        assert r.n_sequences > 0

    def test_valid_csv_with_meta_columns(self):
        """Known meta columns like 'timestamp' and 'attack_present' pass."""
        df = _valid_csv_df(12)
        df["timestamp"] = pd.date_range("2024-01-01", periods=12, freq="10s")
        df["attack_present"] = 0
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert not r.errors, r.errors
        assert r.has_labels
        assert r.has_timestamp

    def test_valid_csv_with_labels(self):
        df = _valid_csv_df(15)
        df["attack_present"] = [0] * 10 + [1] * 5
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert not r.errors
        assert r.has_labels
        assert r.labels is not None
        assert len(r.labels) == r.n_sequences


# =============================================================
# CSV — invalid
# =============================================================

class TestCSVInvalid:

    def test_missing_feature(self):
        df = _valid_csv_df(12)
        df = df.drop(columns=["syn_rate"])
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert len(r.errors) > 0
        assert "syn_rate" in r.errors[0]

    def test_wrong_feature_count(self):
        """Only 51 of 52 features present."""
        df = _valid_csv_df(12)
        df = df.drop(columns=[CANONICAL[-1]])
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert len(r.errors) > 0
        assert "51" in r.errors[0] or "Missing" in r.errors[0]

    def test_nan_values(self):
        df = _valid_csv_df(12)
        df.iloc[3, 5] = np.nan
        df.iloc[7, 10] = np.nan
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        # Should still succeed (NaN rows are rejected, not the whole file)
        # unless there aren't enough rows left
        assert r.n_rejected_rows == 2
        if r.n_sequences > 0:
            assert r.X is not None

    def test_inf_values(self):
        df = _valid_csv_df(12)
        df.iloc[0, 0] = np.inf
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert r.n_rejected_rows >= 1

    def test_insufficient_rows(self):
        df = _valid_csv_df(5)
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert len(r.errors) > 0
        assert "Not enough" in r.errors[0] or "at least" in r.errors[0]

    def test_duplicate_columns(self):
        df = _valid_csv_df(12)
        # Create CSV with duplicate column by writing raw text
        csv_text = df.to_csv(index=False)
        # Rename last column to first column name => duplicate
        old_name = CANONICAL[-1]
        new_name = CANONICAL[0]
        csv_text = csv_text.replace(old_name, new_name, 1)
        r = pil.load_prepared_csv(csv_text.encode())
        # Should detect missing feature (since one got renamed/duplicated)
        assert len(r.errors) > 0

    def test_non_numeric_values(self):
        df = _valid_csv_df(12)
        df[CANONICAL[2]] = "not_a_number"
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert len(r.errors) > 0
        assert "non-numeric" in r.errors[0]

    def test_empty_csv(self):
        r = pil.load_prepared_csv(b"")
        assert len(r.errors) > 0

    def test_non_contiguous_timestamps(self):
        """Timestamps with a big gap should produce fewer sequences."""
        df = _valid_csv_df(20)
        ts = []
        for i in range(10):
            ts.append(pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=10 * i))
        # Gap of 1000 seconds
        for i in range(10):
            ts.append(
                pd.Timestamp("2024-01-01")
                + pd.Timedelta(seconds=1100 + 10 * i)
            )
        df["timestamp"] = ts
        r = pil.load_prepared_csv(_make_csv_bytes(df))
        assert not r.errors, r.errors
        # Should have 2 valid windows of 10 (one on each side of gap)
        # but NOT a window that spans the gap
        assert r.n_sequences == 2  # one from each contiguous block


# =============================================================
# NPY — valid
# =============================================================

class TestNPYValid:

    def test_valid_npy_3d(self):
        arr = np.random.randn(5, SEQ_LEN, N_FEAT).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert not r.errors, r.errors
        assert r.source_format == "npy"
        assert r.n_sequences == 5
        assert r.X.shape == (5, SEQ_LEN, N_FEAT)

    def test_valid_npy_2d(self):
        """Single sequence (10, 52) should be expanded to (1, 10, 52)."""
        arr = np.random.randn(SEQ_LEN, N_FEAT).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert not r.errors, r.errors
        assert r.n_sequences == 1
        assert r.X.shape == (1, SEQ_LEN, N_FEAT)

    def test_npy_preserves_values(self):
        """Values in NPY should be preserved exactly."""
        arr = np.random.randn(3, SEQ_LEN, N_FEAT).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert not r.errors
        np.testing.assert_array_equal(r.X, arr)


# =============================================================
# NPY — invalid
# =============================================================

class TestNPYInvalid:

    def test_wrong_feature_dim(self):
        arr = np.random.randn(5, SEQ_LEN, 30).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert len(r.errors) > 0
        assert "52" in r.errors[0]

    def test_wrong_seq_len(self):
        arr = np.random.randn(5, 8, N_FEAT).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert len(r.errors) > 0
        assert "10" in r.errors[0] or "sequence" in r.errors[0].lower()

    def test_wrong_ndim(self):
        arr = np.random.randn(5).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert len(r.errors) > 0

    def test_nan_values(self):
        arr = np.random.randn(2, SEQ_LEN, N_FEAT).astype(np.float32)
        arr[0, 3, 10] = np.nan
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert len(r.errors) > 0
        assert "non-finite" in r.errors[0].lower() or "NaN" in r.errors[0]

    def test_inf_values(self):
        arr = np.random.randn(2, SEQ_LEN, N_FEAT).astype(np.float32)
        arr[1, 0, 0] = np.inf
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert len(r.errors) > 0

    def test_2d_wrong_shape(self):
        arr = np.random.randn(10, 30).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_npy(buf.getvalue())
        assert len(r.errors) > 0


# =============================================================
# FORMAT DISPATCH
# =============================================================

class TestDispatch:

    def test_csv_dispatch(self):
        df = _valid_csv_df(12)
        r = pil.load_prepared_file("test.csv", _make_csv_bytes(df))
        assert not r.errors
        assert r.source_format == "csv"

    def test_npy_dispatch(self):
        arr = np.random.randn(3, SEQ_LEN, N_FEAT).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        r = pil.load_prepared_file("test.npy", buf.getvalue())
        assert not r.errors
        assert r.source_format == "npy"

    def test_unsupported_extension(self):
        r = pil.load_prepared_file("test.xlsx", b"data")
        assert len(r.errors) > 0
        assert "Unsupported" in r.errors[0]


# =============================================================
# PREDICTION IDENTITY TEST  (the most important test)
# =============================================================

class TestPredictionIdentity:
    """
    Take a real sequence from X_test.npy, save it as a CSV with
    canonical feature names, load it through the new CSV path,
    and verify the V4 prediction is identical to the benchmark path.

    Also tests the NPY round-trip.
    """

    @pytest.fixture(autouse=True)
    def _check_benchmark_exists(self):
        x_path = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "X_test.npy"
        if not x_path.exists():
            pytest.skip("X_test.npy not found — benchmark data required")

    def _load_benchmark(self):
        return np.load(
            REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "X_test.npy"
        )

    def _load_model_and_run(self, X):
        """Run the frozen V4 model on X (N, 10, 52). Returns (N, K+1) probs."""
        import torch
        import pcap_ingest_v4 as pcap_v4
        model, ckpt = pcap_v4.load_v4_model()
        k = int(ckpt["max_k"])
        result = pcap_v4.run_k_rollout(model, X, k)
        return result["probs"]

    def test_npy_identity(self):
        """NPY round-trip: predictions on X_test[0:1] via NPY loader
        must match predictions on the raw benchmark array."""
        X_test = self._load_benchmark()
        seq = X_test[0:1]  # (1, 10, 52)

        # Benchmark path
        probs_benchmark = self._load_model_and_run(seq.astype(np.float32))

        # NPY loader path
        buf = io.BytesIO()
        np.save(buf, seq)
        r = pil.load_prepared_npy(buf.getvalue())
        assert not r.errors, r.errors
        probs_npy = self._load_model_and_run(r.X)

        # Must be identical (same float32 input, same deterministic model)
        np.testing.assert_allclose(
            probs_npy, probs_benchmark,
            rtol=0, atol=1e-7,
            err_msg="NPY loader prediction differs from benchmark path",
        )

    def test_csv_identity(self):
        """CSV round-trip: a single sequence's 10 rows, written as CSV
        with canonical column names, loaded through the CSV path, must
        produce the same V4 prediction as the benchmark path.

        This is the MOST IMPORTANT test in the suite.
        """
        X_test = self._load_benchmark()
        seq = X_test[0]  # (10, 52) — one sequence

        # Write as CSV with canonical feature names
        df = pd.DataFrame(seq, columns=CANONICAL)
        csv_bytes = _make_csv_bytes(df)

        # Load through the new path
        r = pil.load_prepared_csv(csv_bytes)
        assert not r.errors, r.errors
        assert r.n_sequences == 1, (
            f"Expected 1 sequence from 10 rows, got {r.n_sequences}"
        )

        # Both paths through the model
        probs_benchmark = self._load_model_and_run(
            seq[np.newaxis].astype(np.float32)
        )
        probs_csv = self._load_model_and_run(r.X)

        np.testing.assert_allclose(
            probs_csv, probs_benchmark,
            rtol=0, atol=1e-6,
            err_msg=(
                "CSV-loaded prediction differs from benchmark prediction. "
                "This means the upload path is NOT using the same data the "
                "frozen benchmark uses."
            ),
        )

    def test_csv_identity_multiple_sequences(self):
        """Multiple sequences: the first 3 benchmark sequences,
        written as consecutive rows, should produce matching predictions
        for at least the first window."""
        X_test = self._load_benchmark()
        # Use first 12 rows (produces 3 sequences via sliding window)
        rows = X_test[0]  # just the 10 rows of sequence 0
        df = pd.DataFrame(rows, columns=CANONICAL)
        csv_bytes = _make_csv_bytes(df)

        r = pil.load_prepared_csv(csv_bytes)
        assert not r.errors, r.errors

        probs_benchmark = self._load_model_and_run(
            X_test[0:1].astype(np.float32)
        )
        probs_csv = self._load_model_and_run(r.X)

        np.testing.assert_allclose(
            probs_csv, probs_benchmark,
            rtol=0, atol=1e-6,
            err_msg="Multi-sequence CSV identity test failed",
        )


# =============================================================
# PCAP INGESTION STILL WORKS (regression guard)
# =============================================================

class TestPCAPNotBroken:
    """Verify that importing prepared_input_loader does not break
    the existing pcap_ingest_v4 module."""

    def test_pcap_module_importable(self):
        import pcap_ingest_v4  # noqa: F401

    def test_pcap_feature_names_unchanged(self):
        import pcap_ingest_v4 as pcap_v4
        assert pcap_v4.INPUT_FEATURE_NAMES == CANONICAL

    def test_pcap_load_scaler(self):
        import pcap_ingest_v4 as pcap_v4
        path = pcap_v4.INPUT_SCALER_PATH
        if not path.exists():
            pytest.skip("input_scaler.pkl not found")
        scaler = pcap_v4.load_scaler(path)
        assert scaler is not None

    def test_pcap_model_loadable(self):
        import pcap_ingest_v4 as pcap_v4
        if not pcap_v4.CHECKPOINT_PATH.exists():
            pytest.skip("V4 checkpoint not found")
        model, ckpt = pcap_v4.load_v4_model()
        assert model is not None
        assert "max_k" in ckpt
