"""
=============================================================
PREPARED INPUT LOADER — V4 feature data validator
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Validates and loads prepared V4 feature data from CSV or NPY files
for inference through the FROZEN V4 pipeline.

This module is a FILE PARSER AND VALIDATOR only. It:
  - reads CSV or NPY files uploaded by the user
  - validates that they conform to the frozen V4 52-feature contract
  - builds (N, 10, 52) sequence tensors suitable for the frozen model
  - does NOT retrain, re-threshold, or modify any V4 artifact

The canonical 52-feature order is read from:
    data/processed/multiday_benchmark_K5/input_feature_names.txt

This is the SAME file that pcap_ingest_v4.py, feature_contract.py,
screening_data.py and v4_explainer.py all read.

Usage:
    from prepared_input_loader import load_prepared_csv, load_prepared_npy

    result = load_prepared_csv(csv_bytes_or_path)
    result = load_prepared_npy(npy_bytes_or_path)

    if result.errors:
        for e in result.errors:
            print(e)
    else:
        X = result.X                 # (N, 10, 52) float32
        print(result.n_sequences)
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================
# CANONICAL CONTRACT (read from the frozen benchmark artifacts)
# =============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
INPUT_FEATURE_NAMES_PATH = BENCHMARK_DIR / "input_feature_names.txt"

FEATURE_DIMENSION = 52
SEQUENCE_LENGTH = 10
WINDOW_SECONDS = 10

# Read the canonical feature order ONCE at import time.
# This is the single source of truth shared with every other V4 module.
_CANONICAL_FEATURE_NAMES: list[str] = [
    ln.strip()
    for ln in INPUT_FEATURE_NAMES_PATH.read_text().splitlines()
    if ln.strip()
]
assert len(_CANONICAL_FEATURE_NAMES) == FEATURE_DIMENSION, (
    f"canonical feature file has {len(_CANONICAL_FEATURE_NAMES)} names, "
    f"expected {FEATURE_DIMENSION}"
)


def canonical_feature_names() -> list[str]:
    """The frozen V4 52-feature order. Defensive copy."""
    return list(_CANONICAL_FEATURE_NAMES)


# =============================================================
# RESULT CONTAINER
# =============================================================

@dataclass
class PreparedInputResult:
    """Outcome of loading a prepared data file."""

    # Validated sequences ready for V4 inference.
    # Shape (N, 10, 52) float32, or None if validation failed.
    X: np.ndarray | None = None

    # Metadata
    source_format: str = ""          # "csv" or "npy"
    source_shape: tuple = ()         # raw shape before windowing
    n_sequences: int = 0             # number of valid 10-window sequences
    n_rows_raw: int = 0              # rows in the original file (CSV only)
    n_rejected_rows: int = 0         # rows rejected (NaN/Inf, CSV only)
    rejected_row_details: list[str] = field(default_factory=list)
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    has_timestamp: bool = False
    has_labels: bool = False         # ground-truth attack labels present
    labels: np.ndarray | None = None # (N,) binary labels if available

    # Validation
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================
# CSV LOADER
# =============================================================

# Columns that are NOT V4 features but may be present in a CSV.
# They are silently ignored (not treated as unknown columns).
_KNOWN_META_COLUMNS = {
    "timestamp", "window_idx", "window_index",
    "attack_present", "attack_label", "label",
    "sequence_id", "day", "source_day",
}

# Label column names we recognise for optional evaluation mode.
_LABEL_COLUMN_NAMES = {"attack_present", "attack_label", "label"}


def load_prepared_csv(
    source: bytes | str | Path,
) -> PreparedInputResult:
    """Parse and validate a CSV of V4 52-feature rows.

    The CSV must contain all 52 canonical feature columns.  Rows are
    windowed into contiguous 10-row sequences.  If a ``timestamp``
    column is present, rows are sorted chronologically and contiguity
    is validated at the 10-second level.  If no timestamp column is
    present, rows are taken in file order and every consecutive block
    of 10 is accepted.

    Returns a PreparedInputResult with errors populated on failure.
    """
    result = PreparedInputResult(source_format="csv")

    # --- Read CSV ---
    try:
        if isinstance(source, bytes):
            df = pd.read_csv(io.BytesIO(source))
        elif isinstance(source, (str, Path)):
            df = pd.read_csv(source)
        else:
            result.errors.append(
                f"Unsupported source type: {type(source).__name__}"
            )
            return result
    except Exception as exc:
        result.errors.append(f"CSV parsing failed: {exc}")
        return result

    result.n_rows_raw = len(df)
    result.source_shape = tuple(df.shape)

    if len(df) == 0:
        result.errors.append("CSV file is empty (0 rows).")
        return result

    # --- Validate columns ---
    csv_columns = list(df.columns)
    canonical = _CANONICAL_FEATURE_NAMES

    # Check for duplicate column names
    seen = set()
    dupes = []
    for c in csv_columns:
        if c in seen:
            dupes.append(c)
        seen.add(c)
    if dupes:
        result.errors.append(
            f"CSV contains duplicate column names: {dupes}. "
            "Remove duplicates and retry."
        )
        return result

    # Separate meta columns from feature columns
    meta_cols = {c for c in csv_columns if c in _KNOWN_META_COLUMNS}
    feature_cols_present = [c for c in csv_columns if c not in _KNOWN_META_COLUMNS]
    unknown_cols = [c for c in feature_cols_present if c not in canonical]

    if unknown_cols:
        result.warnings.append(
            f"Ignoring {len(unknown_cols)} unrecognised column(s): "
            f"{unknown_cols[:10]}{'…' if len(unknown_cols) > 10 else ''}"
        )

    # Check that all 52 canonical features are present
    missing = [f for f in canonical if f not in csv_columns]
    if missing:
        result.errors.append(
            f"Expected {FEATURE_DIMENSION} V4 features, found "
            f"{len([c for c in canonical if c in csv_columns])}. "
            f"Missing required feature(s): {missing}"
        )
        return result

    # --- Extract the 52-feature matrix in canonical order ---
    feature_df = df[canonical].copy()

    # Validate numeric
    for col in canonical:
        if not pd.api.types.is_numeric_dtype(feature_df[col]):
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="raise")
            except (ValueError, TypeError):
                result.errors.append(
                    f"CSV contains non-numeric values in feature column '{col}'."
                )
                return result

    # --- Check for NaN / Inf ---
    values = feature_df.to_numpy(dtype=np.float64)
    bad_mask = ~np.isfinite(values)
    bad_rows = np.where(bad_mask.any(axis=1))[0]

    valid_mask = np.ones(len(df), dtype=bool)
    if len(bad_rows) > 0:
        valid_mask[bad_rows] = False
        result.n_rejected_rows = len(bad_rows)
        for row_idx in bad_rows[:20]:  # cap detail output
            bad_cols = np.where(bad_mask[row_idx])[0]
            col_names = [canonical[c] for c in bad_cols]
            result.rejected_row_details.append(
                f"Row {row_idx}: non-finite values in {col_names}"
            )
        if len(bad_rows) > 20:
            result.rejected_row_details.append(
                f"… and {len(bad_rows) - 20} more rows with non-finite values"
            )
        result.warnings.append(
            f"{len(bad_rows)} row(s) rejected for NaN/Inf values."
        )

    # --- Handle timestamp ordering and contiguity ---
    has_ts = "timestamp" in df.columns
    result.has_timestamp = has_ts

    if has_ts:
        try:
            ts = pd.to_datetime(df["timestamp"])
            df = df.assign(_parsed_ts=ts)
            # Sort chronologically
            sort_idx = ts.argsort().to_numpy()
            df = df.iloc[sort_idx].reset_index(drop=True)
            feature_df = feature_df.iloc[sort_idx].reset_index(drop=True)
            valid_mask = valid_mask[sort_idx]
            ts = df["_parsed_ts"]

            result.first_timestamp = str(ts.iloc[0])
            result.last_timestamp = str(ts.iloc[len(ts) - 1])
        except Exception as exc:
            result.warnings.append(
                f"Could not parse timestamp column: {exc}. "
                "Treating rows in file order."
            )
            has_ts = False

    # --- Detect ground-truth labels ---
    label_col = None
    for lc in _LABEL_COLUMN_NAMES:
        if lc in df.columns:
            label_col = lc
            break
    result.has_labels = label_col is not None

    # --- Build 10-window sequences ---
    # Filter to valid rows (finite values only)
    valid_indices = np.where(valid_mask)[0]
    valid_values = feature_df.to_numpy(dtype=np.float64)[valid_mask]

    if len(valid_values) < SEQUENCE_LENGTH:
        result.errors.append(
            f"Not enough valid rows to form a {SEQUENCE_LENGTH}-step sequence. "
            f"Need at least {SEQUENCE_LENGTH}, have {len(valid_values)} "
            f"(after rejecting {result.n_rejected_rows} rows with NaN/Inf)."
        )
        return result

    if has_ts:
        ts_valid = ts.iloc[valid_indices].reset_index(drop=True)
        sequences, seq_labels = _build_sequences_with_timestamps(
            valid_values, ts_valid,
            df[label_col].iloc[valid_indices].to_numpy() if label_col else None,
            result,
        )
    else:
        sequences, seq_labels = _build_sequences_sequential(
            valid_values,
            df[label_col].iloc[valid_indices].to_numpy() if label_col else None,
        )

    if len(sequences) == 0:
        if has_ts:
            result.errors.append(
                f"Not enough contiguous {WINDOW_SECONDS}-second windows to form "
                f"a {SEQUENCE_LENGTH}-step sequence. Have {len(valid_values)} "
                f"valid rows, but none form {SEQUENCE_LENGTH} consecutive "
                f"{WINDOW_SECONDS}s-apart windows."
            )
        else:
            result.errors.append(
                f"Not enough rows ({len(valid_values)}) to form a "
                f"{SEQUENCE_LENGTH}-step sequence."
            )
        return result

    result.X = np.array(sequences, dtype=np.float32)
    result.n_sequences = len(sequences)
    result.source_shape = (result.n_rows_raw, FEATURE_DIMENSION)

    if seq_labels is not None:
        result.labels = np.array(seq_labels, dtype=np.float32)

    return result


def _build_sequences_with_timestamps(
    values: np.ndarray,
    timestamps: pd.Series,
    labels: np.ndarray | None,
    result: PreparedInputResult,
) -> tuple[list[np.ndarray], list | None]:
    """Slide a 10-row window, keeping only windows where all 10
    consecutive rows are exactly WINDOW_SECONDS apart."""
    sequences = []
    seq_labels = [] if labels is not None else None

    ts_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()

    for i in range(len(values) - SEQUENCE_LENGTH + 1):
        block_ts = ts_seconds[i:i + SEQUENCE_LENGTH]
        diffs = np.diff(block_ts)
        if np.allclose(diffs, WINDOW_SECONDS, atol=1.0):
            sequences.append(values[i:i + SEQUENCE_LENGTH])
            if labels is not None:
                # The label for a sequence is the label at the last window
                seq_labels.append(float(labels[i + SEQUENCE_LENGTH - 1]))
        # else: non-contiguous gap, silently skip

    return sequences, seq_labels


def _build_sequences_sequential(
    values: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[list[np.ndarray], list | None]:
    """Slide a 10-row window over sequential data (no timestamp check)."""
    sequences = []
    seq_labels = [] if labels is not None else None

    for i in range(len(values) - SEQUENCE_LENGTH + 1):
        sequences.append(values[i:i + SEQUENCE_LENGTH])
        if labels is not None:
            seq_labels.append(float(labels[i + SEQUENCE_LENGTH - 1]))

    return sequences, seq_labels


# =============================================================
# NPY LOADER
# =============================================================

def load_prepared_npy(
    source: bytes | str | Path,
) -> PreparedInputResult:
    """Load and validate an NPY file as prepared V4 input.

    Accepted shapes:
      (N, 10, 52)  — N sequences of 10 windows of 52 features
      (10, 52)     — a single sequence (expanded to (1, 10, 52))

    The data is treated as ALREADY PREPARED (scaled, windowed) V4
    input.  No additional scaling or windowing is applied.
    """
    result = PreparedInputResult(source_format="npy")

    # --- Load ---
    try:
        if isinstance(source, bytes):
            arr = np.load(io.BytesIO(source))
        elif isinstance(source, (str, Path)):
            arr = np.load(source)
        else:
            result.errors.append(
                f"Unsupported source type: {type(source).__name__}"
            )
            return result
    except Exception as exc:
        result.errors.append(f"NPY loading failed: {exc}")
        return result

    result.source_shape = tuple(arr.shape)

    # --- Validate shape ---
    if arr.ndim == 2:
        if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
            result.errors.append(
                f"NPY has shape {arr.shape}; a 2-D array must be "
                f"({SEQUENCE_LENGTH}, {FEATURE_DIMENSION})."
            )
            return result
        arr = arr[np.newaxis, ...]  # (1, 10, 52)

    elif arr.ndim == 3:
        if arr.shape[1] != SEQUENCE_LENGTH:
            result.errors.append(
                f"NPY must have sequence length {SEQUENCE_LENGTH} "
                f"(axis 1); received shape {arr.shape}."
            )
            return result
        if arr.shape[2] != FEATURE_DIMENSION:
            result.errors.append(
                f"NPY must have {FEATURE_DIMENSION} features (axis 2); "
                f"received shape {arr.shape}."
            )
            return result
    else:
        result.errors.append(
            f"NPY must have shape (N, {SEQUENCE_LENGTH}, {FEATURE_DIMENSION}) "
            f"or ({SEQUENCE_LENGTH}, {FEATURE_DIMENSION}); "
            f"received {arr.ndim}-D array with shape {arr.shape}."
        )
        return result

    # --- Validate values ---
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        result.errors.append(
            f"NPY contains {n_bad} non-finite value(s) (NaN or Inf). "
            "All values must be finite."
        )
        return result

    result.X = arr.astype(np.float32)
    result.n_sequences = arr.shape[0]

    return result


# =============================================================
# DISPATCH (auto-detect format from filename)
# =============================================================

def load_prepared_file(
    filename: str,
    data: bytes,
) -> PreparedInputResult:
    """Auto-detect format from filename and delegate."""
    ext = Path(filename).suffix.lower()
    if ext == ".npy":
        return load_prepared_npy(data)
    elif ext == ".csv":
        return load_prepared_csv(data)
    else:
        r = PreparedInputResult()
        r.errors.append(
            f"Unsupported file extension '{ext}'. "
            "Expected '.csv' or '.npy'."
        )
        return r
