"""
=============================================================
UNIFIED DATA INPUT LAYER — one entry point for every V4 source
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

The dashboard used to carry two competing upload pages (Live PCAP
Ingestion and Upload Prepared V4 Data), each with its own inference and
rendering code. This module replaces that with ONE representation and ONE
inference call, so all three sources converge before anything is computed:

    Historical Benchmark ─┐
    Upload PCAP          ─┼─►  PreparedInput  ─►  run_inference()  ─►  …
    Upload Prepared Data ─┘    (N, 10, 52)        frozen V4 rollout

WHAT THIS MODULE IS
-------------------
A VALIDATOR and an ADAPTER. It decides whether a file really is V4 model
input, explains itself when it is not, and normalises whatever survives
into a single container.

WHAT IT IS NOT
--------------
It is NOT a second inference implementation. `run_inference()` delegates to
the provenance-locked V4 explainer's own rollout, which is the same
`model.k_step_rollout` call the benchmark path has always used. It does not
retrain, re-scale, re-threshold or reorder anything, and it owns no model
weights.

WHY IDENTIFICATION COMES BEFORE VALIDATION
------------------------------------------
This project ships several arrays that are the same family of data but NOT
model input: future target states, per-horizon ground-truth labels, window
metadata, stored predictions. Handing one of those to a shape check
produces "wrong shape", which tells the user nothing about what they
actually uploaded. So every upload is first IDENTIFIED — by content shape,
corroborated by filename — and only then validated. A rejected file is told
what it is and what to upload instead.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import prepared_input_loader as pil

REPO_ROOT = Path(__file__).resolve().parent.parent

LAYER_VERSION = "v4-data-input-1.0"

SEQUENCE_LENGTH = pil.SEQUENCE_LENGTH          # 10
FEATURE_DIMENSION = pil.FEATURE_DIMENSION      # 52
WINDOW_SECONDS = pil.WINDOW_SECONDS            # 10
N_STATE_FEATURES = 26

# ---- source types (the only four values `source_type` may take) ----
SRC_BENCHMARK = "benchmark"
SRC_PCAP = "pcap"
SRC_CSV = "csv"
SRC_NPY = "npy"
SOURCE_TYPES = (SRC_BENCHMARK, SRC_PCAP, SRC_CSV, SRC_NPY)

SOURCE_DISPLAY = {
    SRC_BENCHMARK: "Historical Benchmark",
    SRC_PCAP: "Upload PCAP",
    SRC_CSV: "Upload Prepared V4 Data (CSV)",
    SRC_NPY: "Upload Prepared V4 Data (NPY)",
}

# ---- accepted upload extensions, by mode ----
PCAP_EXTENSIONS = ("pcap", "pcapng")
PREPARED_EXTENSIONS = ("csv", "npy")

STATUS_VALID = "VALID"
STATUS_INVALID = "INVALID"

VALID_BADGE = "✅ Valid V4 Input"
INVALID_BADGE = "❌ Invalid V4 Input"


# =============================================================
# 1. ARTIFACT ROLES
# =============================================================
#
# A role says what a file IS, so the dashboard can explain a rejection
# instead of reporting a shape mismatch. Roles are assigned from the
# content first; the filename only corroborates, because an uploaded file
# can be renamed to anything.

ROLE_MODEL_INPUT = "model_input"
ROLE_TARGET_STATES = "future_target_states"
ROLE_BINARY_LABELS = "ground_truth_labels"
ROLE_SOFT_LABELS = "ground_truth_labels_soft"
ROLE_METADATA = "metadata"
ROLE_PREDICTIONS = "predictions"
ROLE_ANNOTATIONS = "annotations"
ROLE_UNKNOWN = "unknown"

ROLE_DISPLAY = {
    ROLE_MODEL_INPUT: "V4 MODEL INPUT",
    ROLE_TARGET_STATES: "FUTURE TARGET STATES — not model input",
    ROLE_BINARY_LABELS: "GROUND-TRUTH LABELS — not model input",
    ROLE_SOFT_LABELS: "GROUND-TRUTH LABELS (soft) — not model input",
    ROLE_METADATA: "METADATA ONLY — not model input",
    ROLE_PREDICTIONS: "PREDICTIONS ONLY — not model input",
    ROLE_ANNOTATIONS: "DISPLAY ANNOTATIONS — not model input",
    ROLE_UNKNOWN: "UNRECOGNISED",
}

# The one sentence the task requires for a target-state array, reused
# verbatim wherever that role is reported.
MSG_TARGET_STATES = (
    "This file contains future target states, not V4 model input. "
    "Upload X_test.npy or a prepared V4 input with shape (N,10,52)."
)
MSG_BINARY_LABELS = (
    "This file contains per-horizon ground-truth attack labels, not V4 "
    "model input. Labels are what the model is scored against, never what "
    "it is fed. Upload X_test.npy or a prepared V4 input with shape "
    "(N,10,52)."
)
MSG_SOFT_LABELS = (
    "This file contains per-horizon soft ground-truth attack ratios, not "
    "V4 model input. Upload X_test.npy or a prepared V4 input with shape "
    "(N,10,52)."
)
MSG_METADATA = (
    "This file contains per-window metadata (timestamps, attack "
    "annotations, source day), not the 52 V4 feature columns. It describes "
    "the windows; it is not the model's input. Upload a CSV carrying all "
    "52 canonical V4 feature columns, or X_test.npy."
)
MSG_PREDICTIONS = (
    "This file contains stored model predictions (rollout probabilities, "
    "thresholds and alert bits), not V4 model input. It is the OUTPUT of a "
    "previous run. Upload a CSV carrying all 52 canonical V4 feature "
    "columns, or X_test.npy."
)
MSG_ANNOTATIONS = (
    "This file contains display-only region annotations, not V4 model "
    "input. It is never fed to the model. Upload a CSV carrying all 52 "
    "canonical V4 feature columns, or X_test.npy."
)

# The recognised project artifacts, for the reference table the page shows.
# `valid` is what the file IS, independent of whether it was uploaded.
KNOWN_ARTIFACTS = {
    "X_test.npy": {
        "role": ROLE_MODEL_INPUT, "shape": "(3115, 10, 52)", "valid": True,
        "note": "the frozen benchmark model-input tensor"},
    "screening_X.npy": {
        "role": ROLE_MODEL_INPUT, "shape": "(3115, 10, 52)", "valid": True,
        "note": "byte-identical copy of X_test.npy used by the replay"},
    "Y_multistep_test.npy": {
        "role": ROLE_TARGET_STATES, "shape": "(3115, 5, 26)", "valid": False,
        "note": "the 26-D next-state targets the decoder is trained against"},
    "A_binary_multistep_test.npy": {
        "role": ROLE_BINARY_LABELS, "shape": "(3115, 5)", "valid": False,
        "note": "per-horizon binary attack labels"},
    "A_soft_multistep_test.npy": {
        "role": ROLE_SOFT_LABELS, "shape": "(3115, 5)", "valid": False,
        "note": "per-horizon soft attack ratios"},
    "test_metadata.csv": {
        "role": ROLE_METADATA, "shape": "(3115, 5)", "valid": False,
        "note": "timestamp / attack_present / attack_ratio / dominant_attack"},
    "screening_metadata.csv": {
        "role": ROLE_METADATA, "shape": "(3115, …)", "valid": False,
        "note": "per-sequence window and label context"},
    "screening_predictions.csv": {
        "role": ROLE_PREDICTIONS, "shape": "(3115, 31)", "valid": False,
        "note": "frozen rollout probabilities, thresholds and alert bits"},
    "event_timeline.csv": {
        "role": ROLE_ANNOTATIONS, "shape": "(3115, …)", "valid": False,
        "note": "UI-only region annotation, never a model input"},
}

ROLE_MESSAGE = {
    ROLE_TARGET_STATES: MSG_TARGET_STATES,
    ROLE_BINARY_LABELS: MSG_BINARY_LABELS,
    ROLE_SOFT_LABELS: MSG_SOFT_LABELS,
    ROLE_METADATA: MSG_METADATA,
    ROLE_PREDICTIONS: MSG_PREDICTIONS,
    ROLE_ANNOTATIONS: MSG_ANNOTATIONS,
}


@dataclass
class Identification:
    """What a file is, and whether it may proceed to validation."""
    role: str
    is_model_input: bool
    headline: str                     # one line naming what the file is
    explanation: str = ""             # why it is rejected, if it is
    recognised_as: str = ""           # matched project artifact, if any
    detail: str = ""                  # observed shape / column evidence


def _npy_role_from_shape(shape: tuple) -> tuple[str, str]:
    """Structural role for an ndarray, from its shape alone.

    Model input wins first, so a legitimately-shaped array is never
    mis-explained as something else.
    """
    nd = len(shape)
    if nd == 3 and shape[1] == SEQUENCE_LENGTH and shape[2] == FEATURE_DIMENSION:
        return ROLE_MODEL_INPUT, "3-D (N, 10, 52)"
    if nd == 2 and shape == (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        return ROLE_MODEL_INPUT, "2-D (10, 52), a single sequence"
    if nd == 3 and shape[2] == N_STATE_FEATURES:
        return (ROLE_TARGET_STATES,
                f"3-D with a trailing {N_STATE_FEATURES}-D state axis "
                f"and a {shape[1]}-step horizon axis")
    if nd == 2 and shape[1] not in (FEATURE_DIMENSION,) and 1 <= shape[1] <= 8:
        return (ROLE_BINARY_LABELS,
                f"2-D with a short trailing axis of {shape[1]} — the shape of "
                f"a per-horizon label array, not of a feature window")
    return ROLE_UNKNOWN, f"{nd}-D array of shape {tuple(shape)}"


def identify_npy(filename: str, arr: np.ndarray) -> Identification:
    """Identify an uploaded .npy by content, corroborated by filename."""
    base = Path(filename).name
    shape = tuple(int(s) for s in arr.shape)
    role, evidence = _npy_role_from_shape(shape)

    known = KNOWN_ARTIFACTS.get(base)
    # Binary and soft label arrays are structurally identical ((N, K)
    # float), so content alone cannot separate them. Where the filename
    # names one of the two, it refines within the label family only -- it
    # never moves a file between families.
    _LABEL_FAMILY = (ROLE_BINARY_LABELS, ROLE_SOFT_LABELS)
    if (known is not None and role in _LABEL_FAMILY
            and known["role"] in _LABEL_FAMILY):
        role = known["role"]

    recognised = ""
    if known is not None and known["role"] == role:
        # Name and content agree — name the artifact in the message.
        recognised = base
    elif known is not None and known["role"] != role:
        # Renamed or altered file: the CONTENT decides, and the mismatch is
        # reported rather than silently trusting either one.
        evidence += (f" — note: the filename '{base}' normally carries "
                     f"{ROLE_DISPLAY[known['role']]} with shape "
                     f"{known['shape']}, which this file's shape does not "
                     f"match. The content was used, not the name.")

    if role == ROLE_MODEL_INPUT:
        return Identification(
            role=role, is_model_input=True,
            headline=(f"Recognised as {recognised} — the benchmark model-input "
                      f"tensor." if recognised
                      else "Shaped as V4 model input."),
            recognised_as=recognised, detail=evidence)

    if role == ROLE_UNKNOWN:
        return Identification(
            role=role, is_model_input=False,
            headline="This file is not recognisable as V4 model input.",
            explanation=(
                f"A V4 input tensor must have shape (N, {SEQUENCE_LENGTH}, "
                f"{FEATURE_DIMENSION}) or ({SEQUENCE_LENGTH}, "
                f"{FEATURE_DIMENSION}). This file is {evidence}. Upload "
                f"X_test.npy or a prepared V4 input with shape "
                f"(N,{SEQUENCE_LENGTH},{FEATURE_DIMENSION})."),
            detail=evidence)

    return Identification(
        role=role, is_model_input=False,
        headline=(f"Recognised as {recognised} — "
                  f"{ROLE_DISPLAY[role]}." if recognised
                  else f"Identified as {ROLE_DISPLAY[role]}."),
        explanation=ROLE_MESSAGE[role],
        recognised_as=recognised, detail=evidence)


# Column signatures used to identify a CSV that is NOT feature data.
_PREDICTION_SIGNATURE = {"nowcast_probability_t0", "prob_t1",
                         "max_forecast_probability", "alert_threshold"}
_METADATA_SIGNATURE = {"attack_ratio", "dominant_attack"}
_ANNOTATION_SIGNATURE = {"region", "target_t1_timestamp"}


def identify_csv(filename: str, columns: list) -> Identification:
    """Identify an uploaded .csv from its header, corroborated by filename."""
    base = Path(filename).name
    cols = set(columns)
    canonical = pil.canonical_feature_names()
    present = [f for f in canonical if f in cols]
    missing = [f for f in canonical if f not in cols]

    evidence = (f"{len(present)} of {FEATURE_DIMENSION} canonical V4 feature "
                f"columns present, {len(columns)} columns in total")

    if not missing:
        return Identification(
            role=ROLE_MODEL_INPUT, is_model_input=True,
            headline="All 52 canonical V4 feature columns are present.",
            detail=evidence)

    # Not feature data — say what it is instead.
    if _PREDICTION_SIGNATURE & cols or base == "screening_predictions.csv":
        role = ROLE_PREDICTIONS
    elif _ANNOTATION_SIGNATURE <= cols or base == "event_timeline.csv":
        role = ROLE_ANNOTATIONS
    elif (_METADATA_SIGNATURE & cols
          or base in ("test_metadata.csv", "screening_metadata.csv")):
        role = ROLE_METADATA
    else:
        role = ROLE_UNKNOWN

    known = KNOWN_ARTIFACTS.get(base)
    recognised = base if (known is not None and known["role"] == role) else ""

    if role == ROLE_UNKNOWN:
        shown = missing[:8]
        return Identification(
            role=role, is_model_input=False,
            headline="This CSV does not carry the V4 feature contract.",
            explanation=(
                f"A prepared V4 CSV must contain all {FEATURE_DIMENSION} "
                f"canonical feature columns, one row per {WINDOW_SECONDS}-"
                f"second window. {len(present)} were found and "
                f"{len(missing)} are missing, for example: "
                f"{', '.join(shown)}"
                f"{'…' if len(missing) > len(shown) else ''}."),
            detail=evidence)

    return Identification(
        role=role, is_model_input=False,
        headline=(f"Recognised as {recognised} — "
                  f"{ROLE_DISPLAY[role]}." if recognised
                  else f"Identified as {ROLE_DISPLAY[role]}."),
        explanation=ROLE_MESSAGE[role],
        recognised_as=recognised, detail=evidence)


def identify_upload(filename: str, data: bytes) -> Identification:
    """Identify an uploaded prepared-data file without fully loading it."""
    ext = Path(filename).suffix.lower()
    if ext == ".npy":
        try:
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as exc:                          # noqa: BLE001
            return Identification(
                role=ROLE_UNKNOWN, is_model_input=False,
                headline="This .npy file could not be read as an array.",
                explanation=f"NumPy could not load it: {exc}",
                detail="unreadable")
        return identify_npy(filename, arr)
    if ext == ".csv":
        try:
            head = pd.read_csv(io.BytesIO(data), nrows=0)
        except Exception as exc:                          # noqa: BLE001
            return Identification(
                role=ROLE_UNKNOWN, is_model_input=False,
                headline="This .csv file could not be parsed.",
                explanation=f"pandas could not read its header: {exc}",
                detail="unreadable")
        return identify_csv(filename, list(head.columns))
    return Identification(
        role=ROLE_UNKNOWN, is_model_input=False,
        headline=f"Unsupported file extension '{ext or "(none)"}'.",
        explanation=("Prepared V4 data must be a .csv of 52 feature columns "
                     "or a .npy of shape (N,10,52). To analyse raw traffic, "
                     "choose the Upload PCAP source instead."),
        detail="")


# =============================================================
# 2. THE COMMON REPRESENTATION
# =============================================================

@dataclass
class PreparedInput:
    """The one container every source converges on before inference.

    `x` is ALWAYS (N, 10, 52) float32 in frozen-scaler space — exactly what
    `model.k_step_rollout` consumes. Nothing downstream needs to know which
    source produced it.
    """
    x: np.ndarray | None = None
    timestamps: list | None = None
    source_type: str = ""
    source_name: str = ""

    # ---- validation report (drives the card the page renders) ----
    status: str = STATUS_INVALID
    format_label: str = ""
    source_shape: tuple = ()
    n_sequences: int = 0
    n_rejected: int = 0
    rejected_detail: list = field(default_factory=list)
    identification: Identification | None = None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    # ---- optional extras ----
    labels: np.ndarray | None = None
    has_labels: bool = False

    @property
    def is_valid(self) -> bool:
        return self.status == STATUS_VALID and self.x is not None

    @property
    def badge(self) -> str:
        return VALID_BADGE if self.is_valid else INVALID_BADGE

    @property
    def source_display(self) -> str:
        return SOURCE_DISPLAY.get(self.source_type, self.source_type)

    def card_rows(self) -> list[tuple]:
        """The compact validation card, as ordered (label, value) pairs."""
        return [
            ("Source", self.source_name or "—"),
            ("Format", self.format_label or "—"),
            ("Shape", str(self.source_shape) if self.source_shape else "—"),
            ("Valid sequences", str(self.n_sequences)),
            ("Rejected sequences", str(self.n_rejected)),
            ("Status", self.badge),
        ]


def _invalid(source_type, source_name, fmt, ident, shape=()) -> PreparedInput:
    p = PreparedInput(source_type=source_type, source_name=source_name,
                      format_label=fmt, status=STATUS_INVALID,
                      identification=ident, source_shape=shape)
    if ident is not None:
        p.errors.append(ident.headline)
        if ident.explanation:
            p.errors.append(ident.explanation)
    return p


# =============================================================
# 3. THE THREE SOURCES
# =============================================================

def from_benchmark(x: np.ndarray, timestamps=None,
                   source_name: str = "X_test.npy") -> PreparedInput:
    """The frozen benchmark tensor, wrapped in the common representation.

    No copy, no rescale, no reorder: the same array the benchmark path has
    always used, simply described in the shared container.
    """
    arr = np.asarray(x)
    return PreparedInput(
        x=arr, timestamps=list(timestamps) if timestamps is not None else None,
        source_type=SRC_BENCHMARK, source_name=source_name,
        status=STATUS_VALID, format_label="NPY (frozen benchmark)",
        source_shape=tuple(int(s) for s in arr.shape),
        n_sequences=int(arr.shape[0]), n_rejected=0,
        identification=Identification(
            role=ROLE_MODEL_INPUT, is_model_input=True,
            headline="The frozen benchmark model-input tensor.",
            recognised_as=source_name,
            detail="loaded from data/processed/multiday_benchmark_K5/"),
    )


def from_pcap(x: np.ndarray, timestamps=None, source_name: str = "",
              n_rejected: int = 0, notes=None) -> PreparedInput:
    """The output of the existing PCAP ingestion pipeline.

    The pipeline itself is untouched: this only wraps its (N, 10, 52)
    result so the rest of the console does not need a PCAP-specific path.
    """
    arr = np.asarray(x)
    p = PreparedInput(
        x=arr, timestamps=list(timestamps) if timestamps is not None else None,
        source_type=SRC_PCAP, source_name=source_name,
        status=STATUS_VALID if len(arr) else STATUS_INVALID,
        format_label="PCAP → 52-D V4 windows",
        source_shape=tuple(int(s) for s in arr.shape),
        n_sequences=int(arr.shape[0]) if arr.ndim == 3 else 0,
        n_rejected=int(n_rejected),
        identification=Identification(
            role=ROLE_MODEL_INPUT, is_model_input=True,
            headline="Built by the canonical PCAP ingestion pipeline.",
            detail="packets → flows → 10 s windows → 52-D V4 state → "
                   "10-window sequences"),
    )
    for n in (notes or []):
        p.warnings.append(n)
    if not len(arr):
        p.errors.append(
            "The PCAP produced no complete 10-window sequence. At least "
            f"{SEQUENCE_LENGTH} contiguous {WINDOW_SECONDS}-second windows "
            "of traffic are required.")
    return p


def from_upload(filename: str, data: bytes) -> PreparedInput:
    """Identify, then validate, then normalise an uploaded CSV / NPY.

    Identification runs FIRST, so a target-state array, a label array, a
    metadata table or a predictions table is explained for what it is
    rather than reported as a shape mismatch.
    """
    ext = Path(filename).suffix.lower()
    fmt = {"npy": "NPY", "csv": "CSV"}.get(ext.lstrip("."), ext.upper() or "—")
    source_type = SRC_NPY if ext == ".npy" else SRC_CSV

    ident = identify_upload(filename, data)
    if not ident.is_model_input:
        shape = ()
        if ext == ".npy":
            try:
                shape = tuple(int(s) for s in
                              np.load(io.BytesIO(data), allow_pickle=False).shape)
            except Exception:                             # noqa: BLE001
                shape = ()
        return _invalid(source_type, filename, fmt, ident, shape)

    # Identification passed — hand over to the frozen parser/validator,
    # which is reused unchanged and still owns every numeric check.
    res = pil.load_prepared_file(filename, data)

    p = PreparedInput(
        source_type=source_type, source_name=filename, format_label=fmt,
        source_shape=tuple(res.source_shape), n_sequences=res.n_sequences,
        n_rejected=res.n_rejected_rows,
        rejected_detail=list(res.rejected_row_details),
        identification=ident,
        errors=list(res.errors), warnings=list(res.warnings),
        labels=res.labels, has_labels=res.has_labels,
    )
    if res.errors or res.X is None:
        p.status = STATUS_INVALID
        return p

    p.x = res.X
    p.status = STATUS_VALID
    if res.has_timestamp and res.first_timestamp:
        p.timestamps = [res.first_timestamp, res.last_timestamp]
    return p


# =============================================================
# 4. THE SINGLE INFERENCE CALL
# =============================================================

def run_inference(prepared: PreparedInput, explainer, k: int) -> np.ndarray:
    """The ONE inference path, shared by every source.

    Delegates to the provenance-locked explainer's own rollout, which is
    the same `model.k_step_rollout` the benchmark path has always used. No
    source-specific branch exists here and no second implementation is
    created: a benchmark sequence, a PCAP window and an uploaded tensor all
    reach the frozen model through this function.

    Returns (N, k+1) probabilities; column 0 is the nowcast.
    """
    if not prepared.is_valid:
        raise ValueError(
            "run_inference received an input that did not validate; the "
            "caller must surface the validation errors instead.")
    x = np.asarray(prepared.x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        raise ValueError(
            f"PreparedInput.x has shape {x.shape}, expected "
            f"(N, {SEQUENCE_LENGTH}, {FEATURE_DIMENSION})")
    return explainer.predict_rollout(x, K=int(k))


# =============================================================
# 5. METADATA
# =============================================================

def layer_metadata() -> dict:
    return {
        "layer_version": LAYER_VERSION,
        "module": "src/data_input.py",
        "purpose": ("one input page, one validated representation, one "
                    "inference call for all three sources"),
        "source_types": list(SOURCE_TYPES),
        "accepted_extensions": {
            "pcap_mode": list(PCAP_EXTENSIONS),
            "prepared_mode": list(PREPARED_EXTENSIONS),
        },
        "common_representation": {
            "x": f"(N, {SEQUENCE_LENGTH}, {FEATURE_DIMENSION}) float32, "
                 f"frozen-scaler space",
            "timestamps": "optional",
            "source_type": list(SOURCE_TYPES),
            "source_name": "the file or dataset name",
        },
        "inference": {
            "function": "run_inference()",
            "delegates_to": "V4Explainer.predict_rollout -> "
                            "model.k_step_rollout",
            "second_implementation_created": False,
        },
        "known_artifacts": KNOWN_ARTIFACTS,
        "roles": ROLE_DISPLAY,
        "guarantees": {
            "retrains_v4": False,
            "modifies_v4_checkpoint": False,
            "modifies_scaler": False,
            "modifies_thresholds": False,
            "modifies_feature_order": False,
            "modifies_benchmark_predictions": False,
            "modifies_pcap_feature_extraction": False,
            "duplicate_inference_implementation": False,
        },
    }
