"""
=============================================================
V4 SCREENING SUBSTRATE LOADER — read-only
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

The single access point to the validated Task-10 screening replay
substrate in results/v4_screening_mvp/. The dashboard and the dashboard
validator both read through here so they cannot disagree about what the
demo is showing.

This module is READ-ONLY. It never writes, never recomputes a threshold,
never fabricates a row, and never falls back to synthetic data: if an
artifact is missing or malformed it raises ScreeningDataError and the
caller is expected to surface that to the user.

Authoritative artifacts (all produced by src/build_v4_screening_mvp.py):
    screening_X.npy            (N, 10, 52) float32 — byte-identical to X_test.npy
    screening_metadata.csv     per-sequence window/label context
    screening_predictions.csv  frozen V4 rollout probabilities + alerts
    event_timeline.csv         UI-only region annotation (NEVER model input)
    feature_names.txt          the frozen 52-D feature order
    screening_manifest.json    provenance
    readiness_report.json      14 PASS/FAIL gates
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCREENING_DIR = REPO_ROOT / "results" / "v4_screening_mvp"

SEQ_LEN = 10
N_FEATURES = 52
MAX_K = 5
WINDOW_SECONDS = 10
HORIZON_SECONDS = WINDOW_SECONDS * MAX_K          # 50 s forecast horizon

# Horizon labels the UI shows. t+4 is displayed but carries NO frozen
# point threshold (the reference evaluation covers horizons 1,2,3,5).
HORIZON_LABELS = {1: "+10s", 2: "+20s", 3: "+30s", 4: "+40s", 5: "+50s"}

REQUIRED_FILES = [
    "screening_X.npy",
    "screening_metadata.csv",
    "screening_predictions.csv",
    "event_timeline.csv",
    "feature_names.txt",
    "screening_manifest.json",
    "readiness_report.json",
]

PRED_REQUIRED_COLUMNS = [
    "sequence_id", "target_t1_timestamp", "observed_window_start",
    "observed_window_end", "nowcast_probability_t0",
    "prob_t1", "prob_t2", "prob_t3", "prob_t4", "prob_t5",
    "max_forecast_probability", "alert_threshold", "alert",
]

# Region vocabulary written by the Task-10 builder. These are CONTEXT
# annotations for replay only — see ScreeningSubstrate.model_input().
REGION_ORDER = [
    "benign",
    "pre_attack",
    "documented_attack_onset",
    "attack_active",
    "attack_active_at_start_no_observable_onset",
]

REGION_DISPLAY = {
    "benign": "Benign",
    "pre_attack": "Pre-attack (≤50 s before onset)",
    "documented_attack_onset": "Documented attack onset",
    "attack_active": "Attack active",
    "attack_active_at_start_no_observable_onset": "Attack active at capture start (no observable onset)",
}

REGION_COLORS = {
    "benign": "#1b3a2f",
    "pre_attack": "#6b5310",
    "documented_attack_onset": "#8c3b12",
    "attack_active": "#5c1622",
    "attack_active_at_start_no_observable_onset": "#3d2230",
}

# Compact, analyst-meaningful subset of the frozen 52-D contract used for
# the network-state forecast panel. Every name MUST exist in the contract;
# load() verifies that, so a typo fails loudly instead of silently plotting
# the wrong column.
STATE_FORECAST_FEATURES = [
    "flow_count",
    "log_total_packets",
    "log_total_bytes",
    "window_packet_rate",
    "window_byte_rate",
    "syn_rate",
    "rst_rate",
    "unique_protocols",
    "dst_port_entropy",
]


# Plain-English gloss for each of the 26 base features, for analyst-facing
# attribution text. Descriptive only -- these say what the feature MEASURES,
# never what it causes.
FEATURE_PLAIN = {
    "flow_count": "number of network flows in the window",
    "log_flow_count": "flow count (log-scaled)",
    "log_total_packets": "total packets in the window (log-scaled)",
    "log_total_bytes": "total bytes in the window (log-scaled)",
    "window_packet_rate": "packets per second",
    "window_byte_rate": "bytes per second",
    "syn_rate": "share of SYN-flagged traffic (connection attempts)",
    "ack_rate": "share of ACK-flagged traffic",
    "rst_rate": "share of RST-flagged traffic (refused/reset connections)",
    "fin_rate": "share of FIN-flagged traffic (connection teardown)",
    "psh_rate": "share of PSH-flagged traffic",
    "urg_rate": "share of URG-flagged traffic",
    "mean_pkt_len": "average packet size",
    "std_pkt_len": "variability in packet size",
    "log_mean_flow_duration": "average flow duration (log-scaled)",
    "log_mean_fwd_iat": "average gap between outbound packets (log-scaled)",
    "log_mean_bwd_iat": "average gap between inbound packets (log-scaled)",
    "fwd_packet_ratio": "fraction of packets sent outbound",
    "bwd_packet_ratio": "fraction of packets received inbound",
    "fwd_byte_ratio": "fraction of bytes sent outbound",
    "bwd_byte_ratio": "fraction of bytes received inbound",
    "mean_down_up_ratio": "average download-to-upload ratio",
    "log_unique_dst_ports": "number of distinct destination ports (log-scaled)",
    "unique_protocols": "number of distinct protocols in use",
    "dst_port_entropy": "how spread out destination ports are (scan-like if high)",
    "protocol_entropy": "how mixed the protocol usage is",
}


def feature_plain(name: str) -> str:
    """Plain-English gloss for any of the 52 contract features."""
    if name.endswith("_rel_change"):
        base = name[: -len("_rel_change")]
        return f"rate of change in {FEATURE_PLAIN.get(base, base)}"
    return FEATURE_PLAIN.get(name, name)


def attribution_sentence(name: str, value: float) -> str:
    """Analyst-facing, non-causal one-liner for a signed attribution."""
    direction = ("contributes toward **higher** predicted threat" if value > 0
                 else "contributes toward **lower** predicted threat")
    return f"{feature_plain(name)} — {direction}"


class ScreeningDataError(RuntimeError):
    """Raised when the screening substrate is missing or malformed.

    Callers must surface this to the user. There is deliberately no
    synthetic fallback."""


class ScreeningSubstrate:
    """The loaded, validated screening replay substrate."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.X: np.ndarray
        self.meta: pd.DataFrame
        self.preds: pd.DataFrame
        self.timeline: pd.DataFrame
        self.feature_names: list[str]
        self.manifest: dict
        self.readiness: dict

    # ---------------------------------------------------------
    # DERIVED PROPERTIES
    # ---------------------------------------------------------

    @property
    def n_sequences(self) -> int:
        return len(self.preds)

    @property
    def alert_threshold(self) -> float:
        """The frozen alert threshold. Read from the substrate, never
        chosen here."""
        return float(self.preds["alert_threshold"].iloc[0])

    @property
    def threshold_source(self) -> str:
        return str(self.manifest["alert_rule"]["threshold_source"])

    @property
    def point_thresholds(self) -> dict:
        """Frozen per-horizon point thresholds. K=4 is absent by design."""
        return {int(k): float(v)
                for k, v in self.manifest["alert_rule"]["per_horizon_point_thresholds"].items()}

    @property
    def probs(self) -> np.ndarray:
        """(N, 6) rollout probabilities: column 0 is the nowcast,
        columns 1..5 are the t+1..t+5 forecast."""
        cols = ["nowcast_probability_t0"] + [f"prob_t{k}" for k in range(1, MAX_K + 1)]
        return self.preds[cols].to_numpy(dtype=float)

    @property
    def max_forecast(self) -> np.ndarray:
        return self.preds["max_forecast_probability"].to_numpy(dtype=float)

    @property
    def alerts(self) -> np.ndarray:
        return self.preds["alert"].to_numpy(dtype=int)

    @property
    def timestamps(self) -> pd.Series:
        return self.preds["target_t1_timestamp"]

    @property
    def default_sequence_id(self) -> int:
        """Reproducible demo default: the sequence with the highest max
        forecast probability, ties broken by lowest sequence_id.

        Chosen from MODEL OUTPUT only — never from event labels — so the
        choice introduces no label leakage."""
        mf = self.max_forecast
        return int(np.flatnonzero(mf == mf.max())[0])

    # ---------------------------------------------------------
    # CAPTURE GAPS
    # ---------------------------------------------------------

    def capture_gaps(self) -> list[dict]:
        """Discontinuities BETWEEN consecutive replay sequences.

        These are real gaps in the raw CSE-CIC-IDS2018 capture. They are
        never interpolated; the UI must break its lines across them."""
        ts = pd.to_datetime(self.preds["target_t1_timestamp"])
        delta = ts.diff().dt.total_seconds()
        out = []
        for i in np.flatnonzero((delta.notna() & (delta != WINDOW_SECONDS)).to_numpy()):
            out.append({
                "break_before_sequence_id": int(i),
                "last_sequence_id": int(i - 1),
                "from_timestamp": str(ts.iloc[i - 1]),
                "to_timestamp": str(ts.iloc[i]),
                "gap_seconds": float(delta.iloc[i]),
            })
        return out

    def segments(self) -> list[tuple[int, int]]:
        """Contiguous [start, end] sequence-id runs between capture gaps."""
        breaks = [g["break_before_sequence_id"] for g in self.capture_gaps()]
        bounds = [0] + breaks + [self.n_sequences]
        return [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]

    def segment_of(self, seq_id: int) -> tuple[int, int]:
        for lo, hi in self.segments():
            if lo <= seq_id <= hi:
                return lo, hi
        return 0, self.n_sequences - 1

    def insert_gap_breaks(self, seq_ids, values):
        """Return (x, y) with NaN inserted at every capture gap so a line
        chart does NOT draw a misleading continuous segment across missing
        data. Plotly renders NaN as a break."""
        seq_ids = list(seq_ids)
        values = list(values)
        gap_at = {g["break_before_sequence_id"] for g in self.capture_gaps()}
        xs, ys = [], []
        for i, (s, v) in enumerate(zip(seq_ids, values)):
            if i > 0 and s in gap_at:
                xs.append(None)
                ys.append(None)
            xs.append(s)
            ys.append(v)
        return xs, ys

    # ---------------------------------------------------------
    # MODEL INPUT  (the leakage boundary)
    # ---------------------------------------------------------

    def model_input(self, seq_id: int) -> np.ndarray:
        """The ONLY tensor the model ever sees for a sequence: (1, 10, 52)
        of frozen, scaled V4 features.

        Nothing from event_timeline.csv, screening_metadata.csv label
        columns, or any ground-truth field is present here or reachable
        from here. The event annotations are display context only."""
        if not (0 <= seq_id < self.n_sequences):
            raise ScreeningDataError(
                f"sequence_id {seq_id} out of range 0..{self.n_sequences - 1}")
        return self.X[seq_id: seq_id + 1]

    def history_frame(self, seq_id: int, features: list[str] | None = None) -> pd.DataFrame:
        """The 10 observed input windows, labelled t-90 … t (scaled units)."""
        feats = features or STATE_FORECAST_FEATURES
        idx = [self.feature_names.index(f) for f in feats]
        end = pd.to_datetime(self.preds["observed_window_end"].iloc[seq_id])
        rows = []
        for step in range(SEQ_LEN):
            offset = (SEQ_LEN - 1 - step) * WINDOW_SECONDS
            rows.append({
                "step_label": "t" if offset == 0 else f"t-{offset}",
                "timestamp": end - pd.Timedelta(seconds=offset),
                **{f: float(self.X[seq_id, step, i]) for f, i in zip(feats, idx)},
            })
        return pd.DataFrame(rows)

    def region_of(self, seq_id: int) -> str:
        return str(self.timeline["region"].iloc[seq_id])


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise ScreeningDataError(f"{path.name} is not readable JSON: {exc}") from exc


def load(directory: Path | str = SCREENING_DIR) -> ScreeningSubstrate:
    """Load and validate the screening substrate.

    Raises ScreeningDataError on any missing file, shape mismatch, column
    mismatch or non-finite probability. Never returns partial or
    synthesised data."""
    directory = Path(directory)
    if not directory.exists():
        raise ScreeningDataError(
            f"screening substrate directory not found: {directory}. "
            "Run: python src/build_v4_screening_mvp.py")

    missing = [f for f in REQUIRED_FILES if not (directory / f).exists()]
    if missing:
        raise ScreeningDataError(
            f"missing screening artifact(s) in {directory}: {', '.join(missing)}. "
            "Run: python src/build_v4_screening_mvp.py")

    sub = ScreeningSubstrate(directory)

    try:
        sub.X = np.load(directory / "screening_X.npy")
    except Exception as exc:
        raise ScreeningDataError(f"screening_X.npy could not be loaded: {exc}") from exc

    if sub.X.ndim != 3 or sub.X.shape[1:] != (SEQ_LEN, N_FEATURES):
        raise ScreeningDataError(
            f"screening_X.npy has shape {sub.X.shape}, expected (N, {SEQ_LEN}, {N_FEATURES})")
    if not np.isfinite(sub.X).all():
        raise ScreeningDataError("screening_X.npy contains non-finite values")

    sub.feature_names = [ln.strip() for ln
                         in (directory / "feature_names.txt").read_text().splitlines()
                         if ln.strip()]
    if len(sub.feature_names) != N_FEATURES:
        raise ScreeningDataError(
            f"feature_names.txt has {len(sub.feature_names)} names, expected {N_FEATURES}")

    unknown = [f for f in STATE_FORECAST_FEATURES if f not in sub.feature_names]
    if unknown:
        raise ScreeningDataError(
            f"state-forecast features absent from the frozen contract: {unknown}")

    try:
        sub.meta = pd.read_csv(directory / "screening_metadata.csv")
        sub.preds = pd.read_csv(directory / "screening_predictions.csv")
        sub.timeline = pd.read_csv(directory / "event_timeline.csv")
    except Exception as exc:
        raise ScreeningDataError(f"a screening CSV could not be parsed: {exc}") from exc

    missing_cols = [c for c in PRED_REQUIRED_COLUMNS if c not in sub.preds.columns]
    if missing_cols:
        raise ScreeningDataError(
            f"screening_predictions.csv is missing column(s): {', '.join(missing_cols)}")

    n = len(sub.X)
    for name, df in [("screening_metadata.csv", sub.meta),
                     ("screening_predictions.csv", sub.preds),
                     ("event_timeline.csv", sub.timeline)]:
        if len(df) != n:
            raise ScreeningDataError(
                f"{name} has {len(df)} rows but screening_X.npy has {n} sequences")

    for col in ["target_t1_timestamp", "observed_window_start", "observed_window_end"]:
        sub.preds[col] = pd.to_datetime(sub.preds[col])
    for col in [c for c in sub.meta.columns if c.endswith("_timestamp")
                or c.startswith("observed_window")]:
        sub.meta[col] = pd.to_datetime(sub.meta[col])
    sub.timeline["target_t1_timestamp"] = pd.to_datetime(sub.timeline["target_t1_timestamp"])

    prob_cols = ["nowcast_probability_t0"] + [f"prob_t{k}" for k in range(1, MAX_K + 1)]
    probs = sub.preds[prob_cols].to_numpy(dtype=float)
    if not np.isfinite(probs).all():
        raise ScreeningDataError("screening_predictions.csv contains non-finite probabilities")
    if probs.min() < 0.0 or probs.max() > 1.0:
        raise ScreeningDataError(
            f"probabilities out of [0,1]: [{probs.min():.4f}, {probs.max():.4f}]")

    if sub.preds["alert_threshold"].nunique() != 1:
        raise ScreeningDataError(
            "screening_predictions.csv carries more than one alert_threshold; "
            "the demo requires a single frozen threshold")

    bad_regions = sorted(set(sub.timeline["region"]) - set(REGION_ORDER))
    if bad_regions:
        raise ScreeningDataError(f"event_timeline.csv has unknown region(s): {bad_regions}")

    sub.manifest = _read_json(directory / "screening_manifest.json")
    sub.readiness = _read_json(directory / "readiness_report.json")

    for key in ["alert_rule", "early_warning", "contiguity"]:
        if key not in sub.manifest:
            raise ScreeningDataError(f"screening_manifest.json is missing '{key}'")

    return sub


def early_warning_summary(sub: ScreeningSubstrate) -> dict:
    """The STRICT pre-attack early-warning figures — the ones the demo is
    allowed to headline. The 50 s reference figure is carried alongside for
    disclosure but must not be presented as the headline."""
    ew = sub.manifest["early_warning"]
    strict = ew.get("strict_pre_attack_variant", {})
    return {
        "evaluable_onsets": int(ew["evaluable_onsets"]),
        "strict_warned": int(strict.get("warned", 0)),
        "strict_missed": int(strict.get("missed", 0)),
        "strict_coverage_pct": float(strict.get("coverage_pct", 0.0)),
        "strict_mean_lead_s": float(strict.get("mean_lead_s", 0.0)),
        "strict_median_lead_s": float(strict.get("median_lead_s", 0.0)),
        "strict_episodes": strict.get("episodes", []),
        "reference_coverage_pct": float(ew["coverage_pct"]),
        "reference_mean_lead_s": float(ew["mean_lead_s"]),
        "threshold": float(ew["threshold"]),
        "false_positive_windows_per_hour": float(ew["false_positive_windows_per_hour"]),
        "caveat": ew.get("caveat", ""),
    }
