"""
=============================================================
V5 INPUT DISTRIBUTION-SHIFT DIAGNOSTIC
=============================================================

Measures how far the real host-level 82-D replay input sits from the V5
input scaler's own training reference statistics.

Task 4 showed the OUTPUT collapses to a low-probability attractor. This
script asks the input-side question directly: is the INPUT outside the
distribution the V5 scaler was fitted on, and if so, is the shift in the
52-D V4 flow prefix, the 30-D packet suffix, or both?

-------------------------------------------------------------
WHAT THIS IS AND IS NOT
-------------------------------------------------------------
Purely diagnostic and READ-ONLY. No scaler is refitted, no model is
loaded, no data is altered, no calibration or OOD logic is added.

The z-score used is the scaler's own definition:

    z = (x - scaler.mean_) / scaler.scale_

This is a PER-FEATURE marginal distance. It is NOT a multivariate OOD
score and NOT a validated OOD detector. No Mahalanobis distance is
computed — the scaler carries no covariance, so one cannot be derived
from it honestly.

The descriptive bands (LOW / MODERATE / HIGH on median |z|) are labels
for reading a table, not a decision rule.

-------------------------------------------------------------
THE OVERLAP CAVEAT
-------------------------------------------------------------
The 2,267 sequences are a SLIDING window over the joined window table, so
each underlying 10 s window appears in up to 10 sequences. Statistics
taken over the (2267, 10, 82) tensor therefore weight windows by how many
sequences contain them — interior windows count ~10x, windows at the edge
of a contiguous run count once.

Both views are computed and reported:

    A. per-sequence-position : all 2267 x 10 rows (overlap-weighted)
    B. unique-window         : each window_idx exactly once (unweighted)

View B is recovered cleanly because row t of a sequence is by construction
window_idx_start + t; the recovery is VERIFIED by checking that every
duplicate occurrence of a window is bitwise identical.

Usage:
    python src/analyze_v5_input_shift.py

=============================================================
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pcap_ingest_v4 as v4
import packet_feature_builder as pfb

REPO_ROOT = SRC_DIR.parent
DEMO_DIR = REPO_ROOT / "results" / "v5_demo"

HOST = "172.31.69.30"
X_PATH = DEMO_DIR / f"host_{HOST}_X.npy"
METADATA_PATH = DEMO_DIR / f"host_{HOST}_metadata.csv"

V5_SCALER_PATH = (
    REPO_ROOT / "data" / "processed" / "v5_multistep_K5" / "v5_input_scaler.pkl"
)
V4_SCALER_PATH = (
    REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_scaler.pkl"
)

FEATURE_CSV = DEMO_DIR / "input_shift_report.csv"
GROUP_CSV = DEMO_DIR / "input_shift_groups.csv"
SEQUENCE_CSV = DEMO_DIR / "sequence_shift.csv"
TOP_CSV = DEMO_DIR / "top_shifted_features.csv"
FIG_HEATMAP = DEMO_DIR / "input_shift_heatmap.png"
FIG_GROUPS = DEMO_DIR / "input_shift_group_summary.png"
FIG_SEQUENCE = DEMO_DIR / "sequence_shift_distribution.png"

Z_BANDS = (1, 2, 3, 4, 5)

# Sequential single hue (magnitude), from the documented blue ramp.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"


def _rule(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _load_scaler(path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as handle:
            return pickle.load(handle)


def _band(median_abs_z: float) -> str:
    """Descriptive reading aid — NOT a validated OOD detector."""
    if median_abs_z >= 2:
        return "HIGH SHIFT"
    if median_abs_z >= 1:
        return "MODERATE SHIFT"
    return "LOW SHIFT"


# =============================================================
# GROUPING (from the feature-name contract, never from position)
# =============================================================


def build_groups(names: list[str]) -> list[str]:
    """Assign each of the 82 features to its contract group by NAME."""
    v4_base = set(v4.BASE_FEATURES)              # 26
    v4_relative = set(v4.REL_CHANGE_FEATURES)    # 26
    packet_base = set(pfb.PACKET_BASE_FEATURES)  # 21
    packet_relative = set(pfb.PACKET_RELATIVE_FEATURES)  # 9

    groups = []
    for name in names:
        if name in v4_base:
            groups.append("V4_BASE")
        elif name in v4_relative:
            groups.append("V4_RELATIVE")
        elif name in packet_base:
            groups.append("PACKET_BASE")
        elif name in packet_relative:
            groups.append("PACKET_RELATIVE")
        else:
            raise RuntimeError(f"feature {name!r} is in no contract group")

    counts = pd.Series(groups).value_counts().to_dict()
    expected = {
        "V4_BASE": 26, "V4_RELATIVE": 26,
        "PACKET_BASE": 21, "PACKET_RELATIVE": 9,
    }
    if counts != expected:
        raise RuntimeError(f"group sizes {counts} != contract {expected}")
    return groups


# =============================================================
# UNIQUE-WINDOW RECOVERY
# =============================================================


def recover_unique_windows(
    X: np.ndarray, metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the overlapping sequence tensor to one row per window_idx.

    Row t of sequence s is, by construction, window_idx_start[s] + t. Every
    duplicate occurrence must therefore be identical; that is verified
    rather than assumed.
    """
    starts = metadata["window_idx_start"].to_numpy()
    table: dict[int, np.ndarray] = {}
    mismatches = 0
    for s in range(X.shape[0]):
        for t in range(X.shape[1]):
            w = int(starts[s]) + t
            if w in table:
                if not np.array_equal(table[w], X[s, t]):
                    mismatches += 1
            else:
                table[w] = X[s, t]
    if mismatches:
        raise RuntimeError(
            f"{mismatches} overlapping occurrences disagree — the unique-window "
            "view cannot be recovered cleanly"
        )
    order = np.array(sorted(table))
    unique = np.stack([table[w] for w in order])
    return unique, order


# =============================================================
# CORE STATISTICS
# =============================================================


def feature_stats(values: np.ndarray, z: np.ndarray) -> dict:
    """Per-feature stats for a (n_rows, n_features) pair of value/z arrays."""
    absolute = np.abs(z)
    stats = {
        "demo_mean": values.mean(axis=0),
        "demo_std": values.std(axis=0, ddof=1),
        "demo_min": values.min(axis=0),
        "demo_max": values.max(axis=0),
        "mean_abs_z": absolute.mean(axis=0),
        "median_abs_z": np.median(absolute, axis=0),
        "p95_abs_z": np.percentile(absolute, 95, axis=0),
        "p99_abs_z": np.percentile(absolute, 99, axis=0),
        "signed_mean_z": z.mean(axis=0),
    }
    for band in Z_BANDS:
        stats[f"fraction_abs_z_gt_{band}"] = (absolute > band).mean(axis=0)
    return stats


def main() -> int:
    X = np.load(X_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded {X_PATH.relative_to(REPO_ROOT)}  shape={X.shape}")

    v5_scaler = _load_scaler(V5_SCALER_PATH)
    v4_scaler = _load_scaler(V4_SCALER_PATH)
    names = [str(n) for n in np.asarray(v5_scaler.feature_names_in_).ravel()]
    if X.shape[2] != len(names) != 82:
        raise RuntimeError("feature-count mismatch against the V5 scaler")

    groups = build_groups(names)
    print(f"V5 scaler: n_features_in_={v5_scaler.n_features_in_}, "
          f"n_samples_seen_={v5_scaler.n_samples_seen_} (not refitted)")
    print(f"V4 scaler: n_features_in_={v4_scaler.n_features_in_}, "
          f"n_samples_seen_={v4_scaler.n_samples_seen_} (not refitted)")

    # ---------- the two views ----------
    _rule("STEP 1 — overlap handling: two views of the same data")
    flat = X.reshape(-1, 82)                                   # view A
    unique, window_order = recover_unique_windows(X, metadata)  # view B
    print(f"  A. per-sequence-position rows : {flat.shape[0]:>6}  "
          f"({X.shape[0]} sequences x 10 positions, overlap-weighted)")
    print(f"  B. unique underlying windows  : {unique.shape[0]:>6}  "
          f"(window_idx {window_order.min()}..{window_order.max()}, unweighted)")
    print(f"  overlap factor                : {flat.shape[0] / unique.shape[0]:.2f}x")
    print("  duplicate-occurrence consistency: VERIFIED bitwise identical")

    mean = np.asarray(v5_scaler.mean_, dtype=np.float64)
    scale = np.asarray(v5_scaler.scale_, dtype=np.float64)
    z_flat = (flat - mean) / scale
    z_unique = (unique - mean) / scale

    # ---------- feature-level report ----------
    _rule("STEP 2 — feature-level shift (V5 scaler reference)")
    primary = feature_stats(flat, z_flat)        # documented primary view
    secondary = feature_stats(unique, z_unique)  # unique-window view

    report = pd.DataFrame(
        {
            "feature_index": np.arange(82),
            "feature_name": names,
            "feature_group": groups,
            "training_mean": mean,
            "training_scale": scale,
            "demo_mean": primary["demo_mean"],
            "demo_std": primary["demo_std"],
            "demo_min": primary["demo_min"],
            "demo_max": primary["demo_max"],
            "mean_abs_z": primary["mean_abs_z"],
            "median_abs_z": primary["median_abs_z"],
            "p95_abs_z": primary["p95_abs_z"],
            "p99_abs_z": primary["p99_abs_z"],
            **{
                f"fraction_abs_z_gt_{b}": primary[f"fraction_abs_z_gt_{b}"]
                for b in Z_BANDS
            },
            "signed_mean_z": primary["signed_mean_z"],
            # out_of_range_fraction is DEFINED as fraction |z| > 3
            "out_of_range_fraction": primary["fraction_abs_z_gt_3"],
            "shift_band": [_band(v) for v in primary["median_abs_z"]],
            "mean_abs_z_unique_windows": secondary["mean_abs_z"],
            "median_abs_z_unique_windows": secondary["median_abs_z"],
        }
    )
    report.to_csv(FEATURE_CSV, index=False)
    print(f"  wrote {FEATURE_CSV.relative_to(REPO_ROOT)} (82 rows)")

    band_counts = report["shift_band"].value_counts()
    print("\n  descriptive bands on median |z| (reading aid, NOT a detector):")
    for band in ("LOW SHIFT", "MODERATE SHIFT", "HIGH SHIFT"):
        print(f"    {band:<16}{int(band_counts.get(band, 0)):>3} / 82")
    print(f"\n  A vs B agreement: max |mean_abs_z difference| = "
          f"{np.abs(primary['mean_abs_z'] - secondary['mean_abs_z']).max():.4f}")

    # ---------- group-level report ----------
    _rule("STEP 3 — group-level shift")
    rows = []
    for group in ("V4_BASE", "V4_RELATIVE", "PACKET_BASE", "PACKET_RELATIVE"):
        mask = np.array([g == group for g in groups])
        absolute = np.abs(z_flat[:, mask])
        per_feature_mean = primary["mean_abs_z"][mask]
        best = int(np.argmax(per_feature_mean))
        rows.append(
            {
                "group": group,
                "feature_count": int(mask.sum()),
                "mean_abs_z": float(absolute.mean()),
                "median_abs_z": float(np.median(absolute)),
                "p95_abs_z": float(np.percentile(absolute, 95)),
                "p99_abs_z": float(np.percentile(absolute, 99)),
                "mean_fraction_abs_z_gt_3": float(
                    primary["fraction_abs_z_gt_3"][mask].mean()
                ),
                "max_feature_mean_abs_z": float(per_feature_mean[best]),
                "max_feature_name": [n for n, g in zip(names, groups) if g == group][best],
                "median_abs_z_unique_windows": float(
                    np.median(np.abs(z_unique[:, mask]))
                ),
            }
        )
    group_frame = pd.DataFrame(rows)
    group_frame.to_csv(GROUP_CSV, index=False)
    print(f"  wrote {GROUP_CSV.relative_to(REPO_ROOT)}")
    print(
        f"\n  {'group':<17}{'n':>4}{'mean|z|':>10}{'med|z|':>9}"
        f"{'p95|z|':>9}{'p99|z|':>10}{'frac>3':>9}  worst feature"
    )
    for row in rows:
        print(
            f"  {row['group']:<17}{row['feature_count']:>4}"
            f"{row['mean_abs_z']:>10.3f}{row['median_abs_z']:>9.3f}"
            f"{row['p95_abs_z']:>9.3f}{row['p99_abs_z']:>10.3f}"
            f"{row['mean_fraction_abs_z_gt_3']:>9.4f}  "
            f"{row['max_feature_name']} ({row['max_feature_mean_abs_z']:.2f})"
        )

    # ---------- V4 prefix under BOTH scalers ----------
    _rule("STEP 4 — 52-D prefix: V5 scaler vs V4 benchmark scaler")
    v4_mean = np.asarray(v4_scaler.mean_, dtype=np.float64)
    v4_scale = np.asarray(v4_scaler.scale_, dtype=np.float64)
    prefix = flat[:, :52]
    z_prefix_v5 = (prefix - mean[:52]) / scale[:52]
    z_prefix_v4 = (prefix - v4_mean) / v4_scale

    stats_same = np.allclose(mean[:52], v4_mean) and np.allclose(scale[:52], v4_scale)
    print(f"  V5 prefix stats identical to V4 benchmark stats? {stats_same}")
    print(
        f"  {'reference':<26}{'mean|z|':>10}{'med|z|':>9}"
        f"{'p95|z|':>9}{'frac>3':>9}"
    )
    for label, zp in (
        ("V5 scaler (prefix 0:52)", z_prefix_v5),
        ("V4 benchmark scaler", z_prefix_v4),
    ):
        a = np.abs(zp)
        print(
            f"  {label:<26}{a.mean():>10.3f}{np.median(a):>9.3f}"
            f"{np.percentile(a, 95):>9.3f}{(a > 3).mean():>9.4f}"
        )
    suffix_abs = np.abs(z_flat[:, 52:])
    print(
        f"  {'V5 scaler (suffix 52:82)':<26}{suffix_abs.mean():>10.3f}"
        f"{np.median(suffix_abs):>9.3f}{np.percentile(suffix_abs, 95):>9.3f}"
        f"{(suffix_abs > 3).mean():>9.4f}"
    )

    # ---------- sequence-level ----------
    _rule("STEP 5 — sequence-level shift")
    z_seq = (X - mean) / scale
    abs_seq = np.abs(z_seq)
    sequence_frame = pd.DataFrame(
        {
            "sequence_index": metadata["sequence_index"],
            "window_idx_start": metadata["window_idx_start"],
            "window_idx_end": metadata["window_idx_end"],
            "timestamp_start": metadata["timestamp_start"],
            "timestamp_end": metadata["timestamp_end"],
            "timestamp_start_utc": metadata["timestamp_start_utc"],
            "timestamp_end_utc": metadata["timestamp_end_utc"],
            "mean_abs_z": abs_seq.mean(axis=(1, 2)),
            "max_abs_z": abs_seq.max(axis=(1, 2)),
            "fraction_gt_3": (abs_seq > 3).mean(axis=(1, 2)),
            "fraction_gt_5": (abs_seq > 5).mean(axis=(1, 2)),
        }
    )
    sequence_frame.to_csv(SEQUENCE_CSV, index=False)
    print(f"  wrote {SEQUENCE_CSV.relative_to(REPO_ROOT)} ({len(sequence_frame)} rows)")
    for column in ("mean_abs_z", "max_abs_z", "fraction_gt_3", "fraction_gt_5"):
        series = sequence_frame[column]
        print(
            f"  {column:<15} min {series.min():>8.4f}  median {series.median():>8.4f}"
            f"  mean {series.mean():>8.4f}  p95 {series.quantile(0.95):>8.4f}"
            f"  max {series.max():>8.4f}"
        )
    print(
        f"\n  sequences with mean_abs_z >= 1 : "
        f"{int((sequence_frame['mean_abs_z'] >= 1).sum())} / {len(sequence_frame)}"
    )

    # ---------- top shifted features ----------
    _rule("STEP 6 — top 20 most shifted features")
    columns = [
        "feature_name", "feature_group", "mean_abs_z", "median_abs_z",
        "p95_abs_z", "fraction_abs_z_gt_3", "signed_mean_z", "shift_band",
        "training_mean", "demo_mean",
    ]
    blocks = []
    for label, key in (
        ("mean_abs_z", "mean_abs_z"),
        ("p95_abs_z", "p95_abs_z"),
        ("fraction_abs_z_gt_3", "fraction_abs_z_gt_3"),
    ):
        block = report.nlargest(20, key)[columns].copy()
        block.insert(0, "ranking", label)
        block.insert(1, "rank", range(1, len(block) + 1))
        blocks.append(block)
    top = pd.concat(blocks, ignore_index=True)
    top.to_csv(TOP_CSV, index=False)
    print(f"  wrote {TOP_CSV.relative_to(REPO_ROOT)} ({len(top)} rows)")
    print(
        f"\n  top 20 by mean |z|:\n  {'#':<4}{'feature':<32}{'group':<17}"
        f"{'mean|z|':>9}{'med|z|':>8}{'p95':>8}{'frac>3':>8}{'signed':>9}"
    )
    for _, row in blocks[0].iterrows():
        print(
            f"  {int(row['rank']):<4}{row['feature_name']:<32}"
            f"{row['feature_group']:<17}{row['mean_abs_z']:>9.3f}"
            f"{row['median_abs_z']:>8.3f}{row['p95_abs_z']:>8.3f}"
            f"{row['fraction_abs_z_gt_3']:>8.4f}{row['signed_mean_z']:>9.3f}"
        )

    make_figures(report, group_frame, sequence_frame, z_flat, names, groups)
    print(
        "\nPer-feature marginal z only — not a multivariate OOD score, not a "
        "validated detector. Bands are descriptive reading aids."
    )
    return 0


# =============================================================
# FIGURES (diagnostic only)
# =============================================================


def make_figures(report, group_frame, sequence_frame, z_flat, names, groups) -> None:
    _rule("STEP 7 — diagnostic figures")

    # ---- Heatmap: feature x sequence-position, mean |z| ----
    order = np.argsort([
        ("V4_BASE", "V4_RELATIVE", "PACKET_BASE", "PACKET_RELATIVE").index(g)
        for g in groups
    ], kind="stable")
    absolute = np.abs(z_flat).reshape(-1, 10, 82)
    grid = absolute.mean(axis=0).T[order]  # (82 features, 10 positions)

    fig, ax = plt.subplots(figsize=(8.5, 15))
    fig.patch.set_facecolor(SURFACE)
    image = ax.imshow(grid, aspect="auto", cmap=SEQ_CMAP, vmin=0,
                      vmax=float(np.percentile(grid, 99)))
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"t-{9 - i}" for i in range(10)], fontsize=8)
    ax.set_yticks(range(82))
    ax.set_yticklabels([names[i] for i in order], fontsize=7)
    ax.tick_params(colors=INK_MUTED)
    # group boundaries
    boundary = 0
    for group in ("V4_BASE", "V4_RELATIVE", "PACKET_BASE", "PACKET_RELATIVE"):
        size = sum(1 for g in groups if g == group)
        if boundary:
            ax.axhline(boundary - 0.5, color=INK, linewidth=1.4)
        ax.text(10.1, boundary + size / 2 - 0.5, group.replace("_", "\n"),
                color=INK_MUTED, fontsize=8, va="center")
        boundary += size
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.22)
    bar.set_label("mean |z| vs V5 training reference", color=INK, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.set_xlabel("window position within the 10-step history", color=INK, fontsize=10)
    ax.set_title(
        "V5 input shift by feature and window position\n"
        f"host {HOST} replay — per-feature marginal |z|, diagnostic only",
        color=INK, fontsize=12, pad=14,
    )
    fig.tight_layout()
    fig.savefig(FIG_HEATMAP, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {FIG_HEATMAP.relative_to(REPO_ROOT)}")

    # ---- Group summary ----
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    labels = group_frame["group"].tolist()
    x = np.arange(len(labels))
    width = 0.26
    for slot, (key, label) in enumerate(
        (("mean_abs_z", "mean |z|"), ("median_abs_z", "median |z|"),
         ("p95_abs_z", "p95 |z|"))
    ):
        offset = (slot - 1) * (width + 0.02)
        bars = ax.bar(x + offset, group_frame[key], width,
                      color=SERIES_COLORS[slot], label=label, edgecolor=SURFACE,
                      linewidth=2)
        for rect, value in zip(bars, group_frame[key]):
            ax.annotate(f"{value:.2f}", (rect.get_x() + rect.get_width() / 2,
                                         rect.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        color=INK_MUTED, fontsize=8)
    ax.axhline(1, color=INK_MUTED, linewidth=1, linestyle="--", alpha=0.7)
    ax.axhline(2, color=INK_MUTED, linewidth=1, linestyle=":", alpha=0.7)
    ax.text(len(labels) - 0.4, 1.03, "MODERATE band floor", color=INK_MUTED, fontsize=8)
    ax.text(len(labels) - 0.4, 2.03, "HIGH band floor", color=INK_MUTED, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=9)
    ax.set_ylabel("|z| vs V5 training reference", color=INK, fontsize=10)
    ax.set_title(
        "Where the V5 input shift lives, by contract feature group\n"
        f"host {HOST} replay — per-feature marginal |z|, diagnostic only",
        color=INK, fontsize=12,
    )
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_MUTED)
    fig.tight_layout()
    fig.savefig(FIG_GROUPS, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {FIG_GROUPS.relative_to(REPO_ROOT)}")

    # ---- Sequence-level distribution ----
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    values = sequence_frame["mean_abs_z"].to_numpy()
    ax.hist(values, bins=60, color=BLUE_RAMP[4], edgecolor="none")
    median = float(np.median(values))
    ax.axvline(median, color=INK, linewidth=1.4, linestyle="--")
    ax.annotate(f"median {median:.3f}", (median, ax.get_ylim()[1] * 0.92),
                textcoords="offset points", xytext=(8, 0), color=INK, fontsize=9)
    ax.axvline(1, color=INK_MUTED, linewidth=1, linestyle=":")
    ax.annotate("MODERATE floor (|z|=1)", (1, ax.get_ylim()[1] * 0.70),
                textcoords="offset points", xytext=(8, 0), color=INK_MUTED, fontsize=8)
    ax.set_xlabel("sequence mean |z| across its 10 x 82 values", color=INK, fontsize=10)
    ax.set_ylabel("sequences", color=INK, fontsize=10)
    ax.set_title(
        "Per-sequence input shift in the host-level replay\n"
        f"{len(values):,} real sequences — per-feature marginal |z|, diagnostic only",
        color=INK, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_SEQUENCE, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {FIG_SEQUENCE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
