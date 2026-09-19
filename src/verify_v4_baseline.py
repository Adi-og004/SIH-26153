"""
=============================================================
V4 BASELINE VERIFICATION — READ-ONLY INTEGRITY CHECK
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Purpose:
    Verify that the repository still matches the frozen V4
    reference baseline recorded in docs/v4_baseline_manifest.json.

This script NEVER trains, fine-tunes, re-scales, re-thresholds
or writes any model/data artifact. It only reads and compares.

Checks performed:
    1.  Manifest present and parseable
    2.  Checkpoint exists + SHA256 matches
    3.  Feature schema exists + order/SHA256 match (52 = 26 + 26)
    4.  Frozen schema copy (docs/v4_feature_schema.json) agrees
    5.  Input scaler exists + SHA256 + fitted statistics match
    6.  Target scaler exists + SHA256 + fitted statistics match
    7.  Benchmark dataset artifacts exist + SHA256 match
    8.  Input shape is 10 x 52 in every split
    9.  Checkpoint loads into the V4 architecture with strict=True
    10. Model dimensions match the manifest
    11. Frozen thresholds present and within (0, 1)
    12. Git revision matches the manifest freeze commit

Usage:
    python src/verify_v4_baseline.py
    python src/verify_v4_baseline.py --manifest docs/v4_baseline_manifest.json

Exit code:
    0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "v4_baseline_manifest.json"

STAT_TOLERANCE = 1e-9


# =============================================================
# V4 ARCHITECTURE (verbatim from src/train_latent_world_model_v4.py)
# =============================================================

class LatentWorldModelV4(nn.Module):
    """Exact V4 architecture — used here only to prove the checkpoint loads."""

    def __init__(
        self,
        input_size,
        hidden_size,
        latent_size,
        output_size,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.encoder_norm = nn.LayerNorm(hidden_size)
        self.encoder_fc = nn.Linear(hidden_size, latent_size)

        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size * 4),
            nn.ReLU(),
            nn.Linear(latent_size * 4, latent_size),
        )
        self.gate_fc = nn.Linear(latent_size, latent_size)
        self.transition_norm = nn.LayerNorm(latent_size)

        self.decoder_fc1 = nn.Linear(latent_size, 64)
        self.decoder_activation = nn.ReLU()
        self.decoder_fc2 = nn.Linear(64, output_size)

        self.attack_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )


# =============================================================
# REPORTING
# =============================================================

class Report:
    """Collects PASS/FAIL lines and detail messages."""

    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        if not ok:
            self.failed = True
        return ok

    def render(self):
        width = max(len(name) for name, _, _ in self.rows)

        print()
        print("V4 BASELINE VERIFICATION")
        print("-" * 24)

        for name, ok, detail in self.rows:
            status = "PASS" if ok else "FAIL"
            line = f"{name:<{width}} : {status}"
            if detail:
                line += f"   ({detail})"
            print(line)

        overall = "FAIL" if self.failed else "PASS"
        print()
        print(f"OVERALL: {overall}")
        print()

        if self.failed:
            print(
                "The repository no longer matches the frozen V4 baseline.\n"
                "The reported V4 results describe a different artifact set.\n"
                "Restore the listed artifacts before comparing any model against V4."
            )
        else:
            print(
                "Repository matches docs/v4_baseline_manifest.json.\n"
                "Reference evaluation (see docs/V4_REFERENCE_BASELINE.md section 10):\n"
                "  git show 2be3073:src/k_step_rollout.py > /tmp/k_step_rollout_reference.py\n"
                "  python /tmp/k_step_rollout_reference.py"
            )


# =============================================================
# HELPERS
# =============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def check_file(report, label, relative_path, expected_sha):
    """Existence + hash check for one artifact."""
    path = resolve(relative_path)

    if not path.exists():
        return report.add(label, False, f"missing: {relative_path}")

    actual = sha256_file(path)

    if actual != expected_sha:
        return report.add(
            label,
            False,
            f"SHA256 mismatch: expected {expected_sha[:12]}…, got {actual[:12]}…",
        )

    return report.add(label, True, f"sha {actual[:12]}…")


# =============================================================
# INDIVIDUAL CHECKS
# =============================================================

def check_checkpoint_file(report, manifest):
    entry = manifest["checkpoint"]
    path = resolve(entry["path"])

    if not path.exists():
        return report.add("Checkpoint", False, f"missing: {entry['path']}")

    size = path.stat().st_size
    if size != entry["size_bytes"]:
        return report.add(
            "Checkpoint",
            False,
            f"size {size} != frozen {entry['size_bytes']}",
        )

    actual = sha256_file(path)
    if actual != entry["sha256"]:
        return report.add(
            "Checkpoint",
            False,
            f"SHA256 mismatch: got {actual[:12]}…",
        )

    return report.add("Checkpoint", True, f"{size} B, sha {actual[:12]}…")


def check_feature_schema(report, manifest):
    spec = manifest["input"]
    path = resolve(spec["feature_order_file"])

    if not path.exists():
        return report.add("Feature schema", False, f"missing: {spec['feature_order_file']}")

    actual_sha = sha256_file(path)
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    base = spec["base_features"]
    relative = spec["relative_features"]
    expected_order = base + relative

    problems = []

    if actual_sha != spec["feature_order_sha256"]:
        problems.append("SHA256 mismatch")
    if len(names) != spec["feature_dimension"]:
        problems.append(f"{len(names)} features, expected {spec['feature_dimension']}")
    if len(base) != spec["base_feature_count"]:
        problems.append(f"{len(base)} base features in manifest")
    if len(relative) != spec["relative_feature_count"]:
        problems.append(f"{len(relative)} relative features in manifest")
    if names != expected_order:
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(zip(names, expected_order))
                if a != b
            ),
            min(len(names), len(expected_order)),
        )
        problems.append(f"order differs at index {first_diff}")
    if relative != [f"{name}_rel_change" for name in base]:
        problems.append("relative features are not base+'_rel_change'")

    if problems:
        return report.add("Feature schema", False, "; ".join(problems))

    return report.add(
        "Feature schema",
        True,
        f"{len(base)} base + {len(relative)} rel = {len(names)}",
    )


def check_frozen_schema_copy(report, manifest):
    spec = manifest["input"]
    relative_path = spec.get("frozen_schema_copy")

    if not relative_path:
        return report.add("Frozen schema copy", False, "not declared in manifest")

    path = resolve(relative_path)
    if not path.exists():
        return report.add("Frozen schema copy", False, f"missing: {relative_path}")

    schema = json.loads(path.read_text())
    names = [item["name"] for item in schema["features"]]
    expected = spec["base_features"] + spec["relative_features"]

    if schema.get("source_sha256") != spec["feature_order_sha256"]:
        return report.add("Frozen schema copy", False, "source SHA256 differs from manifest")

    if names != expected:
        return report.add("Frozen schema copy", False, "feature order differs from manifest")

    if schema.get("input_shape") != [spec["sequence_length"], spec["feature_dimension"]]:
        return report.add("Frozen schema copy", False, "input_shape differs from manifest")

    return report.add("Frozen schema copy", True, relative_path)


def check_scaler(report, label, entry):
    path = resolve(entry["path"])

    if not path.exists():
        return report.add(label, False, f"missing: {entry['path']}")

    actual = sha256_file(path)
    if actual != entry["sha256"]:
        return report.add(label, False, f"SHA256 mismatch: got {actual[:12]}…")

    with open(path, "rb") as handle:
        scaler = pickle.load(handle)

    problems = []

    if type(scaler).__name__ != entry["class"]:
        problems.append(f"class {type(scaler).__name__} != {entry['class']}")
    if int(scaler.n_features_in_) != entry["n_features_in"]:
        problems.append(f"n_features_in_={scaler.n_features_in_}")
    if int(scaler.n_samples_seen_) != entry["n_samples_seen"]:
        problems.append(f"n_samples_seen_={scaler.n_samples_seen_}")
    if abs(float(np.sum(scaler.mean_)) - entry["mean_sum"]) > STAT_TOLERANCE:
        problems.append("fitted mean_ differs")
    if abs(float(np.sum(scaler.scale_)) - entry["scale_sum"]) > STAT_TOLERANCE:
        problems.append("fitted scale_ differs")

    if problems:
        return report.add(label, False, "; ".join(problems))

    return report.add(
        label,
        True,
        f"{entry['class']}, {entry['n_features_in']} feat, n={entry['n_samples_seen']}",
    )


def check_dataset_artifacts(report, manifest):
    splits = manifest["dataset"]["splits"]

    missing = []
    mismatched = []
    checked = 0

    for split_name, split in splits.items():
        for key in ("X", "Y_multistep", "A_binary_multistep", "A_soft_multistep"):
            entry = split[key]
            path = resolve(entry["path"])
            checked += 1

            if not path.exists():
                missing.append(f"{split_name}/{key}")
                continue

            if sha256_file(path) != entry["sha256"]:
                mismatched.append(f"{split_name}/{key}")

        meta = split["metadata_csv"]
        meta_path = resolve(meta["path"])
        checked += 1

        if not meta_path.exists():
            missing.append(f"{split_name}/metadata")
        elif sha256_file(meta_path) != meta["sha256"]:
            mismatched.append(f"{split_name}/metadata")

    if missing or mismatched:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if mismatched:
            detail.append(f"changed: {', '.join(mismatched)}")
        return report.add("Dataset artifacts", False, " | ".join(detail))

    return report.add("Dataset artifacts", True, f"{checked} files verified")


def check_input_shape(report, manifest):
    spec = manifest["input"]
    expected = (spec["sequence_length"], spec["feature_dimension"])
    splits = manifest["dataset"]["splits"]

    problems = []
    observed = []

    for split_name, split in splits.items():
        path = resolve(split["X"]["path"])

        if not path.exists():
            problems.append(f"{split_name}: missing")
            continue

        array = np.load(path, mmap_mode="r")

        if array.ndim != 3 or tuple(array.shape[1:]) != expected:
            problems.append(f"{split_name}: {tuple(array.shape)}")
            continue

        if array.shape[0] != split["sequences"]:
            problems.append(
                f"{split_name}: {array.shape[0]} sequences != frozen {split['sequences']}"
            )
            continue

        observed.append(f"{split_name}={array.shape[0]}")

    if problems:
        return report.add("Input shape", False, "; ".join(problems))

    return report.add(
        "Input shape",
        True,
        f"{expected[0]} x {expected[1]}; " + ", ".join(observed),
    )


def check_model_loads(report, manifest):
    entry = manifest["checkpoint"]
    arch = manifest["architecture"]
    path = resolve(entry["path"])

    if not path.exists():
        report.add("Model loads (strict)", False, "checkpoint missing")
        return report.add("Model architecture", False, "checkpoint missing")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        report.add("Model loads (strict)", False, "no model_state_dict in checkpoint")
        return report.add("Model architecture", False, "no model_state_dict in checkpoint")

    model = LatentWorldModelV4(
        input_size=arch["input_size"],
        hidden_size=arch["hidden_size"],
        latent_size=arch["latent_dimension"],
        output_size=arch["output_size"],
        num_layers=arch["num_layers"],
        dropout=arch["dropout"],
    )

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as error:
        first_line = str(error).splitlines()[0]
        report.add("Model loads (strict)", False, first_line)
        return report.add("Model architecture", False, "state_dict incompatible")

    report.add("Model loads (strict)", True, "0 missing / 0 unexpected keys")

    # ----- Dimensions must match the manifest -----
    embedded = entry["embedded_metadata"]
    problems = []

    expectations = [
        ("input_size", arch["input_size"]),
        ("hidden_size", arch["hidden_size"]),
        ("latent_size", arch["latent_dimension"]),
        ("output_size", arch["output_size"]),
        ("num_layers", arch["num_layers"]),
        ("sequence_length", manifest["input"]["sequence_length"]),
        ("max_k", manifest["rollout"]["max_k"]),
    ]

    for key, expected_value in expectations:
        if key in checkpoint and checkpoint[key] != expected_value:
            problems.append(f"{key}={checkpoint[key]} != {expected_value}")
        if key in embedded and embedded[key] != expected_value:
            problems.append(f"manifest {key}={embedded[key]} != {expected_value}")

    state_dict = checkpoint["model_state_dict"]

    tensor_expectations = [
        ("encoder_lstm.weight_ih_l0", (4 * arch["hidden_size"], arch["input_size"])),
        ("encoder_fc.weight", (arch["latent_dimension"], arch["hidden_size"])),
        ("transition_mlp.0.weight", (arch["latent_dimension"] * 4, arch["latent_dimension"])),
        ("gate_fc.weight", (arch["latent_dimension"], arch["latent_dimension"])),
        ("decoder_fc2.weight", (arch["output_size"], 64)),
        ("attack_head.3.weight", (1, 32)),
    ]

    for key, expected_shape in tensor_expectations:
        if key not in state_dict:
            problems.append(f"missing tensor {key}")
        elif tuple(state_dict[key].shape) != expected_shape:
            problems.append(f"{key} {tuple(state_dict[key].shape)} != {expected_shape}")

    parameter_count = int(sum(v.numel() for v in state_dict.values()))
    if parameter_count != entry["parameter_count"]:
        problems.append(
            f"{parameter_count} parameters != frozen {entry['parameter_count']}"
        )

    if problems:
        return report.add("Model architecture", False, "; ".join(problems))

    return report.add(
        "Model architecture",
        True,
        f"52→128×2→32→26, {parameter_count} params",
    )


def check_thresholds(report, manifest):
    spec = manifest["threshold"]
    problems = []

    if spec.get("selection_source") != "validation":
        problems.append("selection_source is not 'validation'")

    if spec.get("test_labels_used_for_selection") is not False:
        problems.append("manifest does not assert validation-only selection")

    groups = [
        ("future", spec.get("per_horizon_future_prediction", {})),
        ("trajectory_max", spec.get("per_horizon_trajectory_max", {})),
    ]

    for label, group in groups:
        if not group:
            problems.append(f"{label} thresholds missing")
            continue
        for horizon, value in group.items():
            if not isinstance(value, (int, float)) or not 0.0 < value < 1.0:
                problems.append(f"{label} K={horizon} threshold {value} out of range")

    if problems:
        return report.add("Frozen thresholds", False, "; ".join(problems))

    summary = ", ".join(
        f"K{horizon}={value}"
        for horizon, value in spec["per_horizon_future_prediction"].items()
    )
    return report.add("Frozen thresholds", True, f"validation-only; {summary}")


def check_git(report, manifest):
    expected = manifest["git"]["commit"]

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return report.add("Git revision", False, f"git unavailable: {error}")

    if commit != expected:
        return report.add(
            "Git revision",
            False,
            f"HEAD {commit[:10]} != freeze commit {expected[:10]} "
            "(artifact hashes above are authoritative)",
        )

    return report.add("Git revision", True, commit[:10])


# =============================================================
# MAIN
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Verify the repository against the frozen V4 reference baseline."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to v4_baseline_manifest.json (default: docs/v4_baseline_manifest.json)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    print("=" * 74)
    print("V4 REFERENCE BASELINE — INTEGRITY VERIFICATION (read-only)")
    print("=" * 74)
    print(f"Repository : {REPO_ROOT}")
    print(f"Manifest   : {manifest_path}")

    report = Report()

    if not manifest_path.exists():
        report.add("Manifest", False, f"missing: {manifest_path}")
        report.render()
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        report.add("Manifest", False, f"unparseable: {error}")
        report.render()
        return 1

    report.add(
        "Manifest",
        True,
        f"{manifest['baseline_version']} frozen {manifest['frozen_on_utc']}",
    )

    check_checkpoint_file(report, manifest)
    check_feature_schema(report, manifest)
    check_frozen_schema_copy(report, manifest)
    check_scaler(report, "Input scaler", manifest["scalers"]["input_scaler"])
    check_scaler(report, "Target scaler", manifest["scalers"]["target_scaler"])
    check_dataset_artifacts(report, manifest)
    check_input_shape(report, manifest)
    check_model_loads(report, manifest)
    check_thresholds(report, manifest)
    check_git(report, manifest)

    report.render()

    if manifest.get("known_issues"):
        print()
        print("Known issues recorded at freeze time "
              "(see docs/V4_REFERENCE_BASELINE.md section 12):")
        for issue in manifest["known_issues"]:
            headline = issue if len(issue) <= 110 else issue[:107].rstrip() + "..."
            print(f"  - {headline}")

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
