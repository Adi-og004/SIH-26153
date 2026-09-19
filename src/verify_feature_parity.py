"""
=============================================================
FEATURE PARITY VERIFICATION (READ-ONLY)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Checks that TRAINING, VALIDATION, TEST and PCAP INFERENCE still share one
52-feature contract, and reports honestly when they only share it
approximately.

It never claims exact parity it cannot demonstrate: the window-aggregation
and transform layers are proven by running both real implementations on
identical flow records, while the flow-record production layer (CICFlowMeter
vs packet reconstruction) is reported as APPROXIMATE with its reasons.

Usage:
    python src/verify_feature_parity.py

Exit code: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import feature_contract as fc  # noqa: E402
import run_feature_parity as rfp  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "feature_parity"
SUMMARY_JSON = REPO_ROOT / "results" / "feature_parity_summary.json"
CONTRACT_JSON = REPO_ROOT / "docs" / "feature_contract_v4.json"
FROZEN_NAMES = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_feature_names.txt"
FROZEN_SCALER = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5" / "input_scaler.pkl"
TESTS = REPO_ROOT / "tests" / "test_feature_parity.py"


class Report:
    def __init__(self):
        self.rows, self.notes, self.mismatches, self.failed = [], [], [], False

    def add(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        if not ok:
            self.failed = True
        return ok

    def note(self, text):
        self.notes.append(text)

    def mismatch(self, feature, training, pcap, difference, severity):
        self.mismatches.append(
            {"feature": feature, "training_definition": training, "pcap_definition": pcap,
             "difference": difference, "severity": severity}
        )

    def render(self, overall_status):
        width = max(len(name) for name, _, _ in self.rows)
        print()
        print("FEATURE PARITY VERIFICATION")
        print("-" * 27)
        for name, ok, detail in self.rows:
            line = f"{name:<{width}} : {'PASS' if ok else 'FAIL'}"
            if detail:
                line += f"   ({detail})"
            print(line)
        print()
        print(f"OVERALL: {'FAIL' if self.failed else 'PASS'}")
        print(f"PARITY STATUS: {overall_status}")

        if self.mismatches:
            print()
            print("Semantic differences (reported, not hidden):")
            for entry in self.mismatches:
                print(f"  - {entry['feature']} [{entry['severity']}]")
                print(f"      training: {entry['training_definition']}")
                print(f"      pcap    : {entry['pcap_definition']}")
                print(f"      diff    : {entry['difference']}")
        if self.notes:
            print()
            print("Notes:")
            for note in self.notes:
                print(f"  - {note}")


def main() -> int:
    print("=" * 78)
    print("FEATURE PARITY — VERIFICATION (read-only)")
    print("=" * 78)
    print(f"Repository : {REPO_ROOT}")

    report = Report()

    if not CONTRACT_JSON.exists() or not SUMMARY_JSON.exists():
        report.add("Artifacts present", False, "run src/run_feature_parity.py first")
        report.render("UNKNOWN")
        return 1

    contract = json.loads(CONTRACT_JSON.read_text())
    summary = json.loads(SUMMARY_JSON.read_text())
    frozen_names = FROZEN_NAMES.read_text().split()

    # 1 + 12 + 13. Schema: 52 features, none missing, none unexpected
    names = [feature["name"] for feature in contract["features"]]
    problems = []
    if len(names) != 52:
        problems.append(f"{len(names)} features in the contract")
    missing = [name for name in frozen_names if name not in names]
    unexpected = [name for name in names if name not in frozen_names]
    if missing:
        problems.append(f"missing: {', '.join(missing[:5])}")
    if unexpected:
        problems.append(f"unexpected: {', '.join(unexpected[:5])}")
    report.add("Schema", not problems,
               "; ".join(problems) if problems else "52 features, none missing, none unexpected")

    # 2. Exact order matches the frozen schema
    schema = json.loads((REPO_ROOT / "docs" / "v4_feature_schema.json").read_text())
    order_ok = names == frozen_names == [item["name"] for item in schema["features"]]
    report.add("Feature order", order_ok,
               "identical to input_feature_names.txt and v4_feature_schema.json" if order_ok
               else "ORDER DRIFT between the contract and the frozen schema")

    # 3. 26 base
    base = [f for f in contract["features"] if f["category"] == "base"]
    relative = [f for f in contract["features"] if f["category"] == "relative_change"]
    report.add("Base features", len(base) == 26 and [f["name"] for f in base] == fc.BASE_FEATURES,
               f"{len(base)} base features in frozen order")

    # 4. 26 relative-change, each tied to its base feature
    relative_ok = (
        len(relative) == 26
        and [f["name"] for f in relative] == [f"{b}_rel_change" for b in fc.BASE_FEATURES]
        and all(f["depends_on"] == [f["name"].replace("_rel_change", "")] for f in relative)
    )
    report.add("Relative-change features", relative_ok,
               f"{len(relative)} features, each derived from its base feature")

    # 5. Window semantics: 10 s in every implementation
    import pcap_ingest_v4
    training_module = rfp.load_training_builder()
    window_ok = (
        contract["window_seconds"] == 10
        and training_module.WINDOW_SECONDS == 10
        and pcap_ingest_v4.WINDOW_SECONDS == 10
        and fc.WINDOW_SECONDS == 10
    )
    report.add("Window semantics", window_ok,
               "10 s [t, t+10) floor bins in training, PCAP and the contract")
    if contract["window_semantics"]["timezone"]["parity"] != fc.EXACT:
        report.mismatch(
            "window timestamp anchoring",
            contract["window_semantics"]["timezone"]["training"],
            contract["window_semantics"]["timezone"]["pcap"],
            "identical bin width and boundary rule, different clock anchor",
            "LOW — affects window alignment between a capture and the dataset, not feature values",
        )

    # 6 + 7. Transform semantics
    transform_status = {name: entry["parity"] for name, entry in contract["transform_semantics"].items()}
    non_exact = {name: status for name, status in transform_status.items() if status != fc.EXACT}
    report.add("Transform semantics", not non_exact,
               "; ".join(f"{k}={v}" for k, v in non_exact.items()) if non_exact
               else f"{len(transform_status)} shared transforms, all exactly equivalent")

    # 8. Zero / NaN handling, checked live on the degenerate fixtures
    problems = []
    for fixture_name in ("11_zero_bytes", "12_zero_packets", "13_near_zero_iat", "15_backward_only"):
        fixture = json.loads((REPO_ROOT / "tests" / "fixtures" / "feature_parity"
                              / f"{fixture_name}.json").read_text())
        training_states = rfp.run_training_path(fixture["flows"], training_module)
        pcap_states = rfp.run_pcap_path(fixture["flows"], pcap_ingest_v4)
        for label, frame in (("training", training_states), ("pcap", pcap_states)):
            values = frame[fc.BASE_FEATURES].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                problems.append(f"{fixture_name}/{label}: non-finite value")
        difference = np.abs(
            training_states[fc.BASE_FEATURES].to_numpy(dtype=float)
            - pcap_states[fc.BASE_FEATURES].to_numpy(dtype=float)
        ).max()
        if difference > 1e-9:
            problems.append(f"{fixture_name}: paths differ by {difference:.3g}")
    report.add("Zero/NaN handling", not problems,
               "; ".join(problems[:3]) if problems else
               "4 degenerate fixtures finite and identical in both paths")

    # 9. Scaler interface
    with open(FROZEN_SCALER, "rb") as handle:
        scaler = pickle.load(handle)
    scaler_ok = (
        int(scaler.n_features_in_) == 52
        and list(getattr(scaler, "feature_names_in_", fc.INPUT_FEATURE_NAMES)) == fc.INPUT_FEATURE_NAMES
        and contract["scaler_semantics"]["other_transforms"] == "none — no PCA, no whitening, no feature selection anywhere in either path"
    )
    report.add("Scaler interface", scaler_ok,
               f"StandardScaler, 52 named columns in contract order, no other transform")

    # 10. Golden tests
    completed = subprocess.run([sys.executable, str(TESTS)], cwd=REPO_ROOT,
                               capture_output=True, text=True)
    passed = sum(1 for line in completed.stdout.splitlines() if line.startswith("PASS "))
    failed = sum(1 for line in completed.stdout.splitlines() if line.startswith("FAIL "))
    report.add("Golden tests", completed.returncode == 0 and failed == 0,
               f"{passed} passed, {failed} failed"
               + ("" if passed else " — tests/test_feature_parity.py produced no result"))

    # 11. Synthetic training-vs-PCAP parity, re-run live
    parity_frame, fixtures = rfp.run_parity_tests(verbose=False)
    failures = parity_frame[~parity_frame["pass"]]
    report.add("Synthetic parity", len(failures) == 0,
               f"{len(fixtures)} fixtures, {len(parity_frame)} comparisons, {len(failures)} mismatches")
    for _, row in failures.iterrows():
        report.mismatch(row["feature"], "see contract", "see contract",
                        f"{row['fixture']}: |Δ| = {row['absolute_difference']:.6g}", "HIGH")

    # 14. Contract hash
    recorded = (RESULTS_DIR / "feature_contract_sha256.txt").read_text().strip()
    live = fc.contract_hash(fc.build_contract())
    hash_ok = recorded == live == contract.get("contract_sha256") == summary.get("contract_hash")
    report.add("Contract hash", hash_ok,
               f"{live[:16]}… consistent across contract, hash file and summary" if hash_ok
               else "hash mismatch — the contract changed without regenerating its artifacts")

    # 15. The production PCAP path consumes the contract rather than hardcoding it
    pcap_ok = (
        pcap_ingest_v4.INPUT_FEATURE_NAMES == fc.INPUT_FEATURE_NAMES
        and pcap_ingest_v4.BASE_FEATURES == fc.BASE_FEATURES
        and pcap_ingest_v4.INPUT_FEATURE_NAMES_PATH == FROZEN_NAMES
    )
    conformance = RESULTS_DIR / "pcap_conformance.csv"
    conforming = ""
    if conformance.exists():
        frame = pd.read_csv(conformance)
        conforming = f", {int((frame['status'] == 'CONFORMS').sum())}/{len(frame)} sample PCAPs conform"
        pcap_ok = pcap_ok and bool((frame["status"] == "CONFORMS").all())
    report.add("PCAP contract", pcap_ok,
               f"pcap_ingest_v4 reads the 52 names from the frozen file{conforming}")

    # ---- flow-layer parity is reported, never asserted away ----
    approximate = [f["name"] for f in contract["features"] if f["flow_layer_parity"] == fc.APPROXIMATE]
    non_parity = [f["name"] for f in contract["features"] if f["flow_layer_parity"] == fc.NON_PARITY]
    unknown = [f["name"] for f in contract["features"] if f["flow_layer_parity"] == fc.UNKNOWN]
    report.add("Flow-layer parity reported", True,
               f"{len(approximate)} APPROXIMATE, {len(non_parity)} NON_PARITY, {len(unknown)} UNKNOWN")

    for feature in contract["features"]:
        if feature["category"] == "base" and feature["flow_layer_parity"] != fc.EXACT:
            severity = "MEDIUM"
            if feature["name"] in ("syn_rate", "fin_rate"):
                severity = "HIGH — measured distributions diverge by several training sigma"
            elif feature["name"] in ("flow_count", "log_flow_count", "log_unique_dst_ports",
                                     "unique_protocols", "dst_port_entropy", "protocol_entropy"):
                severity = "HIGH — capture-scope dependent"
            report.mismatch(feature["name"], feature["training_formula"][:110],
                            feature["pcap_formula"][:110], feature["parity_note"][:150], severity)

    overall_status = summary["overall_status"] if not report.failed else "PARITY_NOT_ESTABLISHED"

    report.note("EXACT parity is demonstrated for the window aggregation, the transforms, the "
                "relative-change stage and the scaler interface — identical flow records produce "
                "identical 52-feature states in both implementations.")
    report.note("APPROXIMATE parity only for the flow-record production layer: CICFlowMeter (Java, "
                "FIN/RST termination) versus packet reconstruction (120 s idle split). Numerical "
                "equality between a dataset window and a PCAP window is NOT claimed.")
    report.note("Capture scope is the largest practical gap: training aggregates the whole victim "
                "subnet, a demo capture covers only the hosts recorded.")
    report.note("pytest is not installed in venv/; tests/test_feature_parity.py runs standalone "
                "(python tests/test_feature_parity.py) and is executed by this verifier.")

    report.render(overall_status)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
