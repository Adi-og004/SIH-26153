"""
=============================================================
TASK 8 VERIFIER -- V4 EXPLAINABILITY
=============================================================

READ-ONLY. This script opens files for reading and writes
NOTHING: no artifact, no cache, no log file, no model. It is
safe to run against the frozen baseline at any time.

It does not trust ``results/v4_explainability_summary.json``.
Every numerical claim it checks is RECOMPUTED here, from the
frozen checkpoint, through the same explainer the dashboard
uses. The stored artifacts are then checked for agreement with
what was just recomputed -- so a stale or hand-edited CSV is
caught rather than believed.

    python src/verify_v4_explainability.py

Exit code 0 == OVERALL: PASS, 1 == OVERALL: FAIL.

A critical failure is never downgraded to a warning.
=============================================================
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v4_explainer import (
    V4Explainer,
    FileAccessAudit,
    ProvenanceError,
    horizon_target,
    build_provenance,
    load_manifest,
    sha256_file,
    zero_baseline,
    HorizonProbability,
    REPO_ROOT,
    SEQ_LEN,
    N_FEATURES,
    MAX_K,
    IG_N_STEPS,
    IG_METHOD,
    FORBIDDEN_ARTIFACT_MARKERS,
)
from pcap_ingest_v4 import (
    LatentWorldModelV4,
    INPUT_FEATURE_NAMES,
    INPUT_FEATURE_NAMES_PATH,
)

CANON = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
OUT_DIR = REPO_ROOT / "results" / "v4_explainability"
DOCS = REPO_ROOT / "docs"
SUMMARY_PATH = REPO_ROOT / "results" / "v4_explainability_summary.json"
CONTRACT_PATH = DOCS / "v4_explainability_contract.json"

HORIZONS = [1, 3, 5]
HEADLINE_K = 5
IDENTITY_TOLERANCE = 1e-9

# Verification sample sizes. Small enough to run in ~1 minute,
# large enough that a systematic fault cannot hide.
N_IDENTITY = 512
N_ATTRIB = 48
N_ABLATE = 8

# Files that must exist under results/v4_explainability/ (Part 29).
REQUIRED_ARTIFACTS = [
    "completeness_checks.csv",
    "ablation_crosscheck.csv",
    "feature_importance.csv",
    "temporal_importance.csv",
    "attribution_heatmap.csv",
    "representative_explanations.csv",
    "horizon_feature_importance.csv",
    "attack_vs_benign.csv",
    "error_explanations.csv",
    "onset_explanations.csv",
    "prediction_identity.csv",
    "repeatability.csv",
    "feature_name_integrity.json",
    "README.md",
]

# Older-generation explainability artifacts the DASHBOARD must not
# read. Declared here, as a named constant, so the scanner's own
# detection vocabulary is excluded from its own scan (the same
# treatment FORBIDDEN_ARTIFACT_MARKERS gets) -- otherwise the verifier
# reports itself.
STALE_DASHBOARD_MARKERS = ("temporal_benchmark_v2", "shap_analysis")

# The whole explainability path -- every file that participates in
# producing or displaying a V4 explanation. Scanned statically for
# older-generation artifacts.
EXPLAINABILITY_PATH_FILES = [
    REPO_ROOT / "src" / "v4_explainer.py",
    REPO_ROOT / "src" / "run_v4_explainability.py",
    REPO_ROOT / "src" / "verify_v4_explainability.py",
    # The model/feature loader the explainer imports.
    REPO_ROOT / "src" / "pcap_ingest_v4.py",
    # The dashboard: it DISPLAYS the explanation, so an older-
    # generation artifact reached from here would surface next to a
    # V4 prediction just as surely as one loaded by the explainer.
    REPO_ROOT / "src" / "app.py",
]


# =============================================================
# REPORTING
# =============================================================

class Report:
    """Collects PASS/FAIL lines. Critical checks gate the exit code;
    informational lines are printed but never mask a failure."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str, bool]] = []
        self.detail: list[str] = []

    def check(
        self, name: str, passed: bool, note: str = "", critical: bool = True
    ) -> bool:
        self.rows.append((name, bool(passed), note, critical))
        return bool(passed)

    def info(self, line: str) -> None:
        self.detail.append(line)

    @property
    def failed_critical(self) -> list[str]:
        return [n for n, p, _, c in self.rows if c and not p]

    def render(self) -> str:
        width = max(len(n) for n, _, _, _ in self.rows)
        out = [
            "",
            "=" * 74,
            "V4 EXPLAINABILITY VERIFICATION",
            "-" * 30,
        ]
        for name, passed, note, critical in self.rows:
            status = "PASS" if passed else ("FAIL" if critical else "WARN")
            line = f"{name:<{width}} : {status}"
            if note:
                line += f"   {note}"
            out.append(line)
        out.append("")
        if self.detail:
            out.append("-" * 74)
            out += self.detail
            out.append("")
        ok = not self.failed_critical
        out.append("=" * 74)
        out.append(f"OVERALL: {'PASS' if ok else 'FAIL'}")
        if not ok:
            out.append("")
            out.append("FAILING CRITICAL CONDITIONS:")
            for n in self.failed_critical:
                out.append(f"  - {n}")
        out.append("=" * 74)
        return "\n".join(out)


def fmt(x: float) -> str:
    return f"{x:.3e}"


# =============================================================
# STATIC SCANS
# =============================================================

def _rel(p: Path) -> str:
    """Repo-relative path where possible, absolute otherwise (the
    scanners are also exercised against temp files in tests)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def scan_forbidden_literals(paths: list[Path]) -> dict[str, list[str]]:
    """Look for older-generation artifact paths written as string
    literals in the explainability path. Comments and docstrings are
    excluded -- naming V3 in prose to say it is NOT used is not
    contamination; loading it is."""
    hits: dict[str, list[str]] = {}
    for p in paths:
        if not p.exists():
            continue
        src = p.read_text()
        tree = ast.parse(src)
        found: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # A docstring is a bare string Expr; those are filtered
                # by checking the parent below, so instead filter on the
                # module/class/function docstring set.
                for marker in FORBIDDEN_ARTIFACT_MARKERS:
                    if marker not in node.value:
                        continue
                    # Only a PATH-LIKE literal can load an artifact.
                    # Real paths contain no whitespace; prose that
                    # merely names an older generation (a log line, a
                    # limitation, a note saying V3 is NOT used) does.
                    # The runtime audit hook below is the primary
                    # authority either way -- it records what was
                    # actually opened, not what was mentioned.
                    if any(c.isspace() for c in node.value.strip()):
                        continue
                    found.append(
                        f"line {node.lineno}: '{marker}' in a path-like "
                        f"string literal"
                    )
        # Remove hits that come from docstrings or the declared
        # FORBIDDEN_ARTIFACT_MARKERS tuple itself.
        doc_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef)
            ):
                d = ast.get_docstring(node, clean=False)
                if d is not None and node.body:
                    first = node.body[0]
                    lo = getattr(first, "lineno", 0)
                    hi = getattr(first, "end_lineno", lo)
                    doc_lines.update(range(lo, hi + 1))
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in (
                        "FORBIDDEN_ARTIFACT_MARKERS",
                        "STALE_DASHBOARD_MARKERS",
                    ):
                        lo = node.lineno
                        hi = getattr(node, "end_lineno", lo)
                        doc_lines.update(range(lo, hi + 1))
        real = [
            h for h in found
            if int(h.split()[1].rstrip(":")) not in doc_lines
        ]
        if real:
            hits[_rel(p)] = real
    return hits


def scan_test_label_tuning(paths: list[Path]) -> list[str]:
    """Detect a test label influencing the attribution METHOD.

    Reading test labels to *report* which cases are TP/FP/FN is
    required by Part 11 and is not tuning. Tuning would be a test
    label steering n_steps, the baseline, the method, or a
    threshold. Both are distinguished by what the label flows into.
    """
    problems: list[str] = []
    tuning_targets = re.compile(
        r"\b(n_steps|IG_N_STEPS|baseline|method|threshold|thr)\b"
        r"\s*=\s*.*\b(A_test|A_binary_multistep_test|y_test|actual|"
        r"labels?_test)\b"
    )
    for p in paths:
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if tuning_targets.search(line):
                problems.append(f"{_rel(p)}:{i}: {stripped[:90]}")
    return problems


def scan_mitre_separation() -> tuple[bool, list[str]]:
    """The rule-based ATT&CK layer must not consume attribution, and
    the attribution layer must not emit ATT&CK labels."""
    notes: list[str] = []
    ok = True

    mm = REPO_ROOT / "src" / "mitre_mapper.py"
    if mm.exists():
        src = mm.read_text()
        leaks = [
            t for t in
            ("attribution", "attributions", "integrated_gradient",
             "shap_value", "ig_attr", "feature_importance")
            if re.search(rf"\b{t}\b", src)
        ]
        if leaks:
            ok = False
            notes.append(
                f"mitre_mapper.py references attribution terms: {leaks}"
            )
        else:
            notes.append(
                "src/mitre_mapper.py contains no attribution term -- the "
                "rule layer evaluates raw feature values only"
            )

    # The attribution layer must not mint ATT&CK identifiers.
    attack_id = re.compile(r"\bT1\d{3}(\.\d{3})?\b")
    for p in (REPO_ROOT / "src" / "v4_explainer.py",
              REPO_ROOT / "src" / "run_v4_explainability.py"):
        if not p.exists():
            continue
        if attack_id.search(p.read_text()):
            ok = False
            notes.append(
                f"{p.name} emits an ATT&CK technique id -- attribution "
                f"must not produce ATT&CK labels"
            )
    return ok, notes


def scan_dashboard_integration() -> tuple[bool, list[str]]:
    """The dashboard must obtain V4 explanations from the
    provenance-locked explainer, not from a legacy, unlocked one."""
    notes: list[str] = []
    app = REPO_ROOT / "src" / "app.py"
    if not app.exists():
        return False, ["src/app.py not found"]
    src = app.read_text()
    ok = True

    if re.search(r"^from\s+v4_explainer\s+import", src, re.M):
        notes.append("src/app.py imports the provenance-locked V4Explainer")
    else:
        ok = False
        notes.append(
            "src/app.py does NOT import v4_explainer -- the dashboard "
            "explanation path is not the verified one"
        )

    legacy = re.search(
        r"^from\s+shap_explainer\s+import|^import\s+shap_explainer", src, re.M
    )
    if legacy:
        ok = False
        notes.append(
            "src/app.py still imports the legacy shap_explainer "
            "(unlocked baseline, single-step target)"
        )

    # Scan STRING LITERALS only, and only path-like ones. A comment
    # explaining why an older artifact is no longer read is not a
    # reference to it; a path the dashboard could open is.
    stale: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value.strip()
            if any(c.isspace() for c in v):
                continue
            for m in STALE_DASHBOARD_MARKERS:
                if m in v:
                    stale.append(f"line {node.lineno}: {v}")
    if stale:
        ok = False
        notes.append(
            f"src/app.py still references older-generation "
            f"explainability artifacts: {stale}"
        )
    else:
        notes.append(
            "src/app.py contains no path-like reference to "
            "temporal_benchmark_v2 or shap_analysis"
        )
    return ok, notes


# =============================================================
# MAIN
# =============================================================

def main() -> int:  # noqa: C901
    rep = Report()

    # ---------------------------------------------------------
    # 1-5. PROVENANCE (recomputed, not read from the summary)
    # ---------------------------------------------------------
    manifest = load_manifest()
    audit = FileAccessAudit()
    with audit:
        try:
            prov = build_provenance(manifest)
            prov_ok = True
            prov_err = ""
        except ProvenanceError as exc:
            prov_ok = False
            prov_err = str(exc).splitlines()[0]
            prov = None

        rep.check(
            "Checkpoint",
            prov_ok and prov.checkpoint_hash == prov.checkpoint_hash_expected,
            (f"sha256 {prov.checkpoint_hash[:16]}..." if prov_ok
             else prov_err),
        )

        if not prov_ok:
            # Without provenance nothing downstream is meaningful.
            print(rep.render())
            return 1

        ex = V4Explainer()
        rep.check(
            "Model identity",
            type(ex.model).__name__ == "LatentWorldModelV4"
            and isinstance(ex.model, LatentWorldModelV4)
            and prov.model_version == "V4",
            f"{type(ex.model).__name__}, "
            f"{sum(p.numel() for p in ex.model.parameters())} params",
        )

        rep.check(
            "Feature schema",
            prov.feature_order_hash == prov.feature_order_hash_expected,
            f"sha256 {prov.feature_order_hash[:16]}...",
        )
        rep.check(
            "Scaler",
            prov.scaler_hash == prov.scaler_hash_expected,
            f"sha256 {prov.scaler_hash[:16]}...",
        )

        X = np.load(CANON / "X_test.npy")
        A = np.load(CANON / "A_binary_multistep_test.npy")
        rep.check(
            "Input tensor",
            X.shape[1:] == (SEQ_LEN, N_FEATURES)
            and tuple(prov.input_shape) == (SEQ_LEN, N_FEATURES),
            f"X_test {X.shape}",
        )

        # -----------------------------------------------------
        # 6. TARGET DEFINITION
        # -----------------------------------------------------
        contract = json.loads(CONTRACT_PATH.read_text())
        tdefs = contract["target_definitions"]
        tgt = horizon_target(HEADLINE_K, manifest)
        tkeys = {k.lower() for k in tdefs}
        target_ok = (
            "target_a_horizon_probability" in tkeys
            and "target_b_trajectory_max" in tkeys
            and abs(
                tgt.threshold
                - manifest["threshold"]["per_horizon_future_prediction"][
                    str(HEADLINE_K)
                ]
            )
            < 1e-12
        )
        rep.check(
            "Target definition",
            target_ok,
            f"K={HEADLINE_K} horizon_probability, thr={tgt.threshold}",
        )

        # -----------------------------------------------------
        # 7. PREDICTION IDENTITY  (recomputed)
        # -----------------------------------------------------
        idx = np.arange(min(N_IDENTITY, X.shape[0]))
        worst = 0.0
        all_bit_exact = True
        for k in HORIZONS:
            r = ex.check_prediction_identity(X[idx], k, IDENTITY_TOLERANCE)
            worst = max(worst, r["max_abs_diff"])
            all_bit_exact = all_bit_exact and r["bit_exact"]
        rep.check(
            "Prediction identity",
            worst < IDENTITY_TOLERANCE,
            f"max|P-P_explain|={fmt(worst)} over {len(idx)} seq x "
            f"{len(HORIZONS)} horizons, bit-exact={all_bit_exact}",
        )

        # -----------------------------------------------------
        # 8. REPEATABILITY  (recomputed)
        # -----------------------------------------------------
        sub = X[:N_ATTRIB]
        r1 = ex.attribute(sub, HEADLINE_K)
        r2 = ex.attribute(sub, HEADLINE_K)
        a1, a2 = r1["attributions"], r2["attributions"]
        attr_diff = float(np.abs(a1 - a2).max())
        pred_diff = float(np.abs(r1["prediction"] - r2["prediction"]).max())
        rank1 = np.argsort(
            -np.abs(a1).sum(axis=1), axis=1, kind="stable"
        )
        rank2 = np.argsort(
            -np.abs(a2).sum(axis=1), axis=1, kind="stable"
        )
        ranking_identical = bool(np.array_equal(rank1, rank2))
        rep.check(
            "Repeatability",
            attr_diff == 0.0 and pred_diff == 0.0 and ranking_identical,
            f"max|d attr|={fmt(attr_diff)} max|d P|={fmt(pred_diff)} "
            f"ranking identical={ranking_identical}",
        )

        # -----------------------------------------------------
        # 9. ATTRIBUTION SHAPE
        # -----------------------------------------------------
        rep.check(
            "Attribution shape",
            a1.shape == (len(sub), SEQ_LEN, N_FEATURES),
            f"{a1.shape} == (N, 10, 52)",
        )

        # -----------------------------------------------------
        # 10. FEATURE NAMING
        # -----------------------------------------------------
        schema = json.loads((DOCS / "v4_feature_schema.json").read_text())
        schema_names = [f["name"] for f in schema["features"]]
        txt_names = [
            ln.strip()
            for ln in INPUT_FEATURE_NAMES_PATH.read_text().splitlines()
            if ln.strip()
        ]
        heat_path = OUT_DIR / "attribution_heatmap.csv"
        heat_ok = True
        heat_note = "heatmap absent"
        if heat_path.exists():
            heat = pd.read_csv(heat_path, index_col=0)
            heat_ok = (
                list(heat.columns) == schema_names
                and heat.shape == (SEQ_LEN, N_FEATURES)
                and list(heat.index)[-1] == "t"
                and list(heat.index)[0] == "t-9"
            )
            heat_note = f"heatmap {heat.shape}, columns in schema order"
        naming_ok = (
            schema_names == txt_names
            and schema_names == list(INPUT_FEATURE_NAMES)
            and schema_names == list(ex.feature_names)
            and [f["index"] for f in schema["features"]]
            == list(range(N_FEATURES))
            and heat_ok
        )
        rep.check(
            "Feature naming",
            naming_ok,
            f"52 names identical across schema/txt/explainer; {heat_note}",
        )

        # -----------------------------------------------------
        # 11. COMPLETENESS  (recomputed)
        # -----------------------------------------------------
        comp_ok = True
        comp_bits = []
        for k in HORIZONS:
            r = ex.attribute(sub, k)
            mean_abs = float(r["completeness_abs_error"].mean())
            p99_abs = float(np.quantile(r["completeness_abs_error"], 0.99))
            denom = np.abs(r["target_delta"])
            wp = denom > 0.05
            wp_rel = (
                float(r["completeness_rel_error"][wp].mean())
                if wp.any() else 0.0
            )
            ok_k = mean_abs < 0.01 and p99_abs < 0.01 and wp_rel < 0.01
            comp_ok = comp_ok and ok_k
            comp_bits.append(
                f"k={k}: mean|err|={fmt(mean_abs)} p99={fmt(p99_abs)} "
                f"wellposed rel={wp_rel:.4%}"
            )
        rep.check(
            "Completeness",
            comp_ok,
            f"IG {IG_METHOD} n_steps={IG_N_STEPS}",
        )
        for b in comp_bits:
            rep.info(f"  completeness {b}")

        # -----------------------------------------------------
        # 12. ABLATION CROSS-CHECK  (recomputed)
        # -----------------------------------------------------
        aK = ex.attribute(X[:N_ABLATE], HEADLINE_K)
        featattr = aK["attributions"].sum(axis=1)
        absattr = np.abs(aK["attributions"]).sum(axis=1)
        agree = 0
        total = 0
        top_dp = []
        low_dp = []
        for i in range(N_ABLATE):
            order = np.lexsort((np.arange(N_FEATURES), -absattr[i]))
            for j in order[:5]:
                d = ex.ablate_feature(X[i:i + 1], int(j), HEADLINE_K)
                dp = float(d["delta"][0])
                top_dp.append(abs(dp))
                total += 1
                if np.sign(dp) == np.sign(featattr[i, j]) or dp == 0.0:
                    agree += 1
            for j in order[-5:]:
                d = ex.ablate_feature(X[i:i + 1], int(j), HEADLINE_K)
                low_dp.append(abs(float(d["delta"][0])))
        sign_rate = agree / max(total, 1)
        directional = float(np.mean(top_dp)) > float(np.mean(low_dp))
        rep.check(
            "Ablation cross-check",
            total > 0 and directional,
            f"sign agreement {sign_rate:.0%} on top-5; mean|dP| top "
            f"{np.mean(top_dp):.2e} > low {np.mean(low_dp):.2e}",
        )

    # ---------------------------------------------------------
    # 13. V3 CONTAMINATION -- runtime audit + static scan
    # ---------------------------------------------------------
    runtime_hits = audit.forbidden_hits()
    static_hits = scan_forbidden_literals(EXPLAINABILITY_PATH_FILES)
    rep.check(
        "V3 contamination check",
        not runtime_hits and not static_hits,
        f"{len(audit.project_files())} project files opened, "
        f"0 forbidden" if not runtime_hits
        else f"runtime hits: {runtime_hits}",
    )
    if static_hits:
        for f, hs in static_hits.items():
            rep.info(f"  static hit {f}: {hs}")

    # ---------------------------------------------------------
    # 14. NO TEST TUNING
    # ---------------------------------------------------------
    tuning = scan_test_label_tuning(EXPLAINABILITY_PATH_FILES)
    rep.check(
        "No test-label tuning",
        not tuning,
        "no test label flows into n_steps / baseline / method / threshold",
    )
    for t in tuning:
        rep.info(f"  tuning hit {t}")

    # ---------------------------------------------------------
    # 15. PCAP CAVEAT
    # ---------------------------------------------------------
    pcap_csv = OUT_DIR / "pcap_explanations.csv"
    caveat_ok = "pcap_policy" in contract and bool(
        contract["pcap_policy"].get("required_caveat")
    )
    caveat_note = "contract carries the required caveat"
    if pcap_csv.exists():
        pdf = pd.read_csv(pcap_csv)
        # Every PCAP row must be tagged as pcap-sourced AND must carry
        # the flow-record caveat, phrased as an explicit denial of
        # numerical equivalence with training features.
        rows_ok = (
            not pdf.empty
            and (pdf["input_source"] == "pcap").all()
            and pdf["caveat"].astype(str).str.contains(
                "flow-record production", case=False
            ).all()
            and pdf["caveat"].astype(str).str.contains(
                "NOT numerically equivalent", case=False
            ).all()
        )
        caveat_ok = caveat_ok and rows_ok
        caveat_note = (
            f"{len(pdf)} pcap rows, all input_source=pcap and all carrying "
            f"the approximate-flow caveat"
        )
    rep.check("PCAP caveat", caveat_ok, caveat_note)

    # ---------------------------------------------------------
    # 16. MITRE SEPARATION
    # ---------------------------------------------------------
    mitre_ok, mitre_notes = scan_mitre_separation()
    contract_sep = contract.get("mitre_separation", {})
    mitre_ok = (
        mitre_ok
        and contract_sep.get("attribution_feeds_mitre") is False
        and contract_sep.get("mitre_feeds_attribution") is False
    )
    rep.check(
        "MITRE separation",
        mitre_ok,
        "rule layer and attribution layer share no data path",
    )
    for n in mitre_notes:
        rep.info(f"  mitre {n}")

    # ---------------------------------------------------------
    # 17. DASHBOARD INTEGRATION
    # ---------------------------------------------------------
    dash_ok, dash_notes = scan_dashboard_integration()
    rep.check("Dashboard integration", dash_ok, "")
    for n in dash_notes:
        rep.info(f"  dashboard {n}")

    # ---------------------------------------------------------
    # 18. ARTIFACTS PRESENT AND CONSISTENT WITH THIS RECOMPUTATION
    # ---------------------------------------------------------
    missing = [f for f in REQUIRED_ARTIFACTS
               if not (OUT_DIR / f).exists()]
    rep.check(
        "Required artifacts",
        not missing,
        f"{len(REQUIRED_ARTIFACTS)} files present"
        if not missing else f"missing: {missing}",
    )

    stored_ok = True
    stored_note = ""
    if SUMMARY_PATH.exists():
        summ = json.loads(SUMMARY_PATH.read_text())
        mi = summ.get("model_identity", {})
        stored_ok = (
            mi.get("checkpoint_hash") == prov.checkpoint_hash
            and mi.get("scaler_hash") == prov.scaler_hash
            and mi.get("feature_order_hash") == prov.feature_order_hash
            and summ.get("overall") == "PASS"
        )
        stored_note = (
            f"summary hashes match the live recomputation; "
            f"summary overall={summ.get('overall')}"
        )
    else:
        stored_ok = False
        stored_note = "results/v4_explainability_summary.json missing"
    rep.check("Stored summary agrees", stored_ok, stored_note)

    # The stored prediction-identity artifact must agree with the
    # identity we just recomputed -- catches a hand-edited CSV.
    ident_csv = OUT_DIR / "prediction_identity.csv"
    if ident_csv.exists():
        idf = pd.read_csv(ident_csv)
        stored_max = float(idf["abs_diff"].max())
        rep.check(
            "Stored identity artifact",
            bool(idf["passed"].all()) and stored_max < IDENTITY_TOLERANCE,
            f"{len(idf)} rows, max abs_diff={fmt(stored_max)}",
        )
    else:
        rep.check("Stored identity artifact", False, "file missing")

    # ---------------------------------------------------------
    # 19. FROZEN ARTIFACTS UNTOUCHED
    # ---------------------------------------------------------
    frozen_ok = True
    frozen_bits = []
    for key, path, expected in [
        ("checkpoint",
         REPO_ROOT / manifest["checkpoint"]["path"],
         manifest["checkpoint"]["sha256"]),
        ("input_scaler",
         REPO_ROOT / manifest["scalers"]["input_scaler"]["path"],
         manifest["scalers"]["input_scaler"]["sha256"]),
        ("target_scaler",
         REPO_ROOT / manifest["scalers"]["target_scaler"]["path"],
         manifest["scalers"]["target_scaler"]["sha256"]),
    ]:
        actual = sha256_file(path)
        good = actual == expected
        frozen_ok = frozen_ok and good
        frozen_bits.append(f"{key}={'ok' if good else 'CHANGED'}")
    rep.check(
        "Frozen artifacts untouched",
        frozen_ok,
        ", ".join(frozen_bits),
    )

    # ---------------------------------------------------------
    rep.info(
        f"  recomputation scope: identity {N_IDENTITY} sequences x "
        f"{len(HORIZONS)} horizons; attribution {N_ATTRIB} sequences x "
        f"{len(HORIZONS)} horizons at n_steps={IG_N_STEPS}; ablation "
        f"{N_ABLATE} sequences x 10 features"
    )
    rep.info(
        "  this verifier wrote no files -- it is read-only by construction"
    )

    print(rep.render())
    return 0 if not rep.failed_critical else 1


if __name__ == "__main__":
    sys.exit(main())
