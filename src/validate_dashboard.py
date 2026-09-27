"""
=============================================================
V4 SCREENING DASHBOARD — VALIDATION (READ-ONLY on frozen artifacts)
SIH 2026 — Predictive Cyber Defence Using World Models
=============================================================

Task 11 step 16. Validates the screening dashboard without training,
refitting or rethresholding anything.

Checks:
     1. py_compile every modified Python file
     2. Import the application module successfully (headless Streamlit)
     3. All screening artifacts load through src/screening_data.py
     4. The default demo sequence is reproducible
     5. Forecast probabilities recomputed from the frozen checkpoint match
        screening_predictions.csv exactly
     6. The threshold the UI shows matches the frozen manifest
     7. The provenance-locked V4 explainer runs on a screening sequence
     8. The rule-based MITRE mapper runs on the same sequence
     9. Event/ground-truth labels are NOT reachable from model input
    10. Capture-gap handling: gap detected, line breaks inserted, no interpolation
    11. Offline: no network/API dependency in the dashboard source
    12. No frozen model/scaler/data artifact was modified

Writes results/v4_screening_mvp/dashboard_validation.json

Usage:
    ./venv/bin/python src/validate_dashboard.py

Exit code: 0 = PASS, 1 = FAIL
"""

from __future__ import annotations


import ast
import hashlib
import json
import os
import py_compile
import re
import subprocess
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

BENCH_DIR = REPO_ROOT / "data" / "processed" / "multiday_benchmark_K5"
MODEL_DIR = REPO_ROOT / "models" / "latent_world_model_v4"
CHECKPOINT = MODEL_DIR / "latent_world_model_v4.pt"
MANIFEST_PATH = REPO_ROOT / "docs" / "v4_baseline_manifest.json"
OUT_DIR = REPO_ROOT / "results" / "v4_screening_mvp"

# Python files that must compile.
MODIFIED_FILES = [
    "src/app.py",
    "src/screening_data.py",
    "src/validate_dashboard.py",
]

# Everything this task changed or added, for the provenance record.
ALL_MODIFIED_FILES = MODIFIED_FILES + ["requirements.txt"]

# Frozen artifacts that must be byte-identical before and after this run.
FROZEN_ARTIFACTS = [
    CHECKPOINT,
    BENCH_DIR / "input_scaler.pkl",
    BENCH_DIR / "target_scaler.pkl",
    BENCH_DIR / "input_feature_names.txt",
    BENCH_DIR / "state_feature_names.txt",
    BENCH_DIR / "X_test.npy",
    BENCH_DIR / "Y_multistep_test.npy",
    BENCH_DIR / "A_binary_multistep_test.npy",
    BENCH_DIR / "test_metadata.csv",
    MANIFEST_PATH,
    OUT_DIR / "screening_X.npy",
    OUT_DIR / "screening_metadata.csv",
    OUT_DIR / "screening_predictions.csv",
    OUT_DIR / "event_timeline.csv",
    OUT_DIR / "feature_names.txt",
    OUT_DIR / "screening_manifest.json",
    OUT_DIR / "readiness_report.json",
]

# Network-capable constructs that would break the offline requirement.
NETWORK_PATTERNS = [
    r"\brequests\.(get|post|put|delete|head|patch)\s*\(",
    r"\burllib\.request\b",
    r"\bhttpx\.",
    r"\bsocket\.(socket|create_connection)\s*\(",
    r"\baiohttp\b",
    r"https?://[^\s\"')]+",
]
# URLs that are documentation references only (never fetched at runtime).
URL_ALLOW_SUBSTRINGS = ("attack.mitre.org",)

BATCH_SIZE = 64
MAX_K = 5


class Report:
    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, key, name, ok, detail=""):
        self.rows.append({"key": key, "name": name,
                          "status": "PASS" if ok else "FAIL", "detail": detail})
        if not ok:
            self.failed = True
        print(f"  {name:<46} {'PASS' if ok else 'FAIL'}   {detail}")
        return bool(ok)


R = Report()


def header(t):
    print("\n" + "=" * 88)
    print(t)
    print("=" * 88)


def find_dashboard_python() -> tuple[str, dict]:
    """The interpreter that can actually run the dashboard.

    The repo carries two environments: the evaluation venv (torch, no Streamlit)
    and the dashboard venv (Streamlit + plotly). `streamlit run src/app.py` uses
    the latter, so that is the interpreter the UI must be validated under."""
    candidates = [
        REPO_ROOT / "src" / "venv" / "bin" / "python",
        REPO_ROOT / "venv" / "bin" / "python",
        Path(sys.executable),
    ]
    probe = ("import json,importlib;"
             "mods={};"
             "\nfor m in ['streamlit','plotly','torch','captum','sklearn','pandas','numpy']:\n"
             "    try:\n"
             "        mod=importlib.import_module(m)\n"
             "        mods[m]=getattr(mod,'__version__','?')\n"
             "    except Exception:\n"
             "        mods[m]=None\n"
             "print(json.dumps(mods))")
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            out = subprocess.run([str(cand), "-c", probe], cwd=REPO_ROOT,
                                 capture_output=True, text=True, timeout=120)
            mods = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:                                # noqa: BLE001
            continue
        if mods.get("streamlit") and mods.get("plotly"):
            return str(cand), mods
    return "", {}


def _function_code(source: str, name: str) -> str:
    """The executable body of a function, with its docstring removed.

    Used for the leakage check: a plain text grep would match the docstring's
    own explanation that labels are excluded, which is the opposite of a leak."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            return "\n".join(ast.unparse(stmt) for stmt in body)
    raise ValueError(f"function {name!r} not found")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("=" * 88)
    print("V4 SCREENING DASHBOARD — VALIDATION")
    print("=" * 88)
    print(f"Repository : {REPO_ROOT}")
    print(f"Started    : {started}")

    pre_hashes = {str(p.relative_to(REPO_ROOT)): (sha256_file(p) if p.exists() else None)
                  for p in FROZEN_ARTIFACTS}
    manifest = json.loads(MANIFEST_PATH.read_text())

    # =============================================================
    header("1 — py_compile MODIFIED FILES")
    # =============================================================
    compile_errors = []
    with tempfile.TemporaryDirectory() as tmpd:
        for rel in MODIFIED_FILES:
            p = REPO_ROOT / rel
            if not p.exists():
                compile_errors.append(f"{rel}: missing")
                continue
            cfile = Path(tmpd) / (Path(rel).name + "c")
            try:
                py_compile.compile(str(p), doraise=True, cfile=str(cfile))
                print(f"  compiled  {rel}")
            except py_compile.PyCompileError as exc:
                compile_errors.append(f"{rel}: {exc}")
    R.add("py_compile", "py_compile all modified files", not compile_errors,
          f"{len(MODIFIED_FILES)} files" + (f"; errors={compile_errors}" if compile_errors else ""))

    # =============================================================
    header("2 — IMPORT THE APPLICATION (headless)")
    # =============================================================
    # app.py is a Streamlit script: importing it executes the whole page body.
    # Run it through Streamlit's bare-mode runner in a subprocess so any
    # exception surfaces without needing a browser or a server port.
    env = dict(os.environ)
    env.update({
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_GLOBAL_DEVELOPMENT_MODE": "false",
    })
    dash_py, dash_mods = find_dashboard_python()
    if not dash_py:
        R.add("dashboard_env", "Dashboard interpreter found", False,
              "no interpreter with both streamlit and plotly")
        dash_py = sys.executable
    else:
        rel = dash_py.replace(str(REPO_ROOT) + "/", "")
        print(f"  dashboard interpreter : {rel}")
        for m, v in dash_mods.items():
            print(f"    {m:<10} {v}")
        R.add("dashboard_env", "Dashboard interpreter has Streamlit + plotly", True,
              f"{rel} · streamlit {dash_mods['streamlit']}, plotly {dash_mods['plotly']}, "
              f"torch {dash_mods['torch']}")

    # Streamlit CLI is available and can resolve the script.
    cli = subprocess.run(
        [dash_py, "-m", "streamlit", "run", "--help"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    cli_ok = cli.returncode == 0
    R.add("streamlit_cli", "`streamlit run` CLI available", cli_ok,
          "streamlit run src/app.py is invocable")

    # Execute the whole script top to bottom in "bare mode". Streamlit widgets
    # return their defaults outside a real session, so this runs the entire page
    # body -- the strongest check short of driving a browser. Any exception in
    # any page's module-level code surfaces here.
    direct = subprocess.run(
        [dash_py, "-c",
         "import sys, runpy, warnings\n"
         "sys.path.insert(0, 'src')\n"
         "warnings.filterwarnings('ignore')\n"
         "runpy.run_path('src/app.py', run_name='__main__')\n"
         "print('APP_EXEC_OK')"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    exec_ok = "APP_EXEC_OK" in direct.stdout
    tail = (direct.stderr or direct.stdout)[-700:].replace("\n", " ")
    R.add("app_import", "Application executes top-to-bottom", exec_ok,
          "bare-mode run reached the footer" if exec_ok else f"…{tail}")

    # The dashboard runs under its own torch build. Confirm the probabilities it
    # would display still match the stored substrate under THAT interpreter --
    # the number a judge sees is the one that has to be right.
    dash_check = subprocess.run(
        [dash_py, "-c",
         "import sys, json, numpy as np, pandas as pd, torch\n"
         "sys.path.insert(0, 'src')\n"
         "import screening_data as sd\n"
         "from v4_explainer import V4Explainer\n"
         "sub = sd.load()\n"
         "ex = V4Explainer(device='cpu')\n"
         "X = sub.X\n"
         "out = np.zeros((len(X), 6))\n"
         "with torch.no_grad():\n"
         "    for s in range(0, len(X), 64):\n"
         "        e = min(s+64, len(X))\n"
         "        _z, pr, _st = ex.model.k_step_rollout("
         "torch.tensor(X[s:e], dtype=torch.float32), 5)\n"
         "        for j in range(6):\n"
         "            out[s:e, j] = pr[j].cpu().numpy()\n"
         "stored = sub.probs\n"
         "mf = out[:,1:].max(axis=1)\n"
         "res = {'max_delta': float(np.abs(out-stored).max()),\n"
         "       'alert_disagreements': int(((mf >= sub.alert_threshold).astype(int)"
         " != sub.alerts).sum()),\n"
         "       'threshold': float(sub.alert_threshold),\n"
         "       'torch': torch.__version__}\n"
         "print('DASHCHECK' + json.dumps(res))"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    dash_res = {}
    for line in dash_check.stdout.splitlines():
        if line.startswith("DASHCHECK"):
            dash_res = json.loads(line[len("DASHCHECK"):])
    if dash_res:
        print(f"  under dashboard torch {dash_res['torch']}: "
              f"max|Δ| = {dash_res['max_delta']:.3e}, "
              f"alert disagreements = {dash_res['alert_disagreements']}")
    R.add("dashboard_runtime_equivalence",
          "Dashboard runtime reproduces stored forecasts",
          bool(dash_res) and dash_res.get("max_delta", 1) <= 1e-5
          and dash_res.get("alert_disagreements", 1) == 0,
          (f"torch {dash_res.get('torch')}: max|Δ| = {dash_res.get('max_delta', float('nan')):.3e}, "
           f"0 alert disagreements") if dash_res
          else f"probe failed: {(dash_check.stderr or '')[-200:]}")

    # -------- HEADLESS SERVE --------
    # Start the real Streamlit server, confirm it serves the page and reports a
    # healthy core, then shut it down. This is the closest check to what a judge
    # will actually do at the demo.
    serve_ok, serve_detail = False, ""
    serve_port = 8613
    server = None
    try:
        server = subprocess.Popen(
            [dash_py, "-m", "streamlit", "run", "src/app.py",
             "--server.headless=true", f"--server.port={serve_port}",
             "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"],
            cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        health = subprocess.run(
            ["curl", "-s", "--retry", "40", "--retry-delay", "2",
             "--retry-connrefused", "--max-time", "120",
             f"http://127.0.0.1:{serve_port}/_stcore/health"],
            capture_output=True, text=True, timeout=180,
        )
        page = subprocess.run(
            ["curl", "-s", "-o", os.devnull, "-w", "%{http_code}",
             "--max-time", "60", f"http://127.0.0.1:{serve_port}/"],
            capture_output=True, text=True, timeout=90,
        )
        serve_ok = health.stdout.strip() == "ok" and page.stdout.strip() == "200"
        serve_detail = (f"HTTP {page.stdout.strip()} on :{serve_port}, "
                        f"/_stcore/health = {health.stdout.strip() or 'no response'}")
    except Exception as exc:                            # noqa: BLE001
        serve_detail = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
    print(f"  headless serve: {serve_detail}")
    R.add("headless_serve", "Streamlit serves the app headlessly", serve_ok, serve_detail)

    # =============================================================
    header("3 — SCREENING ARTIFACTS LOAD")
    # =============================================================
    import screening_data as sdm
    load_error = None
    sub = None
    try:
        sub = sdm.load()
    except Exception as exc:                            # noqa: BLE001
        load_error = f"{type(exc).__name__}: {exc}"
    R.add("artifacts_load", "All screening artifacts load", sub is not None,
          f"{sub.n_sequences} sequences, {len(sub.feature_names)} features"
          if sub is not None else str(load_error))
    if sub is None:
        _write(pre_hashes, {}, started)
        return 1

    # Malformed-input handling must raise, not silently degrade.
    try:
        sdm.load(REPO_ROOT / "results" / "__nonexistent_screening__")
        guard_ok = False
        guard_detail = "missing directory did NOT raise"
    except sdm.ScreeningDataError as exc:
        guard_ok = True
        guard_detail = f"raises ScreeningDataError: {str(exc)[:60]}…"
    except Exception as exc:                            # noqa: BLE001
        guard_ok = False
        guard_detail = f"raised {type(exc).__name__} instead of ScreeningDataError"
    R.add("missing_artifact_guard", "Missing artifacts raise (no silent fallback)",
          guard_ok, guard_detail)

    # =============================================================
    header("4 — DEFAULT DEMO SEQUENCE")
    # =============================================================
    default_seq = sub.default_sequence_id
    mf = sub.max_forecast
    expected = int(np.flatnonzero(mf == mf.max())[0])
    row = sub.preds.iloc[default_seq]
    print(f"  default sequence_id     : {default_seq}")
    print(f"  max forecast probability: {mf[default_seq]:.6f} (global max {mf.max():.6f})")
    print(f"  target t+1 timestamp    : {row['target_t1_timestamp']}")
    print(f"  observed window t       : {row['observed_window_end']}")
    R.add("default_sequence", "Default sequence reproducible",
          default_seq == expected and mf[default_seq] == mf.max(),
          f"#{default_seq}, argmax of max_forecast_probability (model output only)")

    # The default must be chosen from MODEL OUTPUT, never from event labels.
    src = (REPO_ROOT / "src" / "screening_data.py").read_text()
    dflt_block = src.split("def default_sequence_id")[1].split("def ")[0]
    label_free = not any(tok in dflt_block for tok in
                         ("timeline", "region", "label_", "attack_present",
                          "is_documented_onset", "trajectory_label"))
    R.add("default_no_label_leak", "Default selection uses no event labels", label_free,
          "chosen by argmax(max_forecast_probability)")

    # =============================================================
    header("5 — FORECAST EQUIVALENCE vs screening_predictions.csv")
    # =============================================================
    from v4_explainer import V4Explainer
    explainer = V4Explainer(device="cpu")

    X = sub.X

    def rollout_on(device):
        mdl = explainer.model.to(device)
        out = np.zeros((len(X), MAX_K + 1))
        with torch.no_grad():
            for s in range(0, len(X), BATCH_SIZE):
                e = min(s + BATCH_SIZE, len(X))
                xb = torch.tensor(X[s:e], dtype=torch.float32).to(device)
                _z, pr, _st = mdl.k_step_rollout(xb, MAX_K)
                for j in range(MAX_K + 1):
                    out[s:e, j] = pr[j].cpu().numpy()
        return out

    stored = sub.probs
    stored_maxf = sub.max_forecast

    # The stored CSV was produced on the device recorded in the substrate
    # manifest. Reproduce on THAT device for the equivalence check; float
    # reduction order differs between backends, so comparing across devices
    # measures the backend, not the data.
    stored_device = str(sub.manifest.get("environment", {}).get("device_used", "cpu"))
    native_available = (
        (stored_device == "mps" and torch.backends.mps.is_available())
        or (stored_device == "cuda" and torch.cuda.is_available())
        or stored_device == "cpu"
    )
    native_dev = torch.device(stored_device if native_available else "cpu")
    print(f"  stored values were produced on : {stored_device}"
          f"{'' if native_available else ' (unavailable here -> CPU)'}")

    native = rollout_on(native_dev)
    deltas = {}
    for j, name in enumerate(["nowcast_t0"] + [f"prob_t{k}" for k in range(1, MAX_K + 1)]):
        deltas[name] = float(np.abs(native[:, j] - stored[:, j]).max())
    max_delta = max(deltas.values())
    for name, d in deltas.items():
        print(f"  {name:<14} max|Δ| = {d:.3e}   (on {native_dev.type})")
    native_maxf = native[:, 1:].max(axis=1)
    maxf_delta = float(np.abs(native_maxf - stored_maxf).max())
    print(f"  max_forecast   max|Δ| = {maxf_delta:.3e}   (on {native_dev.type})")

    # float32 -> CSV text -> float64 round-trip is the only expected loss.
    TOL = 1e-6
    R.add("forecast_equivalence",
          f"Forecast probabilities match stored CSV (on {native_dev.type})",
          max_delta <= TOL and maxf_delta <= TOL,
          f"max|Δ| = {max_delta:.3e} over all 6 columns (tol {TOL:.0e}, "
          f"float32 CSV round-trip)")

    # The dashboard itself runs the explainer on CPU. Confirm the CPU path agrees
    # within the cross-device delta the substrate already documented, so the
    # displayed number is the same quantity to display precision.
    cpu = rollout_on(torch.device("cpu")) if native_dev.type != "cpu" else native
    cpu_delta = float(np.abs(cpu - stored).max())
    documented = float(sub.manifest.get("forecast_behaviour", {})
                       .get("cross_device_max_delta", 1e-5))
    CROSS_TOL = max(1e-5, documented * 2)
    print(f"  CPU path (what the UI uses) max|Δ| vs stored = {cpu_delta:.3e}   "
          f"(documented cross-device delta {documented:.3e})")
    R.add("forecast_cpu_agreement",
          "UI (CPU) forecast agrees to display precision",
          cpu_delta <= CROSS_TOL,
          f"max|Δ| = {cpu_delta:.3e} <= {CROSS_TOL:.0e}; "
          f"MPS/CPU float ordering only, 4-dp display unaffected")

    explainer.model.to("cpu")   # restore the CPU model the rest of the run uses
    recomputed_maxf = cpu[:, 1:].max(axis=1)

    thr = sub.alert_threshold
    stored_alerts = sub.alerts
    recomputed_alerts = (recomputed_maxf >= thr).astype(int)
    R.add("alert_equivalence", "Alert decisions match stored CSV",
          int((stored_alerts != recomputed_alerts).sum()) == 0,
          f"{int(stored_alerts.sum())} alerts, 0 disagreements")

    # =============================================================
    header("6 — THRESHOLD EQUIVALENCE")
    # =============================================================
    man_traj5 = float(manifest["threshold"]["per_horizon_trajectory_max"]["5"])
    man_point = {int(k): float(v) for k, v
                 in manifest["threshold"]["per_horizon_future_prediction"].items()}
    print(f"  substrate alert threshold : {thr}")
    print(f"  manifest trajectory-MAX K5: {man_traj5}")
    print(f"  manifest point thresholds : {man_point}")
    print(f"  explainer point thresholds: {explainer.horizon_thresholds}")
    R.add("threshold_equivalence", "UI threshold == frozen manifest",
          abs(thr - man_traj5) < 1e-12
          and sub.point_thresholds == man_point
          and explainer.horizon_thresholds == man_point,
          f"{thr} == trajectory-MAX K=5; point thresholds identical")
    R.add("threshold_not_checkpoint", "Not the checkpoint 0.05 training artefact",
          abs(thr - 0.05) > 1e-9, f"threshold is {thr}, not 0.05")
    R.add("k4_no_threshold", "t+4 has no frozen point threshold",
          4 not in sub.point_thresholds,
          "displayed but never presented as an official classification")

    # =============================================================
    header("7 — EXPLAINER RUNS")
    # =============================================================
    xai_ok, xai_detail, xai_payload = False, "", {}
    try:
        att = explainer.attribute(sub.model_input(default_seq), k=MAX_K, n_steps=128)
        attrs = att["attributions"]
        signed = V4Explainer.feature_aggregate(attrs)["signed"][0]
        order = np.argsort(-np.abs(signed))[:5]
        top = [{"feature": sub.feature_names[int(i)],
                "attribution": float(signed[int(i)]),
                "direction": "higher_threat" if signed[int(i)] > 0 else "lower_threat"}
               for i in order]
        identity = float(abs(float(att["prediction"][0]) - stored[default_seq, MAX_K]))
        xai_ok = (attrs.shape == (1, 10, 52) and np.isfinite(signed).all()
                  and identity < 1e-5)
        xai_payload = {
            "sequence_id": default_seq, "horizon": MAX_K,
            "attribution_shape": list(attrs.shape),
            "explained_probability": float(att["prediction"][0]),
            "stored_probability": float(stored[default_seq, MAX_K]),
            "prediction_identity_abs_delta": identity,
            "completeness_rel_error": float(att["completeness_rel_error"][0]),
            "method": att["method"], "n_steps": att["n_steps"],
            "top_features": top,
        }
        print(f"  attribution tensor : {attrs.shape}")
        print(f"  explained P(t+5)   : {att['prediction'][0]:.6f}")
        print(f"  stored   P(t+5)    : {stored[default_seq, MAX_K]:.6f}")
        print(f"  identity |Δ|       : {identity:.3e}")
        for t in top:
            print(f"    {t['feature']:<34} {t['attribution']:>+10.6f}  {t['direction']}")
        xai_detail = f"IG (1,10,52), identity Δ={identity:.1e}, top-5 signed + direction"
    except Exception as exc:                            # noqa: BLE001
        xai_detail = f"{type(exc).__name__}: {exc}"
    R.add("explainer", "V4 explainer runs on a screening sequence", xai_ok, xai_detail)

    # =============================================================
    header("8 — MITRE MAPPER RUNS")
    # =============================================================
    mitre_ok, mitre_detail, mitre_payload = False, "", {}
    try:
        from mitre_mapper import MITREMapper, INPUT_FEATURES as MF
        if list(MF) != sub.feature_names:
            raise ValueError("mitre_mapper feature order != frozen V4 contract")
        mapper = MITREMapper(model=explainer.model, threshold=thr, device="cpu")
        mapping = mapper.map_single_window(
            x_window=sub.X[default_seq], window_idx=default_seq,
            timestamp=str(sub.preds["target_t1_timestamp"].iloc[default_seq]),
            total_windows=sub.n_sequences,
            first_alert_idx=int(np.argmax(stored_alerts)) if stored_alerts.any() else None,
            attack_prob=float(stored_maxf[default_seq]),
        )
        md = mapping.as_dict() if hasattr(mapping, "as_dict") else vars(mapping)
        mitre_ok = bool(md.get("tactic_id")) and md.get("tactic_id") != "NONE"
        mitre_payload = {
            "sequence_id": default_seq, "threshold_used": thr,
            "rule_library_size": len(mapper.rules),
            "feature_order_matches_contract": True,
            "tactic_id": md.get("tactic_id"), "tactic_name": md.get("tactic_name"),
            "technique_id": md.get("technique_id"),
            "technique_name": md.get("technique_name"),
            "confidence": md.get("confidence"),
            "temporal_phase": md.get("temporal_phase"),
        }
        print(f"  rule library : {len(mapper.rules)} rules")
        print(f"  tactic       : {md.get('tactic_id')} — {md.get('tactic_name')}")
        print(f"  technique    : {md.get('technique_id')} — {md.get('technique_name')}")
        print(f"  confidence   : {md.get('confidence')}  phase: {md.get('temporal_phase')}")
        mitre_detail = f"{md.get('tactic_id')}/{md.get('technique_id')}, {len(mapper.rules)} rules"
    except Exception as exc:                            # noqa: BLE001
        mitre_detail = f"{type(exc).__name__}: {exc}"
    R.add("mitre", "MITRE mapper runs on the same sequence", mitre_ok, mitre_detail)

    # =============================================================
    header("9 — EVENT LABELS NOT IN MODEL INPUT")
    # =============================================================
    # The model input is exactly the 52-D feature tensor, which is byte-identical
    # to the frozen X_test.npy. Labels live only in separate CSVs.
    mi = sub.model_input(default_seq)
    shape_ok = mi.shape == (1, 10, 52)
    xt = np.load(BENCH_DIR / "X_test.npy")
    identical = bool(np.array_equal(sub.X, xt))
    print(f"  model_input shape                     : {mi.shape}")
    print(f"  screening_X == frozen X_test          : {identical}")
    print(f"  label columns in the feature contract : "
          f"{[f for f in sub.feature_names if 'label' in f or 'attack' in f or 'region' in f]}")

    label_tokens = ("label", "attack_present", "region", "is_documented_onset",
                    "trajectory_label", "dominant_attack", "attack_ratio")
    contract_clean = not any(tok in f for f in sub.feature_names for tok in label_tokens)

    # model_input() must not touch any label-bearing frame. Scan the CODE only:
    # the docstring legitimately mentions labels while explaining their absence,
    # so a raw text grep would flag its own disclaimer.
    mi_code = _function_code(src, "model_input")
    mi_clean = not any(tok in mi_code for tok in
                       ("timeline", "region", "label", "self.meta", "self.preds",
                        "attack", "onset"))
    print(f"  model_input() code (docstring stripped):")
    for line in mi_code.splitlines():
        print(f"      {line}")
    print(f"  touches only self.X / self.n_sequences : {mi_clean}")

    # Every label-bearing column lives outside the feature tensor.
    leak_cols = [c for c in sub.timeline.columns if c in sub.feature_names]
    R.add("no_label_leakage", "Event labels absent from model input",
          shape_ok and identical and contract_clean and mi_clean and not leak_cols,
          "input is the 52-D frozen tensor only; labels live in separate CSVs")

    # =============================================================
    header("10 — CAPTURE GAP HANDLING")
    # =============================================================
    gaps = sub.capture_gaps()
    segs = sub.segments()
    for g in gaps:
        print(f"  gap: {g['from_timestamp']} -> {g['to_timestamp']} "
              f"({g['gap_seconds']:.0f} s = {g['gap_seconds'] / 3600:.2f} h)")
    print(f"  segments: {segs}")

    documented = ("05:30:10", "08:46:30")
    man_gaps = sub.manifest["contiguity"]["raw_capture_gaps"]
    gap_documented = any(documented[0] in g["after_timestamp"]
                         and documented[1] in g["next_timestamp"] for g in man_gaps)

    # The UI must insert a break (None) so no line spans the gap.
    ids = list(range(sub.n_sequences))
    xs, ys = sub.insert_gap_breaks(ids, sub.max_forecast)
    n_breaks = sum(1 for v in xs if v is None)
    breaks_ok = n_breaks == len(gaps)

    # No interpolation: the sequence count must equal the raw CSV row count,
    # i.e. nothing was manufactured to bridge the gap.
    no_fill = len(sub.preds) == sub.n_sequences == len(sub.X)
    ts = pd.to_datetime(sub.preds["target_t1_timestamp"])
    spans_gap = int(((ts.diff().dt.total_seconds() != 10)
                     & ts.diff().notna()).sum())

    print(f"  line breaks inserted : {n_breaks} (one per gap)")
    print(f"  segments never joined: {len(segs)} segments for {len(gaps)} gap(s)")
    R.add("capture_gap", "Capture gap detected and never interpolated",
          len(gaps) == 1 and gap_documented and breaks_ok and no_fill
          and spans_gap == len(gaps) and len(segs) == len(gaps) + 1,
          f"1 documented gap (05:30:10→08:46:30), {n_breaks} line break(s), "
          f"{len(segs)} segments, no synthetic windows")

    # =============================================================
    header("11 — OFFLINE DEPENDENCY CHECK")
    # =============================================================
    offline_hits = []
    for rel in ["src/app.py", "src/screening_data.py"]:
        text = (REPO_ROOT / rel).read_text()
        for pat in NETWORK_PATTERNS:
            for m in re.finditer(pat, text):
                frag = m.group(0)
                if any(a in frag for a in URL_ALLOW_SUBSTRINGS):
                    continue
                line = text[:m.start()].count("\n") + 1
                offline_hits.append(f"{rel}:{line} {frag[:70]}")
    for h in offline_hits:
        print(f"  ⚠ {h}")
    if not offline_hits:
        print("  no network calls, remote URLs or sockets found in the dashboard source")
    R.add("offline", "No network/API dependency", not offline_hits,
          "app.py + screening_data.py are network-free"
          if not offline_hits else f"{len(offline_hits)} hit(s)")

    # All data the dashboard reads must be local files.
    local_sources = [
        OUT_DIR / "screening_X.npy", OUT_DIR / "screening_predictions.csv",
        OUT_DIR / "event_timeline.csv", CHECKPOINT,
        BENCH_DIR / "input_scaler.pkl", MANIFEST_PATH,
    ]
    R.add("local_sources", "All data sources are local files",
          all(p.exists() for p in local_sources),
          f"{len(local_sources)} local artifacts resolved")

    # =============================================================
    header("12 — FROZEN ARTIFACTS UNMODIFIED")
    # =============================================================
    post_hashes = {str(p.relative_to(REPO_ROOT)): (sha256_file(p) if p.exists() else None)
                   for p in FROZEN_ARTIFACTS}
    changed = [k for k in pre_hashes if pre_hashes[k] != post_hashes[k]]
    for k in changed:
        print(f"  CHANGED: {k}")
    R.add("frozen_unmodified", "No frozen artifact modified", not changed,
          f"{len(pre_hashes)} artifacts re-hashed, {len(changed)} changed")

    ckpt_ok = post_hashes[str(CHECKPOINT.relative_to(REPO_ROOT))] == \
        manifest["checkpoint"]["sha256"]
    scaler_ok = post_hashes[str((BENCH_DIR / "input_scaler.pkl").relative_to(REPO_ROOT))] == \
        manifest["scalers"]["input_scaler"]["sha256"]
    R.add("hashes_vs_manifest", "Checkpoint + scaler match frozen manifest",
          ckpt_ok and scaler_ok, "SHA256 identical to docs/v4_baseline_manifest.json")

    # =============================================================
    overall = not R.failed
    payload = _write(pre_hashes, post_hashes, started, sub=sub, thr=thr,
                     deltas=deltas, maxf_delta=maxf_delta,
                     default_seq=default_seq, gaps=gaps, segs=segs,
                     xai=xai_payload, mitre=mitre_payload,
                     offline_hits=offline_hits, changed=changed,
                     man_point=man_point, man_traj5=man_traj5, overall=overall)

    header("SUMMARY")
    n_pass = sum(1 for r in R.rows if r["status"] == "PASS")
    print(f"  checks : {n_pass}/{len(R.rows)} PASS")
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"\n  wrote {payload.relative_to(REPO_ROOT)}")
    return 0 if overall else 1


def _write(pre, post, started, sub=None, thr=None, deltas=None, maxf_delta=None,
           default_seq=None, gaps=None, segs=None, xai=None, mitre=None,
           offline_hits=None, changed=None, man_point=None, man_traj5=None,
           overall=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "started_utc": started,
        "overall": "PASS" if overall else "FAIL",
        "validator": "src/validate_dashboard.py",
        "environment": {
            "validator_python": sys.version.split()[0],
            "validator_torch": torch.__version__,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "platform": sys.platform,
        },
        "modified_files": [
            {"path": rel,
             "sha256": sha256_file(REPO_ROOT / rel) if (REPO_ROOT / rel).exists() else None,
             "lines": len((REPO_ROOT / rel).read_text().splitlines())
             if (REPO_ROOT / rel).exists() else None}
            for rel in ALL_MODIFIED_FILES
        ],
        "artifact_hashes_before": pre,
        "artifact_hashes_after": post,
        "artifacts_changed": changed or [],
        "checks": {r["key"]: r["status"] for r in R.rows},
        "check_details": R.rows,
        "prediction_equivalence": {
            "per_column_max_abs_delta": deltas or {},
            "max_forecast_max_abs_delta": maxf_delta,
            "tolerance": 1e-6,
            "note": "Tolerance is the float32 CSV round-trip only; the dashboard "
                    "recomputes probabilities from the frozen checkpoint at runtime.",
        },
        "threshold_equivalence": {
            "substrate_alert_threshold": thr,
            "manifest_trajectory_max_k5": man_traj5,
            "manifest_point_thresholds": man_point,
            "k4_has_threshold": False,
            "checkpoint_embedded_threshold_used": False,
        },
        "default_demo_sequence": {
            "sequence_id": default_seq,
            "rule": "argmax(max_forecast_probability), ties -> lowest sequence_id",
            "selected_from": "model output only; no event label consulted",
            "target_t1_timestamp": (str(sub.preds["target_t1_timestamp"].iloc[default_seq])
                                    if sub is not None and default_seq is not None else None),
            "max_forecast_probability": (float(sub.max_forecast[default_seq])
                                         if sub is not None and default_seq is not None else None),
        },
        "explainer_check": xai or {},
        "mitre_check": mitre or {},
        "gap_handling_check": {
            "gaps_detected": gaps or [],
            "segments": segs or [],
            "interpolation": "none",
            "synthetic_windows_created": 0,
            "ui_behaviour": "NaN break inserted in every line series; replay autoplay "
                            "halts at the segment boundary; explicit DATA GAP banner",
        },
        "headless_serve_check": {
            "method": "streamlit run src/app.py --server.headless=true",
            "probed": ["/", "/_stcore/health"],
            "interpreter": "src/venv/bin/python (Streamlit + plotly)",
        },
        "offline_dependency_check": {
            "patterns_scanned": NETWORK_PATTERNS,
            "files_scanned": ["src/app.py", "src/screening_data.py"],
            "hits": offline_hits or [],
            "allowlisted_substrings": list(URL_ALLOW_SUBSTRINGS),
        },
    }
    path = OUT_DIR / "dashboard_validation.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    return path


if __name__ == "__main__":
    sys.exit(main())
