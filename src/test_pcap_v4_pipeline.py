"""
=============================================================
VALIDATION HARNESS FOR src/pcap_ingest_v4.py
=============================================================

Runs a fixed battery of contract checks against the canonical
PCAP -> V4 inference pipeline, then executes it end-to-end on
the bundled software-test capture
``data/sample_pcap/smallFlows.pcap``.

  python src/test_pcap_v4_pipeline.py

Exit code 0 iff every check passes.
=============================================================
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import pcap_ingest_v4 as P

REPO_ROOT = P.REPO_ROOT
SAMPLE_PCAP = REPO_ROOT / "data" / "sample_pcap" / "smallFlows.pcap"

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            detail = fn()
            _results.append((name, True, detail or ""))
        except AssertionError as exc:
            _results.append((name, False, f"ASSERT: {exc}"))
        except Exception as exc:  # noqa: BLE001
            _results.append(
                (name, False, f"{type(exc).__name__}: {exc}\n"
                 + traceback.format_exc())
            )
        return fn
    return deco


# Shared artifacts (populated as checks run)
STATE: dict = {}


# -------------------------------------------------------------
# 1. Input dimension = 52
# -------------------------------------------------------------
@check("01 input dimension = 52")
def _c01():
    names = P.INPUT_FEATURE_NAMES
    assert len(names) == 52, f"got {len(names)}"
    scaler = P.load_scaler(P.INPUT_SCALER_PATH)
    STATE["input_scaler"] = scaler
    assert scaler.n_features_in_ == 52, (
        f"input_scaler.n_features_in_ = {scaler.n_features_in_}"
    )
    return "52 canonical feature names; input_scaler expects 52"


# -------------------------------------------------------------
# 2. Sequence dimension = 10 x 52
# -------------------------------------------------------------
@check("02 sequence shape = (N, 10, 52)")
def _c02():
    assert P.SEQ_LEN == 10, f"SEQ_LEN = {P.SEQ_LEN}"
    # synthetic contiguous feature frame
    n = 15
    df = pd.DataFrame({"window_idx": np.arange(n),
                       "timestamp": np.arange(n) * 10.0})
    for f in P.INPUT_FEATURE_NAMES:
        df[f] = np.random.rand(n)
    scaler = STATE["input_scaler"]
    seq = P.build_sequences(df, scaler)
    X = seq["X"]
    assert X.ndim == 3, f"ndim = {X.ndim}"
    assert X.shape[1:] == (10, 52), f"shape = {X.shape}"
    assert X.shape[0] == n - 10 + 1, f"N = {X.shape[0]}"
    STATE["synthetic_X"] = X
    return f"built {X.shape} from {n} contiguous windows"


# -------------------------------------------------------------
# 3. V4 checkpoint loads
# -------------------------------------------------------------
@check("03 V4 checkpoint loads")
def _c03():
    model, ckpt = P.load_v4_model()
    STATE["model"] = model
    STATE["ckpt"] = ckpt
    assert ckpt["model_class"] == "LatentWorldModelV4", ckpt.get("model_class")
    assert ckpt["input_size"] == 52
    assert ckpt["sequence_length"] == 10
    assert ckpt["transition_type"] == "deterministic", ckpt.get("transition_type")
    # gated residual + LayerNorm parameters must be present
    sd = model.state_dict()
    for key in ("gate_fc.weight", "transition_norm.weight",
                "transition_mlp.0.weight", "encoder_norm.weight"):
        assert key in sd, f"missing param {key}"
    return (f"loaded {ckpt['model_class']}, "
            f"transition={ckpt['transition_type']}, "
            f"gated-residual + LayerNorm params present")


# -------------------------------------------------------------
# 4. Latent dimension = 32
# -------------------------------------------------------------
@check("04 latent dimension = 32")
def _c04():
    model, ckpt = STATE["model"], STATE["ckpt"]
    assert ckpt["latent_size"] == 32, ckpt["latent_size"]
    with torch.no_grad():
        z = model.encode(torch.zeros(3, 10, 52))
    assert z.shape == (3, 32), f"encode -> {tuple(z.shape)}"
    with torch.no_grad():
        z2 = model.transition_step(z)
    assert z2.shape == (3, 32), f"transition_step -> {tuple(z2.shape)}"
    return "encode() and transition_step() both yield 32-D latent"


# -------------------------------------------------------------
# 5. Decoder output dimension = 26
# -------------------------------------------------------------
@check("05 decoder output dimension = 26")
def _c05():
    model, ckpt = STATE["model"], STATE["ckpt"]
    assert ckpt["output_size"] == 26, ckpt["output_size"]
    with torch.no_grad():
        out = model.decode(torch.zeros(4, 32))
    assert out.shape == (4, 26), f"decode -> {tuple(out.shape)}"
    tscaler = P.load_scaler(P.TARGET_SCALER_PATH)
    assert tscaler.n_features_in_ == 26
    assert len(P.STATE_FEATURE_NAMES) == 26
    return "decode() -> 26; target_scaler expects 26"


# -------------------------------------------------------------
# 6. K=5 rollout produces exactly 5 future steps (6 cols total)
# -------------------------------------------------------------
@check("06 recursive K=5 rollout -> exactly 5 future steps")
def _c06():
    model, ckpt = STATE["model"], STATE["ckpt"]
    k = int(ckpt["max_k"])
    assert k == 5, f"ckpt.max_k = {k}"
    X = np.random.randn(6, 10, 52).astype(np.float32)
    roll = P.run_k_rollout(model, X, k)
    probs = roll["probs"]
    assert probs.shape == (6, k + 1), f"probs shape {probs.shape}"
    future = probs[:, 1:]
    assert future.shape[1] == 5, f"future steps = {future.shape[1]}"

    # prove it is *recursive*, not 5 independent single-steps:
    with torch.no_grad():
        x_t = torch.tensor(X)
        z0 = model.encode(x_t)
        z = z0
        manual = [torch.sigmoid(model.predict_attack(z)).numpy()]
        for _ in range(k):
            z = model.transition_step(z)   # feed predicted latent back
            manual.append(torch.sigmoid(model.predict_attack(z)).numpy())
    manual = np.stack(manual, axis=1)
    assert np.allclose(manual, probs, atol=1e-5), "rollout != recursive chain"

    # and confirm step t+2 differs from applying transition to z0 twice
    # independently would still match (that's the same thing) -- instead
    # confirm t+1 != t+2 generally (state actually evolves)
    assert not np.allclose(probs[:, 1], probs[:, 2]), (
        "t+1 and t+2 identical -> transition not being re-applied"
    )
    STATE["rollout_probs"] = probs
    return f"probs {probs.shape}; matches recursive latent chain (atol 1e-5)"


# -------------------------------------------------------------
# 7. All probabilities in [0, 1]
# -------------------------------------------------------------
@check("07 all attack probabilities in [0, 1]")
def _c07():
    probs = STATE["rollout_probs"]
    assert probs.min() >= 0.0, probs.min()
    assert probs.max() <= 1.0, probs.max()
    return f"min={probs.min():.4f} max={probs.max():.4f}"


# -------------------------------------------------------------
# 8. Threshold comes from the checkpoint, never the PCAP
# -------------------------------------------------------------
@check("08 threshold frozen from checkpoint (= 0.05)")
def _c08():
    ckpt = STATE["ckpt"]
    thr = float(ckpt["selected_threshold"])
    assert abs(thr - 0.05) < 1e-9, f"selected_threshold = {thr}"
    src = Path(P.__file__).read_text()
    # the pipeline must read the threshold from the checkpoint
    assert 'ckpt["selected_threshold"]' in src, (
        "pipeline does not read threshold from checkpoint"
    )
    # and must not compute one from data
    for banned in ("threshold_search", "roc_curve", "precision_recall_curve",
                   "np.percentile(prob", "quantile(prob"):
        assert banned not in src, f"pipeline derives threshold via {banned!r}"
    return "threshold = 0.05, read from ckpt['selected_threshold']"


# -------------------------------------------------------------
# 9. Feature ordering matches canonical input_feature_names.txt
# -------------------------------------------------------------
@check("09 feature ordering == input_feature_names.txt")
def _c09():
    canon = [ln.strip() for ln in
             P.INPUT_FEATURE_NAMES_PATH.read_text().splitlines() if ln.strip()]
    assert P.INPUT_FEATURE_NAMES == canon, "module list drifted from file"
    assert P.BASE_FEATURES == canon[:26]
    assert P.REL_CHANGE_FEATURES == canon[26:]
    state_canon = [ln.strip() for ln in
                   P.STATE_FEATURE_NAMES_PATH.read_text().splitlines()
                   if ln.strip()]
    assert P.BASE_FEATURES == state_canon, "base != state_feature_names.txt"
    # end-to-end: features frame columns feed the scaler in this exact order
    assert canon[0] == "flow_count" and canon[25] == "protocol_entropy"
    assert canon[26] == "flow_count_rel_change"
    assert canon[51] == "protocol_entropy_rel_change"
    return "52-D order identical to canonical file (26 base + 26 rel_change)"


# -------------------------------------------------------------
# 10. Strict temporal contiguity -- gap-spanning sequences rejected
# -------------------------------------------------------------
@check("10 temporal-gap sequences rejected (no invalid gaps)")
def _c10():
    scaler = STATE["input_scaler"]
    # windows 0..5 then jump to 20..28  (gap between idx 5 and 20)
    idx = list(range(0, 6)) + list(range(20, 29))
    n = len(idx)
    df = pd.DataFrame({"window_idx": idx,
                       "timestamp": np.array(idx) * 10.0})
    for f in P.INPUT_FEATURE_NAMES:
        df[f] = np.random.rand(n)
    seq = P.build_sequences(df, scaler)
    X = seq["X"]
    # only the contiguous run 20..28 (9 windows) is < 10 long -> 0 usable,
    # run 0..5 is 6 windows -> also < 10. So zero usable sequences, and
    # every candidate that spans the gap must be counted as rejected.
    assert len(X) == 0, f"expected 0 usable, got {len(X)}"
    assert seq["n_rejected_gap"] == seq["n_candidate"], (
        f"rejected {seq['n_rejected_gap']} of {seq['n_candidate']} candidates"
    )
    # now a fully contiguous 12-window frame -> 3 usable, 0 rejected
    idx2 = list(range(100, 112))
    df2 = pd.DataFrame({"window_idx": idx2,
                        "timestamp": np.array(idx2) * 10.0})
    for f in P.INPUT_FEATURE_NAMES:
        df2[f] = np.random.rand(12)
    seq2 = P.build_sequences(df2, scaler)
    assert len(seq2["X"]) == 3, len(seq2["X"])
    assert seq2["n_rejected_gap"] == 0, seq2["n_rejected_gap"]
    return ("gap-spanning candidates all rejected; "
            "contiguous frame yields 3 sequences, 0 rejected")


# -------------------------------------------------------------
# 11. End-to-end on the bundled sample PCAP
# -------------------------------------------------------------
@check("11 end-to-end on data/sample_pcap/smallFlows.pcap")
def _c11():
    assert SAMPLE_PCAP.exists(), f"missing {SAMPLE_PCAP}"
    out_dir = REPO_ROOT / "models" / "pcap_analysis_v4"
    summary = P.process_pcap(
        str(SAMPLE_PCAP),
        output_dir=out_dir,
        model=STATE["model"],
        ckpt=STATE["ckpt"],
        input_scaler=STATE["input_scaler"],
    )
    STATE["sample_summary"] = summary
    assert summary["status"] in ("success", "insufficient_data"), summary
    if summary["status"] == "success":
        assert summary["input_dim"] == 52
        assert summary["latent_dim"] == 32
        assert summary["decoder_dim"] == 26
        assert summary["rollout_steps_K"] == 5
        assert summary["threshold"] == 0.05
        assert len(summary["peak_trajectory"]) == 6
        assert 0.0 <= summary["prob_min"] <= summary["prob_max"] <= 1.0
        assert summary["rejected_temporal_gap"] >= 0
        roll_csv = out_dir / f"{SAMPLE_PCAP.stem}_rollout.csv"
        assert roll_csv.exists()
        rdf = pd.read_csv(roll_csv)
        fut_cols = [c for c in rdf.columns if c.startswith("prob_t+")]
        assert len(fut_cols) == 6, fut_cols  # t+0..t+5
        return (f"status=success  seqs={summary['usable_sequences']}  "
                f"peak={summary['peak_probability']:.4f}  "
                f"level={summary['overall_warning_level']}  "
                f"gap_rejected={summary['rejected_temporal_gap']}")
    return (f"status=insufficient_data  "
            f"feature_windows={summary['feature_windows']}  "
            f"reason={summary['reason']}")


# -------------------------------------------------------------
def run() -> int:
    # All checks already ran once, at decoration time (see @check).
    print("=" * 70)
    print("PCAP -> V4 PIPELINE VALIDATION")
    print("=" * 70)
    n_pass = 0
    for name, ok, detail in _results:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}")
        if detail:
            first = detail.splitlines()[0]
            print(f"       {first}")
            if not ok:
                for ln in detail.splitlines()[1:]:
                    print(f"       {ln}")
        n_pass += ok
    print("-" * 70)
    print(f"{n_pass}/{len(_results)} checks passed")
    print("=" * 70)
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    # decorators run checks at import/def time; just report + exit
    raise SystemExit(run())
